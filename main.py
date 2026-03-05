import streamlit as st
import pandas as pd
from kiteconnect import KiteConnect
from datetime import datetime, timedelta
import ta
from streamlit_autorefresh import st_autorefresh

# ================= CONFIG =================

API_KEY = st.secrets["API_KEY"]
API_SECRET = st.secrets["API_SECRET"]

st.set_page_config(page_title="NIFTY Strategy Dashboard", layout="wide")

st_autorefresh(interval=2000, key="refresh")

# ================= SESSION STATE =================

if "trades" not in st.session_state:
    st.session_state.trades = []

if "access_token" not in st.session_state:
    st.session_state.access_token = None

if "market_data" not in st.session_state:
    st.session_state.market_data = None

if "last_candle_time" not in st.session_state:
    st.session_state.last_candle_time = None

kite = KiteConnect(api_key=API_KEY)

# ================= LOGIN =================

query_params = st.query_params

if "request_token" in query_params:
    data = kite.generate_session(query_params["request_token"], api_secret=API_SECRET)
    st.session_state.access_token = data["access_token"]
    st.query_params.clear()

if not st.session_state.access_token:
    st.link_button("🔐 Login to Zerodha", kite.login_url())
    st.stop()

kite.set_access_token(st.session_state.access_token)

# ================= MODE =================

mode = st.sidebar.radio("Mode", ["Live Trading", "Backtest"])

# ================= CACHE =================

@st.cache_data(ttl=86400)
def get_instruments():
    return pd.DataFrame(kite.instruments("NFO"))

@st.cache_data(ttl=5)
def get_quote(symbol):
    return kite.quote([symbol])

# ================= INDICATORS =================

def jurik_ma(series, length=8):

    beta = 0.45 * (length - 1) / (0.45 * (length - 1) + 2)
    alpha = beta ** 2

    jma = series.copy()

    for i in range(1, len(series)):
        jma.iloc[i] = alpha * series.iloc[i] + (1 - alpha) * jma.iloc[i-1]

    return jma


def wavetrend(df):

    ap = (df["high"] + df["low"] + df["close"]) / 3

    esa = ap.ewm(span=10).mean()
    d = (ap - esa).abs().ewm(span=10).mean()

    ci = (ap - esa) / (0.015 * d)

    wt1 = ci.ewm(span=21).mean()
    wt2 = wt1.rolling(4).mean()

    return wt1, wt2


def squeeze_momentum(df):

    length = 20
    mult_bb = 2
    mult_kc = 1.5

    m_avg = df["close"].rolling(length).mean()
    m_std = df["close"].rolling(length).std()

    upper_bb = m_avg + mult_bb * m_std
    lower_bb = m_avg - mult_bb * m_std

    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs()
    ], axis=1).max(axis=1)

    range_ma = tr.rolling(length).mean()

    upper_kc = m_avg + mult_kc * range_ma
    lower_kc = m_avg - mult_kc * range_ma

    squeeze_on = (lower_bb > lower_kc) & (upper_bb < upper_kc)

    momentum = df["close"] - m_avg

    return squeeze_on, momentum


class StrategyLogic:

    @staticmethod
    def compute(df):

        df["jurik"] = jurik_ma(df["close"])

        df["wt1"], df["wt2"] = wavetrend(df)

        df["sqz_on"], df["momentum"] = squeeze_momentum(df)

        df["ema9"] = df["close"].ewm(span=9).mean()
        df["ema21"] = df["close"].ewm(span=21).mean()

        df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=7).rsi()

        adx = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=14)

        df["adx"] = adx.adx()
        df["di_plus"] = adx.adx_pos()

        df["ma_chan_high"] = df["high"].rolling(10).mean()

        return df


# ================= ATM OPTION =================

def get_atm_option():

    quote = get_quote("NSE:NIFTY 50")

    nifty_price = quote["NSE:NIFTY 50"]["last_price"]

    strike = round(nifty_price / 50) * 50

    nifty = st.session_state.nifty_options

    expiry = sorted(nifty["expiry"].unique())[0]

    option = nifty[
        (nifty["expiry"] == expiry) &
        (nifty["strike"] == strike) &
        (nifty["instrument_type"] == "CE")
    ]

    return option.iloc[0], nifty_price


# ================= DATA LOADER =================

def load_initial_data(token):

    data = kite.historical_data(
        token,
        datetime.today() - timedelta(days=1),
        datetime.today(),
        "minute"
    )

    df = pd.DataFrame(data)

    df = StrategyLogic.compute(df)

    st.session_state.market_data = df.tail(300)

    st.session_state.last_candle_time = df.iloc[-1]["date"]


def update_if_new_candle(token):

    latest = kite.historical_data(
        token,
        datetime.today() - timedelta(minutes=2),
        datetime.today(),
        "minute"
    )

    if not latest:
        return

    latest_df = pd.DataFrame(latest)

    if latest_df.empty:
        return

    new_time = latest_df.iloc[-1]["date"]

    if st.session_state.last_candle_time is None:
        st.session_state.last_candle_time = new_time
        return

    if new_time > st.session_state.last_candle_time:

        new_row = latest_df.iloc[-1:]

        df = pd.concat([st.session_state.market_data, new_row]).drop_duplicates("date").tail(300)

        df = StrategyLogic.compute(df.tail(200))

        st.session_state.market_data = df

        st.session_state.last_candle_time = new_time

# ================= BACKTEST =================

def run_backtest(df):

    trades = []

    for i in range(120, len(df)-1):

        signal = df.iloc[i]
        entry = df.iloc[i+1]

        conditions = [
            entry.close > entry.ma_chan_high,
            signal.ema9 < signal.ema21 and entry.ema9 > entry.ema21,
            entry.close > entry.jurik,
            signal.rsi < 60 and entry.rsi > 60,
            entry.di_plus > 20,
            entry.adx > 30,
            not entry.sqz_on and entry.momentum > 0,
            signal.wt1 < signal.wt2 and entry.wt1 > entry.wt2
        ]

        if all(conditions):

            trades.append({
                "Entry Time": entry.date,
                "Entry Price": entry.close,
                "Exit Time": entry.date,
                "Exit Price": entry.low,
                "P/L": entry.low - entry.close
            })

    return trades


# ================= BACKTEST MODE =================

if mode == "Backtest":

    st.title("📊 Backtest")

    option, nifty_price = get_atm_option()

    token = option.instrument_token

    data = kite.historical_data(
        token,
        datetime.today() - timedelta(days=1),
        datetime.today(),
        "minute"
    )

    df = pd.DataFrame(data)

    df = StrategyLogic.compute(df)

    trades = run_backtest(df)

    if trades:

        df_trades = pd.DataFrame(trades)

        st.dataframe(df_trades)

        st.download_button(
            "Download Backtest CSV",
            df_trades.to_csv(index=False),
            "backtest.csv"
        )

    else:
        st.warning("No trades found")

    st.stop()


# ================= LIVE MODE =================

st.title("🚀 NIFTY Strategy Dashboard")

option, nifty_price = get_atm_option()

token = option.instrument_token

if st.session_state.market_data is None:
    load_initial_data(token)
else:
    update_if_new_candle(token)

df = st.session_state.market_data

last_candle_time = df.iloc[-1]["date"]
seconds_left = 60 - last_candle_time.second

st.info(f"⏱ Next candle refresh in **{seconds_left} seconds**")
st.caption(f"Last candle time: {last_candle_time.strftime('%H:%M:%S')}")

col1, col2 = st.columns(2)

col1.metric("NIFTY", round(nifty_price,2))
col2.metric("ATM Option", option.tradingsymbol)

signal = df.iloc[-2]
entry = df.iloc[-1]

conditions = [
    entry.close > entry.ma_chan_high,
    signal.ema9 < signal.ema21 and entry.ema9 > entry.ema21,
    entry.close > entry.jurik,
    signal.rsi < 60 and entry.rsi > 60,
    entry.di_plus > 20,
    entry.adx > 30,
    not entry.sqz_on and entry.momentum > 0,
    signal.wt1 < signal.wt2 and entry.wt1 > entry.wt2
]

names = [
"MA Channel",
"EMA Cross",
"Jurik Trend",
"RSI Momentum",
"DMI Strength",
"ADX Trend",
"Squeeze Release",
"WaveTrend Cross"
]

st.subheader("Checklist")

cols = st.columns(4)

for i,r in enumerate(conditions):
    cols[i%4].write(f"{'🟢' if r else '🔴'} {names[i]}")

if all(conditions):
    st.success("🔥 TRADE SIGNAL")
else:
    st.warning("⏳ WAIT")

price = df.iloc[-1].close

st.metric("Option Price", round(price,2))

# ================= TRADE LOG =================

col1, col2 = st.columns(2)

if col1.button("ENTRY", use_container_width=True):

    if not st.session_state.trades or st.session_state.trades[-1]["Status"]=="Closed":

        st.session_state.trades.append({
            "Entry Time": entry.date,
            "Entry Price": float(entry.close),
            "Status":"Open"
        })

        st.toast("Entry recorded")


if col2.button("EXIT", use_container_width=True):

    if st.session_state.trades and st.session_state.trades[-1]["Status"]=="Open":

        trade = st.session_state.trades[-1]

        trade["Exit Time"] = entry.date
        trade["Exit Price"] = float(entry.close)
        trade["Status"] = "Closed"
        trade["P/L"] = trade["Exit Price"] - trade["Entry Price"]

        st.toast("Exit recorded")
        
if st.session_state.trades:

    st.subheader("Trade Log")

    df_trades = pd.DataFrame(st.session_state.trades)

    st.dataframe(df_trades)

    st.download_button(
        "Download CSV",
        df_trades.to_csv(index=False),
        "trades.csv"
    )

st.caption(f"Last candle time: {df.iloc[-1]['date'].strftime('%H:%M:%S')}")