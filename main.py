import streamlit as st
import pandas as pd
from kiteconnect import KiteConnect
from datetime import datetime, timedelta
import ta
from streamlit_autorefresh import st_autorefresh

# ================= CONFIG =================
API_KEY = st.secrets["API_KEY"]
API_SECRET = st.secrets["API_SECRET"]

NIFTY_TOKEN = 256265

st.set_page_config(page_title="NIFTY Strategy Dashboard", layout="wide")

st_autorefresh(interval=60 * 1000, key="refresh")

# ================= SESSION STATE =================
if "trades" not in st.session_state:
    st.session_state.trades = []

if "access_token" not in st.session_state:
    st.session_state.access_token = None

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
@st.cache_data(ttl=3600)
def get_instruments():
    return pd.DataFrame(kite.instruments("NFO"))

@st.cache_data(ttl=50)
def get_data(token, start, end):
    return kite.historical_data(
        token,
        start,
        end,
        "minute"
    )

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

    quote = kite.quote(["NSE:NIFTY 50"])

    if "NSE:NIFTY 50" not in quote:
        st.error("NIFTY quote unavailable")
        st.stop()

    nifty_price = quote["NSE:NIFTY 50"]["last_price"]

    strike = round(nifty_price / 50) * 50

    inst = get_instruments()

    nifty = inst[
        (inst["name"] == "NIFTY") &
        (inst["segment"] == "NFO-OPT")
    ]

    expiry = sorted(nifty["expiry"].unique())[0]

    option = nifty[
        (nifty["expiry"] == expiry) &
        (nifty["strike"] == strike) &
        (nifty["instrument_type"] == "CE")
    ]

    if option.empty:
        st.error("ATM option not found")
        st.stop()

    return option.iloc[0], nifty_price


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

    end = datetime.now()
    start = end - timedelta(hours=6)

    data = get_data(token, start, end)

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

now = datetime.now()
seconds_left = (60 - now.second) % 60

st.info(f"⏱ Next candle refresh in **{seconds_left} seconds**")

option, nifty_price = get_atm_option()

symbol = option.tradingsymbol
token = option.instrument_token

col1, col2 = st.columns(2)

col1.metric("NIFTY", round(nifty_price,2))
col2.metric("ATM Option", symbol)

end = datetime.now()
start = end - timedelta(hours=6)

data = get_data(token, start, end)
df = pd.DataFrame(data)

df = StrategyLogic.compute(df)

if df.empty:
    st.warning("No data received")
    st.stop()

if len(df) < 3:
    st.warning("Waiting for candles...")
    st.stop()
    
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

if col1.button("ENTRY"):
    st.session_state.trades.append({
        "Entry Time": entry.date,
        "Entry Price": entry.close,
        "Status":"Open"
    })

if col2.button("EXIT"):
    if st.session_state.trades and st.session_state.trades[-1]["Status"]=="Open":

        st.session_state.trades[-1].update({
            "Exit Time": entry.date,
            "Exit Price": entry.low,
            "Status":"Closed",
            "P/L": entry.low - st.session_state.trades[-1]["Entry Price"]
        })

if st.session_state.trades:

    st.subheader("Trade Log")

    df_trades = pd.DataFrame(st.session_state.trades)

    st.dataframe(df_trades)

    st.download_button(
        "Download CSV",
        df_trades.to_csv(index=False),
        "trades.csv"
    )

# ================= LAST UPDATED =================
st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")