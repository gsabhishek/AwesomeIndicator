import streamlit as st
import pandas as pd
from kiteconnect import KiteConnect
from datetime import datetime, timedelta
import ta
from streamlit_autorefresh import st_autorefresh

# ================= CONFIG =================

API_KEY = st.secrets["API_KEY"]
API_SECRET = st.secrets["API_SECRET"]

# ================= SESSION STATE =================
st.set_page_config(page_title="NIFTY Strategy", layout="wide")

state = st.session_state

if "initialized" not in state:
    state.access_token = None
    state.df = None
    state.last_candle = None
    state.trades = []
    state.skip_update = False
    state.current_candle = None
    state.current_minute = None
    state.token = None
    state.initialized = True

st_autorefresh(interval=4000)

kite = KiteConnect(api_key=API_KEY)

# ================= LOGIN =================

params = st.query_params

request_token = params.get("request_token")

if request_token:
    data = kite.generate_session(request_token, api_secret=API_SECRET)
    state.access_token = data["access_token"]
    st.query_params.clear()

if not state.access_token:
    st.link_button("Login Zerodha", kite.login_url())
    st.stop()

kite.set_access_token(state.access_token)

# ================= CACHE =================

@st.cache_data(ttl=86400)
def load_instruments():
    inst = pd.DataFrame(kite.instruments("NFO"))
    return inst[(inst.name == "NIFTY") & (inst.segment == "NFO-OPT")]

def get_quote():
    try:
        return kite.quote(["NSE:NIFTY 50"])["NSE:NIFTY 50"]["last_price"]
    except:
        st.warning("Quote fetch failed. Retrying...")
        st.stop()

def get_option_price(symbol):
    try:
        q = kite.quote([f"NFO:{symbol}"])
        return q[f"NFO:{symbol}"]["last_price"]
    except:
        st.warning("Option price fetch failed.")
        st.stop()

# ================= INDICATORS =================

def jurik_ma(series, length=8):

    beta = 0.45 * (length - 1) / (0.45 * (length - 1) + 2)
    alpha = beta ** 2

    jma = series.copy()

    for i in range(1, len(series)):
        jma.iloc[i] = alpha * series.iloc[i] + (1 - alpha) * jma.iloc[i - 1]

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
    m_avg = df["close"].rolling(length).mean()

    squeeze_on = False
    momentum = df["close"] - m_avg

    return squeeze_on, momentum


class StrategyLogic:

    @staticmethod
    def compute(df):
        df = df.copy()

        df["jma_fast"] = jurik_ma(df["close"], 8)
        df["jma_slow"] = jurik_ma(df["close"], 12)

        df["wt1"], df["wt2"] = wavetrend(df)

        df["sqz_on"], df["momentum"] = squeeze_momentum(df)

        df["ema9"] = df["close"].ewm(span=9).mean()
        df["ema21"] = df["close"].ewm(span=21).mean()

        df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=7).rsi()

        adx = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=14)

        df["adx"] = adx.adx()
        df["di_plus"] = adx.adx_pos()
        df["di_minus"] = adx.adx_neg()

        df["ma_chan_high"] = df["high"].rolling(10).mean()
        df["ma_chan_low"] = df["low"].rolling(10).mean()

        return df

# ================= ATM OPTION =================

def get_option():

    price = get_quote()

    strike = round(price / 50) * 50

    if "nifty_options" not in state:
        inst = load_instruments()
        state.nifty_options = inst

    inst = state.nifty_options

    expiry = sorted(inst.expiry.unique())[0]

    opt = inst[
        (inst.expiry == expiry)
        & (inst.strike == strike)
        & (inst.instrument_type == "CE")
    ]

    if opt.empty:
        st.error("Option not found")
        st.stop()

    return opt.iloc[0], price

# ================= DATA =================

def update_local_candle(price):

    now = datetime.now()
    minute = now.replace(second=0, microsecond=0)

    if state.current_minute != minute:

        if state.current_candle is not None:

            new_row = pd.DataFrame([state.current_candle])

            state.df = pd.concat([state.df, new_row]).tail(200)

            state.df = StrategyLogic.compute(state.df)

        state.current_minute = minute

        state.current_candle = {
            "date": minute,
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": 0
        }

    else:

        state.current_candle["high"] = max(state.current_candle["high"], price)
        state.current_candle["low"] = min(state.current_candle["low"], price)
        state.current_candle["close"] = price


def load_data(token):

    data = kite.historical_data(
        token,
        datetime.now() - timedelta(days=1),
        datetime.now(),
        "minute"
    )

    if not data:
        st.error("No historical data returned from Kite.")
        st.stop()

    df = pd.DataFrame(data)

    # Ensure datetime column exists
    if "date" not in df.columns and "datetime" in df.columns:
        df.rename(columns={"datetime": "date"}, inplace=True)

    df["date"] = pd.to_datetime(df["date"])

    df = StrategyLogic.compute(df)

    state.df = df.tail(200).reset_index(drop=True)

    state.last_candle = state.df.iloc[-1]["date"]

# ================= UI =================

st.title("NIFTY Strategy Dashboard")

opt, price = get_option()

option_price = get_option_price(opt.tradingsymbol)

token = opt.instrument_token

# ================= TOKEN SWITCH =================

if state.token != token:

    state.df = None
    state.last_candle = None
    state.current_candle = None
    state.current_minute = None
    state.token = token

# ================= LOAD DATA =================

if state.df is None:

    load_data(token)

    state.current_minute = pd.to_datetime(state.df.iloc[-1]["date"]).replace(second=0, microsecond=0)

# ================= BUILD LIVE CANDLE =================

update_local_candle(option_price)

# ================= TRADE BUTTONS =================

c1, c2 = st.columns(2)

entry_click = c1.button("ENTRY")
exit_click = c2.button("EXIT")

if entry_click:

    state.skip_update = True

    if not state.trades or state.trades[-1]["Status"] == "Closed":

        state.trades.append(
            {
                "Entry Time": datetime.now(),
                "Entry Price": option_price,
                "Status": "Open",
            }
        )

if exit_click:

    state.skip_update = True

    if state.trades and state.trades[-1]["Status"] == "Open":

        t = state.trades[-1]

        t["Exit Time"] = datetime.now()
        t["Exit Price"] = option_price
        t["Status"] = "Closed"
        t["P/L"] = t["Exit Price"] - t["Entry Price"]

# ================= DATA UPDATE =================

df = state.df

if state.current_candle is not None:

    live_row = pd.DataFrame([state.current_candle])

    df = pd.concat([df, live_row]).tail(200)

    if state.current_candle is not None:

        live_row = pd.DataFrame([state.current_candle])

        df = pd.concat([state.df, live_row]).tail(200)

        df = StrategyLogic.compute(df)

# ================= DASHBOARD =================

sec = max(1, 60 - datetime.now().second)

st.info(f"Next candle in {sec}s")

col1, col2 = st.columns(2)

col1.metric("NIFTY", round(price, 2))
col2.metric("Option", round(option_price, 2))

# ================= SIGNAL =================

if len(df) < 2:
    st.warning("Waiting for enough candle data...")
    st.stop()
signal = df.iloc[-2]
entry = df.iloc[-1]

conditions = [

    entry.close > entry.ma_chan_high,

    signal.ema9 < signal.ema21 and entry.ema9 > entry.ema21,

    entry.jma_fast > entry.jma_slow,

    signal.rsi < 60 and entry.rsi > 60,

    entry.di_plus > 20,

    entry.adx > 50 and entry.adx > signal.adx,

    entry.momentum > 0,

    signal.wt1 < signal.wt2 and entry.wt1 > entry.wt2

]

names = [
    "MA Channel",
    "EMA Cross",
    "Jurik Trend",
    "RSI Cross 60",
    "DMI Strength",
    "ADX Trend",
    "Squeeze Momentum",
    "WaveTrend Cross",
]

st.subheader("Checklist")

cols = st.columns(4)

for i, r in enumerate(conditions):
    cols[i % 4].write(f"{'🟢' if r else '🔴'} {names[i]}")

if all(conditions):
    st.success("🔥 TRADE SIGNAL")
else:
    st.warning("WAIT")

# ================= INDICATOR VALUES =================

st.subheader("Indicator Values")

vals = {

    "JMA Fast": round(entry.jma_fast, 2),
    "JMA Slow": round(entry.jma_slow, 2),
    "RSI": round(entry.rsi, 2),
    "ADX": round(entry.adx, 2),
    "DI+": round(entry.di_plus, 2),
    "WT1": round(entry.wt1, 2),
    "WT2": round(entry.wt2, 2),
    "Momentum": round(entry.momentum, 2),
    "EMA9": round(entry.ema9, 2),
    "EMA21": round(entry.ema21, 2),

}

st.json(vals)

# ================= TRADE LOG =================

if state.trades:

    st.subheader("Trades")

    tdf = pd.DataFrame(state.trades)

    st.dataframe(tdf, use_container_width=True)

    st.download_button(
        "Download",
        tdf.to_csv(index=False),
        "trades.csv"
    )

# ================= LIVE CANDLE =================

if state.current_candle:

    st.subheader("Live Candle")

    st.write({
        "Open": state.current_candle["open"],
        "High": state.current_candle["high"],
        "Low": state.current_candle["low"],
        "Close": state.current_candle["close"]
    })