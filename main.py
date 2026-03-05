import streamlit as st
st.set_page_config(page_title="NIFTY Strategy", layout="wide")

import pytz
import pandas as pd
from kiteconnect import KiteConnect
from datetime import datetime, timedelta
import ta
from streamlit_autorefresh import st_autorefresh
import os
import logging

IST = pytz.timezone("Asia/Kolkata")

logging.basicConfig(level=logging.INFO)

# ================= CONFIG =================

API_KEY = st.secrets["API_KEY"]
API_SECRET = st.secrets["API_SECRET"]

TOKEN_FILE = "token.txt"
TRADES_FILE = "trades.csv"

st_autorefresh(interval=10000)

kite = KiteConnect(api_key=API_KEY)

# ================= SESSION STATE =================

state = st.session_state

if "initialized" not in state:
    state.access_token = None
    state.df = None
    state.last_candle = None
    state.trades = []
    state.token = None
    state.initialized = True
    state.last_trade_candle = None

# ================= TOKEN LOAD =================

if os.path.exists(TOKEN_FILE) and state.access_token is None:
    state.access_token = open(TOKEN_FILE).read().strip()

# ================= LOGIN =================

request_token = st.query_params.get("request_token")

if request_token:
    data = kite.generate_session(request_token, api_secret=API_SECRET)
    state.access_token = data["access_token"]

    with open(TOKEN_FILE, "w") as f:
        f.write(state.access_token)

    st.query_params.clear()

if not state.access_token:
    st.link_button("Login Zerodha", kite.login_url())
    st.stop()

kite.set_access_token(state.access_token)

# ================= TOKEN VALIDATION =================

def is_token_valid():
    try:
        kite.profile()
        return True
    except:
        return False

if not is_token_valid():
    st.warning("Session expired. Login again.")
    os.remove(TOKEN_FILE)
    st.stop()

# ================= SAFE API =================

def safe_call(func):
    try:
        return func()
    except Exception as e:
        logging.error(e)
        st.warning("API issue. Retrying...")
        return None

# ================= CACHE =================

@st.cache_data(ttl=86400)
def load_instruments():
    inst = safe_call(lambda: pd.DataFrame(kite.instruments("NFO")))
    return inst[(inst.name == "NIFTY") & (inst.segment == "NFO-OPT")]

def get_quote():
    data = safe_call(lambda: kite.quote(["NSE:NIFTY 50"]))
    if data:
        return data["NSE:NIFTY 50"]["last_price"]
    return 0

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

    m_avg = df["close"].rolling(20).mean()

    momentum = df["close"] - m_avg

    squeeze_on = False

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

# ================= OPTION SELECT =================

def get_option():

    if "nifty_options" not in state:
        state.nifty_options = load_instruments()

    inst = state.nifty_options

    expiry = sorted(inst.expiry.unique())[0]

    ce_options = inst[
        (inst.expiry == expiry) &
        (inst.instrument_type == "CE")
    ].copy()

    spot = get_quote()

    atm = round(spot / 50) * 50

    ce_options = ce_options[
        (ce_options.strike >= atm - 250) &
        (ce_options.strike <= atm + 250)
    ]

    ce_options = ce_options.sort_values("strike")

    ce_options["label"] = ce_options["strike"].astype(int).astype(str) + " CE"

    selected = st.selectbox(
        "Select NIFTY Call Option",
        ce_options["label"],
        index=len(ce_options)//2
    )

    row = ce_options[ce_options["label"] == selected].iloc[0]

    return row

# ================= DATA =================

def fetch_latest_candle(token):

    data = safe_call(lambda: kite.historical_data(
        token,
        datetime.now(IST) - timedelta(minutes=2),
        datetime.now(IST),
        "minute"
    ))

    if not data:
        return None

    df = pd.DataFrame(data)

    if "datetime" in df.columns:
        df.rename(columns={"datetime": "date"}, inplace=True)

    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(None)

    return df.iloc[-1]


def load_data(token):

    data = safe_call(lambda: kite.historical_data(
        token,
        datetime.now(IST) - timedelta(days=1),
        datetime.now(IST),
        "minute"
    ))

    df = pd.DataFrame(data)

    if "datetime" in df.columns:
        df.rename(columns={"datetime": "date"}, inplace=True)

    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(None)

    df = StrategyLogic.compute(df)

    state.df = df.tail(200).reset_index(drop=True)

    state.last_candle = state.df.iloc[-1]["date"]

# ================= UI =================

st.title("NIFTY Strategy Dashboard")

opt = get_option()

token = opt.instrument_token
symbol = opt.tradingsymbol

st.write("Selected Option:", symbol)

price = get_quote()

# ================= TOKEN SWITCH =================

if state.token != token:
    state.df = None
    state.last_candle = None
    state.token = token

# ================= LOAD DATA =================

if state.df is None:
    load_data(token)

# ================= UPDATE CANDLE =================

latest = fetch_latest_candle(token)

if latest is None:
    st.warning("Waiting for candle data...")
    st.stop()

if latest is not None and state.last_candle != latest["date"]:

    new_row = pd.DataFrame([latest])

    state.df = pd.concat([state.df, new_row]).tail(200)

    state.df = StrategyLogic.compute(state.df)

    state.last_candle = latest["date"]

option_price = latest["close"] if latest is not None else price

df = state.df

# ================= LOAD PREVIOUS TRADES =================

if os.path.exists(TRADES_FILE) and not state.trades:
    state.trades = pd.read_csv(TRADES_FILE).to_dict("records")

# ================= TRADE BUTTONS =================

c1, c2 = st.columns(2)

if c1.button("ENTRY"):

    if latest is None:
        st.warning("No candle data yet. Try again.")
        st.stop()

    if not state.trades or state.trades[-1]["Status"] == "Closed":

        if state.last_trade_candle == latest["date"]:
            st.warning("Trade already taken this candle")
            st.stop()

        trade = {
            "Option": symbol,
            "Entry Time": latest["date"].strftime("%Y-%m-%d %H:%M:%S"),
            "Entry Price": option_price,
            "Status": "Open"
        }

        state.last_trade_candle = latest["date"]

        state.trades.append(trade)

        pd.DataFrame(state.trades).to_csv(TRADES_FILE, index=False)

if c2.button("EXIT"):

    if state.trades and state.trades[-1]["Status"] == "Open":

        t = state.trades[-1]

        t["Exit Time"] = latest["date"].strftime("%Y-%m-%d %H:%M:%S")
        t["Exit Price"] = option_price
        t["Status"] = "Closed"
        t["P/L"] = t["Exit Price"] - t["Entry Price"]

        pd.DataFrame(state.trades).to_csv(TRADES_FILE, index=False)

# ================= DASHBOARD =================

if latest is not None:
    sec = max(1, 60 - latest["date"].second)
else:
    sec = 60

st.info(f"Next candle in {sec}s")

col1, col2 = st.columns(2)

col1.metric("NIFTY", round(price, 2))
col2.metric("Option", round(option_price, 2))

# ================= SIGNAL =================

if df is None or len(df) < 30:
    st.warning("Indicators warming up...")
    st.stop()

signal = df.iloc[-2]
entry = df.iloc[-1]

conditions = [

    entry.close > entry.ma_chan_high, #Price above MA Channel
    signal.ema9 < signal.ema21 and entry.ema9 > entry.ema21, #EMA Bullish Crossover
    entry.jma_fast > entry.jma_slow, #Jurik Trend Bullish
    signal.rsi < 50 and entry.rsi > 50, #RSI Crossing 50
    entry.di_plus > 20, #DMI Strength
    entry.adx > 50 and entry.adx > signal.adx, #ADX Strengthening
    entry.momentum > 0, #Squeeze Momentum Positive
    signal.wt1 < signal.wt2 and entry.wt1 > entry.wt2 #WaveTrend Bullish Crossover
]

names = [
    "MA Channel",
    "EMA Cross",
    "Jurik Trend",
    "RSI Cross 60",
    "DMI Strength",
    "ADX Trend",
    "Squeeze Momentum",
    "WaveTrend Cross"
]

st.subheader("Checklist")

cols = st.columns(4)

for i, r in enumerate(conditions):
    cols[i % 4].write(f"{'🟢' if r else '🔴'} {names[i]}")

if all(conditions):
    st.success("🔥 TRADE SIGNAL")
else:
    st.warning("WAIT")

# ================= INDICATORS =================

st.subheader("Indicator Values")

vals = {
    "EMA9            ": round(entry.ema9, 2),
    "EMA21           ": round(entry.ema21, 2),
    "JuricMA Fast.   ": round(entry.jma_fast, 2),
    "JuricMA Slow.   ": round(entry.jma_slow, 2),
    "RSI             ": round(entry.rsi, 2),
    "ADX             ": round(entry.adx, 2),
    "DI+             ": round(entry.di_plus, 2),
    "WaveTrend 1     ": round(entry.wt1, 2),
    "WaveTrend 2     ": round(entry.wt2, 2),
    "SqueezeMomentum ": round(entry.momentum, 2),
}

st.json(vals)

# ================= TRADE LOG =================

if state.trades:

    st.subheader("Trade Log")

    tdf = pd.DataFrame(state.trades)
    tdf = tdf.fillna("")

    st.dataframe(tdf, use_container_width=True)

    st.download_button("Download", tdf.to_csv(index=False), "trades.csv")