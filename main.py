import streamlit as st
import pandas as pd
from kiteconnect import KiteConnect
from datetime import datetime, timedelta
import ta
from streamlit_autorefresh import st_autorefresh

# ================= CONFIG =================

API_KEY = st.secrets["API_KEY"]
API_SECRET = st.secrets["API_SECRET"]

st.set_page_config(page_title="NIFTY Strategy", layout="wide")

st_autorefresh(interval=3000)

kite = KiteConnect(api_key=API_KEY)

# ---------------- SESSION STATE ----------------

state = st.session_state

state.setdefault("access_token", None)
state.setdefault("df", None)
state.setdefault("last_candle", None)
state.setdefault("trades", [])
state.setdefault("skip_update", False)

# ---------------- LOGIN ----------------

params = st.query_params

if "request_token" in params:
    data = kite.generate_session(params["request_token"], api_secret=API_SECRET)
    state.access_token = data["access_token"]
    st.query_params.clear()

if not state.access_token:
    st.link_button("Login Zerodha", kite.login_url())
    st.stop()

kite.set_access_token(state.access_token)

# ---------------- CACHE ----------------

@st.cache_data(ttl=86400)
def load_instruments():
    inst = pd.DataFrame(kite.instruments("NFO"))
    return inst[(inst.name=="NIFTY") & (inst.segment=="NFO-OPT")]

@st.cache_data(ttl=5)
def get_quote():
    return kite.quote(["NSE:NIFTY 50"])["NSE:NIFTY 50"]["last_price"]

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

# ---------------- ATM OPTION ----------------

def get_option():

    price = get_quote()

    strike = round(price/50)*50

    inst = load_instruments()

    expiry = sorted(inst.expiry.unique())[0]

    opt = inst[
        (inst.expiry==expiry) &
        (inst.strike==strike) &
        (inst.instrument_type=="CE")
    ]

    return opt.iloc[0], price

# ---------------- DATA ----------------

def load_data(token):

    data = kite.historical_data(
        token,
        datetime.today()-timedelta(days=1),
        datetime.today(),
        "minute"
    )

    df = pd.DataFrame(data)

    df = compute(df)

    state.df = df.tail(200)

    state.last_candle = df.iloc[-1].date

def update_data(token):

    latest = kite.historical_data(
        token,
        datetime.today()-timedelta(minutes=2),
        datetime.today(),
        "minute"
    )

    if not latest:
        return

    new = pd.DataFrame(latest).iloc[-1]

    if new.date > state.last_candle:

        df = pd.concat([state.df, pd.DataFrame([new])])

        df = df.drop_duplicates("date").tail(200)

        df = compute(df)

        state.df = df
        state.last_candle = new.date

# ---------------- UI ----------------

st.title("NIFTY Strategy Dashboard")

opt, price = get_option()

token = opt.instrument_token

if not state.skip_update:

    if state.df is None:
        load_data(token)
    else:
        update_data(token)

else:
    state.skip_update = False

df = state.df

last = df.iloc[-1].date

sec = 60-last.second

st.info(f"Next candle in {sec}s")

col1,col2 = st.columns(2)

col1.metric("NIFTY",round(price,2))
col2.metric("ATM",opt.tradingsymbol)

# ---------------- SIGNAL ----------------

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
    st.success("TRADE SIGNAL")
else:
    st.warning("WAIT")

# ---------------- TRADES ----------------

c1,c2 = st.columns(2)

if c1.button("ENTRY"):

    state.skip_update=True

    if not state.trades or state.trades[-1]["Status"]=="Closed":

        state.trades.append({
            "Entry Time":entry.date,
            "Entry Price":float(entry.close),
            "Status":"Open"
        })

if c2.button("EXIT"):

    state.skip_update=True

    if state.trades and state.trades[-1]["Status"]=="Open":

        t=state.trades[-1]

        t["Exit Time"]=entry.date
        t["Exit Price"]=float(entry.close)
        t["Status"]="Closed"
        t["P/L"]=t["Exit Price"]-t["Entry Price"]

# ---------------- TRADE LOG ----------------

if state.trades:

    st.subheader("Trades")

    tdf=pd.DataFrame(state.trades)

    st.dataframe(tdf,use_container_width=True)

    st.download_button(
        "Download",
        tdf.to_csv(index=False),
        "trades.csv"
    )