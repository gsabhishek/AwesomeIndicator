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
import json
import requests
import base64

IST = pytz.timezone("Asia/Kolkata")

logging.basicConfig(level=logging.INFO)

def load_file_from_github(repo_path, local_file):

    try:
        token = st.secrets["GITHUB_TOKEN"]
        repo = st.secrets["GITHUB_REPO"]

        url = f"https://api.github.com/repos/{repo}/contents/{repo_path}"

        headers = {"Authorization": f"token {token}"}

        r = requests.get(url, headers=headers)

        if r.status_code == 200:
            content = base64.b64decode(r.json()["content"])

            os.makedirs(os.path.dirname(local_file), exist_ok=True)

            with open(local_file, "wb") as f:
                f.write(content)

            return True

    except Exception as e:
        logging.error(f"GitHub load failed: {e}")

    return False

def save_file_to_github(local_file, repo_path):
    try:
        token = st.secrets["GITHUB_TOKEN"]
        repo = st.secrets["GITHUB_REPO"]

        url = f"https://api.github.com/repos/{repo}/contents/{repo_path}"

        with open(local_file, "rb") as f:
            content = base64.b64encode(f.read()).decode()

        headers = {"Authorization": f"token {token}"}

        r = requests.get(url, headers=headers)
        sha = r.json().get("sha") if r.status_code == 200 else None

        data = {
            "message": f"Update {repo_path}",
            "content": content,
            "branch": "main"
        }

        if sha:
            data["sha"] = sha

        requests.put(url, headers=headers, json=data)

    except Exception as e:
        logging.error(f"GitHub save failed: {e}")

# ================= CONFIG =================

API_KEY = st.secrets["API_KEY"]
API_SECRET = st.secrets["API_SECRET"]

TOKEN_FILE = "token.txt"
TRADES_FILE = "trades.csv"

st_autorefresh(interval=10000)

kite = KiteConnect(api_key=API_KEY)

# ================= SESSION SETTINGS =================

import json

SETTINGS_FILE = "settings.json"

DEFAULT_SETTINGS = {
    "ema_fast": 9,
    "ema_slow": 21,
    "jma_fast_len": 8,
    "jma_slow_len": 12,
    "rsi_window": 7,
    "rsi_threshold": 50,
    "adx_threshold": 50,
    "di_plus_threshold": 20,
    "ma_chan_window": 10
}

def load_settings():
    load_file_from_github("data/settings.json", SETTINGS_FILE)
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "r") as f:
            return json.load(f)
    return DEFAULT_SETTINGS.copy()

def save_settings(settings):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f)
    save_file_to_github(SETTINGS_FILE, "data/settings.json")

# ================= SESSION STATE =================

state = st.session_state

defaults = {
    "access_token": None,
    "df": None,
    "last_candle": None,
    "trades": [],
    "token": None,
    "initialized": True,
    "last_trade_candle": None,
    "compute_params": {},
    "option_type": "CALL"
}

for k, v in defaults.items():
    if k not in state:
        state[k] = v

if "initialized" not in state:
    state.access_token = None
    state.df = None
    state.last_candle = None
    state.trades = []
    state.token = None
    state.initialized = True
    state.last_trade_candle = None
    state.compute_params = {}
    state.option_type = "CALL"

if "settings" not in state:
    state.settings = load_settings()

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

    if os.path.exists(TOKEN_FILE):
        os.remove(TOKEN_FILE)

    state.access_token = None
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
    return inst[(inst.name.str.contains("NIFTY")) & (inst.segment == "NFO-OPT")]

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


def squeeze_momentum(df, length=20, mult=2.0, lengthKC=20, multKC=1.5):

    close = df["close"]
    high = df["high"]
    low = df["low"]

    # ---------------- Bollinger Bands ----------------
    basis = close.rolling(length).mean()
    dev = close.rolling(length).std() * mult

    upperBB = basis + dev
    lowerBB = basis - dev

    # ---------------- Keltner Channel ----------------
    maKC = close.rolling(lengthKC).mean()

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    rangeKC = tr
    rangema = rangeKC.rolling(lengthKC).mean()

    upperKC = maKC + rangema * multKC
    lowerKC = maKC - rangema * multKC

    # ---------------- Squeeze Conditions ----------------
    sqz_on = (lowerBB > lowerKC) & (upperBB < upperKC)
    sqz_off = (lowerBB < lowerKC) & (upperBB > upperKC)

    # ---------------- Momentum Calculation ----------------
    highest_high = high.rolling(lengthKC).max()
    lowest_low = low.rolling(lengthKC).min()

    avg_price = (highest_high + lowest_low) / 2
    mid = (avg_price + close.rolling(lengthKC).mean()) / 2

    val = close - mid

    x = pd.Series(range(lengthKC))

    def linreg_slope(y):
        return pd.Series(y).cov(x) / x.var()

    momentum = val.rolling(lengthKC).apply(linreg_slope, raw=False)
    
    df["sqz_on"] = sqz_on
    df["sqz_off"] = sqz_off

    return sqz_on, momentum


class StrategyLogic:

    @staticmethod
    def compute(df, ema_fast=9, ema_slow=21, jma_fast_len=8, jma_slow_len=12,
                rsi_window=7, ma_chan_window=10):

        df = df.copy()

        df["jma_fast"] = jurik_ma(df["close"], jma_fast_len)
        df["jma_slow"] = jurik_ma(df["close"], jma_slow_len)

        df["wt1"], df["wt2"] = wavetrend(df)

        df["sqz_on"], df["momentum"] = squeeze_momentum(df)

        df["ema_fast"] = df["close"].ewm(span=ema_fast, adjust=False).mean()
        df["ema_slow"] = df["close"].ewm(span=ema_slow, adjust=False).mean()
        # Detect crossover
        df["ema_cross_up"] = (
            (df["ema_fast"] > df["ema_slow"]) &
            (df["ema_fast"].shift(1) <= df["ema_slow"].shift(1))
        )

        df["ema_cross_down"] = (
            (df["ema_fast"] < df["ema_slow"]) &
            (df["ema_fast"].shift(1) >= df["ema_slow"].shift(1))
        )

        df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=rsi_window).rsi()

        adx = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=14)

        df["adx"] = adx.adx()
        df["di_plus"] = adx.adx_pos()
        df["di_minus"] = adx.adx_neg()

        df["ma_chan_high"] = df["high"].rolling(ma_chan_window).mean()
        df["ma_chan_low"] = df["low"].rolling(ma_chan_window).mean()

        return df

# ================= OPTION SELECT =================

def get_option():

    if "nifty_options" not in state:
        state.nifty_options = load_instruments()

    inst = state.nifty_options

    if inst is None or len(inst) == 0:
        st.error("Instrument list not loaded")
        st.stop()

    expiry = sorted(inst.expiry.unique())[0]

    inst_type = "CE" if state.option_type == "CALL" else "PE"

    ce_options = inst[
        (inst.expiry == expiry) &
        (inst.instrument_type == inst_type)
    ].copy()

    if ce_options.empty:
        st.error(f"No {inst_type} options found")
        st.stop()

    spot = get_quote()

    if spot == 0:
        st.warning("Waiting for NIFTY price...")
        st.stop()

    atm = round(spot / 50) * 50

    ce_options = ce_options[
        (ce_options.strike >= atm - 250) &
        (ce_options.strike <= atm + 250)
    ]

    if ce_options.empty:
        st.warning("ATM strikes not available, showing full list")
        ce_options = inst[
            (inst.expiry == expiry) &
            (inst.instrument_type == inst_type)
        ].copy()

    ce_options = ce_options.sort_values("strike")

    ce_options["label"] = ce_options["strike"].astype(int).astype(str) + f" {inst_type}"

    selected = st.selectbox(
        f"Select NIFTY {inst_type}",
        ce_options["label"],
        index=len(ce_options)//2
    )

    row = ce_options.loc[ce_options["label"] == selected]

    if row.empty:
        st.error("Selected option not found")
        st.stop()

    return row.iloc[0]

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


def load_data(token, **kwargs):

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

    df = StrategyLogic.compute(df, **kwargs)

    state.df = df.tail(200).reset_index(drop=True)

    state.last_candle = state.df.iloc[-1]["date"]

# ================= UI =================

st.title("NIFTY Strategy Dashboard")

option_type = st.segmented_control(
    "Option Type",
    ["CALL", "PUT"],
    default="CALL"
)

if state.option_type != option_type:
    state.df = None
    state.token = None
    state.last_candle = None
    state.option_type = option_type

opt = get_option()

token = opt.instrument_token
symbol = opt.tradingsymbol

st.write("Selected Option:", symbol)

price = get_quote()

# ================= SIDEBAR PARAMETERS =================
with st.sidebar.expander("⚙️ Strategy Parameters", expanded=True):
    ema_fast = st.number_input("EMA Fast Period", min_value=2, max_value=50, step=1, value=state.settings["ema_fast"])
    ema_slow = st.number_input("EMA Slow Period", min_value=2, max_value=200, step=1, value=state.settings["ema_slow"])
    jma_fast_len = st.number_input("JMA Fast Length", min_value=2, max_value=50, step=1, value=state.settings["jma_fast_len"])
    jma_slow_len = st.number_input("JMA Slow Length", min_value=2, max_value=50, step=1, value=state.settings["jma_slow_len"])
    rsi_window = st.number_input("RSI Window", min_value=2, max_value=30, step=1, value=state.settings["rsi_window"])
    rsi_threshold = st.slider("RSI Threshold", min_value=30, max_value=70, value=state.settings["rsi_threshold"])
    adx_threshold = st.slider("ADX Threshold", min_value=10, max_value=80, value=state.settings["adx_threshold"])
    di_plus_threshold = st.slider("DI+ Threshold", min_value=5, max_value=50, value=state.settings["di_plus_threshold"])
    ma_chan_window = st.number_input("MA Channel Window", min_value=2, max_value=50, step=1, value=state.settings["ma_chan_window"])

st.sidebar.markdown("---")

new_settings = {
    "ema_fast": int(ema_fast),
    "ema_slow": int(ema_slow),
    "jma_fast_len": int(jma_fast_len),
    "jma_slow_len": int(jma_slow_len),
    "rsi_window": int(rsi_window),
    "rsi_threshold": rsi_threshold,
    "adx_threshold": adx_threshold,
    "di_plus_threshold": di_plus_threshold,
    "ma_chan_window": int(ma_chan_window)
}

if new_settings != state.settings:
    logging.info("Settings changed, saving...")
    save_settings(new_settings)
    state.settings = new_settings

_compute_params = {
    "ema_fast": state.settings["ema_fast"],
    "ema_slow": state.settings["ema_slow"],
    "jma_fast_len": state.settings["jma_fast_len"],
    "jma_slow_len": state.settings["jma_slow_len"],
    "rsi_window": state.settings["rsi_window"],
    "ma_chan_window": state.settings["ma_chan_window"],
}

# ================= TOKEN SWITCH =================

if state.token != token:
    state.df = None
    state.last_candle = None
    state.token = token
    state.compute_params = {}

# ================= LOAD DATA =================

if state.df is None:
    load_data(token, **_compute_params)
    state.compute_params = _compute_params

# ================= UPDATE CANDLE =================

latest = fetch_latest_candle(token)

if latest is None:
    st.warning("Waiting for candle data...")
    st.stop()

if latest is not None and state.last_candle != latest["date"]:

    new_row = pd.DataFrame([latest])

    state.df = pd.concat([state.df, new_row]).tail(200)

    state.df = StrategyLogic.compute(state.df, **_compute_params)

    state.last_candle = latest["date"]
    state.compute_params = _compute_params

elif state.compute_params != _compute_params and state.df is not None:
    state.df = StrategyLogic.compute(state.df, **_compute_params)
    state.compute_params = _compute_params

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

    entry.close > entry.ma_chan_high,
    entry.ema_cross_up,
    entry.jma_fast > entry.jma_slow,
    signal.rsi < state.settings["rsi_threshold"] and entry.rsi > state.settings["rsi_threshold"],
    entry.di_plus > state.settings["di_plus_threshold"],
    entry.adx > state.settings["adx_threshold"] and entry.adx > signal.adx,
    signal.sqz_on and entry.sqz_off and entry.momentum > 0,
    signal.wt1 < signal.wt2 and entry.wt1 > entry.wt2
]

names = [
    "MA Channel",
    "EMA Cross",
    "Jurik Trend",
    "RSI Cross",
    "DMI Strength",
    "ADX Trend",
    "Squeeze Momentum",
    "WaveTrend Cross"
]

rules = [
    "Price > MA Channel High",
    f"EMA{state.settings['ema_fast']} crossed above EMA{state.settings['ema_slow']}",
    "JMA Fast > JMA Slow",
    f"RSI crossed above {state.settings['rsi_threshold']}",
    f"DI+ > {state.settings['di_plus_threshold']}",
    f"ADX > {state.settings['adx_threshold']} & rising",
    "Squeeze release + Momentum > 0",
    "WT1 crossed above WT2",
]

met = sum(conditions)
total = len(conditions)

# ================= CRITERIA SUMMARY =================

summary_color = "green" if met == total else ("orange" if met >= total // 2 else "red")
st.markdown(
    f"<h3 style='color:{summary_color}'>{'✅' if met == total else '⚠️'} {met} / {total} criteria met</h3>",
    unsafe_allow_html=True
)
st.progress(met / total)

if all(conditions):
    st.success("🔥 TRADE SIGNAL")
else:
    st.warning("WAIT")

# ================= CHECKLIST =================

st.subheader("Checklist")

for i, r in enumerate(conditions):
    col_a, col_b, col_c = st.columns([2, 5, 3])

    col_a.write(f"{'🟢' if r else '🔴'} **{names[i]}**")
    col_b.write(rules[i])

    if i == 0:  # MA Channel
        col_c.write(f"Price: {entry.close:.1f} / MA High: {entry.ma_chan_high:.1f}")
    elif i == 1:  # EMA Cross
        col_c.write(f"EMA{state.settings['ema_fast']}: {entry.ema_fast:.0f} / EMA{state.settings['ema_slow']}: {entry.ema_slow:.0f}")
    elif i == 2:  # Jurik Trend
        col_c.write(f"JMAf: {entry.jma_fast:.1f} / JMAs: {entry.jma_slow:.1f}")
    elif i == 3:  # RSI Cross
        col_c.write(f"RSI: {entry.rsi:.1f}")
        col_c.progress(min(max(entry.rsi / 100, 0.0), 1.0))
    elif i == 4:  # DMI Strength
        col_c.write(f"DI+: {entry.di_plus:.1f}")
        col_c.progress(min(max(entry.di_plus / 100, 0.0), 1.0))
    elif i == 5:  # ADX Trend
        direction = "↑" if entry.adx > signal.adx else "↓"
        col_c.write(f"ADX: {entry.adx:.1f} ({direction})")
        col_c.progress(min(max(entry.adx / 100, 0.0), 1.0))
    elif i == 6:  # Squeeze Momentum
        col_c.write(f"Momentum: {entry.momentum:.2f}")
    elif i == 7:  # WaveTrend Cross
        col_c.write(f"WT1: {entry.wt1:.1f} / WT2: {entry.wt2:.1f}")

# ================= INDICATORS =================

st.subheader("Indicator Values")

indicator_metrics = [
    (f"EMA{state.settings['ema_fast']}", round(entry.ema_fast, 2), round(entry.ema_fast - signal.ema_fast, 2)),
    (f"EMA{state.settings['ema_slow']}", round(entry.ema_slow, 2), round(entry.ema_slow - signal.ema_slow, 2)),
    ("JMA Fast", round(entry.jma_fast, 2), round(entry.jma_fast - signal.jma_fast, 2)),
    ("JMA Slow", round(entry.jma_slow, 2), round(entry.jma_slow - signal.jma_slow, 2)),
    ("RSI", round(entry.rsi, 2), round(entry.rsi - signal.rsi, 2)),
    ("ADX", round(entry.adx, 2), round(entry.adx - signal.adx, 2)),
    ("DI+", round(entry.di_plus, 2), round(entry.di_plus - signal.di_plus, 2)),
    ("WaveTrend 1", round(entry.wt1, 2), round(entry.wt1 - signal.wt1, 2)),
    ("WaveTrend 2", round(entry.wt2, 2), round(entry.wt2 - signal.wt2, 2)),
    ("Momentum", round(entry.momentum, 2), round(entry.momentum - signal.momentum, 2)),
]

metric_cols = st.columns(4)
for idx, (label, value, delta) in enumerate(indicator_metrics):
    metric_cols[idx % 4].metric(label, value, delta)

# ================= TRADE LOG =================

if state.trades:

    st.subheader("Trade Log")

    tdf = pd.DataFrame(state.trades)
    tdf = tdf.fillna("")

    st.dataframe(tdf, width="stretch")

    st.download_button("Download", tdf.to_csv(index=False), "trades.csv")