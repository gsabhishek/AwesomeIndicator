import streamlit as st
st.set_page_config(page_title="NIFTY Strategy", layout="wide")

import pytz
import pandas as pd
import numpy as np
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

# ================= GITHUB PERSISTENCE =================

def load_file_from_github(repo_path, local_file):
    try:
        token = st.secrets["GITHUB_TOKEN"]
        repo  = st.secrets["GITHUB_REPO"]
        url   = f"https://api.github.com/repos/{repo}/contents/{repo_path}"
        r = requests.get(url, headers={"Authorization": f"token {token}"})
        if r.status_code == 200:
            content = base64.b64decode(r.json()["content"])
            parent = os.path.dirname(local_file)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(local_file, "wb") as f:
                f.write(content)
            return True
    except Exception as e:
        logging.error(f"GitHub load failed: {e}")
    return False

def save_file_to_github(local_file, repo_path):
    try:
        token = st.secrets["GITHUB_TOKEN"]
        repo  = st.secrets["GITHUB_REPO"]
        url   = f"https://api.github.com/repos/{repo}/contents/{repo_path}"
        with open(local_file, "rb") as f:
            content = base64.b64encode(f.read()).decode()
        r   = requests.get(url, headers={"Authorization": f"token {token}"})
        sha = r.json().get("sha") if r.status_code == 200 else None
        data = {"message": f"Update {repo_path}", "content": content, "branch": "main"}
        if sha:
            data["sha"] = sha
        requests.put(url, headers={"Authorization": f"token {token}"}, json=data)
    except Exception as e:
        logging.error(f"GitHub save failed: {e}")

# ================= CONFIG =================

API_KEY    = st.secrets["API_KEY"]
API_SECRET = st.secrets["API_SECRET"]

TOKEN_FILE  = "token.txt"
TRADES_FILE = "trades.csv"
SETTINGS_FILE = "settings.json"

st_autorefresh(interval=10000)

kite = KiteConnect(api_key=API_KEY)

# ================= SETTINGS =================

DEFAULT_SETTINGS = {
    "ema_fast":          9,
    "ema_slow":         21,
    "jma_window":        8,
    "jma_phase":        50,
    "jma_power":         1,
    "rsi_window":        7,
    "rsi_threshold":    50,
    "adx_window":       14,
    "adx_threshold":    50,
    "di_plus_threshold": 20,
    "ma_chan_upper_window": 20,
    "ma_chan_lower_window": 20,
    "bb_length":   20,
    "bb_mult":    2.0,
    "kc_length":   20,
    "kc_mult":    1.5,
    "wt_channel_length": 10,
    "wt_average_length": 21,
}

def load_settings():
    load_file_from_github("data/settings.json", SETTINGS_FILE)
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                saved = json.load(f)
            # merge so new keys from DEFAULT_SETTINGS always exist
            return {**DEFAULT_SETTINGS, **saved}
        except Exception:
            pass
    return DEFAULT_SETTINGS.copy()


def save_settings(settings):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=2)
    save_file_to_github(SETTINGS_FILE, "data/settings.json")

# ================= SESSION STATE =================

state = st.session_state

_state_defaults = {
    "access_token":    None,
    "df":              None,
    "last_candle":     None,
    "trades":          [],
    "token":           None,
    "last_trade_candle": None,
    "compute_params":  {},
    "option_type":     "CALL",
}
for k, v in _state_defaults.items():
    if k not in state:
        state[k] = v

if "settings" not in state:
    state.settings = load_settings()

# ================= TOKEN LOAD =================

if os.path.exists(TOKEN_FILE) and state.access_token is None:
    state.access_token = open(TOKEN_FILE).read().strip()

# ================= LOGIN =================

request_token = st.query_params.get("request_token")

if request_token:
    session_data       = kite.generate_session(request_token, api_secret=API_SECRET)
    state.access_token = session_data["access_token"]
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
    except Exception:
        return False

if not is_token_valid():
    st.warning("Session expired. Please login again.")
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

# ================= INSTRUMENTS =================

@st.cache_data(ttl=86400)
def load_instruments():
    inst = safe_call(lambda: pd.DataFrame(kite.instruments("NFO")))
    if inst is None:
        return pd.DataFrame()
    return inst[inst.name.str.contains("NIFTY") & (inst.segment == "NFO-OPT")]


def get_quote():
    data = safe_call(lambda: kite.quote(["NSE:NIFTY 50"]))
    if data:
        return data["NSE:NIFTY 50"]["last_price"]
    return 0

# ================= INDICATORS =================

def jurik_moving_average(series, length=8, phase=50, power=1):
    """Jurik Moving Average (JMA)."""
    src        = pd.Series(series.values, index=series.index, dtype=float)
    phase_ratio = np.clip(phase / 100 + 1.5, 0.5, 2.5)
    beta  = 0.45 * (length - 1) / (0.45 * (length - 1) + 2)
    alpha = beta ** power

    e0  = np.zeros(len(src))
    e1  = np.zeros(len(src))
    e2  = np.zeros(len(src))
    jma = np.zeros(len(src))

    for i in range(1, len(src)):
        e0[i]  = (1 - alpha) * src.iloc[i] + alpha * e0[i - 1]
        e1[i]  = (src.iloc[i] - e0[i]) * (1 - beta) + beta * e1[i - 1]
        e2[i]  = (e0[i] + phase_ratio * e1[i] - jma[i - 1]) * (1 - alpha) ** 2 + alpha ** 2 * e2[i - 1]
        jma[i] = e2[i] + jma[i - 1]

    return pd.Series(jma, index=series.index)


def wavetrend(df, channel_length=10, average_length=21):
    ap  = (df["high"] + df["low"] + df["close"]) / 3
    esa = ap.ewm(span=channel_length, adjust=False).mean()
    d   = (ap - esa).abs().ewm(span=channel_length, adjust=False).mean()
    ci  = (ap - esa) / (0.015 * d.replace(0, np.nan)).fillna(0)
    wt1 = ci.ewm(span=average_length, adjust=False).mean()
    wt2 = wt1.rolling(4).mean()
    return wt1, wt2


def squeeze_momentum_color(df, bb_length=20, bb_mult=2.0, kc_length=20, kc_mult=1.5):
    close, high, low = df["close"], df["high"], df["low"]

    # Bollinger Bands
    basis   = close.rolling(bb_length).mean()
    dev     = bb_mult * close.rolling(bb_length).std()
    upper_bb = basis + dev
    lower_bb = basis - dev

    # Keltner Channels (true range)
    ma    = close.rolling(kc_length).mean()
    tr    = pd.concat([high - low,
                       (high - close.shift()).abs(),
                       (low  - close.shift()).abs()], axis=1).max(axis=1)
    rangema  = tr.rolling(kc_length).mean()
    upper_kc = ma + rangema * kc_mult
    lower_kc = ma - rangema * kc_mult

    sqz_on = (lower_bb > lower_kc) & (upper_bb < upper_kc)

    # Momentum via linear regression
    highest_high = high.rolling(kc_length).max()
    lowest_low   = low.rolling(kc_length).min()
    mid          = ((highest_high + lowest_low) / 2 + close.rolling(kc_length).mean()) / 2
    val_input    = close - mid

    def linreg(series, length):
        x      = np.arange(length, dtype=float)
        x_mean = x.mean()
        denom  = ((x - x_mean) ** 2).sum()
        def _calc(y):
            y_mean = y.mean()
            slope  = ((x - x_mean) * (y - y_mean)).sum() / denom
            return y_mean + slope * (x[-1] - x_mean)
        return series.rolling(length).apply(_calc, raw=True)

    val  = linreg(val_input, kc_length)
    prev = val.shift(1)

    color = np.where(val > 0,
                     np.where(val >= prev, "lime",   "green"),
                     np.where(val <= prev, "red",    "maroon"))

    return pd.Series(color, index=df.index), val, sqz_on


def moving_average_channel_slopes(df, upper_length=20, lower_length=20):
    upper       = df["high"].rolling(upper_length).mean()
    lower       = df["low"].rolling(lower_length).mean()
    upper_slope = upper - upper.shift(1)
    lower_slope = lower - lower.shift(1)
    return upper_slope, lower_slope

# ================= STRATEGY =================

class StrategyLogic:

    @staticmethod
    def compute(df,
                ema_fast=9, ema_slow=21,
                jma_window=8, jma_phase=50, jma_power=1,
                rsi_window=7,
                adx_window=14,
                ma_chan_upper_window=20, ma_chan_lower_window=20,
                bb_length=20, bb_mult=2.0,
                kc_length=20, kc_mult=1.5,
                wt_channel_length=10, wt_average_length=21,
                **_):  # absorb any extra kwargs gracefully
        df = df.copy()
        if len(df) < max(adx_window, rsi_window, bb_length, kc_length, wt_average_length):
            return df
        # MA Channel
        upper_slope, lower_slope = moving_average_channel_slopes(
            df, upper_length=ma_chan_upper_window, lower_length=ma_chan_lower_window)
        df["ma_chan_upper_slope"] = upper_slope
        df["ma_chan_lower_slope"] = lower_slope
        df["ma_chan_upward"]      = (upper_slope > 0) & (lower_slope > 0)

        # EMAs
        df["ema_fast"] = df["close"].ewm(span=ema_fast, adjust=False).mean()
        df["ema_slow"] = df["close"].ewm(span=ema_slow, adjust=False).mean()

        # JMA
        df["jma"] = jurik_moving_average(df["close"], length=jma_window,
                                         phase=jma_phase, power=jma_power)

        # RSI
        df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=rsi_window).rsi()

        # ADX / DMI
        adx_ind     = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=adx_window)
        df["adx"]      = adx_ind.adx()
        df["di_plus"]  = adx_ind.adx_pos()
        df["di_minus"] = adx_ind.adx_neg()

        # Squeeze Momentum
        color_series, val_series, sqz_on = squeeze_momentum_color(
            df, bb_length=bb_length, bb_mult=bb_mult,
            kc_length=kc_length, kc_mult=kc_mult)
        df["sqz_color"]    = color_series
        df["sqz_momentum"] = val_series
        df["sqz_on"]       = sqz_on

        # WaveTrend
        df["wt1"], df["wt2"] = wavetrend(df,
                                          channel_length=wt_channel_length,
                                          average_length=wt_average_length)
        return df

# ================= OPTION SELECT =================

def get_option():
    if "nifty_options" not in state:
        state.nifty_options = load_instruments()

    inst = state.nifty_options
    if inst is None or inst.empty:
        st.error("Instrument list not loaded.")
        st.stop()

    expiry    = sorted(inst.expiry.unique())[0]
    inst_type = "CE" if state.option_type == "CALL" else "PE"

    opts = inst[(inst.expiry == expiry) & (inst.instrument_type == inst_type)].copy()

    if opts.empty:
        st.error(f"No {inst_type} options found.")
        st.stop()

    spot = get_quote()
    if spot == 0:
        st.warning("Waiting for NIFTY price...")
        st.stop()

    atm      = round(spot / 50) * 50
    filtered = opts[(opts.strike >= atm - 250) & (opts.strike <= atm + 250)]
    if not filtered.empty:
        opts = filtered

    opts          = opts.sort_values("strike")
    opts["label"] = opts["strike"].astype(int).astype(str) + f" {inst_type}"

    selected = st.selectbox(
        f"Select NIFTY {inst_type} Option",
        opts["label"],
        index=len(opts) // 2,
    )

    row = opts[opts["label"] == selected]
    if row.empty:
        st.error("Selected option not found.")
        st.stop()

    return row.iloc[0]

# ================= DATA =================

def fetch_latest_candle(token):
    data = safe_call(lambda: kite.historical_data(
        token,
        datetime.now(IST) - timedelta(minutes=2),
        datetime.now(IST),
        "minute",
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
        "minute",
    ))
    if not data:
        st.warning("No historical data returned.")
        st.stop()
    df = pd.DataFrame(data)
    if "datetime" in df.columns:
        df.rename(columns={"datetime": "date"}, inplace=True)
    df["date"]     = pd.to_datetime(df["date"], utc=True).dt.tz_convert(None)
    df             = StrategyLogic.compute(df, **kwargs)
    state.df       = df.tail(200).reset_index(drop=True)
    state.last_candle = state.df.iloc[-1]["date"]

# ================= SIDEBAR =================

with st.sidebar:
    st.markdown("### ⚙️ Strategy Parameters")
    with st.expander("EMA", expanded=True):
        ema_fast = st.number_input("EMA Fast Period",  min_value=2,  max_value=50,  step=1, value=state.settings["ema_fast"])
        ema_slow = st.number_input("EMA Slow Period",  min_value=2,  max_value=200, step=1, value=state.settings["ema_slow"])

    with st.expander("JMA", expanded=True):
        jma_window = st.number_input("JMA Window", min_value=2, max_value=50,  step=1,   value=state.settings["jma_window"])
        jma_phase  = st.slider("JMA Phase",         min_value=-100, max_value=100, step=1, value=state.settings["jma_phase"])
        jma_power  = st.number_input("JMA Power",  min_value=1, max_value=5,   step=1,   value=state.settings["jma_power"])

    with st.expander("RSI", expanded=True):
        rsi_window    = st.number_input("RSI Window",    min_value=2,  max_value=30, step=1,  value=state.settings["rsi_window"])
        rsi_threshold = st.slider("RSI Threshold",       min_value=30, max_value=70, value=state.settings["rsi_threshold"])

    with st.expander("ADX / DMI", expanded=True):
        adx_window         = st.number_input("ADX Window",   min_value=2,  max_value=50, step=1,  value=state.settings["adx_window"])
        adx_threshold      = st.slider("ADX Threshold",      min_value=10, max_value=80, value=state.settings["adx_threshold"])
        di_plus_threshold  = st.slider("DI+ Threshold",      min_value=5,  max_value=50, value=state.settings["di_plus_threshold"])

    with st.expander("MA Channel", expanded=False):
        ma_chan_upper_window = st.number_input("Upper Window", min_value=2, max_value=50, step=1, value=state.settings["ma_chan_upper_window"])
        ma_chan_lower_window = st.number_input("Lower Window", min_value=2, max_value=50, step=1, value=state.settings["ma_chan_lower_window"])

    with st.expander("Squeeze Momentum", expanded=False):
        bb_length = st.number_input("BB Length", min_value=5,   max_value=50, step=1,   value=state.settings["bb_length"])
        bb_mult   = st.number_input("BB Mult",   min_value=0.5, max_value=5.0, step=0.1, value=state.settings["bb_mult"])
        kc_length = st.number_input("KC Length", min_value=5,   max_value=50, step=1,   value=state.settings["kc_length"])
        kc_mult   = st.number_input("KC Mult",   min_value=0.5, max_value=5.0, step=0.1, value=state.settings["kc_mult"])

    with st.expander("WaveTrend", expanded=False):
        wt_channel_length = st.number_input("WT Channel Length", min_value=2, max_value=50, step=1, value=state.settings["wt_channel_length"])
        wt_average_length = st.number_input("WT Average Length", min_value=2, max_value=50, step=1, value=state.settings["wt_average_length"])

    st.markdown("---")

new_settings = {
    "ema_fast":              int(ema_fast),
    "ema_slow":              int(ema_slow),
    "jma_window":            int(jma_window),
    "jma_phase":             int(jma_phase),
    "jma_power":             int(jma_power),
    "rsi_window":            int(rsi_window),
    "rsi_threshold":         int(rsi_threshold),
    "adx_window":            int(adx_window),
    "adx_threshold":         int(adx_threshold),
    "di_plus_threshold":     int(di_plus_threshold),
    "ma_chan_upper_window":  int(ma_chan_upper_window),
    "ma_chan_lower_window":  int(ma_chan_lower_window),
    "bb_length":             int(bb_length),
    "bb_mult":               float(bb_mult),
    "kc_length":             int(kc_length),
    "kc_mult":               float(kc_mult),
    "wt_channel_length":     int(wt_channel_length),
    "wt_average_length":     int(wt_average_length),
}

if new_settings != state.settings:
    logging.info("Settings changed — saving...")
    save_settings(new_settings)
    state.settings = new_settings

_compute_params = {k: v for k, v in new_settings.items()}

# ================= MAIN UI =================

st.title("NIFTY Strategy Dashboard")

option_type = st.segmented_control("Option Type", ["CALL", "PUT"], default="CALL")

if state.option_type != option_type:
    state.df          = None
    state.token       = None
    state.last_candle = None
    state.option_type = option_type

opt    = get_option()
token  = opt.instrument_token
symbol = opt.tradingsymbol

st.caption(f"Selected option: **{symbol}**")

price = get_quote()

# ================= TOKEN / DATA SWITCH =================

if state.token != token:
    state.df             = None
    state.last_candle    = None
    state.token          = token
    state.compute_params = {}

if state.df is None:
    load_data(token, **_compute_params)
    state.compute_params = _compute_params

# ================= UPDATE CANDLE =================

latest = fetch_latest_candle(token)

if latest is None:
    st.warning("Waiting for candle data...")
    st.stop()

if state.last_candle != latest["date"]:
    new_row         = pd.DataFrame([latest])
    state.df        = pd.concat([state.df, new_row]).tail(200)
    state.df        = StrategyLogic.compute(state.df, **_compute_params)
    state.last_candle   = latest["date"]
    state.compute_params = _compute_params
elif state.compute_params != _compute_params:
    state.df        = StrategyLogic.compute(state.df, **_compute_params)
    state.compute_params = _compute_params

option_price = latest["close"]
df           = state.df

# ================= LOAD PREVIOUS TRADES =================

if os.path.exists(TRADES_FILE) and not state.trades:
    state.trades = pd.read_csv(TRADES_FILE).to_dict("records")

# ================= TRADE BUTTONS =================

c1, c2 = st.columns(2)

if c1.button("ENTRY", use_container_width=True):
    if not state.trades or state.trades[-1]["Status"] == "Closed":
        if state.last_trade_candle == latest["date"]:
            st.warning("Trade already taken this candle.")
        else:
            trade = {
                "Option":      symbol,
                "Entry Time":  latest["date"].strftime("%Y-%m-%d %H:%M:%S"),
                "Entry Price": option_price,
                "Status":      "Open",
            }
            state.last_trade_candle = latest["date"]
            state.trades.append(trade)
            pd.DataFrame(state.trades).to_csv(TRADES_FILE, index=False)

if c2.button("EXIT", use_container_width=True):
    if state.trades and state.trades[-1]["Status"] == "Open":
        t = state.trades[-1]
        t["Exit Time"]  = latest["date"].strftime("%Y-%m-%d %H:%M:%S")
        t["Exit Price"] = option_price
        t["Status"]     = "Closed"
        t["P/L"]        = round(t["Exit Price"] - t["Entry Price"], 2)
        pd.DataFrame(state.trades).to_csv(TRADES_FILE, index=False)

# ================= DASHBOARD HEADER =================

if latest is not None:
    sec = max(1, 60 - latest["date"].second)
else:
    sec = 60

st.info(f"⏱ Next candle in **{sec}s**")

col1, col2 = st.columns(2)
col1.metric("NIFTY",  round(price, 2))
col2.metric("Option", round(option_price, 2))

# ================= SIGNAL COMPUTATION =================

if df is None or len(df) < 30:
    st.warning("Indicators warming up — need more candles...")
    st.stop()

prev  = df.iloc[-2]
entry = df.iloc[-1]

s = state.settings  # shorthand

conditions = [
    bool(entry["ma_chan_upward"]),
    bool(entry["ema_fast"] > entry["ema_slow"]),
    bool(entry["jma"] > prev["jma"]),
    bool(entry["rsi"] > s["rsi_threshold"]),
    bool(entry["di_plus"] > s["di_plus_threshold"] and entry["di_plus"] > entry["di_minus"]),
    bool(entry["adx"] > s["adx_threshold"] and entry["adx"] > prev["adx"]),
    bool(entry["sqz_momentum"] > 0),
    bool(entry["wt1"] > entry["wt2"]),
]

names = [
    "MA Channel Upward",
    f"EMA{s['ema_fast']} > EMA{s['ema_slow']}",
    "JMA Rising",
    "RSI Strength",
    "DMI Strength",
    "ADX Trend Strength",
    "Squeeze Momentum",
    "WaveTrend Upward",
]

rules = [
    "Both MA channel slopes are positive",
    f"EMA{s['ema_fast']} is above EMA{s['ema_slow']}",
    "JMA is rising (current > previous)",
    f"RSI > {s['rsi_threshold']}",
    f"DI+ > {s['di_plus_threshold']} and DI+ > DI−",
    f"ADX > {s['adx_threshold']} and rising",
    "Squeeze momentum value > 0",
    "WT1 > WT2",
]

met   = sum(conditions)
total = len(conditions)

# ================= CRITERIA SUMMARY =================

summary_color = "green" if met == total else ("orange" if met >= total // 2 else "red")
st.markdown(
    f"<h3 style='color:{summary_color}'>{'✅' if met == total else '⚠️'} {met} / {total} criteria met</h3>",
    unsafe_allow_html=True,
)
st.progress(met / total)

if all(conditions):
    st.success("🔥 TRADE SIGNAL — all criteria met!")
else:
    st.warning("⏳ WAIT — not all criteria met yet.")

# ================= CHECKLIST =================

st.subheader("Checklist")

_sqz_color_map = {"lime": "🟩", "green": "🟢", "red": "🟥", "maroon": "🔴"}
_sqz_label     = _sqz_color_map.get(entry["sqz_color"], "⬛")

for i, (met_flag, name, rule) in enumerate(zip(conditions, names, rules)):
    col_a, col_b, col_c = st.columns([2, 5, 3])
    col_a.write(f"{'🟢' if met_flag else '🔴'} **{name}**")
    col_b.write(rule)

    if i == 0:   # MA Channel
        upward = "✅ Yes" if entry["ma_chan_upward"] else "❌ No"
        col_c.write(f"Channel upward: {upward}")

    elif i == 1:   # EMA
        spread = entry["ema_fast"] - entry["ema_slow"]
        col_c.write(f"EMA{s['ema_fast']}: {entry['ema_fast']:.0f} / EMA{s['ema_slow']}: {entry['ema_slow']:.0f}")
        col_c.write(f"Spread: {spread:+.1f}")

    elif i == 2:   # JMA
        delta = entry["jma"] - prev["jma"]
        col_c.write(f"JMA: {entry['jma']:.1f} ({delta:+.1f})")

    elif i == 3:   # RSI
        col_c.write(f"RSI: {entry['rsi']:.1f}")
        col_c.progress(min(max(entry["rsi"] / 100, 0.0), 1.0))

    elif i == 4:   # DMI
        col_c.write(f"DI+: {entry['di_plus']:.1f}  DI−: {entry['di_minus']:.1f}")
        col_c.progress(min(max(entry["di_plus"] / 100, 0.0), 1.0))

    elif i == 5:   # ADX
        direction = "↑" if entry["adx"] > prev["adx"] else "↓"
        col_c.write(f"ADX: {entry['adx']:.1f} {direction}")
        col_c.progress(min(max(entry["adx"] / 100, 0.0), 1.0))

    elif i == 6:   # Squeeze
        col_c.write(f"Momentum: {entry['sqz_momentum']:.2f}  Color: {_sqz_label}")
        col_c.write("Squeezed 🔒" if entry["sqz_on"] else "Released 🔓")

    elif i == 7:   # WaveTrend
        col_c.write(f"WT1: {entry['wt1']:.1f}  WT2: {entry['wt2']:.1f}")

# ================= INDICATOR VALUES =================

st.subheader("Indicator Values")

indicator_metrics = [
    (f"EMA{s['ema_fast']}",    entry["ema_fast"],    entry["ema_fast"]    - prev["ema_fast"]),
    (f"EMA{s['ema_slow']}",    entry["ema_slow"],    entry["ema_slow"]    - prev["ema_slow"]),
    ("JMA",                     entry["jma"],          entry["jma"]          - prev["jma"]),
    ("RSI",                     entry["rsi"],          entry["rsi"]          - prev["rsi"]),
    ("ADX",                     entry["adx"],          entry["adx"]          - prev["adx"]),
    ("DI+",                     entry["di_plus"],      entry["di_plus"]      - prev["di_plus"]),
    ("DI−",                     entry["di_minus"],     entry["di_minus"]     - prev["di_minus"]),
    ("WT1",                     entry["wt1"],          entry["wt1"]          - prev["wt1"]),
    ("WT2",                     entry["wt2"],          entry["wt2"]          - prev["wt2"]),
    ("Sqz Momentum",            entry["sqz_momentum"], entry["sqz_momentum"] - prev["sqz_momentum"]),
]

metric_cols = st.columns(4)
for idx, (label, value, delta) in enumerate(indicator_metrics):
    metric_cols[idx % 4].metric(label, round(float(value), 2), round(float(delta), 2))

# ================= TRADE LOG =================

if state.trades:
    st.subheader("Trade Log")
    tdf = pd.DataFrame(state.trades).fillna("")
    st.dataframe(tdf, use_container_width=True)
    st.download_button("⬇️ Download trades.csv", tdf.to_csv(index=False), "trades.csv")
