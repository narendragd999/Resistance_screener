import streamlit as st
import pandas as pd
import yfinance as yf
import time
import json
import os
import aiohttp
import asyncio
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date, timedelta
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import logging

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
TICKERS_PATH      = "tickers.csv"
CONFIG_FILE       = "config.json"
SCREENING_FILE    = "screening_data.json"
BACKTEST_FILE     = "backtest_data.json"

NSE_HOMEPAGE      = "https://www.nseindia.com/"
NSE_DERIVATIVES   = "https://www.nseindia.com/market-data/equity-derivatives-watch"
NSE_OPTION_CHAIN  = "https://www.nseindia.com/api/option-chain-equities?symbol={}"
NSE_QUOTE         = "https://www.nseindia.com/api/quote-equity?symbol={}"

NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": NSE_DERIVATIVES,
}

CACHE_TTL_PRICE   = 60       # seconds — option chain / price
CACHE_TTL_HISTORY = 3600     # seconds — OHLCV history

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────
DEFAULTS = {
    "nse_session": None,
    "cache": {},
    "api_call_counter": 0,
    "screening_data": [],
    "backtest_data": [],
    "scan_in_progress": False,
    "last_scan_time": 0.0,
    "refresh_key": time.time(),
    "logs": [],
}

def init_state():
    for k, v in DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "min_gain_percent": 20.0,
    "min_green_candles": 3,
    "lookback_days": 30,
    "telegram_bot_token": "",
    "telegram_chat_id": "",
    "price_proximity_percent": 1.0,
    "max_oi_strikes": 5,
    "session_retry_count": 2,
}

def load_config() -> Dict:
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            cfg = json.load(f)
        for k, v in DEFAULT_CONFIG.items():
            cfg.setdefault(k, v)
        return cfg
    return DEFAULT_CONFIG.copy()

def save_config(cfg: Dict):
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=4)

# ─────────────────────────────────────────────────────────────────────────────
# PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────
def load_json(path: str) -> list:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []

def save_json(path: str, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4, default=str)

# ─────────────────────────────────────────────────────────────────────────────
# TICKER LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_tickers() -> List[str]:
    try:
        if os.path.exists(TICKERS_PATH):
            df = pd.read_csv(TICKERS_PATH)
            if "SYMBOL" in df.columns:
                return df["SYMBOL"].dropna().str.strip().tolist()
            st.error("CSV must have a 'SYMBOL' column.")
    except Exception as e:
        st.error(f"Error loading tickers: {e}")
    return ["HDFCBANK", "RELIANCE", "INFY", "TCS"]

# ─────────────────────────────────────────────────────────────────────────────
# NSE SESSION
# ─────────────────────────────────────────────────────────────────────────────
def _log(msg: str, level: str = "info"):
    """Append to in-app log and Python log."""
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state["logs"].append({"ts": ts, "msg": msg, "level": level})
    getattr(log, level, log.info)(msg)

def initialize_nse_session(retries: int = 2) -> bool:
    if st.session_state["nse_session"] is not None:
        return True
    for attempt in range(1, retries + 2):
        try:
            s = requests.Session()
            r = s.get(NSE_HOMEPAGE, headers=NSE_HEADERS, timeout=15)
            if r.status_code != 200:
                _log(f"NSE homepage failed ({r.status_code}), attempt {attempt}", "warning")
                time.sleep(3)
                continue
            r2 = s.get(NSE_DERIVATIVES, headers=NSE_HEADERS, timeout=15)
            time.sleep(3)
            if r2.status_code != 200:
                _log(f"NSE derivatives page failed ({r2.status_code}), attempt {attempt}", "warning")
                continue
            st.session_state["nse_session"] = s
            _log("NSE session initialized successfully.")
            return True
        except Exception as e:
            _log(f"Session init error: {e}", "error")
            time.sleep(3)
    return False

def reset_nse_session():
    st.session_state["nse_session"] = None

# ─────────────────────────────────────────────────────────────────────────────
# CACHE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _cache_get(key: str, ttl: int):
    cache = st.session_state["cache"]
    if key in cache:
        ts_key = f"{key}__ts"
        if time.time() - cache.get(ts_key, 0) < ttl:
            return cache[key]
    return None

def _cache_set(key: str, value):
    st.session_state["cache"][key] = value
    st.session_state["cache"][f"{key}__ts"] = time.time()

# ─────────────────────────────────────────────────────────────────────────────
# NSE DATA FETCHING
# ─────────────────────────────────────────────────────────────────────────────
def fetch_nse_data(ticker: str) -> Optional[Dict]:
    """Fetch option-chain + quote for a ticker from NSE."""
    cache_key = f"nse_{ticker}"
    cached = _cache_get(cache_key, CACHE_TTL_PRICE)
    if cached:
        _log(f"[{ticker}] Cache hit (NSE data)")
        return cached

    session = st.session_state.get("nse_session")
    if session is None:
        _log(f"[{ticker}] No NSE session available", "warning")
        return None

    st.session_state["api_call_counter"] += 1
    try:
        oc_resp = session.get(NSE_OPTION_CHAIN.format(ticker), headers=NSE_HEADERS, timeout=15)
        if oc_resp.status_code != 200:
            _log(f"[{ticker}] Option chain HTTP {oc_resp.status_code}", "warning")
            return None
        data = oc_resp.json()

        q_resp = session.get(NSE_QUOTE.format(ticker), headers=NSE_HEADERS, timeout=15)
        if q_resp.status_code == 200:
            last_price = q_resp.json().get("priceInfo", {}).get("lastPrice", 0)
            if last_price > 0 and "records" in data:
                data["records"]["underlyingValue"] = last_price

        _cache_set(cache_key, data)
        _log(f"[{ticker}] NSE data fetched (price: {data.get('records',{}).get('underlyingValue','?')})")
        return data
    except Exception as e:
        _log(f"[{ticker}] NSE fetch error: {e}", "error")
        return None

# ─────────────────────────────────────────────────────────────────────────────
# HISTORICAL DATA
# ─────────────────────────────────────────────────────────────────────────────
def fetch_historical(ticker: str, start: date, end: date) -> Optional[pd.DataFrame]:
    cache_key = f"hist_{ticker}_{start}_{end}"
    cached = _cache_get(cache_key, CACHE_TTL_HISTORY)
    if cached is not None:
        _log(f"[{ticker}] Cache hit (historical data)")
        return cached

    st.session_state["api_call_counter"] += 1
    try:
        stock = yf.Ticker(f"{ticker}.NS")
        hist = stock.history(start=start, end=end + timedelta(days=1))
        if hist.empty:
            _log(f"[{ticker}] No historical data from yfinance", "warning")
            return None
        _cache_set(cache_key, hist)
        _log(f"[{ticker}] History fetched: {len(hist)} candles ({hist.index[0].date()} – {hist.index[-1].date()})")
        return hist
    except Exception as e:
        _log(f"[{ticker}] yfinance error: {e}", "error")
        return None

# ─────────────────────────────────────────────────────────────────────────────
# OPTION CHAIN PROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def parse_option_chain(data: Dict, expiry: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    records = data.get("filtered", {}).get("data", [])
    if not records:
        return pd.DataFrame(), pd.DataFrame()

    rows = []
    for r in records:
        if expiry and r.get("expiryDate") != expiry:
            continue
        rows.append({
            "strike":    r.get("strikePrice", 0),
            "call_oi":   r.get("CE", {}).get("openInterest", 0),
            "call_chg":  r.get("CE", {}).get("changeinOpenInterest", 0),
            "call_ltp":  r.get("CE", {}).get("lastPrice", 0),
            "call_iv":   r.get("CE", {}).get("impliedVolatility", 0),
            "put_oi":    r.get("PE", {}).get("openInterest", 0),
            "put_chg":   r.get("PE", {}).get("changeinOpenInterest", 0),
            "put_ltp":   r.get("PE", {}).get("lastPrice", 0),
            "put_iv":    r.get("PE", {}).get("impliedVolatility", 0),
            "total_oi":  r.get("CE", {}).get("openInterest", 0) + r.get("PE", {}).get("openInterest", 0),
            "pcr":       (r.get("PE", {}).get("openInterest", 0) / r.get("CE", {}).get("openInterest", 1))
                         if r.get("CE", {}).get("openInterest", 0) > 0 else 0,
        })
    df = pd.DataFrame(rows)
    return df

def get_resistance_strikes(data: Dict, spot: float, top_n: int = 5) -> Optional[pd.DataFrame]:
    df = parse_option_chain(data)
    if df.empty:
        return None
    res = df[df["strike"] > spot].copy()
    if res.empty:
        return None
    res = res.sort_values("total_oi", ascending=False).head(top_n).reset_index(drop=True)
    res["dist_pct"] = ((res["strike"] - spot) / spot * 100).round(2)
    return res

def get_support_strikes(data: Dict, spot: float, top_n: int = 5) -> Optional[pd.DataFrame]:
    df = parse_option_chain(data)
    if df.empty:
        return None
    sup = df[df["strike"] < spot].copy()
    if sup.empty:
        return None
    sup = sup.sort_values("total_oi", ascending=False).head(top_n).reset_index(drop=True)
    sup["dist_pct"] = ((spot - sup["strike"]) / spot * 100).round(2)
    return sup

def get_expiries(data: Dict) -> List[str]:
    return data.get("records", {}).get("expiryDates", [])

# ─────────────────────────────────────────────────────────────────────────────
# MOMENTUM DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def find_best_momentum_run(closes: np.ndarray, min_green: int, min_gain: float):
    """
    Returns (max_green, max_gain, best_start_idx, best_end_idx)
    for the longest green-candle run meeting criteria.
    """
    best = {"green": 0, "gain": 0.0, "start": None, "end": None}
    run_len = 0
    run_start = None

    for i in range(1, len(closes)):
        if closes[i] > closes[i - 1]:
            run_len += 1
            if run_start is None:
                run_start = i - 1
            run_end = i
            if run_len >= min_green:
                gain = (closes[run_end] - closes[run_start]) / closes[run_start] * 100
                if gain >= min_gain and run_len >= best["green"]:
                    best.update(green=run_len, gain=gain, start=run_start, end=run_end)
        else:
            run_len = 0
            run_start = None

    return best["green"], best["gain"], best["start"], best["end"]

def check_momentum(
    ticker: str,
    hist: pd.DataFrame,
    spot: float,
    cfg: Dict,
) -> Optional[Dict]:
    min_gain      = cfg["min_gain_percent"]
    min_green     = cfg["min_green_candles"]
    bot_token     = cfg["telegram_bot_token"]
    chat_id       = cfg["telegram_chat_id"]
    top_n         = cfg.get("max_oi_strikes", 5)

    if hist.empty or len(hist) < min_green + 2:
        _log(f"[{ticker}] Insufficient candles ({len(hist)})", "warning")
        return None

    hist = hist.sort_index()
    closes = hist["Close"].values
    dates  = hist.index

    last_close = closes[-1]
    prev_close = closes[-2]

    # Must be below previous close (momentum loss / red day)
    if spot >= prev_close:
        _log(f"[{ticker}] No red day (spot={spot:.2f} >= prev_close={prev_close:.2f})")
        return None

    n_green, gain, si, ei = find_best_momentum_run(closes, min_green, min_gain)

    if n_green < min_green or gain < min_gain:
        _log(f"[{ticker}] No qualifying run (green={n_green}, gain={gain:.1f}%)")
        return None

    _log(f"[{ticker}] Momentum run: {n_green} green candles, +{gain:.1f}%")

    # Fetch NSE option chain
    nse = fetch_nse_data(ticker)
    if nse is None:
        _log(f"[{ticker}] NSE data unavailable for strike selection", "warning")
        return None

    resistance_df = get_resistance_strikes(nse, spot, top_n)
    support_df    = get_support_strikes(nse, spot, top_n)
    expiries      = get_expiries(nse)

    if resistance_df is None or resistance_df.empty:
        _log(f"[{ticker}] No resistance strikes found above {spot:.2f}", "warning")
        return None

    # Best call-selling strike = highest call premium above spot
    best_row   = resistance_df.loc[resistance_df["call_ltp"].idxmax()]
    best_strike = float(best_row["strike"])
    call_premium = float(best_row["call_ltp"])

    mom_start = dates[si].strftime("%Y-%m-%d")
    mom_end   = dates[ei].strftime("%Y-%m-%d")

    # Day-after momentum-end high (trigger for recovery alert)
    mom_end_ts = pd.to_datetime(mom_end)
    if hist.index.tz is not None:
        mom_end_ts = mom_end_ts.tz_localize(hist.index.tz)
    post_days = hist[hist.index > mom_end_ts]
    if not post_days.empty:
        next_candle = post_days.iloc[0]
        trigger_high = max(next_candle["Open"], next_candle["High"])
    else:
        trigger_high = max(hist.iloc[-1]["Open"], hist.iloc[-1]["High"])

    price_drop_pct = (prev_close - spot) / prev_close * 100
    recovery       = spot >= trigger_high
    status         = "Recovery" if recovery else "Momentum Loss"

    result = {
        "Ticker":               ticker,
        "Spot_Price":           round(spot, 2),
        "Prev_Close":           round(prev_close, 2),
        "Price_Drop_Pct":       round(price_drop_pct, 2),
        "Momentum_Gain_Pct":    round(gain, 2),
        "Green_Candles":        n_green,
        "Momentum_Start":       mom_start,
        "Momentum_End":         mom_end,
        "Trigger_High":         round(trigger_high, 2),
        "Best_Strike":          round(best_strike, 2),
        "Call_Premium":         round(call_premium, 2),
        "Max_Profit_Per_Lot":   round(call_premium * 50, 2),   # placeholder lot-size 50
        "Nearest_Expiry":       expiries[0] if expiries else "N/A",
        "Status":               status,
        "Resistance_Strikes":   resistance_df.to_dict("records"),
        "Support_Strikes":      support_df.to_dict("records") if support_df is not None else [],
        "Scanned_At":           datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Telegram
    _send_telegram_alert(result, bot_token, chat_id)
    return result

# ─────────────────────────────────────────────────────────────────────────────
# TELEGRAM
# ─────────────────────────────────────────────────────────────────────────────
async def _async_send(token: str, chat_id: str, text: str):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    async with aiohttp.ClientSession() as sess:
        await sess.post(url, json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"})

def _send_telegram_alert(result: Dict, token: str, chat_id: str):
    if not token or not chat_id:
        return
    emoji = "🔄" if result["Status"] == "Recovery" else "⚠️"
    msg = (
        f"{emoji} *{result['Status']} — {result['Ticker']}*\n"
        f"Spot: ₹{result['Spot_Price']} | Drop: {result['Price_Drop_Pct']:.1f}%\n"
        f"Prior run: +{result['Momentum_Gain_Pct']:.1f}% over {result['Green_Candles']} candles\n"
        f"Period: {result['Momentum_Start']} → {result['Momentum_End']}\n"
        f"Suggested strike: *₹{result['Best_Strike']}*  (premium ₹{result['Call_Premium']})\n"
        f"Nearest expiry: {result['Nearest_Expiry']}\n"
        f"Scanned: {result['Scanned_At']}"
    )
    MAX = 4096
    chunks = [msg[i:i+MAX] for i in range(0, len(msg), MAX)]
    for chunk in chunks:
        asyncio.run(_async_send(token, chat_id, chunk))
        time.sleep(0.4)

# ─────────────────────────────────────────────────────────────────────────────
# SCREENING ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────
def screen_tickers(tickers: List[str], cfg: Dict) -> List[Dict]:
    end   = date.today()
    start = end - timedelta(days=cfg["lookback_days"])
    results = []

    if not initialize_nse_session(retries=cfg.get("session_retry_count", 2)):
        st.error("⚠️ Could not initialise NSE session. Please retry.")
        return results

    progress = st.progress(0, text="Starting scan…")
    for idx, ticker in enumerate(tickers):
        progress.progress((idx + 1) / len(tickers), text=f"Scanning {ticker}…")

        hist = fetch_historical(ticker, start, end)
        if hist is None:
            continue

        nse = fetch_nse_data(ticker)
        if nse is None or "records" not in nse:
            continue

        spot = nse["records"].get("underlyingValue")
        if not spot:
            continue

        res = check_momentum(ticker, hist, spot, cfg)
        if res:
            results.append(res)

    progress.empty()
    reset_nse_session()
    return results

# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST
# ─────────────────────────────────────────────────────────────────────────────
def backtest_momentum(ticker: str, start: date, end: date, cfg: Dict) -> List[Dict]:
    hist = fetch_historical(ticker, start, end)
    if hist is None:
        return []

    min_gain  = cfg["min_gain_percent"]
    min_green = cfg["min_green_candles"]
    results   = []

    hist = hist.sort_index()
    closes = hist["Close"].values
    dates  = hist.index

    for i in range(min_green + 1, len(closes)):
        window_closes = closes[:i + 1]
        spot          = closes[i]
        prev          = closes[i - 1]
        if spot >= prev:
            continue

        n, gain, si, ei = find_best_momentum_run(window_closes, min_green, min_gain)
        if n < min_green or gain < min_gain:
            continue

        # Proxy strike from historical high
        local_high = hist["High"].values[:ei + 1].max()
        proxy_strike = round(local_high / 50) * 50

        results.append({
            "Date":           dates[i].strftime("%Y-%m-%d"),
            "Spot":           round(float(spot), 2),
            "Prev_Close":     round(float(prev), 2),
            "Drop_Pct":       round((prev - spot) / prev * 100, 2),
            "Momentum_Gain":  round(float(gain), 2),
            "Green_Candles":  int(n),
            "Momentum_Start": dates[si].strftime("%Y-%m-%d"),
            "Momentum_End":   dates[ei].strftime("%Y-%m-%d"),
            "Proxy_Strike":   float(proxy_strike),
        })

    return results

# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────
CHART_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#0d1117",
    font=dict(color="#c9d1d9", family="monospace"),
    xaxis=dict(gridcolor="#21262d", linecolor="#30363d", showgrid=True),
    yaxis=dict(gridcolor="#21262d", linecolor="#30363d", showgrid=True),
    margin=dict(l=40, r=20, t=40, b=40),
)

def candlestick_chart(ticker: str, hist: pd.DataFrame, result: Optional[Dict] = None) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.04,
    )

    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=hist.index, open=hist["Open"], high=hist["High"],
        low=hist["Low"], close=hist["Close"],
        increasing_line_color="#3fb950", decreasing_line_color="#f85149",
        increasing_fillcolor="#3fb950", decreasing_fillcolor="#f85149",
        name="Price",
    ), row=1, col=1)

    # Volume bars
    colors = ["#3fb950" if c >= o else "#f85149"
              for c, o in zip(hist["Close"], hist["Open"])]
    fig.add_trace(go.Bar(
        x=hist.index, y=hist["Volume"],
        marker_color=colors, name="Volume", opacity=0.6,
    ), row=2, col=1)

    # Annotations from result
    if result:
        spot   = result["Spot_Price"]
        strike = result["Best_Strike"]
        trig   = result["Trigger_High"]

        fig.add_hline(y=spot,   line=dict(color="#79c0ff", dash="dot", width=1.5),
                      annotation_text=f"Spot ₹{spot}", annotation_position="left",
                      annotation_font_color="#79c0ff", row=1, col=1)
        fig.add_hline(y=strike, line=dict(color="#f78166", dash="dash", width=1.5),
                      annotation_text=f"Strike ₹{strike}", annotation_position="right",
                      annotation_font_color="#f78166", row=1, col=1)
        fig.add_hline(y=trig,   line=dict(color="#e3b341", dash="longdash", width=1),
                      annotation_text=f"Trigger ₹{trig}", annotation_position="right",
                      annotation_font_color="#e3b341", row=1, col=1)

        # Shade momentum period
        try:
            m_s = pd.to_datetime(result["Momentum_Start"])
            m_e = pd.to_datetime(result["Momentum_End"])
            if hist.index.tz:
                m_s = m_s.tz_localize(hist.index.tz)
                m_e = m_e.tz_localize(hist.index.tz)
            fig.add_vrect(x0=m_s, x1=m_e, fillcolor="rgba(63,185,80,0.08)",
                          line_width=0, row=1, col=1)
        except Exception:
            pass

    fig.update_layout(
        title=f"{ticker} — Momentum Analysis",
        showlegend=False,
        xaxis_rangeslider_visible=False,
        height=520,
        **CHART_THEME,
    )
    return fig

def oi_heatmap(resistance_df: pd.DataFrame, support_df: pd.DataFrame, spot: float) -> go.Figure:
    frames = []
    if resistance_df is not None and not resistance_df.empty:
        r = resistance_df.copy()
        r["side"] = "Resistance (CE)"
        frames.append(r)
    if support_df is not None and not support_df.empty:
        s = support_df.copy()
        s["side"] = "Support (PE)"
        frames.append(s)

    if not frames:
        return go.Figure()

    combined = pd.concat(frames, ignore_index=True)

    fig = go.Figure()

    for side, color_oi, color_bar in [
        ("Resistance (CE)", "#f85149", "rgba(248,81,73,0.7)"),
        ("Support (PE)", "#3fb950", "rgba(63,185,80,0.7)"),
    ]:
        sub = combined[combined["side"] == side]
        if sub.empty:
            continue
        oi_col  = "call_oi" if "CE" in side else "put_oi"
        ltp_col = "call_ltp" if "CE" in side else "put_ltp"
        fig.add_trace(go.Bar(
            x=sub["strike"].astype(str),
            y=sub[oi_col] / 1000,
            name=side,
            marker_color=color_bar,
            text=[f"₹{p:.1f}" for p in sub[ltp_col]],
            textposition="outside",
            textfont=dict(size=10),
        ))

    fig.add_vline(x=str(combined["strike"].iloc[len(combined) // 2]),
                  line=dict(color="#79c0ff", dash="dot"), annotation_text=f"Spot ₹{spot:.0f}")

    fig.update_layout(
        title="Open Interest — Resistance & Support Strikes",
        barmode="group",
        xaxis_title="Strike Price",
        yaxis_title="OI (thousands)",
        height=350,
        **CHART_THEME,
    )
    return fig

def backtest_equity_chart(bt_results: List[Dict]) -> go.Figure:
    if not bt_results:
        return go.Figure()
    df = pd.DataFrame(bt_results)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["Momentum_Gain"].cumsum(),
        mode="lines+markers", line=dict(color="#79c0ff", width=2),
        fill="tozeroy", fillcolor="rgba(121,192,255,0.1)",
        name="Cumulative Momentum Gain %",
    ))
    fig.update_layout(
        title="Backtest — Cumulative Momentum Gains",
        xaxis_title="Date", yaxis_title="Cumulative Gain %",
        height=340, **CHART_THEME,
    )
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────
STATUS_COLORS = {
    "Recovery":      "#3fb950",
    "Momentum Loss": "#f85149",
}

def status_badge(status: str) -> str:
    color = STATUS_COLORS.get(status, "#8b949e")
    return f'<span style="background:{color}22;color:{color};padding:2px 10px;border-radius:4px;font-size:12px;font-weight:600;border:1px solid {color}44;">{status}</span>'

def metric_pill(label: str, value: str, delta_color: str = "#8b949e") -> str:
    return (
        f'<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px 18px;text-align:center;">'
        f'<div style="font-size:11px;color:#8b949e;letter-spacing:0.08em;text-transform:uppercase;">{label}</div>'
        f'<div style="font-size:22px;font-weight:700;color:{delta_color};margin-top:4px;">{value}</div>'
        f'</div>'
    )

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: #0d1117;
}
.block-container {
    max-width: 1400px;
    padding-top: 1.5rem;
}
h1, h2, h3 {
    font-family: 'JetBrains Mono', monospace !important;
    letter-spacing: -0.03em;
}
.ticker-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 12px;
    transition: border-color 0.2s;
}
.ticker-card:hover {
    border-color: #58a6ff;
}
.ticker-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
}
.ticker-symbol {
    font-family: 'JetBrains Mono', monospace;
    font-size: 20px;
    font-weight: 700;
    color: #e6edf3;
}
.metrics-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: 10px;
    margin-top: 14px;
}
.metric-box {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 10px 14px;
}
.metric-label {
    font-size: 10px;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 16px;
    font-weight: 600;
    color: #e6edf3;
    margin-top: 2px;
}
.section-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    color: #8b949e;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin: 0 0 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid #21262d;
}
.log-entry {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    padding: 3px 0;
    border-bottom: 1px solid #21262d11;
}
.log-info    { color: #8b949e; }
.log-warning { color: #e3b341; }
.log-error   { color: #f85149; }
div[data-testid="stSidebarContent"] {
    background: #010409;
    border-right: 1px solid #21262d;
}
.stButton button {
    background: #21262d !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 13px !important;
    border-radius: 8px !important;
    transition: all 0.15s !important;
}
.stButton button:hover {
    background: #30363d !important;
    border-color: #58a6ff !important;
}
.primary-btn button {
    background: #238636 !important;
    border-color: #2ea043 !important;
    color: #ffffff !important;
}
.primary-btn button:hover {
    background: #2ea043 !important;
}
</style>
"""

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar(cfg: Dict) -> Dict:
    """Render sidebar settings and return updated config."""
    with st.sidebar:
        st.markdown('<p class="section-title">⚙ Screening Params</p>', unsafe_allow_html=True)

        cfg["min_gain_percent"] = st.number_input(
            "Min Momentum Gain (%)", value=cfg["min_gain_percent"],
            min_value=5.0, step=1.0,
        )
        cfg["min_green_candles"] = st.number_input(
            "Min Green Candles", value=cfg["min_green_candles"],
            min_value=1, step=1,
        )
        cfg["lookback_days"] = st.number_input(
            "Lookback Period (days)", value=cfg["lookback_days"],
            min_value=4, step=1,
        )
        cfg["price_proximity_percent"] = st.number_input(
            "Recovery Proximity (%)", value=cfg["price_proximity_percent"],
            min_value=0.1, step=0.1,
        )
        cfg["max_oi_strikes"] = st.number_input(
            "Top OI Strikes to Show", value=cfg.get("max_oi_strikes", 5),
            min_value=1, max_value=15, step=1,
        )

        st.markdown('<p class="section-title">📡 Telegram Alerts</p>', unsafe_allow_html=True)
        cfg["telegram_bot_token"] = st.text_input(
            "Bot Token", value=cfg["telegram_bot_token"], type="password",
        )
        cfg["telegram_chat_id"] = st.text_input(
            "Chat ID", value=cfg["telegram_chat_id"],
        )
        if not cfg["telegram_bot_token"]:
            st.caption("⚠ No token — alerts disabled.")

        st.markdown('<p class="section-title">📂 Ticker List</p>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload CSV (SYMBOL column)", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
            if "SYMBOL" in df.columns:
                df.to_csv(TICKERS_PATH, index=False)
                st.success(f"✓ {len(df)} tickers saved.")
            else:
                st.error("CSV must have 'SYMBOL' column.")

        st.markdown('<p class="section-title">🔢 Session Stats</p>', unsafe_allow_html=True)
        st.caption(f"API calls this session: **{st.session_state['api_call_counter']}**")
        st.caption(f"Cache entries: **{len(st.session_state['cache']) // 2}**")
        if st.button("Clear Cache"):
            st.session_state["cache"] = {}
            st.success("Cache cleared.")

    save_config(cfg)
    return cfg

# ─────────────────────────────────────────────────────────────────────────────
# RESULT CARD
# ─────────────────────────────────────────────────────────────────────────────
def render_result_card(r: Dict, cfg: Dict):
    status_color = STATUS_COLORS.get(r["Status"], "#8b949e")
    with st.container():
        st.markdown(f"""
<div class="ticker-card" style="border-left: 3px solid {status_color};">
  <div class="ticker-header">
    <span class="ticker-symbol">{r['Ticker']}</span>
    {status_badge(r['Status'])}
  </div>
  <div class="metrics-row">
    <div class="metric-box">
      <div class="metric-label">Spot</div>
      <div class="metric-value">₹{r['Spot_Price']:,.2f}</div>
    </div>
    <div class="metric-box">
      <div class="metric-label">Drop</div>
      <div class="metric-value" style="color:#f85149;">-{r['Price_Drop_Pct']:.2f}%</div>
    </div>
    <div class="metric-box">
      <div class="metric-label">Prior Run</div>
      <div class="metric-value" style="color:#3fb950;">+{r['Momentum_Gain_Pct']:.2f}%</div>
    </div>
    <div class="metric-box">
      <div class="metric-label">Green Candles</div>
      <div class="metric-value">{r['Green_Candles']}</div>
    </div>
    <div class="metric-box">
      <div class="metric-label">Best Strike</div>
      <div class="metric-value" style="color:#f78166;">₹{r['Best_Strike']:,.0f}</div>
    </div>
    <div class="metric-box">
      <div class="metric-label">Call Premium</div>
      <div class="metric-value" style="color:#79c0ff;">₹{r['Call_Premium']:.2f}</div>
    </div>
    <div class="metric-box">
      <div class="metric-label">Max Profit/Lot</div>
      <div class="metric-value" style="color:#e3b341;">₹{r['Max_Profit_Per_Lot']:,.0f}</div>
    </div>
    <div class="metric-box">
      <div class="metric-label">Expiry</div>
      <div class="metric-value" style="font-size:13px;">{r['Nearest_Expiry']}</div>
    </div>
  </div>
  <div style="margin-top:10px;font-size:11px;color:#8b949e;">
    Momentum: {r['Momentum_Start']} → {r['Momentum_End']} &nbsp;|&nbsp; Trigger High: ₹{r['Trigger_High']}
    &nbsp;|&nbsp; Scanned: {r['Scanned_At']}
  </div>
</div>
""", unsafe_allow_html=True)

        with st.expander(f"📊 Charts & OI Analysis — {r['Ticker']}"):
            end   = date.today()
            start = end - timedelta(days=cfg["lookback_days"])
            hist  = fetch_historical(r["Ticker"], start, end)
            if hist is not None:
                st.plotly_chart(candlestick_chart(r["Ticker"], hist, r),
                                use_container_width=True)

            res_df = pd.DataFrame(r.get("Resistance_Strikes", []))
            sup_df = pd.DataFrame(r.get("Support_Strikes", []))
            if not res_df.empty:
                st.plotly_chart(oi_heatmap(res_df, sup_df, r["Spot_Price"]),
                                use_container_width=True)
                st.markdown("**Top Resistance Strikes (OI)**")
                st.dataframe(
                    res_df[["strike", "call_oi", "call_ltp", "call_iv", "dist_pct"]].rename(columns={
                        "strike": "Strike", "call_oi": "Call OI",
                        "call_ltp": "Premium ₹", "call_iv": "IV %", "dist_pct": "Dist %",
                    }),
                    use_container_width=True, hide_index=True,
                )

# ─────────────────────────────────────────────────────────────────────────────
# SCREENER TAB
# ─────────────────────────────────────────────────────────────────────────────
def tab_screener(cfg: Dict):
    st.markdown("## 🔍 Real-Time Screener")

    # Summary row
    data = st.session_state["screening_data"]
    n_recovery = sum(1 for r in data if r.get("Status") == "Recovery")
    n_loss     = sum(1 for r in data if r.get("Status") == "Momentum Loss")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Signals",   len(data))
    col2.metric("Momentum Loss",   n_loss,     delta=f"{n_loss}", delta_color="inverse")
    col3.metric("Recovery",        n_recovery, delta=f"{n_recovery}", delta_color="normal")
    col4.metric("Last Scan",
                datetime.fromtimestamp(st.session_state["last_scan_time"]).strftime("%H:%M:%S")
                if st.session_state["last_scan_time"] else "–")

    st.divider()

    # Controls
    c1, c2, c3 = st.columns([2, 2, 1])
    specific_input = c1.text_input("Specific tickers (comma-separated)", placeholder="RELIANCE, HDFC, INFY")
    with c2:
        st.write("")  # spacing

    def _run_screen(tickers: List[str]):
        if st.session_state["scan_in_progress"]:
            st.warning("Scan already in progress.")
            return
        st.session_state["scan_in_progress"] = True
        results = screen_tickers(tickers, cfg)
        st.session_state["screening_data"] = results
        save_json(SCREENING_FILE, results)
        st.session_state["last_scan_time"] = time.time()
        st.session_state["scan_in_progress"] = False
        if not results:
            st.info("No stocks met the momentum loss criteria.")
        st.rerun()

    col_a, col_b, col_c = st.columns([1, 1, 4])
    with col_a:
        st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
        if st.button("▶ Screen All", use_container_width=True):
            _run_screen(load_tickers())
        st.markdown("</div>", unsafe_allow_html=True)
    with col_b:
        if st.button("▶ Screen Selected", use_container_width=True):
            tickers = [t.strip() for t in specific_input.split(",") if t.strip()]
            if tickers:
                _run_screen(tickers)
    with col_c:
        if st.button("🗑 Clear Results"):
            st.session_state["screening_data"] = []
            save_json(SCREENING_FILE, [])
            st.rerun()

    st.divider()

    # Results
    if not data:
        st.info("Run a scan to see results.")
        return

    # Filter
    search = st.text_input("🔎 Filter by ticker", "")
    status_filter = st.radio("Status filter", ["All", "Momentum Loss", "Recovery"], horizontal=True)

    filtered = data
    if search:
        filtered = [r for r in filtered if search.upper() in r["Ticker"]]
    if status_filter != "All":
        filtered = [r for r in filtered if r["Status"] == status_filter]

    if not filtered:
        st.warning("No results match your filter.")
        return

    # Sort options
    sort_by = st.selectbox("Sort by", ["Momentum_Gain_Pct", "Price_Drop_Pct", "Call_Premium", "Green_Candles"],
                           format_func=lambda x: x.replace("_", " "))
    filtered.sort(key=lambda x: x.get(sort_by, 0), reverse=True)

    st.markdown(f"**{len(filtered)} signals**")
    for r in filtered:
        render_result_card(r, cfg)

# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST TAB
# ─────────────────────────────────────────────────────────────────────────────
def tab_backtest(cfg: Dict):
    st.markdown("## 📈 Backtest")

    c1, c2, c3 = st.columns(3)
    bt_ticker = c1.text_input("Ticker", "RELIANCE")
    bt_start  = c2.date_input("Start Date", date.today() - timedelta(days=365))
    bt_end    = c3.date_input("End Date", date.today())

    if st.button("▶ Run Backtest"):
        with st.spinner("Running backtest…"):
            results = backtest_momentum(bt_ticker, bt_start, bt_end, cfg)
            st.session_state["backtest_data"] = results
            save_json(BACKTEST_FILE, results)

    data = st.session_state["backtest_data"]
    if not data:
        st.info("Configure parameters and run a backtest.")
        return

    df = pd.DataFrame(data)
    st.plotly_chart(backtest_equity_chart(data), use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Signals Found",    len(df))
    col2.metric("Avg Momentum Gain", f"{df['Momentum_Gain'].mean():.1f}%")
    col3.metric("Avg Drop",         f"{df['Drop_Pct'].mean():.1f}%")

    st.dataframe(df[[
        "Date", "Spot", "Prev_Close", "Drop_Pct",
        "Momentum_Gain", "Green_Candles",
        "Momentum_Start", "Momentum_End", "Proxy_Strike",
    ]].rename(columns={
        "Drop_Pct": "Drop %", "Momentum_Gain": "Run Gain %",
        "Green_Candles": "Candles", "Proxy_Strike": "Strike",
        "Momentum_Start": "Run Start", "Momentum_End": "Run End",
    }), use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# OI EXPLORER TAB
# ─────────────────────────────────────────────────────────────────────────────
def tab_oi_explorer():
    st.markdown("## 🔬 Option Chain Explorer")

    c1, c2 = st.columns([2, 1])
    ticker = c1.text_input("Ticker Symbol", "NIFTY")
    if c2.button("Fetch Chain", use_container_width=True):
        if not initialize_nse_session():
            st.error("NSE session failed.")
            return
        with st.spinner("Fetching…"):
            nse = fetch_nse_data(ticker)
            reset_nse_session()
        if nse is None:
            st.error("No data returned.")
            return

        spot      = nse.get("records", {}).get("underlyingValue", 0)
        expiries  = get_expiries(nse)
        chain_df  = parse_option_chain(nse)

        st.markdown(f"**Spot: ₹{spot:,.2f}** &nbsp;|&nbsp; Expiries: {', '.join(expiries[:4])}")

        if not chain_df.empty:
            res = get_resistance_strikes(nse, spot, 8)
            sup = get_support_strikes(nse, spot, 8)
            st.plotly_chart(oi_heatmap(res, sup, spot), use_container_width=True)

            st.markdown("**Full Chain (nearest strikes)**")
            near = chain_df[
                (chain_df["strike"] >= spot * 0.95) &
                (chain_df["strike"] <= spot * 1.05)
            ].sort_values("strike")

            st.dataframe(near[[
                "strike", "call_oi", "call_ltp", "call_iv",
                "put_oi", "put_ltp", "put_iv", "pcr"
            ]].rename(columns={
                "strike": "Strike", "call_oi": "CE OI", "call_ltp": "CE LTP",
                "call_iv": "CE IV%", "put_oi": "PE OI", "put_ltp": "PE LTP",
                "put_iv": "PE IV%", "pcr": "PCR",
            }), use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOGS TAB
# ─────────────────────────────────────────────────────────────────────────────
def tab_logs():
    st.markdown("## 📋 Activity Logs")
    if st.button("Clear Logs"):
        st.session_state["logs"] = []
        st.rerun()

    logs = list(reversed(st.session_state["logs"]))
    if not logs:
        st.info("No log entries yet.")
        return

    log_html = "<div style='max-height:500px;overflow-y:auto;font-family:monospace;background:#010409;padding:16px;border-radius:8px;border:1px solid #21262d;'>"
    for entry in logs:
        cls = f"log-{entry['level']}"
        log_html += f'<div class="log-entry {cls}">[{entry["ts"]}] {entry["msg"]}</div>'
    log_html += "</div>"
    st.markdown(log_html, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="SNAPSCREENER — NSE Momentum",
        page_icon="📡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    init_state()

    if "config" not in st.session_state:
        st.session_state["config"] = load_config()
    if not st.session_state["screening_data"]:
        st.session_state["screening_data"] = load_json(SCREENING_FILE)
    if not st.session_state["backtest_data"]:
        st.session_state["backtest_data"] = load_json(BACKTEST_FILE)

    # Header
    st.markdown("""
<div style="display:flex;align-items:center;gap:16px;margin-bottom:4px;">
  <div style="font-family:'JetBrains Mono',monospace;font-size:28px;font-weight:700;color:#e6edf3;letter-spacing:-0.04em;">
    📡 SNAPSCREENER
  </div>
  <div style="font-size:13px;color:#8b949e;padding-top:6px;">
    NSE Momentum Loss &amp; Call-Selling Signal Engine
  </div>
</div>
""", unsafe_allow_html=True)

    cfg = render_sidebar(st.session_state["config"])
    st.session_state["config"] = cfg

    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Screener", "📈 Backtest", "🔬 OI Explorer", "📋 Logs"])

    with tab1:
        tab_screener(cfg)
    with tab2:
        tab_backtest(cfg)
    with tab3:
        tab_oi_explorer()
    with tab4:
        tab_logs()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.exception("Fatal error")
        st.error(f"Fatal error: {e}")
