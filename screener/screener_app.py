import streamlit as st
import pandas as pd
import yfinance as yf
import time
import json
import os
import random
import aiohttp
import asyncio
import threading
from typing import Dict, List, Optional, Tuple
from datetime import date, datetime, timedelta, timezone
import pytz
import plotly.graph_objects as go
import requests
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

# ====================== CONSTANTS ======================
STORED_TICKERS_PATH   = "tickers.csv"
CONFIG_FILE           = "config.json"
SCREENING_DATA_FILE   = "screening_data.json"
BREAKOUT_DATA_FILE    = "breakout_data.json"
BASELINE_FILE         = "price_baseline.json"   # ← NEW: stores initial prices
SCAN_LOG_FILE         = "scan_log.json"

IST = pytz.timezone("Asia/Kolkata")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/option-chain",
    "Sec-Fetch-Site": "same-origin",
    "Sec-Fetch-Mode": "cors",
    "Connection": "keep-alive",
}

_nse_session: Optional[requests.Session] = None
_session_lock = threading.Lock()

# ====================== CONFIG ======================
def load_config() -> Dict:
    defaults = {
        "min_gain_percent":         20.0,
        "min_green_candles":        3,
        "lookback_days":            30,
        "telegram_bot_token":       "",
        "telegram_chat_id":         "",
        "price_proximity_percent":  1.0,
        "surge_filter_percent":     3.0,   # ← NEW: only screen stocks up ≥ this %
        "auto_scan_interval_min":   30,    # ← NEW: auto-scan interval (minutes)
        "auto_scan_enabled":        True,  # ← NEW: toggle auto-scan
        "market_hours_only":        True,  # ← NEW: restrict auto-scan to IST hours
    }
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            saved = json.load(f)
            defaults.update(saved)   # saved values override defaults
    return defaults

def save_config(config: Dict):
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=4)

# ====================== PERSISTENCE ======================
def _load_json(path: str, default):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return default

def _save_json(path: str, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

def load_screening_data() -> List[Dict]:   return _load_json(SCREENING_DATA_FILE, [])
def save_screening_data(d):                _save_json(SCREENING_DATA_FILE, d)
def load_breakout_data() -> List[Dict]:    return _load_json(BREAKOUT_DATA_FILE, [])
def save_breakout_data(d):                 _save_json(BREAKOUT_DATA_FILE, d)
def load_baseline() -> Dict[str, float]:   return _load_json(BASELINE_FILE, {})
def save_baseline(d):                      _save_json(BASELINE_FILE, d)
def load_scan_log() -> List[Dict]:         return _load_json(SCAN_LOG_FILE, [])
def save_scan_log(d):                      _save_json(SCAN_LOG_FILE, d)

# ====================== TELEGRAM ======================
async def _send_tg(bot_token: str, chat_id: str, message: str):
    if not bot_token or not chat_id:
        return
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
    try:
        async with aiohttp.ClientSession() as session:
            await session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10))
    except Exception as e:
        print(f"Telegram error: {e}")

def send_telegram(bot_token: str, chat_id: str, message: str):
    MAX = 4096
    for i in range(0, len(message), MAX):
        chunk = message[i:i + MAX].strip()
        asyncio.run(_send_tg(bot_token, chat_id, chunk))
        time.sleep(0.4)

# ====================== TICKERS ======================
def load_tickers() -> List[str]:
    try:
        if os.path.exists(STORED_TICKERS_PATH):
            df = pd.read_csv(STORED_TICKERS_PATH)
            if "SYMBOL" in df.columns:
                return [str(s).strip().upper() for s in df["SYMBOL"].dropna()]
    except Exception as e:
        print(f"Ticker load error: {e}")
    return ["HDFCBANK", "RELIANCE", "TCS"]

# ====================== NSE SESSION ======================
def get_nse_session() -> Optional[requests.Session]:
    global _nse_session
    with _session_lock:
        if _nse_session is not None:
            return _nse_session
        s = requests.Session()
        s.headers.update(HEADERS)
        try:
            s.get("https://www.nseindia.com/", timeout=12)
            time.sleep(random.uniform(1.5, 2.5))
            s.get("https://www.nseindia.com/option-chain", timeout=12)
            time.sleep(random.uniform(1.8, 3.0))
            _nse_session = s
            print("NSE session ready.")
            return _nse_session
        except Exception as e:
            print(f"NSE session init failed: {e}")
            return None

def reset_nse_session():
    global _nse_session
    with _session_lock:
        _nse_session = None

# ====================== NSE DATA ======================
def get_expiry_list(ticker: str, session: requests.Session) -> List[str]:
    try:
        url = f"https://www.nseindia.com/api/option-chain-contract-info?symbol={ticker}"
        r = session.get(url, timeout=12)
        if r.status_code == 200 and r.text.strip().startswith("{"):
            d = r.json()
            return d.get("expiryDates", []) or d.get("records", {}).get("expiryDates", [])
    except Exception as e:
        print(f"Expiry error {ticker}: {e}")
    return []

def fetch_nse_data(ticker: str) -> Optional[Dict]:
    session = get_nse_session()
    if session is None:
        return None

    expiries = get_expiry_list(ticker, session)
    if not expiries:
        return None
    expiry = expiries[0]

    try:
        url = (f"https://www.nseindia.com/api/option-chain-v3"
               f"?type=Equity&symbol={ticker}&expiry={expiry}")
        r = session.get(url, timeout=15)
        if r.status_code != 200 or not r.text.strip().startswith("{"):
            if r.status_code in (429, 403):
                reset_nse_session()
            return None

        data = r.json()
        # Enrich with live price
        try:
            qr = session.get(
                f"https://www.nseindia.com/api/quote-equity?symbol={ticker}",
                timeout=10)
            if qr.status_code == 200:
                lp = qr.json().get("priceInfo", {}).get("lastPrice")
                if lp:
                    data.setdefault("records", {})["underlyingValue"] = lp
        except:
            pass
        return data
    except Exception as e:
        print(f"Option chain error {ticker}: {e}")
        return None

# ====================== OPTION PROCESSING ======================
def process_option_data(data: Dict):
    records = data.get("filtered", {}).get("data", [])
    calls, puts = [], []
    for r in records:
        s = r.get("strikePrice", 0)
        ce = r.get("CE", {})
        pe = r.get("PE", {})
        calls.append({"strikePrice": s, "callOI": ce.get("openInterest", 0),
                      "callPrice": ce.get("lastPrice", 0)})
        puts.append({"strikePrice": s, "putOI": pe.get("openInterest", 0)})
    return pd.DataFrame(calls), pd.DataFrame(puts)

def identify_resistance(data: Dict, price: float) -> Optional[pd.DataFrame]:
    call_df, put_df = process_option_data(data)
    if call_df.empty or put_df.empty:
        return None
    combined = pd.DataFrame({
        "strikePrice": call_df["strikePrice"],
        "totalOI": call_df["callOI"] + put_df["putOI"],
        "callPrice": call_df["callPrice"],
    })
    above = combined[combined["strikePrice"] > price]
    return above.sort_values("totalOI", ascending=False) if not above.empty else None

# ====================== HISTORICAL DATA ======================
def fetch_historical(ticker: str, start: date, end: date) -> Optional[pd.DataFrame]:
    try:
        hist = yf.Ticker(f"{ticker}.NS").history(
            start=start, end=end + timedelta(days=1))
        return hist if not hist.empty else None
    except Exception as e:
        print(f"yfinance error {ticker}: {e}")
        return None

# ====================== BASELINE PRICE CAPTURE ======================
def capture_baseline(tickers: List[str]) -> Dict[str, float]:
    """
    Fetch current price for every ticker and store as baseline.
    Uses yfinance fast_info to avoid NSE rate issues.
    """
    baseline = {}
    for tk in tickers:
        try:
            info = yf.Ticker(f"{tk}.NS").fast_info
            price = getattr(info, "last_price", None)
            if price:
                baseline[tk] = round(float(price), 2)
            time.sleep(random.uniform(0.3, 0.7))
        except Exception as e:
            print(f"Baseline error {tk}: {e}")
    save_baseline(baseline)
    print(f"Baseline captured for {len(baseline)} tickers.")
    return baseline

def get_surge_candidates(tickers: List[str], baseline: Dict[str, float],
                          surge_pct: float) -> List[Tuple[str, float, float]]:
    """
    Return tickers whose price has moved ≥ surge_pct% from baseline.
    Returns list of (ticker, current_price, change_pct).
    """
    candidates = []
    for tk in tickers:
        if tk not in baseline:
            continue
        try:
            info = yf.Ticker(f"{tk}.NS").fast_info
            cur = getattr(info, "last_price", None)
            if cur is None:
                continue
            cur = float(cur)
            base = baseline[tk]
            chg = ((cur - base) / base) * 100
            if abs(chg) >= surge_pct:
                candidates.append((tk, round(cur, 2), round(chg, 2)))
            time.sleep(random.uniform(0.2, 0.5))
        except:
            pass
    return candidates

# ====================== MOMENTUM LOSS CHECK ======================
def check_momentum(ticker: str, hist: pd.DataFrame, current_price: float,
                   config: Dict) -> Optional[Dict]:
    min_gain = config["min_gain_percent"]
    min_green = config["min_green_candles"]
    bot_token = config["telegram_bot_token"]
    chat_id = config["telegram_chat_id"]

    if len(hist) < min_green + 5:
        return None

    hist = hist.sort_index()
    closes = hist["Close"].values
    dates = hist.index

    max_green = max_gain = 0
    best_start = best_end = None
    curr_green = curr_start = 0

    for i in range(1, len(closes)):
        if closes[i] > closes[i - 1]:
            curr_green += 1
            if curr_green == 1:
                curr_start = i - 1
            if curr_green >= min_green:
                gain = ((closes[i] - closes[curr_start]) / closes[curr_start]) * 100
                if gain >= min_gain and curr_green > max_green:
                    max_green, max_gain = curr_green, gain
                    best_start, best_end = curr_start, i
        else:
            curr_green = curr_start = 0

    if max_gain < min_gain or max_green < min_green or current_price >= closes[-1]:
        return None

    nse_data = fetch_nse_data(ticker)
    if not nse_data:
        return None

    res_df = identify_resistance(nse_data, current_price)
    if res_df is None or res_df.empty:
        return None

    best_row = res_df.loc[res_df["callPrice"].idxmax()]
    strike = float(best_row["strikePrice"])

    result = {
        "Ticker": ticker,
        "Current_Price": round(current_price, 2),
        "Yesterday_Close": round(closes[-1], 2),
        "Price_Drop_Percent": round(((closes[-1] - current_price) / closes[-1]) * 100, 2),
        "Momentum_Gain_Percent": round(max_gain, 2),
        "Green_Candle_Count": int(max_green),
        "Momentum_Start_Date": dates[best_start].strftime("%Y-%m-%d"),
        "Momentum_End_Date": dates[best_end].strftime("%Y-%m-%d"),
        "Strike_Price": strike,
        "Status": "Momentum Loss",
        "Last_Scanned": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
    }

    msg = (f"*📉 Momentum Loss Alert*\n"
           f"Stock: *{ticker}*\nCurrent: *₹{current_price:.2f}*\n"
           f"Drop: *{result['Price_Drop_Percent']:.2f}%*\n"
           f"Suggested Strike: *₹{strike:.2f}*")
    send_telegram(bot_token, chat_id, msg)
    return result

# ====================== BREAKOUT CHECK ======================
def check_breakout(ticker: str, hist: pd.DataFrame, current_price: float,
                   nse_data: Dict, config: Dict) -> Optional[Dict]:
    day_high = hist["Close"].max()
    if current_price <= day_high:
        return None

    res_df = identify_resistance(nse_data, current_price)
    if res_df is None or res_df.empty:
        return None

    best_row = res_df.loc[res_df["callPrice"].idxmax()]
    strike = float(best_row["strikePrice"])
    pct_above = round(((current_price - day_high) / day_high) * 100, 2)

    result = {
        "Ticker": ticker,
        "Current_Price": round(current_price, 2),
        f"{config['lookback_days']}d_High": round(day_high, 2),
        "Percent_Above_High": pct_above,
        "Suggested_Call_Strike": strike,
        "Last_Scanned": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
    }

    # ← NEW: Telegram alert for breakouts
    bot_token = config["telegram_bot_token"]
    chat_id = config["telegram_chat_id"]
    msg = (f"*🚀 Breakout Alert — New High!*\n"
           f"Stock: *{ticker}*\nCurrent: *₹{current_price:.2f}*\n"
           f"{config['lookback_days']}d High: *₹{day_high:.2f}*\n"
           f"Above High: *+{pct_above:.2f}%*\n"
           f"Suggested Call Strike: *₹{strike:.2f}*")
    send_telegram(bot_token, chat_id, msg)
    return result

# ====================== CORE SCREENER ======================
def is_market_open() -> bool:
    now = datetime.now(IST)
    if now.weekday() >= 5:   # Sat/Sun
        return False
    market_open  = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return market_open <= now <= market_close

def run_screening(tickers: List[str], config: Dict,
                  baseline: Dict[str, float],
                  progress_callback=None) -> Tuple[List[Dict], List[Dict]]:
    """
    Main screening logic.
    1. Filter tickers by surge_filter_percent vs baseline.
    2. For each surge candidate, do momentum + breakout checks.
    """
    surge_pct = config["surge_filter_percent"]
    end_date  = date.today()
    start_date = end_date - timedelta(days=config["lookback_days"])

    # --- Step 1: Quick price check to filter candidates ---
    if baseline:
        candidates = get_surge_candidates(tickers, baseline, surge_pct)
        surge_tickers = [(tk, cur) for tk, cur, _ in candidates]
        print(f"Surge filter ({surge_pct}%): {len(surge_tickers)}/{len(tickers)} pass")
    else:
        # No baseline yet — screen all (first run)
        surge_tickers = []
        for tk in tickers:
            try:
                info = yf.Ticker(f"{tk}.NS").fast_info
                cur = getattr(info, "last_price", None)
                if cur:
                    surge_tickers.append((tk, float(cur)))
                time.sleep(random.uniform(0.2, 0.4))
            except:
                pass

    if not surge_tickers:
        return [], []

    session = get_nse_session()
    if session is None:
        return [], []

    momentum_results, breakout_results = [], []
    total = len(surge_tickers)

    for idx, (ticker, current_price) in enumerate(surge_tickers):
        if progress_callback:
            progress_callback(idx, total, ticker)

        hist = fetch_historical(ticker, start_date, end_date)
        if hist is None or len(hist) < config["min_green_candles"] + 5:
            time.sleep(random.uniform(1.0, 2.0))
            continue

        nse_data = fetch_nse_data(ticker)
        if nse_data is None or "records" not in nse_data:
            time.sleep(random.uniform(2.0, 4.0))
            continue

        # Use live price from NSE if available, else use yfinance fast_info
        live_price = nse_data["records"].get("underlyingValue", current_price)

        mom = check_momentum(ticker, hist, live_price, config)
        if mom:
            momentum_results.append(mom)

        brk = check_breakout(ticker, hist, live_price, nse_data, config)
        if brk:
            breakout_results.append(brk)

        # Respectful delay — randomized to avoid pattern detection
        time.sleep(random.uniform(3.0, 5.5))

    # Log scan
    log = load_scan_log()
    log.append({
        "time": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
        "total_tickers": len(tickers),
        "surge_filtered": total,
        "momentum_hits": len(momentum_results),
        "breakout_hits": len(breakout_results),
    })
    log = log[-50:]   # keep last 50 entries
    save_scan_log(log)

    return momentum_results, breakout_results

# ====================== AUTO-SCAN (APScheduler) ======================
_scheduler: Optional[BackgroundScheduler] = None

def start_scheduler(config: Dict):
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)

    _scheduler = BackgroundScheduler(timezone=IST)

    def auto_scan_job():
        cfg = load_config()
        if cfg["market_hours_only"] and not is_market_open():
            print("Auto-scan skipped: market closed")
            return
        print(f"Auto-scan running at {datetime.now(IST).strftime('%H:%M:%S')}")
        tickers  = load_tickers()
        baseline = load_baseline()
        if not baseline:
            print("No baseline found — capturing now.")
            baseline = capture_baseline(tickers)

        mom, brk = run_screening(tickers, cfg, baseline)

        # Persist results (append new unique hits)
        if mom:
            existing = load_screening_data()
            existing_tickers = {r["Ticker"] for r in existing}
            for r in mom:
                if r["Ticker"] not in existing_tickers:
                    existing.append(r)
            save_screening_data(existing)

        if brk:
            existing_brk = load_breakout_data()
            existing_tickers = {r["Ticker"] for r in existing_brk}
            for r in brk:
                if r["Ticker"] not in existing_tickers:
                    existing_brk.append(r)
            save_breakout_data(existing_brk)

        print(f"Auto-scan done. Momentum: {len(mom)}, Breakouts: {len(brk)}")

    interval_min = max(config.get("auto_scan_interval_min", 30), 10)  # min 10min
    _scheduler.add_job(auto_scan_job, IntervalTrigger(minutes=interval_min),
                       id="auto_scan", replace_existing=True)
    _scheduler.start()
    print(f"Scheduler started. Interval: every {interval_min} min.")

# ====================== CHART ======================
def generate_candlestick(ticker: str, strike_price: float):
    hist = fetch_historical(ticker, date.today() - timedelta(days=60), date.today())
    if hist is None or hist.empty:
        return None
    fig = go.Figure(data=[go.Candlestick(
        x=hist.index, open=hist["Open"], high=hist["High"],
        low=hist["Low"], close=hist["Close"])])
    fig.add_hline(y=strike_price, line_dash="dash", line_color="red",
                  annotation_text=f"Strike ₹{strike_price}")
    fig.update_layout(
        title=f"{ticker} — 60 Day Chart (Strike: ₹{strike_price})",
        xaxis_title="Date", yaxis_title="Price (₹)",
        xaxis_rangeslider_visible=False, height=500,
        template="plotly_dark")
    return fig

# ====================== STREAMLIT UI ======================
def main():
    st.set_page_config(
        page_title="NSE Smart Screener",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded")

    # ---- Custom CSS ----
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
    .metric-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 10px;
    }
    .status-pill {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.05em;
    }
    .pill-green  { background: #052e16; color: #4ade80; border: 1px solid #16a34a; }
    .pill-red    { background: #2d0b0b; color: #f87171; border: 1px solid #dc2626; }
    .pill-yellow { background: #1c1a00; color: #facc15; border: 1px solid #ca8a04; }
    .scan-log-row { font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; }
    </style>
    """, unsafe_allow_html=True)

    st.title("📈 NSE Smart Screener")
    st.caption("Momentum Loss · Breakout · Auto-Scan · Surge Filter")

    config = load_config()

    # ---- Init session state ----
    for key, default in [
        ("screening_data", load_screening_data()),
        ("breakout_data",  load_breakout_data()),
        ("scan_log",       load_scan_log()),
        ("baseline",       load_baseline()),
        ("last_scan_time", None),
        ("scheduler_started", False),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # ---- Start scheduler once ----
    if not st.session_state.scheduler_started and config.get("auto_scan_enabled", True):
        start_scheduler(config)
        st.session_state.scheduler_started = True

    # ====================== SIDEBAR ======================
    with st.sidebar:
        st.markdown("### ⚙️ Settings")

        with st.expander("📊 Screening Parameters", expanded=True):
            min_gain  = st.number_input("Min Momentum Gain (%)", value=config["min_gain_percent"], min_value=5.0, step=1.0)
            min_green = st.number_input("Min Green Candles",      value=config["min_green_candles"], min_value=2, step=1)
            lookback  = st.number_input("Lookback Days",           value=config["lookback_days"], min_value=10, step=1)
            proximity = st.number_input("Price Proximity (%)",     value=config["price_proximity_percent"], min_value=0.1, step=0.1)
            surge_pct = st.number_input("Surge Filter (% from baseline)", value=config["surge_filter_percent"],
                                        min_value=0.5, step=0.5,
                                        help="Only screen stocks that moved ≥ this % from baseline price")

        with st.expander("⏱️ Auto-Scan Settings", expanded=True):
            auto_enabled   = st.checkbox("Enable Auto-Scan", value=config["auto_scan_enabled"])
            interval_min   = st.number_input("Scan Interval (min)", value=config["auto_scan_interval_min"],
                                             min_value=10, step=5)
            market_only    = st.checkbox("Market Hours Only (9:15–15:30 IST)",
                                         value=config["market_hours_only"])

        with st.expander("📬 Telegram Alerts"):
            bot_token = st.text_input("Bot Token", value=config["telegram_bot_token"], type="password")
            chat_id   = st.text_input("Chat ID",   value=config["telegram_chat_id"])

        with st.expander("📋 Tickers"):
            uploaded = st.file_uploader("Upload CSV (SYMBOL column)", type=["csv"])
            if uploaded:
                df_up = pd.read_csv(uploaded)
                if "SYMBOL" in df_up.columns:
                    df_up.to_csv(STORED_TICKERS_PATH, index=False)
                    st.session_state.baseline = {}
                    save_baseline({})
                    st.success(f"✅ {len(df_up)} tickers saved! Baseline reset.")
            specific = st.text_input("Specific Tickers (comma-separated)", "")

        # Save config if changed
        new_cfg = {
            "min_gain_percent": min_gain, "min_green_candles": int(min_green),
            "lookback_days": int(lookback), "price_proximity_percent": proximity,
            "surge_filter_percent": surge_pct, "auto_scan_interval_min": int(interval_min),
            "auto_scan_enabled": auto_enabled, "market_hours_only": market_only,
            "telegram_bot_token": bot_token, "telegram_chat_id": chat_id,
        }
        if new_cfg != {k: config.get(k) for k in new_cfg}:
            config.update(new_cfg)
            save_config(config)
            if auto_enabled:
                start_scheduler(config)
            st.success("Settings saved.")

    # ====================== TOP STATUS BAR ======================
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        market_status = "🟢 OPEN" if is_market_open() else "🔴 CLOSED"
        st.metric("Market", market_status)
    with col_b:
        baseline = st.session_state.baseline
        st.metric("Baseline Stocks", len(baseline))
    with col_c:
        auto_label = f"Every {config['auto_scan_interval_min']} min" if config["auto_scan_enabled"] else "Off"
        st.metric("Auto-Scan", auto_label)
    with col_d:
        last = st.session_state.last_scan_time or "—"
        st.metric("Last Manual Scan", last)

    st.divider()

    # ====================== ACTION BUTTONS ======================
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("📸 Capture Baseline", help="Store current prices as baseline for surge detection"):
            tickers = load_tickers()
            with st.spinner(f"Capturing prices for {len(tickers)} tickers…"):
                baseline = capture_baseline(tickers)
                st.session_state.baseline = baseline
            st.success(f"✅ Baseline captured for {len(baseline)} stocks!")
            st.rerun()

    with col2:
        if st.button("🔄 Screen All (Surge Filter)", type="primary"):
            tickers  = load_tickers()
            baseline = st.session_state.baseline
            if not baseline:
                st.warning("⚠️ No baseline found. Capturing now…")
                baseline = capture_baseline(tickers)
                st.session_state.baseline = baseline

            progress = st.progress(0)
            status_text = st.empty()

            def update_progress(idx, total, ticker):
                pct = int((idx / max(total, 1)) * 100)
                progress.progress(pct)
                status_text.text(f"Scanning {ticker} ({idx+1}/{total})…")

            with st.spinner("Running screener on surge candidates…"):
                mom, brk = run_screening(tickers, config, baseline, update_progress)
                st.session_state.screening_data = mom
                st.session_state.breakout_data  = brk
                st.session_state.last_scan_time = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.scan_log = load_scan_log()
                save_screening_data(mom)
                save_breakout_data(brk)

            progress.progress(100)
            status_text.text("Done!")
            st.rerun()

    with col3:
        if st.button("🎯 Screen Specific") and specific.strip():
            tickers  = [t.strip().upper() for t in specific.split(",")]
            baseline = {tk: st.session_state.baseline.get(tk, 0) for tk in tickers}
            with st.spinner("Scanning specific tickers…"):
                mom, brk = run_screening(tickers, config, {})   # No baseline filter for specific
                st.session_state.screening_data = mom
                st.session_state.breakout_data  = brk
                st.session_state.last_scan_time = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
            st.rerun()

    with col4:
        if st.button("🗑️ Clear Results"):
            st.session_state.screening_data = []
            st.session_state.breakout_data  = []
            save_screening_data([])
            save_breakout_data([])
            st.rerun()

    st.divider()

    # ====================== RESULTS ======================
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📉 Momentum Loss", "🚀 Breakouts / New Highs", "📊 Surge Candidates", "🕒 Scan Log"])

    with tab1:
        st.subheader("📉 Momentum Loss Signals")
        data_mom = st.session_state.screening_data
        if data_mom:
            df = pd.DataFrame(data_mom)
            st.dataframe(df, use_container_width=True)
            sel = st.selectbox("View chart for:", df["Ticker"].tolist(), key="mom_sel")
            row = df[df["Ticker"] == sel].iloc[0]
            fig = generate_candlestick(sel, row["Strike_Price"])
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No momentum loss signals yet. Run a scan or wait for auto-scan.")

    with tab2:
        st.subheader("🚀 Breakout / New High Stocks")
        data_brk = st.session_state.breakout_data
        if data_brk:
            df = pd.DataFrame(data_brk)
            st.dataframe(df, use_container_width=True,
                column_config={"Suggested_Call_Strike": st.column_config.NumberColumn(
                    "Suggested Call Strike", format="₹%.2f")})
            sel = st.selectbox("View chart for:", df["Ticker"].tolist(), key="brk_sel")
            row = df[df["Ticker"] == sel].iloc[0]
            fig = generate_candlestick(sel, row["Suggested_Call_Strike"])
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No breakout signals found yet.")

    with tab3:
        st.subheader("⚡ Live Surge Candidates")
        st.caption(f"Stocks with ≥ {config['surge_filter_percent']}% move from baseline")
        baseline = st.session_state.baseline
        if baseline:
            if st.button("🔁 Refresh Surge List"):
                tickers = load_tickers()
                with st.spinner("Checking prices…"):
                    candidates = get_surge_candidates(tickers, baseline, config["surge_filter_percent"])
                if candidates:
                    df_cand = pd.DataFrame(candidates, columns=["Ticker", "Current Price", "Change %"])
                    df_cand = df_cand.sort_values("Change %", ascending=False)
                    st.dataframe(df_cand, use_container_width=True)
                    st.caption(f"{len(candidates)} stocks surged. These will be screened in next scan.")
                else:
                    st.info("No stocks crossed the surge threshold currently.")
        else:
            st.warning("Capture baseline first (📸 button) to enable surge detection.")

    with tab4:
        st.subheader("🕒 Auto-Scan Log")
        log = st.session_state.scan_log or load_scan_log()
        if log:
            df_log = pd.DataFrame(reversed(log))
            st.dataframe(df_log, use_container_width=True)
        else:
            st.info("No scans logged yet.")

    # ---- Auto refresh every 60s to pick up background scan results ----
    st.markdown("""
    <script>
    setTimeout(function() { window.location.reload(); }, 60000);
    </script>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()