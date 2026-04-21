"""
NSE Smart Screener
──────────────────
• Momentum Loss detection
• Breakout / New High detection
• Surge filter (baseline price comparison)
• Auto-scan via APScheduler
• Strike Price LTP Tracker:
    - Jab bhi koi ticker screen hota hai, uski Suggested Strike ka option LTP note hota hai
    - Har scan interval par us option ki LTP dobara fetch hoti hai
    - Agar current LTP < last noted LTP → Telegram alert
"""

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
from datetime import date, datetime, timedelta
import pytz
import plotly.graph_objects as go
import requests
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

# ====================== CONSTANTS ======================
STORED_TICKERS_PATH  = "tickers.csv"
CONFIG_FILE          = "config.json"
SCREENING_DATA_FILE  = "screening_data.json"
BREAKOUT_DATA_FILE   = "breakout_data.json"
BASELINE_FILE        = "price_baseline.json"
SCAN_LOG_FILE        = "scan_log.json"
STRIKE_TRACKER_FILE  = "strike_tracker.json"

IST = pytz.timezone("Asia/Kolkata")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"
    ),
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
        "min_gain_percent":        20.0,
        "min_green_candles":       3,
        "lookback_days":           30,
        "telegram_bot_token":      "",
        "telegram_chat_id":        "",
        "price_proximity_percent": 1.0,
        "surge_filter_percent":    3.0,
        "auto_scan_interval_min":  15,
        "auto_scan_enabled":       True,
        "market_hours_only":       True,
        "strike_drop_alert":       True,
    }
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            saved = json.load(f)
            defaults.update(saved)
    return defaults

def save_config(config: Dict):
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=4)

# ====================== PERSISTENCE ======================
def _load_json(path, default):
    try:
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return default

def _save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

def load_screening_data() -> List[Dict]:  return _load_json(SCREENING_DATA_FILE, [])
def save_screening_data(d):               _save_json(SCREENING_DATA_FILE, d)
def load_breakout_data() -> List[Dict]:   return _load_json(BREAKOUT_DATA_FILE, [])
def save_breakout_data(d):                _save_json(BREAKOUT_DATA_FILE, d)
def load_baseline() -> Dict:              return _load_json(BASELINE_FILE, {})
def save_baseline(d):                     _save_json(BASELINE_FILE, d)
def load_scan_log() -> List[Dict]:        return _load_json(SCAN_LOG_FILE, [])
def save_scan_log(d):                     _save_json(SCAN_LOG_FILE, d)

# ──────────────────────────────────────────────────────────────
#  STRIKE TRACKER
#
#  strike_tracker.json structure:
#  {
#    "HDFCBANK": {
#      "strike": 1700.0,
#      "expiry": "30-Apr-2026",
#      "option_type": "CE",
#      "first_noted_ltp": 45.30,
#      "last_ltp": 42.10,
#      "last_noted_time": "2025-04-15 10:30:00",
#      "history": [{"time": "...", "ltp": 45.30}, ...],
#      "source": "momentum" | "breakout"
#    }
#  }
# ──────────────────────────────────────────────────────────────
def load_strike_tracker() -> Dict:  return _load_json(STRIKE_TRACKER_FILE, {})
def save_strike_tracker(d):         _save_json(STRIKE_TRACKER_FILE, d)

# ====================== TELEGRAM ======================
async def _send_tg(bot_token: str, chat_id: str, message: str):
    if not bot_token or not chat_id:
        return
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
    try:
        async with aiohttp.ClientSession() as sess:
            await sess.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10))
    except Exception as e:
        print(f"Telegram error: {e}")

def send_telegram(bot_token: str, chat_id: str, message: str):
    if not bot_token or not chat_id:
        return
    MAX = 4096
    for i in range(0, len(message), MAX):
        chunk = message[i:i + MAX].strip()
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(_send_tg(bot_token, chat_id, chunk))
            loop.close()
        except Exception as e:
            print(f"Telegram send error: {e}")
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
        url = (
            f"https://www.nseindia.com/api/option-chain-v3"
            f"?type=Equity&symbol={ticker}&expiry={expiry}"
        )
        r = session.get(url, timeout=15)
        if r.status_code != 200 or not r.text.strip().startswith("{"):
            if r.status_code in (429, 403):
                reset_nse_session()
            return None

        data = r.json()
        data["_selected_expiry"] = expiry

        try:
            qr = session.get(
                f"https://www.nseindia.com/api/quote-equity?symbol={ticker}",
                timeout=10,
            )
            if qr.status_code == 200:
                lp = qr.json().get("priceInfo", {}).get("lastPrice")
                if lp:
                    data.setdefault("records", {})["underlyingValue"] = lp
        except Exception:
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
        s  = r.get("strikePrice", 0)
        ce = r.get("CE", {})
        pe = r.get("PE", {})
        calls.append({
            "strikePrice": s,
            "callOI":    ce.get("openInterest", 0),
            "callPrice": ce.get("lastPrice", 0),
        })
        puts.append({"strikePrice": s, "putOI": pe.get("openInterest", 0)})
    return pd.DataFrame(calls), pd.DataFrame(puts)

def identify_resistance(data: Dict, price: float) -> Optional[pd.DataFrame]:
    call_df, put_df = process_option_data(data)
    if call_df.empty or put_df.empty:
        return None
    combined = pd.DataFrame({
        "strikePrice": call_df["strikePrice"],
        "totalOI":     call_df["callOI"] + put_df["putOI"],
        "callPrice":   call_df["callPrice"],
    })
    above = combined[combined["strikePrice"] > price]
    return above.sort_values("totalOI", ascending=False) if not above.empty else None

# ── Fetch specific option contract LTP ───────────────────────
def fetch_option_ltp(ticker: str, strike: float, expiry: str,
                     option_type: str = "CE") -> Optional[float]:
    """Fetch LTP of a specific strike/expiry/type option from NSE."""
    session = get_nse_session()
    if session is None:
        return None
    try:
        url = (
            f"https://www.nseindia.com/api/option-chain-v3"
            f"?type=Equity&symbol={ticker}&expiry={expiry}"
        )
        r = session.get(url, timeout=15)
        if r.status_code != 200 or not r.text.strip().startswith("{"):
            if r.status_code in (429, 403):
                reset_nse_session()
            return None
        records = r.json().get("filtered", {}).get("data", [])
        for rec in records:
            if rec.get("strikePrice") == strike:
                opt = rec.get(option_type, {})
                ltp = opt.get("lastPrice")
                if ltp and ltp > 0:
                    return float(ltp)
    except Exception as e:
        print(f"Option LTP error {ticker} {strike}{option_type}: {e}")
    return None

# ====================== HISTORICAL DATA ======================
def fetch_historical(ticker: str, start: date, end: date) -> Optional[pd.DataFrame]:
    try:
        hist = yf.Ticker(f"{ticker}.NS").history(
            start=start, end=end + timedelta(days=1)
        )
        return hist if not hist.empty else None
    except Exception as e:
        print(f"yfinance error {ticker}: {e}")
        return None

# ====================== BASELINE ======================
def capture_baseline(tickers: List[str]) -> Dict[str, float]:
    baseline = {}
    for tk in tickers:
        try:
            info  = yf.Ticker(f"{tk}.NS").fast_info
            price = getattr(info, "last_price", None)
            if price:
                baseline[tk] = round(float(price), 2)
            time.sleep(random.uniform(0.3, 0.7))
        except Exception as e:
            print(f"Baseline error {tk}: {e}")
    save_baseline(baseline)
    return baseline

def get_surge_candidates(
    tickers: List[str], baseline: Dict[str, float], surge_pct: float
) -> List[Tuple[str, float, float]]:
    candidates = []
    for tk in tickers:
        if tk not in baseline:
            continue
        try:
            info = yf.Ticker(f"{tk}.NS").fast_info
            cur  = getattr(info, "last_price", None)
            if cur is None:
                continue
            cur  = float(cur)
            base = baseline[tk]
            chg  = ((cur - base) / base) * 100
            if abs(chg) >= surge_pct:
                candidates.append((tk, round(cur, 2), round(chg, 2)))
            time.sleep(random.uniform(0.2, 0.5))
        except Exception:
            pass
    return candidates

# ====================== STRIKE TRACKER — REGISTER ======================
def register_strike(ticker: str, strike: float, expiry: str,
                    ltp: float, source: str):
    """Register or update a ticker's strike in the tracker."""
    tracker = load_strike_tracker()
    now_str = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")

    if ticker not in tracker:
        tracker[ticker] = {
            "strike":          strike,
            "expiry":          expiry,
            "option_type":     "CE",
            "first_noted_ltp": ltp,
            "last_ltp":        ltp,
            "last_noted_time": now_str,
            "history":         [{"time": now_str, "ltp": ltp}],
            "source":          source,
        }
        print(f"Tracker: NEW {ticker} | Strike ₹{strike} | LTP ₹{ltp}")
    else:
        # Update strike if changed
        if tracker[ticker]["strike"] != strike:
            tracker[ticker]["strike"] = strike
            tracker[ticker]["expiry"] = expiry
        tracker[ticker]["last_ltp"]        = ltp
        tracker[ticker]["last_noted_time"] = now_str
        tracker[ticker]["history"].append({"time": now_str, "ltp": ltp})
        tracker[ticker]["history"] = tracker[ticker]["history"][-100:]

    save_strike_tracker(tracker)

# ====================== STRIKE TRACKER — PERIODIC DROP CHECK ======================
def check_strike_drops(config: Dict):
    """
    Har scan interval par call hota hai.
    Har tracked ticker ke liye:
      1. Fresh option LTP fetch karo
      2. Agar current LTP < last noted LTP → Telegram drop alert
      3. History update karo
    """
    tracker = load_strike_tracker()
    if not tracker:
        print("Tracker: nothing to check.")
        return

    bot_token = config["telegram_bot_token"]
    chat_id   = config["telegram_chat_id"]
    now_str   = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")

    for ticker, info in list(tracker.items()):
        strike    = info["strike"]
        expiry    = info["expiry"]
        opt_type  = info.get("option_type", "CE")
        last_ltp  = info["last_ltp"]
        first_ltp = info.get("first_noted_ltp", last_ltp)

        current_ltp = fetch_option_ltp(ticker, strike, expiry, opt_type)
        time.sleep(random.uniform(2.5, 4.0))

        if current_ltp is None:
            print(f"Tracker: LTP unavailable for {ticker} {strike}{opt_type}")
            continue

        # Always record in history
        info["history"].append({"time": now_str, "ltp": current_ltp})
        info["history"] = info["history"][-100:]

        drop_pct       = round(((last_ltp - current_ltp) / last_ltp) * 100, 2)
        total_drop_pct = round(((first_ltp - current_ltp) / first_ltp) * 100, 2)

        print(f"Tracker {ticker}: prev={last_ltp} now={current_ltp} ({drop_pct:+.2f}%)")

        # ── DROP DETECTED ──
        if current_ltp < last_ltp:
            msg = (
                f"*📉 Option Price DROP Alert*\n\n"
                f"Stock : *{ticker}*\n"
                f"Strike: *₹{strike:.2f} {opt_type}*\n"
                f"Expiry: {expiry}\n\n"
                f"Pehle note ki gayi LTP : *₹{last_ltp:.2f}*\n"
                f"Abhi ki LTP             : *₹{current_ltp:.2f}*\n"
                f"Is scan mein giri       : *{drop_pct:.2f}%* ⬇️\n"
                f"Pehli reading se giri   : *{total_drop_pct:.2f}%*\n\n"
                f"Pehli noted LTP : ₹{first_ltp:.2f}\n"
                f"Time : {now_str}\n"
                f"Source : {info.get('source','—').upper()}"
            )
            send_telegram(bot_token, chat_id, msg)
            print(f"Tracker: DROP alert sent for {ticker}")

        # Update last_ltp to current so next scan compares against today's reading
        info["last_ltp"]        = current_ltp
        info["last_noted_time"] = now_str

    save_strike_tracker(tracker)

def remove_from_tracker(ticker: str):
    tracker = load_strike_tracker()
    if ticker in tracker:
        del tracker[ticker]
        save_strike_tracker(tracker)

# ====================== MOMENTUM LOSS ======================
def check_momentum(ticker: str, hist: pd.DataFrame, current_price: float,
                   nse_data: Dict, config: Dict) -> Optional[Dict]:
    min_gain  = config["min_gain_percent"]
    min_green = config["min_green_candles"]

    if len(hist) < min_green + 5:
        return None

    hist   = hist.sort_index()
    closes = hist["Close"].values
    dates  = hist.index

    max_green = max_gain = 0.0
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

    res_df = identify_resistance(nse_data, current_price)
    if res_df is None or res_df.empty:
        return None

    best_row   = res_df.loc[res_df["callPrice"].idxmax()]
    strike     = float(best_row["strikePrice"])
    option_ltp = float(best_row["callPrice"])
    expiry     = nse_data.get("_selected_expiry", "")

    result = {
        "Ticker":                ticker,
        "Current_Price":         round(current_price, 2),
        "Yesterday_Close":       round(closes[-1], 2),
        "Price_Drop_Percent":    round(((closes[-1] - current_price) / closes[-1]) * 100, 2),
        "Momentum_Gain_Percent": round(max_gain, 2),
        "Green_Candle_Count":    int(max_green),
        "Momentum_Start_Date":   dates[best_start].strftime("%Y-%m-%d"),
        "Momentum_End_Date":     dates[best_end].strftime("%Y-%m-%d"),
        "Strike_Price":          strike,
        "Strike_Option_LTP":     option_ltp,
        "Expiry":                expiry,
        "Status":                "Momentum Loss",
        "Last_Scanned":          datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Register in tracker
    register_strike(ticker, strike, expiry, option_ltp, source="momentum")

    msg = (
        f"*📉 Momentum Loss Alert*\n"
        f"Stock: *{ticker}*  |  Price: *₹{current_price:.2f}*\n"
        f"Drop: *{result['Price_Drop_Percent']:.2f}%*\n"
        f"Strike: *₹{strike:.2f} CE* ({expiry})\n"
        f"Option LTP abhi: *₹{option_ltp:.2f}*\n"
        f"_Har {config['auto_scan_interval_min']} min mein LTP track hogi…_"
    )
    send_telegram(config["telegram_bot_token"], config["telegram_chat_id"], msg)
    return result

# ====================== BREAKOUT ======================
def check_breakout(ticker: str, hist: pd.DataFrame, current_price: float,
                   nse_data: Dict, config: Dict) -> Optional[Dict]:
    day_high = hist["Close"].max()
    if current_price <= day_high:
        return None

    res_df = identify_resistance(nse_data, current_price)
    if res_df is None or res_df.empty:
        return None

    best_row   = res_df.loc[res_df["callPrice"].idxmax()]
    strike     = float(best_row["strikePrice"])
    option_ltp = float(best_row["callPrice"])
    expiry     = nse_data.get("_selected_expiry", "")
    pct_above  = round(((current_price - day_high) / day_high) * 100, 2)

    result = {
        "Ticker":                ticker,
        "Current_Price":         round(current_price, 2),
        f"{config['lookback_days']}d_High": round(day_high, 2),
        "Percent_Above_High":    pct_above,
        "Suggested_Call_Strike": strike,
        "Strike_Option_LTP":     option_ltp,
        "Expiry":                expiry,
        "Last_Scanned":          datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Register in tracker
    register_strike(ticker, strike, expiry, option_ltp, source="breakout")

    msg = (
        f"*🚀 Breakout Alert — New High!*\n"
        f"Stock: *{ticker}*  |  Price: *₹{current_price:.2f}*\n"
        f"{config['lookback_days']}d High: *₹{day_high:.2f}*  (+{pct_above:.2f}%)\n"
        f"Strike: *₹{strike:.2f} CE* ({expiry})\n"
        f"Option LTP abhi: *₹{option_ltp:.2f}*\n"
        f"_Har {config['auto_scan_interval_min']} min mein LTP track hogi…_"
    )
    send_telegram(config["telegram_bot_token"], config["telegram_chat_id"], msg)
    return result

# ====================== CORE SCREENER ======================
def is_market_open() -> bool:
    now = datetime.now(IST)
    if now.weekday() >= 5:
        return False
    o = now.replace(hour=9,  minute=15, second=0, microsecond=0)
    c = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return o <= now <= c

def run_screening(
    tickers: List[str],
    config: Dict,
    baseline: Dict[str, float],
    progress_callback=None,
) -> Tuple[List[Dict], List[Dict]]:

    surge_pct  = config["surge_filter_percent"]
    end_date   = date.today()
    start_date = end_date - timedelta(days=config["lookback_days"])

    if baseline:
        candidates = get_surge_candidates(tickers, baseline, surge_pct)
        surge_tickers = [(tk, cur) for tk, cur, _ in candidates]
    else:
        surge_tickers = []
        for tk in tickers:
            try:
                info = yf.Ticker(f"{tk}.NS").fast_info
                cur  = getattr(info, "last_price", None)
                if cur:
                    surge_tickers.append((tk, float(cur)))
                time.sleep(random.uniform(0.2, 0.4))
            except Exception:
                pass

    if not surge_tickers:
        return [], []

    if get_nse_session() is None:
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

        live_price = nse_data["records"].get("underlyingValue", current_price)

        mom = check_momentum(ticker, hist, live_price, nse_data, config)
        if mom:
            momentum_results.append(mom)

        brk = check_breakout(ticker, hist, live_price, nse_data, config)
        if brk:
            breakout_results.append(brk)

        time.sleep(random.uniform(3.0, 5.5))

    log = load_scan_log()
    log.append({
        "time":            datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
        "total_tickers":   len(tickers),
        "surge_filtered":  total,
        "momentum_hits":   len(momentum_results),
        "breakout_hits":   len(breakout_results),
        "tracked_strikes": len(load_strike_tracker()),
    })
    save_scan_log(log[-50:])

    return momentum_results, breakout_results

# ====================== SCHEDULER ======================
_scheduler: Optional[BackgroundScheduler] = None

def start_scheduler(config: Dict):
    global _scheduler
    if _scheduler and _scheduler.running:
        try:
            _scheduler.shutdown(wait=False)
        except Exception:
            pass

    _scheduler = BackgroundScheduler(timezone=IST)

    def auto_scan_job():
        cfg = load_config()
        if cfg["market_hours_only"] and not is_market_open():
            print("[AutoScan] Skipped: market closed")
            return

        print(f"[AutoScan] {datetime.now(IST).strftime('%H:%M:%S')}")
        tickers  = load_tickers()
        baseline = load_baseline()

        if not baseline:
            baseline = capture_baseline(tickers)

        # Step 1: Screen surge candidates
        mom, brk = run_screening(tickers, cfg, baseline)

        if mom:
            existing = load_screening_data()
            seen = {r["Ticker"] for r in existing}
            for r in mom:
                if r["Ticker"] not in seen:
                    existing.append(r)
            save_screening_data(existing)

        if brk:
            existing_brk = load_breakout_data()
            seen_brk = {r["Ticker"] for r in existing_brk}
            for r in brk:
                if r["Ticker"] not in seen_brk:
                    existing_brk.append(r)
            save_breakout_data(existing_brk)

        # Step 2: Check drop on all tracked options
        if cfg.get("strike_drop_alert", True):
            print("[AutoScan] Checking strike drops…")
            check_strike_drops(cfg)

        print(f"[AutoScan] Done. Mom={len(mom)} Brk={len(brk)}")

    interval_min = max(config.get("auto_scan_interval_min", 15), 10)
    _scheduler.add_job(
        auto_scan_job,
        IntervalTrigger(minutes=interval_min),
        id="auto_scan",
        replace_existing=True,
    )
    _scheduler.start()
    print(f"[Scheduler] Interval: {interval_min} min.")

# ====================== CHARTS ======================
def generate_candlestick(ticker: str, strike_price: float):
    hist = fetch_historical(ticker, date.today() - timedelta(days=60), date.today())
    if hist is None or hist.empty:
        return None
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=hist.index, open=hist["Open"], high=hist["High"],
        low=hist["Low"], close=hist["Close"], name="Price",
    ))
    fig.add_hline(
        y=strike_price, line_dash="dash", line_color="#f97316",
        annotation_text=f"Strike ₹{strike_price}",
        annotation_font_color="#f97316",
    )
    fig.update_layout(
        title=f"{ticker} — 60 Day Chart",
        xaxis_title="Date", yaxis_title="₹",
        xaxis_rangeslider_visible=False,
        height=460, template="plotly_dark",
        paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
    )
    return fig

def generate_option_ltp_chart(ticker: str, info: Dict):
    history = info.get("history", [])
    if len(history) < 2:
        return None

    times  = [h["time"] for h in history]
    ltps   = [h["ltp"]  for h in history]
    colors = []
    for i, ltp in enumerate(ltps):
        if i == 0:
            colors.append("#94a3b8")
        else:
            colors.append("#4ade80" if ltp >= ltps[i - 1] else "#f87171")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times, y=ltps,
        mode="lines+markers",
        line=dict(color="#38bdf8", width=2),
        marker=dict(color=colors, size=9, line=dict(width=1, color="#0f172a")),
        name="Option LTP",
    ))
    fig.add_hline(
        y=info.get("first_noted_ltp", ltps[0]),
        line_dash="dot", line_color="#facc15",
        annotation_text="First Noted LTP",
        annotation_font_color="#facc15",
    )
    fig.update_layout(
        title=(
            f"{ticker}  |  Strike ₹{info['strike']} {info.get('option_type','CE')}  "
            f"|  Expiry: {info.get('expiry','—')}"
        ),
        xaxis_title="Time", yaxis_title="Option LTP (₹)",
        height=380, template="plotly_dark",
        paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
    )
    return fig

# ====================== STREAMLIT UI ======================
def main():
    st.set_page_config(
        page_title="NSE Smart Screener",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@400;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
    .stTabs [data-baseweb="tab"] { font-weight: 700; }
    </style>
    """, unsafe_allow_html=True)

    st.title("📈 NSE Smart Screener")
    st.caption("Momentum Loss · Breakout · Strike LTP Tracker · Auto-Scan")

    config = load_config()

    for key, default in [
        ("screening_data",    load_screening_data()),
        ("breakout_data",     load_breakout_data()),
        ("scan_log",          load_scan_log()),
        ("baseline",          load_baseline()),
        ("strike_tracker",    load_strike_tracker()),
        ("last_scan_time",    None),
        ("scheduler_started", False),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    if not st.session_state.scheduler_started and config.get("auto_scan_enabled", True):
        start_scheduler(config)
        st.session_state.scheduler_started = True

    # ── SIDEBAR ──────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Settings")

        with st.expander("📊 Screening", expanded=True):
            min_gain  = st.number_input("Min Momentum Gain (%)", value=config["min_gain_percent"], min_value=5.0, step=1.0)
            min_green = st.number_input("Min Green Candles",      value=config["min_green_candles"], min_value=2, step=1)
            lookback  = st.number_input("Lookback Days",          value=config["lookback_days"], min_value=10, step=1)
            proximity = st.number_input("Price Proximity (%)",    value=config["price_proximity_percent"], min_value=0.1, step=0.1)
            surge_pct = st.number_input("Surge Filter (% from baseline)", value=config["surge_filter_percent"], min_value=0.5, step=0.5)

        with st.expander("⏱️ Auto-Scan & Tracker", expanded=True):
            auto_enabled = st.checkbox("Enable Auto-Scan",    value=config["auto_scan_enabled"])
            interval_min = st.number_input("Interval (min)",  value=config["auto_scan_interval_min"], min_value=10, step=5)
            market_only  = st.checkbox("Market Hours Only",   value=config["market_hours_only"])
            drop_alert   = st.checkbox("Strike Option Drop Alert", value=config["strike_drop_alert"],
                                       help="Alert jab option LTP pichle scan se neeche aaye")

        with st.expander("📬 Telegram"):
            bot_token = st.text_input("Bot Token", value=config["telegram_bot_token"], type="password")
            chat_id   = st.text_input("Chat ID",   value=config["telegram_chat_id"])

        with st.expander("📋 Tickers"):
            uploaded = st.file_uploader("CSV (SYMBOL column)", type=["csv"])
            if uploaded:
                df_up = pd.read_csv(uploaded)
                if "SYMBOL" in df_up.columns:
                    df_up.to_csv(STORED_TICKERS_PATH, index=False)
                    st.session_state.baseline = {}
                    save_baseline({})
                    st.success(f"✅ {len(df_up)} tickers saved.")
            specific = st.text_input("Specific (comma-separated)", "")

        new_cfg = {
            "min_gain_percent": min_gain, "min_green_candles": int(min_green),
            "lookback_days": int(lookback), "price_proximity_percent": proximity,
            "surge_filter_percent": surge_pct, "auto_scan_interval_min": int(interval_min),
            "auto_scan_enabled": auto_enabled, "market_hours_only": market_only,
            "strike_drop_alert": drop_alert,
            "telegram_bot_token": bot_token, "telegram_chat_id": chat_id,
        }
        if new_cfg != {k: config.get(k) for k in new_cfg}:
            config.update(new_cfg)
            save_config(config)
            if auto_enabled:
                start_scheduler(config)
            st.toast("Settings saved.", icon="✅")

    # ── STATUS BAR ───────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Market",          "🟢 OPEN" if is_market_open() else "🔴 CLOSED")
    c2.metric("Baseline Stocks", len(st.session_state.baseline))
    c3.metric("Tracked Strikes", len(st.session_state.strike_tracker))
    c4.metric("Auto-Scan",       f"Every {config['auto_scan_interval_min']}m" if config["auto_scan_enabled"] else "Off")
    c5.metric("Last Manual Scan", st.session_state.last_scan_time or "—")

    st.divider()

    # ── BUTTONS ──────────────────────────────────────────────
    bc1, bc2, bc3, bc4, bc5 = st.columns(5)

    with bc1:
        if st.button("📸 Capture Baseline"):
            tickers = load_tickers()
            with st.spinner(f"Capturing {len(tickers)} prices…"):
                bl = capture_baseline(tickers)
                st.session_state.baseline = bl
            st.success(f"✅ {len(bl)} prices stored.")
            st.rerun()

    with bc2:
        if st.button("🔄 Screen All", type="primary"):
            tickers  = load_tickers()
            baseline = st.session_state.baseline
            if not baseline:
                baseline = capture_baseline(tickers)
                st.session_state.baseline = baseline

            prog = st.progress(0)
            stat = st.empty()

            def _cb(idx, total, tk):
                prog.progress(int(idx / max(total, 1) * 100))
                stat.text(f"Scanning {tk} ({idx+1}/{total})")

            mom, brk = run_screening(tickers, config, baseline, _cb)
            st.session_state.screening_data = mom
            st.session_state.breakout_data  = brk
            st.session_state.strike_tracker = load_strike_tracker()
            st.session_state.last_scan_time = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
            save_screening_data(mom); save_breakout_data(brk)
            prog.progress(100); stat.text("Done!")
            st.rerun()

    with bc3:
        if st.button("🎯 Screen Specific") and specific.strip():
            tickers = [t.strip().upper() for t in specific.split(",")]
            with st.spinner("Scanning…"):
                mom, brk = run_screening(tickers, config, {})
                st.session_state.screening_data = mom
                st.session_state.breakout_data  = brk
                st.session_state.strike_tracker = load_strike_tracker()
                st.session_state.last_scan_time = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
            st.rerun()

    with bc4:
        if st.button("🔁 Check Drops Now"):
            with st.spinner("Option LTP drop check ho raha hai…"):
                check_strike_drops(config)
                st.session_state.strike_tracker = load_strike_tracker()
            st.success("Drop check complete.")
            st.rerun()

    with bc5:
        if st.button("🗑️ Clear All"):
            for path, val in [(SCREENING_DATA_FILE, []), (BREAKOUT_DATA_FILE, []),
                               (STRIKE_TRACKER_FILE, {})]:
                _save_json(path, val)
            st.session_state.screening_data = []
            st.session_state.breakout_data  = []
            st.session_state.strike_tracker = {}
            st.rerun()

    st.divider()

    # ── TABS ─────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📉 Momentum Loss",
        "🚀 Breakouts",
        "🎯 Strike LTP Tracker",
        "📊 Surge Candidates",
        "🕒 Scan Log",
    ])

    with tab1:
        st.subheader("📉 Momentum Loss Signals")
        data_mom = st.session_state.screening_data
        if data_mom:
            df = pd.DataFrame(data_mom)
            st.dataframe(df, use_container_width=True)
            sel = st.selectbox("Chart:", df["Ticker"].tolist(), key="mom_sel")
            row = df[df["Ticker"] == sel].iloc[0]
            fig = generate_candlestick(sel, row["Strike_Price"])
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Abhi koi momentum loss signal nahi hai.")

    with tab2:
        st.subheader("🚀 Breakout / New High Stocks")
        data_brk = st.session_state.breakout_data
        if data_brk:
            df = pd.DataFrame(data_brk)
            st.dataframe(df, use_container_width=True)
            sel = st.selectbox("Chart:", df["Ticker"].tolist(), key="brk_sel")
            row = df[df["Ticker"] == sel].iloc[0]
            fig = generate_candlestick(sel, row["Suggested_Call_Strike"])
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Abhi koi breakout signal nahi hai.")

    # ── Tab 3: Strike LTP Tracker (MAIN NEW TAB) ──
    with tab3:
        tracker = st.session_state.strike_tracker or load_strike_tracker()
        st.subheader(f"🎯 Strike LTP Tracker  —  {len(tracker)} stocks tracked")
        st.caption(
            "Jab bhi koi stock screen hota hai → uski Suggested CE Strike ki LTP note hoti hai.  "
            "Har scan interval par LTP dobara fetch hoti hai.  "
            "Agar LTP pichle scan se neeche aayi → Telegram alert."
        )

        if not tracker:
            st.info("Abhi koi tracked stock nahi. Pehle scan run karo.")
        else:
            rows = []
            for tk, info in tracker.items():
                ltps       = [h["ltp"] for h in info.get("history", [])]
                last_ltp   = info["last_ltp"]
                first_ltp  = info.get("first_noted_ltp", last_ltp)
                total_chg  = round(((last_ltp - first_ltp) / first_ltp) * 100, 2) if first_ltp else 0
                prev_ltp   = ltps[-2] if len(ltps) >= 2 else last_ltp
                scan_chg   = round(((last_ltp - prev_ltp) / prev_ltp) * 100, 2) if prev_ltp else 0
                rows.append({
                    "Ticker":          tk,
                    "Strike":          f"₹{info['strike']:.0f} CE",
                    "Expiry":          info.get("expiry", "—"),
                    "First LTP (₹)":   first_ltp,
                    "Last LTP (₹)":    last_ltp,
                    "Last Scan Δ%":    scan_chg,
                    "Total Δ%":        total_chg,
                    "Readings":        len(ltps),
                    "Updated":         info.get("last_noted_time", "—"),
                    "Source":          info.get("source", "—").upper(),
                })

            df_t = pd.DataFrame(rows)
            st.dataframe(
                df_t,
                use_container_width=True,
                column_config={
                    "Last Scan Δ%": st.column_config.NumberColumn(format="%.2f %%"),
                    "Total Δ%":     st.column_config.NumberColumn(format="%.2f %%"),
                    "First LTP (₹)": st.column_config.NumberColumn(format="₹%.2f"),
                    "Last LTP (₹)":  st.column_config.NumberColumn(format="₹%.2f"),
                },
            )

            st.markdown("---")
            sel_tk = st.selectbox(
                "LTP history chart dekho:", list(tracker.keys()), key="tracker_sel"
            )
            if sel_tk and sel_tk in tracker:
                info = tracker[sel_tk]
                fig2 = generate_option_ltp_chart(sel_tk, info)
                if fig2:
                    st.plotly_chart(fig2, use_container_width=True)

                with st.expander("Raw LTP History Table"):
                    h_df = pd.DataFrame(info["history"])
                    h_df["Δ LTP"]  = h_df["ltp"].diff().round(2)
                    h_df["Δ %"]    = (h_df["ltp"].pct_change() * 100).round(2)
                    st.dataframe(h_df[::-1].reset_index(drop=True), use_container_width=True)
            else:
                st.info("Ek reading ke baad chart aayega.")

            st.markdown("---")
            col_r1, col_r2 = st.columns([3, 1])
            with col_r1:
                rem_tk = st.selectbox("Tracker se remove karo:", ["—"] + list(tracker.keys()), key="rem_sel")
            with col_r2:
                st.write("")
                st.write("")
                if st.button("❌ Remove") and rem_tk != "—":
                    remove_from_tracker(rem_tk)
                    st.session_state.strike_tracker = load_strike_tracker()
                    st.success(f"{rem_tk} remove ho gaya.")
                    st.rerun()

    with tab4:
        st.subheader("⚡ Surge Candidates")
        baseline = st.session_state.baseline
        if baseline:
            if st.button("🔁 Refresh"):
                with st.spinner("Prices check ho rahe hain…"):
                    cands = get_surge_candidates(load_tickers(), baseline, config["surge_filter_percent"])
                if cands:
                    df_c = pd.DataFrame(cands, columns=["Ticker", "Current Price", "Change %"])
                    st.dataframe(df_c.sort_values("Change %", ascending=False), use_container_width=True)
                else:
                    st.info("Koi stock surge threshold cross nahi kar raha abhi.")
        else:
            st.warning("Pehle baseline capture karo (📸 button).")

    with tab5:
        st.subheader("🕒 Scan Log")
        log = load_scan_log()
        if log:
            st.dataframe(pd.DataFrame(reversed(log)), use_container_width=True)
        else:
            st.info("Abhi tak koi scan log nahi.")

    st.markdown(
        "<script>setTimeout(()=>window.location.reload(),60000);</script>",
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()