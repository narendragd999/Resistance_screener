"""
NSE Momentum Loss Screener — FastAPI Backend
============================================
Run: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import os, json, time, random, asyncio, threading, uuid, io
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import pytz
import requests
import yfinance as yf
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

from fastapi import FastAPI, BackgroundTasks, UploadFile, File, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Sub-routers (Option Charts + SMA Screener) ────────────────
from option_charts_router import router as oc_router
from sma_router import router as sma_router

BASE_DIR = Path(__file__).parent

# ─────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────
TICKERS_FILE   = "tickers.csv"
CONFIG_FILE    = "config.json"
SIGNALS_FILE   = "signals.json"
PROXIMITY_FILE = "proximity_alerts.json"
SCAN_LOG_FILE  = "scan_log.json"
TRACKER_FILE   = "strike_tracker.json"

IST = pytz.timezone("Asia/Kolkata")

NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"
    ),
    "Accept":          "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer":         "https://www.nseindia.com/option-chain",
    "Sec-Fetch-Site":  "same-origin",
    "Sec-Fetch-Mode":  "cors",
    "Connection":      "keep-alive",
}

_nse_session: Optional[requests.Session] = None
_nse_lock = threading.Lock()

# ─────────────────────────────────────────────────────────────
#  JOB MANAGEMENT
# ─────────────────────────────────────────────────────────────
jobs: Dict[str, Dict] = {}
jobs_lock = threading.Lock()

def create_job() -> str:
    job_id = str(uuid.uuid4())
    with jobs_lock:
        jobs[job_id] = {
            "status": "running",
            "progress": 0,
            "total": 0,
            "current_ticker": "",
            "logs": [],
            "result": [],
            "created_at": now_ist_str() if 'now_ist_str' in dir() else "",
        }
    return job_id

def job_log(job_id: str, msg: str, level: str = "info"):
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id]["logs"].append({
                "time": datetime.now(IST).strftime("%H:%M:%S"),
                "msg":  msg,
                "level": level,
            })

def job_progress(job_id: str, current: int, total: int, ticker: str):
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id]["progress"]        = current
            jobs[job_id]["total"]           = total
            jobs[job_id]["current_ticker"]  = ticker

def job_done(job_id: str, result: list, status: str = "done"):
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id]["status"] = status
            jobs[job_id]["result"] = result

# ─────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "lookback_days":             30,
    "min_gain_percent":          10.0,
    "min_green_candles":         3,
    "price_proximity_percent":   1.5,
    "sell_zone_lookback_days":   10,      # ← NEW: how many days to look back for the high
    "ce_above_historical_high":  True,
    "ce_history_days":           30,
    "auto_scan_enabled":         True,
    "auto_scan_interval_min":    15,
    "market_hours_only":         True,
    "telegram_bot_token":        "",
    "telegram_chat_id":          "",
}

def load_config() -> Dict:
    cfg = DEFAULT_CONFIG.copy()
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            cfg.update(json.load(f))
    return cfg

def save_config(cfg: Dict):
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)

# ─────────────────────────────────────────────────────────────
#  PERSISTENCE
# ─────────────────────────────────────────────────────────────
def _rj(path: str, default):
    try:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    except Exception:
        pass
    return default

def _wj(path: str, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

load_signals   = lambda: _rj(SIGNALS_FILE, [])
save_signals   = lambda d: _wj(SIGNALS_FILE, d)
load_proximity = lambda: _rj(PROXIMITY_FILE, [])
save_proximity = lambda d: _wj(PROXIMITY_FILE, d)
load_scan_log  = lambda: _rj(SCAN_LOG_FILE, [])
save_scan_log  = lambda d: _wj(SCAN_LOG_FILE, d)

# ─────────────────────────────────────────────────────────────
#  TELEGRAM
# ─────────────────────────────────────────────────────────────
def send_telegram(token: str, chat_id: str, text: str):
    if not token or not chat_id:
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    for i in range(0, len(text), 4096):
        chunk = text[i:i+4096].strip()
        try:
            requests.post(url, json={"chat_id": chat_id, "text": chunk,
                                     "parse_mode": "Markdown"}, timeout=10)
        except Exception as e:
            print(f"[Telegram] {e}")
        time.sleep(0.3)

# ─────────────────────────────────────────────────────────────
#  TICKERS
# ─────────────────────────────────────────────────────────────
def load_tickers() -> List[str]:
    try:
        if os.path.exists(TICKERS_FILE):
            df = pd.read_csv(TICKERS_FILE)
            if "SYMBOL" in df.columns:
                return [str(s).strip().upper() for s in df["SYMBOL"].dropna()]
    except Exception:
        pass
    return ["HDFCBANK", "RELIANCE", "TCS", "INFY", "ICICIBANK"]

# ─────────────────────────────────────────────────────────────
#  UTILITY
# ─────────────────────────────────────────────────────────────
def is_market_open() -> bool:
    now = datetime.now(IST)
    if now.weekday() >= 5:
        return False
    o = now.replace(hour=9,  minute=15, second=0, microsecond=0)
    c = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return o <= now <= c

def now_ist_str() -> str:
    return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")

# ─────────────────────────────────────────────────────────────
#  NSE SESSION
# ─────────────────────────────────────────────────────────────
def get_nse_session() -> Optional[requests.Session]:
    global _nse_session
    with _nse_lock:
        if _nse_session is not None:
            return _nse_session
        s = requests.Session()
        s.headers.update(NSE_HEADERS)
        try:
            s.get("https://www.nseindia.com/", timeout=12)
            time.sleep(random.uniform(1.5, 2.5))
            s.get("https://www.nseindia.com/option-chain", timeout=12)
            time.sleep(random.uniform(2.0, 3.0))
            _nse_session = s
            return s
        except Exception as e:
            print(f"[NSE] Session init failed: {e}")
            return None

def reset_nse_session():
    global _nse_session
    with _nse_lock:
        _nse_session = None

# ─────────────────────────────────────────────────────────────
#  NSE OPTION CHAIN
# ─────────────────────────────────────────────────────────────
def fetch_option_chain(ticker: str) -> Tuple[Optional[Dict], Optional[str]]:
    session = get_nse_session()
    if session is None:
        return None, None
    try:
        r = session.get(
            f"https://www.nseindia.com/api/option-chain-contract-info?symbol={ticker}",
            timeout=12,
        )
        if r.status_code in (429, 403):
            reset_nse_session()
            return None, None
        if r.status_code != 200 or not r.text.strip().startswith("{"):
            return None, None
        d = r.json()
        expiries = d.get("expiryDates", []) or d.get("records", {}).get("expiryDates", [])
        if not expiries:
            return None, None
        expiry = expiries[0]
    except Exception as e:
        print(f"[NSE] Expiry error {ticker}: {e}")
        return None, None

    time.sleep(random.uniform(1.0, 2.0))

    try:
        r2 = session.get(
            f"https://www.nseindia.com/api/option-chain-v3"
            f"?type=Equity&symbol={ticker}&expiry={expiry}",
            timeout=15,
        )
        if r2.status_code in (429, 403):
            reset_nse_session()
            return None, None
        if r2.status_code != 200 or not r2.text.strip().startswith("{"):
            return None, None
        records = r2.json().get("filtered", {}).get("data", [])
        strike_map: Dict[float, Dict] = {}
        for rec in records:
            sp  = float(rec.get("strikePrice", 0))
            ce  = rec.get("CE", {})
            ltp = ce.get("lastPrice", 0.0)
            oi  = ce.get("openInterest", 0)
            if sp > 0:
                strike_map[sp] = {"CE_ltp": float(ltp), "CE_oi": int(oi)}
        return strike_map, expiry
    except Exception as e:
        print(f"[NSE] Option chain error {ticker}: {e}")
        return None, None

# ─────────────────────────────────────────────────────────────
#  YFINANCE HELPERS
# ─────────────────────────────────────────────────────────────
def get_price_history(ticker: str, days: int) -> Optional[pd.DataFrame]:
    try:
        now_ist   = datetime.now(IST)
        end_dt    = (now_ist + timedelta(days=1)).date()
        start_dt  = (now_ist - timedelta(days=days + 10)).date()
        hist = yf.Ticker(f"{ticker}.NS").history(
            start=str(start_dt), end=str(end_dt), auto_adjust=True
        )
        if hist.empty:
            return None
        if hist.index.tz is None:
            hist.index = hist.index.tz_localize("UTC")
        hist.index = hist.index.tz_convert(IST)
        hist = hist.sort_index()
        today_ist = now_ist.date()
        hist = hist[hist.index.date <= today_ist]
        return hist
    except Exception as e:
        print(f"[yf] History error {ticker}: {e}")
        return None

def get_current_price(ticker: str) -> Optional[float]:
    try:
        p = yf.Ticker(f"{ticker}.NS").fast_info.last_price
        return float(p) if p else None
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────
#  CE HISTORICAL HIGH
# ─────────────────────────────────────────────────────────────
def get_ce_historical_high(ticker: str, strike: float, expiry_str: str,
                            days: int = 30) -> Tuple[Optional[float], int]:
    try:
        session = get_nse_session()
        if session is None:
            print(f"[{ticker}] ✗ NSE session is None")
            return None, 0

        expiry_dt = datetime.strptime(expiry_str, "%d-%b-%Y")
        exp_nse   = expiry_dt.strftime("%d-%b-%Y")
        year      = expiry_dt.year
        end_dt    = date.today()
        start_dt  = end_dt - timedelta(days=days + 5)

        url = (
            f"https://www.nseindia.com/api/historicalOR/foCPV"
            f"?from={start_dt.strftime('%d-%m-%Y')}"
            f"&to={end_dt.strftime('%d-%m-%Y')}"
            f"&instrumentType=OPTSTK"
            f"&symbol={ticker}"
            f"&year={year}"
            f"&expiryDate={exp_nse}"
            f"&optionType=CE"
            f"&strikePrice={int(strike)}"
        )
        print(f"\n{'='*60}")
        print(f"[{ticker}] NSE CE HIST URL:\n  {url}")
        print(f"  expiry_str received : {expiry_str!r}")
        print(f"  exp_nse formatted   : {exp_nse!r}")
        print(f"  year                : {year}")
        print(f"  strike (int)        : {int(strike)}")
        print(f"  date range          : {start_dt} → {end_dt}")

        time.sleep(random.uniform(1.0, 2.0))
        r = session.get(url, timeout=15)

        print(f"[{ticker}] HTTP STATUS : {r.status_code}")
        print(f"[{ticker}] RESPONSE (first 500 chars):\n  {r.text[:500]!r}")

        if r.status_code in (403, 429):
            print(f"[{ticker}] ✗ Blocked ({r.status_code}) — resetting session")
            reset_nse_session()
            return None, 1

        if r.status_code != 200:
            print(f"[{ticker}] ✗ Non-200 status: {r.status_code}")
            return None, 1

        if not r.text.strip().startswith("{"):
            print(f"[{ticker}] ✗ Response is NOT JSON — got HTML/redirect?")
            print(f"  Full response: {r.text[:1000]!r}")
            return None, 1

        data = r.json()
        print(f"[{ticker}] TOP-LEVEL KEYS : {list(data.keys())}")

        # ── NSE sometimes nests under "data", sometimes under other keys
        rows = (
            data.get("data")
            or data.get("Data")
            or data.get("records", {}).get("data")
            or []
        )
        print(f"[{ticker}] ROWS COUNT : {len(rows)}")

        if rows:
            print(f"[{ticker}] FIRST ROW KEYS : {list(rows[0].keys())}")
            print(f"[{ticker}] FIRST ROW DATA : {rows[0]}")

        if not rows:
            print(f"[{ticker}] ✗ Empty rows — API returned no historical data")
            return None, 1

        ltps = []
        for row in rows:
            # Try every possible field NSE uses across API versions
            val = (
                row.get("FH_CLOSING_PRICE")
                or row.get("FH_LAST_TRADED_PRICE")
                or row.get("close")
                or row.get("lastPrice")
                or row.get("FH_TRADE_HIGH_PRICE")   # ← sometimes this
                or row.get("CH_CLOSING_PRICE")       # ← equity variant
                or 0
            )
            try:
                v = float(str(val).replace(",", ""))  # handle "1,234.56" format
                if v > 0:
                    ltps.append(v)
            except (ValueError, TypeError):
                continue

        print(f"[{ticker}] EXTRACTED LTPs : {ltps[:10]}{'...' if len(ltps)>10 else ''}")

        if not ltps:
            print(f"[{ticker}] ✗ No valid LTP values — check field names above")
            return None, 1

        high = max(ltps)
        print(f"[{ticker}] ✓ CE Configured-30d HIGH = ₹{high}")
        print(f"{'='*60}\n")
        return high, 1

    except Exception as e:
        import traceback
        print(f"[{ticker}] ✗ EXCEPTION in get_ce_historical_high:")
        traceback.print_exc()
        return None, 0

# ─────────────────────────────────────────────────────────────
#  STRIKE SELECTION
# ─────────────────────────────────────────────────────────────
def nearest_round_strike_above(price: float, strike_map: Dict) -> Optional[float]:
    liquid_above = {
        s: data for s, data in strike_map.items()
        if s > price and data["CE_ltp"] > 0
    }
    if not liquid_above:
        above = [s for s in strike_map if s > price]
        return min(above) if above else None
    sorted_strikes = sorted(liquid_above.keys(), key=lambda s: s - price)
    top3 = sorted_strikes[:3]
    round_in_top3 = [s for s in top3 if int(s) % 50 == 0]
    if round_in_top3:
        return min(round_in_top3, key=lambda s: s - price)
    return sorted_strikes[0]

# ─────────────────────────────────────────────────────────────
#  SCREENING FUNCTIONS
# ─────────────────────────────────────────────────────────────
def check_surge_and_loss(ticker: str, cfg: Dict, job_id: str = "") -> Optional[Dict]:
    hist = get_price_history(ticker, cfg["lookback_days"])
    if hist is None or len(hist) < cfg["min_green_candles"] + 3:
        return None

    hist   = hist.sort_index()
    closes = hist["Close"].values.copy()
    highs  = hist["High"].values
    lows   = hist["Low"].values
    opens  = hist["Open"].values
    dates  = hist.index

    if len(closes) < 2:
        return None

    live_price = get_current_price(ticker)
    if live_price is not None:
        closes[-1] = live_price

    today_close     = float(closes[-1])
    yesterday_low   = float(lows[-2])
    yesterday_close = float(closes[-2])

    if today_close >= yesterday_low:
        if job_id:
            job_log(job_id, f"{ticker} ✗ today={today_close:.2f} >= prev_low={yesterday_low:.2f}", "fail")
        return None

    drop_pct = ((yesterday_low - today_close) / yesterday_low) * 100

    # ── Compute N-day high (over completed candles only) ──────────────────────
    # "sell_zone_lookback_days" defaults to 10 if not set in config
    sell_zone_days = max(int(cfg.get("sell_zone_lookback_days", 10)), 1)

    # Only exclude today's row if the last candle IS from today (live/partial session).
    # If the market is closed, the last candle is a fully completed prior session
    # and must be included — this was the source of the Apr 17 high being dropped.
    today_date       = datetime.now(IST).date()
    last_candle_date = hist.index[-1].date()
    if last_candle_date == today_date:
        completed_hist = hist.iloc[:-1]   # strip live partial candle
    else:
        completed_hist = hist             # all rows are completed sessions

    recent_window = completed_hist.tail(sell_zone_days)
    ten_day_high  = float(recent_window["High"].max()) if not recent_window.empty else yesterday_low

    if job_id:
        job_log(job_id,
            f"{ticker} → {sell_zone_days}d High = ₹{ten_day_high:.2f}  "
            f"(from {len(recent_window)} candles, last candle: {last_candle_date})",
            "info")

    min_gain  = cfg["min_gain_percent"]
    min_green = cfg["min_green_candles"]
    scan_closes = closes[:-1]
    scan_opens  = opens[:-1]
    scan_dates  = dates[:-1]
    n           = len(scan_closes)
    window_size = max(min_green + 2, 5)
    best_gain   = 0.0
    best_greens = 0
    best_window = None

    for wsize in range(window_size, n + 1):
        for start in range(0, n - wsize + 1):
            end = start + wsize - 1
            net_gain = ((scan_closes[end] - scan_closes[start]) / scan_closes[start]) * 100
            green_count = sum(1 for i in range(start, end + 1) if scan_closes[i] > scan_opens[i])
            if net_gain >= min_gain and green_count >= min_green:
                if net_gain > best_gain:
                    best_gain   = net_gain
                    best_greens = green_count
                    best_window = (start, end)

    if best_gain < min_gain or best_greens < min_green:
        if job_id:
            job_log(job_id, f"{ticker} ✗ no surge ≥{min_gain}% (best={best_gain:.1f}%)", "fail")
        return None

    surge_start_idx, surge_end_idx = best_window
    if job_id:
        job_log(job_id,
            f"{ticker} ✓ surge={best_gain:.1f}% drop={drop_pct:.1f}% — passing to Step 2",
            "pass")

    return {
        "ticker":           ticker,
        "today_close":      round(today_close, 2),
        "yesterday_close":  round(yesterday_close, 2),
        "yesterday_low":    round(yesterday_low, 2),
        "drop_pct":         round(drop_pct, 2),
        "surge_gain_pct":   round(best_gain, 2),
        "surge_candles":    best_greens,
        "surge_start_date": scan_dates[surge_start_idx].strftime("%Y-%m-%d"),
        "surge_end_date":   scan_dates[surge_end_idx].strftime("%Y-%m-%d"),
        "ten_day_high":     round(ten_day_high, 2),   # ← NEW
        "sell_zone_days":   sell_zone_days,            # ← NEW: record which window was used
    }


def find_best_strike(candidate: Dict, cfg: Dict, job_id: str = "") -> Optional[Dict]:
    ticker      = candidate["ticker"]
    today_close = candidate["today_close"]

    if job_id:
        job_log(job_id, f"{ticker} → fetching option chain…", "info")

    strike_map, expiry = fetch_option_chain(ticker)
    if strike_map is None or not strike_map:
        if job_id:
            job_log(job_id, f"{ticker} ✗ option chain empty", "fail")
        return None

    strike = nearest_round_strike_above(today_close, strike_map)
    if strike is None:
        if job_id:
            job_log(job_id, f"{ticker} ✗ no liquid strike above ₹{today_close}", "fail")
        return None

    ce_data = strike_map[strike]
    ce_ltp  = ce_data["CE_ltp"]
    ce_oi   = ce_data["CE_oi"]

    if job_id:
        job_log(job_id,
            f"{ticker} → selected strike ₹{strike:.0f} CE | LTP ₹{ce_ltp:.2f} | OI {ce_oi:,}",
            "info")

    if ce_ltp <= 0:
        if job_id:
            job_log(job_id, f"{ticker} ✗ Strike ₹{strike:.0f} CE LTP=0 (illiquid)", "fail")
        return None

    ce_hist_high        = None
    ce_above_30d_status = "Unverified"
    nse_hist_calls      = 0
    remark              = "⚪ 30d High Unverified"

    if cfg.get("ce_above_historical_high", True):
        ce_hist_high, nse_hist_calls = get_ce_historical_high(
            ticker, strike, expiry, cfg.get("ce_history_days", 30)
        )
        if ce_hist_high is not None:
            if ce_ltp <= ce_hist_high:
                ce_above_30d_status = False
                remark = "🔴 CE Below 30d High"
                if job_id:
                    job_log(job_id,
                        f"{ticker} ⚠ CE LTP ₹{ce_ltp} ≤ 30d high ₹{ce_hist_high} — "
                        f"included with remark (special case)",
                        "warn")
            else:
                ce_above_30d_status = True
                remark = "🟢 CE at New High"
                if job_id:
                    job_log(job_id,
                        f"{ticker} ✓ CE LTP ₹{ce_ltp} > 30d high ₹{ce_hist_high} — strong signal",
                        "signal")
        else:
            ce_above_30d_status = "Unverified"
            remark = "⚪ 30d High Unverified"
    else:
        remark = "—  (check disabled)"

    ten_day_high  = candidate["ten_day_high"]
    sell_zone_days = candidate["sell_zone_days"]

    if job_id:
        job_log(job_id,
            f"{ticker} ✓ SIGNAL — Strike ₹{strike:.0f} CE  LTP ₹{ce_ltp:.2f}  "
            f"[{remark}]  {sell_zone_days}d High ₹{ten_day_high:.2f}",
            "signal")

    return {
        "Ticker":            ticker,
        "Today_Close":       candidate["today_close"],
        "Yesterday_Close":   candidate["yesterday_close"],
        "Yesterday_Low":     candidate["yesterday_low"],
        "Drop_Pct":          candidate["drop_pct"],
        "Surge_Gain_Pct":    candidate["surge_gain_pct"],
        "Surge_Candles":     candidate["surge_candles"],
        "Surge_Start":       candidate["surge_start_date"],
        "Surge_End":         candidate["surge_end_date"],
        "Ten_Day_High":      ten_day_high,               # ← NEW
        "Sell_Zone_Days":    sell_zone_days,             # ← NEW
        "Suggested_Strike":  strike,
        "Expiry":            expiry,
        "CE_LTP":            round(ce_ltp, 2),
        "CE_OI":             ce_oi,
        "CE_30d_High":       round(ce_hist_high, 2) if ce_hist_high else "N/A",
        "CE_Above_30d_High": ce_above_30d_status,
        "Remark":            remark,
        "NSE_Hist_Calls":    nse_hist_calls,
        "Status":            "Momentum Lost",
        "Sell_Alert_Sent":   False,
        "Scanned_At":        now_ist_str(),
    }


# ─────────────────────────────────────────────────────────────
#  PROXIMITY / SELL ZONE ALERTS
#
#  Logic: fire when current price is within X% BELOW the
#  N-day high (resistance ceiling).  The price must be
#  approaching from below — i.e. cur_price <= ten_day_high.
#
#  sell_zone_floor = ten_day_high * (1 - proximity_pct / 100)
#
#  Alert fires when:
#      sell_zone_floor <= cur_price <= ten_day_high
# ─────────────────────────────────────────────────────────────
def check_proximity_alerts(signals: List[Dict], cfg: Dict) -> List[Dict]:
    proximity_pct  = cfg["price_proximity_percent"]
    sell_zone_hits = []

    for sig in signals:
        ticker       = sig["Ticker"]
        ten_day_high = sig.get("Ten_Day_High")

        # ── Graceful fallback for old signals that pre-date this field ────────
        if not ten_day_high:
            # Old signal: fall back to Suggested_Strike so we don't silently skip
            ten_day_high = sig.get("Suggested_Strike")
            if not ten_day_high:
                time.sleep(random.uniform(0.2, 0.5))
                continue

        cur_price = get_current_price(ticker)
        if cur_price is None:
            time.sleep(random.uniform(0.2, 0.5))
            continue

        # Price is above the N-day high → not a sell-zone approach, skip
        if cur_price > ten_day_high:
            time.sleep(random.uniform(0.2, 0.5))
            continue

        # Distance of current price from the N-day high ceiling (downward %)
        dist_pct = (ten_day_high - cur_price) / ten_day_high * 100

        if dist_pct <= proximity_pct:
            sell_zone_days = sig.get("Sell_Zone_Days", 10)
            hit = {
                **sig,
                "Current_Price":          round(cur_price, 2),
                "Ten_Day_High":           round(ten_day_high, 2),
                "Distance_From_High_Pct": round(dist_pct, 2),
                "Alert_Time":             now_ist_str(),
            }
            sell_zone_hits.append(hit)
            send_telegram(
                cfg["telegram_bot_token"],
                cfg["telegram_chat_id"],
                f"*🔔 SELL ZONE — {ticker}*\n"
                f"Price ₹{cur_price:.2f}  "
                f"{sell_zone_days}d High ₹{ten_day_high:.2f}  "
                f"Dist {dist_pct:.2f}%  (within {proximity_pct}%)"
            )

        time.sleep(random.uniform(0.2, 0.5))

    return sell_zone_hits


def run_screen_job(tickers: List[str], cfg: Dict, job_id: str):
    """Runs in a background thread."""
    signals = []
    total   = len(tickers)
    nse_calls = 0

    job_log(job_id, f"Starting scan of {total} tickers…", "info")

    for idx, ticker in enumerate(tickers):
        job_progress(job_id, idx, total, ticker)
        candidate = check_surge_and_loss(ticker, cfg, job_id)
        if candidate is None:
            time.sleep(random.uniform(0.1, 0.3))
            continue

        time.sleep(random.uniform(3.0, 5.0))
        signal = find_best_strike(candidate, cfg, job_id)
        nse_calls += 2
        if signal is None:
            continue

        signals.append(signal)
        time.sleep(random.uniform(2.0, 4.0))

    # Save + merge
    existing = load_signals()
    seen = {s["Ticker"] for s in existing}
    added = 0
    for s in signals:
        if s["Ticker"] not in seen:
            existing.append(s)
            seen.add(s["Ticker"])
            added += 1
    save_signals(existing)

    log = load_scan_log()
    log.append({
        "time":            now_ist_str(),
        "tickers_scanned": total,
        "signals_found":   len(signals),
        "nse_calls":       nse_calls,
    })
    save_scan_log(log[-100:])

    job_log(job_id,
        f"Scan complete — {len(signals)} signal(s) found, {added} new added.", "info")
    job_progress(job_id, total, total, "")
    job_done(job_id, signals)

# ─────────────────────────────────────────────────────────────
#  SCHEDULER
# ─────────────────────────────────────────────────────────────
_scheduler: Optional[BackgroundScheduler] = None

def start_scheduler(cfg: Dict):
    global _scheduler
    if _scheduler and _scheduler.running:
        try:
            _scheduler.shutdown(wait=False)
        except Exception:
            pass
    _scheduler = BackgroundScheduler(timezone=IST)

    def job():
        c = load_config()
        if c["market_hours_only"] and not is_market_open():
            return
        jid = create_job()
        tickers = load_tickers()
        run_screen_job(tickers, c, jid)
        all_signals = load_signals()
        if all_signals:
            hits = check_proximity_alerts(all_signals, c)
            if hits:
                prox = load_proximity()
                prox.extend(hits)
                save_proximity(prox[-200:])

    interval = max(cfg.get("auto_scan_interval_min", 15), 10)
    _scheduler.add_job(job, IntervalTrigger(minutes=interval),
                       id="main_scan", replace_existing=True)
    _scheduler.start()

# ─────────────────────────────────────────────────────────────
#  FASTAPI APP
# ─────────────────────────────────────────────────────────────
app = FastAPI(title="NSE Tools Suite", version="3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Page routes — MUST be before app.mount() ─────────────────
@app.get("/option-charts", include_in_schema=False)
async def option_charts_page():
    return FileResponse(BASE_DIR / "static" / "option-charts.html")

@app.get("/sma-screener", include_in_schema=False)
async def sma_screener_page():
    return FileResponse(BASE_DIR / "static" / "sma-screener.html")

# ── Include sub-routers (APIs) ────────────────────────────────
app.include_router(oc_router)
app.include_router(sma_router)

# Mount static files (comes AFTER routes)
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Pydantic schemas ──────────────────────────────────────────
class ConfigUpdate(BaseModel):
    lookback_days:             Optional[int]   = None
    min_gain_percent:          Optional[float] = None
    min_green_candles:         Optional[int]   = None
    price_proximity_percent:   Optional[float] = None
    sell_zone_lookback_days:   Optional[int]   = None   # ← NEW
    ce_above_historical_high:  Optional[bool]  = None
    ce_history_days:           Optional[int]   = None
    auto_scan_enabled:         Optional[bool]  = None
    auto_scan_interval_min:    Optional[int]   = None
    market_hours_only:         Optional[bool]  = None
    telegram_bot_token:        Optional[str]   = None
    telegram_chat_id:          Optional[str]   = None

class SpecificScreenRequest(BaseModel):
    tickers: List[str]

# ─────────────────────────────────────────────────────────────
#  ROUTES — STATUS & CONFIG
# ─────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return FileResponse(BASE_DIR / "static" / "index.html")

@app.get("/api/status")
async def get_status():
    cfg = load_config()
    sigs = load_signals()
    prox = load_proximity()
    log  = load_scan_log()
    return {
        "market_open":      is_market_open(),
        "ist_time":         datetime.now(IST).strftime("%H:%M:%S"),
        "ist_date":         datetime.now(IST).strftime("%Y-%m-%d"),
        "active_signals":   len(sigs),
        "proximity_alerts": len(prox),
        "auto_scan":        cfg["auto_scan_enabled"],
        "scan_interval":    cfg["auto_scan_interval_min"],
        "last_scan":        log[-1]["time"] if log else None,
    }

@app.get("/api/config")
async def get_config():
    return load_config()

@app.put("/api/config")
async def update_config(data: ConfigUpdate):
    cfg = load_config()
    updates = {k: v for k, v in data.dict().items() if v is not None}
    cfg.update(updates)
    save_config(cfg)
    if cfg["auto_scan_enabled"]:
        start_scheduler(cfg)
    return cfg

# ─────────────────────────────────────────────────────────────
#  ROUTES — TICKERS
# ─────────────────────────────────────────────────────────────
@app.get("/api/tickers")
async def get_tickers():
    return {"tickers": load_tickers()}

@app.post("/api/tickers/upload")
async def upload_tickers(file: UploadFile = File(...)):
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))
    if "SYMBOL" not in df.columns:
        raise HTTPException(400, "CSV must have a SYMBOL column")
    df.to_csv(TICKERS_FILE, index=False)
    tickers = [str(s).strip().upper() for s in df["SYMBOL"].dropna()]
    return {"count": len(tickers), "tickers": tickers[:20]}

# ─────────────────────────────────────────────────────────────
#  ROUTES — SIGNALS
# ─────────────────────────────────────────────────────────────
@app.get("/api/signals")
async def get_signals():
    return load_signals()

@app.delete("/api/signals")
async def clear_signals():
    _wj(SIGNALS_FILE, [])
    _wj(PROXIMITY_FILE, [])
    return {"ok": True}

@app.delete("/api/signals/{ticker}")
async def remove_signal(ticker: str):
    sigs = [s for s in load_signals() if s["Ticker"] != ticker.upper()]
    save_signals(sigs)
    return {"ok": True, "remaining": len(sigs)}

# ─────────────────────────────────────────────────────────────
#  ROUTES — PROXIMITY ALERTS
# ─────────────────────────────────────────────────────────────
@app.get("/api/proximity")
async def get_proximity():
    return load_proximity()

@app.post("/api/proximity/check")
async def check_proximity():
    sigs = load_signals()
    if not sigs:
        return {"hits": 0, "alerts": []}
    cfg  = load_config()
    hits = await asyncio.to_thread(check_proximity_alerts, sigs, cfg)
    if hits:
        prox = load_proximity()
        prox.extend(hits)
        save_proximity(prox[-200:])
    return {"hits": len(hits), "alerts": hits}

# ─────────────────────────────────────────────────────────────
#  ROUTES — SCAN LOG
# ─────────────────────────────────────────────────────────────
@app.get("/api/scan-log")
async def get_scan_log():
    return list(reversed(load_scan_log()))

# ─────────────────────────────────────────────────────────────
#  ROUTES — SCREENING (SSE streaming)
# ─────────────────────────────────────────────────────────────
@app.post("/api/screen/all")
async def screen_all(background_tasks: BackgroundTasks):
    cfg     = load_config()
    tickers = load_tickers()
    job_id  = create_job()
    background_tasks.add_task(
        asyncio.to_thread, run_screen_job, tickers, cfg, job_id
    )
    return {"job_id": job_id, "total": len(tickers)}

@app.post("/api/screen/specific")
async def screen_specific(req: SpecificScreenRequest, background_tasks: BackgroundTasks):
    cfg    = load_config()
    tickers = [t.strip().upper() for t in req.tickers if t.strip()]
    job_id = create_job()
    background_tasks.add_task(
        asyncio.to_thread, run_screen_job, tickers, cfg, job_id
    )
    return {"job_id": job_id, "total": len(tickers)}

@app.get("/api/jobs/{job_id}/stream")
async def job_stream(job_id: str):
    async def event_gen():
        sent_log_count = 0
        while True:
            with jobs_lock:
                job = jobs.get(job_id)

            if job is None:
                yield f"data: {json.dumps({'type':'error','msg':'Job not found'})}\n\n"
                break

            # Stream new log lines
            logs = job.get("logs", [])
            for entry in logs[sent_log_count:]:
                yield f"data: {json.dumps({'type':'log','time':entry['time'],'msg':entry['msg'],'level':entry['level']})}\n\n"
            sent_log_count = len(logs)

            # Progress update
            yield f"data: {json.dumps({'type':'progress','current':job['progress'],'total':job['total'],'ticker':job.get('current_ticker','')})}\n\n"

            if job["status"] in ("done", "error"):
                yield f"data: {json.dumps({'type':'done','status':job['status'],'count':len(job.get('result',[]))})}\n\n"
                break

            await asyncio.sleep(0.6)

    return StreamingResponse(event_gen(), media_type="text/event-stream",
                              headers={"Cache-Control": "no-cache",
                                       "X-Accel-Buffering": "no"})

@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job

# ─────────────────────────────────────────────────────────────
#  ROUTES — CHART DATA
# ─────────────────────────────────────────────────────────────
@app.get("/api/chart/{ticker}")
async def get_chart_data(ticker: str, days: int = 60):
    hist = await asyncio.to_thread(get_price_history, ticker.upper(), days)
    if hist is None or hist.empty:
        raise HTTPException(404, f"No data for {ticker}")
    signals = load_signals()
    strike      = None
    ten_day_high = None
    sig = next((s for s in signals if s["Ticker"] == ticker.upper()), None)
    if sig:
        strike       = sig.get("Suggested_Strike")
        ten_day_high = sig.get("Ten_Day_High")
    data = []
    for idx, row in hist.iterrows():
        data.append({
            "date":  idx.strftime("%Y-%m-%d"),
            "open":  round(float(row["Open"]),  2),
            "high":  round(float(row["High"]),  2),
            "low":   round(float(row["Low"]),   2),
            "close": round(float(row["Close"]), 2),
            "volume":int(row.get("Volume", 0)),
        })
    return {
        "ticker":       ticker.upper(),
        "strike":       strike,
        "ten_day_high": ten_day_high,   # ← NEW: chart can now draw the sell zone ceiling
        "candles":      data,
    }

# ─────────────────────────────────────────────────────────────
#  STARTUP
# ─────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    cfg = load_config()
    if cfg.get("auto_scan_enabled"):
        start_scheduler(cfg)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)