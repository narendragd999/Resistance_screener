"""
NSE Momentum Loss Screener — FastAPI Backend
============================================
Run: uvicorn main:app --host 0.0.0.0 --port 8000 --reload

THESIS
------
1. Stock surges >= N% continuously (green candles with higher highs;
   red candles allowed only if they do NOT close below the previous
   candle's low — i.e. the higher-low structure stays intact).
2. Surge must have ended within `surge_recency_days` trading sessions
   of today — ensures the breakdown happens right after the surge,
   while market memory of those highs is still fresh.
3. TODAY's candle closes BELOW the previous day's LOW with volume
   >= `min_breakdown_volume_ratio` x 20-day average  -> momentum loss.
4. Today's close must be >= `min_drop_percent` below previous low
   (filters trivial tick-below-low noise like -0.18%).
5. The close must be below the 9-EMA at the time of initial breakdown
   detection — a ONE-TIME gate confirming trend weakening. Once this
   signal is created (EMA breach confirmed), any subsequent day where
   price rallies back into proximity of the surge high is valid without
   re-checking the EMA — the trend damage is already established.
6. The SURGE HIGH (peak of the entire surge window) is the true
   resistance ceiling — this is where original momentum exhausted.
   Even after breakdown + partial recovery, this level rejects price
   again. The CE strike is anchored above this level.
7. When price rallies back within `price_proximity_percent` of the
   surge high -> CE premium is elevated -> SELL CE above that
   resistance. Theta decay + failed retest = profit.
"""

import os, json, time, random, asyncio, threading, uuid, io
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
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

# Sub-routers (Option Charts + SMA Screener)
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

_nse_session:         Optional[requests.Session] = None
_nse_session_created: Optional[datetime]         = None
_nse_lock = threading.Lock()
NSE_SESSION_TTL_SECS = 25 * 60   # proactive refresh after 25 min

# ─────────────────────────────────────────────────────────────
#  JOB MANAGEMENT
# ─────────────────────────────────────────────────────────────
jobs: Dict[str, Dict] = {}
jobs_lock = threading.Lock()

def _prune_old_jobs():
    """Drop completed jobs older than 2 hours to prevent memory growth."""
    cutoff = datetime.now(IST) - timedelta(hours=2)
    with jobs_lock:
        stale = [
            jid for jid, j in jobs.items()
            if j.get("status") in ("done", "error") and j.get("created_at")
            and datetime.strptime(j["created_at"], "%Y-%m-%d %H:%M:%S")
               .replace(tzinfo=IST) < cutoff
        ]
        for jid in stale:
            del jobs[jid]

def create_job() -> str:
    _prune_old_jobs()
    job_id = str(uuid.uuid4())
    with jobs_lock:
        jobs[job_id] = {
            "status":         "running",
            "progress":       0,
            "total":          0,
            "current_ticker": "",
            "logs":           [],
            "result":         [],
            "created_at":     now_ist_str(),
        }
    return job_id

def job_log(job_id: str, msg: str, level: str = "info"):
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id]["logs"].append({
                "time":  datetime.now(IST).strftime("%H:%M:%S"),
                "msg":   msg,
                "level": level,
            })

def job_progress(job_id: str, current: int, total: int, ticker: str):
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id]["progress"]       = current
            jobs[job_id]["total"]          = total
            jobs[job_id]["current_ticker"] = ticker

def job_done(job_id: str, result: list, status: str = "done"):
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id]["status"] = status
            jobs[job_id]["result"] = result

# ─────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    # Surge detection
    "lookback_days":               30,
    "min_gain_percent":            20.0,
    "min_green_candles":           3,
    "surge_recency_days":          5,      # surge_end must be within N trading days of today
    # Breakdown (momentum loss) candle
    "min_drop_percent":            0.5,    # must close this % below prev low (not just a tick)
    "min_breakdown_volume_ratio":  1.2,    # breakdown volume / 20d avg volume
    # Trend filter
    "ema_period":                  9,
    "ema_filter_enabled":          True,   # breakdown close must be below EMA
    # Sell zone
    "price_proximity_percent":     2.0,    # within X% below yesterday_high = sell zone
    # CE option filter
    "ce_above_historical_high":    True,
    "ce_history_days":             30,
    # Signal freshness
    "max_signal_age_days":         5,      # auto-prune signals older than N days
    # Auto scan
    "auto_scan_enabled":           True,
    "auto_scan_interval_min":      15,
    "market_hours_only":           True,
    # Telegram
    "telegram_bot_token":          "",
    "telegram_chat_id":            "",
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

def compute_ema(values: np.ndarray, period: int) -> np.ndarray:
    """EMA via pandas ewm — matches standard charting tools."""
    return pd.Series(values, dtype=float).ewm(span=period, adjust=False).mean().values

# ─────────────────────────────────────────────────────────────
#  NSE SESSION  (TTL-based proactive refresh)
# ─────────────────────────────────────────────────────────────
def get_nse_session() -> Optional[requests.Session]:
    global _nse_session, _nse_session_created
    with _nse_lock:
        if _nse_session is not None and _nse_session_created is not None:
            age = (datetime.now(IST) - _nse_session_created).total_seconds()
            if age > NSE_SESSION_TTL_SECS:
                print(f"[NSE] Session TTL exceeded ({age:.0f}s) — refreshing")
                _nse_session         = None
                _nse_session_created = None

        if _nse_session is not None:
            return _nse_session

        s = requests.Session()
        s.headers.update(NSE_HEADERS)
        try:
            s.get("https://www.nseindia.com/", timeout=12)
            time.sleep(random.uniform(1.5, 2.5))
            s.get("https://www.nseindia.com/option-chain", timeout=12)
            time.sleep(random.uniform(2.0, 3.0))
            _nse_session         = s
            _nse_session_created = datetime.now(IST)
            return s
        except Exception as e:
            print(f"[NSE] Session init failed: {e}")
            return None

def reset_nse_session():
    global _nse_session, _nse_session_created
    with _nse_lock:
        _nse_session         = None
        _nse_session_created = None

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

        # Skip nearest expiry if within 3 calendar days (high gamma risk)
        today_dt      = date.today()
        chosen_expiry = expiries[0]
        try:
            nearest_dt = datetime.strptime(expiries[0], "%d-%b-%Y").date()
            if (nearest_dt - today_dt).days <= 3 and len(expiries) > 1:
                chosen_expiry = expiries[1]
                print(f"[NSE] {ticker}: nearest expiry {expiries[0]} too close "
                      f"({(nearest_dt - today_dt).days}d) -> using {chosen_expiry}")
        except Exception:
            pass

        expiry = chosen_expiry
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
        now_ist  = datetime.now(IST)
        end_dt   = (now_ist + timedelta(days=1)).date()
        start_dt = (now_ist - timedelta(days=days + 15)).date()
        hist = yf.Ticker(f"{ticker}.NS").history(
            start=str(start_dt), end=str(end_dt), auto_adjust=True
        )
        if hist.empty:
            return None
        if hist.index.tz is None:
            hist.index = hist.index.tz_localize("UTC")
        hist.index = hist.index.tz_convert(IST)
        hist = hist.sort_index()
        hist = hist[hist.index.date <= now_ist.date()]
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
            print(f"[{ticker}] NSE session is None")
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
        print(f"  date range          : {start_dt} -> {end_dt}")

        time.sleep(random.uniform(1.0, 2.0))
        r = session.get(url, timeout=15)

        print(f"[{ticker}] HTTP STATUS : {r.status_code}")
        print(f"[{ticker}] RESPONSE (first 500 chars):\n  {r.text[:500]!r}")

        if r.status_code in (403, 429):
            print(f"[{ticker}] Blocked ({r.status_code}) -- resetting session")
            reset_nse_session()
            return None, 1
        if r.status_code != 200:
            print(f"[{ticker}] Non-200 status: {r.status_code}")
            return None, 1
        if not r.text.strip().startswith("{"):
            print(f"[{ticker}] Response is NOT JSON")
            return None, 1

        data = r.json()
        print(f"[{ticker}] TOP-LEVEL KEYS : {list(data.keys())}")

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
            print(f"[{ticker}] Empty rows -- API returned no historical data")
            return None, 1

        ltps = []
        for row in rows:
            val = (
                row.get("FH_CLOSING_PRICE")
                or row.get("FH_LAST_TRADED_PRICE")
                or row.get("close")
                or row.get("lastPrice")
                or row.get("FH_TRADE_HIGH_PRICE")
                or row.get("CH_CLOSING_PRICE")
                or 0
            )
            try:
                v = float(str(val).replace(",", ""))
                if v > 0:
                    ltps.append(v)
            except (ValueError, TypeError):
                continue

        print(f"[{ticker}] EXTRACTED LTPs : {ltps[:10]}{'...' if len(ltps)>10 else ''}")
        if not ltps:
            print(f"[{ticker}] No valid LTP values")
            return None, 1

        high = max(ltps)
        print(f"[{ticker}] CE {days}d HIGH = Rs{high}")
        print(f"{'='*60}\n")
        return high, 1

    except Exception as e:
        import traceback
        print(f"[{ticker}] EXCEPTION in get_ce_historical_high:")
        traceback.print_exc()
        return None, 0

# ─────────────────────────────────────────────────────────────
#  STRIKE SELECTION
#  Anchored to yesterday_high (resistance ceiling), not today_close.
#  We sell a CE that only goes ITM if price breaks above resistance.
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
#  SURGE CONTINUITY CHECK
#
#  A surge window [start, end] is "continuous" when every red
#  candle (close < open) inside it closes AT OR ABOVE the
#  previous candle's low.  This preserves the higher-low
#  structure while tolerating healthy pullback candles.
#
#  A red candle that closes BELOW the previous candle's low
#  is a real structural break and disqualifies the window.
# ─────────────────────────────────────────────────────────────
def check_surge_continuity(
    closes: np.ndarray,
    opens:  np.ndarray,
    lows:   np.ndarray,
    start:  int,
    end:    int,
) -> Tuple[bool, int, int]:
    """
    Returns (is_continuous, allowed_red_count, green_count).
    is_continuous=False means a red candle broke the prior low.
    """
    green_count       = 0
    allowed_red_count = 0

    for i in range(start, end + 1):
        is_green = closes[i] > opens[i]
        if is_green:
            green_count += 1
        else:
            # Red candle -- check higher-low structure
            if i > start and closes[i] < lows[i - 1]:
                # Closed below previous candle's low = structural break
                return False, allowed_red_count, green_count
            allowed_red_count += 1

    return True, allowed_red_count, green_count


# ─────────────────────────────────────────────────────────────
#  SCREENING -- STEP 1
# ─────────────────────────────────────────────────────────────
def check_surge_and_loss(ticker: str, cfg: Dict, job_id: str = "",
                          ema_already_breached: bool = False) -> Optional[Dict]:
    hist = get_price_history(ticker, cfg["lookback_days"])
    if hist is None or len(hist) < cfg["min_green_candles"] + 5:
        return None

    hist   = hist.sort_index()
    closes = hist["Close"].values.copy()
    highs  = hist["High"].values.copy()
    lows   = hist["Low"].values.copy()
    opens  = hist["Open"].values.copy()
    vols   = hist["Volume"].values.copy()
    dates  = hist.index

    if len(closes) < 3:
        return None

    # Inject live price into today's close (and high) during market hours
    live_price = get_current_price(ticker)
    if live_price is not None:
        closes[-1] = live_price
        if live_price > highs[-1]:
            highs[-1] = live_price

    # Key reference values
    # "yesterday" = the candle immediately before the breakdown candle
    today_close     = float(closes[-1])
    yesterday_high  = float(highs[-2])    # RESISTANCE CEILING
    yesterday_low   = float(lows[-2])
    yesterday_close = float(closes[-2])

    # ── CONDITION 1: Breakdown candle closes below previous low ────────────
    if today_close >= yesterday_low:
        if job_id:
            job_log(job_id,
                f"{ticker} no breakdown: close Rs{today_close:.2f} >= prev_low Rs{yesterday_low:.2f}",
                "fail")
        return None

    drop_pct = (yesterday_low - today_close) / yesterday_low * 100

    # ── CONDITION 2: Drop must be meaningful (not just a noise tick) ────────
    min_drop = cfg.get("min_drop_percent", 0.5)
    if drop_pct < min_drop:
        if job_id:
            job_log(job_id,
                f"{ticker} drop too small: {drop_pct:.2f}% < min {min_drop}%",
                "fail")
        return None

    # ── CONDITION 3: Breakdown volume >= ratio x 20-day average ────────────
    min_vol_ratio = cfg.get("min_breakdown_volume_ratio", 1.2)
    breakdown_vol = float(vols[-1]) if vols[-1] > 0 else 0.0

    # 20 completed candles before the breakdown candle
    vol_window   = vols[-21:-1]          # indices -21 to -2 inclusive
    avg_vol_20d  = float(vol_window.mean()) if len(vol_window) > 0 else 0.0
    volume_ratio = (breakdown_vol / avg_vol_20d) if avg_vol_20d > 0 else 0.0

    if volume_ratio < min_vol_ratio:
        if job_id:
            job_log(job_id,
                f"{ticker} low breakdown volume: {volume_ratio:.2f}x avg "
                f"(need >={min_vol_ratio}x)",
                "fail")
        return None

    # ── CONDITION 4: Close must be below EMA — ONE-TIME trend-weakness gate ──
    # This check applies at the moment of initial breakdown detection only.
    # Once a signal is recorded (EMA breach confirmed), the trend is already
    # damaged. On any subsequent scan day where price retests the surge high,
    # this gate is NOT re-applied — proximity alone is sufficient to trigger
    # the sell alert. Re-checking EMA on retest days would incorrectly filter
    # valid setups where price briefly bounced above EMA during the recovery.
    ema_period       = int(cfg.get("ema_period", 9))
    ema_enabled      = cfg.get("ema_filter_enabled", True)
    ema_values       = compute_ema(closes, ema_period)
    ema_at_detection = float(ema_values[-1])
    below_ema        = today_close < ema_at_detection

    if ema_enabled and not below_ema and not ema_already_breached:
        if job_id:
            job_log(job_id,
                f"{ticker} close Rs{today_close:.2f} >= EMA{ema_period} "
                f"Rs{ema_at_detection:.2f} -- trend intact, skip",
                "fail")
        return None

    if ema_already_breached and not below_ema and job_id:
        job_log(job_id,
            f"{ticker} EMA gate skipped (breach confirmed at detection); "
            f"current close Rs{today_close:.2f} vs EMA{ema_period} Rs{ema_at_detection:.2f}",
            "info")

    # ── CONDITION 5: Continuous surge ending recently ──────────────────────
    min_gain     = cfg["min_gain_percent"]
    min_green    = cfg["min_green_candles"]
    recency_days = int(cfg.get("surge_recency_days", 5))

    # Operate on completed candles only (exclude the breakdown candle itself)
    scan_closes = closes[:-1]
    scan_opens  = opens[:-1]
    scan_lows   = lows[:-1]
    scan_dates  = dates[:-1]
    n           = len(scan_closes)

    if n < min_green + 2:
        return None

    # Surge end must be within recency_days sessions of the breakdown candle
    min_end_idx  = n - recency_days
    window_min   = max(min_green + 1, 3)

    best_gain   = 0.0
    best_greens = 0
    best_reds   = 0
    best_window = None

    for wsize in range(window_min, n + 1):
        for start in range(0, n - wsize + 1):
            end = start + wsize - 1

            # Enforce recency
            if end < min_end_idx:
                continue

            net_gain = ((scan_closes[end] - scan_closes[start]) / scan_closes[start]) * 100
            if net_gain < min_gain:
                continue

            # Continuity check: red candles allowed only if higher-low holds
            is_cont, red_count, green_count = check_surge_continuity(
                scan_closes, scan_opens, scan_lows, start, end
            )
            if not is_cont or green_count < min_green:
                continue

            # Keep best by gain; prefer more recent end on tie
            prev_end = best_window[1] if best_window else -1
            if net_gain > best_gain or (net_gain == best_gain and end > prev_end):
                best_gain   = net_gain
                best_greens = green_count
                best_reds   = red_count
                best_window = (start, end)

    if best_window is None or best_gain < min_gain:
        if job_id:
            job_log(job_id,
                f"{ticker} no continuous surge >={min_gain}% ending within "
                f"last {recency_days} sessions",
                "fail")
        return None

    surge_start_idx, surge_end_idx = best_window
    surge_end_date  = scan_dates[surge_end_idx]
    days_since_end  = (datetime.now(IST).date() - surge_end_date.date()).days

    # ── CRITICAL: Use SURGE_HIGH as resistance (not yesterday_high) ────────
    # The surge window's highest point is the true resistance level where
    # price exhausted momentum. Even after breakdown + recovery, this level
    # will likely reject price again (as seen in DIXON 20 May 2025 example).
    surge_window_highs = highs[surge_start_idx:surge_end_idx + 1]
    surge_high         = float(np.max(surge_window_highs))

    if job_id:
        job_log(job_id,
            f"{ticker} PASS  surge={best_gain:.1f}%  greens={best_greens}  "
            f"allowed_reds={best_reds}  ended {days_since_end}d ago  "
            f"drop={drop_pct:.2f}%  vol={volume_ratio:.2f}x  "
            f"EMA{ema_period}={ema_at_detection:.2f} (breached at detection)  "
            f"surge_high=Rs{surge_high:.2f} (resistance)  "
            f"yesterday_high=Rs{yesterday_high:.2f}",
            "pass")

    return {
        "ticker":                  ticker,
        "today_close":             round(today_close, 2),
        "surge_high":              round(surge_high, 2),        # TRUE RESISTANCE (surge peak)
        "yesterday_high":          round(yesterday_high, 2),    # Day before breakdown
        "yesterday_close":         round(yesterday_close, 2),
        "yesterday_low":           round(yesterday_low, 2),
        "drop_pct":                round(drop_pct, 2),
        "breakdown_volume":        int(breakdown_vol),
        "avg_volume_20d":          int(avg_vol_20d),
        "volume_ratio":            round(volume_ratio, 2),
        "ema_at_detection":        round(ema_at_detection, 2),  # EMA value when signal was first detected
        "ema_breached_at_detection": True,                      # Always True here; gate passed
        "below_ema":               below_ema,                   # kept for UI/legacy compat
        "surge_gain_pct":          round(best_gain, 2),
        "surge_candles":           best_greens,
        "surge_allowed_reds":      best_reds,
        "surge_start_date":        scan_dates[surge_start_idx].strftime("%Y-%m-%d"),
        "surge_end_date":          surge_end_date.strftime("%Y-%m-%d"),
        "days_since_surge_end":    days_since_end,
    }


# ─────────────────────────────────────────────────────────────
#  SCREENING -- STEP 2
# ─────────────────────────────────────────────────────────────
def find_best_strike(candidate: Dict, cfg: Dict, job_id: str = "") -> Optional[Dict]:
    ticker      = candidate["ticker"]
    surge_high  = candidate["surge_high"]   # TRUE resistance (surge peak)

    if job_id:
        job_log(job_id, f"{ticker} -> fetching option chain...", "info")

    strike_map, expiry = fetch_option_chain(ticker)
    if strike_map is None or not strike_map:
        if job_id:
            job_log(job_id, f"{ticker} option chain empty", "fail")
        return None

    # Strike anchored ABOVE surge_high (true resistance), not yesterday_high.
    # CE goes ITM only if price breaks above the surge peak -- exactly
    # the scenario our thesis says is unlikely (as seen in DIXON 20 May example).
    strike = nearest_round_strike_above(surge_high, strike_map)
    if strike is None:
        if job_id:
            job_log(job_id,
                f"{ticker} no liquid strike above surge_high Rs{surge_high:.2f}",
                "fail")
        return None

    ce_data = strike_map[strike]
    ce_ltp  = ce_data["CE_ltp"]
    ce_oi   = ce_data["CE_oi"]

    if job_id:
        job_log(job_id,
            f"{ticker} -> strike Rs{strike:.0f} CE (surge_high Rs{surge_high:.2f}) "
            f"| LTP Rs{ce_ltp:.2f} | OI {ce_oi:,}",
            "info")

    if ce_ltp <= 0:
        if job_id:
            job_log(job_id, f"{ticker} strike Rs{strike:.0f} CE LTP=0 (illiquid)", "fail")
        return None

    ce_hist_high        = None
    ce_above_30d_status = "Unverified"
    nse_hist_calls      = 0
    remark              = "Unverified 30d High"

    if cfg.get("ce_above_historical_high", True):
        ce_hist_high, nse_hist_calls = get_ce_historical_high(
            ticker, strike, expiry, cfg.get("ce_history_days", 30)
        )
        if ce_hist_high is not None:
            if ce_ltp <= ce_hist_high:
                ce_above_30d_status = False
                remark = "CE Below 30d High"
                if job_id:
                    job_log(job_id,
                        f"{ticker} CE LTP Rs{ce_ltp} <= 30d high Rs{ce_hist_high} -- included with caution",
                        "warn")
            else:
                ce_above_30d_status = True
                remark = "CE at New High"
                if job_id:
                    job_log(job_id,
                        f"{ticker} CE LTP Rs{ce_ltp} > 30d high Rs{ce_hist_high} -- strong signal",
                        "signal")
    else:
        remark = "check disabled"

    if job_id:
        job_log(job_id,
            f"{ticker} SIGNAL  Rs{strike:.0f} CE  LTP Rs{ce_ltp:.2f}  [{remark}]",
            "signal")

    return {
        "Ticker":               ticker,
        # Price action
        "Today_Close":          candidate["today_close"],
        "Surge_High":           candidate["surge_high"],        # TRUE RESISTANCE (surge peak)
        "Yesterday_High":       candidate["yesterday_high"],    # Day before breakdown
        "Yesterday_Close":      candidate["yesterday_close"],
        "Yesterday_Low":        candidate["yesterday_low"],
        "Drop_Pct":             candidate["drop_pct"],
        # Volume
        "Breakdown_Volume":     candidate["breakdown_volume"],
        "Avg_Volume_20d":       candidate["avg_volume_20d"],
        "Volume_Ratio":         candidate["volume_ratio"],
        # Trend — EMA was breached at initial detection; not re-checked on retest days
        "EMA_At_Detection":         candidate["ema_at_detection"],
        "EMA_Breached_At_Detection": candidate["ema_breached_at_detection"],
        "Below_EMA":                candidate["below_ema"],          # legacy compat
        # Surge
        "Surge_Gain_Pct":       candidate["surge_gain_pct"],
        "Surge_Candles":        candidate["surge_candles"],
        "Surge_Allowed_Reds":   candidate["surge_allowed_reds"],
        "Surge_Start":          candidate["surge_start_date"],
        "Surge_End":            candidate["surge_end_date"],
        "Days_Since_Surge_End": candidate["days_since_surge_end"],
        # Option
        "Suggested_Strike":     strike,
        "Expiry":               expiry,
        "CE_LTP":               round(ce_ltp, 2),
        "CE_OI":                ce_oi,
        "CE_30d_High":          round(ce_hist_high, 2) if ce_hist_high else "N/A",
        "CE_Above_30d_High":    ce_above_30d_status,
        "Remark":               remark,
        "NSE_Hist_Calls":       nse_hist_calls,
        # Meta
        "Status":               "Momentum Lost",
        "Sell_Alert_Sent":      False,
        "Scanned_At":           now_ist_str(),
    }


# ─────────────────────────────────────────────────────────────
#  PROXIMITY / SELL ZONE ALERTS
#
#  Reference: Yesterday_High = resistance ceiling defined at scan time.
#
#  Alert fires when:
#      floor = yesterday_high * (1 - proximity_pct / 100)
#      floor <= cur_price <= yesterday_high
#
#  Price above yesterday_high = breakout, thesis invalidated, skip.
#  Duplicate alerts for the same ticker within 30 min are suppressed.
# ─────────────────────────────────────────────────────────────
def check_proximity_alerts(signals: List[Dict], cfg: Dict) -> List[Dict]:
    proximity_pct  = cfg["price_proximity_percent"]
    sell_zone_hits = []

    for sig in signals:
        ticker = sig["Ticker"]

        # Use Surge_High (surge peak) as primary resistance.
        # This is the strongest resistance level - where original momentum
        # exhausted. Even after breakdown + recovery (DIXON 20 May example),
        # this level will reject price again.
        resistance = (
            sig.get("Surge_High")           # NEW: True resistance
            or sig.get("Yesterday_High")    # Fallback for old signals
            or sig.get("Ten_Day_High")      # Legacy fallback
            or sig.get("Suggested_Strike")  # Last resort
        )
        if not resistance:
            time.sleep(random.uniform(0.2, 0.5))
            continue

        cur_price = get_current_price(ticker)
        if cur_price is None:
            time.sleep(random.uniform(0.2, 0.5))
            continue

        # Price broke above surge_high -> thesis invalidated (breakout confirmed)
        if cur_price > resistance:
            time.sleep(random.uniform(0.2, 0.5))
            continue

        dist_pct = (resistance - cur_price) / resistance * 100

        if dist_pct <= proximity_pct:
            hit = {
                **sig,
                "Current_Price":          round(cur_price, 2),
                "Resistance_High":        round(resistance, 2),
                "Distance_From_High_Pct": round(dist_pct, 2),
                "Alert_Time":             now_ist_str(),
            }
            sell_zone_hits.append(hit)
            send_telegram(
                cfg["telegram_bot_token"],
                cfg["telegram_chat_id"],
                f"*SELL ZONE -- {ticker}*\n"
                f"Price Rs{cur_price:.2f}  "
                f"Surge High Rs{resistance:.2f}  "
                f"Dist {dist_pct:.2f}%  (within {proximity_pct}%)\n"
                f"Strike Rs{sig.get('Suggested_Strike','?')} CE  "
                f"Expiry {sig.get('Expiry','?')}"
            )

        time.sleep(random.uniform(0.2, 0.5))

    return sell_zone_hits


# ─────────────────────────────────────────────────────────────
#  SIGNAL PRUNING
# ─────────────────────────────────────────────────────────────
def prune_stale_signals(signals: List[Dict], cfg: Dict) -> Tuple[List[Dict], int]:
    max_age = int(cfg.get("max_signal_age_days", 5))
    cutoff  = datetime.now(IST) - timedelta(days=max_age)
    fresh, pruned = [], 0
    for s in signals:
        try:
            scanned = datetime.strptime(s["Scanned_At"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=IST)
            if scanned >= cutoff:
                fresh.append(s)
            else:
                pruned += 1
        except Exception:
            fresh.append(s)
    return fresh, pruned


# ─────────────────────────────────────────────────────────────
#  SCREEN JOB
# ─────────────────────────────────────────────────────────────
def run_screen_job(tickers: List[str], cfg: Dict, job_id: str):
    signals   = []
    total     = len(tickers)
    nse_calls = 0

    job_log(job_id, f"Starting scan of {total} tickers...", "info")

    # Build a set of tickers already holding a confirmed signal so the EMA
    # gate can be skipped for them — trend weakness was already established
    # at initial detection. On a retest day, proximity to surge_high is enough.
    existing_signals = load_signals()
    ema_breached_tickers = {
        s["Ticker"] for s in existing_signals
        if s.get("EMA_Breached_At_Detection") or s.get("Below_EMA")
    }

    for idx, ticker in enumerate(tickers):
        job_progress(job_id, idx, total, ticker)
        already_breached = ticker in ema_breached_tickers
        candidate = check_surge_and_loss(ticker, cfg, job_id,
                                          ema_already_breached=already_breached)
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

    # Prune stale then merge
    existing, pruned = prune_stale_signals(load_signals(), cfg)
    if pruned:
        job_log(job_id, f"Pruned {pruned} stale signal(s) (>{cfg.get('max_signal_age_days',5)}d)", "info")

    seen  = {s["Ticker"] for s in existing}
    added = 0
    for s in signals:
        if s["Ticker"] not in seen:
            existing.append(s)
            seen.add(s["Ticker"])
            added += 1
        else:
            # Refresh existing signal with latest scan data
            existing = [s if ex["Ticker"] == s["Ticker"] else ex for ex in existing]

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
        f"Done -- {len(signals)} signal(s) found, {added} new, {pruned} pruned.", "info")
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
        run_screen_job(load_tickers(), c, jid)
        all_sigs = load_signals()
        if all_sigs:
            hits = check_proximity_alerts(all_sigs, c)
            if hits:
                prox = load_proximity()
                # Deduplicate within 30-min window
                recent = {
                    h["Ticker"] for h in prox
                    if h.get("Alert_Time") and
                    (datetime.now(IST) -
                     datetime.strptime(h["Alert_Time"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=IST)
                    ).total_seconds() < 1800
                }
                new_hits = [h for h in hits if h["Ticker"] not in recent]
                if new_hits:
                    prox.extend(new_hits)
                    save_proximity(prox[-200:])

    interval = max(cfg.get("auto_scan_interval_min", 15), 10)
    _scheduler.add_job(job, IntervalTrigger(minutes=interval),
                       id="main_scan", replace_existing=True)
    _scheduler.start()


# ─────────────────────────────────────────────────────────────
#  FASTAPI APP
# ─────────────────────────────────────────────────────────────
app = FastAPI(title="NSE Tools Suite", version="4.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/option-charts", include_in_schema=False)
async def option_charts_page():
    return FileResponse(BASE_DIR / "static" / "option-charts.html")

@app.get("/sma-screener", include_in_schema=False)
async def sma_screener_page():
    return FileResponse(BASE_DIR / "static" / "sma-screener.html")

app.include_router(oc_router)
app.include_router(sma_router)

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Pydantic schemas ──────────────────────────────────────────
class ConfigUpdate(BaseModel):
    lookback_days:               Optional[int]   = None
    min_gain_percent:            Optional[float] = None
    min_green_candles:           Optional[int]   = None
    surge_recency_days:          Optional[int]   = None
    min_drop_percent:            Optional[float] = None
    min_breakdown_volume_ratio:  Optional[float] = None
    ema_period:                  Optional[int]   = None
    ema_filter_enabled:          Optional[bool]  = None
    price_proximity_percent:     Optional[float] = None
    ce_above_historical_high:    Optional[bool]  = None
    ce_history_days:             Optional[int]   = None
    max_signal_age_days:         Optional[int]   = None
    auto_scan_enabled:           Optional[bool]  = None
    auto_scan_interval_min:      Optional[int]   = None
    market_hours_only:           Optional[bool]  = None
    telegram_bot_token:          Optional[str]   = None
    telegram_chat_id:            Optional[str]   = None

class SpecificScreenRequest(BaseModel):
    tickers: List[str]


# ─────────────────────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return FileResponse(BASE_DIR / "static" / "index.html")

@app.get("/api/status")
async def get_status():
    cfg  = load_config()
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
    cfg     = load_config()
    updates = {k: v for k, v in data.dict().items() if v is not None}
    cfg.update(updates)
    save_config(cfg)
    if cfg["auto_scan_enabled"]:
        start_scheduler(cfg)
    return cfg

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
        prox   = load_proximity()
        recent = {
            h["Ticker"] for h in prox
            if h.get("Alert_Time") and
            (datetime.now(IST) -
             datetime.strptime(h["Alert_Time"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=IST)
            ).total_seconds() < 1800
        }
        new_hits = [h for h in hits if h["Ticker"] not in recent]
        if new_hits:
            prox.extend(new_hits)
            save_proximity(prox[-200:])
        hits = new_hits
    return {"hits": len(hits), "alerts": hits}

@app.get("/api/scan-log")
async def get_scan_log():
    return list(reversed(load_scan_log()))

@app.post("/api/screen/all")
async def screen_all(background_tasks: BackgroundTasks):
    cfg     = load_config()
    tickers = load_tickers()
    job_id  = create_job()
    background_tasks.add_task(asyncio.to_thread, run_screen_job, tickers, cfg, job_id)
    return {"job_id": job_id, "total": len(tickers)}

@app.post("/api/screen/specific")
async def screen_specific(req: SpecificScreenRequest, background_tasks: BackgroundTasks):
    cfg     = load_config()
    tickers = [t.strip().upper() for t in req.tickers if t.strip()]
    job_id  = create_job()
    background_tasks.add_task(asyncio.to_thread, run_screen_job, tickers, cfg, job_id)
    return {"job_id": job_id, "total": len(tickers)}

@app.get("/api/jobs/{job_id}/stream")
async def job_stream(job_id: str):
    async def event_gen():
        sent = 0
        while True:
            with jobs_lock:
                job = jobs.get(job_id)
            if job is None:
                yield f"data: {json.dumps({'type':'error','msg':'Job not found'})}\n\n"
                break
            logs = job.get("logs", [])
            for entry in logs[sent:]:
                yield f"data: {json.dumps({'type':'log','time':entry['time'],'msg':entry['msg'],'level':entry['level']})}\n\n"
            sent = len(logs)
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

@app.get("/api/chart/{ticker}")
async def get_chart_data(ticker: str, days: int = 60):
    hist = await asyncio.to_thread(get_price_history, ticker.upper(), days)
    if hist is None or hist.empty:
        raise HTTPException(404, f"No data for {ticker}")

    signals = load_signals()
    sig = next((s for s in signals if s["Ticker"] == ticker.upper()), None)

    data = []
    for idx, row in hist.iterrows():
        data.append({
            "date":   idx.strftime("%Y-%m-%d"),
            "open":   round(float(row["Open"]),   2),
            "high":   round(float(row["High"]),   2),
            "low":    round(float(row["Low"]),    2),
            "close":  round(float(row["Close"]),  2),
            "volume": int(row.get("Volume", 0)),
        })

    return {
        "ticker":          ticker.upper(),
        "strike":          sig.get("Suggested_Strike")  if sig else None,
        "surge_high":      sig.get("Surge_High")        if sig else None,  # TRUE resistance (surge peak)
        "yesterday_high":  sig.get("Yesterday_High")    if sig else None,  # Day before breakdown
        "surge_start":     sig.get("Surge_Start")       if sig else None,  # for surge highlight
        "surge_end":       sig.get("Surge_End")         if sig else None,
        "candles":         data,
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