"""
NSE Momentum Loss Screener  —  v3.0
=====================================

CORE LOGIC (in order):
─────────────────────────────────────────────────────────────────
STEP 1 │ SURGE DETECTION  (yfinance only, zero NSE calls)
       │  • Stock must have gained >= min_gain_percent over
       │    lookback_days at some point (momentum existed)
       │  • Today's close < yesterday's LOW  →  momentum LOST

STEP 2 │ STRIKE SELECTION  (ONE NSE option-chain call per stock)
       │  • Find nearest round-figure strike ABOVE today's close
       │  • That strike's CE LTP must be > its own 30-day high
       │    (option is at premium → selling is advantageous)

STEP 3 │ PROXIMITY SELL ALERT  (yfinance, no extra NSE call)
       │  • Every scan: re-fetch stock's current price
       │  • If price is within price_proximity_percent of the
       │    suggested strike → fire Telegram SELL alert
       │    "Stock approaching strike — time to sell CE"

API CALL BUDGET:
  Screening 500 stocks:
    yfinance  : 500 calls  (history, fast — no rate limit issues)
    NSE       : only for stocks that PASS steps 1+2  (~5–20 stocks)
  Proximity check (every scan interval):
    yfinance  : N calls  (N = number of tracked stocks, typically tiny)
    NSE       : 0 calls
─────────────────────────────────────────────────────────────────
"""

import os
import json
import time
import random
import asyncio
import threading
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import aiohttp
import pandas as pd
import pytz
import requests
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

# ─────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────
TICKERS_FILE       = "tickers.csv"
CONFIG_FILE        = "config.json"
SIGNALS_FILE       = "signals.json"          # screened momentum-loss stocks
PROXIMITY_FILE     = "proximity_alerts.json" # stocks near strike (sell zone)
SCAN_LOG_FILE      = "scan_log.json"
TRACKER_FILE       = "strike_tracker.json"   # CE LTP history per tracked stock
BASELINE_FILE      = "price_baseline.json"

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
#  CONFIG
# ─────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    # Screening
    "lookback_days":           30,      # days of price history to fetch
    "min_gain_percent":        10.0,    # stock must have surged >= this% in lookback
    "min_green_candles":       3,       # consecutive green candles during surge
    "price_proximity_percent": 1.5,     # how close to strike = "sell zone"
    # Option filter
    "ce_above_historical_high": True,   # only pick strike whose CE LTP > 30d high
    "ce_history_days":         30,      # days to compute CE LTP historical high
    # Auto-scan
    "auto_scan_enabled":       True,
    "auto_scan_interval_min":  15,
    "market_hours_only":       True,
    # Telegram
    "telegram_bot_token":      "",
    "telegram_chat_id":        "",
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
#  PERSISTENCE HELPERS
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

load_signals     = lambda: _rj(SIGNALS_FILE, [])
save_signals     = lambda d: _wj(SIGNALS_FILE, d)
load_proximity   = lambda: _rj(PROXIMITY_FILE, [])
save_proximity   = lambda d: _wj(PROXIMITY_FILE, d)
load_scan_log    = lambda: _rj(SCAN_LOG_FILE, [])
save_scan_log    = lambda d: _wj(SCAN_LOG_FILE, d)
load_tracker     = lambda: _rj(TRACKER_FILE, {})
save_tracker     = lambda d: _wj(TRACKER_FILE, d)
load_baseline    = lambda: _rj(BASELINE_FILE, {})
save_baseline    = lambda d: _wj(BASELINE_FILE, d)

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
#  NSE SESSION  (created once, reused)
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
            print("[NSE] Session ready")
            return s
        except Exception as e:
            print(f"[NSE] Session init failed: {e}")
            return None

def reset_nse_session():
    global _nse_session
    with _nse_lock:
        _nse_session = None

# ─────────────────────────────────────────────────────────────
#  NSE OPTION CHAIN  (single call per ticker)
#
#  Returns dict keyed by strikePrice:
#    { 1700: {"CE_ltp": 45.3, "CE_oi": 12000, "expiry": "24-Apr-2026"}, ... }
# ─────────────────────────────────────────────────────────────
def fetch_option_chain(ticker: str) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Returns (strike_map, expiry_str) or (None, None) on failure.
    strike_map = { strike_price: {"CE_ltp": float, "CE_oi": int} }
    Makes EXACTLY 2 NSE API calls:
      1. contract-info  → get nearest expiry
      2. option-chain-v3 → get CE data
    """
    session = get_nse_session()
    if session is None:
        return None, None

    # Call 1: get expiry list
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

    time.sleep(random.uniform(1.0, 2.0))   # polite gap between the two calls

    # Call 2: option chain for nearest expiry
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
    """
    Returns OHLCV DataFrame for NSE ticker, index localized to IST dates.
    Fetches up to TODAY (inclusive) by using tomorrow as the end date.
    """
    try:
        now_ist   = datetime.now(IST)
        end_dt    = (now_ist + timedelta(days=1)).date()   # tomorrow → includes today's bar
        start_dt  = (now_ist - timedelta(days=days + 10)).date()
        hist = yf.Ticker(f"{ticker}.NS").history(
            start=str(start_dt), end=str(end_dt), auto_adjust=True
        )
        if hist.empty:
            return None
        # yfinance NSE bars are timestamped at midnight UTC of the TRADING date.
        # Convert index to IST date so [-1] is always today's session.
        if hist.index.tz is None:
            hist.index = hist.index.tz_localize("UTC")
        hist.index = hist.index.tz_convert(IST)
        hist = hist.sort_index()
        # Drop any bar whose date is in the future (shouldn't happen, but guard)
        today_ist = now_ist.date()
        hist = hist[hist.index.date <= today_ist]
        return hist
    except Exception as e:
        print(f"[yf] History error {ticker}: {e}")
        return None


def get_current_price(ticker: str) -> Optional[float]:
    """Fast live price via yfinance fast_info — zero NSE calls."""
    try:
        p = yf.Ticker(f"{ticker}.NS").fast_info.last_price
        return float(p) if p else None
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────
#  CE OPTION HISTORICAL HIGH  (yfinance)
#
#  We use the underlying stock's ATM CE option symbol to fetch
#  historical option prices from yfinance.
#  yfinance supports NSE option history with symbol format:
#    <TICKER><YYMMDD><CE/PE><STRIKE>
#  Example: HDFCBANK260424CE1700
#
#  If yfinance cannot find the option (common for far-OTM),
#  we fall back to using the STOCK's own historical high as proxy.
# ─────────────────────────────────────────────────────────────
def get_ce_historical_high(ticker: str, strike: float, expiry_str: str,
                            days: int = 30) -> Optional[float]:
    """
    Returns the 30-day historical high of the CE option's LTP.
    Uses yfinance — ZERO NSE calls.
    Falls back to stock price range as proxy if option data unavailable.
    """
    # Build option symbol: e.g. "HDFCBANK260424CE1700.NS"
    try:
        # Parse expiry like "24-Apr-2026"
        expiry_dt = datetime.strptime(expiry_str, "%d-%b-%Y")
        exp_code  = expiry_dt.strftime("%y%m%d")
        strike_int = int(strike)
        opt_symbol = f"{ticker}{exp_code}CE{strike_int}.NS"
        end   = date.today()
        start = end - timedelta(days=days + 5)
        hist  = yf.Ticker(opt_symbol).history(start=start, end=end)
        if not hist.empty and "High" in hist.columns:
            high = float(hist["High"].max())
            if high > 0:
                return high
    except Exception as e:
        print(f"[yf] CE history error {ticker} {strike}: {e}")

    # Fallback: not critical, return None (caller will skip this check)
    return None

# ─────────────────────────────────────────────────────────────
#  ROUND FIGURE STRIKE FINDER
# ─────────────────────────────────────────────────────────────
def nearest_round_strike_above(price: float, available_strikes: List[float]) -> Optional[float]:
    """
    From available_strikes, find the smallest strike that is:
      a) above current price
      b) a "round figure" — divisible by 50 or 100
    If none qualify as round, pick smallest strike above price.
    """
    above = [s for s in available_strikes if s > price]
    if not above:
        return None
    # Prefer round figures (divisible by 50)
    round_strikes = [s for s in above if int(s) % 50 == 0]
    if round_strikes:
        return min(round_strikes)
    return min(above)

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

def check_surge_and_loss(ticker: str, cfg: Dict) -> Optional[Dict]:
    hist = get_price_history(ticker, cfg["lookback_days"])
    if hist is None or len(hist) < cfg["min_green_candles"] + 3:
        print(f"[{ticker}] ❌ FAIL: Not enough history ({len(hist) if hist is not None else 0} bars)")
        return None

    hist   = hist.sort_index()
    closes = hist["Close"].values.copy()
    lows   = hist["Low"].values
    opens  = hist["Open"].values
    dates  = hist.index

    if len(closes) < 2:
        return None

    # ── Patch today's close with live price ─────────────────────────────────
    live_price = get_current_price(ticker)
    if live_price is not None:
        print(f"[{ticker}] Patching today close: history={closes[-1]:.2f} → live={live_price:.2f}")
        closes[-1] = live_price

    today_close     = float(closes[-1])
    yesterday_low   = float(lows[-2])
    yesterday_close = float(closes[-2])

    print(f"\n[{ticker}] Last 2 bars used:")
    print(f"  'Today'     {dates[-1].strftime('%Y-%m-%d')}  Close={today_close:.2f}  Low={lows[-1]:.2f}")
    print(f"  'Yesterday' {dates[-2].strftime('%Y-%m-%d')}  Close={yesterday_close:.2f}  Low={yesterday_low:.2f}")
    print(f"Ticker        — {ticker}")
    print(f"Today Close   — {today_close}")
    print(f"Yesterday Low — {yesterday_low}")

    # ── Condition A: momentum lost ───────────────────────────────────────────
    if today_close >= yesterday_low:
        print(f"[{ticker}] ❌ FAIL Step1-A: today_close {today_close} >= yesterday_low {yesterday_low}")
        return None
    print(f"[{ticker}] ✅ PASS Step1-A: today_close {today_close} < yesterday_low {yesterday_low}")

    drop_pct = ((yesterday_low - today_close) / yesterday_low) * 100

    # ── Condition B: surge detection (rolling window, allows mixed candles) ──
    #
    #  OLD logic: required UNBROKEN consecutive green candles → too strict,
    #             one red candle resets entire streak even mid-surge
    #
    #  NEW logic: sliding window of `window_size` bars
    #    • count green candles in window (close > open)
    #    • compute net gain from window_start to window_end
    #    • passes if: net_gain >= min_gain AND green_count >= min_green_candles
    #    • scans ALL windows across lookback, picks the BEST one
    #
    #  This correctly catches surges like: green-red-green-green-green
    #  which have clear upward momentum but fail the consecutive check.
    # ─────────────────────────────────────────────────────────────────────────

    min_gain   = cfg["min_gain_percent"]
    min_green  = cfg["min_green_candles"]

    # Use lookback window = max(min_green + 2, 5) bars, slide across history
    # Exclude today's bar (index -1) from surge scan — we only look at past
    scan_closes = closes[:-1]   # exclude today
    scan_opens  = opens[:-1]
    scan_dates  = dates[:-1]
    n           = len(scan_closes)

    window_size = max(min_green + 2, 5)   # e.g. min_green=2 → window=5 bars

    best_gain    = 0.0
    best_greens  = 0
    best_window  = None   # (start_idx, end_idx)

    # Also check: full lookback net gain (catches slow steady surges)
    candidate_windows = []

    # Sliding windows of increasing sizes: window_size up to full lookback
    for wsize in range(window_size, n + 1):
        for start in range(0, n - wsize + 1):
            end = start + wsize - 1
            net_gain = ((scan_closes[end] - scan_closes[start]) / scan_closes[start]) * 100
            green_count = sum(
                1 for i in range(start, end + 1)
                if scan_closes[i] > scan_opens[i]
            )
            if net_gain >= min_gain and green_count >= min_green:
                if net_gain > best_gain:
                    best_gain   = net_gain
                    best_greens = green_count
                    best_window = (start, end)

    if best_gain < min_gain or best_greens < min_green:
        print(
            f"[{ticker}] ❌ FAIL Step1-B: No surge found. "
            f"best_gain={best_gain:.2f}% (need {min_gain}%), "
            f"best_greens={best_greens} (need {min_green})"
        )
        # Debug: show the best net gain seen across any window
        best_raw = max(
            ((scan_closes[end] - scan_closes[start]) / scan_closes[start]) * 100
            for wsize in range(2, min(n, 10))
            for start in range(0, n - wsize + 1)
            for end in [start + wsize - 1]
        ) if n >= 2 else 0
        print(f"[{ticker}]   (best raw gain seen in any window: {best_raw:.2f}%)")
        return None

    surge_start_idx, surge_end_idx = best_window
    print(
        f"[{ticker}] ✅ PASS Step1-B: surge {best_gain:.2f}% over {surge_end_idx - surge_start_idx + 1} bars "
        f"({best_greens} green candles) "
        f"from {scan_dates[surge_start_idx].strftime('%Y-%m-%d')} "
        f"to {scan_dates[surge_end_idx].strftime('%Y-%m-%d')}"
    )

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
        "closes":           closes,
    }


def nearest_round_strike_above(price: float, strike_map: Dict) -> Optional[float]:
    """
    Find the best CE strike above current price:
      1. Must be above current price
      2. Must have non-zero CE LTP (actually traded)
      3. Among liquid strikes, prefer the NEAREST one
         (closest to ATM = highest premium, best to sell)
      4. If multiple at same distance, prefer round figure (div by 50/100)

    Changed signature: now takes the full strike_map dict instead of just
    a list of strikes, so we can check liquidity (CE_LTP > 0) here.
    """
    # Filter: above price AND has actual LTP
    liquid_above = {
        s: data for s, data in strike_map.items()
        if s > price and data["CE_ltp"] > 0
    }

    if not liquid_above:
        # Fallback: above price, any LTP (even 0) — pick nearest
        above = [s for s in strike_map if s > price]
        return min(above) if above else None

    # Sort by distance from current price (nearest first)
    sorted_strikes = sorted(liquid_above.keys(), key=lambda s: s - price)

    # Among the 3 nearest liquid strikes, prefer round figures
    top3 = sorted_strikes[:3]
    round_in_top3 = [s for s in top3 if int(s) % 50 == 0]
    if round_in_top3:
        return min(round_in_top3, key=lambda s: s - price)

    return sorted_strikes[0]   # nearest liquid strike regardless of round


def find_best_strike(candidate: Dict, cfg: Dict) -> Optional[Dict]:
    ticker      = candidate["ticker"]
    today_close = candidate["today_close"]

    print(f"\n[{ticker}] Step 2: fetching option chain…")
    strike_map, expiry = fetch_option_chain(ticker)

    if strike_map is None or not strike_map:
        print(f"[{ticker}] ❌ FAIL Step2-A: option chain returned None or empty")
        return None

    # ── Pass full strike_map so liquidity check happens inside ──
    strike = nearest_round_strike_above(today_close, strike_map)
    if strike is None:
        print(f"[{ticker}] ❌ FAIL Step2-B: no liquid strike found above {today_close}")
        return None

    ce_data = strike_map[strike]
    ce_ltp  = ce_data["CE_ltp"]
    ce_oi   = ce_data["CE_oi"]
    print(f"[{ticker}] Strike selected: ₹{strike}  CE_LTP={ce_ltp}  CE_OI={ce_oi}")

    if ce_ltp <= 0:
        print(f"[{ticker}] ❌ FAIL Step2-C: CE_LTP is 0 — no liquid options above price")
        return None

    ce_hist_high = None
    if cfg.get("ce_above_historical_high", True):
        ce_hist_high = get_ce_historical_high(
            ticker, strike, expiry, cfg.get("ce_history_days", 30)
        )
        print(f"[{ticker}] CE 30d hist high = {ce_hist_high}")
        if ce_hist_high is not None and ce_ltp <= ce_hist_high:
            print(f"[{ticker}] ❌ FAIL Step2-D: CE LTP ₹{ce_ltp} <= 30d high ₹{ce_hist_high}")
            return None
        elif ce_hist_high is None:
            print(f"[{ticker}] ⚠️  CE hist high unavailable — passing as Unverified")
        else:
            print(f"[{ticker}] ✅ PASS Step2-D: CE LTP ₹{ce_ltp} > 30d high ₹{ce_hist_high}")

    signal = {
        "Ticker":            ticker,
        "Today_Close":       candidate["today_close"],
        "Yesterday_Close":   candidate["yesterday_close"],
        "Yesterday_Low":     candidate["yesterday_low"],
        "Drop_Pct":          candidate["drop_pct"],
        "Surge_Gain_Pct":    candidate["surge_gain_pct"],
        "Surge_Candles":     candidate["surge_candles"],
        "Surge_Start":       candidate["surge_start_date"],
        "Surge_End":         candidate["surge_end_date"],
        "Suggested_Strike":  strike,
        "Expiry":            expiry,
        "CE_LTP":            round(ce_ltp, 2),
        "CE_OI":             ce_oi,
        "CE_30d_High":       round(ce_hist_high, 2) if ce_hist_high else "N/A",
        "CE_Above_30d_High": True if (ce_hist_high and ce_ltp > ce_hist_high) else "Unverified",
        "Status":            "Momentum Lost",
        "Sell_Alert_Sent":   False,
        "Scanned_At":        now_ist_str(),
    }
    print(f"[{ticker}] ✅ SIGNAL CREATED — Strike ₹{strike} CE LTP ₹{ce_ltp}")
    return signal

# ─────────────────────────────────────────────────────────────
#  STEP 3 — PROXIMITY SELL ALERT CHECK
#  Checks if stock price is near suggested strike → SELL signal
#  Uses yfinance only. Zero NSE calls.
# ─────────────────────────────────────────────────────────────
def check_proximity_alerts(signals: List[Dict], cfg: Dict) -> List[Dict]:
    """
    For every active signal, check if current stock price is within
    price_proximity_percent of the suggested strike.
    Returns list of signals that hit the sell zone this round.
    NSE API calls: 0
    yfinance calls: 1 per tracked stock
    """
    proximity_pct = cfg["price_proximity_percent"]
    sell_zone_hits: List[Dict] = []

    for sig in signals:
        ticker = sig["Ticker"]
        strike = sig["Suggested_Strike"]

        cur_price = get_current_price(ticker)
        if cur_price is None:
            time.sleep(random.uniform(0.2, 0.5))
            continue

        # Distance from strike as percentage
        dist_pct = abs(cur_price - strike) / strike * 100

        if dist_pct <= proximity_pct:
            hit = {**sig, "Current_Price": round(cur_price, 2),
                   "Distance_From_Strike_Pct": round(dist_pct, 2),
                   "Alert_Time": now_ist_str()}
            sell_zone_hits.append(hit)

            # Send Telegram
            arrow = "⬆️" if cur_price > sig["Today_Close"] else "⬇️"
            msg = (
                f"*🔔 SELL ZONE ALERT — {ticker}*\n\n"
                f"Stock price has returned near the suggested CE strike.\n\n"
                f"Current Price   : *₹{cur_price:.2f}* {arrow}\n"
                f"Suggested Strike: *₹{strike:.2f} CE* ({sig['Expiry']})\n"
                f"Distance        : *{dist_pct:.2f}%* "
                f"(threshold: {proximity_pct}%)\n\n"
                f"CE LTP when screened: ₹{sig['CE_LTP']:.2f}\n"
                f"CE above 30d high   : {sig['CE_Above_30d_High']}\n\n"
                f"Original surge  : +{sig['Surge_Gain_Pct']:.2f}% "
                f"({sig['Surge_Candles']} candles)\n"
                f"Today's drop    : -{sig['Drop_Pct']:.2f}% below yesterday's Low ₹{sig.get('Yesterday_Low', 'N/A')}\n\n"
                f"_Consider selling the ₹{strike:.0f} CE now._\n"
                f"Time: {now_ist_str()}"
            )
            send_telegram(cfg["telegram_bot_token"], cfg["telegram_chat_id"], msg)
            print(f"[Proximity] SELL alert sent for {ticker}")

        time.sleep(random.uniform(0.2, 0.5))

    return sell_zone_hits

# ─────────────────────────────────────────────────────────────
#  FULL SCREENING PIPELINE
# ─────────────────────────────────────────────────────────────
def run_full_screen(
    tickers: List[str],
    cfg: Dict,
    progress_cb=None,
) -> List[Dict]:
    """
    Runs all 3 steps for each ticker.
    Returns list of final signals (passed all filters).
    """
    signals: List[Dict] = []
    total = len(tickers)
    nse_call_count = 0

    for idx, ticker in enumerate(tickers):
        if progress_cb:
            progress_cb(idx, total, ticker)

        # ── Step 1: Surge + Loss (yfinance only) ─────────────
        candidate = check_surge_and_loss(ticker, cfg)
        if candidate is None:
            time.sleep(random.uniform(0.1, 0.3))
            continue

        print(f"[Screen] {ticker} passed Step 1 — surge {candidate['surge_gain_pct']:.1f}%, "
              f"drop {candidate['drop_pct']:.1f}% below yesterday Low ₹{candidate['yesterday_low']}")

        # ── Step 2: Strike + CE high check (2 NSE calls) ─────
        time.sleep(random.uniform(3.0, 5.0))   # respectful delay before NSE
        signal = find_best_strike(candidate, cfg)
        nse_call_count += 2   # always 2 per qualifying stock

        if signal is None:
            print(f"[Screen] {ticker} failed Step 2 — no valid strike")
            continue

        print(f"[Screen] {ticker} SIGNAL — Strike ₹{signal['Suggested_Strike']} CE "
              f"LTP ₹{signal['CE_LTP']}, 30d high: {signal['CE_30d_High']}")

        signals.append(signal)
        time.sleep(random.uniform(2.0, 4.0))

    # Update scan log
    log = load_scan_log()
    log.append({
        "time":            now_ist_str(),
        "tickers_scanned": total,
        "step1_pass":      "N/A",
        "signals_found":   len(signals),
        "nse_calls":       nse_call_count,
    })
    save_scan_log(log[-100:])

    return signals

# ─────────────────────────────────────────────────────────────
#  AUTO-SCAN JOB  (runs in background thread via APScheduler)
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
            print("[AutoScan] Market closed — skip")
            return

        print(f"[AutoScan] Starting at {now_ist_str()}")
        tickers = load_tickers()

        # Full screen
        new_signals = run_full_screen(tickers, c)

        if new_signals:
            # Merge with existing (avoid duplicates by ticker)
            existing = load_signals()
            existing_tickers = {s["Ticker"] for s in existing}
            added = 0
            for s in new_signals:
                if s["Ticker"] not in existing_tickers:
                    existing.append(s)
                    added += 1
                    existing_tickers.add(s["Ticker"])
                    # Send initial discovery alert
                    msg = (
                        f"*📉 Momentum Loss Signal — {s['Ticker']}*\n\n"
                        f"Surge    : +{s['Surge_Gain_Pct']:.2f}% over {s['Surge_Candles']} candles\n"
                        f"Drop Today: -{s['Drop_Pct']:.2f}% below yesterday's Low ₹{s.get('Yesterday_Low', 'N/A')}\n"
                        f"Today Close: ₹{s['Today_Close']:.2f}\n\n"
                        f"Suggested Strike: *₹{s['Suggested_Strike']:.0f} CE* ({s['Expiry']})\n"
                        f"CE LTP now: ₹{s['CE_LTP']:.2f}\n"
                        f"CE above 30d high: {s['CE_Above_30d_High']}\n\n"
                        f"_Watching for price proximity alert at "
                        f"{c['price_proximity_percent']}% from ₹{s['Suggested_Strike']:.0f}_\n"
                        f"Scanned: {now_ist_str()}"
                    )
                    send_telegram(c["telegram_bot_token"], c["telegram_chat_id"], msg)
            save_signals(existing)
            print(f"[AutoScan] {added} new signals added.")

        # Proximity check on all tracked signals
        all_signals = load_signals()
        if all_signals:
            hits = check_proximity_alerts(all_signals, c)
            if hits:
                prox = load_proximity()
                prox.extend(hits)
                save_proximity(prox[-200:])

        print(f"[AutoScan] Done. Signals={len(new_signals)}")

    interval = max(cfg.get("auto_scan_interval_min", 15), 10)
    _scheduler.add_job(job, IntervalTrigger(minutes=interval),
                       id="main_scan", replace_existing=True)
    _scheduler.start()
    print(f"[Scheduler] Started — every {interval} min")

# ─────────────────────────────────────────────────────────────
#  CHARTS
# ─────────────────────────────────────────────────────────────
def make_price_chart(ticker: str, strike: float) -> Optional[go.Figure]:
    hist = get_price_history(ticker, 60)
    if hist is None or hist.empty:
        return None
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=hist.index, open=hist["Open"], high=hist["High"],
        low=hist["Low"],  close=hist["Close"], name="Price",
        increasing_line_color="#4ade80", decreasing_line_color="#f87171",
    ))
    fig.add_hline(y=strike, line_dash="dash", line_color="#f97316",
                  annotation_text=f"Strike ₹{strike:.0f}",
                  annotation_font_color="#f97316")
    fig.update_layout(
        title=f"{ticker} — 60d Price Chart",
        xaxis_title="Date", yaxis_title="Price (₹)",
        xaxis_rangeslider_visible=False, height=420,
        template="plotly_dark",
        paper_bgcolor="#0a0f1e", plot_bgcolor="#0a0f1e",
    )
    return fig

def make_signals_summary_chart(signals: List[Dict]) -> Optional[go.Figure]:
    if not signals:
        return None
    df = pd.DataFrame(signals)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["Ticker"], y=df["Surge_Gain_Pct"],
        name="Surge Gain %", marker_color="#38bdf8",
    ))
    fig.add_trace(go.Bar(
        x=df["Ticker"], y=df["Drop_Pct"],
        name="Drop % (below prev Low)", marker_color="#f87171",
    ))
    fig.update_layout(
        barmode="group", title="Surge vs Drop — Active Signals",
        template="plotly_dark", height=320,
        paper_bgcolor="#0a0f1e", plot_bgcolor="#0a0f1e",
    )
    return fig

# ─────────────────────────────────────────────────────────────
#  STREAMLIT UI
# ─────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="NSE Momentum Loss Screener",
        page_icon="📉",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .logic-box {
        background: #0f172a; border: 1px solid #1e3a5f;
        border-radius: 10px; padding: 14px 18px; margin-bottom: 12px;
        font-size: 0.85rem; line-height: 1.7;
    }
    .step-badge {
        background: #1e3a5f; color: #38bdf8;
        border-radius: 6px; padding: 2px 10px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem; font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("📉 NSE Momentum Loss Screener")
    st.caption("Surge Detection  ·  Lost Momentum Filter  ·  CE Strike Selector  ·  Proximity Sell Alert")

    # Logic explainer
    with st.expander("📖 How it works", expanded=False):
        st.markdown("""
        <div class="logic-box">
        <span class="step-badge">STEP 1</span>&nbsp; <b>Surge + Loss Detection</b> &nbsp;(yfinance only — 0 NSE calls)<br>
        &nbsp;&nbsp;→ Stock must have gained ≥ <i>Min Gain %</i> in lookback period (surge existed)<br>
        &nbsp;&nbsp;→ Today's close must be BELOW <b>yesterday's LOW</b> (momentum lost — broke prior candle's low)<br><br>
        <span class="step-badge">STEP 2</span>&nbsp; <b>Strike Selection</b> &nbsp;(2 NSE calls per qualifying stock only)<br>
        &nbsp;&nbsp;→ Find nearest <i>round-figure CE strike</i> above today's close<br>
        &nbsp;&nbsp;→ That CE's current LTP must be > its own 30-day historical high<br>
        &nbsp;&nbsp;&nbsp;&nbsp; (option is at premium → ideal to sell)<br><br>
        <span class="step-badge">STEP 3</span>&nbsp; <b>Proximity Sell Alert</b> &nbsp;(yfinance only — 0 NSE calls)<br>
        &nbsp;&nbsp;→ Every scan: re-check if stock price returned near the strike<br>
        &nbsp;&nbsp;→ If within <i>Price Proximity %</i> → Telegram SELL alert fires<br>
        &nbsp;&nbsp;&nbsp;&nbsp; (stock came back to strike level = good CE sell point)
        </div>
        """, unsafe_allow_html=True)

    cfg = load_config()

    # Session state init
    for k, v in [
        ("signals",           load_signals()),
        ("proximity_alerts",  load_proximity()),
        ("scan_log",          load_scan_log()),
        ("last_scan",         None),
        ("scheduler_started", False),
    ]:
        if k not in st.session_state:
            st.session_state[k] = v

    if not st.session_state.scheduler_started and cfg["auto_scan_enabled"]:
        start_scheduler(cfg)
        st.session_state.scheduler_started = True

    # ── SIDEBAR ──────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Settings")

        with st.expander("📊 Screening Filters", expanded=True):
            min_gain   = st.number_input("Min Surge Gain (%)", value=cfg["min_gain_percent"],
                                         min_value=3.0, step=1.0,
                                         help="Stock must have risen at least this % during lookback period")
            min_green  = st.number_input("Min Green Candles", value=cfg["min_green_candles"],
                                         min_value=2, step=1,
                                         help="Minimum consecutive green candles in the surge")
            lookback   = st.number_input("Lookback Days", value=cfg["lookback_days"],
                                         min_value=7, step=1)
            proximity  = st.number_input("Price Proximity % (Sell Zone)",
                                         value=cfg["price_proximity_percent"],
                                         min_value=0.1, step=0.1,
                                         help="Alert when stock price is within this % of suggested strike")

        with st.expander("📈 CE Option Filter", expanded=True):
            ce_filter  = st.checkbox("CE LTP must be > its 30d High",
                                     value=cfg["ce_above_historical_high"],
                                     help="Only select strikes whose CE option is currently at elevated premium")
            ce_days    = st.number_input("CE Historical High Lookback (days)",
                                         value=cfg["ce_history_days"], min_value=7, step=1)

        with st.expander("⏱️ Auto-Scan", expanded=True):
            auto_en    = st.checkbox("Enable Auto-Scan", value=cfg["auto_scan_enabled"])
            interval   = st.number_input("Interval (min)", value=cfg["auto_scan_interval_min"],
                                         min_value=10, step=5)
            mkt_only   = st.checkbox("Market Hours Only (9:15–15:30 IST)",
                                     value=cfg["market_hours_only"])

        with st.expander("📬 Telegram Alerts"):
            bot_tok    = st.text_input("Bot Token", value=cfg["telegram_bot_token"], type="password")
            chat_id    = st.text_input("Chat ID",   value=cfg["telegram_chat_id"])

        with st.expander("📋 Tickers"):
            uploaded = st.file_uploader("Upload CSV (SYMBOL column)", type=["csv"])
            if uploaded:
                df_up = pd.read_csv(uploaded)
                if "SYMBOL" in df_up.columns:
                    df_up.to_csv(TICKERS_FILE, index=False)
                    st.success(f"✅ {len(df_up)} tickers saved.")
            specific = st.text_input("Specific tickers (comma-separated)", "")

        new_cfg = {
            "min_gain_percent": min_gain, "min_green_candles": int(min_green),
            "lookback_days": int(lookback), "price_proximity_percent": proximity,
            "ce_above_historical_high": ce_filter, "ce_history_days": int(ce_days),
            "auto_scan_enabled": auto_en, "auto_scan_interval_min": int(interval),
            "market_hours_only": mkt_only,
            "telegram_bot_token": bot_tok, "telegram_chat_id": chat_id,
        }
        if any(new_cfg.get(k) != cfg.get(k) for k in new_cfg):
            cfg.update(new_cfg)
            save_config(cfg)
            if auto_en:
                start_scheduler(cfg)
            st.toast("Settings saved.", icon="✅")

    # ── STATUS BAR ───────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Market",        "🟢 OPEN" if is_market_open() else "🔴 CLOSED")
    c2.metric("Active Signals", len(st.session_state.signals))
    c3.metric("Sell Zone Hits", len(st.session_state.proximity_alerts))
    c4.metric("Auto-Scan",     f"Every {cfg['auto_scan_interval_min']}m" if cfg["auto_scan_enabled"] else "Off")
    c5.metric("Last Scan",     st.session_state.last_scan or "—")

    st.divider()

    # ── BUTTONS ──────────────────────────────────────────────
    bc1, bc2, bc3, bc4 = st.columns(4)

    with bc1:
        if st.button("🔄 Screen All Tickers", type="primary"):
            tickers = load_tickers()
            prog = st.progress(0)
            stat = st.empty()

            def _cb(idx, total, tk):
                prog.progress(int(idx / max(total, 1) * 100))
                stat.text(f"[{idx+1}/{total}]  Checking {tk}…")

            with st.spinner(f"Screening {len(tickers)} tickers…"):
                new_sigs = run_full_screen(tickers, cfg, _cb)

            existing = load_signals()
            seen = {s["Ticker"] for s in existing}
            added = 0
            for s in new_sigs:
                if s["Ticker"] not in seen:
                    existing.append(s)
                    seen.add(s["Ticker"])
                    added += 1
                    send_telegram(
                        cfg["telegram_bot_token"], cfg["telegram_chat_id"],
                        f"*📉 New Signal: {s['Ticker']}*\n"
                        f"Surge +{s['Surge_Gain_Pct']:.2f}% → Drop -{s['Drop_Pct']:.2f}% below prev Low ₹{s.get('Yesterday_Low','N/A')}\n"
                        f"Strike: ₹{s['Suggested_Strike']:.0f} CE  LTP ₹{s['CE_LTP']:.2f}\n"
                        f"CE > 30d High: {s['CE_Above_30d_High']}"
                    )

            save_signals(existing)
            st.session_state.signals = existing
            st.session_state.last_scan = now_ist_str()
            prog.progress(100)
            stat.text(f"Done — {added} new signals found.")
            st.rerun()

    with bc2:
        if st.button("🎯 Screen Specific") and specific.strip():
            tickers = [t.strip().upper() for t in specific.split(",")]
            with st.spinner("Screening…"):
                new_sigs = run_full_screen(tickers, cfg)
            existing = load_signals()
            seen = {s["Ticker"] for s in existing}
            for s in new_sigs:
                if s["Ticker"] not in seen:
                    existing.append(s)
            save_signals(existing)
            st.session_state.signals = existing
            st.session_state.last_scan = now_ist_str()
            st.rerun()

    with bc3:
        if st.button("🔔 Check Sell Zone Now"):
            all_sigs = load_signals()
            if all_sigs:
                with st.spinner("Checking proximity…"):
                    hits = check_proximity_alerts(all_sigs, cfg)
                if hits:
                    prox = load_proximity()
                    prox.extend(hits)
                    save_proximity(prox[-200:])
                    st.session_state.proximity_alerts = prox
                    st.success(f"✅ {len(hits)} sell zone hit(s) found!")
                else:
                    st.info("No stocks in sell zone right now.")
            else:
                st.warning("No signals to check. Run a screen first.")

    with bc4:
        if st.button("🗑️ Clear Signals"):
            for path, val in [(SIGNALS_FILE, []), (PROXIMITY_FILE, [])]:
                _wj(path, val)
            st.session_state.signals = []
            st.session_state.proximity_alerts = []
            st.rerun()

    st.divider()

    # ── TABS ─────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📉 Active Signals",
        "🔔 Sell Zone Alerts",
        "📊 Overview Chart",
        "🕒 Scan Log",
    ])

    # ── Tab 1: Active Signals ─────────────────────────────────
    with tab1:
        signals = st.session_state.signals or load_signals()
        st.subheader(f"📉 Active Momentum Loss Signals  ({len(signals)} stocks)")

        if signals:
            # Build display dataframe
            df = pd.DataFrame([{
                "Ticker":           s["Ticker"],
                "Today Close":      f"₹{s['Today_Close']:.2f}",
                "Yesterday Low":    f"₹{s.get('Yesterday_Low', 'N/A')}",
                "Drop % (vs Low)":  f"-{s['Drop_Pct']:.2f}%",
                "Surge Gain %":     f"+{s['Surge_Gain_Pct']:.2f}%",
                "Surge Candles":    s["Surge_Candles"],
                "Suggested Strike": f"₹{s['Suggested_Strike']:.0f} CE",
                "CE LTP":           f"₹{s['CE_LTP']:.2f}",
                "CE > 30d High":    s["CE_Above_30d_High"],
                "Expiry":           s["Expiry"],
                "Scanned At":       s["Scanned_At"],
            } for s in signals])

            st.dataframe(df, use_container_width=True, hide_index=True)

            # Individual stock detail
            st.markdown("---")
            sel = st.selectbox("View chart for:", [s["Ticker"] for s in signals])
            sel_sig = next((s for s in signals if s["Ticker"] == sel), None)
            if sel_sig:
                col_a, col_b = st.columns([2, 1])
                with col_a:
                    fig = make_price_chart(sel, sel_sig["Suggested_Strike"])
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                with col_b:
                    st.markdown("#### Signal Details")
                    st.metric("Today Close",       f"₹{sel_sig['Today_Close']:.2f}")
                    st.metric("Yesterday Close",   f"₹{sel_sig['Yesterday_Close']:.2f}")
                    st.metric("Yesterday Low",     f"₹{sel_sig.get('Yesterday_Low', 'N/A')}")
                    st.metric("Drop (vs prev Low)", f"-{sel_sig['Drop_Pct']:.2f}%")
                    st.metric("Surge Peak Gain",   f"+{sel_sig['Surge_Gain_Pct']:.2f}%")
                    st.metric("Suggested Strike",  f"₹{sel_sig['Suggested_Strike']:.0f} CE")
                    st.metric("CE LTP (screened)", f"₹{sel_sig['CE_LTP']:.2f}")
                    st.metric("CE 30d High",        str(sel_sig.get("CE_30d_High", "N/A")))
                    st.metric("Expiry",             sel_sig["Expiry"])
                    st.info(
                        f"Sell Zone: stock price within "
                        f"{cfg['price_proximity_percent']}% of ₹{sel_sig['Suggested_Strike']:.0f} "
                        f"= ₹{sel_sig['Suggested_Strike'] * (1 - cfg['price_proximity_percent']/100):.2f}"
                        f" – ₹{sel_sig['Suggested_Strike'] * (1 + cfg['price_proximity_percent']/100):.2f}"
                    )

            # Remove individual signal
            st.markdown("---")
            rem_col1, rem_col2 = st.columns([3, 1])
            with rem_col1:
                rem = st.selectbox("Remove signal:", ["—"] + [s["Ticker"] for s in signals], key="rem_sig")
            with rem_col2:
                st.write("")
                st.write("")
                if st.button("❌ Remove") and rem != "—":
                    updated = [s for s in signals if s["Ticker"] != rem]
                    save_signals(updated)
                    st.session_state.signals = updated
                    st.rerun()
        else:
            st.info("No signals yet. Click **Screen All Tickers** or wait for auto-scan.")

    # ── Tab 2: Sell Zone Alerts ──────────────────────────────
    with tab2:
        prox = st.session_state.proximity_alerts or load_proximity()
        st.subheader(f"🔔 Sell Zone Alerts  ({len(prox)} hits)")
        st.caption(
            f"Fired when stock price comes within "
            f"{cfg['price_proximity_percent']}% of the suggested CE strike."
        )
        if prox:
            df_p = pd.DataFrame([{
                "Alert Time":      p["Alert_Time"],
                "Ticker":          p["Ticker"],
                "Current Price":   f"₹{p['Current_Price']:.2f}",
                "Strike":          f"₹{p['Suggested_Strike']:.0f} CE",
                "Distance":        f"{p['Distance_From_Strike_Pct']:.2f}%",
                "CE LTP (initial)":f"₹{p['CE_LTP']:.2f}",
                "CE > 30d High":   p["CE_Above_30d_High"],
                "Surge Gain":      f"+{p['Surge_Gain_Pct']:.2f}%",
                "Expiry":          p["Expiry"],
            } for p in reversed(prox)])
            st.dataframe(df_p, use_container_width=True, hide_index=True)
        else:
            st.info("No sell zone alerts yet.")

    # ── Tab 3: Overview Chart ────────────────────────────────
    with tab3:
        signals = st.session_state.signals or load_signals()
        if signals:
            fig_sum = make_signals_summary_chart(signals)
            if fig_sum:
                st.plotly_chart(fig_sum, use_container_width=True)

            # Metrics grid
            st.markdown("---")
            cols = st.columns(min(len(signals), 4))
            for i, s in enumerate(signals[:8]):
                with cols[i % 4]:
                    st.metric(
                        label=s["Ticker"],
                        value=f"₹{s['Today_Close']:.2f}",
                        delta=f"Strike ₹{s['Suggested_Strike']:.0f}",
                        delta_color="off",
                    )
        else:
            st.info("No signals to display.")

    # ── Tab 4: Scan Log ──────────────────────────────────────
    with tab4:
        log = load_scan_log()
        st.subheader("🕒 Scan History")
        if log:
            df_log = pd.DataFrame(reversed(log))
            st.dataframe(df_log, use_container_width=True, hide_index=True)
            total_nse = sum(r.get("nse_calls", 0) for r in log)
            st.caption(f"Total NSE API calls across all scans: **{total_nse}**")
        else:
            st.info("No scan history yet.")

    # Auto-refresh every 60s
    st.markdown(
        "<script>setTimeout(()=>window.location.reload(),60000);</script>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()