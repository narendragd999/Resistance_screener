"""
NSE Momentum Loss Screener — Walk-Forward Backtesting Engine
============================================================
Simulates the 3-gate CE sell strategy over historical daily prices.

Gate 1 : Stock surges ≥ min_gain_percent continuously (green candles /
          red candles allowed only if they hold higher-low structure).
          Followed by a breakdown candle: close < prev_low with volume
          ≥ ratio × 20d avg.

Gate 2 : After the breakdown, price closes below 9-EMA at least once
          with above-average volume  (sticky confirmation).

Gate 3 : Price rallies back within price_proximity_percent of surge_high
          → Sell a slightly OTM CE above surge_high.
          Trade held until that month's NSE expiry (last Thursday).

Outcome :
    WIN  → stock closes below strike at expiry   (CE expires worthless)
    LOSS → stock closes ≥ strike at expiry       (CE ITM / assignment risk)

Run standalone:
    python backtest.py --tickers RELIANCE TCS HDFCBANK --years 2

As FastAPI router, mount in main.py:
    from backtest import router as bt_router
    app.include_router(bt_router)
"""

import argparse
import hashlib
import json
import math
import sys
from calendar import monthrange
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytz
import yfinance as yf
from scipy.stats import norm


# ── Optional FastAPI import (not required for standalone use) ──────────────
try:
    from fastapi import APIRouter, BackgroundTasks
    from fastapi.responses import JSONResponse
    _has_fastapi = True
except ImportError:
    _has_fastapi = False

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
IST   = pytz.timezone("Asia/Kolkata")
RISK_FREE_RATE = 0.065      # India approximate 10Y G-Sec yield
VOL_WINDOW     = 30         # days for historical vol estimate
MIN_WARMUP     = 60         # trading days needed before first signal scan

DEFAULT_CONFIG = {
    "lookback_days":              252,    # 1 trading year history per scan window
    "min_gain_percent":           18.0,
    "min_green_candles":          2,
    "surge_recency_days":         45,
    "min_drop_percent":           0.1,
    "min_breakdown_volume_ratio": 0.5,
    "ema_period":                 9,
    "price_proximity_percent":    1.0,
    "sell_zone_lookback_days":    10,
}

# ─────────────────────────────────────────────────────────────────────────────
#  DISK CACHE
#  Results are keyed by MD5( sorted_tickers + years + full_config ).
#  Cache lives in ./backtest_cache/<hash>.json beside this file.
#  A cached entry is reused unless the caller passes force=True.
# ─────────────────────────────────────────────────────────────────────────────
CACHE_DIR = Path(__file__).parent / "backtest_cache"


def _cache_key(tickers: List[str], years: int, cfg: Dict) -> str:
    """Stable MD5 key for a (tickers, years, config) combination."""
    payload = json.dumps(
        {
            "tickers": sorted(t.upper() for t in tickers),
            "years":   years,
            "config":  {k: cfg[k] for k in sorted(cfg)},
        },
        sort_keys=True,
    )
    return hashlib.md5(payload.encode()).hexdigest()


def _cache_load(key: str) -> Optional[Dict]:
    """Return cached result dict or None if not found."""
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass   # corrupt file — treat as cache miss
    return None


def _cache_save(key: str, data: Dict) -> None:
    """Persist result dict to disk under the given key."""
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = CACHE_DIR / f"{key}.json"
        with open(path, "w") as f:
            json.dump(data, f, separators=(",", ":"))
        print(f"[cache] saved → {path.name}")
    except Exception as e:
        print(f"[cache] save failed: {e}", file=sys.stderr)


def _cache_list() -> List[Dict]:
    """Return metadata for every cached entry (for the /cache endpoint)."""
    if not CACHE_DIR.exists():
        return []
    entries = []
    for p in sorted(CACHE_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            with open(p) as f:
                d = json.load(f)
            agg = d.get("aggregate", {})
            entries.append({
                "key":       p.stem,
                "tickers":   agg.get("tickers", []),
                "years":     d.get("years_tested"),
                "trades":    agg.get("total_trades", 0),
                "accuracy":  agg.get("accuracy_pct", 0),
                "cached_at": d.get("cached_at", ""),
                "size_kb":   round(p.stat().st_size / 1024, 1),
            })
        except Exception:
            pass
    return entries


def _cache_delete(key: str) -> bool:
    """Delete a single cache entry. Returns True if deleted."""
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        path.unlink()
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
#  OPTION MATHS
# ─────────────────────────────────────────────────────────────────────────────
def bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes European call price. T in years."""
    if T <= 0:
        return max(0.0, S - K)
    if sigma <= 0:
        return max(0.0, S - K)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def hist_vol(closes: np.ndarray, window: int = VOL_WINDOW) -> float:
    """Annualised historical volatility from daily close prices."""
    if len(closes) < window + 1:
        window = len(closes) - 1
    if window < 2:
        return 0.30   # fallback 30%
    log_ret = np.log(closes[-(window + 1):][1:] / closes[-(window + 1):][:-1])
    return float(np.std(log_ret, ddof=1) * math.sqrt(252))


# ─────────────────────────────────────────────────────────────────────────────
#  NSE UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def nse_monthly_expiry(year: int, month: int) -> date:
    """Last Thursday of the given month (NSE monthly F&O expiry)."""
    _, last_day_num = monthrange(year, month)
    d = date(year, month, last_day_num)
    while d.weekday() != 3:   # 3 = Thursday
        d -= timedelta(days=1)
    return d


def nse_strike_interval(price: float) -> float:
    """NSE standard strike interval for a given stock price."""
    if price < 250:
        return 10
    if price < 500:
        return 20
    if price < 1000:
        return 50
    if price < 2500:
        return 100
    if price < 5000:
        return 200
    return 500


def nearest_otm_strike(surge_high: float) -> float:
    """
    Nearest strike ABOVE surge_high, rounded to NSE standard interval.
    'Slightly OTM' = closest available strike above the resistance level.
    """
    interval = nse_strike_interval(surge_high)
    return math.ceil(surge_high / interval) * interval


# ─────────────────────────────────────────────────────────────────────────────
#  DATA FETCHING
# ─────────────────────────────────────────────────────────────────────────────
def fetch_history(ticker: str, years: int = 2) -> Optional[pd.DataFrame]:
    """Fetch `years` of daily OHLCV for `ticker` (NSE) via yfinance."""
    try:
        end   = datetime.now(IST).date() + timedelta(days=1)
        start = end - timedelta(days=years * 366)
        df = yf.Ticker(f"{ticker}.NS").history(
            start=str(start), end=str(end), auto_adjust=False
        )
        if df.empty:
            return None
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df.index = df.index.tz_convert(IST)
        df = df.sort_index()
        # auto_adjust=False may return MultiIndex columns on newer yfinance versions;
        # flatten to simple column names so downstream code works unchanged.
        if isinstance(df.columns, type(df.columns)) and hasattr(df.columns, 'get_level_values'):
            try:
                df.columns = df.columns.get_level_values(0)
            except Exception:
                pass
        # Keep only the OHLCV columns we need (drops Adj Close, Dividends, etc.)
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col not in df.columns:
                print(f"[yf] {ticker}: missing column {col}", file=sys.stderr)
                return None
        return df[["Open", "High", "Low", "Close", "Volume"]]
    except Exception as e:
        print(f"[yf] {ticker}: {e}", file=sys.stderr)
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  CORE SIGNAL LOGIC  (ported from main.py — no live price injection)
# ─────────────────────────────────────────────────────────────────────────────
def compute_ema(values: np.ndarray, period: int) -> np.ndarray:
    return pd.Series(values, dtype=float).ewm(span=period, adjust=False).mean().values


def check_surge_continuity(
    closes: np.ndarray, opens: np.ndarray, lows: np.ndarray,
    start: int, end: int
) -> Tuple[bool, int, int]:
    green_count = allowed_red = 0
    for i in range(start, end + 1):
        if closes[i] > opens[i]:
            green_count += 1
        else:
            if i > start and closes[i] < lows[i - 1]:
                return False, allowed_red, green_count
            allowed_red += 1
    return True, allowed_red, green_count


def detect_breakdown(
    closes: np.ndarray, highs: np.ndarray, lows: np.ndarray,
    opens: np.ndarray, vols: np.ndarray,
    idx: int, cfg: Dict
) -> Optional[Dict]:
    """
    Check if candle at `idx` is a valid breakdown candle.
    Returns breakdown info dict or None.
    Looks back up to surge_recency_days to find breakdown, then checks surge.
    """
    min_drop      = cfg.get("min_drop_percent", 0.1)
    min_vol_ratio = cfg.get("min_breakdown_volume_ratio", 0.5)
    lookback_bd   = int(cfg.get("surge_recency_days", 45))

    # ── Step 1: Find a valid breakdown candle within lookback window ──────────
    breakdown_idx = None
    yesterday_high = yesterday_low = drop_pct = breakdown_vol = None
    avg_vol_20d = volume_ratio = ema_val = None

    for offset in range(0, min(lookback_bd + 1, idx)):
        bd_idx   = idx - offset
        prev_idx = bd_idx - 1
        if prev_idx < 1:
            break

        bd_close  = float(closes[bd_idx])
        prev_low  = float(lows[prev_idx])
        prev_high = float(highs[prev_idx])

        if bd_close >= prev_low:
            continue
        _drop = (prev_low - bd_close) / prev_low * 100
        if _drop < min_drop:
            continue

        _bd_vol  = float(vols[bd_idx]) if vols[bd_idx] > 0 else 0.0
        _vol_win = vols[max(0, bd_idx - 20):bd_idx]
        _avg_vol = float(_vol_win.mean()) if len(_vol_win) > 0 else 0.0
        _v_ratio = (_bd_vol / _avg_vol) if _avg_vol > 0 else 0.0
        if _v_ratio < min_vol_ratio:
            continue

        breakdown_idx  = bd_idx
        yesterday_high = prev_high
        yesterday_low  = prev_low
        drop_pct       = _drop
        breakdown_vol  = _bd_vol
        avg_vol_20d    = _avg_vol
        volume_ratio   = _v_ratio
        ema_val        = None   # computed below
        break

    if breakdown_idx is None:
        return None

    # ── Step 2: Verify continuous surge ending just before breakdown ──────────
    min_gain  = cfg.get("min_gain_percent", 18.0)
    min_green = cfg.get("min_green_candles", 2)
    recency   = int(cfg.get("surge_recency_days", 45))
    ema_period= int(cfg.get("ema_period", 9))

    scan_closes = closes[:breakdown_idx]
    scan_opens  = opens[:breakdown_idx]
    scan_lows   = lows[:breakdown_idx]
    scan_highs  = highs[:breakdown_idx]
    n = len(scan_closes)
    if n < min_green + 2:
        return None

    min_end_idx  = max(0, n - recency)
    window_min   = max(min_green + 1, 3)
    best_gain    = 0.0
    best_window  = None
    best_greens  = 0

    for wsize in range(window_min, n + 1):
        for start in range(0, n - wsize + 1):
            end_ = start + wsize - 1
            if end_ < min_end_idx:
                continue
            net_gain = (scan_closes[end_] - scan_closes[start]) / scan_closes[start] * 100
            if net_gain < min_gain:
                continue
            is_cont, _, green_count = check_surge_continuity(
                scan_closes, scan_opens, scan_lows, start, end_
            )
            if not is_cont or green_count < min_green:
                continue
            prev_end = best_window[1] if best_window else -1
            if net_gain > best_gain or (net_gain == best_gain and end_ > prev_end):
                best_gain   = net_gain
                best_greens = green_count
                best_window = (start, end_)

    if best_window is None or best_gain < min_gain:
        return None

    surge_start_idx, surge_end_idx = best_window
    surge_high = float(np.max(scan_highs[surge_start_idx:surge_end_idx + 1]))

    # EMA at breakdown
    ema_vals  = compute_ema(closes[:breakdown_idx + 1], ema_period)
    ema_at_bd = float(ema_vals[-1])

    return {
        "breakdown_idx":  breakdown_idx,
        "surge_high":     round(surge_high, 2),
        "yesterday_high": round(float(yesterday_high), 2),
        "yesterday_low":  round(float(yesterday_low), 2),
        "drop_pct":       round(drop_pct, 2),
        "volume_ratio":   round(volume_ratio, 2),
        "surge_gain_pct": round(best_gain, 2),
        "surge_candles":  best_greens,
        "surge_start_idx":surge_start_idx,
        "surge_end_idx":  surge_end_idx,
        "ema_at_breakdown": round(ema_at_bd, 2),
    }


def check_gate2(
    closes: np.ndarray, vols: np.ndarray,
    bd_idx: int, cfg: Dict
) -> Optional[int]:
    """
    Scan from bd_idx onward. Return first index where
    close < 9-EMA AND volume ≥ min_breakdown_volume_ratio × 20d avg.
    """
    ema_period    = int(cfg.get("ema_period", 9))
    min_vol_ratio = float(cfg.get("min_breakdown_volume_ratio", 0.5))
    ema_vals      = compute_ema(closes, ema_period)
    avg_vol_20d   = float(np.mean(vols[-20:])) if len(vols) >= 20 else float(np.mean(vols))
    if avg_vol_20d <= 0:
        return None
    for i in range(bd_idx, len(closes)):
        if closes[i] < ema_vals[i]:
            if float(vols[i]) / avg_vol_20d >= min_vol_ratio:
                return i
    return None


def check_gate3(
    closes: np.ndarray, highs: np.ndarray,
    g2_idx: int, surge_high: float, cfg: Dict
) -> Optional[int]:
    """
    Scan from g2_idx onward. Return first index where price rallies back
    within price_proximity_percent of surge_high AND price ≤ surge_high.
    A breakout above surge_high invalidates the thesis for that episode.
    """
    prox_pct = float(cfg.get("price_proximity_percent", 1.0))
    floor    = surge_high * (1 - prox_pct / 100)
    for i in range(g2_idx, len(closes)):
        price = float(closes[i])
        # Breakout: thesis invalidated
        if price > surge_high:
            return None
        if floor <= price <= surge_high:
            return i
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  TRADE EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_trade(
    closes: np.ndarray, dates: pd.DatetimeIndex,
    g3_idx: int, surge_high: float, cfg: Dict
) -> Optional[Dict]:
    """
    At Gate 3 trigger (day g3_idx):
      1. Compute strike = nearest OTM above surge_high
      2. Estimate CE premium via Black-Scholes
      3. Find monthly NSE expiry
      4. Look up closing price on or after expiry date
      5. Determine WIN / LOSS and P&L
    """
    entry_price  = float(closes[g3_idx])
    entry_date   = dates[g3_idx].date()

    # --- Strike ---
    strike = nearest_otm_strike(surge_high)

    # --- Time to expiry ---
    expiry_date = nse_monthly_expiry(entry_date.year, entry_date.month)
    # If expiry is within 5 calendar days, roll to next month
    if (expiry_date - entry_date).days <= 5:
        next_month = entry_date.month + 1 if entry_date.month < 12 else 1
        next_year  = entry_date.year + (1 if entry_date.month == 12 else 0)
        expiry_date = nse_monthly_expiry(next_year, next_month)

    T_years = max(0.001, (expiry_date - entry_date).days / 365.0)

    # --- Historical volatility at entry ---
    sigma = hist_vol(closes[:g3_idx + 1], VOL_WINDOW)

    # --- Black-Scholes CE premium at entry ---
    entry_ce_premium = round(bs_call(entry_price, strike, T_years, RISK_FREE_RATE, sigma), 2)
    if entry_ce_premium < 0.01:
        return None   # negligible premium — skip

    # --- Find expiry-day close (or nearest available date) ---
    exp_dates = [d.date() for d in dates]
    exp_idx   = None
    for search_offset in range(0, 5):   # allow up to 4 days slippage for holidays
        target = expiry_date + timedelta(days=search_offset)
        if target in exp_dates:
            exp_idx = exp_dates.index(target)
            break

    if exp_idx is None or exp_idx >= len(closes):
        return None   # no data at/after expiry

    expiry_close  = float(closes[exp_idx])
    actual_expiry = dates[exp_idx].date()

    # --- P&L ---
    # Seller receives entry_ce_premium; pays max(0, expiry_close - strike) at settlement
    intrinsic_at_expiry = max(0.0, expiry_close - strike)
    pnl_per_lot_raw = entry_ce_premium - intrinsic_at_expiry   # per unit
    result = "WIN" if expiry_close < strike else "LOSS"

    # Max loss cap at 3× premium (reasonable for sold CE)
    pnl = max(pnl_per_lot_raw, -3 * entry_ce_premium)

    return {
        # Dates
        "entry_date":       str(entry_date),
        "expiry_date":      str(expiry_date),
        "actual_data_date": str(actual_expiry),
        # Prices
        "entry_price":      round(entry_price, 2),
        "surge_high":       round(surge_high, 2),
        "strike":           int(strike),
        "entry_ce_premium": entry_ce_premium,
        "expiry_close":     round(expiry_close, 2),
        "intrinsic_value":  round(intrinsic_at_expiry, 2),
        # Greeks
        "iv_sigma":         round(sigma * 100, 1),    # %
        "T_days":           (expiry_date - entry_date).days,
        # Outcome
        "result":           result,
        "pnl_per_unit":     round(pnl, 2),
        "return_pct":       round(pnl / entry_ce_premium * 100, 1) if entry_ce_premium > 0 else 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  WALK-FORWARD ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def backtest_ticker(
    ticker: str, cfg: Dict, years: int = 2, verbose: bool = False
) -> Dict:
    """
    Full walk-forward backtest for a single ticker.
    Returns summary + list of individual trades.
    """
    df = fetch_history(ticker, years)
    if df is None or len(df) < MIN_WARMUP + 10:
        return {"ticker": ticker, "error": "Insufficient data", "trades": []}

    closes = df["Close"].values.astype(float)
    highs  = df["High"].values.astype(float)
    lows   = df["Low"].values.astype(float)
    opens  = df["Open"].values.astype(float)
    vols   = df["Volume"].values.astype(float)
    dates  = df.index

    signals: List[Dict] = []    # Gate 1 hits
    trades:  List[Dict] = []    # completed Gate 3 trades

    # Track processed breakdown candles to avoid duplicate signals
    seen_breakdown_dates: set = set()
    # Deduplicate completed trades: same surge_high + same Gate-3 date = same
    # trade episode regardless of which breakdown candle triggered it.
    seen_trade_keys: set = set()   # (surge_high_rounded, g3_date)

    print(f"[{ticker}] {len(closes)} candles from {dates[0].date()} to {dates[-1].date()}")

    # Walk-forward: simulate being on each day starting from MIN_WARMUP
    for today_idx in range(MIN_WARMUP, len(closes)):
        # ── Gate 1: Detect new breakdown signal at today_idx ─────────────────
        bd_info = detect_breakdown(
            closes[:today_idx + 1], highs[:today_idx + 1],
            lows[:today_idx + 1], opens[:today_idx + 1],
            vols[:today_idx + 1], today_idx, cfg
        )
        if bd_info is None:
            continue

        bd_date_str = str(dates[bd_info["breakdown_idx"]].date())
        if bd_date_str in seen_breakdown_dates:
            continue
        seen_breakdown_dates.add(bd_date_str)

        surge_high  = bd_info["surge_high"]
        bd_abs_idx  = bd_info["breakdown_idx"]   # absolute index in full arrays

        if verbose:
            print(f"  Gate 1 @ {bd_date_str}  surge_high={surge_high}  "
                  f"drop={bd_info['drop_pct']}%  vol={bd_info['volume_ratio']}x")

        # ── Gate 2: Scan forward from breakdown to find EMA+vol breach ────────
        g2_idx = check_gate2(closes, vols, bd_abs_idx, cfg)
        if g2_idx is None:
            if verbose:
                print(f"    Gate 2 NOT confirmed — skip")
            continue

        g2_date = str(dates[g2_idx].date())
        if verbose:
            print(f"    Gate 2 @ {g2_date}")

        # ── Recompute surge_high: max high from (surge_recency_days before    ──
        # ── breakdown) up to (not including) the Gate-2 EMA-break candle.   ──
        #                                                                     ──
        # WHY NOT surge_start_idx: detect_breakdown's continuity check       ──
        # (no red candle may close below prev low) rejects the full visible   ──
        # surge and returns a tiny late window near the top.  Using that      ──
        # start index misses the real price peak.  Instead we scan the same   ──
        # lookback window used to find the breakdown and let np.max find the  ──
        # actual highest high — which IS the resistance level the user sees.  ──
        recency_days   = int(cfg.get("surge_recency_days", 45))
        lookback_start = max(0, bd_abs_idx - recency_days)
        if lookback_start < g2_idx:
            surge_high = float(np.max(highs[lookback_start:g2_idx]))
        # (fallback: gate-1 surge_high retained if window is degenerate)

        if verbose:
            print(f"    Surge-high (pre-EMA-break, {recency_days}d lookback) → {surge_high:.2f}")

        # ── Gate 3: Scan forward from Gate 2 for sell-zone retest ────────────
        g3_idx = check_gate3(closes, highs, g2_idx, surge_high, cfg)
        if g3_idx is None:
            if verbose:
                print(f"    Gate 3 never triggered (breakout or no retest)")
            continue

        g3_date = str(dates[g3_idx].date())

        # ── Dedup: skip if this exact trade episode already recorded ──────────
        trade_key = (round(surge_high, 2), g3_date)
        if trade_key in seen_trade_keys:
            if verbose:
                print(f"    Duplicate trade episode (surge={surge_high:.2f}, g3={g3_date}) — skip")
            continue
        seen_trade_keys.add(trade_key)

        if verbose:
            print(f"    Gate 3 @ {g3_date}  price={closes[g3_idx]:.2f}")

        # ── Evaluate CE sell trade ────────────────────────────────────────────
        trade = evaluate_trade(closes, dates, g3_idx, surge_high, cfg)
        if trade is None:
            if verbose:
                print(f"    Trade evaluation skipped (no expiry data or zero premium)")
            continue

        trade.update({
            "ticker":           ticker,
            "gate1_date":       bd_date_str,
            "gate2_date":       g2_date,
            "gate3_date":       g3_date,
            "surge_gain_pct":   bd_info["surge_gain_pct"],
            "surge_candles":    bd_info["surge_candles"],
            "volume_ratio":     bd_info["volume_ratio"],
        })
        trades.append(trade)

        if verbose:
            tag = "✅ WIN" if trade["result"] == "WIN" else "❌ LOSS"
            print(f"    {tag}  strike={trade['strike']}  "
                  f"entry_ce={trade['entry_ce_premium']}  "
                  f"expiry_close={trade['expiry_close']}  "
                  f"P&L={trade['pnl_per_unit']}")

    # ── Summary ───────────────────────────────────────────────────────────────
    total   = len(trades)
    wins    = sum(1 for t in trades if t["result"] == "WIN")
    losses  = total - wins
    accuracy = round(wins / total * 100, 1) if total > 0 else 0.0
    avg_pnl  = round(sum(t["pnl_per_unit"] for t in trades) / total, 2) if total > 0 else 0.0
    avg_premium = round(sum(t["entry_ce_premium"] for t in trades) / total, 2) if total > 0 else 0.0

    return {
        "ticker":        ticker,
        "data_from":     str(dates[0].date()),
        "data_to":       str(dates[-1].date()),
        "total_candles": len(closes),
        "summary": {
            "total_trades":  total,
            "wins":          wins,
            "losses":        losses,
            "accuracy_pct":  accuracy,
            "avg_pnl_per_unit": avg_pnl,
            "avg_premium":   avg_premium,
            "total_pnl":     round(sum(t["pnl_per_unit"] for t in trades), 2),
        },
        "trades": trades,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  MULTI-TICKER BACKTEST
# ─────────────────────────────────────────────────────────────────────────────
def backtest_multiple(
    tickers: List[str], cfg: Dict, years: int = 2, verbose: bool = False
) -> Dict:
    results = []
    for tkr in tickers:
        print(f"\n{'='*60}\nBacktesting {tkr}...\n{'='*60}")
        res = backtest_ticker(tkr, cfg, years=years, verbose=verbose)
        results.append(res)

    # Aggregate
    all_trades  = [t for r in results for t in r.get("trades", [])]
    total       = len(all_trades)
    wins        = sum(1 for t in all_trades if t["result"] == "WIN")
    losses      = total - wins
    accuracy    = round(wins / total * 100, 1) if total > 0 else 0.0
    total_pnl   = round(sum(t["pnl_per_unit"] for t in all_trades), 2)

    return {
        "config":      cfg,
        "years_tested":years,
        "aggregate": {
            "tickers":       tickers,
            "total_trades":  total,
            "wins":          wins,
            "losses":        losses,
            "accuracy_pct":  accuracy,
            "total_pnl":     total_pnl,
            "avg_pnl":       round(total_pnl / total, 2) if total > 0 else 0.0,
        },
        "per_ticker": results,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  FASTAPI ROUTER  (optional — only mounted if FastAPI is available)
# ─────────────────────────────────────────────────────────────────────────────
if _has_fastapi:
    from pydantic import BaseModel as _BM

    class BacktestRequest(_BM):
        tickers:  List[str]
        years:    int        = 2
        config:   Dict       = {}
        force:    bool       = False   # True → bypass cache, always run fresh

    router = APIRouter(prefix="/api/backtest", tags=["backtest"])

    @router.post("/run")
    async def run_backtest(req: BacktestRequest):
        cfg = {**DEFAULT_CONFIG, **req.config}
        key = _cache_key(req.tickers, req.years, cfg)

        # ── Cache hit ────────────────────────────────────────────────────────
        if not req.force:
            cached = _cache_load(key)
            if cached is not None:
                cached["from_cache"] = True
                print(f"[cache] HIT  {key[:10]}…  tickers={req.tickers}")
                return JSONResponse(content=cached)

        # ── Cache miss → run full backtest ───────────────────────────────────
        print(f"[cache] MISS {key[:10]}…  tickers={req.tickers}  force={req.force}")
        try:
            result = backtest_multiple(req.tickers, cfg, years=req.years, verbose=False)
            result["from_cache"] = False
            result["cached_at"]  = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")
            _cache_save(key, result)
            return JSONResponse(content=result)
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

    @router.get("/config")
    async def backtest_config():
        return DEFAULT_CONFIG

    @router.get("/cache")
    async def list_cache():
        """List all cached backtest results with metadata."""
        return JSONResponse(content={"entries": _cache_list()})

    @router.delete("/cache/{key}")
    async def delete_cache_entry(key: str):
        """Delete a single cache entry by its MD5 key."""
        deleted = _cache_delete(key)
        return JSONResponse(content={"deleted": deleted, "key": key})

    @router.delete("/cache")
    async def clear_all_cache():
        """Wipe the entire cache directory."""
        count = 0
        if CACHE_DIR.exists():
            for p in CACHE_DIR.glob("*.json"):
                p.unlink()
                count += 1
        return JSONResponse(content={"cleared": count})


# ─────────────────────────────────────────────────────────────────────────────
#  STANDALONE CLI
# ─────────────────────────────────────────────────────────────────────────────
def _print_results(result: Dict):
    agg = result.get("aggregate", {})
    print(f"\n{'═'*70}")
    print(f"  BACKTEST RESULTS  ({result.get('years_tested', '?')} years)")
    print(f"{'═'*70}")
    print(f"  Tickers  : {', '.join(agg.get('tickers', []))}")
    print(f"  Trades   : {agg.get('total_trades', 0)}")
    print(f"  Wins     : {agg.get('wins', 0)}")
    print(f"  Losses   : {agg.get('losses', 0)}")
    print(f"  Accuracy : {agg.get('accuracy_pct', 0):.1f}%")
    print(f"  Avg P&L  : ₹{agg.get('avg_pnl', 0):.2f} per unit")
    print(f"  Total P&L: ₹{agg.get('total_pnl', 0):.2f} (sum across all trades)")
    print(f"{'═'*70}\n")

    for tkr_res in result.get("per_ticker", []):
        tkr = tkr_res["ticker"]
        s   = tkr_res.get("summary", {})
        err = tkr_res.get("error")
        if err:
            print(f"  {tkr}: ERROR — {err}")
            continue
        print(f"  {tkr}: {s.get('total_trades',0)} trades  "
              f"Acc={s.get('accuracy_pct',0)}%  "
              f"AvgP&L=₹{s.get('avg_pnl_per_unit',0):.2f}  "
              f"AvgPremium=₹{s.get('avg_premium',0):.2f}")

        for t in tkr_res.get("trades", []):
            tag = "✅" if t["result"] == "WIN" else "❌"
            print(
                f"    {tag} G1={t['gate1_date']} G3={t['gate3_date']}"
                f"  Surge={t['surge_high']}  Strike={t['strike']}"
                f"  Premium=₹{t['entry_ce_premium']}"
                f"  ExpiryClose={t['expiry_close']}"
                f"  P&L=₹{t['pnl_per_unit']}"
                f"  ({t['result']})"
            )
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NSE Momentum Loss Screener — Backtesting Engine"
    )
    parser.add_argument(
        "--tickers", nargs="+", default=["RELIANCE"],
        help="NSE ticker symbols (without .NS suffix)"
    )
    parser.add_argument(
        "--years", type=int, default=2,
        help="Years of historical data to test over (default: 2)"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config.json (uses defaults if omitted)"
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Save results as JSON to this path"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print per-signal debug output"
    )
    args = parser.parse_args()

    cfg = DEFAULT_CONFIG.copy()
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            cfg.update(json.load(f))

    result = backtest_multiple(args.tickers, cfg, years=args.years, verbose=args.verbose)
    _print_results(result)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved → {args.out}")