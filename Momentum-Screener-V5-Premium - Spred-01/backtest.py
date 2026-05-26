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
    # Expiry control ─────────────────────────────────────────────────────────
    "expiry_mode":          "auto",
    "min_days_to_expiry":   5,
    # Premium Surge Analysis
    "premium_vol_surge_threshold": 1.5,
    # ── NEW: Accuracy-improvement filters ────────────────────────────────────
    # 1. Bear Call Spread hedge — buy leg N intervals above short CE
    #    Set to 0 to disable BCS (naked CE sell only)
    "bcs_width_intervals":        1,
    # 2. Stop-loss multiplier — exit if live CE premium >= entry × multiplier
    #    Set to 0 to disable stop-loss (hold to expiry)
    "stop_loss_multiplier":       2.5,
    # 3. Minimum CE premium to enter — skip trades where premium is too thin
    "min_ce_premium":             2.0,
    # 4. Minimum IV% to enter — skip low-IV environments (cheap premiums)
    "min_iv_pct":                 15.0,
    # 5. Minimum DTE to enter (overrides expiry-mode roll for very short windows)
    "min_dte":                    7,
    # 6. Minimum surge gain for high-conviction filter
    "min_surge_for_highconv":     25.0,
    # 7. Lifetime-high filter — skip if stock is within X% of 52-week high
    "lifetime_high_filter":       True,
    "lifetime_high_buffer_pct":   2.0,
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


def lifetime_high_in_window(highs: np.ndarray, lookback: int = 252) -> float:
    """Max high over last `lookback` candles (or all if fewer)."""
    window = highs[-lookback:] if len(highs) >= lookback else highs
    return float(np.max(window)) if len(window) > 0 else 0.0


def simulate_stop_loss(
    closes: np.ndarray, dates: pd.DatetimeIndex,
    g3_idx: int, expiry_date: date,
    entry_price: float, strike: float, sigma: float,
    entry_ce_premium: float, stop_multiplier: float,
    risk_free_rate: float = RISK_FREE_RATE,
) -> Optional[Tuple[str, float, float]]:
    """
    Simulate intraday stop-loss check day-by-day between g3_idx and expiry.
    Each day recompute the BS CE price; if it exceeds entry × stop_multiplier,
    exit immediately at that day's estimated CE price (stop triggered).

    Returns (exit_type, exit_ce_price, exit_date_str) or None if no stop triggered.
      exit_type: "STOP"
    """
    if stop_multiplier <= 0:
        return None  # stop-loss disabled
    stop_level = entry_ce_premium * stop_multiplier
    exp_dates  = [d.date() for d in dates]

    for i in range(g3_idx + 1, len(closes)):
        d = dates[i].date()
        if d > expiry_date:
            break
        T = max(0.001, (expiry_date - d).days / 365.0)
        spot      = float(closes[i])
        ce_today  = bs_call(spot, strike, T, risk_free_rate, sigma)
        if ce_today >= stop_level:
            return ("STOP", round(ce_today, 2), str(d))
    return None


def bcs_pnl(
    entry_short_ce: float, exit_short_ce: float,
    entry_long_ce: float, exit_long_ce: float,
) -> float:
    """
    Bear Call Spread P&L per unit.
    SELL short_ce, BUY long_ce (hedge). Both exit at settlement (or stop).
    Net credit  = short_entry - long_entry
    Net debit   = short_exit  - long_exit
    P&L         = (short_entry - long_entry) - (short_exit - long_exit)
    """
    net_credit = entry_short_ce - entry_long_ce
    net_debit  = exit_short_ce  - exit_long_ce
    return round(net_credit - net_debit, 2)


def diagnose_loss(trade: Dict, cfg: Dict) -> Dict:
    """
    Analyse a LOSS trade and return:
      - loss_reason: primary category of loss
      - rescue_action: specific actionable improvement
      - improvement_if_applied: estimated P&L improvement (qualitative)
    """
    surge_gain  = trade.get("surge_gain_pct", 0)
    iv          = trade.get("iv_sigma", 0)
    dte         = trade.get("T_days", 0)
    vol_g3      = trade.get("vol_surge_at_g3") or 0
    prem        = trade.get("entry_ce_premium", 0)
    intrinsic   = trade.get("intrinsic_value", 0)
    entry_price = trade.get("entry_price", 0)
    strike      = trade.get("strike", 0)
    surge_high  = trade.get("surge_high", 0)
    pnl         = trade.get("pnl_per_unit", 0)

    # ── Rule 1: Breakout loss — stock surged well above strike ──────────
    overshoot = entry_price - strike if entry_price > strike else 0
    breakout_pct = (overshoot / strike * 100) if strike > 0 else 0
    if intrinsic > prem * 1.5:
        reason = "FULL_BREAKOUT"
        rescue = (
            f"Use BCS: sell ₹{int(strike)} CE + buy ₹{int(strike + nse_strike_interval(strike))} CE. "
            f"Caps max loss to spread width. Also consider skipping Gate 3 entries "
            f"when vol@G3 < 1.2x (was {vol_g3:.1f}x) — low-vol retests lack conviction."
        )
    # ── Rule 2: Low DTE — time ran out before thesis played ─────────────
    elif dte < 8:
        reason = "LOW_DTE"
        rescue = (
            f"Only {dte}d to expiry at entry. Switch expiry_mode to 'next' for entries "
            f"with < 10 DTE. Next-month expiry gives price more time to stay below strike."
        )
    # ── Rule 3: Low IV — premium too thin to absorb loss ────────────────
    elif iv < 18:
        reason = "LOW_IV"
        rescue = (
            f"IV was only {iv:.1f}% — premium ₹{prem:.2f} is too thin. "
            f"Apply min_iv_pct=20% filter to skip low-vol environments. "
            f"This trade would have been skipped."
        )
    # ── Rule 4: Weak surge — setup lacks momentum exhaustion ────────────
    elif surge_gain < cfg.get("min_surge_for_highconv", 25.0):
        reason = "WEAK_SURGE"
        rescue = (
            f"Surge was only {surge_gain:.1f}% (threshold {cfg.get('min_surge_for_highconv',25):.0f}%). "
            f"Weak surges = weaker resistance. Raise min_gain_percent to filter out marginal setups."
        )
    # ── Rule 5: Low vol at G3 — retest lacked conviction ────────────────
    elif vol_g3 < 1.0:
        reason = "LOW_VOL_RETEST"
        rescue = (
            f"Vol@G3 = {vol_g3:.2f}x (below 1.0x average) — price drifted into sell zone "
            f"without conviction. Apply premium_vol_surge_threshold >= 1.2 to skip these."
        )
    # ── Rule 6: Near-lifetime-high — no real overhead resistance ────────
    elif (surge_high / entry_price - 1) < 0.01:
        reason = "NEAR_LIFETIME_HIGH"
        rescue = (
            f"Entry ₹{entry_price:.2f} was within 1% of surge high ₹{surge_high:.2f}. "
            f"Stock was near lifetime high — CE selling is risky when there is no overhead resistance. "
            f"Apply lifetime_high_filter=True to skip these setups."
        )
    else:
        reason = "MARKET_OVERRUN"
        rescue = (
            f"General market momentum overran resistance. Mitigation: "
            f"(1) Use BCS to cap max loss. "
            f"(2) Set stop_loss_multiplier=2.5× to exit early. "
            f"(3) Consider partial position sizing when IV < 25%."
        )

    return {
        "loss_reason":   reason,
        "rescue_action": rescue,
    }


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
    prox_pct = float(cfg.get("price_proximity_percent", 1.0))
    floor    = surge_high * (1 - prox_pct / 100)

    pulled_back = False   # True once close has dropped below floor after Gate 2

    # Start from g2_idx+1: Gate 2 candle itself is never a valid entry.
    # elif prevents Phase 1 + Phase 2 firing on the same candle.
    for i in range(g2_idx + 1, len(closes)):
        close = float(closes[i])
        high  = float(highs[i])

        # Breakout confirmed by CLOSE -> thesis dead, stop scanning
        if close > surge_high:
            return None

        # Phase 1: mark pullback when close falls below the floor
        elif not pulled_back and close < floor:
            pulled_back = True

        # Phase 2: retest — HIGH re-enters the sell zone AFTER confirmed pullback
        elif pulled_back and high >= floor:
            return i

    return None


# ─────────────────────────────────────────────────────────────────────────────
#  TRADE EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_trade(
    closes: np.ndarray, dates: pd.DatetimeIndex,
    g3_idx: int, surge_high: float, cfg: Dict,
    vols: Optional[np.ndarray] = None,
) -> Optional[Dict]:
    """
    At Gate 3 trigger (day g3_idx):
      1. Compute strike = nearest OTM above surge_high
      2. Estimate CE premium via Black-Scholes
      3. Find monthly NSE expiry
      4. Look up closing price on or after expiry date
      5. Determine WIN / LOSS and P&L
      6. Compute volume surge at Gate 3 (vol / 20d avg) and Premium Score
         = vol_surge_at_g3 x entry_ce_premium — higher score means the
         retest happened with elevated volume AND rich premium, flagging
         the highest-conviction sell setups.
    """
    entry_price  = float(closes[g3_idx])
    entry_date   = dates[g3_idx].date()

    # --- Strike ---
    strike = nearest_otm_strike(surge_high)

    # --- Time to expiry (respects expiry_mode config) ---
    expiry_mode   = str(cfg.get("expiry_mode", "auto")).lower()
    min_days_roll = int(cfg.get("min_days_to_expiry", 5))

    def _next_expiry(d: date) -> date:
        nm = d.month + 1 if d.month < 12 else 1
        ny = d.year  + (1 if d.month == 12 else 0)
        return nse_monthly_expiry(ny, nm)

    if expiry_mode == "next":
        expiry_date = _next_expiry(entry_date)
    elif expiry_mode == "current":
        expiry_date = nse_monthly_expiry(entry_date.year, entry_date.month)
    else:   # "auto" (default)
        expiry_date = nse_monthly_expiry(entry_date.year, entry_date.month)
        if (expiry_date - entry_date).days <= min_days_roll:
            expiry_date = _next_expiry(entry_date)

    T_years = max(0.001, (expiry_date - entry_date).days / 365.0)
    dte     = (expiry_date - entry_date).days

    # ── NEW: Hard DTE filter ─────────────────────────────────────────────────
    min_dte = int(cfg.get("min_dte", 7))
    if dte < min_dte:
        return None   # not enough time — skip trade

    # --- Historical volatility at entry ---
    sigma = hist_vol(closes[:g3_idx + 1], VOL_WINDOW)
    iv_pct = sigma * 100

    # ── NEW: IV filter — skip if implied/historical vol is too low ───────────
    min_iv = float(cfg.get("min_iv_pct", 15.0))
    if iv_pct < min_iv:
        return None   # premium too thin to be worth selling

    # --- Black-Scholes CE premium at entry ---
    entry_ce_premium = round(bs_call(entry_price, strike, T_years, RISK_FREE_RATE, sigma), 2)
    if entry_ce_premium < max(0.01, float(cfg.get("min_ce_premium", 2.0))):
        return None   # negligible premium — skip

    # ── NEW: Bear Call Spread hedge leg ─────────────────────────────────────
    bcs_width    = int(cfg.get("bcs_width_intervals", 1))
    hedge_strike = None
    entry_hedge_ce = 0.0
    bcs_net_credit = None
    bcs_max_profit = None
    bcs_max_loss   = None
    bcs_pnl_val    = None
    bcs_result     = None

    if bcs_width > 0:
        interval     = nse_strike_interval(strike)
        hedge_strike = strike + bcs_width * interval
        entry_hedge_ce = round(bs_call(entry_price, hedge_strike, T_years, RISK_FREE_RATE, sigma), 2)
        if entry_hedge_ce > 0:
            bcs_net_credit = round(entry_ce_premium - entry_hedge_ce, 2)
            bcs_max_loss   = round(bcs_width * interval - bcs_net_credit, 2)
            bcs_max_profit = bcs_net_credit

    # --- Find expiry-day close (or nearest available date) ---
    exp_dates = [d.date() for d in dates]
    exp_idx   = None
    for search_offset in range(0, 5):
        target = expiry_date + timedelta(days=search_offset)
        if target in exp_dates:
            exp_idx = exp_dates.index(target)
            break

    if exp_idx is None or exp_idx >= len(closes):
        return None

    expiry_close  = float(closes[exp_idx])
    actual_expiry = dates[exp_idx].date()

    # ── NEW: Stop-loss simulation ────────────────────────────────────────────
    stop_mult  = float(cfg.get("stop_loss_multiplier", 2.5))
    stop_event = simulate_stop_loss(
        closes, dates, g3_idx, expiry_date,
        entry_price, strike, sigma, entry_ce_premium, stop_mult,
    )

    # --- P&L (base: naked CE sell) ---
    intrinsic_at_expiry = max(0.0, expiry_close - strike)
    pnl_per_lot_raw     = entry_ce_premium - intrinsic_at_expiry
    result              = "WIN" if expiry_close < strike else "LOSS"

    # Apply stop-loss if triggered BEFORE expiry
    stop_triggered = False
    stop_exit_price = None
    stop_exit_date  = None
    if stop_event is not None:
        _, stop_exit_price, stop_exit_date = stop_event
        pnl_per_lot_raw = entry_ce_premium - stop_exit_price  # exit at stop
        if pnl_per_lot_raw < 0:
            result = "LOSS"
            stop_triggered = True
        else:
            result = "WIN"   # stopped out profitably (rare but possible)
        intrinsic_at_expiry = stop_exit_price  # use for display

    # Max loss cap at 3× premium (reasonable for sold CE)
    pnl = max(pnl_per_lot_raw, -3 * entry_ce_premium)

    # BCS settlement P&L
    if hedge_strike is not None and entry_hedge_ce > 0:
        if stop_triggered and stop_exit_price is not None:
            # Exit both legs at stop
            exit_hedge_ce = round(bs_call(
                float(closes[exp_dates.index(date.fromisoformat(stop_exit_date))] if stop_exit_date in [str(d) for d in exp_dates] else closes[g3_idx]),
                hedge_strike, max(0.001, (expiry_date - date.fromisoformat(stop_exit_date)).days / 365),
                RISK_FREE_RATE, sigma,
            ), 2)
            bcs_pnl_val = bcs_pnl(entry_ce_premium, stop_exit_price, entry_hedge_ce, exit_hedge_ce)
        else:
            expiry_hedge_intrinsic = max(0.0, expiry_close - hedge_strike)
            bcs_pnl_val = bcs_pnl(entry_ce_premium, intrinsic_at_expiry, entry_hedge_ce, expiry_hedge_intrinsic)
        bcs_result = "WIN" if bcs_pnl_val >= 0 else "LOSS"

    # ── Premium Surge metrics at Gate 3 ──────────────────────────────────────
    vol_surge_at_g3  = None
    premium_score    = None
    vol_at_g3_raw    = None
    avg_vol_20d_g3   = None

    if vols is not None and len(vols) > g3_idx:
        vol_at_g3_raw  = float(vols[g3_idx])
        window         = vols[max(0, g3_idx - 20):g3_idx]
        avg_vol_20d_g3 = float(window.mean()) if len(window) > 0 else None
        if avg_vol_20d_g3 and avg_vol_20d_g3 > 0:
            vol_surge_at_g3 = round(vol_at_g3_raw / avg_vol_20d_g3, 2)
            premium_score   = round(vol_surge_at_g3 * entry_ce_premium, 2)

    threshold    = float(cfg.get("premium_vol_surge_threshold", 1.5))
    is_high_prem = bool(vol_surge_at_g3 is not None and vol_surge_at_g3 >= threshold)

    trade_dict = {
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
        "iv_sigma":         round(iv_pct, 1),
        "T_days":           dte,
        # Outcome (naked CE sell)
        "expiry_mode":      expiry_mode,
        "expiry_label":     expiry_mode,
        "result":           result,
        "pnl_per_unit":     round(pnl, 2),
        "return_pct":       round(pnl / entry_ce_premium * 100, 1) if entry_ce_premium > 0 else 0,
        # Stop-loss fields
        "stop_triggered":   stop_triggered,
        "stop_exit_price":  stop_exit_price,
        "stop_exit_date":   stop_exit_date,
        # BCS fields
        "bcs_hedge_strike":  hedge_strike,
        "bcs_net_credit":    bcs_net_credit,
        "bcs_max_loss":      bcs_max_loss,
        "bcs_pnl":           bcs_pnl_val,
        "bcs_result":        bcs_result,
        # Premium Surge
        "vol_at_g3":        int(vol_at_g3_raw) if vol_at_g3_raw is not None else None,
        "avg_vol_20d_g3":   int(avg_vol_20d_g3) if avg_vol_20d_g3 is not None else None,
        "vol_surge_at_g3":  vol_surge_at_g3,
        "premium_score":    premium_score,
        "is_high_prem":     is_high_prem,
    }

    # ── Loss diagnosis ───────────────────────────────────────────────────────
    if result == "LOSS":
        diagnosis = diagnose_loss({**trade_dict, "surge_gain_pct": 0}, cfg)
        trade_dict.update(diagnosis)

    return trade_dict


# ─────────────────────────────────────────────────────────────────────────────
#  DUAL-EXPIRY HELPER
# ─────────────────────────────────────────────────────────────────────────────
def _expiry_side(
    closes: np.ndarray, dates: pd.DatetimeIndex,
    g3_idx: int, surge_high: float, base_cfg: Dict,
    forced_mode: str,
    vols: Optional[np.ndarray] = None,
) -> Optional[Dict]:
    """
    Evaluate the CE sell trade for a single specific expiry month.
    Returns a compact dict with only the expiry-specific fields.
    Returns None if there is no historical data at/after that expiry.
    """
    cfg_override = {**base_cfg, "expiry_mode": forced_mode}
    t = evaluate_trade(closes, dates, g3_idx, surge_high, cfg_override, vols=vols)
    if t is None:
        return None
    return {
        "expiry_date":       t["expiry_date"],
        "actual_data_date":  t["actual_data_date"],
        "T_days":            t["T_days"],
        "entry_ce_premium":  t["entry_ce_premium"],
        "expiry_close":      t["expiry_close"],
        "intrinsic_value":   t["intrinsic_value"],
        "result":            t["result"],
        "pnl_per_unit":      t["pnl_per_unit"],
        "return_pct":        t["return_pct"],
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

    OPTIMISED: O(N × recency) instead of the original O(N³).

    Old approach:
      for every today_idx (≈440):            ← O(N)
        detect_breakdown():
          for wsize (up to 440):             ← O(N)
            for start (up to 440):           ← O(N)
              check_surge_continuity()       ← O(wsize) each
    → ~850 million Python iterations for RELIANCE 2-year run

    New approach:
      Step 1 — 3 linear precompute passes    ← O(N)
      Step 2 — one-pass breakdown scan,
               surge search only for the
               ~10-20 actual breakdowns      ← O(N + breaks × recency²)
      Step 3 — walk-forward = dict lookups   ← O(N)
    → ~50,000 iterations; RELIANCE < 5 s
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
    n      = len(closes)

    print(f"[{ticker}] {n} candles from {dates[0].date()} to {dates[-1].date()}")

    # ── Config ───────────────────────────────────────────────────────────────
    min_drop      = float(cfg.get("min_drop_percent", 0.1))
    min_vol_ratio = float(cfg.get("min_breakdown_volume_ratio", 0.5))
    recency       = int(cfg.get("surge_recency_days", 45))
    min_gain      = float(cfg.get("min_gain_percent", 18.0))
    min_green_cfg = int(cfg.get("min_green_candles", 2))
    ema_period    = int(cfg.get("ema_period", 9))

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 1 — Precompute helper arrays  (O(N) total, done ONCE)
    # ═════════════════════════════════════════════════════════════════════════

    # 1a. Continuity DP
    #     earliest_start[i] = farthest back we can trace a continuous surge
    #     ending at candle i, where "continuous" means no candle in the
    #     window closes below the previous candle's low.
    #     This single array replaces the inner check_surge_continuity loop.
    earliest_start = np.empty(n, dtype=np.intp)
    earliest_start[0] = 0
    for i in range(1, n):
        earliest_start[i] = earliest_start[i - 1] if closes[i] >= lows[i - 1] else i

    # 1b. Prefix sum of green candles
    #     green_count(start, end) = green_prefix[end+1] - green_prefix[start]
    #     Turns the green-candle count from O(window) to O(1).
    green_prefix = np.zeros(n + 1, dtype=np.int32)
    green_prefix[1:] = np.cumsum(closes > opens)

    # 1c. Rolling 20-day average volume via cumsum  → O(N), no per-candle loop
    vol_cs = np.empty(n + 1)
    vol_cs[0] = 0.0
    np.cumsum(vols, out=vol_cs[1:])
    avg_vol_20 = np.empty(n)
    for i in range(n):
        j = max(0, i - 20)
        cnt = i - j
        avg_vol_20[i] = (vol_cs[i] - vol_cs[j]) / cnt if cnt > 0 else 0.0

    # 1d. Full-series EMA — computed ONCE instead of once-per-breakdown
    ema_full = compute_ema(closes, ema_period)

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 2 — One-pass breakdown detection + surge window search
    #
    # Old code called detect_breakdown() for every today_idx (440 calls).
    # Each call ran an O(n²) double loop + O(n) check_surge_continuity →
    # the SAME breakdown candle triggered the expensive surge search 45×
    # (once per today_idx within lookback window).
    #
    # New: scan candles once; the O(recency²) surge search only fires for
    # the ~10-20 genuine breakdown candles found in a 2-year history.
    # ═════════════════════════════════════════════════════════════════════════
    precomputed_signals: Dict[int, Dict] = {}
    seen_bd_dates: set = set()

    for bd_idx in range(MIN_WARMUP, n):
        # ── Breakdown candle check (O(1) per candle) ──────────────────────
        if closes[bd_idx] >= lows[bd_idx - 1]:
            continue
        drop = (lows[bd_idx - 1] - closes[bd_idx]) / lows[bd_idx - 1] * 100
        if drop < min_drop:
            continue
        if avg_vol_20[bd_idx] <= 0:
            continue
        v_ratio = vols[bd_idx] / avg_vol_20[bd_idx]
        if v_ratio < min_vol_ratio:
            continue

        bd_date_str = str(dates[bd_idx].date())
        if bd_date_str in seen_bd_dates:
            continue
        seen_bd_dates.add(bd_date_str)

        # ── Surge window search — O(recency²) but only ~10-20 times total ─
        #    For each possible surge end (within recency before breakdown),
        #    use earliest_start[] to skip non-continuous windows entirely,
        #    and green_prefix[] for O(1) green-candle counts.
        search_start = max(0, bd_idx - recency)
        best_gain    = 0.0
        best_window  = None
        best_greens  = 0

        for end_idx in range(bd_idx - 1, search_start - 1, -1):
            cont_start  = int(earliest_start[end_idx])
            valid_start = max(cont_start, search_start)
            if valid_start >= end_idx:
                continue   # no room for a multi-candle window

            for start_idx in range(valid_start, end_idx):
                gain = (closes[end_idx] - closes[start_idx]) / closes[start_idx] * 100
                if gain < min_gain:
                    continue
                gc = int(green_prefix[end_idx + 1] - green_prefix[start_idx])
                if gc < min_green_cfg:
                    continue
                if gain > best_gain:
                    best_gain   = gain
                    best_window = (start_idx, end_idx)
                    best_greens = gc

        if best_window is None:
            continue

        surge_start_idx, surge_end_idx = best_window
        surge_high = float(np.max(highs[surge_start_idx:surge_end_idx + 1]))

        if verbose:
            print(f"  Gate 1 @ {bd_date_str}  surge_high={surge_high:.2f}  "
                  f"drop={drop:.2f}%  vol={v_ratio:.2f}x")

        precomputed_signals[bd_idx] = {
            "breakdown_idx":    bd_idx,
            "surge_high":       round(surge_high, 2),
            "yesterday_high":   round(float(highs[bd_idx - 1]), 2),
            "yesterday_low":    round(float(lows[bd_idx - 1]), 2),
            "drop_pct":         round(drop, 2),
            "volume_ratio":     round(v_ratio, 2),
            "surge_gain_pct":   round(best_gain, 2),
            "surge_candles":    best_greens,
            "surge_start_idx":  surge_start_idx,
            "surge_end_idx":    surge_end_idx,
            "ema_at_breakdown": round(float(ema_full[bd_idx]), 2),
        }

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 3 — Walk-forward trade simulation (O(N) — just dict lookups)
    #          Gate 2 uses precomputed ema_full/avg_vol_20 (no recompute).
    #          Gate 3 and trade evaluation are unchanged.
    # ═════════════════════════════════════════════════════════════════════════
    trades:          List[Dict] = []
    seen_trade_keys: set        = set()

    for bd_idx, bd_info in sorted(precomputed_signals.items()):
        surge_high  = bd_info["surge_high"]
        bd_date_str = str(dates[bd_idx].date())

        # ── Gate 2: inline with precomputed EMA — avoids recomputing full EMA
        g2_idx    = None
        g2_avgvol = float(np.mean(vols[max(0, bd_idx - 20):bd_idx])) if bd_idx > 0 else 0.0
        if g2_avgvol > 0:
            for i in range(bd_idx, n):
                if closes[i] < ema_full[i] and vols[i] / g2_avgvol >= min_vol_ratio:
                    g2_idx = i
                    break

        if g2_idx is None:
            if verbose:
                print(f"    Gate 2 NOT confirmed — skip")
            continue

        g2_date = str(dates[g2_idx].date())
        if verbose:
            print(f"    Gate 2 @ {g2_date}")

        # Recompute surge_high: max high from recency window before bd up to g2
        # (same logic as original — anchors resistance to visible peak)
        lookback_start = max(0, bd_idx - recency)
        if lookback_start < g2_idx:
            surge_high = float(np.max(highs[lookback_start:g2_idx]))

        if verbose:
            print(f"    Surge-high (pre-EMA-break, {recency}d lookback) → {surge_high:.2f}")

        # ── Gate 3 (unchanged) ───────────────────────────────────────────────
        g3_idx = check_gate3(closes, highs, g2_idx, surge_high, cfg)
        if g3_idx is None:
            if verbose:
                print(f"    Gate 3 never triggered (breakout or no retest)")
            continue

        g3_date   = str(dates[g3_idx].date())
        trade_key = (round(surge_high, 2), g3_date)
        if trade_key in seen_trade_keys:
            if verbose:
                print(f"    Duplicate trade episode — skip")
            continue
        seen_trade_keys.add(trade_key)

        if verbose:
            print(f"    Gate 3 @ {g3_date}  price={closes[g3_idx]:.2f}")

        # ── Trade evaluation (unchanged) ─────────────────────────────────────
        trade = evaluate_trade(closes, dates, g3_idx, surge_high, cfg, vols=vols)
        if trade is None:
            if verbose:
                print(f"    Trade evaluation skipped (no expiry data or zero premium)")
            continue

        cur_side = _expiry_side(closes, dates, g3_idx, surge_high, cfg, "current", vols=vols)
        nxt_side = _expiry_side(closes, dates, g3_idx, surge_high, cfg, "next",    vols=vols)

        trade.update({
            "ticker":         ticker,
            "gate1_date":     bd_date_str,
            "gate2_date":     g2_date,
            "gate3_date":     g3_date,
            "surge_gain_pct": bd_info["surge_gain_pct"],
            "surge_candles":  bd_info["surge_candles"],
            "volume_ratio":   bd_info["volume_ratio"],
            "cur_expiry":     cur_side,
            "nxt_expiry":     nxt_side,
        })
        trades.append(trade)

        if verbose:
            tag = "✅ WIN" if trade["result"] == "WIN" else "❌ LOSS"
            print(f"    {tag}  strike={trade['strike']}  "
                  f"entry_ce={trade['entry_ce_premium']}  "
                  f"expiry_close={trade['expiry_close']}  "
                  f"P&L={trade['pnl_per_unit']}")

    # ── Summary (identical structure to original) ─────────────────────────────
    total   = len(trades)
    wins    = sum(1 for t in trades if t["result"] == "WIN")
    losses  = total - wins
    avg_pnl     = round(sum(t["pnl_per_unit"] for t in trades) / total, 2) if total > 0 else 0.0
    avg_premium = round(sum(t["entry_ce_premium"] for t in trades) / total, 2) if total > 0 else 0.0

    threshold     = float(cfg.get("premium_vol_surge_threshold", 1.5))
    hi_vol_trades = [t for t in trades if (t.get("vol_surge_at_g3") or 0) >= threshold]
    lo_vol_trades = [t for t in trades if (t.get("vol_surge_at_g3") or 0) <  threshold]

    def _cohort_stats(cohort: list) -> dict:
        nc = len(cohort)
        if nc == 0:
            return {"trades": 0, "wins": 0, "losses": 0, "accuracy_pct": None,
                    "avg_premium": None, "avg_pnl": None, "avg_prem_score": None}
        w      = sum(1 for t in cohort if t["result"] == "WIN")
        prem   = round(sum(t["entry_ce_premium"] for t in cohort) / nc, 2)
        pnl    = round(sum(t["pnl_per_unit"]     for t in cohort) / nc, 2)
        scores = [t["premium_score"] for t in cohort if t.get("premium_score") is not None]
        return {
            "trades":         nc,
            "wins":           w,
            "losses":         nc - w,
            "accuracy_pct":   round(w / nc * 100, 1),
            "avg_premium":    prem,
            "avg_pnl":        pnl,
            "avg_prem_score": round(sum(scores) / len(scores), 2) if scores else None,
        }

    def _side_summary(key: str) -> Dict:
        sides = [t[key] for t in trades if t.get(key)]
        ns    = len(sides)
        if ns == 0:
            return {"trades": 0, "wins": 0, "losses": 0, "accuracy_pct": None,
                    "avg_premium": None, "avg_pnl": None, "total_pnl": None}
        w   = sum(1 for s in sides if s["result"] == "WIN")
        pnl = sum(s["pnl_per_unit"] for s in sides)
        return {
            "trades":       ns,
            "wins":         w,
            "losses":       ns - w,
            "accuracy_pct": round(w / ns * 100, 1),
            "avg_premium":  round(sum(s["entry_ce_premium"] for s in sides) / ns, 2),
            "avg_pnl":      round(pnl / ns, 2),
            "total_pnl":    round(pnl, 2),
        }

    return {
        "ticker": ticker,
        "summary": {
            "total_trades":       total,
            "wins":               wins,
            "losses":             losses,
            "accuracy_pct":       round(wins / total * 100, 1) if total > 0 else 0.0,
            "avg_pnl_per_unit":   avg_pnl,
            "avg_premium":        avg_premium,
            "total_pnl":          round(sum(t["pnl_per_unit"] for t in trades), 2),
            "premium_analysis": {
                "threshold":       threshold,
                "high_vol_cohort": _cohort_stats(hi_vol_trades),
                "normal_cohort":   _cohort_stats(lo_vol_trades),
            },
            "dual_expiry_comparison": {
                "current_month": _side_summary("cur_expiry"),
                "next_month":    _side_summary("nxt_expiry"),
            },
        },
        "trades": trades,
    }

def backtest_multiple(
    tickers: List[str], cfg: Dict, years: int = 2, verbose: bool = False
) -> Dict:
    results = []
    for tkr in tickers:
        print(f"\n{'='*60}\nBacktesting {tkr}...\n{'='*60}")
        res = backtest_ticker(tkr, cfg, years=years, verbose=verbose)
        results.append(res)

    # Aggregate — overall
    all_trades  = [t for r in results for t in r.get("trades", [])]
    total       = len(all_trades)
    wins        = sum(1 for t in all_trades if t["result"] == "WIN")
    losses      = total - wins
    accuracy    = round(wins / total * 100, 1) if total > 0 else 0.0
    total_pnl   = round(sum(t["pnl_per_unit"] for t in all_trades), 2)

    # Aggregate — Premium Surge cohort (cross-ticker)
    threshold    = float(cfg.get("premium_vol_surge_threshold", 1.5))
    hi_all  = [t for t in all_trades if (t.get("vol_surge_at_g3") or 0) >= threshold]
    lo_all  = [t for t in all_trades if (t.get("vol_surge_at_g3") or 0) <  threshold]

    def _agg_cohort(cohort: list) -> dict:
        n = len(cohort)
        if n == 0:
            return {"trades": 0, "wins": 0, "losses": 0, "accuracy_pct": None,
                    "avg_premium": None, "avg_pnl": None, "avg_prem_score": None}
        w     = sum(1 for t in cohort if t["result"] == "WIN")
        prem  = round(sum(t["entry_ce_premium"] for t in cohort) / n, 2)
        pnl   = round(sum(t["pnl_per_unit"]     for t in cohort) / n, 2)
        scores = [t["premium_score"] for t in cohort if t.get("premium_score") is not None]
        return {
            "trades":         n,
            "wins":           w,
            "losses":         n - w,
            "accuracy_pct":   round(w / n * 100, 1),
            "avg_premium":    prem,
            "avg_pnl":        pnl,
            "avg_prem_score": round(sum(scores) / len(scores), 2) if scores else None,
        }

    # Aggregate dual-expiry across all tickers
    def _agg_side(key: str) -> Dict:
        sides = [t[key] for r in results for t in r.get("trades", []) if t.get(key)]
        n = len(sides)
        if n == 0:
            return {"trades": 0, "wins": 0, "losses": 0, "accuracy_pct": None,
                    "avg_premium": None, "avg_pnl": None, "total_pnl": None}
        w   = sum(1 for s in sides if s["result"] == "WIN")
        pnl = sum(s["pnl_per_unit"] for s in sides)
        return {
            "trades":       n,
            "wins":         w,
            "losses":       n - w,
            "accuracy_pct": round(w / n * 100, 1),
            "avg_premium":  round(sum(s["entry_ce_premium"] for s in sides) / n, 2),
            "avg_pnl":      round(pnl / n, 2),
            "total_pnl":    round(pnl, 2),
        }

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
            # Premium Surge aggregate
            "premium_analysis": {
                "threshold":       threshold,
                "high_vol_cohort": _agg_cohort(hi_all),
                "normal_cohort":   _agg_cohort(lo_all),
            },
            # Dual-expiry aggregate comparison
            "dual_expiry_comparison": {
                "current_month": _agg_side("cur_expiry"),
                "next_month":    _agg_side("nxt_expiry"),
            },
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

    @router.get("/candles")
    async def get_candles(ticker: str, from_date: str, to_date: str):
        """
        Return OHLCV candles for ticker between from_date and to_date (inclusive).
        Used by the frontend to render per-trade candlestick charts.
        """
        try:
            from_d = date.fromisoformat(from_date)
            to_d   = date.fromisoformat(to_date)
            years_needed = max(1, math.ceil((to_d - from_d).days / 300) + 1)
            df = fetch_history(ticker, years=years_needed + 1)
            if df is None or df.empty:
                return JSONResponse(status_code=404, content={"error": f"No data for {ticker}"})
            candles = []
            for dt, row in df.iterrows():
                d = dt.date()
                if d < from_d or d > to_d:
                    continue
                candles.append({
                    "date":   str(d),
                    "open":   round(float(row["Open"]),   2),
                    "high":   round(float(row["High"]),   2),
                    "low":    round(float(row["Low"]),    2),
                    "close":  round(float(row["Close"]),  2),
                    "volume": int(row["Volume"]),
                })
            return JSONResponse(content={"ticker": ticker, "candles": candles})
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})


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