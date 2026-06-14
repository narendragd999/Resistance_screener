"""
9EMA Breakout + Higher High + Undervalued Screener — FastAPI Router
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Strategy:
  1. Current price must be UNDERVALUED vs SMA fair value (composite_gain_pct > 0)
  2. Stock must have broken out above 9 EMA (a candle closed above 9EMA after being below)
  3. A subsequent candle formed a Higher High (close > previous candle's high = confirmation)
  4. That confirmation candle had HIGHER VOLUME than the breakout candle
  5. Stock is now near 9 EMA (within `retest_pct` %) → BUY ZONE for retest entry
  
Routes:
  GET  /api/ema9/tickers?q=          → autocomplete
  GET  /api/ema9/tickers/list        → full list
  POST /api/ema9/screen              → batch screener
  GET  /api/ema9/detail/{ticker}     → single ticker full detail
"""

import os, re, asyncio, datetime as dt_module
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

DATA_DIR = "data"
FNO_CSV  = "tickers.csv"
ALL_CSV  = "tickers_all.csv"
os.makedirs(DATA_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
#  TICKER LOADING  (mirrors sma_router pattern exactly)
# ─────────────────────────────────────────────────────────────
_fno_df: Optional[pd.DataFrame] = None
_all_df: Optional[pd.DataFrame] = None


def _load_csv_df(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        return pd.DataFrame(columns=["symbol", "company_name"])
    try:
        df = pd.read_csv(csv_path, dtype=str)
        df.columns = [c.strip().lower() for c in df.columns]
        if "name of company" in df.columns:
            df = df.rename(columns={"name of company": "company_name"})
        elif "security" in df.columns:
            df = df.rename(columns={"security": "company_name"})
        elif "company_name" not in df.columns:
            df["company_name"] = ""
        if "symbol" not in df.columns:
            return pd.DataFrame(columns=["symbol", "company_name"])
        df["symbol"]       = df["symbol"].str.strip().str.upper()
        df["company_name"] = df.get("company_name", pd.Series([""] * len(df))).fillna("").str.strip()
        out = df[["symbol", "company_name"]].dropna(subset=["symbol"])
        return out[out["symbol"] != ""].reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=["symbol", "company_name"])


def _load_fno_df() -> pd.DataFrame:
    global _fno_df
    if _fno_df is None:
        _fno_df = _load_csv_df(FNO_CSV)
    return _fno_df


def _load_all_df() -> pd.DataFrame:
    global _all_df
    if _all_df is None:
        _all_df = _load_csv_df(ALL_CSV)
    return _all_df


# ─────────────────────────────────────────────────────────────
#  EMA HELPER
# ─────────────────────────────────────────────────────────────
def _calc_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


# ─────────────────────────────────────────────────────────────
#  FAIR VALUE  (simple 52w SMA-based fair value for speed)
#  For full screener speed we use a fast heuristic:
#    fair_value = 200-day SMA of closing price
#  Current price < fair_value  →  UNDERVALUED
#  (Deep value via regression is done in sma_router; this
#   screener prioritises speed for batch scanning)
# ─────────────────────────────────────────────────────────────
def _get_fair_value(df: pd.DataFrame) -> Optional[float]:
    """200-day SMA as proxy fair value for fast batch screening."""
    closes = df["Close"].dropna()
    if len(closes) < 50:
        return None
    # Use 200-day SMA if enough data, else 100-day
    period = 200 if len(closes) >= 200 else 100
    return round(float(closes.rolling(period).mean().iloc[-1]), 2)


# ─────────────────────────────────────────────────────────────
#  CORE SCREENER LOGIC
# ─────────────────────────────────────────────────────────────
def _screen_ticker(
    ticker: str,
    interval: str = "1wk",
    lookback_days: int = 365,
    retest_pct: float = 3.0,
    min_breakout_candles_ago: int = 1,
    max_breakout_candles_ago: int = 8,
) -> Dict:
    """
    Returns a dict with screening result for one ticker.

    Signal conditions (ALL must be true):
      A) current_price < fair_value  (undervalued)
      B) A breakout candle exists: candle that closed ABOVE 9EMA after being below
      C) Next candle after breakout: close > breakout_candle.high  (Higher High)
      D) Confirmation candle volume > breakout candle volume
      E) Current price is within retest_pct% of 9EMA  (buy zone)
    """
    ticker = ticker.strip().upper()
    yf_symbol = f"{ticker}.NS"

    try:
        end   = dt_module.date.today()
        start = end - dt_module.timedelta(days=lookback_days + 60)
        raw   = yf.download(yf_symbol, start=str(start), end=str(end),
                            interval=interval, progress=False, auto_adjust=True)
    except Exception as exc:
        return {"ticker": ticker, "status": "ERROR", "error": str(exc)}

    if raw is None or raw.empty or len(raw) < 20:
        return {"ticker": ticker, "status": "NO_DATA", "error": "Insufficient price data"}

    # Flatten MultiIndex columns if present
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy().dropna()

    if len(df) < 15:
        return {"ticker": ticker, "status": "NO_DATA", "error": "Not enough candles"}

    # ── Compute 9 EMA ──
    df["ema9"] = _calc_ema(df["Close"], 9)
    df["ema21"] = _calc_ema(df["Close"], 21)

    # ── Fair value ──
    fair_value = _get_fair_value(df)
    current_price = round(float(df["Close"].iloc[-1]), 2)
    current_ema9  = round(float(df["ema9"].iloc[-1]), 2)

    # Condition A: Undervalued
    if fair_value is None:
        return {"ticker": ticker, "status": "NO_DATA", "error": "Cannot compute fair value"}

    undervalued = current_price < fair_value
    upside_pct  = round((fair_value - current_price) / current_price * 100, 2)

    # ── Scan for breakout + higher high pattern ──
    # We look back max_breakout_candles_ago candles for the MOST RECENT valid setup
    n = len(df)
    breakout_found   = False
    breakout_idx     = None
    confirm_idx      = None
    breakout_candle  = None
    confirm_candle   = None

    # Scan from recent to older (skip last candle = current)
    scan_range = range(
        max(1, n - max_breakout_candles_ago - 1),
        n - min_breakout_candles_ago
    )

    for i in reversed(list(scan_range)):
        if i < 1:
            continue
        prev  = df.iloc[i - 1]
        curr  = df.iloc[i]
        curr_ema = df["ema9"].iloc[i]
        prev_ema = df["ema9"].iloc[i - 1]

        # Breakout: previous close BELOW ema9, current close ABOVE ema9
        prev_below = float(prev["Close"]) < float(prev_ema)
        curr_above = float(curr["Close"]) > float(curr_ema)

        if not (prev_below and curr_above):
            continue

        # Need at least one candle after breakout for confirmation
        if i + 1 >= n:
            continue

        conf = df.iloc[i + 1]
        # Higher High: confirm close > breakout candle HIGH
        higher_high = float(conf["Close"]) > float(curr["High"])
        # Higher Volume
        higher_vol  = float(conf["Volume"]) > float(curr["Volume"]) if float(curr["Volume"]) > 0 else False

        if higher_high and higher_vol:
            breakout_found  = True
            breakout_idx    = i
            confirm_idx     = i + 1
            breakout_candle = curr
            confirm_candle  = conf
            break   # most recent valid setup

    if not breakout_found:
        return {
            "ticker":        ticker,
            "status":        "NO_SIGNAL",
            "current_price": current_price,
            "fair_value":    fair_value,
            "upside_pct":    upside_pct,
            "undervalued":   undervalued,
            "ema9":          current_ema9,
            "error":         "No breakout+HH+vol pattern found",
        }

    # Condition E: Is current price near 9 EMA (retest zone)?
    ema9_distance_pct = round(abs(current_price - current_ema9) / current_ema9 * 100, 2)
    in_retest_zone    = ema9_distance_pct <= retest_pct

    # ── Build candle series for chart (last 60 candles) ──
    chart_start = max(0, n - 60)
    candles = []
    for i in range(chart_start, n):
        row = df.iloc[i]
        candles.append({
            "date":   str(df.index[i].date()) if hasattr(df.index[i], "date") else str(df.index[i])[:10],
            "open":   round(float(row["Open"]),  2),
            "high":   round(float(row["High"]),  2),
            "low":    round(float(row["Low"]),   2),
            "close":  round(float(row["Close"]), 2),
            "volume": int(row["Volume"]),
            "ema9":   round(float(df["ema9"].iloc[i]),  2),
            "ema21":  round(float(df["ema21"].iloc[i]), 2),
        })

    # Mark breakout and confirm candles in chart
    bo_date  = str(df.index[breakout_idx].date()) if hasattr(df.index[breakout_idx], "date") else str(df.index[breakout_idx])[:10]
    con_date = str(df.index[confirm_idx].date())  if hasattr(df.index[confirm_idx],  "date") else str(df.index[confirm_idx])[:10]
    for c in candles:
        c["is_breakout"] = (c["date"] == bo_date)
        c["is_confirm"]  = (c["date"] == con_date)

    # ── Profit targets (3%, 4%, 5%) from current price ──
    targets = {
        "t3pct": round(current_price * 1.03, 2),
        "t4pct": round(current_price * 1.04, 2),
        "t5pct": round(current_price * 1.05, 2),
    }

    # Final signal grade
    if not undervalued:
        signal = "PARTIAL"   # pattern ok but not undervalued
    elif not in_retest_zone:
        signal = "WATCH"     # undervalued + pattern, but not in retest zone yet
    else:
        signal = "BUY_ZONE"  # all conditions met

    return {
        "ticker":              ticker,
        "status":              "SIGNAL",
        "signal":              signal,
        "current_price":       current_price,
        "fair_value":          fair_value,
        "upside_pct":          upside_pct,
        "undervalued":         undervalued,
        "ema9":                current_ema9,
        "ema9_distance_pct":   ema9_distance_pct,
        "in_retest_zone":      in_retest_zone,
        "breakout_date":       bo_date,
        "breakout_close":      round(float(breakout_candle["Close"]), 2),
        "breakout_high":       round(float(breakout_candle["High"]),  2),
        "breakout_volume":     int(breakout_candle["Volume"]),
        "confirm_date":        con_date,
        "confirm_close":       round(float(confirm_candle["Close"]), 2),
        "confirm_volume":      int(confirm_candle["Volume"]),
        "candles_ago":         n - 1 - confirm_idx,
        "targets":             targets,
        "candles":             candles,
        "interval":            interval,
    }


# ─────────────────────────────────────────────────────────────
#  PYDANTIC REQUEST MODELS
# ─────────────────────────────────────────────────────────────
class Ema9ScreenRequest(BaseModel):
    tickers:                  List[str]
    interval:                 str   = "1wk"    # 1d, 1wk
    lookback_days:            int   = 365
    retest_pct:               float = 3.0      # % from 9EMA to qualify as retest zone
    max_breakout_candles_ago: int   = 8        # how far back to look for breakout
    min_breakout_candles_ago: int   = 1


class Ema9DetailRequest(BaseModel):
    interval:      str   = "1wk"
    lookback_days: int   = 365
    retest_pct:    float = 3.0


# ─────────────────────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────────────────────
@router.get("/api/ema9/tickers")
async def ema9_tickers(q: str = "", source: str = "fno"):
    df = _load_all_df() if source == "all" else _load_fno_df()
    if df.empty:
        return {"tickers": [], "total": 0}
    q = q.strip().upper()
    if q:
        mask = df["symbol"].str.contains(q, na=False)
        if "company_name" in df.columns:
            mask = mask | df["company_name"].str.upper().str.contains(q, na=False)
        filtered = df[mask].head(30)
    else:
        filtered = df.head(30)
    return {
        "total": len(df),
        "tickers": [
            {"symbol": row["symbol"], "name": row["company_name"] or row["symbol"]}
            for _, row in filtered.iterrows()
        ],
    }


@router.get("/api/ema9/tickers/list")
async def ema9_tickers_list(source: str = "fno"):
    df = _load_all_df() if source == "all" else _load_fno_df()
    return {
        "source":  source,
        "total":   len(df),
        "symbols": df["symbol"].tolist(),
        "tickers": [
            {"symbol": row["symbol"], "name": row["company_name"] or row["symbol"]}
            for _, row in df.iterrows()
        ],
    }


@router.post("/api/ema9/screen")
async def ema9_screen(req: Ema9ScreenRequest):
    tickers = [t.strip().upper() for t in req.tickers if t.strip()][:100]
    if not tickers:
        raise HTTPException(400, "No tickers provided.")

    results, failed, no_signal = [], [], []

    async def _run(ticker):
        return await asyncio.to_thread(
            _screen_ticker,
            ticker,
            req.interval,
            req.lookback_days,
            req.retest_pct,
            req.min_breakout_candles_ago,
            req.max_breakout_candles_ago,
        )

    tasks = [_run(t) for t in tickers]
    raw   = await asyncio.gather(*tasks, return_exceptions=True)

    for res in raw:
        if isinstance(res, Exception):
            failed.append({"ticker": "?", "error": str(res)})
        elif res["status"] in ("ERROR", "NO_DATA"):
            failed.append({"ticker": res["ticker"], "error": res.get("error","")})
        elif res["status"] == "NO_SIGNAL":
            no_signal.append(res)
        else:
            results.append(res)

    # Sort: BUY_ZONE first, then WATCH, then PARTIAL
    order = {"BUY_ZONE": 0, "WATCH": 1, "PARTIAL": 2}
    results.sort(key=lambda r: (order.get(r.get("signal",""), 9), -r.get("upside_pct", 0)))

    return {
        "results":   results,
        "no_signal": no_signal,
        "failed":    failed,
        "count":     len(results),
        "interval":  req.interval,
    }


@router.get("/api/ema9/detail/{ticker}")
async def ema9_detail(ticker: str, interval: str = "1wk",
                      lookback_days: int = 365, retest_pct: float = 3.0):
    res = await asyncio.to_thread(
        _screen_ticker, ticker.strip().upper(), interval, lookback_days, retest_pct
    )
    if res["status"] in ("ERROR", "NO_DATA"):
        raise HTTPException(404, res.get("error", "No data"))
    return res
