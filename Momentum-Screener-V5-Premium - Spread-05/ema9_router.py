"""
9EMA Breakout Screener — FastAPI Router (Simplified)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Signal (both must be true):
  1. A candle closes ABOVE 9 EMA (breakout candle)
     — previous candle must have been BELOW 9 EMA
  2. The very next candle closes ABOVE the breakout candle's close (confirmation)

Fair Value columns (from sma_router logic):
  composite_fair_price  — weighted average of OP / Sales / TTM regression models
  composite_gain_pct    — (fair - current) / current * 100
  fair_gap_pct          — alias of composite_gain_pct (positive = upside to fair value)
  valuation_bucket      — UNDERVALUED / FAIR / OVERVALUED

Routes:
  GET  /api/ema9/tickers        → autocomplete
  GET  /api/ema9/tickers/list   → full list
  POST /api/ema9/screen         → batch screener
"""

import os, asyncio, datetime as dt_module, time, math
from typing import Optional, List, Dict

import pandas as pd
import yfinance as yf
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# Import fair value logic from sma_router
try:
    from sma_router import _analyze_ticker as _sma_analyze_ticker
    _FV_AVAILABLE = True
except ImportError:
    _FV_AVAILABLE = False

router = APIRouter()

DATA_DIR = "data"
FNO_CSV  = "tickers.csv"
ALL_CSV  = "tickers_all.csv"
os.makedirs(DATA_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
#  RATE LIMIT CONFIGURATION
#  Tune these if you hit 429s or Screener.in blocks.
# ─────────────────────────────────────────────────────────────

# ── yfinance (Yahoo Finance) ──────────────────────────────────
# One batch download per chunk; Yahoo allows ~2 000 symbols/hour
# comfortably.  We cap each chunk at 100 symbols and pause between
# chunks so sustained 1 000+ scans stay well under the limit.
YF_CHUNK_SIZE   = 100    # symbols per single yf.download() call
YF_CHUNK_DELAY  = 2.0    # seconds to sleep between yf chunks
                          # → 100 sym/2 s = 50 sym/s ≈ 180 000/hr (safe)

# ── Screener.in ───────────────────────────────────────────────
# Screener.in is aggressive about rate-limiting scrapers.
# Each FV enrichment hits screener.in up to 2× (consolidated + standalone).
# 1.5 s between calls keeps it under ~40 req/min — well within tolerance.
# Results are disk-cached so re-runs within the same day are instant.
FV_INTER_DELAY  = 1.5    # seconds between consecutive FV enrichments

# ─────────────────────────────────────────────────────────────
#  TICKER LOADING
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
#  CORE SCREENER LOGIC
# ─────────────────────────────────────────────────────────────
def _process_ticker_df(ticker: str, df: pd.DataFrame, max_candles_ago: int) -> Dict:
    """
    Run EMA9 breakout logic on a pre-downloaded OHLCV DataFrame for one ticker.
    df must have columns: Open, High, Low, Close, Volume  (DatetimeIndex).
    """
    df = df.copy().dropna()
    if len(df) < 12:
        return {"ticker": ticker, "status": "NO_DATA", "error": "Not enough candles"}

    df["ema9"] = df["Close"].ewm(span=9, adjust=False).mean()

    n             = len(df)
    current_price = round(float(df["Close"].iloc[-1]), 2)
    current_ema9  = round(float(df["ema9"].iloc[-1]), 2)

    scan_start = max(1, n - max_candles_ago - 1)
    scan_end   = n - 2

    found = False
    breakout_idx = confirm_idx = None

    for i in range(scan_end, scan_start - 1, -1):
        prev_close = float(df["Close"].iloc[i - 1])
        prev_ema   = float(df["ema9"].iloc[i - 1])
        curr_close = float(df["Close"].iloc[i])
        curr_ema   = float(df["ema9"].iloc[i])
        conf_close = float(df["Close"].iloc[i + 1])

        if prev_close < prev_ema and curr_close > curr_ema and conf_close > curr_close:
            found = True
            breakout_idx = i
            confirm_idx  = i + 1
            break

    if not found:
        return {"ticker": ticker, "status": "NO_SIGNAL",
                "current_price": current_price, "ema9": current_ema9}

    breakout_candle = df.iloc[breakout_idx]
    confirm_candle  = df.iloc[confirm_idx]

    def _date(idx):
        d = df.index[idx]
        return str(d.date()) if hasattr(d, "date") else str(d)[:10]

    bo_date  = _date(breakout_idx)
    con_date = _date(confirm_idx)

    chart_start = max(0, n - 60)
    candles = []
    for j in range(chart_start, n):
        row = df.iloc[j]
        candles.append({
            "date":        str(df.index[j].date()) if hasattr(df.index[j], "date") else str(df.index[j])[:10],
            "open":        round(float(row["Open"]),  2),
            "high":        round(float(row["High"]),  2),
            "low":         round(float(row["Low"]),   2),
            "close":       round(float(row["Close"]), 2),
            "volume":      int(row["Volume"]),
            "ema9":        round(float(df["ema9"].iloc[j]), 2),
            "is_breakout": (_date(j) == bo_date),
            "is_confirm":  (_date(j) == con_date),
        })

    return {
        "ticker":         ticker,
        "status":         "SIGNAL",
        "current_price":  current_price,
        "ema9":           current_ema9,
        "ema9_dist_pct":  round(abs(current_price - current_ema9) / current_ema9 * 100, 2),
        "breakout_date":  bo_date,
        "breakout_close": round(float(breakout_candle["Close"]), 2),
        "breakout_high":  round(float(breakout_candle["High"]),  2),
        "confirm_date":   con_date,
        "confirm_close":  round(float(confirm_candle["Close"]), 2),
        "candles_ago":    n - 1 - confirm_idx,
        "interval":       "batch",
        "candles":        candles,
    }


def _batch_download(
    tickers: List[str],
    interval: str,
    lookback_days: int,
) -> Dict[str, pd.DataFrame]:
    """
    Single yf.download call for a chunk of tickers (≤ YF_CHUNK_SIZE).
    Returns {ticker: ohlcv_df}.  Caller is responsible for chunking and
    inter-chunk delays.
    """
    if not tickers:
        return {}

    yf_symbols = [f"{t}.NS" for t in tickers]
    end   = dt_module.date.today()
    start = end - dt_module.timedelta(days=lookback_days + 30)

    try:
        raw = yf.download(
            yf_symbols,
            start=str(start), end=str(end),
            interval=interval,
            group_by="ticker",
            progress=False,
            auto_adjust=True,
            threads=False,        # single-threaded inside yfinance
        )
    except Exception:
        return {}

    if raw is None or raw.empty:
        return {}

    result: Dict[str, pd.DataFrame] = {}
    for ticker, sym in zip(tickers, yf_symbols):
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                df = raw[sym][["Open", "High", "Low", "Close", "Volume"]].copy()
            else:
                # Single-symbol fallback — flat columns
                df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
            df = df.dropna(how="all")
            if len(df) >= 12:
                result[ticker] = df
        except Exception:
            continue
    return result


def _screen_ticker(
    ticker: str,
    interval: str = "1d",
    lookback_days: int = 180,
    max_candles_ago: int = 10,
) -> Dict:
    """Single-ticker fallback (kept for direct calls / testing)."""
    ticker    = ticker.strip().upper()
    yf_symbol = f"{ticker}.NS"
    try:
        end   = dt_module.date.today()
        start = end - dt_module.timedelta(days=lookback_days + 30)
        raw   = yf.download(
            yf_symbol, start=str(start), end=str(end),
            interval=interval, progress=False, auto_adjust=True, threads=False,
        )
    except Exception as exc:
        return {"ticker": ticker, "status": "ERROR", "error": str(exc)}

    if raw is None or raw.empty or len(raw) < 15:
        return {"ticker": ticker, "status": "NO_DATA", "error": "Insufficient price data"}

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    return _process_ticker_df(ticker, df, max_candles_ago)


# ─────────────────────────────────────────────────────────────
#  FAIR VALUE ENRICHMENT  (uses sma_router logic)
# ─────────────────────────────────────────────────────────────
_FV_NULL = {
    "composite_fair_price": None,
    "composite_gain_pct":   None,
    "fair_gap_pct":         None,   # alias: positive = stock is below fair value
    "valuation_bucket":     "N/A",
    "fv_model_count":       0,
}


def _enrich_fair_value(ticker: str) -> Dict:
    """
    Fetch fundamental fair value for a ticker using sma_router's regression models.
    Returns ONLY composite_fair_price, composite_gain_pct, fair_gap_pct,
    valuation_bucket, fv_model_count — never overwrites current_price / ema9 / etc.
    Falls back to _FV_NULL on any error.
    """
    if not _FV_AVAILABLE:
        return _FV_NULL
    try:
        res = _sma_analyze_ticker(
            ticker,
            fy_start=2014,
            force=False,
            include_other_income=True,
        )
        if "error" in res:
            return _FV_NULL
        comp_fair  = res.get("composite_fair_price")
        comp_gain  = res.get("composite_gain_pct")
        bucket     = res.get("valuation_bucket", "N/A")
        model_cnt  = res.get("model_count", 0)
        if comp_fair is None or comp_gain is None:
            return _FV_NULL
        # Return ONLY these 5 keys — never any price/ema/candle fields
        return {
            "composite_fair_price": comp_fair,
            "composite_gain_pct":   comp_gain,
            "fair_gap_pct":         comp_gain,   # positive = upside to fair value
            "valuation_bucket":     bucket,
            "fv_model_count":       model_cnt,
        }
    except Exception:
        return _FV_NULL


# ─────────────────────────────────────────────────────────────
#  REQUEST MODELS
# ─────────────────────────────────────────────────────────────
class Ema9ScreenRequest(BaseModel):
    tickers:         List[str]
    interval:        str = "1d"
    lookback_days:   int = 180
    max_candles_ago: int = 10


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
    tickers = [t.strip().upper() for t in req.tickers if t.strip()][:2000]
    if not tickers:
        raise HTTPException(400, "No tickers provided.")

    signals, failed = [], []
    total   = len(tickers)
    n_chunks = math.ceil(total / YF_CHUNK_SIZE)

    # ── Step 1: Chunked batch yf.download with inter-chunk delay ──────────────
    # Each chunk = one yf.download() call (safe, no cross-contamination).
    # Sleep YF_CHUNK_DELAY seconds between chunks to avoid Yahoo 429s on
    # large scans (e.g. 1 000–2 000 symbols split into 10–20 chunks).
    ticker_dfs: Dict[str, pd.DataFrame] = {}

    for chunk_idx in range(n_chunks):
        chunk = tickers[chunk_idx * YF_CHUNK_SIZE : (chunk_idx + 1) * YF_CHUNK_SIZE]

        chunk_dfs = await asyncio.to_thread(
            _batch_download, chunk, req.interval, req.lookback_days
        )
        ticker_dfs.update(chunk_dfs)

        # Pause between chunks (skip after the last one)
        if chunk_idx < n_chunks - 1:
            await asyncio.sleep(YF_CHUNK_DELAY)

    # ── Step 2: Process each ticker's DataFrame slice (pure Python) ────────────
    for ticker in tickers:
        if ticker not in ticker_dfs:
            failed.append({"ticker": ticker, "error": "No data from batch download"})
            continue
        try:
            res = _process_ticker_df(ticker, ticker_dfs[ticker], req.max_candles_ago)
            res["interval"] = req.interval
            if res["status"] in ("ERROR", "NO_DATA"):
                failed.append({"ticker": res["ticker"], "error": res.get("error", "")})
            elif res["status"] == "SIGNAL":
                signals.append(res)
            # NO_SIGNAL: silently dropped
        except Exception as exc:
            failed.append({"ticker": ticker, "error": str(exc)})

    # Sort: most recent confirmation first
    signals.sort(key=lambda r: r.get("candles_ago", 999))

    # ── Step 3: Fair Value enrichment — serial with inter-call delay ───────────
    # Screener.in will block rapid scraping.  We run enrichments one-at-a-time
    # and sleep FV_INTER_DELAY seconds between each call.
    # Disk cache (data/<TICKER>_pl.csv / _qr.csv) means previously-fetched
    # tickers skip the network entirely, so cached runs add zero delay.
    _SAFE_FV_KEYS = {
        "composite_fair_price", "composite_gain_pct",
        "fair_gap_pct", "valuation_bucket", "fv_model_count",
    }

    for i, sig in enumerate(signals):
        fv = await asyncio.to_thread(_enrich_fair_value, sig["ticker"])
        for k in _SAFE_FV_KEYS:
            sig[k] = fv.get(k)

        # Pause between Screener.in calls (skip after last signal)
        if i < len(signals) - 1:
            await asyncio.sleep(FV_INTER_DELAY)

    return {
        "signals":  signals,
        "count":    len(signals),
        "failed":   failed,
        "interval": req.interval,
    }