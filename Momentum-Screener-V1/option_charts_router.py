"""
Historical NSE Option Chart — FastAPI Router
Ported from core_new.py — zero Streamlit dependency
"""
import os, time, asyncio, threading
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

# ─────────────────────────────────────────────────────────────
#  NSE SESSION (isolated from main.py's session)
# ─────────────────────────────────────────────────────────────
_oc_session: Optional[requests.Session] = None
_oc_lock = threading.Lock()

NSE_OC_URL = "https://www.nseindia.com/api/historicalOR/foCPV"

INSTRUMENT_TYPES = ["OPTSTK", "OPTIDX", "FUTIDX", "FUTSTK", "FUTIVX"]
OPTION_TYPES     = ["CE", "PE"]
QUICK_RANGES     = ["Custom", "1D", "1W", "1M", "1.5M", "3M"]


def _get_oc_session() -> requests.Session:
    global _oc_session
    with _oc_lock:
        if _oc_session is not None:
            return _oc_session
        s = requests.Session()
        s.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate",
            "Connection":      "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        })
        _oc_session = s
        return s


def _reset_oc_session():
    global _oc_session
    with _oc_lock:
        _oc_session = None


def _warm_up(session: requests.Session) -> bool:
    warm_pages = [
        "https://www.nseindia.com/",
        "https://www.nseindia.com/market-data/equity-derivatives-watch",
    ]
    for page in warm_pages:
        try:
            session.get(page, timeout=15)
            time.sleep(1.5)
        except Exception:
            return False
    return True


# ─────────────────────────────────────────────────────────────
#  CORE FETCH (blocking — run via asyncio.to_thread)
# ─────────────────────────────────────────────────────────────
def _do_fetch(
    from_dt: datetime, to_dt: datetime,
    symbol: str, year: int, expiry_dt: datetime,
    option_type: str, strike_price: int, instrument_type: str,
) -> list:
    session = _get_oc_session()
    _warm_up(session)

    params = {
        "from":           from_dt.strftime("%d-%m-%Y"),
        "to":             to_dt.strftime("%d-%m-%Y"),
        "instrumentType": instrument_type,
        "symbol":         symbol,
        "year":           str(year),
        "expiryDate":     expiry_dt.strftime("%d-%b-%Y").upper(),
        "optionType":     option_type,
        "strikePrice":    str(strike_price),
    }
    api_hdrs = {
        "Accept":           "application/json, text/plain, */*",
        "Referer":          "https://www.nseindia.com/market-data/equity-derivatives-watch",
        "X-Requested-With": "XMLHttpRequest",
    }

    resp_data = None
    for attempt in range(2):
        try:
            r = session.get(NSE_OC_URL, params=params, headers=api_hdrs, timeout=15)
            if r.status_code == 401:
                _warm_up(session)
                time.sleep(3)
                continue
            if r.status_code == 403:
                _reset_oc_session()
                raise ValueError("HTTP 403 — NSE blocked the request. Try again after a few seconds.")
            if r.status_code != 200:
                raise ValueError(f"HTTP {r.status_code} from NSE.")
            resp_data = r.json()
            break
        except ValueError:
            raise
        except requests.RequestException as exc:
            if attempt == 1:
                raise ValueError(f"Network error: {exc}")
            time.sleep(3)

    if not resp_data or "data" not in resp_data or not resp_data["data"]:
        raise ValueError(
            "No data returned by NSE. Verify: symbol exists in F&O, "
            "expiry date is correct, strike price is valid for that expiry."
        )

    df = pd.DataFrame(resp_data["data"])

    rename_map = {
        "FH_TIMESTAMP":         "date",
        "FH_OPENING_PRICE":     "open",
        "FH_TRADE_HIGH_PRICE":  "high",
        "FH_TRADE_LOW_PRICE":   "low",
        "FH_CLOSING_PRICE":     "close",
        "FH_LAST_TRADED_PRICE": "ltp",
        "FH_STRIKE_PRICE":      "strike_price",
        "FH_EXPIRY_DT":         "expiry",
        "FH_OPTION_TYPE":       "option_type_col",
        "FH_UNDERLYING_VALUE":  "underlying",
        "FH_TOT_TRADED_QTY":    "volume",
        "FH_OPEN_INT":          "oi",
        "FH_CHG_IN_OI":         "change_oi",
        "FH_SETTLE_PRICE":      "settle_price",
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    numeric = ["open", "high", "low", "close", "ltp", "volume", "oi", "change_oi",
               "underlying", "settle_price"]
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.replace("-", None, inplace=True)
    df.dropna(subset=["open", "high", "low", "close"], inplace=True)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], format="%d-%b-%Y", errors="coerce")
        df.sort_values("date", inplace=True)
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    # Round numeric cols
    for col in ["open", "high", "low", "close", "ltp", "underlying"]:
        if col in df.columns:
            df[col] = df[col].round(2)

    # ── Sanitize: replace ALL NaN / Inf / -Inf with None so JSON serialises cleanly
    import math
    def _safe(v):
        if v is None:
            return None
        try:
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return None
        except Exception:
            pass
        return v

    records = df.to_dict(orient="records")
    clean   = [{k: _safe(v) for k, v in row.items()} for row in records]
    return clean


# ─────────────────────────────────────────────────────────────
#  PYDANTIC MODELS
# ─────────────────────────────────────────────────────────────
class OcFetchRequest(BaseModel):
    symbol:          str
    instrument_type: str = "OPTSTK"
    expiry_date:     str            # DD-MM-YYYY
    option_type:     str = "CE"
    strike_price:    int
    from_date:       str            # DD-MM-YYYY
    to_date:         str            # DD-MM-YYYY


# ─────────────────────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────────────────────
@router.get("/api/option-charts/meta")
async def oc_meta():
    """Return tickers + static dropdown values for the UI."""
    tickers = []
    try:
        if os.path.exists("tickers.csv"):
            df = pd.read_csv("tickers.csv")
            if "SYMBOL" in df.columns:
                tickers = sorted(df["SYMBOL"].dropna().str.strip().str.upper().unique().tolist())
    except Exception:
        pass
    return {
        "tickers":          tickers,
        "instrument_types": INSTRUMENT_TYPES,
        "option_types":     OPTION_TYPES,
        "quick_ranges":     QUICK_RANGES,
    }


@router.post("/api/option-charts/fetch")
async def oc_fetch(req: OcFetchRequest):
    # ── parse dates
    fmt = "%d-%m-%Y"
    try:
        from_dt   = datetime.strptime(req.from_date,   fmt)
        to_dt     = datetime.strptime(req.to_date,     fmt)
        expiry_dt = datetime.strptime(req.expiry_date, fmt)
    except ValueError as exc:
        raise HTTPException(400, f"Date format error (use DD-MM-YYYY): {exc}")

    if from_dt >= to_dt:
        raise HTTPException(400, "from_date must be before to_date.")
    if to_dt.date() > expiry_dt.date():
        raise HTTPException(400, "to_date cannot be after expiry_date.")
    if req.strike_price <= 0:
        raise HTTPException(400, "strike_price must be > 0.")

    try:
        rows = await asyncio.to_thread(
            _do_fetch,
            from_dt, to_dt,
            req.symbol.strip().upper(),
            expiry_dt.year,
            expiry_dt,
            req.option_type.upper(),
            req.strike_price,
            req.instrument_type,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(500, f"Internal error: {exc}")

    if not rows:
        raise HTTPException(404, "No rows returned for the given parameters.")

    last = rows[-1]
    return {
        "symbol":      req.symbol.strip().upper(),
        "expiry":      expiry_dt.strftime("%d-%b-%Y"),
        "strike":      req.strike_price,
        "option_type": req.option_type.upper(),
        "rows":        len(rows),
        "last_close":  last.get("close"),
        "last_ltp":    last.get("ltp"),
        "data":        rows,
    }