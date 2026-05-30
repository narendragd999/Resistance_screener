"""
Historical NSE Option Chart — FastAPI Router
Ported from core_new.py — zero Streamlit dependency

Added: Bear Call Spread analyzer (/api/option-charts/spread/bear-call)
"""
import os, time, asyncio, threading, math
from datetime import datetime, timedelta
from typing import Optional, List

import pandas as pd
import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

# ─────────────────────────────────────────────────────────────
#  NSE SESSION (isolated from main.py's session)
# ─────────────────────────────────────────────────────────────
_oc_session: Optional[requests.Session] = None
_oc_lock    = threading.Lock()
_oc_warmed  = False

NSE_OC_URL      = "https://www.nseindia.com/api/historicalOR/foCPV"

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
            "Accept":                    "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language":           "en-US,en;q=0.9",
            "Accept-Encoding":           "gzip, deflate",
            "Connection":                "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        })
        _oc_session = s
        return s


def _reset_oc_session():
    global _oc_session, _oc_warmed
    with _oc_lock:
        _oc_session = None
        _oc_warmed  = False


def _warm_up(session: requests.Session) -> bool:
    global _oc_warmed
    if _oc_warmed:
        return True
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
    _oc_warmed = True
    return True


# ─────────────────────────────────────────────────────────────
#  CORE FETCH — OHLC DATA (blocking)
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

    for col in ["open", "high", "low", "close", "ltp", "underlying"]:
        if col in df.columns:
            df[col] = df[col].round(2)

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
#  STRIKES FETCH
# ─────────────────────────────────────────────────────────────
def _do_fetch_strikes(
    symbol: str,
    instrument_type: str,
    expiry_dt: datetime,
    option_type: str,
) -> list:
    session = _get_oc_session()
    _warm_up(session)

    today = datetime.now().date()
    if expiry_dt.date() >= today:
        to_dt   = datetime.now()
        from_dt = to_dt - timedelta(days=7)
    else:
        to_dt   = expiry_dt
        from_dt = expiry_dt - timedelta(days=3)

    params = {
        "from":           from_dt.strftime("%d-%m-%Y"),
        "to":             to_dt.strftime("%d-%m-%Y"),
        "instrumentType": instrument_type,
        "symbol":         symbol,
        "year":           str(expiry_dt.year),
        "expiryDate":     expiry_dt.strftime("%d-%b-%Y").upper(),
        "optionType":     option_type,
    }
    api_hdrs = {
        "Accept":           "application/json, text/plain, */*",
        "Referer":          "https://www.nseindia.com/market-data/equity-derivatives-watch",
        "X-Requested-With": "XMLHttpRequest",
    }

    for attempt in range(2):
        try:
            r = session.get(NSE_OC_URL, params=params, headers=api_hdrs, timeout=15)
            if r.status_code == 401:
                _warm_up(session)
                time.sleep(2)
                continue
            if r.status_code == 403:
                return []
            if r.status_code != 200:
                return []

            data = r.json()
            if not data or "data" not in data or not data["data"]:
                return []

            strikes = set()
            for row in data["data"]:
                strike_val = row.get("FH_STRIKE_PRICE")
                if strike_val is not None:
                    try:
                        s = int(float(strike_val))
                        if s > 0:
                            strikes.add(s)
                    except (ValueError, TypeError):
                        pass

            return sorted(strikes)

        except requests.RequestException:
            if attempt == 1:
                return []
            time.sleep(2)
        except Exception:
            return []

    return []


# ─────────────────────────────────────────────────────────────
#  BEAR CALL SPREAD — CORE CALCULATION ENGINE
# ─────────────────────────────────────────────────────────────

def _build_daily_pnl_series(
    short_data: list,      # OHLC rows for short (sold) call
    long_data: list,       # OHLC rows for long (bought) call
    entry_date: str,       # YYYY-MM-DD — the trade initiation date
    short_entry_premium: float,   # premium received on short leg
    long_entry_premium: float,    # premium paid on long leg
    lot_size: int,
    num_lots: int,
) -> list:
    """
    Build day-by-day P&L for a Bear Call Spread.
    Short Call Sold @ short_entry_premium, Long Call Bought @ long_entry_premium.
    Net credit = short_entry_premium - long_entry_premium.

    Daily unrealised P&L:
      pnl_per_lot = ((short_entry_premium - short_close) - (long_close - long_entry_premium)) * lot_size
    """
    # Index by date
    short_map = {r["date"]: r for r in short_data}
    long_map  = {r["date"]: r for r in long_data}

    all_dates = sorted(set(short_map.keys()) | set(long_map.keys()))
    # Only from entry_date onward
    all_dates = [d for d in all_dates if d >= entry_date]

    net_credit_per_share = round(short_entry_premium - long_entry_premium, 2)
    max_profit_per_lot   = net_credit_per_share * lot_size
    total_max_profit     = max_profit_per_lot * num_lots

    rows = []
    for d in all_dates:
        sr = short_map.get(d)
        lr = long_map.get(d)
        if sr is None or lr is None:
            continue

        sc = sr.get("close") or sr.get("ltp")
        lc = lr.get("close") or lr.get("ltp")
        if sc is None or lc is None:
            continue

        # P&L per share = net_credit - current_net_debit_to_close
        # Current net debit to close = (buy back short) - (sell long) = sc - lc
        pnl_per_share  = net_credit_per_share - (sc - lc)
        pnl_per_lot    = round(pnl_per_share * lot_size, 2)
        pnl_total      = round(pnl_per_lot * num_lots, 2)
        pct_of_max     = round((pnl_total / total_max_profit * 100) if total_max_profit != 0 else 0, 1)

        underlying = sr.get("underlying")

        rows.append({
            "date":            d,
            "short_close":     round(sc, 2),
            "long_close":      round(lc, 2),
            "pnl_per_share":   round(pnl_per_share, 2),
            "pnl_per_lot":     pnl_per_lot,
            "pnl_total":       pnl_total,
            "pct_of_max":      pct_of_max,
            "underlying":      round(underlying, 2) if underlying else None,
        })

    return rows


def _calc_spread_stats(
    pnl_rows: list,
    short_strike: int,
    long_strike: int,
    short_entry: float,
    long_entry: float,
    lot_size: int,
    num_lots: int,
    expiry_str: str,         # label only
) -> dict:
    """Calculate summary statistics for the spread."""
    if not pnl_rows:
        return {}

    net_credit      = round(short_entry - long_entry, 2)
    spread_width    = long_strike - short_strike
    max_profit_ps   = net_credit
    max_loss_ps     = round(spread_width - net_credit, 2)
    breakeven       = round(short_strike + net_credit, 2)

    total_lots      = num_lots
    max_profit_tot  = round(max_profit_ps * lot_size * total_lots, 2)
    max_loss_tot    = round(max_loss_ps * lot_size * total_lots, 2)
    capital_at_risk = max_loss_tot

    # Final row = expiry (last available close)
    final_row   = pnl_rows[-1]
    final_pnl   = final_row["pnl_total"]
    final_pct   = final_row["pct_of_max"]

    # Peak unrealised profit and drawdown
    pnl_series  = [r["pnl_total"] for r in pnl_rows]
    peak_pnl    = max(pnl_series)
    trough_pnl  = min(pnl_series)
    max_dd      = round(peak_pnl - trough_pnl, 2) if peak_pnl > trough_pnl else 0

    # Days in profit vs loss
    days_profit = sum(1 for p in pnl_series if p > 0)
    days_loss   = sum(1 for p in pnl_series if p < 0)
    days_flat   = len(pnl_series) - days_profit - days_loss

    # Risk/reward
    rr = round(max_profit_tot / max_loss_tot, 3) if max_loss_tot != 0 else None

    # Outcome
    if final_pnl >= max_profit_tot * 0.95:
        outcome = "FULL PROFIT 🎯"
    elif final_pnl > 0:
        outcome = "PARTIAL PROFIT ✅"
    elif final_pnl == 0:
        outcome = "BREAKEVEN ⚖️"
    elif final_pnl < 0 and final_pnl > max_loss_tot * 0.5:
        outcome = "PARTIAL LOSS ⚠️"
    else:
        outcome = "FULL LOSS ❌"

    return {
        "expiry":           expiry_str,
        "short_strike":     short_strike,
        "long_strike":      long_strike,
        "spread_width":     spread_width,
        "net_credit":       net_credit,
        "breakeven":        breakeven,
        "max_profit_ps":    max_profit_ps,
        "max_loss_ps":      max_loss_ps,
        "max_profit_total": max_profit_tot,
        "max_loss_total":   max_loss_tot,
        "capital_at_risk":  capital_at_risk,
        "risk_reward":      rr,
        "final_pnl":        final_pnl,
        "final_pct_of_max": final_pct,
        "peak_pnl":         round(peak_pnl, 2),
        "trough_pnl":       round(trough_pnl, 2),
        "max_drawdown":     max_dd,
        "days_total":       len(pnl_series),
        "days_profit":      days_profit,
        "days_loss":        days_loss,
        "days_flat":        days_flat,
        "outcome":          outcome,
    }


def _build_payoff_curve(
    short_strike: int,
    long_strike: int,
    net_credit: float,
    lot_size: int,
    num_lots: int,
    underlying_price: Optional[float] = None,
) -> dict:
    """Build at-expiry payoff curve data for the Bear Call Spread."""
    spread_width = long_strike - short_strike
    max_profit   = net_credit
    max_loss     = spread_width - net_credit
    breakeven    = short_strike + net_credit

    # Price range: from short_strike * 0.85 to long_strike * 1.15
    lo = int(short_strike * 0.85)
    hi = int(long_strike  * 1.15)
    step = max(1, (hi - lo) // 200)

    prices, pnls_ps, pnls_total = [], [], []
    for price in range(lo, hi + step, step):
        # Short call P&L at expiry
        short_pnl = net_credit - max(0, price - short_strike)
        # Long call offsets losses above long_strike
        long_pnl  = max(0, price - long_strike)
        pnl_ps    = round(short_pnl + long_pnl, 4)
        # Clamp to theoretical bounds
        pnl_ps    = max(-max_loss, min(max_profit, pnl_ps))
        prices.append(price)
        pnls_ps.append(pnl_ps)
        pnls_total.append(round(pnl_ps * lot_size * num_lots, 2))

    return {
        "prices":      prices,
        "pnl_ps":      pnls_ps,
        "pnl_total":   pnls_total,
        "breakeven":   round(breakeven, 2),
        "max_profit":  max_profit,
        "max_loss":    max_loss,
        "underlying":  underlying_price,
    }


def _do_bear_call_spread(
    symbol: str,
    instrument_type: str,
    short_strike: int,
    long_strike: int,
    expiry_dt: datetime,
    entry_date: str,         # YYYY-MM-DD
    lot_size: int,
    num_lots: int,
) -> dict:
    """
    Full Bear Call Spread analysis for one expiry.
    Fetches OHLC for both legs from entry_date → expiry_dt.
    """
    fmt = "%d-%m-%Y"
    from_dt = datetime.strptime(entry_date, "%Y-%m-%d")

    # Fetch short (sold) call leg
    short_rows = _do_fetch(
        from_dt, expiry_dt,
        symbol, expiry_dt.year, expiry_dt,
        "CE", short_strike, instrument_type,
    )
    # Fetch long (bought) call leg
    long_rows = _do_fetch(
        from_dt, expiry_dt,
        symbol, expiry_dt.year, expiry_dt,
        "CE", long_strike, instrument_type,
    )

    if not short_rows:
        raise ValueError(f"No data for short leg (CE {short_strike}) on expiry {expiry_dt.strftime('%d-%b-%Y')}")
    if not long_rows:
        raise ValueError(f"No data for long leg (CE {long_strike}) on expiry {expiry_dt.strftime('%d-%b-%Y')}")

    # Entry premiums = close/ltp on entry_date (first available row ≥ entry_date)
    def _find_entry_premium(rows, entry):
        for r in sorted(rows, key=lambda x: x["date"]):
            if r["date"] >= entry:
                v = r.get("close") or r.get("ltp")
                if v:
                    return float(v), r["date"]
        return None, None

    short_entry_prem, short_entry_actual = _find_entry_premium(short_rows, entry_date)
    long_entry_prem,  long_entry_actual  = _find_entry_premium(long_rows,  entry_date)

    if short_entry_prem is None:
        raise ValueError(f"Could not determine entry premium for short leg (CE {short_strike})")
    if long_entry_prem is None:
        raise ValueError(f"Could not determine entry premium for long leg (CE {long_strike})")

    actual_entry_date = max(short_entry_actual, long_entry_actual)

    # Build daily P&L series
    pnl_rows = _build_daily_pnl_series(
        short_rows, long_rows,
        actual_entry_date,
        short_entry_prem, long_entry_prem,
        lot_size, num_lots,
    )

    # Stats
    stats = _calc_spread_stats(
        pnl_rows,
        short_strike, long_strike,
        short_entry_prem, long_entry_prem,
        lot_size, num_lots,
        expiry_dt.strftime("%d-%b-%Y"),
    )

    # Payoff curve
    net_credit    = short_entry_prem - long_entry_prem
    last_underlying = None
    if pnl_rows:
        last_underlying = pnl_rows[-1].get("underlying")
    payoff = _build_payoff_curve(
        short_strike, long_strike, net_credit,
        lot_size, num_lots, last_underlying,
    )

    # Underlying price on entry date (first row of pnl_series)
    entry_underlying = None
    entry_underlying_date = None
    if pnl_rows:
        entry_underlying      = pnl_rows[0].get("underlying")
        entry_underlying_date = pnl_rows[0].get("date")
    # Fallback: scan short_rows for closest to actual_entry_date
    if entry_underlying is None:
        for row in short_rows:
            if row.get("date") == actual_entry_date:
                entry_underlying      = row.get("underlying")
                entry_underlying_date = row.get("date")
                break
        if entry_underlying is None and short_rows:
            # take the first available row after or on entry
            for row in sorted(short_rows, key=lambda x: x["date"]):
                if row.get("date") >= actual_entry_date and row.get("underlying") is not None:
                    entry_underlying      = row.get("underlying")
                    entry_underlying_date = row.get("date")
                    break

    # OTM % for both strikes relative to underlying at entry
    short_otm_pct = None
    long_otm_pct  = None
    if entry_underlying and entry_underlying > 0:
        short_otm_pct = round((short_strike - entry_underlying) / entry_underlying * 100, 2)
        long_otm_pct  = round((long_strike  - entry_underlying) / entry_underlying * 100, 2)

    return {
        "entry_date":           actual_entry_date,
        "short_entry_premium":  round(short_entry_prem, 2),
        "long_entry_premium":   round(long_entry_prem, 2),
        "entry_underlying":     round(entry_underlying, 2) if entry_underlying else None,
        "entry_underlying_date": entry_underlying_date,
        "short_otm_pct":        short_otm_pct,
        "long_otm_pct":         long_otm_pct,
        "stats":                stats,
        "pnl_series":           pnl_rows,
        "payoff":               payoff,
        "short_ohlc":           short_rows,
        "long_ohlc":            long_rows,
    }


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


class BearCallSpreadRequest(BaseModel):
    symbol:          str
    instrument_type: str = "OPTSTK"
    short_strike:    int            # Sold (lower) call strike
    long_strike:     int            # Bought (higher) call strike
    entry_date:      str            # DD-MM-YYYY — trade initiation date
    expiry_date_1:   str            # DD-MM-YYYY — first (nearest) expiry
    expiry_date_2:   Optional[str] = None  # DD-MM-YYYY — second expiry (optional)
    lot_size:        int = 1        # NSE lot size for the symbol
    num_lots:        int = 1        # Number of lots to trade


# ─────────────────────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────────────────────
@router.get("/api/option-charts/meta")
async def oc_meta():
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


@router.get("/api/option-charts/strikes")
async def oc_strikes(
    symbol:          str,
    instrument_type: str = "OPTSTK",
    expiry_date:     str = "",
    option_type:     str = "CE",
):
    if not symbol or not expiry_date:
        return {"strikes": [], "source": "missing_params", "count": 0}

    if instrument_type.upper().startswith("FUT"):
        return {"strikes": [], "source": "futures_no_strikes", "count": 0}

    try:
        expiry_dt = datetime.strptime(expiry_date, "%d-%m-%Y")
    except ValueError:
        raise HTTPException(400, "Invalid expiry_date — use DD-MM-YYYY")

    try:
        strikes = await asyncio.to_thread(
            _do_fetch_strikes,
            symbol.strip().upper(),
            instrument_type.upper(),
            expiry_dt,
            option_type.upper(),
        )
    except Exception:
        strikes = []

    source = "historicalOR" if strikes else "not_found"
    return {"strikes": strikes, "source": source, "count": len(strikes)}


@router.post("/api/option-charts/fetch")
async def oc_fetch(req: OcFetchRequest):
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


def _do_fetch_strikes_with_underlying(
    symbol: str,
    instrument_type: str,
    expiry_dt: datetime,
    option_type: str,
    entry_dt: datetime,
) -> dict:
    """
    Fetch available CE strikes for a given expiry AND the underlying price
    on/near the entry date. Returns both so the frontend can show OTM context.
    """
    session = _get_oc_session()
    _warm_up(session)

    # To get strikes: use a narrow window around expiry
    today = datetime.now().date()
    if expiry_dt.date() >= today:
        to_dt   = datetime.now()
        from_dt = to_dt - timedelta(days=7)
    else:
        to_dt   = expiry_dt
        from_dt = expiry_dt - timedelta(days=3)

    params = {
        "from":           from_dt.strftime("%d-%m-%Y"),
        "to":             to_dt.strftime("%d-%m-%Y"),
        "instrumentType": instrument_type,
        "symbol":         symbol,
        "year":           str(expiry_dt.year),
        "expiryDate":     expiry_dt.strftime("%d-%b-%Y").upper(),
        "optionType":     option_type,
    }
    api_hdrs = {
        "Accept":           "application/json, text/plain, */*",
        "Referer":          "https://www.nseindia.com/market-data/equity-derivatives-watch",
        "X-Requested-With": "XMLHttpRequest",
    }

    strikes = set()
    underlying_price = None
    underlying_date  = None

    # ── Pass 1: fetch strikes list (narrow window near expiry, no strikePrice needed) ──
    for attempt in range(2):
        try:
            r = session.get(NSE_OC_URL, params=params, headers=api_hdrs, timeout=15)
            if r.status_code == 401:
                _warm_up(session)
                time.sleep(2)
                continue
            if r.status_code not in (200,):
                break
            data = r.json()
            if not data or "data" not in data or not data["data"]:
                break
            for row in data["data"]:
                sv = row.get("FH_STRIKE_PRICE")
                if sv is not None:
                    try:
                        s = int(float(sv))
                        if s > 0:
                            strikes.add(s)
                    except (ValueError, TypeError):
                        pass
            break
        except requests.RequestException:
            if attempt == 1:
                break
            time.sleep(2)
        except Exception:
            break

    # ── Pass 2: fetch underlying price on/near entry_date using a wide window ──
    # Use entry_date as the centre of the window; fetch full range entry→expiry
    # with a specific strike so NSE returns data for old series too.
    if strikes:
        try:
            any_strike  = sorted(strikes)[len(strikes) // 2]   # pick a middle strike
            entry_from  = entry_dt - timedelta(days=5)
            entry_to    = entry_dt + timedelta(days=10)        # wider look-ahead
            # Don't go past expiry
            if entry_to.date() > expiry_dt.date():
                entry_to = expiry_dt
            entry_params = {
                "from":           entry_from.strftime("%d-%m-%Y"),
                "to":             entry_to.strftime("%d-%m-%Y"),
                "instrumentType": instrument_type,
                "symbol":         symbol,
                "year":           str(expiry_dt.year),
                "expiryDate":     expiry_dt.strftime("%d-%b-%Y").upper(),
                "optionType":     option_type,
                "strikePrice":    str(any_strike),
            }
            r2 = session.get(NSE_OC_URL, params=entry_params, headers=api_hdrs, timeout=15)
            if r2.status_code == 200:
                d2 = r2.json()
                if d2 and "data" in d2 and d2["data"]:
                    best_diff = None
                    for row in d2["data"]:
                        uv = row.get("FH_UNDERLYING_VALUE")
                        ts = row.get("FH_TIMESTAMP")
                        if uv and ts:
                            try:
                                uval   = float(uv)
                                row_dt = datetime.strptime(ts, "%d-%b-%Y")
                                diff   = abs((row_dt.date() - entry_dt.date()).days)
                                if best_diff is None or diff < best_diff:
                                    best_diff        = diff
                                    underlying_price = uval
                                    underlying_date  = row_dt.strftime("%d-%b-%Y")
                            except Exception:
                                pass
        except Exception:
            pass

    # ── Pass 3: if still no underlying, try each available strike until one returns data ──
    if underlying_price is None and strikes:
        for try_strike in sorted(strikes)[:5]:
            try:
                entry_from = entry_dt - timedelta(days=5)
                entry_to   = min(entry_dt + timedelta(days=10), expiry_dt)
                p3 = {
                    "from":           entry_from.strftime("%d-%m-%Y"),
                    "to":             entry_to.strftime("%d-%m-%Y"),
                    "instrumentType": instrument_type,
                    "symbol":         symbol,
                    "year":           str(expiry_dt.year),
                    "expiryDate":     expiry_dt.strftime("%d-%b-%Y").upper(),
                    "optionType":     option_type,
                    "strikePrice":    str(try_strike),
                }
                r3 = session.get(NSE_OC_URL, params=p3, headers=api_hdrs, timeout=15)
                if r3.status_code == 200:
                    d3 = r3.json()
                    if d3 and "data" in d3 and d3["data"]:
                        best_diff = None
                        for row in d3["data"]:
                            uv = row.get("FH_UNDERLYING_VALUE")
                            ts = row.get("FH_TIMESTAMP")
                            if uv and ts:
                                try:
                                    uval   = float(uv)
                                    row_dt = datetime.strptime(ts, "%d-%b-%Y")
                                    diff   = abs((row_dt.date() - entry_dt.date()).days)
                                    if best_diff is None or diff < best_diff:
                                        best_diff        = diff
                                        underlying_price = uval
                                        underlying_date  = row_dt.strftime("%d-%b-%Y")
                                except Exception:
                                    pass
                        if underlying_price is not None:
                            break
            except Exception:
                continue

    strikes_sorted = sorted(strikes)

    # Build strike_detail with OTM/ATM/ITM classification for CE
    strikes_detail = []
    for s in strikes_sorted:
        if underlying_price is not None:
            pct_otm = round((s - underlying_price) / underlying_price * 100, 1)
            if abs(pct_otm) <= 0.5:
                zone = "ATM"
            elif pct_otm > 0:
                zone = "OTM"
            else:
                zone = "ITM"
        else:
            pct_otm = None
            zone    = "UNK"
        strikes_detail.append({"strike": s, "pct_otm": pct_otm, "zone": zone})

    # Build suggested spread pairs: pairs of (short, long) both OTM, spread_width ≈ 1-3 strike steps
    spread_pairs = []
    if underlying_price is not None:
        otm_strikes = [s for s in strikes_sorted if s > underlying_price]
        step = None
        if len(otm_strikes) >= 2:
            diffs = [otm_strikes[i+1] - otm_strikes[i] for i in range(min(5, len(otm_strikes)-1))]
            if diffs:
                step = int(sorted(diffs)[len(diffs)//2])  # median step

        for i, short_s in enumerate(otm_strikes[:8]):
            for width_mult in [1, 2, 3]:
                long_s = short_s + (step * width_mult if step else 0)
                if long_s in strikes and long_s != short_s:
                    short_pct = round((short_s - underlying_price) / underlying_price * 100, 1)
                    entry = {"short_strike": short_s, "long_strike": long_s,
                             "spread_width": long_s - short_s, "short_pct_otm": short_pct}
                    if entry not in spread_pairs:
                        spread_pairs.append(entry)
                    if len(spread_pairs) >= 6:
                        break
            if len(spread_pairs) >= 6:
                break

    return {
        "strikes":        strikes_sorted,
        "strikes_detail": strikes_detail,
        "underlying":     round(underlying_price, 2) if underlying_price else None,
        "underlying_date": underlying_date,
        "spread_pairs":   spread_pairs,
    }


@router.get("/api/option-charts/spread/suggest-strikes")
async def spread_suggest_strikes(
    symbol:          str,
    instrument_type: str = "OPTSTK",
    expiry_date:     str = "",
    entry_date:      str = "",
    option_type:     str = "CE",
    num_otm_levels:  int = 5,
):
    """
    Returns available CE strikes for the given expiry with OTM/ATM/ITM classification
    based on historical underlying price on the entry_date.
    Also returns suggested Bear Call Spread pairs.
    """
    if not symbol or not expiry_date or not entry_date:
        raise HTTPException(400, "symbol, expiry_date and entry_date are required")

    try:
        expiry_dt = datetime.strptime(expiry_date, "%d-%m-%Y")
    except ValueError:
        raise HTTPException(400, "Invalid expiry_date — use DD-MM-YYYY")

    try:
        entry_dt = datetime.strptime(entry_date, "%d-%m-%Y")
    except ValueError:
        raise HTTPException(400, "Invalid entry_date — use DD-MM-YYYY")

    try:
        result = await asyncio.to_thread(
            _do_fetch_strikes_with_underlying,
            symbol.strip().upper(),
            instrument_type.upper(),
            expiry_dt,
            option_type.upper(),
            entry_dt,
        )
    except Exception as exc:
        raise HTTPException(500, f"Error fetching strike suggestions: {exc}")

    return result


@router.post("/api/option-charts/spread/bear-call")
async def bear_call_spread(req: BearCallSpreadRequest):
    """
    Bear Call Spread historical analysis.

    Fetches OHLC data for both legs (short lower CE + long higher CE),
    computes daily unrealised P&L, final stats, and payoff curve.
    Supports analysis across two expiry dates for comparison.

    Parameters
    ----------
    symbol          : NSE F&O ticker (e.g. RELIANCE, NIFTY)
    instrument_type : OPTSTK or OPTIDX
    short_strike    : Strike of the SOLD (lower) call
    long_strike     : Strike of the BOUGHT (higher) call
    entry_date      : Date you initiated the trade (DD-MM-YYYY)
    expiry_date_1   : First expiry to evaluate (DD-MM-YYYY)
    expiry_date_2   : Second expiry to evaluate (DD-MM-YYYY), optional
    lot_size        : NSE lot size for the symbol
    num_lots        : Number of lots

    Returns
    -------
    JSON with:
    - expiry_1: { entry_date, short_entry_premium, long_entry_premium,
                  stats, pnl_series, payoff, short_ohlc, long_ohlc }
    - expiry_2: same structure (if expiry_date_2 provided)
    """
    fmt = "%d-%m-%Y"

    # ── Validate inputs ──
    try:
        entry_dt   = datetime.strptime(req.entry_date, fmt)
        expiry1_dt = datetime.strptime(req.expiry_date_1, fmt)
    except ValueError as exc:
        raise HTTPException(400, f"Date format error (use DD-MM-YYYY): {exc}")

    if req.short_strike <= 0:
        raise HTTPException(400, "short_strike must be > 0")
    if req.long_strike <= 0:
        raise HTTPException(400, "long_strike must be > 0")
    if req.long_strike <= req.short_strike:
        raise HTTPException(400, "long_strike must be > short_strike for a Bear Call Spread")
    if entry_dt.date() > expiry1_dt.date():
        raise HTTPException(400, "entry_date must be before or on expiry_date_1")
    if req.lot_size <= 0:
        raise HTTPException(400, "lot_size must be > 0")
    if req.num_lots <= 0:
        raise HTTPException(400, "num_lots must be > 0")

    symbol    = req.symbol.strip().upper()
    inst_type = req.instrument_type.upper()
    entry_str = entry_dt.strftime("%Y-%m-%d")   # internal YYYY-MM-DD

    result = {"symbol": symbol, "instrument_type": inst_type}

    # ── Expiry 1 ──
    try:
        e1 = await asyncio.to_thread(
            _do_bear_call_spread,
            symbol, inst_type,
            req.short_strike, req.long_strike,
            expiry1_dt, entry_str,
            req.lot_size, req.num_lots,
        )
        result["expiry_1"] = e1
    except ValueError as exc:
        raise HTTPException(400, f"Expiry 1 error: {exc}")
    except Exception as exc:
        raise HTTPException(500, f"Expiry 1 internal error: {exc}")

    # ── Expiry 2 (optional) ──
    if req.expiry_date_2:
        try:
            expiry2_dt = datetime.strptime(req.expiry_date_2, fmt)
        except ValueError as exc:
            raise HTTPException(400, f"expiry_date_2 format error: {exc}")

        if entry_dt.date() > expiry2_dt.date():
            raise HTTPException(400, "entry_date must be before or on expiry_date_2")

        try:
            e2 = await asyncio.to_thread(
                _do_bear_call_spread,
                symbol, inst_type,
                req.short_strike, req.long_strike,
                expiry2_dt, entry_str,
                req.lot_size, req.num_lots,
            )
            result["expiry_2"] = e2
        except ValueError as exc:
            result["expiry_2_error"] = str(exc)
        except Exception as exc:
            result["expiry_2_error"] = f"Internal error: {exc}"

    return result