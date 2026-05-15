"""
SMA / Value Screener — FastAPI Router
Ported from value-invest.py — zero Streamlit dependency
"""
import os, re, asyncio, datetime as dt_module
from typing import Optional, List, Dict, Any

import difflib
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from sklearn.linear_model import LinearRegression
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel

router = APIRouter()

DATA_DIR        = "data"
NSE_TICKER_CSV  = "NSE_ALL_TICKERS_LIST.csv"
os.makedirs(DATA_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
#  NSE TICKER AUTOCOMPLETE LIST
# ─────────────────────────────────────────────────────────────
_nse_df: Optional[pd.DataFrame] = None

def _load_nse_df() -> pd.DataFrame:
    global _nse_df
    if _nse_df is not None:
        return _nse_df

    # ── Primary: tickers.csv (same file the main screener uses)
    for csv_path in ["tickers.csv", "NSE_ALL_TICKERS_LIST.csv"]:
        if not os.path.exists(csv_path):
            continue
        try:
            df = pd.read_csv(csv_path, dtype=str)
            df.columns = [c.strip().lower() for c in df.columns]

            # Normalise column names across different CSV formats
            if "name of company" in df.columns:
                df = df.rename(columns={"name of company": "company_name"})
            elif "company_name" not in df.columns:
                # tickers.csv only has SYMBOL — synthesise a company_name col
                df["company_name"] = ""

            if "symbol" not in df.columns:
                continue

            df["symbol"]       = df["symbol"].str.strip().str.upper()
            df["company_name"] = df["company_name"].fillna("").str.strip()
            _nse_df = df[["symbol", "company_name"]].dropna(subset=["symbol"])
            _nse_df = _nse_df[_nse_df["symbol"] != ""].reset_index(drop=True)
            return _nse_df
        except Exception:
            continue

    _nse_df = pd.DataFrame(columns=["symbol", "company_name"])
    return _nse_df

# ─────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────
def is_bank_or_finance(ticker: str) -> bool:
    ticker = ticker.upper()
    df = _load_nse_df()
    if not df.empty:
        row = df[df["symbol"] == ticker]
        if not row.empty:
            name = row["company_name"].iloc[0].upper()
            return any(k in name for k in
                       ["BANK", "FINANCE", "NBFC", "FINANCIAL", "LENDING", "MICROFINANCE"])
    # Fallback: well-known bank tickers
    return ticker in {
        "HDFCBANK","ICICIBANK","AXISBANK","KOTAKBANK","SBIN","BANDHANBNK",
        "INDUSINDBK","FEDERALBNK","RBLBANK","IDFCFIRSTB","YESBANK","PNB",
        "CANBK","BANKBARODA","UNIONBANK","AUBANK","DCBBANK","KARNATAKABAN"
    }


def _requests_html(url: str) -> Optional[str]:
    try:
        r = requests.get(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/91.0.4472.124 Safari/537.36"
        }, timeout=15)
        if r.status_code == 200:
            return r.text
    except Exception:
        pass
    return None


def _parse_table(table) -> Optional[pd.DataFrame]:
    try:
        thead = table.find("thead")
        if not thead:
            return None
        headers = [th.get_text(strip=True) for th in thead.find_all("th")]
        if len(headers) < 2:
            return None
        if not headers[0]:
            headers[0] = "Metric"
        rows = []
        tbody = table.find("tbody")
        if not tbody:
            return None
        for tr in tbody.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all("td")]
            if len(cells) == len(headers):
                rows.append(cells)
        if not rows:
            return None
        df = pd.DataFrame(rows, columns=headers)
        df.set_index("Metric", inplace=True)
        return df
    except Exception:
        return None


def _scrape_section(ticker: str, section_id: str) -> Optional[pd.DataFrame]:
    """Scrape a specific section from Screener.in (consolidated preferred)."""
    urls = [
        f"https://www.screener.in/company/{ticker}/consolidated/",
        f"https://www.screener.in/company/{ticker}/",
    ]
    dfs = []
    for url in urls:
        html = _requests_html(url)
        if not html:
            continue
        soup = BeautifulSoup(html, "html.parser")
        section = soup.find("section", {"id": section_id})
        if not section:
            continue
        table = section.find("table", class_="data-table")
        if not table:
            continue
        df = _parse_table(table)
        if df is not None and not df.empty:
            dfs.append(df)
    if not dfs:
        return None
    if len(dfs) == 1:
        return dfs[0]
    # Pick the one with more non-zero profit rows
    best, best_n = dfs[0], 0
    for d in dfs:
        idx, _ = _find_row(d, "profit")
        if idx:
            s = _extract_metric(idx, d)
            n = int((s != 0).sum())
            if n > best_n:
                best, best_n = d, n
    return best


def _clean_numeric(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.replace(",", "", regex=False)
         .str.replace(r"[^\d.-]", "", regex=True)
         .str.strip()
         .replace("", "0")
         .pipe(pd.to_numeric, errors="coerce")
         .fillna(0)
    )


def _find_row(df: pd.DataFrame, name: str):
    candidates = {
        "profit": [
            ("Operating Profit", "Operating Profit"), ("OP", "Operating Profit"),
            ("EBIT", "Operating Profit"), ("EBITDA", "Operating Profit"),
            ("Financing Profit", "Financing Profit"),
            ("Interest Income", "Interest Income"),
            ("Net Interest Income", "Net Interest Income"),
        ],
        "sales": [
            ("Sales", "Sales"), ("Revenue", "Sales"), ("Net Sales", "Sales"),
            ("Total Income", "Sales"), ("Interest Earned", "Interest Earned"),
            ("Income", "Income"),
        ],
        "other_income": [
            ("Other Income", "Other Income"),
            ("Other Operating Income", "Other Income"),
        ],
    }
    target = name.lower()
    for keyword, display in candidates.get(target, []):
        for idx in df.index:
            if keyword.lower() in idx.lower():
                return idx, display
    idx_lower = [i.lower().strip() for i in df.index]
    matches = difflib.get_close_matches(target, idx_lower, n=1, cutoff=0.6)
    if matches:
        matched = df.index[idx_lower.index(matches[0])]
        label = "Operating Profit" if target == "profit" else "Sales"
        return matched, label
    return None, "Unknown"


def _extract_metric(row_idx, df: pd.DataFrame) -> pd.Series:
    if row_idx is None:
        return pd.Series(dtype=float)
    series   = df.loc[row_idx]
    raw_cols = df.columns.tolist()
    years, vals = [], []
    for col in raw_cols:
        col_str = str(col).strip()
        if col_str.upper() == "TTM":
            continue
        m = re.search(r"\d{2,4}", col_str)
        if m:
            yr_str = m.group(0)
            yr = int(yr_str)
            if len(yr_str) == 2:
                yr += 2000
            if 2000 <= yr <= 2100:
                years.append(yr)
                vals.append(series[col])
                continue
        try:
            yr = int(col_str)
            if 2000 <= yr <= 2100:
                years.append(yr)
                vals.append(series[col])
        except Exception:
            continue
    if not years:
        return pd.Series(dtype=float)
    clean = _clean_numeric(pd.Series(vals, index=years))
    return clean[clean != 0].dropna()


def _extract_quarterly_metric(row_idx, df: pd.DataFrame) -> pd.Series:
    if row_idx is None:
        return pd.Series(dtype=float)
    series   = df.loc[row_idx]
    raw_cols = df.columns.tolist()
    month_num  = {"Mar": 3, "Jun": 6, "Sep": 9, "Dec": 12}
    month_days = {"Mar": 31, "Jun": 30, "Sep": 30, "Dec": 31}
    q_dates, vals = [], []
    for col in raw_cols:
        col_str = str(col).strip()
        m = re.match(r"([A-Za-z]{3})\s+(\d{4})", col_str)
        if m:
            mon_abbr = m.group(1)
            year     = int(m.group(2))
            if mon_abbr in month_num:
                mon = month_num[mon_abbr]
                day = month_days[mon_abbr]
                try:
                    q_date = dt_module.date(year, mon, day)
                    q_dates.append(pd.to_datetime(q_date))
                    vals.append(series[col])
                except ValueError:
                    continue
    if not q_dates:
        return pd.Series(dtype=float)
    return pd.Series(_clean_numeric(pd.Series(vals)).values, index=q_dates).dropna()


def _get_cached(ticker: str, kind: str, force: bool) -> Optional[pd.DataFrame]:
    """Load from disk cache if available."""
    path = os.path.join(DATA_DIR, f"{ticker}_{kind}.csv")
    if not force and os.path.exists(path):
        try:
            return pd.read_csv(path, index_col=0)
        except Exception:
            pass
    return None


def _save_cache(ticker: str, kind: str, df: pd.DataFrame):
    try:
        df.to_csv(os.path.join(DATA_DIR, f"{ticker}_{kind}.csv"))
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────
#  CORE ANALYSIS (blocking)
# ─────────────────────────────────────────────────────────────
def _analyze_ticker(ticker: str, fy_start: int, force: bool, include_other_income: bool) -> Dict:
    ticker = ticker.strip().upper()

    # ── P&L data
    pl_df = _get_cached(ticker, "pl", force)
    if pl_df is None:
        pl_df = _scrape_section(ticker, "profit-loss")
        if pl_df is not None:
            _save_cache(ticker, "pl", pl_df)
    if pl_df is None or pl_df.empty:
        return {"ticker": ticker, "error": "Could not scrape P&L from Screener.in"}

    is_bank = is_bank_or_finance(ticker)

    profit_row, profit_label = _find_row(pl_df, "profit")
    if not profit_row:
        return {"ticker": ticker, "error": "No Operating Profit row found"}

    sales_row, sales_label = _find_row(pl_df, "sales")
    if is_bank and sales_row is None and len(pl_df) > 0:
        first_idx = pl_df.index[0]
        if any(w in first_idx.lower() for w in ["income", "total", "revenue", ""]):
            sales_row, sales_label = first_idx, "Total Income"

    other_row = None
    if include_other_income and is_bank:
        other_row, _ = _find_row(pl_df, "other_income")

    # ── local extract that closes over pl_df
    def _ext(row_idx):
        if row_idx is None:
            return pd.Series(dtype=float)
        series   = pl_df.loc[row_idx].iloc[1:]
        raw_cols = pl_df.columns[1:]
        years, vals = [], []
        for i, col in enumerate(raw_cols):
            col_str = str(col).strip()
            if col_str.upper() == "TTM":
                continue
            m = re.search(r"\d{2,4}", col_str)
            if m:
                yr = int(m.group())
                if 2000 <= yr <= 2100:
                    years.append(yr)
                    vals.append(series.iloc[i])
                    continue
            try:
                yr = int(col_str)
                if 2000 <= yr <= 2100:
                    years.append(yr)
                    vals.append(series.iloc[i])
            except Exception:
                continue
        if not years:
            return pd.Series(dtype=float)
        clean = _clean_numeric(pd.Series(vals, index=years))
        return clean[clean != 0].dropna()

    profit_s  = _ext(profit_row)
    sales_s   = _ext(sales_row) if sales_row else pd.Series(dtype=float)
    other_s   = _ext(other_row) if other_row else pd.Series(dtype=float)

    if include_other_income and not other_s.empty and is_bank:
        common = profit_s.index.intersection(other_s.index)
        if len(common) >= 2:
            profit_s    = profit_s.loc[common] + other_s.loc[common]
            profit_label = f"{profit_label} + Other Income"

    if profit_s.empty:
        return {"ticker": ticker, "error": "No numeric Profit data found"}

    # ── Price history via yfinance
    try:
        start_str = f"{fy_start}-04-01"
        end_str   = f"{dt_module.date.today().year + 1}-03-31"
        price_df  = yf.download(f"{ticker}.NS", start=start_str, end=end_str, progress=False)
    except Exception as exc:
        return {"ticker": ticker, "error": f"yfinance error: {exc}"}

    if price_df is None or price_df.empty:
        return {"ticker": ticker, "error": "No price data from yfinance"}

    if isinstance(price_df.columns, pd.MultiIndex):
        price_df.columns = price_df.columns.get_level_values(0)

    price_df["FY_Year"] = price_df.index.year
    price_df.loc[price_df.index.month <= 3, "FY_Year"] -= 1
    fy_end_price   = price_df.groupby("FY_Year")["Close"].last().round(2)
    current_price  = round(float(price_df["Close"].iloc[-1]), 2)

    # ── OP model
    common_yrs = sorted(profit_s.index.intersection(fy_end_price.index))
    if len(common_yrs) < 2:
        return {"ticker": ticker, "error": "Need ≥2 overlapping years of data"}

    profit_v  = np.array(profit_s.loc[common_yrs]).flatten()
    price_v   = np.array(fy_end_price.loc[common_yrs]).flatten()
    model_op  = LinearRegression().fit(profit_v.reshape(-1, 1), price_v)
    pred_op   = round(float(model_op.predict([[profit_v[-1]]])[0]), 2)
    gain_op   = round((pred_op - current_price) / current_price * 100, 2) if current_price else 0
    r2_op     = round(model_op.score(profit_v.reshape(-1, 1), price_v), 3)
    eq_op     = (f"Price = {round(model_op.intercept_,2)} + "
                 f"{round(model_op.coef_[0],6)} × {profit_label.split()[0]}")

    ratio_op  = np.round(price_v / profit_v, 4)
    profit_series_out = [
        {"year": int(y), "profit": round(float(profit_v[i]),2),
         "price": round(float(price_v[i]),2), "ratio": round(float(ratio_op[i]),4)}
        for i, y in enumerate(common_yrs)
    ]

    hist_op = {}
    for i, yr in enumerate(common_yrs):
        fair = round(float(model_op.predict([[profit_v[i]]])[0]), 2)
        actual = round(float(price_v[i]), 2)
        mp   = round((actual - fair) / fair * 100, 1) if fair else 0
        hist_op[str(yr)] = {
            "profit": round(float(profit_v[i]),2), "fair": fair,
            "actual": actual, "misprice_pct": mp,
            "status": "Overvalued" if mp > 20 else ("Undervalued" if mp < -20 else "Fair"),
        }

    # ── Sales model (optional)
    sales_out = {}
    has_sales = False
    if not sales_s.empty:
        sy = sorted(sales_s.index.intersection(fy_end_price.index))
        if len(sy) >= 2:
            has_sales  = True
            sv = np.array(sales_s.loc[sy]).flatten()
            pv = np.array(fy_end_price.loc[sy]).flatten()
            model_s  = LinearRegression().fit(sv.reshape(-1, 1), pv)
            pred_s   = round(float(model_s.predict([[sv[-1]]])[0]), 2)
            gain_s   = round((pred_s - current_price) / current_price * 100, 2) if current_price else 0
            r2_s     = round(model_s.score(sv.reshape(-1, 1), pv), 3)
            eq_s     = (f"Price = {round(model_s.intercept_,2)} + "
                        f"{round(model_s.coef_[0],6)} × {sales_label.split()[0]}")
            ratio_s  = np.round(pv / sv, 4)
            sales_series_out = [
                {"year": int(y), "sales": round(float(sv[i]),2),
                 "price": round(float(pv[i]),2), "ratio": round(float(ratio_s[i]),4)}
                for i, y in enumerate(sy)
            ]
            hist_s = {}
            for i, yr in enumerate(sy):
                fair   = round(float(model_s.predict([[sv[i]]])[0]), 2)
                actual = round(float(pv[i]), 2)
                mp     = round((actual - fair) / fair * 100, 1) if fair else 0
                hist_s[str(yr)] = {
                    "sales": round(float(sv[i]),2), "fair": fair,
                    "actual": actual, "misprice_pct": mp,
                    "status": "Overvalued" if mp > 20 else ("Undervalued" if mp < -20 else "Fair"),
                }
            sales_out = {
                "pred_price": pred_s, "gain_pct": gain_s, "r2": r2_s,
                "eq": eq_s, "label": sales_label,
                "series": sales_series_out, "historical": hist_s,
            }

    # ── Full price history for chart
    price_hist = [
        {"date": str(idx.date()), "close": round(float(row["Close"]), 2)}
        for idx, row in price_df.iterrows()
        if pd.notna(row["Close"])
    ]

    # ── Quarterly / TTM
    ttm_out = {}
    has_ttm = False
    qr_df = _get_cached(ticker, "qr", force)
    if qr_df is None:
        qr_df = _scrape_section(ticker, "quarters")
        if qr_df is not None:
            _save_cache(ticker, "qr", qr_df)

    if qr_df is not None and not qr_df.empty:
        p_row_q, _ = _find_row(qr_df, "profit")
        s_row_q, s_lbl_q = _find_row(qr_df, "sales")
        if is_bank and s_row_q is None and len(qr_df) > 0:
            fi = qr_df.index[0]
            if any(w in fi.lower() for w in ["income","total","revenue",""]):
                s_row_q, s_lbl_q = fi, "Total Income"

        if p_row_q:
            profit_q = _extract_quarterly_metric(p_row_q, qr_df).sort_index()
            if include_other_income and is_bank:
                or_q_row, _ = _find_row(qr_df, "other_income")
                if or_q_row:
                    oi_q = _extract_quarterly_metric(or_q_row, qr_df)
                    cq = profit_q.index.intersection(oi_q.index)
                    if len(cq):
                        profit_q.loc[cq] += oi_q.loc[cq]

            if len(profit_q) >= 4:
                price_q = price_df["Close"].resample("QE").last().round(2)
                ttm_vals = [profit_q.iloc[i-3:i+1].sum() for i in range(3, len(profit_q))]
                ttm_idx  = profit_q.index[3:]
                ttm_s    = pd.Series(ttm_vals, index=ttm_idx)
                common_t = sorted(ttm_s.index.intersection(price_q.index))
                if len(common_t) >= 2:
                    has_ttm = True
                    tv  = np.array(ttm_s.loc[common_t]).flatten()
                    pqv = np.array(price_q.loc[common_t]).flatten()
                    model_t = LinearRegression().fit(tv.reshape(-1,1), pqv)
                    pred_t  = round(float(model_t.predict([[tv[-1]]])[0]), 2)
                    gain_t  = round((pred_t - current_price)/current_price*100, 2) if current_price else 0
                    r2_t    = round(model_t.score(tv.reshape(-1,1), pqv), 3)
                    eq_t    = (f"Price = {round(model_t.intercept_,2)} + "
                               f"{round(model_t.coef_[0],6)} × {profit_label.split()[0]} TTM")
                    ratio_t = np.round(pqv / tv, 4)
                    ttm_series_out = [
                        {"quarter": str(common_t[i].date()),
                         "ttm_profit": round(float(tv[i]),2),
                         "price": round(float(pqv[i]),2),
                         "ratio": round(float(ratio_t[i]),4)}
                        for i in range(len(common_t))
                    ]
                    hist_t = {}
                    for i, dt_q in enumerate(common_t):
                        fair   = round(float(model_t.predict([[tv[i]]])[0]), 2)
                        actual = round(float(pqv[i]), 2)
                        mp     = round((actual - fair)/fair*100, 1) if fair else 0
                        hist_t[str(dt_q.date())] = {
                            "ttm": round(float(tv[i]),2), "fair": fair,
                            "actual": actual, "misprice_pct": mp,
                            "status": "Overvalued" if mp > 20 else ("Undervalued" if mp < -20 else "Fair"),
                        }
                    ttm_out = {
                        "pred_price": pred_t, "gain_pct": gain_t, "r2": r2_t,
                        "eq": eq_t, "label": profit_label,
                        "series": ttm_series_out, "historical": hist_t,
                    }

        # TTM Sales
        if s_row_q:
            sales_q = _extract_quarterly_metric(s_row_q, qr_df).sort_index()
            if len(sales_q) >= 4:
                price_q2 = price_df["Close"].resample("QE").last().round(2)
                ttm_sv   = [sales_q.iloc[i-3:i+1].sum() for i in range(3, len(sales_q))]
                ttm_si   = sales_q.index[3:]
                ttm_ss   = pd.Series(ttm_sv, index=ttm_si)
                common_ts = sorted(ttm_ss.index.intersection(price_q2.index))
                if len(common_ts) >= 2:
                    sv2  = np.array(ttm_ss.loc[common_ts]).flatten()
                    pv2  = np.array(price_q2.loc[common_ts]).flatten()
                    m_ts = LinearRegression().fit(sv2.reshape(-1,1), pv2)
                    pred_ts  = round(float(m_ts.predict([[sv2[-1]]])[0]), 2)
                    gain_ts  = round((pred_ts - current_price)/current_price*100, 2) if current_price else 0
                    r2_ts    = round(m_ts.score(sv2.reshape(-1,1), pv2), 3)
                    eq_ts    = (f"Price = {round(m_ts.intercept_,2)} + "
                                f"{round(m_ts.coef_[0],6)} × {s_lbl_q.split()[0]} TTM")
                    ratio_ts = np.round(pv2/sv2, 4)
                    ttm_sales_series = [
                        {"quarter": str(common_ts[i].date()),
                         "ttm_sales": round(float(sv2[i]),2),
                         "price": round(float(pv2[i]),2),
                         "ratio": round(float(ratio_ts[i]),4)}
                        for i in range(len(common_ts))
                    ]
                    hist_ts = {}
                    for i, dt_q in enumerate(common_ts):
                        fair   = round(float(m_ts.predict([[sv2[i]]])[0]), 2)
                        actual = round(float(pv2[i]), 2)
                        mp     = round((actual-fair)/fair*100,1) if fair else 0
                        hist_ts[str(dt_q.date())] = {
                            "ttm_sales": round(float(sv2[i]),2), "fair": fair,
                            "actual": actual, "misprice_pct": mp,
                            "status": "Overvalued" if mp>20 else ("Undervalued" if mp<-20 else "Fair"),
                        }
                    ttm_out["sales"] = {
                        "pred_price": pred_ts, "gain_pct": gain_ts, "r2": r2_ts,
                        "eq": eq_ts, "label": s_lbl_q,
                        "series": ttm_sales_series, "historical": hist_ts,
                    }

    return {
        "ticker":        ticker,
        "current_price": current_price,
        "is_bank":       is_bank,
        "profit_label":  profit_label,
        "years_count":   len(common_yrs),
        # OP model
        "op": {
            "pred_price": pred_op, "gain_pct": gain_op,
            "r2": r2_op, "eq": eq_op,
            "series": profit_series_out, "historical": hist_op,
        },
        # Sales model
        "has_sales": has_sales,
        "sales":     sales_out if has_sales else {},
        # TTM
        "has_ttm": has_ttm,
        "ttm":     ttm_out if has_ttm else {},
        # Full daily price history
        "price_history": price_hist,
    }


# ─────────────────────────────────────────────────────────────
#  PYDANTIC MODELS
# ─────────────────────────────────────────────────────────────
class SmaAnalyzeRequest(BaseModel):
    tickers:              List[str]
    fy_start:             int  = 2014
    force_scrape:         bool = False
    include_other_income: bool = True


# ─────────────────────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────────────────────
@router.get("/api/sma/tickers")
async def sma_tickers(q: str = ""):
    df = _load_nse_df()
    if df.empty:
        return {"tickers": [], "total": 0}
    q = q.strip().upper()
    if q:
        mask = df["symbol"].str.contains(q, na=False)
        # Also search company name if available
        if "company_name" in df.columns:
            mask = mask | df["company_name"].str.upper().str.contains(q, na=False)
        filtered = df[mask].head(30)
    else:
        filtered = df.head(30)
    return {
        "total": len(df),
        "tickers": [
            {"symbol": row["symbol"],
             "name":   row["company_name"] if row["company_name"] else row["symbol"]}
            for _, row in filtered.iterrows()
        ],
    }


@router.post("/api/sma/tickers/reload")
async def sma_tickers_reload():
    """Force-reload ticker list from tickers.csv"""
    global _nse_df
    _nse_df = None
    df = _load_nse_df()
    return {"ok": True, "total": len(df)}


@router.post("/api/sma/analyze")
async def sma_analyze(req: SmaAnalyzeRequest):
    tickers = [t.strip().upper() for t in req.tickers if t.strip()][:10]
    if not tickers:
        raise HTTPException(400, "No tickers provided.")

    results, failed = [], []
    for ticker in tickers:
        try:
            res = await asyncio.to_thread(
                _analyze_ticker, ticker, req.fy_start, req.force_scrape,
                req.include_other_income
            )
            if "error" in res:
                failed.append({"ticker": ticker, "error": res["error"]})
            else:
                results.append(res)
        except Exception as exc:
            failed.append({"ticker": ticker, "error": str(exc)})

    return {"results": results, "failed": failed, "count": len(results)}


@router.delete("/api/sma/cache")
async def sma_clear_cache():
    removed = 0
    for f in os.listdir(DATA_DIR):
        if f.endswith("_pl.csv") or f.endswith("_qr.csv"):
            try:
                os.remove(os.path.join(DATA_DIR, f))
                removed += 1
            except Exception:
                pass
    return {"removed": removed}