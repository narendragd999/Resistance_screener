"""
Naked Call Scanner & P&L Dashboard
Live data via NSE India public API (free, no auth required)
Fallback spot price via yfinance

Install deps:
    pip install flask requests yfinance
"""

from flask import Flask, render_template, jsonify, request
import math, os, time, requests
from datetime import datetime, date
from functools import lru_cache

app = Flask(__name__)

# ── NSE session (required to bypass NSE's anti-scrape cookies) ──────────────
NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept":          "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer":         "https://www.nseindia.com/",
    "Origin":          "https://www.nseindia.com",
    "Connection":      "keep-alive",
}

_nse_session   = None
_session_born  = 0
SESSION_TTL    = 300   # refresh cookie every 5 min


def get_nse_session() -> requests.Session:
    """Return a warmed-up NSE session with valid cookies."""
    global _nse_session, _session_born
    if _nse_session is None or (time.time() - _session_born) > SESSION_TTL:
        s = requests.Session()
        s.headers.update(NSE_HEADERS)
        # Warm up: hit homepage + option-chain page to seed cookies
        s.get("https://www.nseindia.com", timeout=10)
        s.get("https://www.nseindia.com/option-chain", timeout=10)
        _nse_session = s
        _session_born = time.time()
    return _nse_session


def nse_get(path: str, params: dict = None) -> dict:
    """GET from NSE API with automatic session retry."""
    url = f"https://www.nseindia.com/api/{path}"
    for attempt in range(3):
        try:
            s = get_nse_session()
            r = s.get(url, params=params, timeout=15)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == 2:
                raise RuntimeError(f"NSE API error ({path}): {e}")
            global _nse_session
            _nse_session = None   # force session refresh
            time.sleep(1)


# ── NSE symbol helpers ──────────────────────────────────────────────────────
# Symbols used in NSE option-chain API
NSE_INDEX_NAMES = {
    "NIFTY":      "NIFTY",
    "BANKNIFTY":  "BANKNIFTY",
    "FINNIFTY":   "FINNIFTY",
    "MIDCPNIFTY": "MIDCPNIFTY",
}

# Symbols used in NSE quote / allIndices
NSE_QUOTE_SYMBOLS = {
    "NIFTY":      "NIFTY 50",
    "BANKNIFTY":  "NIFTY BANK",
    "FINNIFTY":   "NIFTY FIN SERVICE",
    "MIDCPNIFTY": "NIFTY MIDCAP SELECT",
}


# ── Black-Scholes helpers ───────────────────────────────────────────────────
def norm_cdf(x):
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def bs_call_price(S, K, T, r, sigma):
    if T <= 0:
        return max(0, S - K)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)


def bs_delta(S, K, T, r, sigma):
    if T <= 0:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return norm_cdf(d1)


def bs_theta(S, K, T, r, sigma):
    if T <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    theta = (
        -(S * math.exp(-0.5 * d1 ** 2) * sigma) / (2 * math.sqrt(2 * math.pi * T))
        - r * K * math.exp(-r * T) * norm_cdf(d2)
    )
    return theta / 365


def bs_vega(S, K, T, r, sigma):
    if T <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return S * math.exp(-0.5 * d1 ** 2) / math.sqrt(2 * math.pi) * math.sqrt(T) / 100


def implied_vol(market_price, S, K, T, r, initial_sigma=0.2):
    """Newton-Raphson IV solver."""
    sigma = initial_sigma
    for _ in range(100):
        price = bs_call_price(S, K, T, r, sigma)
        vega  = bs_vega(S, K, T, r, sigma) * 100
        if abs(vega) < 1e-10:
            break
        sigma -= (price - market_price) / vega
        sigma  = max(0.001, min(sigma, 5.0))
    return sigma


# ── Spot price helpers ──────────────────────────────────────────────────────
def get_spot_nse(symbol: str) -> float:
    """Fetch index LTP from NSE allIndices endpoint."""
    target = NSE_QUOTE_SYMBOLS.get(symbol.upper(), symbol.upper())
    data = nse_get("allIndices")
    for idx in data.get("data", []):
        if idx.get("indexSymbol") == target or idx.get("index") == target:
            return float(idx["last"])
    raise ValueError(f"Index '{symbol}' not found in NSE allIndices")


def get_spot_ohlc_nse(symbol: str) -> dict:
    """Return spot with OHLC for the /api/quote route."""
    target = NSE_QUOTE_SYMBOLS.get(symbol.upper(), symbol.upper())
    data = nse_get("allIndices")
    for idx in data.get("data", []):
        if idx.get("indexSymbol") == target or idx.get("index") == target:
            ltp    = float(idx["last"])
            open_  = float(idx.get("open", ltp))
            high   = float(idx.get("high", ltp))
            low    = float(idx.get("low", ltp))
            prev   = float(idx.get("previousClose", ltp))
            return {
                "symbol":     symbol,
                "ltp":        ltp,
                "open":       open_,
                "high":       high,
                "low":        low,
                "close":      prev,
                "change":     round(ltp - prev, 2),
                "change_pct": round((ltp - prev) / prev * 100, 2) if prev else 0,
            }
    raise ValueError(f"Index '{symbol}' not found")


# ── Option chain from NSE ───────────────────────────────────────────────────
def fetch_nse_option_chain(symbol: str) -> dict:
    """
    Fetch the full option chain JSON from NSE.
    Returns the raw NSE response dict.
    """
    nse_sym = NSE_INDEX_NAMES.get(symbol.upper(), symbol.upper())
    return nse_get("option-chain-indices", params={"symbol": nse_sym})


def parse_expiries(nse_data: dict) -> list:
    return nse_data.get("records", {}).get("expiryDates", [])


def parse_chain_for_expiry(nse_data: dict, expiry_str: str, spot: float) -> list:
    """
    Filter NSE option-chain data for a specific expiry and compute Greeks.
    expiry_str: e.g. "27-Jun-2024"  (NSE format) or "2024-06-27"
    """
    # normalise to NSE format "DD-Mon-YYYY"
    try:
        if "-" in expiry_str and len(expiry_str) == 10 and expiry_str[4] == "-":
            # ISO format → NSE format
            d = datetime.strptime(expiry_str, "%Y-%m-%d")
            exp_nse = d.strftime("%d-%b-%Y")
        else:
            exp_nse = expiry_str
        exp_date = datetime.strptime(exp_nse, "%d-%b-%Y").date()
    except ValueError:
        raise ValueError(f"Cannot parse expiry: {expiry_str}")

    today = date.today()
    T = max((exp_date - today).days / 365, 1 / 365)
    r = 0.065

    results = []
    data_records = nse_data.get("records", {}).get("data", [])
    for rec in data_records:
        if rec.get("expiryDate") != exp_nse:
            continue
        ce = rec.get("CE")
        if not ce:
            continue

        K   = float(ce.get("strikePrice", 0))
        ltp = float(ce.get("lastPrice",   0))
        oi  = int(ce.get("openInterest",  0))
        vol = int(ce.get("totalTradedVolume", 0))

        if ltp <= 0 or K <= 0:
            continue

        iv    = implied_vol(ltp, spot, K, T, r)
        delta = bs_delta(spot, K, T, r, iv)
        theta = bs_theta(spot, K, T, r, iv)
        vega  = bs_vega(spot, K, T, r, iv)
        theo  = bs_call_price(spot, K, T, r, iv)

        # Scanner signals
        score = 0
        if delta <= 0.20: score += 1
        if delta <= 0.15: score += 1
        if iv    > 0.20:  score += 1
        if oi    > 100000: score += 1

        signal = "SELL" if score >= 3 else ("WATCH" if score == 2 else "NEUTRAL")

        results.append({
            "strike":        K,
            "ltp":           round(ltp, 2),
            "theo":          round(theo, 2),
            "iv":            round(iv * 100, 2),
            "delta":         round(delta, 3),
            "theta":         round(theta, 2),
            "vega":          round(vega, 2),
            "oi":            oi,
            "volume":        vol,
            "signal":        signal,
            "score":         score,
            "tradingsymbol": ce.get("identifier", ""),
        })

    results.sort(key=lambda x: x["strike"])
    return results


# ── Routes ──────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/quote/<symbol>")
def get_quote(symbol):
    """Live LTP + OHLC for index — sourced from NSE allIndices."""
    try:
        return jsonify(get_spot_ohlc_nse(symbol))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/expiries/<symbol>")
def expiries(symbol):
    """Return next 6 expiry dates for the given index."""
    try:
        data = fetch_nse_option_chain(symbol)
        raw  = parse_expiries(data)          # ["27-Jun-2024", ...]
        # Convert to ISO for consistency with the front-end
        iso  = []
        for d in raw[:6]:
            try:
                iso.append(datetime.strptime(d, "%d-%b-%Y").strftime("%Y-%m-%d"))
            except ValueError:
                iso.append(d)
        return jsonify({"expiries": iso})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/option-chain/<symbol>/<expiry>")
def option_chain(symbol, expiry):
    """
    Full option chain with Greeks + scanner signals.
    expiry: YYYY-MM-DD
    """
    try:
        spot = get_spot_nse(symbol)
        data = fetch_nse_option_chain(symbol)
        chain = parse_chain_for_expiry(data, expiry, spot)
        if not chain:
            return jsonify({"error": "No CE data found for given expiry"}), 404

        exp_date = datetime.strptime(expiry, "%Y-%m-%d").date()
        t_days   = (exp_date - date.today()).days

        return jsonify({
            "spot":   round(spot, 2),
            "expiry": expiry,
            "symbol": symbol,
            "T_days": t_days,
            "chain":  chain,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/scanner")
def scanner():
    """
    Best naked-call opportunities filtered by delta range.
    Query params: symbol, expiry, min_delta, max_delta
    """
    symbol    = request.args.get("symbol",    "NIFTY")
    expiry    = request.args.get("expiry",    "")
    min_delta = float(request.args.get("min_delta", 0.05))
    max_delta = float(request.args.get("max_delta", 0.20))

    if not expiry:
        # Auto-pick nearest expiry
        try:
            data = fetch_nse_option_chain(symbol)
            raw  = parse_expiries(data)
            expiry = datetime.strptime(raw[0], "%d-%b-%Y").strftime("%Y-%m-%d")
        except Exception as e:
            return jsonify({"error": f"Could not fetch expiries: {e}"}), 500

    try:
        spot  = get_spot_nse(symbol)
        data  = fetch_nse_option_chain(symbol)
        chain = parse_chain_for_expiry(data, expiry, spot)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    filtered = [
        row for row in chain
        if min_delta <= row["delta"] <= max_delta and row["ltp"] > 0
    ]
    filtered.sort(key=lambda x: x["score"], reverse=True)

    return jsonify({
        "spot":    round(spot, 2),
        "symbol":  symbol,
        "expiry":  expiry,
        "results": filtered[:10],
    })


@app.route("/api/pl")
def pl_calc():
    """P&L computation endpoint (pure maths, no live data needed)."""
    K      = float(request.args.get("strike",    0))
    prem   = float(request.args.get("premium",   0))
    lot    = int(request.args.get("lot_size",    50))
    lots   = int(request.args.get("lots",         1))
    brok   = float(request.args.get("brokerage", 40))   # Zerodha-style flat ₹20 each leg
    sl_pct = float(request.args.get("sl_pct",   100))

    total_prem = prem * lot * lots
    max_profit = total_prem - brok
    be         = K + prem
    sl_trigger = prem * (1 + sl_pct / 100)
    sl_loss    = (sl_trigger - prem) * lot * lots + brok

    scenarios = []
    for pct in range(-20, 35, 5):
        ep  = K * (1 + pct / 100)
        cp  = max(0, ep - K) * lot * lots
        pnl = total_prem - cp - brok
        scenarios.append({
            "move_pct":  pct,
            "expiry_px": round(ep),
            "pnl":       round(pnl),
        })

    return jsonify({
        "total_premium": round(total_prem),
        "max_profit":    round(max_profit),
        "breakeven":     round(be, 2),
        "sl_trigger":    round(sl_trigger, 2),
        "sl_loss":       round(sl_loss),
        "scenarios":     scenarios,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)