import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import time

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Stock universe
# ─────────────────────────────────────────────────────────────────────────────
NIFTY50_STOCKS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "HINDUNILVR", "ICICIBANK",
    "KOTAKBANK", "BHARTIARTL", "ITC", "LT", "AXISBANK", "ASIANPAINT",
    "MARUTI", "TITAN", "SUNPHARMA", "ULTRACEMCO", "BAJFINANCE",
    "HCLTECH", "WIPRO", "NESTLEIND", "POWERGRID", "NTPC", "ONGC",
    "COALINDIA", "JSWSTEEL", "TMCV", "TATASTEEL", "SBIN",
    "INDUSINDBK", "TECHM", "DRREDDY", "CIPLA", "DIVISLAB", "EICHERMOT",
    "BAJAJFINSV", "BPCL", "GRASIM", "HEROMOTOCO", "HINDALCO",
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "BRITANNIA", "TATACONSUM",
    "SHRIRAMFIN", "LTM", "SBILIFE", "HDFCLIFE", "M&M", "BAJAJ-AUTO"
]

NIFTY_MIDCAP = [
    "PIDILITIND", "SIEMENS", "GODREJCP", "DABUR", "BERGEPAINT",
    "MUTHOOTFIN", "LUPIN", "TORNTPHARM", "HAVELLS", "MARICO",
    "NAUKRI", "COLPAL", "INDIGO", "DLF", "AUROPHARMA",
    "BANKBARODA", "FEDERALBNK", "IDFCFIRSTB", "CONCOR", "IRCTC",
    "VOLTAS", "ABCAPITAL", "ABFRL", "ACC", "ALKEM", "AMBUJACEM",
    "ASTRAL", "ATUL", "AUROPHARMA", "BALKRISIND", "BATAINDIA",
    "BEL", "BIOCON", "BOSCHLTD", "CANFINHOME", "CHOLAFIN",
    "CUMMINSIND", "DEEPAKNTR", "Dixon", "ESCORTS", "GMRAIRPORT"
]

ALL_STOCKS = list(set(NIFTY50_STOCKS + NIFTY_MIDCAP))

# ─────────────────────────────────────────────────────────────────────────────
# Sector proxy mapping  (NSE sector index tickers via Yahoo Finance)
# Each stock maps to its closest sector benchmark.
# ─────────────────────────────────────────────────────────────────────────────
SECTOR_MAP = {
    # Banking & Finance
    "HDFCBANK": "^NSEBANK", "ICICIBANK": "^NSEBANK", "KOTAKBANK": "^NSEBANK",
    "AXISBANK": "^NSEBANK", "SBIN": "^NSEBANK", "INDUSINDBK": "^NSEBANK",
    "BANKBARODA": "^NSEBANK", "FEDERALBNK": "^NSEBANK", "IDFCFIRSTB": "^NSEBANK",
    "BAJFINANCE": "^NSEBANK", "BAJAJFINSV": "^NSEBANK", "MUTHOOTFIN": "^NSEBANK",
    "CHOLAFIN": "^NSEBANK", "CANFINHOME": "^NSEBANK", "ABCAPITAL": "^NSEBANK",
    "SHRIRAMFIN": "^NSEBANK", "SBILIFE": "^NSEBANK", "HDFCLIFE": "^NSEBANK",
    # IT
    "TCS": "^CNXIT", "INFY": "^CNXIT", "HCLTECH": "^CNXIT",
    "WIPRO": "^CNXIT", "TECHM": "^CNXIT", "NAUKRI": "^CNXIT",
    # Pharma
    "SUNPHARMA": "^CNXPHARMA", "DRREDDY": "^CNXPHARMA", "CIPLA": "^CNXPHARMA",
    "DIVISLAB": "^CNXPHARMA", "LUPIN": "^CNXPHARMA", "TORNTPHARM": "^CNXPHARMA",
    "AUROPHARMA": "^CNXPHARMA", "ALKEM": "^CNXPHARMA", "BIOCON": "^CNXPHARMA",
    # Auto
    "MARUTI": "^CNXAUTO", "EICHERMOT": "^CNXAUTO", "HEROMOTOCO": "^CNXAUTO",
    "BAJAJ-AUTO": "^CNXAUTO", "TMCV": "^CNXAUTO", "ESCORTS": "^CNXAUTO",
    # FMCG
    "HINDUNILVR": "^CNXFMCG", "ITC": "^CNXFMCG", "NESTLEIND": "^CNXFMCG",
    "BRITANNIA": "^CNXFMCG", "DABUR": "^CNXFMCG", "MARICO": "^CNXFMCG",
    "GODREJCP": "^CNXFMCG", "COLPAL": "^CNXFMCG", "TATACONSUM": "^CNXFMCG",
    # Energy & Utilities
    "RELIANCE": "^CNXENERGY", "ONGC": "^CNXENERGY", "BPCL": "^CNXENERGY",
    "POWERGRID": "^CNXENERGY", "NTPC": "^CNXENERGY", "COALINDIA": "^CNXENERGY",
    # Metals & Materials
    "JSWSTEEL": "^CNXMETAL", "TATASTEEL": "^CNXMETAL", "HINDALCO": "^CNXMETAL",
    "AMBUJACEM": "^CNXMETAL", "ACC": "^CNXMETAL",
    # Infra & Real Estate
    "LT": "^CNXINFRA", "ADANIENT": "^CNXINFRA", "ADANIPORTS": "^CNXINFRA",
    "DLF": "^CNXINFRA", "GMRAIRPORT": "^CNXINFRA", "CONCOR": "^CNXINFRA",
    "IRCTC": "^CNXINFRA",
    # Consumer Discretionary / Others → default to Nifty500
}
DEFAULT_SECTOR = "^CRSLDX"   # Nifty 500 as fallback


class StockScreener:
    def __init__(self):
        self._cache = {}
        self._cache_time = {}
        self.cache_duration = 300  # 5-min cache

    # ── Cache helpers ────────────────────────────────────────────────────────
    def _get_cached(self, key):
        if key in self._cache:
            if time.time() - self._cache_time.get(key, 0) < self.cache_duration:
                return self._cache[key]
        return None

    def _set_cached(self, key, data):
        self._cache[key] = data
        self._cache_time[key] = time.time()

    # ── Data fetching ────────────────────────────────────────────────────────
    def get_stock_data(self, symbol: str, period: str = "6mo", interval: str = "1d"):
        cache_key = f"{symbol}_{period}_{interval}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached
        try:
            ticker = yf.Ticker(f"{symbol}.NS")
            data = ticker.history(period=period, interval=interval, auto_adjust=True)
            if data.empty:
                ticker = yf.Ticker(f"{symbol}.BO")
                data = ticker.history(period=period, interval=interval, auto_adjust=True)
            if data.empty:
                return None
            data.index = pd.to_datetime(data.index).tz_localize(None)
            self._set_cached(cache_key, data)
            return data
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return None

    def _get_index_data(self, ticker_sym: str, period: str = "6mo", interval: str = "1d"):
        """Fetch index data (Nifty, sector indices) — no .NS/.BO suffix."""
        cache_key = f"IDX_{ticker_sym}_{period}_{interval}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached
        try:
            data = yf.Ticker(ticker_sym).history(period=period, interval=interval, auto_adjust=True)
            if data.empty:
                return None
            data.index = pd.to_datetime(data.index).tz_localize(None)
            self._set_cached(cache_key, data)
            return data
        except Exception as e:
            logger.error(f"Error fetching index {ticker_sym}: {e}")
            return None

    # ── Indicator calculations ───────────────────────────────────────────────
    def calculate_ema(self, series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    def calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    def calculate_macd(self, series: pd.Series):
        ema12 = self.calculate_ema(series, 12)
        ema26 = self.calculate_ema(series, 26)
        macd = ema12 - ema26
        signal = self.calculate_ema(macd, 9)
        histogram = macd - signal
        return macd, signal, histogram

    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        hl = data["High"] - data["Low"]
        hc = (data["High"] - data["Close"].shift(1)).abs()
        lc = (data["Low"] - data["Close"].shift(1)).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        return tr.ewm(span=period, adjust=False).mean()

    def calculate_bollinger_bands(self, series: pd.Series, period: int = 20) -> tuple:
        """Returns (upper, middle, lower, bandwidth)."""
        mid = series.rolling(period).mean()
        std = series.rolling(period).std()
        upper = mid + 2 * std
        lower = mid - 2 * std
        bandwidth = (upper - lower) / mid * 100
        return upper, mid, lower, bandwidth

    # ── NEW: Weekly EMA9 alignment ───────────────────────────────────────────
    def check_weekly_ema_alignment(self, symbol: str) -> dict:
        """
        CONDITION 6: Higher timeframe alignment.
        Weekly close must be above weekly EMA9, and weekly EMA9 slope must
        be positive. Eliminates counter-trend daily trades.
        """
        data = self.get_stock_data(symbol, period="1y", interval="1wk")
        if data is None or len(data) < 12:
            return {"aligned": True, "reason": "Insufficient weekly data — skipping"}
        closes = data["Close"]
        ema9w = self.calculate_ema(closes, 9)
        curr  = float(closes.iloc[-1])
        e9w   = float(ema9w.iloc[-1])
        slope = (e9w - float(ema9w.iloc[-4])) / float(ema9w.iloc[-4]) * 100
        above = curr > e9w
        aligned = above and slope > 0
        return {
            "aligned":          aligned,
            "weekly_close":     round(curr, 2),
            "weekly_ema9":      round(e9w, 2),
            "weekly_ema_slope": round(slope, 3),
            "above_weekly_ema9": above,
        }

    # ── NEW: Sector momentum ─────────────────────────────────────────────────
    def check_sector_momentum(self, symbol: str) -> dict:
        """
        CONDITION 7: Sector index must be above its own EMA9 and sloping up.
        Strong stocks in weak sectors fail. Uses SECTOR_MAP for proxy.
        """
        sector_ticker = SECTOR_MAP.get(symbol.upper(), DEFAULT_SECTOR)
        data = self._get_index_data(sector_ticker, period="6mo")
        if data is None or len(data) < 15:
            return {"strong": True, "reason": "Sector data unavailable — skipping",
                    "sector_ticker": sector_ticker}
        closes = data["Close"]
        ema9s  = self.calculate_ema(closes, 9)
        curr   = float(closes.iloc[-1])
        e9s    = float(ema9s.iloc[-1])
        slope  = (e9s - float(ema9s.iloc[-4])) / float(ema9s.iloc[-4]) * 100
        strong = curr > e9s and slope > 0
        return {
            "strong":         strong,
            "sector_ticker":  sector_ticker,
            "sector_close":   round(curr, 2),
            "sector_ema9":    round(e9s, 2),
            "sector_slope":   round(slope, 3),
            "above_ema9":     curr > e9s,
        }

    # ── NEW: Relative strength vs Nifty50 ───────────────────────────────────
    def check_relative_strength(self, symbol: str, closes: pd.Series) -> dict:
        """
        CONDITION 8: Stock must outperform Nifty50 over the last 20 days.
        Leaders trend; laggards don't.
        """
        nifty = self._get_index_data("^NSEI", period="3mo")
        if nifty is None or len(nifty) < 22:
            return {"outperforming": True, "reason": "Nifty data unavailable — skipping"}
        nifty_closes = nifty["Close"]
        # Align lengths via last 20 common bars
        stock_ret = (float(closes.iloc[-1]) - float(closes.iloc[-21])) / float(closes.iloc[-21]) * 100
        nifty_ret = (float(nifty_closes.iloc[-1]) - float(nifty_closes.iloc[-21])) / float(nifty_closes.iloc[-21]) * 100
        rs = stock_ret - nifty_ret
        outperforming = rs > 0
        return {
            "outperforming":   outperforming,
            "stock_20d_ret":   round(stock_ret, 2),
            "nifty_20d_ret":   round(nifty_ret, 2),
            "relative_strength": round(rs, 2),
        }

    # ── Original: 9 EMA support bounce (enhanced) ───────────────────────────
    def detect_9ema_support(self, data: pd.DataFrame) -> dict:
        """
        CONDITION 4 (enhanced): Detects quality EMA9 bounce.
        Now also checks:
          • green_candle mandatory in signal (was computed but not required)
          • pullback_quality: touch candle had below-average volume (healthy dip)
          • touch_count: counts how many times EMA was touched in last 20 bars
            (>= 4 touches weakens support — penalised but not hard-blocked)
        """
        if len(data) < 15:
            return {"signal": False, "reason": "Insufficient data"}

        closes  = data["Close"]
        lows    = data["Low"]
        opens   = data["Open"]
        volumes = data["Volume"]
        ema9    = self.calculate_ema(closes, 9)

        curr_close = float(closes.iloc[-1])
        curr_ema9  = float(ema9.iloc[-1])
        curr_open  = float(opens.iloc[-1])

        price_above_ema = curr_close > curr_ema9
        proximity_pct   = (curr_close - curr_ema9) / curr_ema9 * 100
        ema_slope       = (float(ema9.iloc[-1]) - float(ema9.iloc[-4])) / float(ema9.iloc[-4]) * 100

        # Touch detection in last 5 days
        touch_found      = False
        touch_day        = None
        touch_vol_ratio  = 1.0   # vol on touch day vs 20-day avg
        avg_vol_20       = float(volumes.tail(20).mean()) if len(volumes) >= 20 else float(volumes.mean())

        for i in range(-1, -6, -1):
            low_i   = float(lows.iloc[i])
            ema_i   = float(ema9.iloc[i])
            close_i = float(closes.iloc[i])
            vol_i   = float(volumes.iloc[i])
            pct_from_ema = (low_i - ema_i) / ema_i * 100
            if -2.0 <= pct_from_ema <= 0.5 and close_i > ema_i:
                touch_found     = True
                touch_day       = abs(i)
                touch_vol_ratio = vol_i / avg_vol_20 if avg_vol_20 > 0 else 1.0
                break
            elif 0.0 <= (close_i - ema_i) / ema_i * 100 <= 1.5:
                touch_found     = True
                touch_day       = abs(i)
                touch_vol_ratio = vol_i / avg_vol_20 if avg_vol_20 > 0 else 1.0
                break

        # Pullback quality: touch on LOW volume = healthy dip, not panic
        pullback_quality_ok = touch_vol_ratio <= 1.0   # touch volume below average

        # Touch count in last 20 bars (too many = weakening support)
        touch_count = 0
        lookback    = min(20, len(data) - 1)
        for i in range(-1, -(lookback + 1), -1):
            low_i = float(lows.iloc[i])
            ema_i = float(ema9.iloc[i])
            if abs((low_i - ema_i) / ema_i * 100) <= 1.5:
                touch_count += 1
        fresh_support = touch_count <= 3   # 4+ touches = weakening

        # Green recovery candle (NOW REQUIRED in signal)
        green_candle = curr_close > curr_open

        signal = (
            price_above_ema
            and touch_found
            and ema_slope > 0
            and 0.0 <= proximity_pct <= 5.0
            and green_candle           # ← NEW: mandatory
            and pullback_quality_ok    # ← NEW: healthy pullback volume
            and fresh_support          # ← NEW: not overused support
        )

        return {
            "signal":              signal,
            "price_above_ema9":    price_above_ema,
            "proximity_pct":       round(proximity_pct, 2),
            "ema_slope_pct":       round(ema_slope, 3),
            "touch_found":         touch_found,
            "touch_day":           touch_day,
            "touch_vol_ratio":     round(touch_vol_ratio, 2),
            "pullback_quality_ok": pullback_quality_ok,
            "touch_count_20d":     touch_count,
            "fresh_support":       fresh_support,
            "green_candle":        green_candle,
            "curr_ema9":           round(curr_ema9, 2),
            "curr_close":          round(curr_close, 2),
        }

    # ── Market status (unchanged) ────────────────────────────────────────────
    def get_market_status(self) -> dict:
        try:
            nifty = yf.Ticker("^NSEI")
            data  = nifty.history(period="60d", interval="1d", auto_adjust=True)
            if data.empty:
                return {"status": "unknown", "healthy": True, "current": 0}

            closes      = data["Close"]
            current     = closes.iloc[-1]
            ema20       = self.calculate_ema(closes, 20).iloc[-1]
            ema50       = self.calculate_ema(closes, 50).iloc[-1]
            rsi         = self.calculate_rsi(closes).iloc[-1]
            above_ema20 = current > ema20
            above_ema50 = current > ema50
            week_chg    = ((current - closes.iloc[-6]) / closes.iloc[-6]) * 100

            if not above_ema20 and week_chg < -2:
                status = "breakdown"
            elif not above_ema20:
                status = "weak"
            elif above_ema20 and above_ema50 and rsi > 50:
                status = "bullish"
            else:
                status = "neutral"

            prev_close = closes.iloc[-2]
            day_chg    = ((current - prev_close) / prev_close) * 100

            return {
                "status":         status,
                "healthy":        status not in ("breakdown",),
                "current":        round(current, 2),
                "prev_close":     round(prev_close, 2),
                "day_change_pct": round(day_chg, 2),
                "week_change_pct": round(week_chg, 2),
                "ema20":          round(ema20, 2),
                "ema50":          round(ema50, 2),
                "rsi":            round(rsi, 1),
                "above_ema20":    bool(above_ema20),
                "above_ema50":    bool(above_ema50),
            }
        except Exception as e:
            logger.error(f"Market status error: {e}")
            return {"status": "unknown", "healthy": True, "current": 0}

    # ── Master confluence check (13 conditions total) ────────────────────────
    def check_confluence(self, symbol: str, market_healthy: bool = True) -> dict:
        data = self.get_stock_data(symbol)
        if data is None or len(data) < 30:
            return {
                "symbol": symbol,
                "error":  "Data unavailable",
                "all_conditions_met": False,
                "score":  0,
            }

        closes  = data["Close"]
        highs   = data["High"]
        lows    = data["Low"]
        volumes = data["Volume"]

        ema9        = self.calculate_ema(closes, 9)
        ema21       = self.calculate_ema(closes, 21)
        rsi_series  = self.calculate_rsi(closes)
        macd_line, macd_sig, macd_hist = self.calculate_macd(closes)
        atr         = self.calculate_atr(data)
        _, _, _, bb_bandwidth = self.calculate_bollinger_bands(closes)

        curr_close  = float(closes.iloc[-1])
        curr_ema9   = float(ema9.iloc[-1])
        curr_ema21  = float(ema21.iloc[-1])
        curr_rsi    = float(rsi_series.iloc[-1])
        curr_vol    = int(volumes.iloc[-1])
        avg_vol_20  = float(volumes.tail(20).mean())

        # ── Original 5 conditions ────────────────────────────────────────────

        # 1. Price above 9 EMA
        cond_price_above_ema9 = curr_close > curr_ema9

        # 2. RSI 55–70
        cond_rsi = 55.0 <= curr_rsi <= 70.0

        # 3. Volume > 1M
        cond_volume = curr_vol > 1_000_000

        # 4. 9 EMA support bounce (enhanced — now includes green candle,
        #    pullback quality, and fresh support check)
        ema9_check        = self.detect_9ema_support(data)
        cond_ema9_support = ema9_check["signal"]

        # 5. Macro alignment
        cond_macro = market_healthy

        # ── NEW condition 6: MACD histogram positive & rising ────────────────
        # Histogram must be above zero AND today's histogram > yesterday's.
        # Already computed in get_chart_data — zero extra fetch.
        curr_hist = float(macd_hist.iloc[-1])
        prev_hist = float(macd_hist.iloc[-2])
        cond_macd = curr_hist > 0 and curr_hist > prev_hist

        # ── NEW condition 7: Volume spike (1.5× avg) ─────────────────────────
        # Institutional participation check — not just liquidity.
        vol_ratio        = curr_vol / avg_vol_20 if avg_vol_20 > 0 else 0
        cond_vol_spike   = vol_ratio >= 1.5

        # ── NEW condition 8: Near 52-week high (within 15%) ──────────────────
        # Breakout momentum: stocks near highs have trend tailwind.
        w52_data    = self.get_stock_data(symbol, period="1y")
        high52      = float(w52_data["High"].max()) if w52_data is not None else curr_close
        low52       = float(w52_data["Low"].min())  if w52_data is not None else curr_close
        pct_from_h  = (curr_close - high52) / high52 * 100  # negative = below high
        cond_near_52wh = pct_from_h >= -15.0   # within 15% of 52-week high

        # ── NEW condition 9: Relative strength vs Nifty50 (20-day) ──────────
        rs_check           = self.check_relative_strength(symbol, closes)
        cond_rel_strength  = rs_check.get("outperforming", True)

        # ── NEW condition 10: Weekly EMA9 alignment ───────────────────────────
        weekly_check       = self.check_weekly_ema_alignment(symbol)
        cond_weekly_aligned = weekly_check.get("aligned", True)

        # ── NEW condition 11: Sector momentum ────────────────────────────────
        sector_check       = self.check_sector_momentum(symbol)
        cond_sector_strong = sector_check.get("strong", True)

        # ── NEW condition 12: Bollinger Band squeeze (low bandwidth = contraction
        # before expansion; bandwidth < 10 = squeeze in place) ────────────────
        curr_bw       = float(bb_bandwidth.iloc[-1]) if not pd.isna(bb_bandwidth.iloc[-1]) else 20.0
        prev_bw       = float(bb_bandwidth.iloc[-2]) if not pd.isna(bb_bandwidth.iloc[-2]) else 20.0
        cond_bb_squeeze = curr_bw < 15.0 and curr_bw <= prev_bw   # narrowing bands

        # ── NEW condition 13: First/second touch (≤ 3 touches in 20 bars) ────
        # Already computed inside detect_9ema_support as fresh_support.
        cond_fresh_touch = ema9_check.get("fresh_support", True)

        # ── Assemble conditions dict ─────────────────────────────────────────
        conditions = {
            # ── Original 5 ──
            "price_above_9ema": {
                "met":   cond_price_above_ema9,
                "label": "Price above 9 EMA (daily trend)",
                "value": f"₹{round(curr_close,2)} vs EMA ₹{round(curr_ema9,2)}",
                "group": "core",
            },
            "rsi_55_70": {
                "met":   cond_rsi,
                "label": "RSI (14) between 55–70",
                "value": f"RSI = {round(curr_rsi,1)}",
                "group": "core",
            },
            "volume_1m": {
                "met":   cond_volume,
                "label": "Volume > 1M shares (liquidity)",
                "value": f"{curr_vol:,} (avg {int(avg_vol_20):,})",
                "group": "core",
            },
            "ema9_support": {
                "met":   cond_ema9_support,
                "label": "9 EMA quality bounce (green + low-vol touch + fresh)",
                "value": (
                    f"Touch: {'Yes' if ema9_check.get('touch_found') else 'No'} | "
                    f"Slope: {ema9_check.get('ema_slope_pct',0):+.3f}% | "
                    f"Green: {'Yes' if ema9_check.get('green_candle') else 'No'} | "
                    f"TouchVol: {ema9_check.get('touch_vol_ratio',1):.2f}x | "
                    f"Touches(20d): {ema9_check.get('touch_count_20d',0)}"
                ),
                "group": "core",
            },
            "macro_ok": {
                "met":   cond_macro,
                "label": "Nifty not in breakdown",
                "value": "Market aligned" if cond_macro else "Market in breakdown",
                "group": "core",
            },
            # ── New 8 ──
            "macd_positive_rising": {
                "met":   cond_macd,
                "label": "MACD histogram positive & rising",
                "value": f"Hist={round(curr_hist,4)} (prev {round(prev_hist,4)})",
                "group": "momentum",
            },
            "volume_spike": {
                "met":   cond_vol_spike,
                "label": "Volume spike ≥ 1.5× 20-day avg",
                "value": f"{round(vol_ratio,2)}× average ({int(avg_vol_20):,})",
                "group": "momentum",
            },
            "near_52w_high": {
                "met":   cond_near_52wh,
                "label": "Within 15% of 52-week high",
                "value": f"{round(pct_from_h,2)}% from ₹{round(high52,2)} high",
                "group": "momentum",
            },
            "relative_strength": {
                "met":   cond_rel_strength,
                "label": "Outperforming Nifty50 (20-day RS)",
                "value": (
                    f"Stock {rs_check.get('stock_20d_ret',0):+.2f}% vs "
                    f"Nifty {rs_check.get('nifty_20d_ret',0):+.2f}% "
                    f"(RS = {rs_check.get('relative_strength',0):+.2f}%)"
                ),
                "group": "momentum",
            },
            "weekly_ema_aligned": {
                "met":   cond_weekly_aligned,
                "label": "Weekly EMA9 aligned (higher TF trend)",
                "value": (
                    f"Weekly close ₹{weekly_check.get('weekly_close',0)} vs "
                    f"EMA9 ₹{weekly_check.get('weekly_ema9',0)} | "
                    f"Slope {weekly_check.get('weekly_ema_slope',0):+.3f}%"
                ),
                "group": "trend",
            },
            "sector_strong": {
                "met":   cond_sector_strong,
                "label": "Sector index above its EMA9",
                "value": (
                    f"{sector_check.get('sector_ticker','')} | "
                    f"Slope {sector_check.get('sector_slope',0):+.3f}%"
                ),
                "group": "trend",
            },
            "bb_squeeze": {
                "met":   cond_bb_squeeze,
                "label": "Bollinger Band squeeze (bandwidth < 15 & narrowing)",
                "value": f"BW={round(curr_bw,2)}% (prev {round(prev_bw,2)}%)",
                "group": "timing",
            },
            "fresh_ema_touch": {
                "met":   cond_fresh_touch,
                "label": "Fresh EMA support (≤ 3 touches in 20 days)",
                "value": f"{ema9_check.get('touch_count_20d',0)} touches in last 20 days",
                "group": "timing",
            },
        }

        score   = sum(1 for c in conditions.values() if c["met"])
        max_sc  = len(conditions)

        # Core conditions (original 5) must ALL be met.
        # New 8 are bonus — we require at least 5 of 8 to fire a signal.
        core_met  = all(conditions[k]["met"] for k in
                        ["price_above_9ema", "rsi_55_70", "volume_1m",
                         "ema9_support", "macro_ok"])
        bonus_keys = ["macd_positive_rising", "volume_spike", "near_52w_high",
                      "relative_strength", "weekly_ema_aligned", "sector_strong",
                      "bb_squeeze", "fresh_ema_touch"]
        bonus_met = sum(1 for k in bonus_keys if conditions[k]["met"])
        all_met   = core_met and bonus_met >= 5   # 5-of-8 bonus required

        # ── Trade setup (ATR-based stops) ────────────────────────────────────
        curr_atr  = float(atr.iloc[-1])
        sl_price  = curr_close - (1.5 * curr_atr)        # 1.5×ATR dynamic SL
        sl_pct    = (curr_close - sl_price) / curr_close * 100
        t1        = curr_close * 1.05
        t2        = curr_close * 1.10
        risk      = curr_close - sl_price
        rr        = round((t1 - curr_close) / risk, 2) if risk > 0 else 0

        prev_close  = float(closes.iloc[-2])
        day_chg     = round((curr_close - prev_close) / prev_close * 100, 2)

        return {
            "symbol":             symbol,
            "all_conditions_met": all_met,
            "core_conditions_met": core_met,
            "bonus_conditions_met": bonus_met,
            "bonus_required":     5,
            "score":              score,
            "max_score":          max_sc,
            "conditions":         conditions,
            "metrics": {
                "close":              round(curr_close, 2),
                "prev_close":         round(prev_close, 2),
                "day_change_pct":     day_chg,
                "ema9":               round(curr_ema9, 2),
                "ema21":              round(curr_ema21, 2),
                "rsi":                round(curr_rsi, 1),
                "volume":             curr_vol,
                "avg_volume_20d":     int(avg_vol_20),
                "volume_ratio":       round(vol_ratio, 2),
                "atr":                round(curr_atr, 2),
                "atr_pct":            round(curr_atr / curr_close * 100, 2),
                "macd_hist":          round(curr_hist, 4),
                "bb_bandwidth":       round(curr_bw, 2),
                "high52":             round(high52, 2),
                "low52":              round(low52, 2),
                "pct_from_52high":    round(pct_from_h, 2),
                "proximity_to_ema9":  ema9_check.get("proximity_pct", 0),
                "relative_strength":  rs_check.get("relative_strength", 0),
                "touch_count_20d":    ema9_check.get("touch_count_20d", 0),
            },
            "trade_setup": {
                "entry":        round(curr_close, 2),
                "stop_loss":    round(sl_price, 2),    # 1.5×ATR
                "stop_loss_pct": round(sl_pct, 2),
                "target1":      round(t1, 2),
                "target2":      round(t2, 2),
                "risk_reward":  rr,
                "atr":          round(curr_atr, 2),
            },
            "ema9_detail":    ema9_check,
            "weekly_detail":  weekly_check,
            "sector_detail":  sector_check,
            "rs_detail":      rs_check,
        }

    # ── Batch screener ───────────────────────────────────────────────────────
    def screen_stocks(self, symbols: list = None, market_status: dict = None) -> list:
        if symbols is None:
            symbols = ALL_STOCKS
        market_healthy = (market_status or {}).get("healthy", True)
        results = []
        for symbol in symbols:
            try:
                r = self.check_confluence(symbol, market_healthy)
                results.append(r)
            except Exception as e:
                logger.error(f"Screen error {symbol}: {e}")
        results.sort(
            key=lambda x: (
                x.get("all_conditions_met", False),
                x.get("bonus_conditions_met", 0),
                x.get("score", 0),
            ),
            reverse=True,
        )
        return results

    # ── Chart data (unchanged) ───────────────────────────────────────────────
    def get_chart_data(self, symbol: str) -> dict:
        data = self.get_stock_data(symbol, period="3mo")
        if data is None:
            return {}

        closes   = data["Close"]
        ema9     = self.calculate_ema(closes, 9)
        ema21    = self.calculate_ema(closes, 21)
        rsi      = self.calculate_rsi(closes)
        macd, macd_sig, macd_hist = self.calculate_macd(closes)
        upper_bb, mid_bb, lower_bb, bw = self.calculate_bollinger_bands(closes)

        dates = [d.strftime("%Y-%m-%d") for d in data.index]

        return {
            "dates": dates,
            "ohlcv": {
                "open":   [round(v, 2) for v in data["Open"].tolist()],
                "high":   [round(v, 2) for v in data["High"].tolist()],
                "low":    [round(v, 2) for v in data["Low"].tolist()],
                "close":  [round(v, 2) for v in closes.tolist()],
                "volume": [int(v) for v in data["Volume"].tolist()],
            },
            "indicators": {
                "ema9":        [round(v, 2) for v in ema9.tolist()],
                "ema21":       [round(v, 2) for v in ema21.tolist()],
                "rsi":         [round(v, 2) for v in rsi.tolist()],
                "macd":        [round(v, 4) for v in macd.tolist()],
                "macd_signal": [round(v, 4) for v in macd_sig.tolist()],
                "macd_hist":   [round(v, 4) for v in macd_hist.tolist()],
                "bb_upper":    [round(v, 2) for v in upper_bb.tolist()],
                "bb_mid":      [round(v, 2) for v in mid_bb.tolist()],
                "bb_lower":    [round(v, 2) for v in lower_bb.tolist()],
                "bb_bandwidth":[round(v, 2) if not pd.isna(v) else 0
                                for v in bw.tolist()],
            },
        }