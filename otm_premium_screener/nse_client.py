"""
nse_client.py — NSE Option Chain fetcher with session warm-up & retry logic
"""
import requests
import time
import json
import logging
from datetime import datetime, date, timedelta
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class NSEClient:
    """Handles NSE session, cookies, and option chain API calls."""

    def __init__(self, config: dict):
        self.cfg = config["nse"]
        self.base_url = self.cfg["base_url"]
        self.timeout = self.cfg.get("timeout", 15)
        self.max_retries = self.cfg.get("max_retries", 3)
        self.session = requests.Session()
        self.session.headers.update(self.cfg["headers"])
        self._session_ready = False
        self._last_warmup: Optional[float] = None
        self._warmup_interval = 300  # re-warm every 5 min

    # ------------------------------------------------------------------ #
    #  Session management                                                  #
    # ------------------------------------------------------------------ #

    def _warm_up(self) -> bool:
        """Visit NSE homepage to get cookies/tokens before API calls."""
        now = time.time()
        if self._session_ready and self._last_warmup and (now - self._last_warmup) < self._warmup_interval:
            return True
        try:
            r = self.session.get(
                self.cfg["session_init_url"],
                timeout=self.timeout,
                allow_redirects=True
            )
            if r.status_code == 200:
                self._session_ready = True
                self._last_warmup = now
                logger.debug("NSE session warmed up.")
                return True
        except Exception as e:
            logger.warning(f"NSE warm-up failed: {e}")
        return False

    def _get(self, url: str) -> Optional[Dict]:
        """GET with retry + auto session re-warm."""
        for attempt in range(self.max_retries):
            if not self._warm_up():
                time.sleep(2)
                continue
            try:
                r = self.session.get(url, timeout=self.timeout)
                if r.status_code == 200:
                    return r.json()
                elif r.status_code == 401:
                    logger.warning("401 — re-warming session.")
                    self._session_ready = False
                    time.sleep(2)
                else:
                    logger.warning(f"HTTP {r.status_code} for {url}")
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout on attempt {attempt + 1}: {url}")
            except Exception as e:
                logger.warning(f"Error on attempt {attempt + 1}: {e}")
            time.sleep(1.5 * (attempt + 1))
        return None

    # ------------------------------------------------------------------ #
    #  Option chain                                                        #
    # ------------------------------------------------------------------ #

    def fetch_option_chain(self, symbol: str) -> Optional[Dict]:
        """Fetch full option chain for equity or index symbol."""
        indices = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX", "NIFTYNXT50"]
        if symbol.upper() in indices:
            url = self.base_url + self.cfg["option_chain_index"] + symbol.upper()
        else:
            url = self.base_url + self.cfg["option_chain_equity"] + symbol.upper()
        return self._get(url)

    def parse_option_chain(self, raw: Dict, symbol: str) -> Optional[Dict]:
        """
        Parse raw option chain JSON into structured data:
        {
          "symbol": str,
          "spot": float,
          "expiries": [date_str, ...],
          "chain": {
            expiry: {
              strike: {"CE": {...}, "PE": {...}}
            }
          }
        }
        """
        try:
            records = raw.get("records", {})
            filtered = raw.get("filtered", {})
            spot = (
                records.get("underlyingValue")
                or filtered.get("CE", {}).get("underlyingValue")
                or 0.0
            )
            expiry_dates: List[str] = records.get("expiryDates", [])
            data_rows = records.get("data", [])

            chain: Dict[str, Dict] = {}
            for row in data_rows:
                expiry = row.get("expiryDate", "")
                strike = float(row.get("strikePrice", 0))
                if expiry not in chain:
                    chain[expiry] = {}
                if strike not in chain[expiry]:
                    chain[expiry][strike] = {}

                for opt_type in ("CE", "PE"):
                    if opt_type in row:
                        d = row[opt_type]
                        chain[expiry][strike][opt_type] = {
                            "ltp": d.get("lastPrice", 0.0),
                            "volume": d.get("totalTradedVolume", 0),
                            "oi": d.get("openInterest", 0),
                            "change_oi": d.get("changeinOpenInterest", 0),
                            "iv": d.get("impliedVolatility", 0.0),
                            "bid": d.get("bidprice", 0.0),
                            "ask": d.get("askPrice", 0.0),
                            "pchange": d.get("pChange", 0.0),
                            "change": d.get("change", 0.0),
                        }

            return {
                "symbol": symbol,
                "spot": float(spot),
                "expiries": expiry_dates,
                "chain": chain,
                "fetched_at": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"parse_option_chain failed for {symbol}: {e}")
            return None

    # ------------------------------------------------------------------ #
    #  Expiry helpers                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def parse_expiry_date(expiry_str: str) -> Optional[date]:
        """Parse '26-Jun-2025' → date object."""
        for fmt in ("%d-%b-%Y", "%Y-%m-%d", "%d-%m-%Y"):
            try:
                return datetime.strptime(expiry_str, fmt).date()
            except ValueError:
                continue
        return None

    @staticmethod
    def days_to_expiry(expiry_str: str) -> int:
        """Return calendar days from today to expiry."""
        dt = NSEClient.parse_expiry_date(expiry_str)
        if dt is None:
            return 9999
        return (dt - date.today()).days

    @staticmethod
    def get_atm_strike(spot: float, strikes: List[float]) -> float:
        """Return the strike closest to spot price."""
        if not strikes:
            return spot
        return min(strikes, key=lambda s: abs(s - spot))

    @staticmethod
    def get_otm_strikes_ce(spot: float, strikes: List[float], min_otm: int, max_otm: int) -> List[float]:
        """Strikes above spot, from min_otm-th to max_otm-th OTM (for CE selling)."""
        above = sorted([s for s in strikes if s > spot])
        return above[min_otm - 1: max_otm]

    @staticmethod
    def get_otm_strikes_pe(spot: float, strikes: List[float], min_otm: int, max_otm: int) -> List[float]:
        """Strikes below spot, from min_otm-th to max_otm-th OTM (for PE selling)."""
        below = sorted([s for s in strikes if s < spot], reverse=True)
        return below[min_otm - 1: max_otm]
