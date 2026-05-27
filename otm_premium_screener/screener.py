"""
screener.py — OTM Premium Sell Zone Screener
Scans 2-6 OTM options for huge premium + premium surge + volume surge
within 3 days of expiry (ideal for options selling).
"""
import json
import os
import logging
import time
from datetime import datetime, date
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Tuple

import pandas as pd

from nse_client import NSEClient

logger = logging.getLogger(__name__)


class OTMPremiumScreener:

    def __init__(self, config: dict):
        self.config = config
        self.scan_cfg = config["scan"]
        self.cache_cfg = config["cache"]
        self.client = NSEClient(config)
        self.data_dir = self.cache_cfg["data_dir"]
        os.makedirs(self.data_dir, exist_ok=True)
        self._scan_progress: Dict = {"total": 0, "done": 0, "status": "idle", "current": ""}

    # ------------------------------------------------------------------ #
    #  Ticker loading                                                      #
    # ------------------------------------------------------------------ #

    def load_tickers(self, csv_path: str = "tickers.csv") -> List[str]:
        """Load symbols from tickers.csv."""
        try:
            df = pd.read_csv(csv_path)
            symbols = df["SYMBOL"].dropna().str.strip().unique().tolist()
            logger.info(f"Loaded {len(symbols)} tickers from {csv_path}")
            return symbols
        except Exception as e:
            logger.error(f"Failed to load tickers: {e}")
            return []

    # ------------------------------------------------------------------ #
    #  Historical cache (premium / volume history for surge detection)     #
    # ------------------------------------------------------------------ #

    def _cache_path(self, symbol: str, expiry: str, strike: float, opt_type: str) -> str:
        safe_expiry = expiry.replace("-", "")
        return os.path.join(self.data_dir, f"{symbol}_{safe_expiry}_{int(strike)}_{opt_type}.json")

    def _load_history(self, symbol: str, expiry: str, strike: float, opt_type: str) -> List[Dict]:
        path = self._cache_path(symbol, expiry, strike, opt_type)
        if os.path.exists(path):
            try:
                with open(path) as f:
                    return json.load(f)
            except Exception:
                pass
        return []

    def _save_history(self, symbol: str, expiry: str, strike: float, opt_type: str, data: List[Dict]):
        path = self._cache_path(symbol, expiry, strike, opt_type)
        # Keep only last N days
        keep = self.cache_cfg.get("history_days", 5)
        data = data[-keep:]
        try:
            with open(path, "w") as f:
                json.dump(data, f)
        except Exception as e:
            logger.warning(f"Could not save history for {symbol} {strike}{opt_type}: {e}")

    def _append_to_history(self, symbol: str, expiry: str, strike: float, opt_type: str, snapshot: Dict):
        history = self._load_history(symbol, expiry, strike, opt_type)
        today_str = date.today().isoformat()
        # Overwrite today's entry if exists
        history = [h for h in history if h.get("date") != today_str]
        history.append({"date": today_str, **snapshot})
        self._save_history(symbol, expiry, strike, opt_type, history)

    # ------------------------------------------------------------------ #
    #  Premium surge detection                                             #
    # ------------------------------------------------------------------ #

    def _calc_premium_surge(self, symbol: str, expiry: str, strike: float,
                             opt_type: str, current_ltp: float) -> Tuple[float, float]:
        """
        Returns (surge_pct, avg_3d_premium).
        surge_pct = % increase of today's LTP vs 3-day average.
        """
        history = self._load_history(symbol, expiry, strike, opt_type)
        today_str = date.today().isoformat()
        past = [h for h in history if h.get("date") != today_str]

        if not past:
            return 0.0, 0.0

        # Use up to 3 most recent past days
        recent = sorted(past, key=lambda x: x["date"])[-3:]
        avg = sum(r.get("ltp", 0) for r in recent) / len(recent)
        if avg <= 0:
            return 0.0, avg

        surge_pct = ((current_ltp - avg) / avg) * 100
        return round(surge_pct, 2), round(avg, 2)

    def _calc_volume_surge(self, symbol: str, expiry: str, strike: float,
                            opt_type: str, current_vol: int) -> Tuple[float, float]:
        """Returns (volume_ratio, avg_3d_volume)."""
        history = self._load_history(symbol, expiry, strike, opt_type)
        today_str = date.today().isoformat()
        past = [h for h in history if h.get("date") != today_str]
        if not past:
            return 0.0, 0.0

        recent = sorted(past, key=lambda x: x["date"])[-3:]
        avg = sum(r.get("volume", 0) for r in recent) / len(recent)
        if avg <= 0:
            return 0.0, avg

        ratio = current_vol / avg
        return round(ratio, 2), round(avg, 0)

    # ------------------------------------------------------------------ #
    #  Per-symbol scan                                                     #
    # ------------------------------------------------------------------ #

    def _scan_symbol(self, symbol: str) -> List[Dict]:
        """
        Fetch option chain for symbol and return matching OTM signals.
        """
        results = []
        cfg = self.scan_cfg

        raw = self.client.fetch_option_chain(symbol)
        if not raw:
            logger.warning(f"No data for {symbol}")
            return results

        parsed = self.client.parse_option_chain(raw, symbol)
        if not parsed:
            return results

        spot = parsed["spot"]
        if spot <= 0:
            return results

        expiries = parsed["expiries"]
        chain = parsed["chain"]

        for expiry in expiries:
            dte = NSEClient.days_to_expiry(expiry)

            # Only scan near-expiry
            if not (cfg["days_to_expiry_min"] <= dte <= cfg["days_to_expiry_max"]):
                continue

            strikes = sorted(chain.get(expiry, {}).keys())
            if not strikes:
                continue

            atm = NSEClient.get_atm_strike(spot, strikes)

            # OTM CE strikes (above spot)
            ce_strikes = NSEClient.get_otm_strikes_ce(spot, strikes, cfg["otm_strikes_min"], cfg["otm_strikes_max"])
            # OTM PE strikes (below spot)
            pe_strikes = NSEClient.get_otm_strikes_pe(spot, strikes, cfg["otm_strikes_min"], cfg["otm_strikes_max"])

            for strike, opt_type in [(s, "CE") for s in ce_strikes] + [(s, "PE") for s in pe_strikes]:
                opt_data = chain[expiry].get(strike, {}).get(opt_type)
                if not opt_data:
                    continue

                ltp = opt_data.get("ltp", 0.0)
                volume = int(opt_data.get("volume", 0))
                oi = int(opt_data.get("oi", 0))
                iv = opt_data.get("iv", 0.0)
                pchange = opt_data.get("pchange", 0.0)

                # Always snapshot today's data into history
                self._append_to_history(symbol, expiry, strike, opt_type, {
                    "ltp": ltp,
                    "volume": volume,
                    "oi": oi,
                    "iv": iv,
                })

                # --- Gate 1: Minimum premium ---
                if ltp < cfg["min_premium"]:
                    continue

                # --- Gate 2: Minimum OI ---
                if oi < cfg["min_oi"]:
                    continue

                # --- Gate 3: Premium surge over 3-day avg ---
                surge_pct, avg_3d_premium = self._calc_premium_surge(symbol, expiry, strike, opt_type, ltp)

                # --- Gate 4: Volume surge ---
                vol_ratio, avg_3d_vol = self._calc_volume_surge(symbol, expiry, strike, opt_type, volume)

                # Calculate how many strikes OTM
                if opt_type == "CE":
                    otm_count = strikes.index(strike) - strikes.index(atm) if strike in strikes and atm in strikes else 0
                else:
                    otm_count = strikes.index(atm) - strikes.index(strike) if strike in strikes and atm in strikes else 0

                # Score for sorting (0-100)
                score = self._compute_score(ltp, surge_pct, vol_ratio, iv, oi)

                signal = {
                    "symbol": symbol,
                    "expiry": expiry,
                    "strike": strike,
                    "type": opt_type,
                    "spot": round(spot, 2),
                    "atm_strike": atm,
                    "otm_count": otm_count,
                    "ltp": round(ltp, 2),
                    "iv": round(iv, 2),
                    "volume": volume,
                    "oi": oi,
                    "oi_change": opt_data.get("change_oi", 0),
                    "pchange": round(pchange, 2),
                    "premium_surge_pct": surge_pct,
                    "avg_3d_premium": avg_3d_premium,
                    "volume_ratio": vol_ratio,
                    "avg_3d_volume": avg_3d_vol,
                    "days_to_expiry": dte,
                    "score": score,
                    "gates": {
                        "premium_ok": ltp >= cfg["min_premium"],
                        "oi_ok": oi >= cfg["min_oi"],
                        "premium_surge": surge_pct >= cfg["premium_surge_pct"],
                        "volume_surge": vol_ratio >= cfg["volume_surge_multiplier"],
                    },
                    "scanned_at": datetime.now().isoformat(),
                }

                # Must pass at least premium + oi gates; surge gates are informational
                results.append(signal)

        return results

    # ------------------------------------------------------------------ #
    #  Scoring                                                             #
    # ------------------------------------------------------------------ #

    def _compute_score(self, ltp: float, surge_pct: float, vol_ratio: float, iv: float, oi: int) -> int:
        """0-100 composite score for ranking."""
        s = 0
        # Premium weight: up to 25 pts
        if ltp >= 100:
            s += 25
        elif ltp >= 50:
            s += 18
        elif ltp >= 30:
            s += 12
        elif ltp >= 15:
            s += 6

        # Premium surge: up to 30 pts
        if surge_pct >= 100:
            s += 30
        elif surge_pct >= 60:
            s += 22
        elif surge_pct >= 30:
            s += 15
        elif surge_pct >= 10:
            s += 7

        # Volume ratio: up to 25 pts
        if vol_ratio >= 5:
            s += 25
        elif vol_ratio >= 3:
            s += 18
        elif vol_ratio >= 2:
            s += 12
        elif vol_ratio >= 1:
            s += 5

        # IV: up to 10 pts
        if iv >= 50:
            s += 10
        elif iv >= 30:
            s += 6
        elif iv >= 15:
            s += 3

        # OI: up to 10 pts
        if oi >= 100000:
            s += 10
        elif oi >= 50000:
            s += 7
        elif oi >= 10000:
            s += 4
        elif oi >= 500:
            s += 2

        return min(s, 100)

    # ------------------------------------------------------------------ #
    #  Full scan                                                           #
    # ------------------------------------------------------------------ #

    def run_scan(self, symbols: Optional[List[str]] = None, csv_path: str = "tickers.csv") -> Dict:
        """
        Run the full screener scan.
        Returns dict with signals and metadata.
        """
        if symbols is None:
            symbols = self.load_tickers(csv_path)

        self._scan_progress = {
            "total": len(symbols),
            "done": 0,
            "status": "running",
            "current": "",
            "started_at": datetime.now().isoformat(),
        }

        all_signals = []
        errors = []
        delay = self.scan_cfg.get("scan_delay_seconds", 1.5)

        logger.info(f"Starting OTM premium scan for {len(symbols)} symbols...")

        # Sequential scan (NSE doesn't like parallel hammering)
        for i, symbol in enumerate(symbols):
            self._scan_progress["current"] = symbol
            self._scan_progress["done"] = i
            try:
                signals = self._scan_symbol(symbol)
                all_signals.extend(signals)
                logger.info(f"[{i+1}/{len(symbols)}] {symbol}: {len(signals)} signals")
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
                errors.append({"symbol": symbol, "error": str(e)})
            time.sleep(delay)

        # Sort by score descending
        all_signals.sort(key=lambda x: x["score"], reverse=True)

        self._scan_progress["status"] = "done"
        self._scan_progress["done"] = len(symbols)

        result = {
            "scan_time": datetime.now().isoformat(),
            "total_symbols": len(symbols),
            "signals_found": len(all_signals),
            "errors": len(errors),
            "signals": all_signals,
            "error_list": errors,
        }

        # Persist signals
        self._save_signals(result)
        self._log_scan(result)
        return result

    # ------------------------------------------------------------------ #
    #  Persistence                                                         #
    # ------------------------------------------------------------------ #

    def _save_signals(self, result: Dict):
        path = self.cache_cfg.get("signals_file", "signals.json")
        try:
            with open(path, "w") as f:
                json.dump(result, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save signals: {e}")

    def _load_signals(self) -> Dict:
        path = self.cache_cfg.get("signals_file", "signals.json")
        if os.path.exists(path):
            try:
                with open(path) as f:
                    return json.load(f)
            except Exception:
                pass
        return {"signals": [], "scan_time": None, "signals_found": 0}

    def _log_scan(self, result: Dict):
        path = self.cache_cfg.get("scan_log_file", "scan_log.json")
        log = []
        if os.path.exists(path):
            try:
                with open(path) as f:
                    log = json.load(f)
            except Exception:
                pass
        log.append({
            "scan_time": result["scan_time"],
            "symbols": result["total_symbols"],
            "signals": result["signals_found"],
            "errors": result["errors"],
        })
        # Keep last 50 log entries
        log = log[-50:]
        try:
            with open(path, "w") as f:
                json.dump(log, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save scan log: {e}")

    def get_progress(self) -> Dict:
        return dict(self._scan_progress)

    def get_last_signals(self) -> Dict:
        return self._load_signals()
