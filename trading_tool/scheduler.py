"""
scheduler.py — APScheduler auto-scan during NSE market hours
Scans watchlist for 9 EMA support triggers and fires Telegram alerts.
"""
import asyncio
import logging
from datetime import datetime, time as dtime
from typing import List, Optional

import pytz

logger = logging.getLogger("scheduler")

# ── Config (overrideable at runtime via API) ──────────────────────────────
_config = {
    "enabled"       : False,
    "watchlist"     : [
        "HDFCAMC", "RELIANCE", "TCS", "INFY", "HDFCBANK",
        "ICICIBANK", "AXISBANK", "SBIN", "BAJFINANCE", "KOTAKBANK",
        "WIPRO", "HCLTECH", "TITAN", "NESTLEIND", "MARUTI",
        "ADANIENT", "LTIM", "TECHM", "SUNPHARMA", "DRREDDY",
    ],
    "telegram_token"  : "",
    "telegram_chat_id": "",
    "alert_on_signals": ["BUY"],          # signals that trigger Telegram
    "min_confidence"  : 55,               # minimum confidence to alert
    "scan_interval"   : 15,               # minutes
    "last_scan"       : None,
    "last_alerts"     : [],
}

IST = pytz.timezone("Asia/Kolkata")


def is_market_hours() -> bool:
    """NSE market: Mon–Fri, 09:15–15:30 IST."""
    now = datetime.now(IST)
    if now.weekday() >= 5:                  # Saturday / Sunday
        return False
    market_open  = dtime(9, 15)
    market_close = dtime(15, 35)
    return market_open <= now.time() <= market_close


def get_config() -> dict:
    return dict(_config)


def update_config(updates: dict) -> dict:
    _config.update(updates)
    return get_config()


async def scan_once(force: bool = False) -> List[dict]:
    """
    Run one scan pass.  Called by scheduler or manually via API.
    Returns list of triggered signals.
    """
    if not force and not is_market_hours():
        logger.info("Outside market hours — skipping scan")
        return []

    # Import here to avoid circular imports
    import yfinance as yf
    import pandas as pd
    from strategies import ema9_support_check
    from telegram_bot import send_signal_alert

    triggered = []
    watchlist = _config["watchlist"]
    logger.info(f"Scanning {len(watchlist)} symbols …")

    for sym in watchlist:
        nsym = sym.upper().strip()
        if not nsym.endswith(".NS"):
            nsym += ".NS"
        try:
            ticker = yf.Ticker(nsym)
            df = ticker.history(period="3mo", interval="1d", auto_adjust=True)
            if df is None or len(df) < 30:
                continue
            df.index = pd.to_datetime(df.index).tz_localize(None)

            result = ema9_support_check(df)
            result["symbol"] = nsym

            signal     = result.get("signal", "HOLD")
            confidence = result.get("confidence", 0)

            # Should we alert?
            should_alert = (
                signal in _config["alert_on_signals"]
                and confidence >= _config["min_confidence"]
                and result.get("ema9_triggered", False)
            )

            if should_alert:
                triggered.append(result)
                # Fire Telegram if configured
                if _config.get("telegram_token") and _config.get("telegram_chat_id"):
                    asyncio.create_task(
                        send_signal_alert(
                            result,
                            token  =_config["telegram_token"],
                            chat_id=_config["telegram_chat_id"],
                        )
                    )

        except Exception as e:
            logger.warning(f"{nsym}: {e}")

    _config["last_scan"]   = datetime.now().isoformat()
    _config["last_alerts"] = triggered[:20]
    logger.info(f"Scan complete — {len(triggered)} trigger(s)")
    return triggered


def setup_scheduler(app):
    """Attach APScheduler to the FastAPI app lifecycle."""
    try:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from apscheduler.triggers.interval  import IntervalTrigger

        scheduler = AsyncIOScheduler(timezone=IST)
        scheduler.add_job(
            scan_once,
            trigger  = IntervalTrigger(minutes=_config["scan_interval"]),
            id       = "market_scan",
            replace_existing=True,
        )

        @app.on_event("startup")
        async def start_sched():
            scheduler.start()
            logger.info("Scheduler started")

        @app.on_event("shutdown")
        async def stop_sched():
            scheduler.shutdown()

        return scheduler

    except ImportError:
        logger.warning("APScheduler not installed — auto-scan disabled. pip install apscheduler")
        return None
