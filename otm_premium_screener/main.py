"""
main.py — OTM Premium Sell Zone Screener — FastAPI Server
Screens 2-6 OTM options with huge premium + surge + volume near expiry.
"""
import json
import logging
import os
import threading
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, BackgroundTasks, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from screener import OTMPremiumScreener

# ── Logging ──────────────────────────────────────────────────────────── #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ── Load config ───────────────────────────────────────────────────────── #
with open("config.json") as f:
    CONFIG = json.load(f)

# ── Init screener ─────────────────────────────────────────────────────── #
screener = OTMPremiumScreener(CONFIG)
_scan_lock = threading.Lock()
_scan_running = False


# ── App ───────────────────────────────────────────────────────────────── #
app = FastAPI(
    title="OTM Premium Sell Zone Screener",
    description="NSE options screener: 2-6 OTM with huge premium near expiry",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Pydantic models ───────────────────────────────────────────────────── #
class ScanRequest(BaseModel):
    symbols: Optional[List[str]] = None  # None → use tickers.csv
    days_to_expiry_max: Optional[int] = None
    min_premium: Optional[float] = None
    otm_strikes_min: Optional[int] = None
    otm_strikes_max: Optional[int] = None
    premium_surge_pct: Optional[float] = None
    volume_surge_multiplier: Optional[float] = None


class ConfigUpdate(BaseModel):
    scan: Optional[dict] = None


# ── Background scan ──────────────────────────────────────────────────── #
def _run_scan_bg(symbols: Optional[List[str]], overrides: dict):
    global _scan_running
    try:
        # Apply overrides to config temporarily
        cfg = json.loads(json.dumps(CONFIG))  # deep copy
        if overrides:
            cfg["scan"].update(overrides)

        sc = OTMPremiumScreener(cfg)
        sc._scan_progress = screener._scan_progress  # share progress ref
        sc.run_scan(symbols=symbols)
        # Propagate results back
        screener._scan_progress = sc._scan_progress
        screener._save_signals(sc._load_signals())
    except Exception as e:
        logger.error(f"Background scan failed: {e}")
        screener._scan_progress["status"] = "error"
    finally:
        _scan_running = False


# ── Routes ────────────────────────────────────────────────────────────── #

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve frontend."""
    try:
        with open("static/index.html", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    except FileNotFoundError:
        return HTMLResponse("<h1>OTM Premium Screener — static/index.html not found</h1>")


@app.post("/api/scan")
async def start_scan(req: ScanRequest, background_tasks: BackgroundTasks):
    """Start a background scan."""
    global _scan_running
    if _scan_running:
        return JSONResponse({"status": "already_running", "message": "Scan already in progress"}, status_code=409)

    overrides = {}
    for field in ["days_to_expiry_max", "min_premium", "otm_strikes_min",
                  "otm_strikes_max", "premium_surge_pct", "volume_surge_multiplier"]:
        val = getattr(req, field, None)
        if val is not None:
            overrides[field] = val

    _scan_running = True
    screener._scan_progress = {
        "total": len(req.symbols) if req.symbols else "?",
        "done": 0,
        "status": "starting",
        "current": "",
        "started_at": datetime.now().isoformat(),
    }

    background_tasks.add_task(_run_scan_bg, req.symbols, overrides)
    return {"status": "started", "message": "Scan started in background"}


@app.get("/api/scan/progress")
async def scan_progress():
    """Get live scan progress."""
    return screener.get_progress()


@app.get("/api/signals")
async def get_signals(
    opt_type: Optional[str] = Query(None, description="CE or PE"),
    min_score: int = Query(0),
    min_premium: float = Query(0),
    min_surge_pct: float = Query(0),
    min_vol_ratio: float = Query(0),
    dte_max: int = Query(3),
    sort_by: str = Query("score"),
    limit: int = Query(100),
):
    """Get last scan signals with filters."""
    data = screener.get_last_signals()
    signals = data.get("signals", [])

    # Filters
    if opt_type:
        signals = [s for s in signals if s["type"] == opt_type.upper()]
    signals = [s for s in signals if s.get("score", 0) >= min_score]
    signals = [s for s in signals if s.get("ltp", 0) >= min_premium]
    signals = [s for s in signals if s.get("premium_surge_pct", 0) >= min_surge_pct]
    signals = [s for s in signals if s.get("volume_ratio", 0) >= min_vol_ratio]
    signals = [s for s in signals if s.get("days_to_expiry", 99) <= dte_max]

    # Sort
    valid_sorts = {"score", "ltp", "premium_surge_pct", "volume_ratio", "iv", "days_to_expiry"}
    if sort_by in valid_sorts:
        signals.sort(key=lambda x: x.get(sort_by, 0), reverse=(sort_by != "days_to_expiry"))

    return {
        "scan_time": data.get("scan_time"),
        "total": len(signals),
        "signals": signals[:limit],
    }


@app.get("/api/signals/top")
async def get_top_signals(limit: int = Query(20)):
    """Get top N signals by score (both CE+PE gates fully passed)."""
    data = screener.get_last_signals()
    signals = data.get("signals", [])

    # Only fully confirmed signals (premium + volume surges both passed)
    confirmed = [
        s for s in signals
        if s.get("gates", {}).get("premium_surge") and s.get("gates", {}).get("volume_surge")
    ]
    confirmed.sort(key=lambda x: x.get("score", 0), reverse=True)

    return {
        "scan_time": data.get("scan_time"),
        "total_confirmed": len(confirmed),
        "signals": confirmed[:limit],
    }


@app.get("/api/scan/log")
async def scan_log():
    """Return scan history log."""
    log_path = CONFIG["cache"].get("scan_log_file", "scan_log.json")
    if os.path.exists(log_path):
        with open(log_path) as f:
            return json.load(f)
    return []


@app.get("/api/config")
async def get_config():
    return CONFIG["scan"]


@app.post("/api/config")
async def update_config(update: ConfigUpdate):
    """Update scan config at runtime."""
    if update.scan:
        CONFIG["scan"].update(update.scan)
        with open("config.json", "w") as f:
            json.dump(CONFIG, f, indent=2)
    return {"status": "updated", "scan": CONFIG["scan"]}


@app.get("/api/tickers")
async def get_tickers():
    """Return loaded ticker list."""
    tickers = screener.load_tickers()
    return {"count": len(tickers), "tickers": tickers}


@app.get("/api/symbol/{symbol}")
async def scan_single(symbol: str):
    """Scan a single symbol on demand."""
    result = screener._scan_symbol(symbol.upper())
    return {"symbol": symbol.upper(), "signals": result, "count": len(result)}


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "scan_running": _scan_running,
        "time": datetime.now().isoformat(),
    }


# ── Entry point ───────────────────────────────────────────────────────── #
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
