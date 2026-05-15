from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import os
import pathlib
import numpy as np
import pandas as pd
import math

from screener import StockScreener, ALL_STOCKS, NIFTY50_STOCKS
from backtester import Backtester
from telegram_notifier import TelegramNotifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s"
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="9 EMA Trading Analyzer",
    description="NSE Stock Screener with 9 EMA Confluence + Telegram Alerts",
    version="1.0.0"
)

# =========================
# FIX NUMPY JSON ISSUE
# =========================
def clean_numpy(obj):

    if isinstance(obj, dict):
        return {
            k: clean_numpy(v)
            for k, v in obj.items()
        }

    elif isinstance(obj, list):
        return [
            clean_numpy(v)
            for v in obj
        ]

    elif isinstance(obj, tuple):
        return tuple(
            clean_numpy(v)
            for v in obj
        )

    elif isinstance(obj, np.bool_):
        return bool(obj)

    elif isinstance(obj, np.integer):
        return int(obj)

    elif isinstance(obj, np.floating):
        val = float(obj)

        if math.isnan(val) or math.isinf(val):
            return 0

        return val

    elif isinstance(obj, float):

        if math.isnan(obj) or math.isinf(obj):
            return 0

        return obj

    elif isinstance(obj, np.ndarray):
        return obj.tolist()

    elif isinstance(obj, pd.Series):
        return obj.tolist()

    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()

    elif hasattr(obj, "item"):
        try:
            val = obj.item()

            if isinstance(val, float):
                if math.isnan(val) or math.isinf(val):
                    return 0

            return val

        except:
            return str(obj)

    return obj

# =========================
# MIDDLEWARE
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# STATIC
# =========================
BASE_DIR = pathlib.Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"

STATIC_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# =========================
# SERVICES
# =========================
executor = ThreadPoolExecutor(max_workers=8)

screener = StockScreener()
backtester = Backtester()
notifier = TelegramNotifier()

# =========================
# MODELS
# =========================
class ScreenRequest(BaseModel):
    symbols: Optional[List[str]] = None
    send_telegram: bool = False

class AnalyzeRequest(BaseModel):
    symbol: str

class BacktestRequest(BaseModel):
    symbol: str
    period: str = "2y"
    stop_loss_pct: float = 2.0
    target1_pct: float = 5.0
    target2_pct: float = 10.0
    rsi_min: float = 55.0
    rsi_max: float = 70.0
    min_volume: int = 1_000_000
    partial_exit: bool = True
    bonus_required: int = 3          # how many of 8 bonus conditions needed (lowered default for more trades)
    vol_spike_mult: float = 1.5
    stop_loss_atr_mult: float = 1.5

class TelegramConfig(BaseModel):
    token: str
    chat_id: str

class AlertRequest(BaseModel):
    symbol: str

# =========================
# HELPER
# =========================
async def run_sync(fn, *args):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, fn, *args)

# =========================
# ROUTES
# =========================
@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = STATIC_DIR / "index.html"

    if html_path.exists():
        return HTMLResponse(
            content=html_path.read_text(encoding="utf-8")
        )

    return HTMLResponse(
        "<h1>index.html not found in /static/</h1>",
        status_code=404
    )

@app.get("/api/stocks")
async def get_stock_list():
    return clean_numpy({
        "nifty50": NIFTY50_STOCKS,
        "all": ALL_STOCKS,
        "count": len(ALL_STOCKS)
    })

@app.get("/api/market")
async def get_market():
    result = await run_sync(screener.get_market_status)
    return clean_numpy(result)

@app.post("/api/screen")
async def screen_stocks(
    req: ScreenRequest,
    background_tasks: BackgroundTasks
):
    symbols = req.symbols if req.symbols else ALL_STOCKS

    market = await run_sync(
        screener.get_market_status
    )

    def _screen():
        return screener.screen_stocks(
            symbols,
            market
        )

    results = await run_sync(_screen)

    qualified = [
        r for r in results
        if r.get("all_conditions_met")
    ]

    near_miss = [
        r for r in results
        if r.get("score", 0) >= 4
        and not r.get("all_conditions_met")
    ]

    if req.send_telegram and notifier.configured:
        background_tasks.add_task(
            notifier.send_screening_summary,
            results,
            market
        )

    return clean_numpy({
        "market": market,
        "total_screened": len(results),
        "signals_found": len(qualified),
        "near_miss": len(near_miss),
        "results": results,
    })

@app.post("/api/analyze")
async def analyze_stock(req: AnalyzeRequest):

    symbol = req.symbol.upper().strip()

    market = await run_sync(
        screener.get_market_status
    )

    def _analyze():
        analysis = screener.check_confluence(
            symbol,
            market.get("healthy", True)
        )

        chart = screener.get_chart_data(symbol)

        return analysis, chart

    loop = asyncio.get_event_loop()

    analysis, chart = await loop.run_in_executor(
        executor,
        _analyze
    )

    return clean_numpy({
        "analysis": analysis,
        "chart_data": chart,
        "market": market,
    })

@app.post("/api/backtest")
async def backtest(req: BacktestRequest):

    def _bt():
        return backtester.run_backtest(
            symbol=req.symbol.upper().strip(),
            period=req.period,
            stop_loss_atr_mult=req.stop_loss_atr_mult,
            stop_loss_pct=req.stop_loss_pct,
            target1_pct=req.target1_pct,
            target2_pct=req.target2_pct,
            rsi_min=req.rsi_min,
            rsi_max=req.rsi_max,
            min_volume=req.min_volume,
            partial_exit=req.partial_exit,
            vol_spike_mult=req.vol_spike_mult,
            bonus_required=req.bonus_required,
        )

    result = await run_sync(_bt)

    return clean_numpy(result)

@app.post("/api/telegram/configure")
async def configure_telegram(
    config: TelegramConfig
):
    notifier.configure(
        config.token,
        config.chat_id
    )

    result = await run_sync(
        notifier.test_connection
    )

    return clean_numpy(result)

@app.get("/api/telegram/status")
async def telegram_status():
    return clean_numpy({
        "configured": notifier.configured
    })

@app.post("/api/telegram/test")
async def test_telegram():

    if not notifier.configured:
        return clean_numpy({
            "success": False,
            "message": "Telegram not configured yet"
        })

    result = await run_sync(
        notifier.test_connection
    )

    return clean_numpy(result)

@app.post("/api/alert")
async def send_alert(
    req: AlertRequest,
    background_tasks: BackgroundTasks
):

    if not notifier.configured:
        raise HTTPException(
            status_code=400,
            detail="Telegram not configured"
        )

    symbol = req.symbol.upper().strip()

    market = await run_sync(
        screener.get_market_status
    )

    def _get_and_send():
        analysis = screener.check_confluence(
            symbol,
            market.get("healthy", True)
        )

        msg = notifier.format_stock_alert(
            analysis,
            market
        )

        notifier.send_message(msg)

        return analysis

    background_tasks.add_task(_get_and_send)

    return clean_numpy({
        "queued": True,
        "symbol": symbol
    })

@app.get("/api/health")
async def health():
    return clean_numpy({
        "status": "ok",
        "version": "1.0.0"
    })

# =========================
# ENTRY
# =========================
if __name__ == "__main__":

    import uvicorn

    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True
    )