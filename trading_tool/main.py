"""
main.py — FastAPI Trading Advisor
Endpoints:
  GET  /                         → Bloomberg terminal UI
  GET  /api/analyze/{symbol}     → 9 EMA Support + confluence analysis
  GET  /api/screen               → Screen watchlist for 9 EMA setups
  POST /api/backtest             → Full historical backtest
  GET  /api/nifty                → Nifty50 macro direction
  GET  /api/scan                 → Manual scan trigger
  POST /api/telegram/setup       → Configure Telegram bot
  POST /api/telegram/test        → Send test message
  GET  /api/telegram/history     → Alert history
  GET  /api/scheduler/config     → View scheduler settings
  POST /api/scheduler/config     → Update scheduler settings
"""

from __future__ import annotations
import os
from datetime import datetime
from typing import Optional, List

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from strategies import ema9_support_check, hdfcamc_confluence, general_signals, ema, rsi
from backtester import run_backtest
from telegram_bot import send_telegram_alert, send_signal_alert, get_alert_history
from scheduler import setup_scheduler, get_config as sched_config, update_config as sched_update, scan_once

# ── App ───────────────────────────────────────────────────────────────────
app = FastAPI(title="NSE 9 EMA Swing Analyzer", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

setup_scheduler(app)

# ── Helpers ───────────────────────────────────────────────────────────────

def _nse(symbol: str) -> str:
    symbol = symbol.upper().strip()
    if symbol in ("^NSEI", "^BSESN"): return symbol
    if not symbol.endswith((".NS", ".BO")): return symbol + ".NS"
    return symbol


def _fetch(symbol: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    try:
        df = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=True)
        if df is None or len(df) < 10:
            raise ValueError("Empty data")
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Cannot fetch {symbol}: {e}")


def _add_info(sym: str, result: dict) -> dict:
    try:
        info = yf.Ticker(sym).info
        result["company_name"] = info.get("shortName", sym)
        result["sector"]       = info.get("sector", "N/A")
        result["market_cap"]   = info.get("marketCap")
        result["52w_high"]     = info.get("fiftyTwoWeekHigh")
        result["52w_low"]      = info.get("fiftyTwoWeekLow")
        result["pe_ratio"]     = info.get("trailingPE")
        result["beta"]         = info.get("beta")
    except Exception:
        result["company_name"] = sym
    return result


# ── Models ────────────────────────────────────────────────────────────────

class BacktestRequest(BaseModel):
    symbol         : str   = "HDFCAMC.NS"
    strategy       : str   = "ema9_support"
    period         : str   = "2y"
    take_profit_pct: float = 5.0
    stop_loss_pct  : float = 3.0
    max_hold_days  : int   = 7
    initial_capital: float = 100_000.0


class TelegramSetup(BaseModel):
    bot_token: str
    chat_id  : str


class TelegramTest(BaseModel):
    bot_token: str
    chat_id  : str
    message  : str = "🟢 SnapTrade Test Alert — Connection Successful!"


class SchedulerConfig(BaseModel):
    enabled          : Optional[bool]       = None
    watchlist        : Optional[List[str]]  = None
    telegram_token   : Optional[str]        = None
    telegram_chat_id : Optional[str]        = None
    min_confidence   : Optional[int]        = None
    scan_interval    : Optional[int]        = None


# ── Routes ────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def root():
    path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h1>NSE Swing Analyzer API — see /docs</h1>")


@app.get("/api/analyze/{symbol}")
def analyze(
    symbol        : str,
    mode          : str   = Query("ema9", description="ema9 | hdfcamc | general"),
    period        : str   = Query("6mo"),
    pullback_low  : float = Query(2780.0),
    pullback_high : float = Query(2800.0),
    breakout_level: float = Query(2870.0),
):
    """Analyze a stock. mode=ema9 (default) uses 9 EMA support detector."""
    sym = _nse(symbol)
    df  = _fetch(sym, period=period)

    if mode == "hdfcamc":
        result = hdfcamc_confluence(df, pullback_low=pullback_low,
                                    pullback_high=pullback_high,
                                    breakout_level=breakout_level)
    elif mode == "general":
        result = general_signals(df)
    else:  # ema9 — default
        result = ema9_support_check(df)

    result["symbol"] = sym
    return _add_info(sym, result)


@app.get("/api/screen")
def screen(
    symbols: str = Query(
        "HDFCAMC,RELIANCE,TCS,INFY,HDFCBANK,BAJFINANCE,ICICIBANK,AXISBANK,"
        "KOTAKBANK,SBIN,WIPRO,HCLTECH,TITAN,NESTLEIND,MARUTI,LTIM,TECHM,SUNPHARMA",
    ),
    mode: str = Query("ema9"),
):
    """Screen multiple stocks for 9 EMA support opportunities."""
    sym_list = [s.strip() for s in symbols.split(",") if s.strip()]
    results  = []

    for sym in sym_list:
        nsym = _nse(sym)
        try:
            df = _fetch(nsym, period="3mo")
            if mode == "ema9":
                r = ema9_support_check(df)
                r["symbol"] = nsym
                results.append({
                    "symbol"       : nsym.replace(".NS",""),
                    "price"        : r.get("current_price", 0),
                    "ema9"         : r.get("ema9", 0),
                    "ema9_dist_pct": r.get("ema9_distance_pct", 0),
                    "ema9_trigger" : r.get("ema9_triggered", False),
                    "signal"       : r.get("signal", "HOLD"),
                    "action"       : r.get("action", ""),
                    "confidence"   : r.get("confidence", 0),
                    "score"        : r.get("score", 0),
                    "rsi"          : r.get("indicators", {}).get("rsi"),
                    "vol_ratio"    : r.get("indicators", {}).get("vol_ratio"),
                    "target_5pct"  : r.get("targets", {}).get("target_5pct"),
                    "stop_loss"    : r.get("targets", {}).get("stop_loss"),
                    "conditions"   : r.get("conditions", {}),
                })
            else:
                r = general_signals(df)
                r["symbol"] = nsym
                results.append({
                    "symbol"      : nsym.replace(".NS",""),
                    "price"       : r.get("price", 0),
                    "signal"      : r.get("recommendation","HOLD"),
                    "confidence"  : r.get("confidence", 0),
                    "score"       : r.get("score", 0),
                    "rsi"         : r.get("indicators",{}).get("rsi"),
                    "target_5pct" : r.get("targets",{}).get("target_5pct"),
                })
        except Exception as e:
            results.append({"symbol": nsym.replace(".NS",""), "error": str(e)})

    # Sort: triggered first, then by confidence desc
    results.sort(key=lambda x: (-(1 if x.get("ema9_trigger") else 0), -x.get("confidence", 0)))
    return {"results": results, "count": len(results), "scanned_at": datetime.now().isoformat()}


@app.post("/api/backtest")
def backtest(req: BacktestRequest):
    """Run historical backtest."""
    sym = _nse(req.symbol)
    df  = _fetch(sym, period=req.period)
    result = run_backtest(
        df,
        strategy       = req.strategy,
        take_profit_pct= req.take_profit_pct,
        stop_loss_pct  = req.stop_loss_pct,
        max_hold_days  = req.max_hold_days,
        initial_capital= req.initial_capital,
    )
    result["symbol"]   = sym
    result["period"]   = req.period
    result["strategy"] = req.strategy
    return result


@app.get("/api/nifty")
def nifty_direction():
    """Nifty50 macro alignment check."""
    df    = _fetch("^NSEI", period="3mo")
    close = df["Close"]
    e20   = ema(close, 20)
    e50   = ema(close, 50)
    _rsi  = rsi(close)

    p    = float(close.iloc[-1])
    e20v = float(e20.iloc[-1])
    e50v = float(e50.iloc[-1])
    rv   = float(_rsi.iloc[-1])
    c1d  = float((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100)
    c5d  = float((close.iloc[-1] - close.iloc[-6]) / close.iloc[-6] * 100) if len(close) > 6 else 0

    if p > e20v and p > e50v and rv > 45:
        trend, macro, color = "UPTREND",   "✅ Macro Aligned — Safe for Longs",    "green"
    elif p < e20v and p < e50v:
        trend, macro, color = "DOWNTREND", "❌ Market Breakdown — Avoid New Longs", "red"
    else:
        trend, macro, color = "SIDEWAYS",  "⚠️ Choppy Market — Use Tight Stops",    "amber"

    return {"nifty_price":round(p,2),"trend":trend,"macro_signal":macro,"color":color,
            "change_1d_pct":round(c1d,2),"change_5d_pct":round(c5d,2),
            "above_ema20":p>e20v,"above_ema50":p>e50v,"rsi":round(rv,2)}


@app.get("/api/scan")
async def manual_scan(force: bool = Query(False)):
    """Manually trigger a watchlist scan."""
    triggered = await scan_once(force=force)
    return {
        "triggered_count": len(triggered),
        "triggered"       : triggered,
        "scanned_at"      : datetime.now().isoformat(),
    }


@app.post("/api/telegram/setup")
async def telegram_setup(req: TelegramSetup):
    """Save Telegram config and send a welcome message."""
    sched_update({"telegram_token": req.bot_token, "telegram_chat_id": req.chat_id})
    msg = (f"✅ <b>NSE 9 EMA Swing Analyzer</b> connected!\n\n"
           f"You will receive Telegram alerts when stocks trigger the 9 EMA support setup.\n"
           f"⏰ {datetime.now().strftime('%d %b %Y %H:%M IST')}")
    result = await send_telegram_alert(msg, token=req.bot_token, chat_id=req.chat_id)
    return result


@app.post("/api/telegram/test")
async def telegram_test(req: TelegramTest):
    """Send a test Telegram message."""
    return await send_telegram_alert(req.message, token=req.bot_token, chat_id=req.chat_id)


@app.get("/api/telegram/history")
def telegram_history():
    """Get recent alert history."""
    return {"alerts": get_alert_history()}


@app.get("/api/scheduler/config")
def scheduler_config_get():
    return sched_config()


@app.post("/api/scheduler/config")
def scheduler_config_post(req: SchedulerConfig):
    updates = {k: v for k, v in req.dict().items() if v is not None}
    return sched_update(updates)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
