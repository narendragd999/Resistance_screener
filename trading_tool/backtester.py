"""
backtester.py — Historical trade simulation engine
Supports: rsi | macd | bollinger | ema_cross | volume_breakout | ema9_support | composite
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from strategies import rsi, macd, bollinger_bands, ema, atr


# ── Signal functions ──────────────────────────────────────────────────────

def _sig_rsi(close, rsi_s, i):
    return float(rsi_s.iloc[i]) < 35

def _sig_macd(ml, sl, i):
    return float(ml.iloc[i-1]) < float(sl.iloc[i-1]) and float(ml.iloc[i]) > float(sl.iloc[i])

def _sig_bollinger(close, bb_lo, rsi_s, i):
    return float(close.iloc[i]) <= float(bb_lo.iloc[i]) * 1.005 and float(rsi_s.iloc[i]) < 48

def _sig_ema_cross(e9, e21, i):
    return float(e9.iloc[i-1]) < float(e21.iloc[i-1]) and float(e9.iloc[i]) > float(e21.iloc[i])

def _sig_volume_breakout(close, vol, vol_sma, rsi_s, i):
    return (float(vol.iloc[i]) > 1.5 * float(vol_sma.iloc[i]) and
            float(close.iloc[i]) > float(close.iloc[i-1]) and
            float(rsi_s.iloc[i]) < 68)

def _sig_ema9_support(close, low_s, rsi_s, e9, vol, vol_sma, i,
                       tolerance_pct: float = 2.0,
                       min_volume: float = 500_000):
    """
    9 EMA Support signal (backtest-tuned):
    - Previous candle's LOW came within tolerance% of EMA9 (touched it)
    - Current close is above EMA9 (bounce confirmed)
    - RSI 42–76 (relaxed for backtest scanning)
    - Volume ≥ 70% of 20-day average
    """
    p       = float(close.iloc[i])
    lo_prev = float(low_s.iloc[i-1])
    e9_prev = float(e9.iloc[i-1])
    e9_cur  = float(e9.iloc[i])
    r       = float(rsi_s.iloc[i])
    v       = float(vol.iloc[i])
    vs_raw  = float(vol_sma.iloc[i])
    vs      = vs_raw if vs_raw > 0 else 1

    # Core: low touched EMA zone and price bounced above it
    touched_ema  = lo_prev <= e9_prev * (1 + tolerance_pct / 100)
    above_ema    = p > e9_cur
    rsi_ok       = 42 <= r <= 76
    vol_ok       = v >= min_volume and v >= vs * 0.7

    return touched_ema and above_ema and rsi_ok and vol_ok

def _sig_composite(close, rsi_s, ml, sl_m, bb_lo, e9, e21, vol, vol_sma, i):
    score = 0
    if float(rsi_s.iloc[i]) < 42:                                             score += 2
    if float(ml.iloc[i]) > float(sl_m.iloc[i]):                               score += 1
    if float(close.iloc[i]) <= float(bb_lo.iloc[i]) * 1.01:                   score += 2
    if float(e9.iloc[i]) > float(e21.iloc[i]):                                score += 1
    if (float(vol.iloc[i]) > 1.5 * float(vol_sma.iloc[i])
            and float(close.iloc[i]) > float(close.iloc[i-1])):               score += 2
    return score >= 5


# ── Main backtest ─────────────────────────────────────────────────────────

def run_backtest(df: pd.DataFrame,
                 strategy        : str   = "ema9_support",
                 take_profit_pct : float = 5.0,
                 stop_loss_pct   : float = 3.0,
                 max_hold_days   : int   = 7,
                 initial_capital : float = 100_000.0) -> Dict[str, Any]:

    if len(df) < 50:
        return {"error": "Need at least 50 bars"}

    close   = df["Close"]
    high_s  = df["High"]
    low_s   = df["Low"]
    volume  = df["Volume"]

    # Pre-compute indicators
    rsi_s   = rsi(close)
    ml, sl_m, _ = macd(close)
    _, _, bb_lo = bollinger_bands(close)
    e9      = ema(close, 9)
    e21     = ema(close, 21)
    vol_sma = volume.rolling(20).mean()

    capital  = initial_capital
    position = None
    trades: List[Dict] = []
    equity_curve: List[Dict] = []

    for i in range(50, len(df)):
        bar_date = df.index[i]
        price    = float(close.iloc[i])
        bar_high = float(high_s.iloc[i])
        bar_low  = float(low_s.iloc[i])

        # ── Exit check ────────────────────────────────────────────────
        if position is not None:
            hold_days   = (bar_date - position["entry_date"]).days
            ep, er      = None, None

            if bar_high >= position["tp"]:
                ep, er = position["tp"], "Take Profit ✓"
            elif bar_low <= position["sl"]:
                ep, er = position["sl"], "Stop Loss ✗"
            elif hold_days >= max_hold_days:
                ep, er = price, "Timeout ⏱"

            if ep is not None:
                pnl     = (ep - position["entry_price"]) * position["shares"]
                pnl_pct = (ep / position["entry_price"] - 1) * 100
                capital += pnl
                trades.append({
                    "entry_date"  : str(position["entry_date"])[:10],
                    "exit_date"   : str(bar_date)[:10],
                    "entry_price" : round(position["entry_price"], 2),
                    "exit_price"  : round(ep, 2),
                    "shares"      : position["shares"],
                    "pnl"         : round(pnl, 2),
                    "pnl_pct"     : round(pnl_pct, 2),
                    "exit_reason" : er,
                    "win"         : pnl > 0,
                    "capital"     : round(capital, 2),
                    "hold_days"   : hold_days,
                })
                position = None

        # ── Entry check ───────────────────────────────────────────────
        if position is None and capital > price * 10:
            sig = False
            try:
                if   strategy == "rsi":
                    sig = _sig_rsi(close, rsi_s, i)
                elif strategy == "macd":
                    sig = _sig_macd(ml, sl_m, i)
                elif strategy == "bollinger":
                    sig = _sig_bollinger(close, bb_lo, rsi_s, i)
                elif strategy == "ema_cross":
                    sig = _sig_ema_cross(e9, e21, i)
                elif strategy == "volume_breakout":
                    sig = _sig_volume_breakout(close, volume, vol_sma, rsi_s, i)
                elif strategy in ("ema9_support", "hdfcamc"):
                    sig = _sig_ema9_support(close, low_s, rsi_s, e9, volume, vol_sma, i)
                else:  # composite
                    sig = _sig_composite(close, rsi_s, ml, sl_m, bb_lo, e9, e21, volume, vol_sma, i)
            except Exception:
                pass

            if sig:
                shares = max(1, int(capital * 0.95 / price))
                position = {
                    "entry_date"  : bar_date,
                    "entry_price" : price,
                    "shares"      : shares,
                    "tp"          : round(price * (1 + take_profit_pct / 100), 2),
                    "sl"          : round(price * (1 - stop_loss_pct   / 100), 2),
                }

        equity_curve.append({"date": str(bar_date)[:10], "capital": round(capital, 2)})

    return _build_result(trades, equity_curve, initial_capital, take_profit_pct, stop_loss_pct, strategy)


def _build_result(trades, equity_curve, init_cap, tp_pct, sl_pct, strategy):
    if not trades:
        return {"error": "No closed trades — try longer period or different strategy",
                "summary": {}, "trades": [], "equity_curve": equity_curve}

    df_t  = pd.DataFrame(trades)
    wins  = df_t[df_t["win"] == True]
    loss  = df_t[df_t["win"] == False]

    total   = len(df_t)
    n_wins  = len(wins)
    n_loss  = len(loss)
    wr      = n_wins / total * 100
    aw      = float(wins["pnl_pct"].mean())   if n_wins else 0.0
    al      = float(loss["pnl_pct"].mean())   if n_loss else 0.0
    tw      = float(wins["pnl"].sum())         if n_wins else 0.0
    tl      = float(loss["pnl"].sum())         if n_loss else 0.0
    pf      = abs(tw / tl)                     if tl != 0 else float("inf")
    final   = float(df_t["capital"].iloc[-1])
    ret     = (final - init_cap) / init_cap * 100
    rm      = df_t["capital"].cummax()
    dd      = ((df_t["capital"] - rm) / rm * 100).min()
    pls     = df_t["pnl_pct"] / 100
    sharpe  = (float(pls.mean()) / float(pls.std()) * np.sqrt(252)
               if float(pls.std()) > 0 else 0.0)

    summary = {
        "strategy"        : strategy,
        "total_trades"    : total,
        "winning_trades"  : n_wins,
        "losing_trades"   : n_loss,
        "win_rate_pct"    : round(wr, 2),
        "avg_win_pct"     : round(aw, 2),
        "avg_loss_pct"    : round(al, 2),
        "profit_factor"   : round(pf, 2),
        "total_return_pct": round(ret, 2),
        "initial_capital" : init_cap,
        "final_capital"   : round(final, 2),
        "max_drawdown_pct": round(float(dd), 2),
        "sharpe_ratio"    : round(sharpe, 2),
        "take_profit_hits": int((df_t["exit_reason"].str.contains("Take Profit")).sum()),
        "stop_loss_hits"  : int((df_t["exit_reason"].str.contains("Stop Loss")).sum()),
        "timeout_exits"   : int((df_t["exit_reason"].str.contains("Timeout")).sum()),
        "take_profit_pct" : tp_pct,
        "stop_loss_pct"   : sl_pct,
    }
    return {"summary": summary, "trades": trades, "equity_curve": equity_curve}