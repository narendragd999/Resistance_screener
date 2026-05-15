"""
strategies.py — Pure technical analysis engine
9 EMA Support Detection + Full Confluence Framework
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List


# ── Indicator Calculations ─────────────────────────────────────────────────

def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta    = prices.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
         ) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast    = prices.ewm(span=fast,   adjust=False).mean()
    ema_slow    = prices.ewm(span=slow,   adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


def bollinger_bands(prices: pd.Series, period: int = 20, std_mult: float = 2.0
                    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
    sma = prices.rolling(period).mean()
    std = prices.rolling(period).std()
    return sma + std_mult * std, sma, sma - std_mult * std


def ema(prices: pd.Series, period: int) -> pd.Series:
    return prices.ewm(span=period, adjust=False).mean()


def vwap(df: pd.DataFrame) -> pd.Series:
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    return (typical * df["Volume"]).cumsum() / df["Volume"].cumsum()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.Series:
    _atr     = atr(df, period)
    hl_avg   = (df["High"] + df["Low"]) / 2
    upper    = hl_avg + multiplier * _atr
    lower    = hl_avg - multiplier * _atr
    vals     = [True] * len(df)
    for i in range(1, len(df)):
        if   df["Close"].iloc[i] > upper.iloc[i - 1]: vals[i] = True
        elif df["Close"].iloc[i] < lower.iloc[i - 1]: vals[i] = False
        else:                                          vals[i] = vals[i - 1]
    return pd.Series(vals, index=df.index)


# ── 9 EMA SUPPORT DETECTOR ─────────────────────────────────────────────────
# Core signal: price pulls back to 9 EMA, holds it, then bounces with volume

def ema9_support_check(df: pd.DataFrame,
                       tolerance_pct: float = 0.8,
                       min_volume: float = 1_000_000,
                       rsi_low: float = 50,
                       rsi_high: float = 72) -> Dict[str, Any]:
    """
    Detects when a stock is taking SUPPORT on the 9 EMA.

    Signal Logic (all checked on last 3 candles):
    ─────────────────────────────────────────────
    TRIGGER  : Previous candle's LOW came within tolerance% of 9 EMA
               AND current candle CLOSED above 9 EMA  ← bounce confirmed
    C1       : Price > 9 EMA  (trend filter)
    C2       : RSI 50–72      (momentum, not overbought)
    C3       : Volume > 1M    (participation)
    C4       : Volume > 20-day average (confirm the move)

    Returns trigger flag, condition breakdown, score, action.
    """
    if len(df) < 30:
        return {"error": "Need 30+ bars"}

    close   = df["Close"]
    low_s   = df["Low"]
    high_s  = df["High"]
    vol     = df["Volume"]

    _ema9   = ema(close, 9)
    _ema21  = ema(close, 21)
    _ema50  = ema(close, 50)
    _rsi    = rsi(close)
    vol_sma = vol.rolling(20).mean()
    _atr_s  = atr(df)

    # Latest values
    p      = float(close.iloc[-1])
    lo     = float(low_s.iloc[-1])
    hi     = float(high_s.iloc[-1])
    e9     = float(_ema9.iloc[-1])
    e9_p   = float(_ema9.iloc[-2])   # yesterday's EMA9
    e21    = float(_ema21.iloc[-1])
    e50    = float(_ema50.iloc[-1])
    r      = float(_rsi.iloc[-1])
    r_p    = float(_rsi.iloc[-2])
    v      = float(vol.iloc[-1])
    vs     = float(vol_sma.iloc[-1])
    v_ratio= v / vs if vs > 0 else 0
    cur_atr= float(_atr_s.iloc[-1])

    # Previous candle low touched 9 EMA (within tolerance%)
    prev_lo      = float(low_s.iloc[-2])
    ema9_touch   = prev_lo <= e9_p * (1 + tolerance_pct / 100)   # came to EMA or below
    bounce_above = p > e9                                          # closed back above EMA9
    
    # Also check current candle for intraday support
    intra_touch  = lo <= e9 * (1 + tolerance_pct / 100) and p > e9

    ema9_triggered = (ema9_touch and bounce_above) or intra_touch

    # ── 4 Confluence Conditions ────────────────────────────────────────
    c1_above_ema9 = p > e9
    c2_rsi_zone   = rsi_low <= r <= rsi_high
    c3_vol_1m     = v >= min_volume
    c4_vol_avg    = v_ratio >= 1.0   # at least avg volume (1.0× for normal, 1.5× strong)

    # Distance from EMA9 (how close is price to EMA?)
    ema9_distance_pct = abs(p - e9) / e9 * 100

    # Score
    score = sum([c1_above_ema9, c2_rsi_zone, c3_vol_1m, c4_vol_avg])

    # Entry zone — only if triggered or very close
    near_ema9 = ema9_distance_pct <= 1.5

    if ema9_triggered and score >= 3:
        action    = "🟢 BUY SIGNAL — 9 EMA Support Confirmed"
        signal    = "BUY"
        confidence= min(95, 60 + score * 8 + (5 if v_ratio >= 1.5 else 0))
    elif near_ema9 and score >= 3:
        action    = "🟡 WATCH — Approaching 9 EMA Support"
        signal    = "WATCH"
        confidence= min(75, 45 + score * 6)
    elif ema9_triggered and score == 2:
        action    = "🟡 WEAK SETUP — 9 EMA Touch, Partial Confluence"
        signal    = "WATCH"
        confidence= 40
    elif not c1_above_ema9:
        action    = "🔴 BELOW 9 EMA — No Long Setup"
        signal    = "AVOID"
        confidence= 10
    else:
        action    = "⚪ HOLD — No Setup Yet"
        signal    = "HOLD"
        confidence= 25

    target_5pct = round(p * 1.05, 2)
    stop_loss   = round(e9 - cur_atr * 0.5, 2)
    risk        = p - stop_loss
    reward      = target_5pct - p
    rr_ratio    = round(reward / risk, 2) if risk > 0 else 0

    # 30-day history for charting
    hist   = df.tail(30).copy()
    chart_data = {
        "dates"   : [str(d)[:10] for d in hist.index],
        "close"   : [round(float(x), 2) for x in hist["Close"]],
        "ema9"    : [round(float(x), 2) if not np.isnan(x) else None for x in _ema9.tail(30)],
        "ema21"   : [round(float(x), 2) if not np.isnan(x) else None for x in _ema21.tail(30)],
        "volume"  : [int(x) for x in hist["Volume"]],
        "vol_avg" : [round(float(x), 0) for x in vol_sma.tail(30)],
        "rsi"     : [round(float(x), 2) if not np.isnan(x) else None for x in _rsi.tail(30)],
    }

    return {
        "symbol"          : None,
        "current_price"   : round(p, 2),
        "ema9"            : round(e9, 2),
        "ema9_distance_pct": round(ema9_distance_pct, 2),
        "ema9_triggered"  : ema9_triggered,
        "near_ema9"       : near_ema9,
        "action"          : action,
        "signal"          : signal,
        "confidence"      : confidence,
        "score"           : score,
        "conditions"      : {
            "above_9ema" : {
                "pass" : c1_above_ema9,
                "value": f"Price ₹{p:.1f} vs EMA9 ₹{e9:.1f}  ({'+' if p>e9 else ''}{p-e9:.1f})",
                "label": "Price Above 9 EMA (Trend Filter)",
                "weight": 3,
            },
            "rsi_zone"   : {
                "pass" : c2_rsi_zone,
                "value": f"RSI = {r:.1f}  (target: {rsi_low}–{rsi_high})",
                "label": "RSI(14) in 50–72 (Momentum Zone)",
                "weight": 2,
            },
            "vol_1m"     : {
                "pass" : c3_vol_1m,
                "value": f"{v/1e6:.2f}M shares",
                "label": "Volume > 1M Shares (Participation)",
                "weight": 2,
            },
            "vol_avg"    : {
                "pass" : c4_vol_avg,
                "value": f"{v_ratio:.2f}× 20-day avg ({vs/1e6:.2f}M)",
                "label": "Volume ≥ 20-Day Average",
                "weight": 1,
            },
        },
        "ema9_touch_detail": {
            "prev_candle_low"  : round(prev_lo, 2),
            "ema9_yesterday"   : round(e9_p, 2),
            "touch_confirmed"  : ema9_touch,
            "bounce_confirmed" : bounce_above,
            "intraday_touch"   : intra_touch,
        },
        "indicators"    : {
            "rsi"      : round(r, 2),
            "ema9"     : round(e9, 2),
            "ema21"    : round(e21, 2),
            "ema50"    : round(e50, 2),
            "vol_ratio": round(v_ratio, 2),
            "volume"   : int(v),
            "vol_avg"  : int(vs),
            "atr"      : round(cur_atr, 2),
        },
        "targets"       : {
            "entry"      : round(p, 2),
            "stop_loss"  : stop_loss,
            "target_3pct": round(p * 1.03, 2),
            "target_5pct": target_5pct,
            "target_7pct": round(p * 1.07, 2),
            "rr_ratio"   : rr_ratio,
        },
        "chart_data"    : chart_data,
    }


# ── HDFCAMC Full Confluence Checker ──────────────────────────────────────

def hdfcamc_confluence(df: pd.DataFrame,
                        pullback_low: float = 2780,
                        pullback_high: float = 2800,
                        breakout_level: float = 2870,
                        vol_multiplier: float = 1.5,
                        min_volume: float = 1_000_000) -> Dict[str, Any]:
    if len(df) < 30:
        return {"error": "Need 30+ days"}

    close = df["Close"]
    vol   = df["Volume"]

    _rsi           = rsi(close)
    _ema9          = ema(close, 9)
    _ema21         = ema(close, 21)
    _ema50         = ema(close, 50)
    _macd, _sig, _ = macd(close)
    _bb_up, _bb_md, _bb_lo = bollinger_bands(close)
    _atr_s         = atr(df)
    vol_sma20      = vol.rolling(20).mean()

    p        = float(close.iloc[-1])
    r        = float(_rsi.iloc[-1])
    v        = float(vol.iloc[-1])
    va       = float(vol_sma20.iloc[-1])
    e9       = float(_ema9.iloc[-1])
    e21      = float(_ema21.iloc[-1])
    e50      = float(_ema50.iloc[-1])
    cur_atr  = float(_atr_s.iloc[-1])
    cur_low  = float(df["Low"].iloc[-1])
    vr       = v / va if va > 0 else 0

    c1 = p > e9
    c2 = 55 <= r <= 70
    c3 = v >= min_volume
    c4 = vr >= vol_multiplier
    c5 = cur_low >= pullback_low

    conditions = {
        "above_9ema"  : {"pass":c1,"value":f"₹{p:.0f} vs EMA9 ₹{e9:.0f}","label":"Price > 9 EMA"},
        "rsi_zone"    : {"pass":c2,"value":f"RSI = {r:.1f}","label":"RSI(14) 55–70"},
        "vol_1m"      : {"pass":c3,"value":f"{v/1e6:.2f}M shares","label":"Volume > 1M"},
        "vol_spike"   : {"pass":c4,"value":f"{vr:.2f}× avg","label":"Vol > 1.5× Avg"},
        "support_hold": {"pass":c5,"value":f"Low ₹{cur_low:.0f}","label":f"Above ₹{pullback_low:,.0f}"},
    }
    score = sum(1 for v_ in conditions.values() if v_["pass"])

    in_pullback = pullback_low <= p <= pullback_high
    at_breakout = p > breakout_level and vr >= vol_multiplier

    if at_breakout:
        entry_type, conf_adj = "BREAKOUT_BUY", 10
        entry_price          = breakout_level
    elif in_pullback:
        entry_type, conf_adj = "PULLBACK_BUY", 5
        entry_price          = p
    else:
        entry_type, conf_adj = "WAIT", 0
        entry_price          = None

    confidence = min(95, round((score / 5) * 75 + conf_adj))

    if score == 5 and entry_type != "WAIT": action = "STRONG BUY ✓"
    elif score >= 3 and entry_type != "WAIT": action = "BUY"
    elif score >= 3: action = "WATCH — Not In Zone"
    else: action = "HOLD / WAIT"

    tp5    = round(entry_price * 1.05, 2) if entry_price else None
    sl     = round(pullback_low - cur_atr * 0.5, 2) if entry_price else None
    rr     = round((tp5 - entry_price) / (entry_price - sl), 2) if (entry_price and sl and entry_price > sl) else None

    hist  = df.tail(30).copy()
    chart_data = {
        "dates"   : [str(d)[:10] for d in hist.index],
        "close"   : [round(float(x),2) for x in hist["Close"]],
        "ema9"    : [round(float(x),2) if not np.isnan(x) else None for x in _ema9.tail(30)],
        "ema21"   : [round(float(x),2) if not np.isnan(x) else None for x in _ema21.tail(30)],
        "volume"  : [int(x) for x in hist["Volume"]],
        "vol_avg" : [round(float(x),0) for x in vol_sma20.tail(30)],
        "rsi"     : [round(float(x),2) if not np.isnan(x) else None for x in _rsi.tail(30)],
        "bb_upper": [round(float(x),2) for x in _bb_up.tail(30)],
        "bb_lower": [round(float(x),2) for x in _bb_lo.tail(30)],
    }

    return {
        "symbol":"","current_price":round(p,2),"action":action,
        "entry_type":entry_type,"confidence":confidence,"score":score,
        "conditions":conditions,"entry_price":entry_price,
        "target_5pct":tp5,"stop_loss":sl,"rr_ratio":rr,
        "indicators":{"rsi":round(r,2),"ema9":round(e9,2),"ema21":round(e21,2),
                      "ema50":round(e50,2),"vol_ratio":round(vr,2),
                      "volume":int(v),"vol_avg":int(va),"atr":round(cur_atr,2)},
        "chart_data":chart_data,
        "zones":{"pullback_low":pullback_low,"pullback_high":pullback_high,"breakout_level":breakout_level},
    }


# ── General Multi-Strategy Signal Engine ──────────────────────────────────

def general_signals(df: pd.DataFrame) -> Dict[str, Any]:
    if len(df) < 50:
        return {"error":"Need 50+ bars"}

    close = df["Close"]
    vol   = df["Volume"]

    _rsi            = rsi(close)
    ml, sl_m, _     = macd(close)
    bb_up, _, bb_lo = bollinger_bands(close)
    e9,e21,e50      = ema(close,9),ema(close,21),ema(close,50)
    _atr_s          = atr(df)
    vol_sma         = vol.rolling(20).mean()

    p  = float(close.iloc[-1])
    r  = float(_rsi.iloc[-1])
    v  = float(vol.iloc[-1])
    vs = float(vol_sma.iloc[-1])
    ca = float(_atr_s.iloc[-1])

    strats = {}
    score  = 0

    if r < 35:   strats["RSI Oversold Reversal"]  = {"signal":"BUY","strength":"Strong","score":3};  score += 3
    elif r < 45: strats["RSI Near Oversold"]       = {"signal":"BUY","strength":"Moderate","score":2};score += 2
    elif r > 72: strats["RSI Overbought"]          = {"signal":"SELL","strength":"Strong","score":-3};score -= 3

    prev_diff = float(ml.iloc[-2]) - float(sl_m.iloc[-2])
    curr_diff = float(ml.iloc[-1]) - float(sl_m.iloc[-1])
    if prev_diff < 0 < curr_diff:  strats["MACD Bullish Crossover"] = {"signal":"BUY","strength":"Strong","score":3};  score += 3
    elif prev_diff > 0 > curr_diff:strats["MACD Bearish Crossover"] = {"signal":"SELL","strength":"Strong","score":-3};score -= 3
    elif curr_diff > 0:            strats["MACD Bullish Momentum"]  = {"signal":"BUY","strength":"Weak","score":1};   score += 1

    if p <= float(bb_lo.iloc[-1]) * 1.005:  strats["BB Lower Bounce"]    = {"signal":"BUY","strength":"Strong","score":3}; score += 3
    elif p >= float(bb_up.iloc[-1]) * 0.995:strats["BB Upper Rejection"] = {"signal":"SELL","strength":"Moderate","score":-2};score -= 2

    pe = float(e9.iloc[-2]) - float(e21.iloc[-2])
    ce = float(e9.iloc[-1]) - float(e21.iloc[-1])
    if pe < 0 < ce:         strats["EMA 9/21 Golden Cross"] = {"signal":"BUY","strength":"Strong","score":3};  score += 3
    elif ce > 0 and p>float(e50.iloc[-1]): strats["EMA Stack Bullish"] = {"signal":"BUY","strength":"Moderate","score":2};score += 2
    elif pe > 0 > ce:       strats["EMA Death Cross"]       = {"signal":"SELL","strength":"Strong","score":-3};score -= 3

    if v > 1.5 * vs and p > float(close.iloc[-2]): strats["Volume Breakout"] = {"signal":"BUY","strength":"Strong","score":3};score += 3

    if score >= 7:   rec,conf = "STRONG BUY",  min(95, 65+score*2)
    elif score >= 4: rec,conf = "BUY",          min(80, 55+score*2)
    elif score <= -6:rec,conf = "STRONG SELL",  min(95, 65+abs(score)*2)
    elif score <= -3:rec,conf = "SELL",          min(80, 55+abs(score)*2)
    else:            rec,conf = "HOLD/NEUTRAL",  50

    return {
        "price":round(p,2),"recommendation":rec,"confidence":int(conf),"score":score,
        "strategies":strats,
        "indicators":{"rsi":round(r,2),"ema9":round(float(e9.iloc[-1]),2),
                      "ema21":round(float(e21.iloc[-1]),2),"ema50":round(float(e50.iloc[-1]),2),
                      "macd":round(float(ml.iloc[-1]),4),"atr":round(ca,2),
                      "vol_ratio":round(v/vs,2) if vs else 0},
        "targets":{"entry":round(p,2),"target_5pct":round(p*1.05,2),
                   "stop_loss":round(p-ca*1.5,2)},
    }