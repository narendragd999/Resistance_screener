import pandas as pd
import numpy as np
from screener import StockScreener
import logging

logger = logging.getLogger(__name__)


class Backtester:
    def __init__(self):
        self.screener = StockScreener()

    # ──────────────────────────────────────────────────────────────────────────
    # Helper: evaluate ALL 13 entry conditions at bar index `i`
    # using pre-computed arrays (no re-fetching on every candle).
    # ──────────────────────────────────────────────────────────────────────────
    def _check_entry_conditions_at(
        self,
        i: int,
        closes, highs, lows, opens, volumes,
        ema9, ema21,
        rsi_arr,
        macd_hist_arr,
        bb_bandwidth_arr,
        avg_vol_20_arr,
        high_52w: float,
        nifty_closes,
        weekly_closes,
        weekly_ema9_arr,
        sector_closes,
        sector_ema9_arr,
        rsi_min: float,
        rsi_max: float,
        min_volume: int,
        vol_spike_mult: float,
        bonus_required: int,
    ) -> dict:

        c   = float(closes.iloc[i])
        lo  = float(lows.iloc[i])
        o   = float(opens.iloc[i])
        vol = float(volumes.iloc[i])
        e9  = float(ema9.iloc[i])
        r   = float(rsi_arr.iloc[i])

        avg_vol = float(avg_vol_20_arr.iloc[i]) if avg_vol_20_arr is not None else float(min_volume)

        # ── CORE 1: Price above EMA9 ──────────────────────────────────────────
        cond1 = c > e9

        # ── CORE 2: RSI in band ───────────────────────────────────────────────
        cond2 = rsi_min <= r <= rsi_max

        # ── CORE 3: Volume liquidity ──────────────────────────────────────────
        cond3 = vol > min_volume

        # ── CORE 4: 9 EMA quality bounce (full screener logic) ───────────────
        proximity_pct = (c - e9) / e9 * 100
        ema_slope     = (e9 - float(ema9.iloc[i - 3])) / float(ema9.iloc[i - 3]) * 100

        touch_found     = False
        touch_vol_ratio = 1.0
        for k in range(0, 5):
            idx = i - k
            if idx < 0:
                break
            low_k   = float(lows.iloc[idx])
            ema_k   = float(ema9.iloc[idx])
            close_k = float(closes.iloc[idx])
            vol_k   = float(volumes.iloc[idx])
            pct_k   = (low_k - ema_k) / ema_k * 100
            prox_k  = (close_k - ema_k) / ema_k * 100
            if (-2.0 <= pct_k <= 0.5 and close_k > ema_k) or (0.0 <= prox_k <= 1.5):
                touch_found     = True
                touch_vol_ratio = vol_k / avg_vol if avg_vol > 0 else 1.0
                break

        pullback_quality_ok = touch_vol_ratio <= 1.0

        touch_count = 0
        for k in range(1, min(21, i + 1)):
            lk = float(lows.iloc[i - k])
            ek = float(ema9.iloc[i - k])
            if abs((lk - ek) / ek * 100) <= 1.5:
                touch_count += 1
        fresh_support = touch_count <= 3

        green_candle = c > o

        cond4 = (
            c > e9
            and touch_found
            and ema_slope > 0
            and 0.0 <= proximity_pct <= 5.0
            and green_candle
            and pullback_quality_ok
            and fresh_support
        )

        core_met = cond1 and cond2 and cond3 and cond4
        # (CORE 5 = macro_ok, checked by caller before this function)

        # ── BONUS 6: MACD histogram positive & rising ─────────────────────────
        try:
            curr_hist = float(macd_hist_arr.iloc[i])
            prev_hist = float(macd_hist_arr.iloc[i - 1])
            b6 = curr_hist > 0 and curr_hist > prev_hist
        except Exception:
            b6 = True

        # ── BONUS 7: Volume spike ≥ N× avg ───────────────────────────────────
        vol_ratio = vol / avg_vol if avg_vol > 0 else 0.0
        b7 = vol_ratio >= vol_spike_mult

        # ── BONUS 8: Near 52-week high (within 15%) ───────────────────────────
        pct_from_h = (c - high_52w) / high_52w * 100 if high_52w > 0 else 0.0
        b8 = pct_from_h >= -15.0

        # ── BONUS 9: Relative strength vs Nifty50 (20-bar) ───────────────────
        b9 = True
        if nifty_closes is not None and i >= 20:
            try:
                stock_ret = (c - float(closes.iloc[i - 20])) / float(closes.iloc[i - 20]) * 100
                nifty_ret = (
                    float(nifty_closes.iloc[-1]) - float(nifty_closes.iloc[-21])
                ) / float(nifty_closes.iloc[-21]) * 100
                b9 = stock_ret > nifty_ret
            except Exception:
                b9 = True

        # ── BONUS 10: Weekly EMA9 aligned ────────────────────────────────────
        b10 = True
        if weekly_closes is not None and weekly_ema9_arr is not None:
            try:
                wc   = float(weekly_closes.iloc[-1])
                we9  = float(weekly_ema9_arr.iloc[-1])
                wslp = (we9 - float(weekly_ema9_arr.iloc[-4])) / float(weekly_ema9_arr.iloc[-4]) * 100
                b10  = wc > we9 and wslp > 0
            except Exception:
                b10 = True

        # ── BONUS 11: Sector above its EMA9 ──────────────────────────────────
        b11 = True
        if sector_closes is not None and sector_ema9_arr is not None:
            try:
                sc   = float(sector_closes.iloc[-1])
                se9  = float(sector_ema9_arr.iloc[-1])
                sslp = (se9 - float(sector_ema9_arr.iloc[-4])) / float(sector_ema9_arr.iloc[-4]) * 100
                b11  = sc > se9 and sslp > 0
            except Exception:
                b11 = True

        # ── BONUS 12: Bollinger Band squeeze ─────────────────────────────────
        b12 = True
        if bb_bandwidth_arr is not None:
            try:
                bw_curr = float(bb_bandwidth_arr.iloc[i])
                bw_prev = float(bb_bandwidth_arr.iloc[i - 1])
                b12 = (not pd.isna(bw_curr)) and bw_curr < 15.0 and bw_curr <= bw_prev
            except Exception:
                b12 = True

        # ── BONUS 13: Fresh EMA touch ─────────────────────────────────────────
        b13 = fresh_support

        bonus_flags = [b6, b7, b8, b9, b10, b11, b12, b13]
        bonus_met   = sum(bonus_flags)
        all_met     = core_met and bonus_met >= bonus_required

        return {
            "all_conditions_met": all_met,
            "core_met":           core_met,
            "bonus_met":          bonus_met,
            "proximity_pct":      round(proximity_pct, 2),
            "ema_slope":          round(ema_slope, 3),
            "touch_found":        touch_found,
            "green_candle":       green_candle,
            "pullback_quality":   pullback_quality_ok,
            "fresh_support":      fresh_support,
            "touch_count":        touch_count,
            "touch_vol_ratio":    round(touch_vol_ratio, 2),
            "vol_ratio":          round(vol_ratio, 2),
            "pct_from_52h":       round(pct_from_h, 2),
            "bonus_details": {
                "macd_pos_rising": b6,
                "vol_spike":       b7,
                "near_52w_high":   b8,
                "rs_vs_nifty":     b9,
                "weekly_aligned":  b10,
                "sector_strong":   b11,
                "bb_squeeze":      b12,
                "fresh_touch":     b13,
            },
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Main backtest
    # ──────────────────────────────────────────────────────────────────────────
    def run_backtest(
        self,
        symbol: str,
        period: str = "2y",
        stop_loss_atr_mult: float = 1.5,   # ATR-based stop loss
        stop_loss_pct: float = 2.0,         # % fallback SL (if ATR unavailable)
        target1_pct: float = 5.0,
        target2_pct: float = 10.0,
        rsi_min: float = 55.0,
        rsi_max: float = 70.0,
        min_volume: int = 1_000_000,
        partial_exit: bool = True,
        vol_spike_mult: float = 1.5,        # volume spike multiplier
        bonus_required: int = 5,            # how many of 8 bonus conditions needed
        require_macro_ok: bool = True,
        trailing_stop_after_t1: bool = True,
    ) -> dict:

        # ── 1. OHLCV ─────────────────────────────────────────────────────────
        data = self.screener.get_stock_data(symbol, period=period)
        if data is None or len(data) < 50:
            return {"error": f"Insufficient data for {symbol}"}

        closes  = data["Close"]
        highs   = data["High"]
        lows    = data["Low"]
        opens   = data["Open"]
        volumes = data["Volume"]

        # ── 2. Indicators ────────────────────────────────────────────────────
        ema9     = self.screener.calculate_ema(closes, 9)
        ema21    = self.screener.calculate_ema(closes, 21)
        rsi_arr  = self.screener.calculate_rsi(closes)
        atr_arr  = self.screener.calculate_atr(data)
        _, _, macd_hist = self.screener.calculate_macd(closes)

        bb_mid  = closes.rolling(20).mean()
        bb_std  = closes.rolling(20).std()
        bb_bw   = ((bb_mid + 2 * bb_std) - (bb_mid - 2 * bb_std)) / bb_mid * 100

        avg_vol_20 = volumes.rolling(20).mean()
        high_52w   = float(highs.max())

        # ── 3. External series (fetched once) ────────────────────────────────
        nifty_data   = self.screener._get_index_data("^NSEI", period="3mo")
        nifty_closes = nifty_data["Close"] if nifty_data is not None else None

        weekly_data     = self.screener.get_stock_data(symbol, period="1y", interval="1wk")
        weekly_closes   = weekly_data["Close"] if weekly_data is not None else None
        weekly_ema9_arr = self.screener.calculate_ema(weekly_closes, 9) if weekly_closes is not None else None

        from screener import SECTOR_MAP, DEFAULT_SECTOR
        sector_ticker   = SECTOR_MAP.get(symbol.upper(), DEFAULT_SECTOR)
        sector_data     = self.screener._get_index_data(sector_ticker, period="6mo")
        sector_closes   = sector_data["Close"] if sector_data is not None else None
        sector_ema9_arr = self.screener.calculate_ema(sector_closes, 9) if sector_closes is not None else None

        # ── 4. Macro filter ───────────────────────────────────────────────────
        macro_ok = True
        if require_macro_ok:
            try:
                market   = self.screener.get_market_status()
                macro_ok = market.get("healthy", True)
            except Exception as e:
                logger.warning(f"Market status fetch failed: {e}. Assuming OK.")

        # ── 5. Simulation ────────────────────────────────────────────────────
        trades      = []
        in_trade    = False
        entry_price = 0.0
        entry_date  = None
        stop_price  = 0.0
        t1_price    = 0.0
        t2_price    = 0.0
        t1_hit      = False

        warmup = 30

        for i in range(warmup, len(data)):
            c    = float(closes.iloc[i])
            h    = float(highs.iloc[i])
            lo   = float(lows.iloc[i])
            e9   = float(ema9.iloc[i])
            date = data.index[i]

            # ── EXIT ─────────────────────────────────────────────────────────
            if in_trade:
                # Trailing stop after T1: trail at EMA9 - 0.5%
                if t1_hit and trailing_stop_after_t1:
                    stop_price = max(stop_price, e9 * 0.995)

                # Stop loss
                if lo <= stop_price:
                    pnl_pct = (stop_price - entry_price) / entry_price * 100
                    trades.append(self._make_trade(
                        symbol, entry_date, date,
                        entry_price, stop_price, stop_price,
                        t1_price, t2_price, pnl_pct, "Stop Loss"
                    ))
                    in_trade = False
                    t1_hit   = False
                    continue

                # Target 1
                if not t1_hit and h >= t1_price:
                    if not partial_exit:
                        pnl_pct = (t1_price - entry_price) / entry_price * 100
                        trades.append(self._make_trade(
                            symbol, entry_date, date,
                            entry_price, t1_price, stop_price,
                            t1_price, t2_price, pnl_pct, "Target 1"
                        ))
                        in_trade = False
                        t1_hit   = False
                        continue
                    else:
                        t1_hit     = True
                        stop_price = max(stop_price, entry_price * 1.002)

                # Target 2
                if h >= t2_price:
                    pnl_pct     = (t2_price - entry_price) / entry_price * 100
                    exit_reason = "Target 2 (Partial)" if t1_hit else "Target 2"
                    trades.append(self._make_trade(
                        symbol, entry_date, date,
                        entry_price, t2_price, stop_price,
                        t1_price, t2_price, pnl_pct, exit_reason
                    ))
                    in_trade = False
                    t1_hit   = False
                    continue

                # EMA9 breakdown (before T1)
                if c < e9 and not t1_hit:
                    pnl_pct = (c - entry_price) / entry_price * 100
                    trades.append(self._make_trade(
                        symbol, entry_date, date,
                        entry_price, c, stop_price,
                        t1_price, t2_price, pnl_pct, "EMA9 Breakdown"
                    ))
                    in_trade = False
                    t1_hit   = False
                    continue

            # ── ENTRY ─────────────────────────────────────────────────────────
            else:
                if not macro_ok:
                    continue

                ec = self._check_entry_conditions_at(
                    i=i,
                    closes=closes, highs=highs, lows=lows,
                    opens=opens, volumes=volumes,
                    ema9=ema9, ema21=ema21,
                    rsi_arr=rsi_arr,
                    macd_hist_arr=macd_hist,
                    bb_bandwidth_arr=bb_bw,
                    avg_vol_20_arr=avg_vol_20,
                    high_52w=high_52w,
                    nifty_closes=nifty_closes,
                    weekly_closes=weekly_closes,
                    weekly_ema9_arr=weekly_ema9_arr,
                    sector_closes=sector_closes,
                    sector_ema9_arr=sector_ema9_arr,
                    rsi_min=rsi_min,
                    rsi_max=rsi_max,
                    min_volume=min_volume,
                    vol_spike_mult=vol_spike_mult,
                    bonus_required=bonus_required,
                )

                if ec["all_conditions_met"]:
                    in_trade    = True
                    entry_price = c
                    entry_date  = date

                    curr_atr   = float(atr_arr.iloc[i])
                    sl_by_atr  = c - stop_loss_atr_mult * curr_atr
                    sl_by_pct  = e9 * (1 - stop_loss_pct / 100)
                    stop_price = max(sl_by_atr, sl_by_pct)

                    t1_price = entry_price * (1 + target1_pct / 100)
                    t2_price = entry_price * (1 + target2_pct / 100)

                    logger.debug(
                        f"[{symbol}] ENTRY {date.date()} @ ₹{entry_price:.2f} | "
                        f"SL=₹{stop_price:.2f} ATR={curr_atr:.2f} | "
                        f"Bonus {ec['bonus_met']}/8"
                    )

        # Close open trade at end
        if in_trade:
            c   = float(closes.iloc[-1])
            pnl = (c - entry_price) / entry_price * 100
            trades.append(self._make_trade(
                symbol, entry_date, data.index[-1],
                entry_price, c, stop_price,
                t1_price, t2_price, pnl, "End of Period"
            ))

        # ── 6. No trades ──────────────────────────────────────────────────────
        if not trades:
            return {
                "symbol": symbol, "period": period,
                "total_trades": 0,
                "winning_trades": 0, "losing_trades": 0,
                "win_rate": 0,
                "total_return": 0, "avg_gain": 0, "avg_loss": 0,
                "best_trade": 0, "worst_trade": 0,
                "max_drawdown": 0, "sharpe_ratio": 0,
                "profit_factor": 0, "expectancy_pct": 0,
                "avg_holding_days": 0,
                "equity_curve": [100], "equity_dates": [],
                "trades": [], "exit_reasons": {},
                "message": "No trades triggered. Try loosening RSI range, reducing bonus_required, or using a longer period.",
                "params": self._params_summary(
                    period, stop_loss_atr_mult, stop_loss_pct,
                    target1_pct, target2_pct, rsi_min, rsi_max,
                    min_volume, vol_spike_mult, bonus_required,
                    require_macro_ok, trailing_stop_after_t1,
                ),
            }

        # ── 7. Statistics ─────────────────────────────────────────────────────
        wins   = [t for t in trades if t["win"]]
        losses = [t for t in trades if not t["win"]]

        win_rate     = len(wins) / len(trades) * 100
        avg_gain     = float(np.mean([t["pnl_pct"] for t in wins]))   if wins   else 0.0
        avg_loss     = float(np.mean([t["pnl_pct"] for t in losses])) if losses else 0.0
        total_return = sum(t["pnl_pct"] for t in trades)
        avg_hold     = float(np.mean([t["holding_days"] for t in trades]))
        expectancy   = round((win_rate / 100 * avg_gain) + ((1 - win_rate / 100) * avg_loss), 2)

        equity   = [100.0]
        eq_dates = []
        for t in trades:
            equity.append(equity[-1] * (1 + t["pnl_pct"] / 100))
            eq_dates.append(t["exit_date"])

        max_dd = 0.0
        peak   = equity[0]
        for v in equity:
            peak  = max(peak, v)
            max_dd = max(max_dd, (peak - v) / peak * 100)

        returns = [t["pnl_pct"] for t in trades]
        sharpe  = (float(np.mean(returns) / np.std(returns))
                   if len(returns) > 1 and np.std(returns) > 0 else 0.0)

        gross_win  = sum(t["pnl_pct"] for t in wins)
        gross_loss = abs(sum(t["pnl_pct"] for t in losses))
        pf = round(gross_win / gross_loss, 2) if gross_loss > 0 else 99.0

        exit_reasons: dict = {}
        for t in trades:
            exit_reasons[t["exit_reason"]] = exit_reasons.get(t["exit_reason"], 0) + 1

        return {
            "symbol":           symbol,
            "period":           period,
            "total_trades":     len(trades),
            "winning_trades":   len(wins),
            "losing_trades":    len(losses),
            "win_rate":         round(win_rate, 1),
            "total_return":     round(total_return, 2),
            "avg_gain":         round(avg_gain, 2),
            "avg_loss":         round(avg_loss, 2),
            "best_trade":       round(max((t["pnl_pct"] for t in trades), default=0), 2),
            "worst_trade":      round(min((t["pnl_pct"] for t in trades), default=0), 2),
            "max_drawdown":     round(max_dd, 2),
            "sharpe_ratio":     round(sharpe, 2),
            "profit_factor":    pf,
            "expectancy_pct":   expectancy,
            "avg_holding_days": round(avg_hold, 1),
            "equity_curve":     [round(e, 2) for e in equity],
            "equity_dates":     eq_dates,
            "trades":           trades,
            "exit_reasons":     exit_reasons,
            "params":           self._params_summary(
                period, stop_loss_atr_mult, stop_loss_pct,
                target1_pct, target2_pct, rsi_min, rsi_max,
                min_volume, vol_spike_mult, bonus_required,
                require_macro_ok, trailing_stop_after_t1,
            ),
        }

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _make_trade(
        self, symbol, entry_date, exit_date,
        entry, exit_p, sl, t1, t2, pnl, reason
    ) -> dict:
        return {
            "symbol":       symbol,
            "entry_date":   entry_date.strftime("%Y-%m-%d"),
            "exit_date":    exit_date.strftime("%Y-%m-%d"),
            "entry_price":  round(entry, 2),
            "exit_price":   round(exit_p, 2),
            "stop_loss":    round(sl, 2),
            "target1":      round(t1, 2),
            "target2":      round(t2, 2),
            "pnl_pct":      round(pnl, 2),
            "exit_reason":  reason,
            "win":          pnl > 0,
            "holding_days": max(1, (exit_date - entry_date).days),
        }

    def _params_summary(
        self, period, atr_mult, sl_pct, t1, t2,
        rsi_min, rsi_max, min_vol, vol_mult,
        bonus_req, macro, trailing
    ) -> dict:
        return {
            "period":                  period,
            "stop_loss_atr_mult":      atr_mult,
            "stop_loss_pct_fallback":  sl_pct,
            "target1_pct":             t1,
            "target2_pct":             t2,
            "rsi_range":               f"{rsi_min}–{rsi_max}",
            "min_volume":              min_vol,
            "vol_spike_mult":          vol_mult,
            "bonus_required":          f"{bonus_req}/8",
            "require_macro_ok":        macro,
            "trailing_stop_after_t1":  trailing,
            "total_conditions":        13,
            "core_conditions":         5,
            "bonus_conditions":        8,
        }