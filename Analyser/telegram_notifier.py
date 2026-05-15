import requests
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class TelegramNotifier:
    def __init__(self):
        self.token = None
        self.chat_id = None
        self.configured = False

    def configure(self, token: str, chat_id: str):
        self.token = token.strip()
        self.chat_id = chat_id.strip()
        self.configured = True

    def _post(self, payload: dict) -> bool:
        if not self.configured:
            return False
        try:
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            r = requests.post(url, json=payload, timeout=10)
            if r.status_code != 200:
                logger.error(f"Telegram error: {r.text}")
            return r.status_code == 200
        except Exception as e:
            logger.error(f"Telegram post error: {e}")
            return False

    def test_connection(self) -> dict:
        ok = self._post({
            "chat_id": self.chat_id,
            "text": "✅ <b>Trading Analyzer Connected!</b>\n\nYour 9 EMA Confluence screener is active and ready to send alerts.",
            "parse_mode": "HTML"
        })
        return {"success": ok, "message": "Connected successfully!" if ok else "Connection failed. Check token and chat_id."}

    def send_message(self, text: str) -> bool:
        return self._post({
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "HTML"
        })

    def format_stock_alert(self, stock: dict, market: dict) -> str:
        sym = stock.get("symbol", "")
        metrics = stock.get("metrics", {})
        setup = stock.get("trade_setup", {})
        conds = stock.get("conditions", {})
        score = stock.get("score", 0)

        score_bar = "🟢" * score + "⚪" * (5 - score)
        mkt_emoji = {"bullish": "🟢", "neutral": "🟡", "weak": "🟠", "breakdown": "🔴"}.get(
            market.get("status", ""), "⚪"
        )
        day_chg = metrics.get("day_change_pct", 0)
        chg_emoji = "📈" if day_chg >= 0 else "📉"

        def ck(key):
            return "✅" if conds.get(key, {}).get("met") else "❌"

        text = f"""🔔 <b>TRADING SIGNAL — {sym}</b>
━━━━━━━━━━━━━━━━━━━━

📊 <b>Confluence Score:</b> {score}/5  {score_bar}

{chg_emoji} <b>Price:</b> ₹{metrics.get('close')} ({day_chg:+.2f}%)
📐 <b>9 EMA:</b> ₹{metrics.get('ema9')}  |  <b>21 EMA:</b> ₹{metrics.get('ema21')}
⚡ <b>RSI:</b> {metrics.get('rsi')}
📦 <b>Volume:</b> {int(metrics.get('volume', 0)):,}
📏 <b>ATR:</b> ₹{metrics.get('atr')}

<b>📋 Confluence Checklist:</b>
{ck('price_above_9ema')} Price above 9 EMA (daily trend)
{ck('rsi_55_70')} RSI between 55–70
{ck('volume_1m')} Volume &gt; 1M shares
{ck('ema9_support')} 9 EMA Support Bounce
{ck('macro_ok')} Nifty/Sensex aligned

<b>🎯 Trade Setup:</b>
▶️ <b>Entry:</b>     ₹{setup.get('entry')}
🛑 <b>Stop Loss:</b>  ₹{setup.get('stop_loss')} (-{setup.get('risk_pct')}%)
🎯 <b>Target 1:</b>  ₹{setup.get('target1')} (+5%)
🚀 <b>Target 2:</b>  ₹{setup.get('target2')} (+10%)
⚖️ <b>Risk:Reward:</b> 1:{setup.get('risk_reward')}

{mkt_emoji} <b>Market:</b> Nifty {market.get('status','').upper()} @ {market.get('current')} ({market.get('day_change_pct', 0):+.2f}%)

⏰ {datetime.now().strftime('%d %b %Y, %H:%M')} IST
⚠️ <i>For educational purposes only. Not SEBI registered advice.</i>"""
        return text

    def send_screening_summary(self, results: list, market: dict) -> bool:
        qualified = [r for r in results if r.get("all_conditions_met")]
        near_miss = [r for r in results if r.get("score", 0) >= 4 and not r.get("all_conditions_met")]
        mkt_emoji = {"bullish": "🟢", "neutral": "🟡", "weak": "🟠", "breakdown": "🔴"}.get(
            market.get("status", ""), "⚪"
        )
        now = datetime.now().strftime('%d %b %Y, %H:%M')

        if not qualified and not near_miss:
            return self.send_message(
                f"📊 <b>SCREENER RESULTS — {now}</b>\n\n"
                f"No stocks met all 5 confluence conditions.\n\n"
                f"{mkt_emoji} Nifty: {market.get('status','').upper()} @ {market.get('current')} "
                f"({market.get('day_change_pct',0):+.2f}%)"
            )

        text = f"📊 <b>SCREENER RESULTS — {now}</b>\n\n"

        if qualified:
            text += f"🚀 <b>{len(qualified)} stock(s) — ALL conditions met:</b>\n"
            for s in qualified[:8]:
                m = s.get("metrics", {})
                t = s.get("trade_setup", {})
                chg = m.get("day_change_pct", 0)
                text += (
                    f"  🔹 <b>{s['symbol']}</b>  ₹{m.get('close')} "
                    f"({chg:+.2f}%) | RSI {m.get('rsi')} | "
                    f"RR 1:{t.get('risk_reward')} | SL ₹{t.get('stop_loss')}\n"
                )

        if near_miss:
            text += f"\n⚡ <b>{len(near_miss)} stock(s) — Near miss (4/5):</b>\n"
            for s in near_miss[:5]:
                m = s.get("metrics", {})
                missing = [k for k, v in s.get("conditions", {}).items() if not v.get("met")]
                text += f"  🔸 <b>{s['symbol']}</b>  ₹{m.get('close')} | Missing: {', '.join(missing)}\n"

        text += (
            f"\n{mkt_emoji} <b>Market:</b> Nifty {market.get('status','').upper()} "
            f"@ {market.get('current')} ({market.get('day_change_pct',0):+.2f}%)\n"
            f"⚠️ <i>Not SEBI registered advice. DYOR.</i>"
        )
        return self.send_message(text)
