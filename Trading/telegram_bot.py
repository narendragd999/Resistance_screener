"""
telegram_bot.py — Send trade alerts to Telegram
"""
import os
import httpx
from typing import Optional
from datetime import datetime


TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")

# In-memory alert history (last 100)
_alert_history: list = []


def _fmt_signal_emoji(signal: str) -> str:
    return {"BUY": "🟢", "WATCH": "🟡", "AVOID": "🔴", "HOLD": "⚪"}.get(signal, "📊")


def build_alert_message(data: dict) -> str:
    """Build a rich Telegram alert message from signal data."""
    symbol    = data.get("symbol", "UNKNOWN").replace(".NS", "")
    price     = data.get("current_price") or data.get("price", 0)
    signal    = data.get("signal") or data.get("recommendation", "WATCH")
    action    = data.get("action", "")
    confidence= data.get("confidence", 0)
    score     = data.get("score", 0)
    e9        = data.get("ema9") or data.get("indicators", {}).get("ema9", 0)
    rsi_val   = data.get("indicators", {}).get("rsi", 0)
    vol_ratio = data.get("indicators", {}).get("vol_ratio", 0)
    targets   = data.get("targets", {})
    conditions= data.get("conditions", {})
    ema9_dist = data.get("ema9_distance_pct", 0)
    triggered = data.get("ema9_triggered", False)

    emoji = _fmt_signal_emoji(signal)

    lines = [
        f"{emoji} <b>NSE ALERT — {symbol}</b>",
        f"━━━━━━━━━━━━━━━━━━━━━━━",
        f"📌 <b>Signal:</b> {signal}",
        f"💰 <b>CMP:</b> ₹{price:,.2f}",
        f"📊 <b>9 EMA:</b> ₹{e9:,.2f}  ({'+' if price >= e9 else ''}{price - e9:.1f})",
    ]

    if triggered:
        lines.append(f"✅ <b>9 EMA SUPPORT TRIGGERED</b>")

    lines += [
        f"",
        f"<b>Confluence Score: {score}/4</b>",
    ]

    # Conditions
    cond_map = {
        "above_9ema" : "Price > 9 EMA",
        "rsi_zone"   : f"RSI {rsi_val:.0f} (target 50–72)",
        "vol_1m"     : "Volume > 1M",
        "vol_avg"    : f"Vol {vol_ratio:.1f}× avg",
    }
    for key, lbl in cond_map.items():
        if key in conditions:
            icon = "✅" if conditions[key].get("pass") else "❌"
            lines.append(f"  {icon} {lbl}")

    lines += [""]

    if targets:
        entry = targets.get("entry", price)
        sl    = targets.get("stop_loss", 0)
        t5    = targets.get("target_5pct", 0)
        rr    = targets.get("rr_ratio", 0)
        lines += [
            f"🎯 <b>Targets</b>",
            f"  Entry    : ₹{entry:,.2f}",
            f"  Target 5%: ₹{t5:,.2f}",
            f"  Stop Loss: ₹{sl:,.2f}",
            f"  R:R Ratio: {rr}",
        ]

    lines += [
        f"",
        f"🔐 Confidence: <b>{confidence}%</b>",
        f"⏰ {datetime.now().strftime('%d %b %Y  %H:%M IST')}",
        f"",
        f"<i>⚠ Not SEBI-registered advice. Do your own research.</i>",
    ]

    return "\n".join(lines)


async def send_telegram_alert(message: str,
                               token: Optional[str] = None,
                               chat_id: Optional[str] = None) -> dict:
    """Send a Telegram message. Returns status dict."""
    tok = token   or TELEGRAM_BOT_TOKEN
    cid = chat_id or TELEGRAM_CHAT_ID

    if not tok or not cid:
        return {"ok": False, "error": "Token or Chat ID not configured"}

    url = f"https://api.telegram.org/bot{tok}/sendMessage"
    payload = {
        "chat_id"   : cid,
        "text"      : message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(url, json=payload)
            result = resp.json()
            if result.get("ok"):
                _alert_history.insert(0, {
                    "time"   : datetime.now().isoformat(),
                    "message": message[:120] + "...",
                    "status" : "sent",
                })
                if len(_alert_history) > 100:
                    _alert_history.pop()
                return {"ok": True}
            else:
                return {"ok": False, "error": result.get("description", "Unknown error")}
    except Exception as e:
        return {"ok": False, "error": str(e)}


async def send_signal_alert(data: dict,
                             token: Optional[str] = None,
                             chat_id: Optional[str] = None) -> dict:
    """Build message from signal data and send."""
    msg = build_alert_message(data)
    return await send_telegram_alert(msg, token, chat_id)


def get_alert_history() -> list:
    return _alert_history
