#!/bin/bash
# ──────────────────────────────────────────────────────
#  NSE 9 EMA SWING ANALYZER — Startup Script
#  Usage: bash start.sh
# ──────────────────────────────────────────────────────
cd "$(dirname "$0")"

echo ""
echo "  ███████╗    ███████╗███╗   ███╗ █████╗     ███████╗██╗    ██╗██╗███╗   ██╗ ██████╗ "
echo "  ██╔═══╝    ██╔════╝████╗ ████║██╔══██╗    ██╔════╝██║    ██║██║████╗  ██║██╔════╝ "
echo "  ███████╗   █████╗  ██╔████╔██║███████║    ███████╗██║ █╗ ██║██║██╔██╗ ██║██║  ███╗"
echo "  ╚════██║   ██╔══╝  ██║╚██╔╝██║██╔══██║    ╚════██║██║███╗██║██║██║╚██╗██║██║   ██║"
echo "  ███████║   ███████╗██║ ╚═╝ ██║██║  ██║    ███████║╚███╔███╔╝██║██║ ╚████║╚██████╔╝"
echo "  ╚══════╝   ╚══════╝╚═╝     ╚═╝╚═╝  ╚═╝    ╚══════╝ ╚══╝╚══╝ ╚═╝╚═╝  ╚═══╝ ╚═════╝ "
echo ""
echo "  NSE 9 EMA Support Detector · Screener · Backtester · Telegram Alerts"
echo "  ─────────────────────────────────────────────────────────────────────"
echo ""

# ── Check Python ──────────────────────────────────────
if ! command -v python3 &>/dev/null && ! command -v python &>/dev/null; then
  echo "  ❌  Python not found. Install Python 3.9+ and try again."
  exit 1
fi

PYTHON=$(command -v python3 || command -v python)
echo "  ✓  Python: $($PYTHON --version)"

# ── Install dependencies ──────────────────────────────
echo "  Checking dependencies..."
if ! $PYTHON -c "import fastapi,yfinance,pandas,apscheduler" 2>/dev/null; then
  echo "  Installing packages from requirements.txt..."
  $PYTHON -m pip install -r requirements.txt -q
  if [ $? -ne 0 ]; then
    echo "  ❌  pip install failed. Try: pip install -r requirements.txt"
    exit 1
  fi
fi
echo "  ✓  All dependencies OK"

# ── Telegram env vars (optional) ─────────────────────
if [ -n "$TELEGRAM_BOT_TOKEN" ]; then
  echo "  ✓  Telegram token found in environment"
fi

# ── Start server ──────────────────────────────────────
echo ""
echo "  ┌────────────────────────────────────────────┐"
echo "  │   Dashboard : http://localhost:8000        │"
echo "  │   API Docs  : http://localhost:8000/docs   │"
echo "  │   Press Ctrl+C to stop                     │"
echo "  └────────────────────────────────────────────┘"
echo ""

$PYTHON -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
