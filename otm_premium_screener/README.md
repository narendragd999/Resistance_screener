# OTM Premium Sell Zone Screener
NSE Options Screener — 2–6 OTM options with huge premium surge + volume near expiry.

## What it does
- Scans all 201 NSE F&O tickers from `tickers.csv`
- Identifies options that are **2–6 strikes OTM** from ATM
- Screens for options that have:
  - **Huge Premium** (LTP > threshold, default ₹15)
  - **Premium Surge** (today's premium vs 3-day average, default 30%)
  - **Volume Surge** (today's volume vs 3-day average, default 2.5x)
  - **Near Expiry** (0–3 days to expiry, configurable)
- Built for **options selling** strategy (sell OTM options before expiry)

## Scoring (0–100)
| Component       | Weight |
|----------------|--------|
| Premium (LTP)   | 25 pts |
| Premium Surge % | 30 pts |
| Volume Ratio    | 25 pts |
| Implied Vol     | 10 pts |
| Open Interest   | 10 pts |

## Project Structure
```
otm_premium_screener/
├── main.py           # FastAPI server + all API endpoints
├── screener.py       # Core screening logic + scoring + caching
├── nse_client.py     # NSE session management + option chain parsing
├── config.json       # All tunable parameters
├── tickers.csv       # NSE F&O ticker list (input)
├── requirements.txt
├── signals.json      # Last scan results (auto-generated)
├── scan_log.json     # Scan history (auto-generated)
├── data/             # Per-symbol premium/volume history cache
│   └── SYMBOL_EXPIRY_STRIKE_TYPE.json
├── backtest_cache/   # Reserved for future backtesting
└── static/
    └── index.html    # Dark UI frontend
```

## Setup & Run

```bash
pip install -r requirements.txt
python main.py
```
Open: http://localhost:8000

## API Reference
| Endpoint | Description |
|----------|-------------|
| `POST /api/scan` | Start background scan (body: config overrides) |
| `GET /api/scan/progress` | Live scan progress |
| `GET /api/signals` | Get signals with filters |
| `GET /api/signals/top` | Top confirmed signals (both surge gates passed) |
| `GET /api/scan/log` | Scan history |
| `GET /api/symbol/{sym}` | Quick single-symbol scan |
| `GET /api/config` | Current scan config |
| `POST /api/config` | Update config at runtime |
| `GET /api/tickers` | Loaded ticker list |

## Signal Gates
- 🟢 **Premium OK** — LTP ≥ min_premium  
- 🟢 **OI OK** — Open Interest ≥ min_oi  
- 🟡 **Premium Surge** — Today's LTP > 3-day avg by surge_pct% (requires 1+ day of cached history)  
- 🟡 **Volume Surge** — Today's volume > 3-day avg × multiplier (requires 1+ day of cached history)  

> Surge gates require at least 1 prior day of cached data. Run the screener daily to build history.

## Notes
- NSE has rate limits — scan_delay_seconds (default 1.5s) prevents bans
- Session is auto-warmed up before each API call
- History is cached in `data/` folder (JSON per strike)
- Signals are persisted to `signals.json` after each scan
