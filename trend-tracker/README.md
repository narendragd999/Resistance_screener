# 📡 SNAPSCREENER — NSE Momentum Loss Screener

A Streamlit-based tool that identifies NSE stocks that had a strong upward momentum run (multiple consecutive green candles with significant gain), then experienced a reversal day. It suggests call-selling strikes based on live NSE option chain Open Interest data.

---

## Architecture

```
app.py
├── Config layer          load_config / save_config (config.json)
├── Persistence layer     load_json / save_json  (screening_data.json, backtest_data.json)
├── NSE Session Manager   initialize_nse_session / reset_nse_session
├── Cache layer           _cache_get / _cache_set  (in st.session_state["cache"])
│
├── Data Sources
│   ├── fetch_nse_data()       → NSE option-chain + live quote API
│   └── fetch_historical()     → yfinance OHLCV (NSE suffix .NS)
│
├── Signal Engine
│   ├── find_best_momentum_run()   → pure NumPy scan for longest qualifying green run
│   ├── parse_option_chain()       → normalises NSE option chain into a DataFrame
│   ├── get_resistance_strikes()   → top-N CE OI strikes above spot
│   ├── get_support_strikes()      → top-N PE OI strikes below spot
│   └── check_momentum()           → combines all above → signal Dict
│
├── Orchestrator
│   └── screen_tickers()           → loops tickers, progress bar, resets session
│
├── Backtest
│   └── backtest_momentum()        → historical signal detection (no live OI)
│
├── Charts (Plotly)
│   ├── candlestick_chart()        → OHLCV + annotations + volume
│   ├── oi_heatmap()               → grouped bar chart of CE/PE OI by strike
│   └── backtest_equity_chart()    → cumulative momentum gain over time
│
└── UI Tabs
    ├── tab_screener()     → live scanning, result cards, filter/sort
    ├── tab_backtest()     → historical replay
    ├── tab_oi_explorer()  → standalone OI chain viewer for any ticker
    └── tab_logs()         → in-app activity log viewer
```

---

## Key Improvements Over v1

| Area | Before | After |
|---|---|---|
| NSE session | Global mutable variable | `st.session_state["nse_session"]` with retry logic |
| Caching | Global `dict` lost on reload | Session-state cache with per-key TTL |
| Momentum detection | Duplicated loop in 3 places | Single `find_best_momentum_run()` pure function |
| OI data | Only strike + total OI | Full: CE/PE split, IV, OI change, PCR, LTP |
| Strike selection | Highest call premium | Resistance DataFrame exposed for full OI analysis |
| Telegram | Hardcoded message format | Clean structured alert with all relevant fields |
| Logging | `print()` scattered everywhere | `_log()` → Python logger + in-app log tab |
| UI | Basic Streamlit defaults | Dark trading terminal theme, metric cards, expandable charts |
| Charts | Single candlestick | Candlestick + volume + OI heatmap + backtest equity curve |
| Backtest | Partially broken | Clean backtest with cumulative chart and summary stats |
| New: OI Explorer | Missing | Standalone tab to inspect any ticker's full option chain |
| Config | Partially saved | All settings persist to config.json |
| Error handling | Bare `except Exception` | Structured error paths with log levels |

---

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Ticker CSV Format

Upload a CSV with a single column:

```
SYMBOL
RELIANCE
HDFCBANK
INFY
TCS
```

---

## Signal Logic

1. Fetch last N days of OHLCV (configurable lookback).
2. Find the longest consecutive green-candle run where:
   - Candles ≥ `min_green_candles`
   - Total gain ≥ `min_gain_percent`
3. Check current spot is **below** the previous close (reversal day).
4. Fetch live NSE option chain, find top-OI resistance strikes above spot.
5. Select best call-selling strike = highest call premium among top OI strikes.
6. If spot ≥ trigger high (high of first post-run candle) → mark as **Recovery**.
7. Send Telegram alert.

---

## Deployment (Hetzner CX43)

```bash
# Install
git clone <your-repo>
cd snapscreener
pip install -r requirements.txt

# Run with persistent session
nohup streamlit run app.py --server.port 8501 --server.headless true &

# Or with systemd (recommended)
# See: https://docs.streamlit.io/deploy/tutorials/docker
```

---

## Notes on NSE Rate Limiting

- NSE requires a valid browser session (cookies from homepage + derivatives page).
- The session is initialised once per screening run and reset afterwards.
- Between each ticker, no artificial sleep is added; the yfinance call acts as a natural delay.
- If you get repeated 403s, increase `session_retry_count` in config.json.
