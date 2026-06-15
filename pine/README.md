# Fair Value — Pine Script Integration

Three files, one goal: show the same UNDERVALUED / FAIR / OVERVALUED
stepped fair-value line on your TradingView chart that your SMA screener
already computes.

---

## Files

| File | Purpose |
|------|---------|
| `fv_router.py` | New FastAPI router — bridge between Pine and your sma_router |
| `fair_value.pine` | Pine Script v5 indicator — paste into TradingView Pine Editor |
| `fv_test_harness.html` | Browser test — verify the endpoint draws the chart correctly first |

---

## Step 1 — Wire the router into main.py

```python
# main.py (add these two lines)
from fv_router import router as fv_router
app.include_router(fv_router)
```

Restart FastAPI. Test it manually:

```
http://localhost:8002/api/fv/pine?ticker=RELIANCE&fy=2023
# Should return one CSV line:
# 2023,1842.50,0.923,1790.00,0.851,0.00,0.000,1820.00,UNDERVALUED,+12.3
```

---

## Step 2 — Open the test harness

Open `fv_test_harness.html` in your browser (double-click it).

Enter your server URL, a ticker, and click **Fetch & Plot**.

You should see:
- Summary cards (CMP / FV / Gap% / Bucket)
- Price chart with the stepped FV line and ±15% band — exactly matching the screener
- Per-FY table (OP model, Sales model, TTM model, Composite)
- The raw CSV string that Pine Script will receive

If the chart looks right, proceed to Step 3.

---

## Step 3 — Load the Pine Script indicator

1. Open TradingView → Pine Editor (bottom panel) → New script
2. Paste the full contents of `fair_value.pine`
3. Change the `host` input to your server URL
   - If running locally: `http://localhost:8002`
   - If using ngrok: `https://xxxx.ngrok.io`
4. Click **Add to chart**
5. TradingView will show a security permission dialog — click **Allow**

---

## What you see on the chart

```
  ┌──────────────────────────────────────────────────────────────┐
  │  Yellow stepped line = Composite Fair Value                  │
  │  Green shaded band   = FV ± 15% (UNDERVALUED zone)          │
  │  Red shaded band     = FV ± 15% (OVERVALUED zone)           │
  │  Label at Apr 1      = FY2023  FV: ₹1820  UNDERVALUED +12%  │
  │  Label at last bar   = current CMP vs FV info box            │
  └──────────────────────────────────────────────────────────────┘
```

Optional dashed lines (toggle in Settings):
- Blue dashes = OP model alone
- Purple dashes = Sales model alone
- Orange dashes = TTM model alone

---

## How the FY stepping works

Indian FY: April 1 (year Y) → March 31 (year Y+1)

Pine computes `fy_year = month >= 4 ? year : year - 1` on every bar.
The bridge is called once per new FY year. When you scroll back to 2019,
Pine sends `fy=2019` and the bridge returns the 2019 regression fair value —
the same number your screener shows in the FY2019 row.

---

## Composite formula (mirrors sma_router.py exactly)

```
weight_op    = max(0.1, R²_op)  × 1.0
weight_sales = max(0.1, R²_s)   × 0.8
weight_ttm   = max(0.1, R²_ttm) × 1.2

composite = (FV_op × w_op + FV_s × w_s + FV_t × w_t)
            / (w_op + w_s + w_t)

bucket = UNDERVALUED  if composite_gain > +15%
         OVERVALUED   if composite_gain < −15%
         FAIR         otherwise
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Test harness shows "Error: Failed to fetch" | CORS header missing — check `fv_router.py` returns `Access-Control-Allow-Origin: *` |
| Pine shows `na` / flat line | TradingView blocked the URL — go to Chart Settings → Security and allow your domain |
| FV line looks flat at 0 | Ticker not found in screener data; check the CSV endpoint manually |
| "NODATA" in bucket | `_analyze_ticker` returned an error; check FastAPI logs |
| FV line doesn't step at April | Make sure you are on Daily or higher timeframe — Pine fetches at `"D"` resolution |
