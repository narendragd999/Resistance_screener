import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz

# Set timezone to IST
ist = pytz.timezone('Asia/Kolkata')

# Streamlit app title
st.title("Range-Bound NSE Stock Screener (Using Closing Prices)")

# File uploader for CSV
uploaded_file = st.file_uploader("Upload CSV with NSE Ticker List", type=["csv"])

# Slider to select years of historical data (2 or 3)
years = st.slider("Select years of historical data", min_value=2, max_value=3, value=3)

# Dropdown to select minimum touches at high/low boundaries
min_touches = st.selectbox("Minimum touches at high/low boundaries", [2, 3], index=1)

# Dropdown to select touch threshold (5% or 7%)
touch_threshold_pct = st.selectbox("Touch threshold for boundaries (%)", [5, 7], index=0)
touch_threshold = 1 - (touch_threshold_pct / 100)  # For high: 0.95 if 5%, 0.93 if 7%
touch_threshold_low = 1 + (touch_threshold_pct / 100)  # For low: 1.05 if 5%, 1.07 if 7%

# Checkbox to include stocks regardless of current price proximity
include_all_ranges = st.checkbox("Include stocks regardless of current price proximity to boundary", value=True)

# Debug log display
st.write("### Debug Log")
debug_log = st.empty()

if uploaded_file is not None:
    # Read CSV file
    try:
        tickers_df = pd.read_csv(uploaded_file)
        if 'Ticker' not in tickers_df.columns:
            st.error("CSV must contain a 'Ticker' column with NSE symbols (e.g., SRF, RELIANCE, TCS).")
        else:
            tickers = tickers_df['Ticker'].tolist()
            st.write(f"Loaded {len(tickers)} NSE tickers from CSV.")

            # Initialize results and debug logs
            results = []
            debug_logs = []

            # Calculate start and end dates in IST
            end_date = datetime.now(ist)
            start_date = end_date - timedelta(days=years * 365)
            start_date_52w = end_date - timedelta(days=365)

            # Progress bar and status text
            st.write("### Processing Tickers")
            progress_text = st.empty()
            progress_bar = st.progress(0)
            total_tickers = len(tickers)

            for i, ticker in enumerate(tickers):
                # Update progress text
                progress_text.text(f"Processing {ticker} ({i + 1}/{total_tickers}, {((i + 1) / total_tickers * 100):.1f}% complete)")

                try:
                    # Append .NS for NSE stocks
                    nse_ticker = f"{ticker}.NS"
                    stock = yf.Ticker(nse_ticker)
                    hist = stock.history(start=start_date, end=end_date)
                    hist_52w = stock.history(start=start_date_52w, end=end_date)

                    if hist.empty or hist_52w.empty:
                        debug_logs.append(f"{ticker}: No data available")
                        continue

                    # Use closing prices to determine initial range
                    close_prices = hist['Close']
                    initial_high = close_prices.max()
                    initial_low = close_prices.min()
                    initial_range_pct = ((initial_high - initial_low) / initial_low) * 100

                    # Log initial range (based on closing prices)
                    debug_logs.append(f"{ticker}: Initial High (Close): {initial_high:.2f}, Low (Close): {initial_low:.2f}, Range: {initial_range_pct:.2f}%")

                    # Compress the range by clustering frequent closing price levels
                    # Bin closing prices into 50 INR intervals
                    bin_size = 50
                    close_bins = np.floor(close_prices / bin_size) * bin_size

                    # Find the most frequent price levels
                    close_counts = close_bins.value_counts()

                    # Select the compressed high and low with sufficient touches
                    compressed_high = None
                    compressed_low = None
                    for high in sorted(close_counts.index, reverse=True):
                        for low in sorted(close_counts.index):
                            if high <= low:
                                continue
                            # Count touches using closing prices within 5% of the bin
                            high_touches = sum(1 for c in close_prices if high * 0.95 <= c <= high * 1.05)
                            low_touches = sum(1 for c in close_prices if low * 0.95 <= c <= low * 1.05)
                            if high_touches >= min_touches and low_touches >= min_touches:
                                compressed_high = high
                                compressed_low = low
                                break
                        if compressed_high is not None:
                            break

                    if compressed_high is None or compressed_low is None:
                        debug_logs.append(f"{ticker}: No compressed range found with sufficient touches")
                        continue

                    # Calculate compressed range percentage
                    compressed_range_pct = ((compressed_high - compressed_low) / compressed_low) * 100

                    # Log compressed range
                    debug_logs.append(f"{ticker}: Compressed High (Close): {compressed_high:.2f}, Low (Close): {compressed_low:.2f}, Range: {compressed_range_pct:.2f}%")

                    # Define boundary thresholds for the compressed range
                    high_threshold = compressed_high * touch_threshold
                    low_threshold = compressed_low * touch_threshold_low

                    # Count touches in the compressed range using closing prices
                    high_touches = sum(1 for c in close_prices if c >= high_threshold)
                    low_touches = sum(1 for c in close_prices if c <= low_threshold)

                    # Check minimum touches
                    if high_touches < min_touches or low_touches < min_touches:
                        debug_logs.append(f"{ticker}: Insufficient touches in compressed range (High: {high_touches} at {high_threshold:.2f}, Low: {low_touches} at {low_threshold:.2f})")
                        continue

                    # Check current price proximity
                    current_price = hist['Close'][-1]
                    near_high = current_price >= compressed_high * touch_threshold
                    near_low = current_price <= compressed_low * touch_threshold_low
                    ready_for_third = near_high or near_low

                    if not include_all_ranges and not ready_for_third:
                        debug_logs.append(f"{ticker}: Current price {current_price:.2f} not near compressed boundary (Low: {compressed_low:.2f}, High: {compressed_high:.2f})")
                        continue

                    # Check 52-week low (using closing prices for consistency)
                    low_52w = hist_52w['Close'].min()
                    near_52w_low = current_price <= low_52w * touch_threshold_low

                    # Store results
                    results.append({
                        'Ticker': ticker,
                        'Low Price (INR)': round(compressed_low, 2),
                        'High Price (INR)': round(compressed_high, 2),
                        'Range %': round(compressed_range_pct, 2),
                        'High Touches': high_touches,
                        'Low Touches': low_touches,
                        'Current Price (INR)': round(current_price, 2),
                        'Near Boundary': 'High' if near_high else 'Low' if near_low else 'N/A',
                        'Near 52-Week Low': 'Yes' if near_52w_low else 'No',
                        'Near Low Touch': 'Yes' if near_low else 'No'
                    })

                    debug_logs.append(f"{ticker}: Included (Compressed Range: {compressed_range_pct:.2f}%, High Touches: {high_touches}, Low Touches: {low_touches})")

                except Exception as e:
                    debug_logs.append(f"{ticker}: Error - {str(e)}")
                    continue

                # Update progress bar
                progress_bar.progress((i + 1) / total_tickers)

            # Update debug log (limit to 10 lines for readability)
            debug_log.text("\n".join(debug_logs[:10]) + ("\n...and more" if len(debug_logs) > 10 else ""))

            # Clear progress text
            progress_text.text("Processing complete!")

            # Display results
            if results:
                results_df = pd.DataFrame(results)
                st.write("### Range-Bound NSE Stocks")
                st.dataframe(results_df)
            else:
                st.write("No range-bound stocks found with at least {} touches at both boundaries.".format(min_touches))

    except Exception as e:
        st.error(f"Error reading CSV: {str(e)}")
else:
    st.write("Please upload a CSV file with a 'Ticker' column containing NSE symbols (e.g., SRF, RELIANCE, TCS).")