import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go

# Streamlit app title and description
st.title("Stock Screener: Stocks Near 200-Day Moving Average")
st.write("""
This app screens stocks whose current price is within a specified percentage range of their 200-day Moving Average (MA).
Enter a list of tickers (e.g., SURYODAY.NS), set the threshold percentage (positive or negative), and view results with price vs. 200 MA charts.
""")

# Default list of Indian small finance bank tickers
default_tickers = [
    'ABBOTINDIA.NS',
    'AKZOINDIA.NS',
    'ASIANPAINT.NS',
    'AXISBANK.NS',
    'BAJAJ-AUTO.NS',
    'BAJFINANCE.NS',
    'BAJAJFINSV.NS',
    'BAJAJHLDNG.NS',
    'BERGEPAINT.NS',
    'COLPAL.NS',
    'DABUR.NS',
    'GILLETTE.NS',
    'GLAXO.NS',
    'HAVELLS.NS',
    'HCLTECH.NS',
    'HDFCAMC.NS',
    'HDFCBANK.NS',
    'HDFCLIFE.NS',
    'HONAUT.NS',
    'ICICIBANK.NS',
    'ICICIGI.NS',
    'ICICIPRULI.NS',
    'INDUSINDBK.NS',
    'INFY.NS',
    'ITC.NS',
    'KOTAKBANK.NS',
    'MARICO.NS',
    'MOTILALOFS.NS',
    'NESTLEIND.NS',
    'NAM-INDIA.NS',
    'PGHH.NS',
    'PAGEIND.NS',
    'PFIZER.NS',
    'PIDILITIND.NS',
    'RELAXO.NS',
    'RELIANCE.NS',
    'SANOFI.NS',
    'TCS.NS',
    'TITAN.NS',
    'WHIRLPOOL.NS'
]

# Input form for user parameters
st.sidebar.header("Screener Parameters")
ticker_input = st.sidebar.text_area(
    "Enter tickers (one per line, e.g., SURYODAY.NS)",
    value="\n".join(default_tickers),
    height=200
)
threshold_percent = st.sidebar.slider(
    "Threshold (% within 200 MA)",
    min_value=-30.0,  # Allow negative threshold
    max_value=10.0,
    value=5.0,
    step=0.5
)
ma_period = 200  # Fixed 200-day MA
data_period = st.sidebar.selectbox(
    "Data Period",
    options=["6mo", "1y", "2y"],
    index=0
)

# Convert data period to days
period_to_days = {"6mo": 180, "1y": 365, "2y": 730}
days = period_to_days[data_period]

# Process ticker input
tickers = [ticker.strip().upper() for ticker in ticker_input.split("\n") if ticker.strip()]

# Parameters for data fetching
end_date = datetime.now()
start_date = pd.to_datetime(end_date) - pd.Timedelta(days=days)

# Function to screen stocks near 200 MA
@st.cache_data
def screen_stocks_near_200ma(tickers, ma_period, threshold_percent, start_date, end_date):
    results = []
    stock_data = {}
    
    for ticker in tickers:
        try:
            # Fetch stock data
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)
            
            if len(df) < ma_period:
                st.warning(f"Not enough data for {ticker} to calculate {ma_period}-day MA")
                continue
            
            # Calculate 200-day MA
            df['200_MA'] = df['Close'].rolling(window=ma_period).mean()
            
            # Get the latest closing price and 200 MA
            latest_price = df['Close'].iloc[-1]
            latest_200ma = df['200_MA'].iloc[-1]
            
            # Check if the price is within ±threshold% of 200 MA
            threshold = abs(threshold_percent) / 100  # Use absolute value for range
            lower_bound = latest_200ma * (1 - threshold)
            upper_bound = latest_200ma * (1 + threshold)
            
            if lower_bound <= latest_price <= upper_bound:
                percent_diff = ((latest_price - latest_200ma) / latest_200ma) * 100
                if threshold_percent < 0:
                    # For negative threshold, only include stocks below 200 MA
                    if percent_diff <= 0:
                        results.append({
                            'Ticker': ticker,
                            'Latest Price': round(latest_price, 2),
                            '200 MA': round(latest_200ma, 2),
                            '% Difference': round(percent_diff, 2)
                        })
                        stock_data[ticker] = df[['Close', '200_MA']]
                else:
                    # For positive or zero threshold, include stocks within range
                    results.append({
                        'Ticker': ticker,
                        'Latest Price': round(latest_price, 2),
                        '200 MA': round(latest_200ma, 2),
                        '% Difference': round(percent_diff, 2)
                    })
                    stock_data[ticker] = df[['Close', '200_MA']]
            
        except Exception as e:
            st.error(f"Error processing {ticker}: {e}")
    
    return pd.DataFrame(results), stock_data

# Function to plot price vs. 200 MA
def plot_stock_data(ticker, df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['200_MA'],
        mode='lines',
        name='200-Day MA',
        line=dict(color='orange')
    ))
    fig.update_layout(
        title=f"{ticker}: Close Price vs. 200-Day MA",
        xaxis_title="Date",
        yaxis_title="Price (INR)",
        template="plotly_white",
        showlegend=True
    )
    return fig

# Run the screener when the user clicks the button
if st.button("Run Screener"):
    if not tickers:
        st.error("Please enter at least one valid ticker.")
    else:
        with st.spinner("Fetching data and screening stocks..."):
            results_df, stock_data = screen_stocks_near_200ma(tickers, ma_period, threshold_percent, start_date, end_date)
        
        # Display results
        if not results_df.empty:
            st.success(f"Found {len(results_df)} stocks within {threshold_percent}% of their 200-day MA:")
            st.dataframe(results_df, use_container_width=True)
            
            # Download button for CSV
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="stocks_near_200ma.csv",
                mime="text/csv"
            )
            
            # Plot charts for each qualifying stock
            st.subheader("Price vs. 200 MA Charts")
            for ticker in results_df['Ticker']:
                st.write(f"**{ticker}**")
                fig = plot_stock_data(ticker, stock_data[ticker])
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No stocks found within {threshold_percent}% of their 200-day MA.")

# Footer
st.markdown("---")
st.write("Built with Streamlit, yfinance, and Plotly. Data as of May 25, 2025.")