import streamlit as st
import pandas as pd
import yfinance as yf
import time
import json
import os
import aiohttp
import asyncio
from typing import Dict, List, Optional
from datetime import datetime, date, timedelta
import numpy as np

# Constants
STORED_TICKERS_PATH = "tickers.csv"
CONFIG_FILE = "config.json"
SCREENING_DATA_FILE = "screening_data.json"
api_call_counter = 0

# Cache for performance
cache = {}

# Load/Save Config
def load_config() -> Dict:
    default_config = {
        "min_gain_percent": 20.0,
        "min_green_candles": 3,
        "lookback_days": 30,
        "telegram_bot_token": "",
        "telegram_chat_id": ""
    }
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
            for key, value in default_config.items():
                if key not in config:
                    config[key] = value
            return config
    return default_config

def save_config(config: Dict):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)

# Load/Save Screening Data to JSON
def load_screening_data() -> List[Dict]:
    if os.path.exists(SCREENING_DATA_FILE):
        with open(SCREENING_DATA_FILE, 'r') as f:
            return json.load(f)
    return []

def save_screening_data(data: List[Dict]):
    with open(SCREENING_DATA_FILE, 'w') as f:
        json.dump(data, f, indent=4)

# Telegram Integration
async def send_telegram_message(bot_token: str, chat_id: str, message: str):
    if not bot_token or not chat_id:
        print("Telegram credentials missing")
        return
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                print(f"Failed to send Telegram message: {response.status}")

def send_split_telegram_message(bot_token: str, chat_id: str, message: str):
    MAX_MESSAGE_LENGTH = 4096
    if len(message) <= MAX_MESSAGE_LENGTH:
        asyncio.run(send_telegram_message(bot_token, chat_id, message))
        return
    chunks = []
    current_chunk = message[:MAX_MESSAGE_LENGTH].strip()
    chunks.append(current_chunk)
    remaining = message[MAX_MESSAGE_LENGTH:]
    while remaining:
        next_chunk = remaining[:MAX_MESSAGE_LENGTH].strip()
        chunks.append(next_chunk)
        remaining = remaining[MAX_MESSAGE_LENGTH:]
    for chunk in chunks:
        asyncio.run(send_telegram_message(bot_token, chat_id, chunk))
        asyncio.sleep(0.5)

# Load Tickers
def load_tickers() -> List[str]:
    try:
        if os.path.exists(STORED_TICKERS_PATH):
            df = pd.read_csv(STORED_TICKERS_PATH)
            if 'SYMBOL' in df.columns:
                return df['SYMBOL'].dropna().tolist()
            st.error("Stored CSV file must contain 'SYMBOL' column")
            return ["HDFCBANK"]
        return ["HDFCBANK"]
    except Exception as e:
        st.error(f"Error loading tickers: {e}")
        return ["HDFCBANK"]

# Fetch Historical Prices using yfinance
def fetch_historical_data(ticker: str, start_date: date, end_date: date, refresh_key: float) -> Optional[pd.DataFrame]:
    global api_call_counter
    api_call_counter += 1
    cache_key = f"{ticker}_{start_date}_{end_date}_{refresh_key}"
    if cache_key in cache and (time.time() - cache.get(f"{cache_key}_timestamp", 0)) < 3600:
        print(f"Using cached data for {ticker}")
        return cache[cache_key]
    print(f"Fetching data for API {api_call_counter}--{ticker}")
    try:
        stock = yf.Ticker(ticker + ".NS")
        hist = stock.history(start=start_date, end=end_date + timedelta(days=1))
        if hist.empty:
            print(f"No historical data found for {ticker}. Possible delisted or invalid ticker.")
            return None
        cache[cache_key] = hist
        cache[f"{cache_key}_timestamp"] = time.time()
        print(f"Successfully fetched {len(hist)} days of data for {ticker}")
        return hist
    except Exception as e:
        print(f"Error fetching historical data for {ticker}: {str(e)}")
        return None

# Check Momentum Loss Criteria
def check_momentum_loss(ticker: str, hist: pd.DataFrame, min_gain_percent: float, min_green_candles: int, bot_token: str, chat_id: str) -> Optional[Dict]:
    if hist.empty or len(hist) < min_green_candles + 2:
        print(f"{ticker}: Insufficient data (only {len(hist)} days)")
        return None
    try:
        today = hist.index[-1]
        yesterday = hist.index[-2]
        today_close = hist.loc[today, 'Close']
        yesterday_close = hist.loc[yesterday, 'Close']
    except KeyError:
        print(f"{ticker}: Missing data for today or yesterday")
        return None
    if today_close >= yesterday_close:
        print(f"{ticker}: Today close ({today_close:.2f}) >= yesterday close ({yesterday_close:.2f})")
        return None
    hist = hist.sort_index()
    closes = hist['Close'].values
    dates = hist.index
    max_green_candles = 0
    max_gain = 0
    best_start_idx = None
    best_end_idx = None
    current_green_candles = 0
    current_start_idx = None
    print(f"{ticker}: Scanning for momentum periods...")
    for i in range(1, len(closes)):
        if closes[i] > closes[i-1]:
            current_green_candles += 1
            if current_green_candles == 1:
                current_start_idx = i - 1
            current_end_idx = i
            if current_green_candles >= min_green_candles:
                start_price = closes[current_start_idx]
                end_price = closes[current_end_idx]
                gain_percent = ((end_price - start_price) / start_price) * 100
                if gain_percent >= min_gain_percent and current_green_candles >= max_green_candles:
                    max_green_candles = current_green_candles
                    max_gain = gain_percent
                    best_start_idx = current_start_idx
                    best_end_idx = current_end_idx
                print(f"{ticker}: Found {current_green_candles} green candles from {dates[current_start_idx].strftime('%Y-%m-%d')} to {dates[current_end_idx].strftime('%Y-%m-%d')}, gain: {gain_percent:.2f}%")
        else:
            current_green_candles = 0
            current_start_idx = None
    if max_gain >= min_gain_percent and max_green_candles >= min_green_candles:
        momentum_start_date = dates[best_start_idx].strftime("%Y-%m-%d")
        momentum_end_date = dates[best_end_idx].strftime("%Y-%m-%d")
        result = {
            "Ticker": ticker,
            "Today_Close": float(today_close),
            "Yesterday_Close": float(yesterday_close),
            "Price_Drop_Percent": float(((yesterday_close - today_close) / yesterday_close) * 100),
            "Momentum_Gain_Percent": float(max_gain),
            "Green_Candle_Count": int(max_green_candles),
            "Momentum_Start_Date": momentum_start_date,
            "Momentum_End_Date": momentum_end_date,
            "Last_Scanned": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        message = (
            f"*Momentum Loss Alert*\n"
            f"Stock: *{ticker}*\n"
            f"Today Close: *₹{today_close:.2f}*\n"
            f"Yesterday Close: *₹{yesterday_close:.2f}*\n"
            f"Price Drop: *{result['Price_Drop_Percent']:.2f}%*\n"
            f"Prior Momentum Gain: *{max_gain:.2f}%*\n"
            f"Green Candles: *{max_green_candles}*\n"
            f"Momentum Period: *{momentum_start_date} to {momentum_end_date}*\n"
            f"Timestamp: *{result['Last_Scanned']}*"
        )
        print(f"{ticker}: Sending Telegram notification - {message}")
        send_split_telegram_message(bot_token, chat_id, message)
        return result
    print(f"{ticker}: No momentum loss (gain: {max_gain:.2f}%, green candles: {max_green_candles})")
    return None

# Screen Tickers for Momentum Loss
def screen_tickers(tickers: List[str], min_gain_percent: float, min_green_candles: int, lookback_days: int, bot_token: str, chat_id: str, refresh_key: float) -> List[Dict]:
    end_date = date.today()
    start_date = end_date - timedelta(days=lookback_days)
    results = []
    for ticker in tickers:
        with st.spinner(f"Screening {ticker} for momentum loss..."):
            hist = fetch_historical_data(ticker, start_date, end_date, refresh_key)
            if hist is None:
                st.warning(f"No data for {ticker}. It may be delisted or invalid.")
                continue
            result = check_momentum_loss(ticker, hist, min_gain_percent, min_green_candles, bot_token, chat_id)
            if result:
                results.append(result)
    print(f"Screening completed. Found {len(results)} stocks: {results}")
    return results

# Main Application
def main():
    st.set_page_config(page_title="Momentum Loss Screener", layout="wide")
    st.title("Real-Time Momentum Loss Screener")

    # Load Config
    config = load_config()
    if 'config' not in st.session_state:
        st.session_state['config'] = config

    # Initialize Session State
    if 'screening_data' not in st.session_state:
        st.session_state['screening_data'] = load_screening_data()
    if 'refresh_key' not in st.session_state:
        st.session_state['refresh_key'] = time.time()
    if 'last_scan_time' not in st.session_state:
        st.session_state['last_scan_time'] = time.time()
    if 'scan_in_progress' not in st.session_state:
        st.session_state['scan_in_progress'] = False

    # Sidebar Configuration
    with st.sidebar:
        st.subheader("Screening Settings")
        min_gain_percent = st.number_input(
            "Minimum Momentum Gain (%):",
            value=st.session_state['config']['min_gain_percent'],
            min_value=5.0,
            step=1.0,
            key="min_gain_percent"
        )
        min_green_candles = st.number_input(
            "Minimum Green Candles:",
            value=st.session_state['config']['min_green_candles'],
            min_value=1,
            step=1,
            key="min_green_candles"
        )
        lookback_days = st.number_input(
            "Lookback Period (Days):",
            value=st.session_state['config']['lookback_days'],
            min_value=10,
            step=1,
            key="lookback_days"
        )

        st.subheader("Telegram Integration")
        telegram_bot_token = st.text_input(
            "Telegram Bot Token:",
            value=st.session_state['config']['telegram_bot_token'],
            type="password",
            key="telegram_bot_token"
        )
        telegram_chat_id = st.text_input(
            "Telegram Chat ID:",
            value=st.session_state['config']['telegram_chat_id'],
            key="telegram_chat_id"
        )

        st.subheader("Upload Tickers")
        uploaded_file = st.file_uploader("Upload CSV with 'SYMBOL' column", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            if 'SYMBOL' in df.columns:
                df.to_csv(STORED_TICKERS_PATH, index=False)
                st.success(f"Tickers saved to {STORED_TICKERS_PATH}")
                st.session_state['refresh_key'] = time.time()
            else:
                st.error("CSV must contain 'SYMBOL' column")

        st.subheader("Screen Specific Stocks")
        specific_tickers = st.text_input("Enter tickers (comma-separated, e.g., HDFCBANK,RELIANCE,HUDCO):", key="specific_tickers")

        # Update config if changed
        config_changed = False
        if st.session_state['config']['min_gain_percent'] != min_gain_percent:
            st.session_state['config']['min_gain_percent'] = min_gain_percent
            config_changed = True
        if st.session_state['config']['min_green_candles'] != min_green_candles:
            st.session_state['config']['min_green_candles'] = min_green_candles
            config_changed = True
        if st.session_state['config']['lookback_days'] != lookback_days:
            st.session_state['config']['lookback_days'] = lookback_days
            config_changed = True
        if st.session_state['config']['telegram_bot_token'] != telegram_bot_token:
            st.session_state['config']['telegram_bot_token'] = telegram_bot_token
            config_changed = True
        if st.session_state['config']['telegram_chat_id'] != telegram_chat_id:
            st.session_state['config']['telegram_chat_id'] = telegram_chat_id
            config_changed = True
        if config_changed:
            save_config(st.session_state['config'])

        if not telegram_bot_token or not telegram_chat_id:
            st.warning("Please configure Telegram Bot Token and Chat ID for notifications.")

    # Main Content
    st.subheader("Momentum Loss Screening Results")
    st.write(f"Last Scan: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(st.session_state['last_scan_time']))}")
    if st.button("Refresh Results"):
        st.rerun()

    def perform_screening(tickers_to_screen):
        if st.session_state['scan_in_progress']:
            st.warning("Scan in progress. Please wait.")
            return
        st.session_state['scan_in_progress'] = True
        screening_data = screen_tickers(
            tickers_to_screen,
            st.session_state['config']['min_gain_percent'],
            st.session_state['config']['min_green_candles'],
            st.session_state['config']['lookback_days'],
            st.session_state['config']['telegram_bot_token'],
            st.session_state['config']['telegram_chat_id'],
            st.session_state['refresh_key']
        )
        print(f"Screening completed. Found {len(screening_data)} stocks: {screening_data}")
        st.session_state['screening_data'] = screening_data
        save_screening_data(screening_data)
        st.session_state['last_scan_time'] = time.time()
        st.session_state['scan_in_progress'] = False
        if not screening_data:
            st.info("No stocks met the momentum loss criteria. Try relaxing the settings or using different tickers.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Screen All Tickers") and not st.session_state['scan_in_progress']:
            tickers = load_tickers()
            perform_screening(tickers)
            st.rerun()
    with col2:
        if st.button("Screen Specific Tickers") and specific_tickers and not st.session_state['scan_in_progress']:
            tickers = [t.strip() for t in specific_tickers.split(',')]
            perform_screening(tickers)
            st.rerun()

    if st.session_state['screening_data']:
        st.write("### Stocks with Lost Momentum (Sortable & Searchable)")
        search_query = st.text_input("Search Results by Ticker", key="screening_search")
        screening_df = pd.DataFrame(st.session_state['screening_data'])
        if search_query:
            screening_df = screening_df[screening_df['Ticker'].str.contains(search_query, case=False, na=False)]
        st.dataframe(
            screening_df,
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker"),
                "Today_Close": st.column_config.NumberColumn("Today Close", format="%.2f"),
                "Yesterday_Close": st.column_config.NumberColumn("Yesterday Close", format="%.2f"),
                "Price_Drop_Percent": st.column_config.NumberColumn("Price Drop %", format="%.2f"),
                "Momentum_Gain_Percent": st.column_config.NumberColumn("Momentum Gain %", format="%.2f"),
                "Green_Candle_Count": st.column_config.NumberColumn("Green Candles"),
                "Momentum_Start_Date": st.column_config.TextColumn("Momentum Start"),
                "Momentum_End_Date": st.column_config.TextColumn("Momentum End"),
                "Last_Scanned": st.column_config.TextColumn("Last Scanned")
            },
            use_container_width=True,
            height=400
        )
        if st.button("Clear Screening Results"):
            st.session_state['screening_data'] = []
            save_screening_data(st.session_state['screening_data'])
            st.rerun()
    else:
        st.info("No results found. Run a scan to check for stocks with momentum loss.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Script error: {str(e)}")
        st.error(f"Script error: {str(e)}")