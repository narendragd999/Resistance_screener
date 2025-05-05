import streamlit as st
import pandas as pd
import yfinance as yf
import time
import json
import os
import aiohttp
import asyncio
import cloudscraper
from typing import Dict, List, Optional
from datetime import datetime, date, timedelta
import numpy as np
import uuid

# Constants
BASE_URL = "https://www.nseindia.com"
STORED_TICKERS_PATH = "tickers.csv"
CONFIG_FILE = "config.json"
SCREENING_DATA_FILE = "screening_data.json"
api_call_counter = 0

# Cache for performance
cache = {}

# Initialize cloudscraper session
scraper = cloudscraper.create_scraper()
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/market-data/equity-derivatives-watch",
}

# Set up cookies by visiting NSE pages
def initialize_nse_session():
    print("Visiting homepage...")
    response = scraper.get("https://www.nseindia.com/", headers=headers)
    if response.status_code != 200:
        print(f"Failed to load homepage: {response.status_code}")
        return False
    print("Visiting derivatives page...")
    response = scraper.get("https://www.nseindia.com/market-data/equity-derivatives-watch", headers=headers)
    time.sleep(2)
    if response.status_code != 200:
        print(f"Failed to load derivatives page: {response.status_code}")
        return False
    return True

# Load/Save Config
def load_config() -> Dict:
    default_config = {
        "min_gain_percent": 20.0,
        "min_green_candles": 3,
        "lookback_days": 30,
        "telegram_bot_token": "",
        "telegram_chat_id": "",
        "min_call_price_increase": 5.0,
        "strike_price_proximity": 1.15,
        "otm_strike_threshold": 1.05  # 5% above current price for OTM suggestion
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
    current_chunk = message[:MAX_MESSAGE_LENGTH].rstrip()
    chunks.append(current_chunk)
    remaining = message[MAX_MESSAGE_LENGTH:]
    while remaining:
        next_chunk = remaining[:MAX_MESSAGE_LENGTH].rstrip()
        chunks.append(next_chunk)
        remaining = remaining[MAX_MESSAGE_LENGTH:]
    for chunk in chunks:
        asyncio.run(send_telegram_message(bot_token, chat_id, chunk))
        time.sleep(0.5)

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
    print(f"Fetching historical data for API {api_call_counter}--{ticker}")
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

# Fetch Options Data from NSE
def fetch_options_data(ticker: str, refresh_key: float) -> Optional[pd.DataFrame]:
    global api_call_counter
    api_call_counter += 1
    cache_key = f"options_{ticker}_{refresh_key}"
    if cache_key in cache and (time.time() - cache.get(f"{cache_key}_timestamp", 0)) < 3600:
        print(f"Using cached options data for {ticker}")
        return cache[cache_key]
    print(f"Fetching options data for API {api_call_counter}--{ticker}")
    try:
        url = f"{BASE_URL}/api/option-chain-equities?symbol={ticker}"
        response = scraper.get(url, headers=headers)
        if response.status_code != 200:
            print(f"Failed to fetch option chain for {ticker}: {response.status_code}")
            return None
        data = response.json()
        if not data.get('records') or not data['records'].get('data'):
            print(f"No option chain data for {ticker}")
            return None

        expiry_dates = sorted(data['records']['expiryDates'])
        if not expiry_dates:
            print(f"No expiry dates found for {ticker}")
            return None
        nearest_expiry = expiry_dates[0]
        print(f"Using nearest expiry for {ticker}: {nearest_expiry}")

        calls_data = []
        for option in data['records']['data']:
            if option['expiryDate'] == nearest_expiry and 'CE' in option:
                ce = option['CE']
                calls_data.append({
                    'contractSymbol': ce['identifier'],
                    'strike': float(ce['strikePrice']),
                    'lastPrice': float(ce['lastPrice']),
                    'bid': float(ce.get('bidprice', ce['lastPrice'])),
                    'ask': float(ce.get('askPrice', ce['lastPrice'])),
                    'volume': int(ce.get('totalTradedVolume', 0)),
                    'openInterest': int(ce.get('openInterest', 0)),
                    'change': float(ce.get('change', 0)),
                    'changeinOpenInterest': int(ce.get('changeinOpenInterest', 0)),
                    'expiration': nearest_expiry
                })

        if not calls_data:
            print(f"No call options found for {ticker} at expiry {nearest_expiry}")
            return None

        calls_df = pd.DataFrame(calls_data)
        cache[cache_key] = calls_df
        cache[f"{cache_key}_timestamp"] = time.time()
        print(f"Successfully fetched {len(calls_df)} call options for {ticker}, expiration {nearest_expiry}")
        return calls_df
    except Exception as e:
        print(f"Error fetching options data for {ticker}: {str(e)}")
        return None

# Analyze Call Options for Selling with OTM Suggestion
def analyze_call_options(ticker: str, current_price: float, calls: pd.DataFrame, min_price_increase: float, strike_proximity: float, otm_strike_threshold: float) -> Dict:
    suggestions = []
    otm_suggestion = None
    if calls is None or calls.empty:
        print(f"{ticker}: No call options data to analyze")
        return {"suggestions": suggestions, "otm_suggestion": otm_suggestion}
    try:
        print(f"{ticker}: Analyzing {len(calls)} call options, current price: {current_price:.2f}")
        max_strike = current_price * strike_proximity
        min_otm_strike = current_price * otm_strike_threshold  # Minimum strike for OTM suggestion
        relevant_calls = calls[calls['strike'] <= max_strike]
        if relevant_calls.empty:
            print(f"{ticker}: No call options within {strike_proximity*100-100}% above current price ({current_price:.2f})")
            return {"suggestions": suggestions, "otm_suggestion": otm_suggestion}

        # General call suggestions
        for _, row in relevant_calls.iterrows():
            last_price = row['lastPrice']
            prev_price = row.get('lastPrice') - row.get('change', 0) if row.get('change', 0) != 0 else row.get('bid', last_price)
            if pd.isna(prev_price) or prev_price == 0:
                print(f"{ticker}: Skipping option (strike {row['strike']}) due to invalid previous price")
                continue
            price_increase = ((last_price - prev_price) / prev_price) * 100
            if price_increase >= min_price_increase:
                suggestion = {
                    "Ticker": ticker,
                    "Strike_Price": float(row['strike']),
                    "Last_Price": float(last_price),
                    "Price_Increase_Percent": float(price_increase),
                    "Expiration_Date": row['expiration'],
                    "Contract_Symbol": row['contractSymbol'],
                    "Volume": int(row.get('volume', 0)),
                    "Open_Interest": int(row.get('openInterest', 0))
                }
                suggestions.append(suggestion)
                print(f"{ticker}: Found call option - Strike: {row['strike']:.2f}, Last Price: {last_price:.2f}, Price Increase: {price_increase:.2f}%")

        # OTM strike suggestion
        otm_calls = calls[calls['strike'] >= min_otm_strike]
        if not otm_calls.empty:
            # Sort by strike price and find the nearest rounded strike (e.g., 100, 105)
            otm_calls = otm_calls.sort_values('strike')
            for _, row in otm_calls.iterrows():
                strike = row['strike']
                # Check if strike is "rounded" (divisible by 5 or 10)
                if strike % 5 == 0:  # Adjust to 10 for larger stocks if needed
                    last_price = row['lastPrice']
                    prev_price = row.get('lastPrice') - row.get('change', 0) if row.get('change', 0) != 0 else row.get('bid', last_price)
                    if pd.isna(prev_price) or prev_price == 0:
                        continue
                    price_increase = ((last_price - prev_price) / prev_price) * 100
                    if price_increase >= min_price_increase and row.get('volume', 0) > 0:  # Ensure some liquidity
                        otm_suggestion = {
                            "Ticker": ticker,
                            "Strike_Price": float(strike),
                            "Last_Price": float(last_price),
                            "Price_Increase_Percent": float(price_increase),
                            "Expiration_Date": row['expiration'],
                            "Contract_Symbol": row['contractSymbol'],
                            "Volume": int(row.get('volume', 0)),
                            "Open_Interest": int(row.get('openInterest', 0))
                        }
                        print(f"{ticker}: OTM suggestion - Strike: {strike:.2f}, Last Price: {last_price:.2f}, Price Increase: {price_increase:.2f}%")
                        break  # Take the first valid OTM strike

        if not suggestions:
            print(f"{ticker}: No call options met the price increase threshold ({min_price_increase}%)")
        if not otm_suggestion:
            print(f"{ticker}: No suitable OTM strike found above {min_otm_strike:.2f} with significant price increase")
        return {"suggestions": suggestions, "otm_suggestion": otm_suggestion}
    except Exception as e:
        print(f"Error analyzing call options for {ticker}: {str(e)}")
        return {"suggestions": suggestions, "otm_suggestion": otm_suggestion}

# Check Momentum Loss Criteria and Include Options Suggestions
def check_momentum_loss(ticker: str, hist: pd.DataFrame, min_gain_percent: float, min_green_candles: int, bot_token: str, chat_id: str, min_call_price_increase: float, strike_proximity: float, otm_strike_threshold: float, refresh_key: float) -> Optional[Dict]:
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
        calls = fetch_options_data(ticker, refresh_key)
        call_analysis = analyze_call_options(ticker, today_close, calls, min_call_price_increase, strike_proximity, otm_strike_threshold) if calls is not None else {"suggestions": [], "otm_suggestion": None}
        call_suggestions = call_analysis["suggestions"]
        otm_suggestion = call_analysis["otm_suggestion"]
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
            "Last_Scanned": time.strftime("%Y-%m-%d %H:%M:%S"),
            "Call_Suggestions": call_suggestions,
            "OTM_Suggestion": otm_suggestion,
            "Raw_Calls_Data": calls.to_dict() if calls is not None else {}
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
            f"Timestamp: *{result['Last_Scanned']}*\n"
        )
        if call_suggestions:
            message += "\n*Call Option Selling Suggestions:*\n"
            for suggestion in call_suggestions:
                message += (
                    f"- Strike: *₹{suggestion['Strike_Price']:.2f}*, "
                    f"Last Price: *₹{suggestion['Last_Price']:.2f}*, "
                    f"Price Increase: *{suggestion['Price_Increase_Percent']:.2f}%*, "
                    f"Expiration: *{suggestion['Expiration_Date']}*, "
                    f"Volume: *{suggestion['Volume']}*, "
                    f"Open Interest: *{suggestion['Open_Interest']}*\n"
                )
        else:
            message += "\n*No Call Option Suggestions*: No options met the general criteria.\n"
        if otm_suggestion:
            message += "\n*Recommended OTM Strike for Call Selling:*\n"
            message += (
                f"- Strike: *₹{otm_suggestion['Strike_Price']:.2f}*, "
                f"Last Price: *₹{otm_suggestion['Last_Price']:.2f}*, "
                f"Price Increase: *{otm_suggestion['Price_Increase_Percent']:.2f}%*, "
                f"Expiration: *{otm_suggestion['Expiration_Date']}*, "
                f"Volume: *{otm_suggestion['Volume']}*, "
                f"Open Interest: *{otm_suggestion['Open_Interest']}*\n"
            )
        else:
            message += "\n*No OTM Strike Suggestion*: No suitable OTM strike with significant price increase.\n"
        print(f"{ticker}: Sending Telegram notification - {message}")
        send_split_telegram_message(bot_token, chat_id, message)
        return result
    print(f"{ticker}: No momentum loss (gain: {max_gain:.2f}%, green candles: {max_green_candles})")
    return None

# Screen Tickers for Momentum Loss
def screen_tickers(tickers: List[str], min_gain_percent: float, min_green_candles: int, lookback_days: int, bot_token: str, chat_id: str, min_call_price_increase: float, strike_proximity: float, otm_strike_threshold: float, refresh_key: float) -> List[Dict]:
    end_date = date.today()
    start_date = end_date - timedelta(days=lookback_days)
    results = []
    for ticker in tickers:
        with st.spinner(f"Screening {ticker} for momentum loss and call options..."):
            hist = fetch_historical_data(ticker, start_date, end_date, refresh_key)
            if hist is None:
                st.warning(f"No data for {ticker}. It may be delisted or invalid.")
                continue
            result = check_momentum_loss(ticker, hist, min_gain_percent, min_green_candles, bot_token, chat_id, min_call_price_increase, strike_proximity, otm_strike_threshold, refresh_key)
            if result:
                results.append(result)
    print(f"Screening completed. Found {len(results)} stocks: {results}")
    return results

# Main Application
def main():
    st.set_page_config(page_title="Momentum Loss Screener with NSE Options", layout="wide")
    st.title("Real-Time Momentum Loss Screener with NSE Call Option Suggestions")

    # Initialize NSE session
    if not initialize_nse_session():
        st.error("Failed to initialize NSE session. Please try again later.")
        return

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

        st.subheader("Call Option Settings")
        min_call_price_increase = st.number_input(
            "Minimum Call Price Increase (%):",
            value=st.session_state['config']['min_call_price_increase'],
            min_value=1.0,
            step=1.0,
            key="min_call_price_increase"
        )
        strike_proximity = st.number_input(
            "Max Strike Price Proximity (% above current):",
            value=(st.session_state['config']['strike_price_proximity'] - 1) * 100,
            min_value=0.0,
            step=1.0,
            key="strike_proximity"
        ) / 100 + 1
        otm_strike_threshold = st.number_input(
            "OTM Strike Threshold (% above current):",
            value=(st.session_state['config']['otm_strike_threshold'] - 1) * 100,
            min_value=0.0,
            step=1.0,
            key="otm_strike_threshold"
        ) / 100 + 1

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
        specific_tickers = st.text_input("Enter tickers (comma-separated, e.g., HDFCBANK,RELIANCE,NBCC):", key="specific_tickers")

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
        if st.session_state['config']['min_call_price_increase'] != min_call_price_increase:
            st.session_state['config']['min_call_price_increase'] = min_call_price_increase
            config_changed = True
        if st.session_state['config']['strike_price_proximity'] != strike_proximity:
            st.session_state['config']['strike_price_proximity'] = strike_proximity
            config_changed = True
        if st.session_state['config']['otm_strike_threshold'] != otm_strike_threshold:
            st.session_state['config']['otm_strike_threshold'] = otm_strike_threshold
            config_changed = True
        if config_changed:
            save_config(st.session_state['config'])

        if not telegram_bot_token or not telegram_chat_id:
            st.warning("Please configure Telegram Bot Token and Chat ID for notifications.")

    # Main Content
    st.subheader("Momentum Loss Screening Results with NSE Call Option Suggestions")
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
            st.session_state['config']['min_call_price_increase'],
            st.session_state['config']['strike_price_proximity'],
            st.session_state['config']['otm_strike_threshold'],
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
        st.write("### Stocks with Lost Momentum and Call Option Suggestions (Sortable & Searchable)")
        search_query = st.text_input("Search Results by Ticker", key="screening_search")
        screening_df = pd.DataFrame(st.session_state['screening_data'])
        if search_query:
            screening_df = screening_df[screening_df['Ticker'].str.contains(search_query, case=False, na=False)]

        # Debug: Print DataFrame columns
        print(f"screening_df columns: {screening_df.columns.tolist()}")

        # Flatten call suggestions for display
        display_data = []
        for _, row in screening_df.iterrows():
            drop_columns = [col for col in ['Call_Suggestions', 'OTM_Suggestion', 'Raw_Calls_Data'] if col in row]
            base_row = row.drop(drop_columns).to_dict() if drop_columns else row.to_dict()
            if 'Call_Suggestions' in row and row['Call_Suggestions']:
                for call in row['Call_Suggestions']:
                    display_row = base_row.copy()
                    display_row.update({
                        'Option_Strike_Price': call['Strike_Price'],
                        'Option_Last_Price': call['Last_Price'],
                        'Option_Price_Increase_Percent': call['Price_Increase_Percent'],
                        'Option_Expiration_Date': call['Expiration_Date'],
                        'Option_Contract_Symbol': call['Contract_Symbol'],
                        'Option_Volume': call['Volume'],
                        'Option_Open_Interest': call['Open_Interest'],
                        'Option_Type': 'General'
                    })
                    display_data.append(display_row)
            if 'OTM_Suggestion' in row and row['OTM_Suggestion']:
                display_row = base_row.copy()
                display_row.update({
                    'Option_Strike_Price': row['OTM_Suggestion']['Strike_Price'],
                    'Option_Last_Price': row['OTM_Suggestion']['Last_Price'],
                    'Option_Price_Increase_Percent': row['OTM_Suggestion']['Price_Increase_Percent'],
                    'Option_Expiration_Date': row['OTM_Suggestion']['Expiration_Date'],
                    'Option_Contract_Symbol': row['OTM_Suggestion']['Contract_Symbol'],
                    'Option_Volume': row['OTM_Suggestion']['Volume'],
                    'Option_Open_Interest': row['OTM_Suggestion']['Open_Interest'],
                    'Option_Type': 'OTM Recommendation'
                })
                display_data.append(display_row)
            if not ('Call_Suggestions' in row and row['Call_Suggestions']) and not ('OTM_Suggestion' in row and row['OTM_Suggestion']):
                display_row = base_row.copy()
                display_row.update({
                    'Option_Strike_Price': None,
                    'Option_Last_Price': None,
                    'Option_Price_Increase_Percent': None,
                    'Option_Expiration_Date': None,
                    'Option_Contract_Symbol': None,
                    'Option_Volume': None,
                    'Option_Open_Interest': None,
                    'Option_Type': None
                })
                display_data.append(display_row)
        display_df = pd.DataFrame(display_data)
        st.dataframe(
            display_df,
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker"),
                "Today_Close": st.column_config.NumberColumn("Today Close", format="%.2f"),
                "Yesterday_Close": st.column_config.NumberColumn("Yesterday Close", format="%.2f"),
                "Price_Drop_Percent": st.column_config.NumberColumn("Price Drop %", format="%.2f"),
                "Momentum_Gain_Percent": st.column_config.NumberColumn("Momentum Gain %", format="%.2f"),
                "Green_Candle_Count": st.column_config.NumberColumn("Green Candles"),
                "Momentum_Start_Date": st.column_config.TextColumn("Momentum Start"),
                "Momentum_End_Date": st.column_config.TextColumn("Momentum End"),
                "Last_Scanned": st.column_config.TextColumn("Last Scanned"),
                "Option_Strike_Price": st.column_config.NumberColumn("Strike Price", format="%.2f"),
                "Option_Last_Price": st.column_config.NumberColumn("Option Last Price", format="%.2f"),
                "Option_Price_Increase_Percent": st.column_config.NumberColumn("Option Price Increase %", format="%.2f"),
                "Option_Expiration_Date": st.column_config.TextColumn("Option Expiration"),
                "Option_Contract_Symbol": st.column_config.TextColumn("Contract Symbol"),
                "Option_Volume": st.column_config.NumberColumn("Option Volume"),
                "Option_Open_Interest": st.column_config.NumberColumn("Option Open Interest"),
                "Option_Type": st.column_config.TextColumn("Option Type")
            },
            use_container_width=True,
            height=400
        )

        # Debugging: Display raw options data
        st.write("### Debug: Raw Options Data")
        if not screening_df.empty:
            for _, row in screening_df.iterrows():
                ticker = row['Ticker']
                raw_calls = row.get('Raw_Calls_Data', {})
                if raw_calls:
                    st.write(f"**{ticker} Raw Options Data**")
                    raw_df = pd.DataFrame.from_dict(raw_calls)
                    st.dataframe(raw_df, use_container_width=True)
                else:
                    st.write(f"**{ticker}**: No raw options data available.")
        else:
            st.write("No raw options data available (empty screening results).")

        if st.button("Clear Screening Results"):
            st.session_state['screening_data'] = []
            save_screening_data(st.session_state['screening_data'])
            st.rerun()
    else:
        st.info("No results found. Run a scan to check for stocks with momentum loss and call option opportunities.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Script error: {str(e)}")
        st.error(f"Script error: {str(e)}")