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
import plotly.graph_objects as go
import requests
import threading

# Constants
STORED_TICKERS_PATH = "tickers.csv"
CONFIG_FILE = "config.json"
SCREENING_DATA_FILE = "screening_data.json"
BACKTEST_DATA_FILE = "backtest_data.json"
NOTIFIED_TICKERS_FILE = "notified_tickers.json"
api_call_counter = 0
SCREENING_INTERVAL_SECONDS = 600  # 10 minutes

# Cache for performance
cache = {}

# Global session for NSE requests
nse_session = None

# Headers for NSE requests
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/market-data/equity-derivatives-watch",
}

# Initialize NSE session
def initialize_nse_session():
    global nse_session
    if nse_session is None:
        nse_session = requests.Session()
        print("Visiting homepage...")
        response = nse_session.get("https://www.nseindia.com/", headers=headers)
        if response.status_code != 200:
            print(f"Failed to load homepage: {response.status_code}")
            return False
        print("Visiting derivatives page...")
        response = nse_session.get("https://www.nseindia.com/market-data/equity-derivatives-watch", headers=headers)
        time.sleep(5)
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
        "price_proximity_percent": 1.0
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

# Load/Save Screening, Backtest, and Notified Tickers Data
def load_screening_data() -> List[Dict]:
    if os.path.exists(SCREENING_DATA_FILE):
        with open(SCREENING_DATA_FILE, 'r') as f:
            return json.load(f)
    return []

def save_screening_data(data: List[Dict]):
    with open(SCREENING_DATA_FILE, 'w') as f:
        json.dump(data, f, indent=4)

def load_backtest_data() -> List[Dict]:
    if os.path.exists(BACKTEST_DATA_FILE):
        with open(BACKTEST_DATA_FILE, 'r') as f:
            return json.load(f)
    return []

def save_backtest_data(data: List[Dict]):
    with open(BACKTEST_DATA_FILE, 'w') as f:
        json.dump(data, f, indent=4)

def load_notified_tickers() -> Dict:
    if os.path.exists(NOTIFIED_TICKERS_FILE):
        with open(NOTIFIED_TICKERS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_notified_tickers(data: Dict):
    with open(NOTIFIED_TICKERS_FILE, 'w') as f:
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

# Fetch Current Price and Option Chain Data from NSE
def fetch_nse_data(ticker: str, refresh_key: float) -> Optional[Dict]:
    global nse_session, api_call_counter
    api_call_counter += 1
    cache_key = f"{ticker}_{refresh_key}"
    if cache_key in cache and (time.time() - cache.get(f"{cache_key}_timestamp", 0)) < 60:
        print(f"Using cached NSE data for {ticker}")
        return cache[cache_key]
    print(f"Fetching NSE data for API {api_call_counter}--{ticker}")
    try:
        if nse_session is None:
            print(f"No session available for {ticker}")
            return None
        option_chain_url = f"https://www.nseindia.com/api/option-chain-equities?symbol={ticker}"
        quote_url = f"https://www.nseindia.com/api/quote-equity?symbol={ticker}"
        response = nse_session.get(option_chain_url, headers=headers)
        print(f"Option chain response for {ticker}: Status {response.status_code}")
        if response.status_code != 200:
            print(f"Failed to load option chain for {ticker}: {response.status_code}, Response: {response.text}")
            return None
        data = response.json()
        quote_response = nse_session.get(quote_url, headers=headers)
        print(f"Quote response for {ticker}: Status {quote_response.status_code}")
        if quote_response.status_code == 200:
            quote_data = quote_response.json()
            last_price = quote_data.get('priceInfo', {}).get('lastPrice', 0)
            if last_price > 0 and 'records' in data:
                data['records']['underlyingValue'] = last_price
                print(f"Updated underlying value for {ticker}: {last_price}")
            else:
                print(f"No valid last price for {ticker}")
        cache[cache_key] = data
        cache[f"{cache_key}_timestamp"] = time.time()
        print(f"Successfully fetched NSE data for {ticker}")
        return data
    except Exception as e:
        print(f"Error fetching NSE data for {ticker}: {str(e)}")
        return None

# Process Option Chain Data
def process_option_data(data: Dict, expiry: Optional[str] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = data.get('filtered', {}).get('data', [])
    if not records:
        return pd.DataFrame(), pd.DataFrame()

    call_data = []
    put_data = []
    for record in records:
        strike = record.get('strikePrice', 0)
        expiry_date = record.get('expiryDate', '')
        if expiry and expiry_date != expiry:
            continue
        call_oi = record.get('CE', {}).get('openInterest', 0)
        call_price = record.get('CE', {}).get('lastPrice', 0)
        put_oi = record.get('PE', {}).get('openInterest', 0)
        call_data.append({'strikePrice': strike, 'callOI': call_oi, 'callPrice': call_price})
        put_data.append({'strikePrice': strike, 'putOI': put_oi})

    call_df = pd.DataFrame(call_data)
    put_df = pd.DataFrame(put_data)
    return call_df, put_df

# Identify Resistance Based on Option Chain OI
def identify_resistance(data: Dict, underlying_price: float) -> Optional[pd.DataFrame]:
    call_df, put_df = process_option_data(data)
    if call_df.empty or put_df.empty:
        return None

    combined_df = pd.DataFrame({
        'strikePrice': call_df['strikePrice'],
        'totalOI': call_df['callOI'] + put_df['putOI'],
        'callPrice': call_df['callPrice']
    })

    resistance_candidates = combined_df[combined_df['strikePrice'] > underlying_price]
    if resistance_candidates.empty:
        return None

    resistance_candidates = resistance_candidates.sort_values(by='totalOI', ascending=False)
    return resistance_candidates

# Fetch Historical Prices using yfinance
def fetch_historical_data(ticker: str, start_date: date, end_date: date, refresh_key: float) -> Optional[pd.DataFrame]:
    global api_call_counter
    api_call_counter += 1
    cache_key = f"{ticker}_{start_date}_{end_date}_{refresh_key}"
    if cache_key in cache and (time.time() - cache.get(f"{cache_key}_timestamp", 0)) < 3600:
        print(f"Using cached historical data for {ticker}")
        return cache[cache_key]
    print(f"Fetching historical data for API {api_call_counter}--{ticker}")
    try:
        stock = yf.Ticker(ticker + ".NS")
        hist = stock.history(start=start_date, end=end_date + timedelta(days=1))
        if hist.empty:
            print(f"No historical data found for {ticker}. Possible delisted or invalid ticker.")
            return None
        print(f"Fetched {len(hist)} days of historical data for {ticker}: {hist.index[0]} to {hist.index[-1]}")
        cache[cache_key] = hist
        cache[f"{cache_key}_timestamp"] = time.time()
        return hist
    except Exception as e:
        print(f"Error fetching historical data for {ticker}: {str(e)}")
        return None

# Generate Candlestick Chart for Option
def generate_option_candlestick(ticker: str, strike_price: float, start_date: date, end_date: date):
    hist = fetch_historical_data(ticker, start_date, end_date, time.time())
    if hist is None:
        return None
    fig = go.Figure(data=[go.Candlestick(
        x=hist.index,
        open=hist['Open'],
        high=hist['High'],
        low=hist['Low'],
        close=hist['Close'],
        name=f"{ticker} Stock (Proxy for {strike_price} CE)"
    )])
    fig.update_layout(
        title=f"Candlestick Chart for {ticker} {strike_price} CE (Proxy)",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False
    )
    return fig

# Check Momentum Loss and Suggest Resistance Strike
def check_momentum(ticker: str, hist: pd.DataFrame, current_price: float, min_gain_percent: float, min_green_candles: int, bot_token: str, chat_id: str, price_proximity_percent: float, notified_tickers: Dict, force_notify: bool = False) -> Optional[Dict]:
    print(f"\n=== Processing {ticker} ===")
    if hist.empty or len(hist) < min_green_candles + 2:
        print(f"{ticker}: Insufficient data (only {len(hist)} days)")
        return None
    try:
        yesterday = hist.index[-1]
        yesterday_close = hist.loc[yesterday, 'Close']
        print(f"{ticker}: Current price={current_price:.2f}, Yesterday close={yesterday_close:.2f}, Price drop={current_price < yesterday_close}")
    except KeyError:
        print(f"{ticker}: Missing data")
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
                print(f"{ticker}: Found {current_green_candles} green candles, Gain={gain_percent:.2f}%")
                if gain_percent >= min_gain_percent and current_green_candles >= max_green_candles:
                    max_green_candles = current_green_candles
                    max_gain = gain_percent
                    best_start_idx = current_start_idx
                    best_end_idx = current_end_idx
        else:
            current_green_candles = 0
            current_start_idx = None
    
    print(f"{ticker}: Max green candles={max_green_candles}, Max gain={max_gain:.2f}%")
    
    if max_gain < min_gain_percent or max_green_candles < min_green_candles or current_price >= yesterday_close:
        print(f"{ticker}: Failed criteria - Gain={max_gain:.2f}% (<{min_gain_percent}%), Green candles={max_green_candles} (<{min_green_candles}), Price drop={current_price < yesterday_close}")
        return None
    
    print(f"{ticker}: Momentum loss detected, fetching resistance")
    nse_data = fetch_nse_data(ticker, time.time())
    if nse_data is None:
        print(f"{ticker}: No option chain data available")
        return None
    
    resistance_df = identify_resistance(nse_data, current_price)
    if resistance_df is None or resistance_df.empty:
        print(f"{ticker}: No resistance strikes found")
        return None
    
    momentum_start_date = dates[best_start_idx].strftime("%Y-%m-%d")
    momentum_end_date = dates[best_end_idx].strftime("%Y-%m-%d")
    momentum_end = pd.to_datetime(momentum_end_date)
    hist.index = pd.to_datetime(hist.index)
    if hist.index.tz is not None:
        momentum_end = momentum_end.tz_localize(hist.index.tz)
    else:
        momentum_end = momentum_end.tz_localize(None)
    
    next_day = None
    yesterday_high = None
    for idx in hist.index:
        if idx > momentum_end:
            next_day = idx
            break
    if next_day is not None and next_day in hist.index:
        red_candle_open = hist.loc[next_day, 'Open']
        red_candle_high = hist.loc[next_day, 'High']
        yesterday_high = max(red_candle_open, red_candle_high)
    else:
        print(f"{ticker}: No data for day after momentum end, using last day")
        red_candle_open = hist.loc[yesterday, 'Open']
        red_candle_high = hist.loc[yesterday, 'High']
        yesterday_high = max(red_candle_open, red_candle_high)
    
    print(f"{ticker}: Red candle high/open={yesterday_high:.2f}")
    
    valid_resistances = resistance_df[resistance_df['strikePrice'] > current_price]
    if valid_resistances.empty:
        print(f"{ticker}: No resistance strikes above current price {current_price:.2f}")
        return None
    
    best_strike = valid_resistances.loc[valid_resistances['callPrice'].idxmax(), 'strikePrice']
    strike_price = float(best_strike)
    print(f"{ticker}: Selected strike={strike_price:.2f}")
    
    result = {
        "Ticker": ticker,
        "Current_Price": current_price,
        "Yesterday_Close": float(yesterday_close),
        "Price_Drop_Percent": float(((yesterday_close - current_price) / yesterday_close) * 100),
        "Momentum_Gain_Percent": float(max_gain),
        "Green_Candle_Count": int(max_green_candles),
        "Momentum_Start_Date": momentum_start_date,
        "Momentum_End_Date": momentum_end_date,
        "Strike_Price": float(strike_price),
        "Status": "Momentum Loss",
        "Last_Scanned": time.strftime("%Y-%m-%d %H:%M:%S"),
        "Red_Candle_High": float(yesterday_high)
    }
    
    ticker_key = f"{ticker}_{momentum_end_date}_{strike_price}"
    should_notify = force_notify or ticker_key not in notified_tickers
    
    if should_notify:
        message = (
            f"*Momentum Loss Alert*\n"
            f"Stock: *{ticker}*\n"
            f"Current Price: *₹{current_price:.2f}*\n"
            f"Yesterday Close: *₹{yesterday_close:.2f}*\n"
            f"Price Drop: *{result['Price_Drop_Percent']:.2f}%*\n"
            f"Prior Momentum Gain: *{max_gain:.2f}%*\n"
            f"Green Candles: *{max_green_candles}*\n"
            f"Momentum Period: *{momentum_start_date} to {momentum_end_date}*\n"
            f"Suggested Strike (Call Selling): *₹{strike_price:.2f}*\n"
            f"Timestamp: *{result['Last_Scanned']}*"
        )
        print(f"{ticker}: Sending Telegram notification for Momentum Loss")
        send_split_telegram_message(bot_token, chat_id, message)
        notified_tickers[ticker_key] = result["Last_Scanned"]
        save_notified_tickers(notified_tickers)
    
    if yesterday_high is not None and current_price >= yesterday_high * (1 - price_proximity_percent / 100):
        result["Status"] = "Momentum Loss Recovery"
        recovery_message = (
            f"*Momentum Loss Recovery Alert*\n"
            f"Stock: *{ticker}*\n"
            f"Current Price: *₹{current_price:.2f}*\n"
            f"Red Candle High: *₹{yesterday_high:.2f}*\n"
            f"Price Proximity: *{((yesterday_high - current_price) / yesterday_high * 100):.2f}%*\n"
            f"Prior Momentum Gain: *{max_gain:.2f}%*\n"
            f"Green Candles: *{max_green_candles}*\n"
            f"Momentum Period: *{momentum_start_date} to {momentum_end_date}*\n"
            f"Suggested Strike (Call Selling): *₹{strike_price:.2f}*\n"
            f"Timestamp: *{result['Last_Scanned']}*\n"
            f"Action: *Consider selling call at suggested strike*"
        )
        print(f"{ticker}: Sending Telegram notification for Momentum Loss Recovery")
        send_split_telegram_message(bot_token, chat_id, recovery_message)
        # Always notify for recovery in real-time, so we don't update notified_tickers here
    
    return result

# Screen Tickers
def screen_tickers(tickers: List[str], min_gain_percent: float, min_green_candles: int, lookback_days: int, bot_token: str, chat_id: str, refresh_key: float, price_proximity_percent: float, force_notify: bool = False) -> List[Dict]:
    global nse_session
    end_date = datetime.now().date()  # Use current date dynamically
    start_date = end_date - timedelta(days=lookback_days)
    results = []
    notified_tickers = load_notified_tickers()
    if not initialize_nse_session():
        st.error("Failed to initialize NSE session. Please try again.")
        return results
    for ticker in tickers:
        with st.spinner(f"Screening {ticker}..."):
            hist = fetch_historical_data(ticker, start_date, end_date, refresh_key)
            if hist is None:
                st.warning(f"No historical data for {ticker}.")
                continue
            nse_data = fetch_nse_data(ticker, refresh_key)
            if nse_data is None or 'records' not in nse_data:
                st.warning(f"No NSE data for {ticker}.")
                continue
            current_price = nse_data['records'].get('underlyingValue', None)
            if current_price is None:
                st.warning(f"No current price for {ticker}.")
                continue
            result = check_momentum(ticker, hist, current_price, min_gain_percent, min_green_candles, bot_token, chat_id, price_proximity_percent, notified_tickers, force_notify)
            if result:
                results.append(result)
    nse_session = None
    return results

# Backtest Momentum
def backtest_momentum(ticker: str, start_date: date, end_date: date, min_gain_percent: float, min_green_candles: int) -> List[Dict]:
    hist = fetch_historical_data(ticker, start_date, end_date, time.time())
    if hist is None:
        return []
    results = []
    for i in range(len(hist) - min_green_candles - 1):
        window = hist.iloc[:i + min_green_candles + 1]
        current_date = window.index[-1]
        current_close = window['Close'][-1]
        prev_close = window['Close'][-2]
        momentum_high = window['High'].max()
        strike_price = round(momentum_high / 50) * 50
        closes = window['Close'].values
        dates = window.index
        max_green_candles = 0
        max_gain = 0
        best_start_idx = None
        best_end_idx = None
        current_green_candles = 0
        current_start_idx = None
        for j in range(1, len(closes)):
            if closes[j] > closes[j-1]:
                current_green_candles += 1
                if current_green_candles == 1:
                    current_start_idx = j - 1
                current_end_idx = j
                if current_green_candles >= min_green_candles:
                    start_price = closes[current_start_idx]
                    end_price = closes[current_end_idx]
                    gain_percent = ((end_price - start_price) / start_price) * 100
                    if gain_percent >= min_gain_percent and current_green_candles >= max_green_candles:
                        max_green_candles = current_green_candles
                        max_gain = gain_percent
                        best_start_idx = current_start_idx
                        best_end_idx = current_end_idx
            else:
                current_green_candles = 0
                current_start_idx = None
        if max_gain >= min_gain_percent and max_green_candles >= min_green_candles and current_close < prev_close:
            momentum_start_date = dates[best_start_idx].strftime("%Y-%m-%d")
            momentum_end_date = dates[best_end_idx].strftime("%Y-%m-%d")
            result = {
                "Ticker": ticker,
                "Date": current_date.strftime("%Y-%m-%d"),
                "Current_Close": float(current_close),
                "Previous_Close": float(prev_close),
                "Price_Change_Percent": float(((current_close - prev_close) / prev_close) * 100),
                "Momentum_Gain_Percent": float(max_gain),
                "Green_Candle_Count": int(max_green_candles),
                "Momentum_Start_Date": momentum_start_date,
                "Momentum_End_Date": momentum_end_date,
                "Strike_Price": float(strike_price),
                "Status": "Momentum Loss"
            }
            results.append(result)
    return results

# Schedule Periodic Screening
def schedule_screening(tickers, min_gain_percent, min_green_candles, lookback_days, bot_token, chat_id, refresh_key, price_proximity_percent, force_notify):
    if 'screening_scheduled' not in st.session_state:
        st.session_state['screening_scheduled'] = False
    if st.session_state['screening_scheduled']:
        return
    st.session_state['screening_scheduled'] = True

    def run_screening():
        print("Running scheduled screening...")
        st.session_state['refresh_key'] = time.time()
        screening_data = screen_tickers(
            tickers,
            min_gain_percent,
            min_green_candles,
            lookback_days,
            bot_token,
            chat_id,
            st.session_state['refresh_key'],
            price_proximity_percent,
            force_notify
        )
        if screening_data:
            st.session_state['screening_data'] = screening_data
            save_screening_data(screening_data)
            st.session_state['last_scan_time'] = time.time()
            st.rerun()  # Updated from st.experimental_rerun()
        st.session_state['scan_in_progress'] = False
        # Schedule the next run
        threading.Timer(SCREENING_INTERVAL_SECONDS, run_screening).start()

    if not st.session_state['scan_in_progress']:
        threading.Timer(SCREENING_INTERVAL_SECONDS, run_screening).start()

# Main Application
def main():
    st.set_page_config(page_title="Momentum Loss Screener", layout="wide")
    tabs = st.tabs(["Real-Time Screener"])

    config = load_config()
    if 'config' not in st.session_state:
        st.session_state['config'] = config

    if 'screening_data' not in st.session_state:
        st.session_state['screening_data'] = load_screening_data()
    if 'backtest_data' not in st.session_state:
        st.session_state['backtest_data'] = load_backtest_data()
    if 'refresh_key' not in st.session_state:
        st.session_state['refresh_key'] = time.time()
    if 'last_scan_time' not in st.session_state:
        st.session_state['last_scan_time'] = time.time()
    if 'scan_in_progress' not in st.session_state:
        st.session_state['scan_in_progress'] = False

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
            min_value=4,
            step=1,
            key="lookback_days"
        )
        price_proximity_percent = st.number_input(
            "Price Proximity for Recovery (%):",
            value=st.session_state['config']['price_proximity_percent'],
            min_value=0.1,
            step=0.1,
            key="price_proximity_percent"
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
        specific_tickers = st.text_input("Enter tickers (comma-separated):", key="specific_tickers")

        fresh_scan = st.checkbox("Force Fresh Scan with Notifications for All Tickers", value=False, key="fresh_scan")

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
        if st.session_state['config']['price_proximity_percent'] != price_proximity_percent:
            st.session_state['config']['price_proximity_percent'] = price_proximity_percent
            config_changed = True
        if config_changed:
            save_config(st.session_state['config'])

        if not telegram_bot_token or not telegram_chat_id:
            st.warning("Configure Telegram Bot Token and Chat ID for notifications.")

    with tabs[0]:
        st.title("Real-Time Momentum Loss Screener")
        st.subheader("Momentum Loss Results")
        st.write(f"Last Scan: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(st.session_state['last_scan_time']))}")
        if st.button("Refresh Results"):
            st.rerun()

        def perform_screening(tickers_to_screen, force_notify):
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
                st.session_state['refresh_key'],
                st.session_state['config']['price_proximity_percent'],
                force_notify
            )
            st.session_state['screening_data'] = screening_data
            save_screening_data(screening_data)
            st.session_state['last_scan_time'] = time.time()
            st.session_state['scan_in_progress'] = False
            if not screening_data:
                st.info("No stocks met the momentum loss criteria.")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Screen All Tickers") and not st.session_state['scan_in_progress']:
                tickers = load_tickers()
                perform_screening(tickers, fresh_scan)
                st.rerun()
        with col2:
            if st.button("Screen Specific Tickers") and specific_tickers and not st.session_state['scan_in_progress']:
                tickers = [t.strip() for t in specific_tickers.split(',')]
                perform_screening(tickers, fresh_scan)
                st.rerun()

        # Schedule periodic screening
        tickers = load_tickers()
        schedule_screening(
            tickers,
            st.session_state['config']['min_gain_percent'],
            st.session_state['config']['min_green_candles'],
            st.session_state['config']['lookback_days'],
            st.session_state['config']['telegram_bot_token'],
            st.session_state['config']['telegram_chat_id'],
            st.session_state['refresh_key'],
            st.session_state['config']['price_proximity_percent'],
            fresh_scan
        )

        if st.session_state['screening_data']:
            st.write("### Stocks with Momentum Loss")
            screening_df = pd.DataFrame(st.session_state['screening_data'])
            momentum_loss_df = screening_df[screening_df['Status'] == "Momentum Loss"]
            recovery_df = screening_df[screening_df['Status'] == "Momentum Loss Recovery"]

            if not momentum_loss_df.empty:
                st.write("#### Momentum Loss")
                search_query = st.text_input("Search Momentum Loss Results by Ticker", key="momentum_loss_search")
                display_df = momentum_loss_df
                if search_query:
                    display_df = momentum_loss_df[momentum_loss_df['Ticker'].str.contains(search_query, case=False, na=False)]
                st.dataframe(
                    display_df,
                    column_config={
                        "Ticker": st.column_config.TextColumn("Ticker"),
                        "Current_Price": st.column_config.NumberColumn("Current Price", format="%.2f"),
                        "Yesterday_Close": st.column_config.NumberColumn("Yesterday Close", format="%.2f"),
                        "Price_Drop_Percent": st.column_config.NumberColumn("Price Drop %", format="%.2f"),
                        "Momentum_Gain_Percent": st.column_config.NumberColumn("Momentum Gain %", format="%.2f"),
                        "Green_Candle_Count": st.column_config.NumberColumn("Green Candles"),
                        "Momentum_Start_Date": st.column_config.TextColumn("Momentum Start"),
                        "Momentum_End_Date": st.column_config.TextColumn("Momentum End"),
                        "Strike_Price": st.column_config.NumberColumn("Suggested Strike", format="%.2f"),
                        "Status": st.column_config.TextColumn("Status"),
                        "Last_Scanned": st.column_config.TextColumn("Last Scanned"),
                        "Red_Candle_High": st.column_config.NumberColumn("Red Candle High", format="%.2f")
                    },
                    use_container_width=True,
                    height=400
                )
            else:
                st.info("No momentum loss results found.")

            if not recovery_df.empty:
                st.write("#### Momentum Loss Recovery")
                search_recovery_query = st.text_input("Search Recovery Results by Ticker", key="recovery_search")
                display_recovery_df = recovery_df
                if search_recovery_query:
                    display_recovery_df = recovery_df[recovery_df['Ticker'].str.contains(search_recovery_query, case=False, na=False)]
                st.dataframe(
                    display_recovery_df,
                    column_config={
                        "Ticker": st.column_config.TextColumn("Ticker"),
                        "Current_Price": st.column_config.NumberColumn("Current Price", format="%.2f"),
                        "Yesterday_Close": st.column_config.NumberColumn("Yesterday Close", format="%.2f"),
                        "Price_Drop_Percent": st.column_config.NumberColumn("Price Drop %", format="%.2f"),
                        "Momentum_Gain_Percent": st.column_config.NumberColumn("Momentum Gain %", format="%.2f"),
                        "Green_Candle_Count": st.column_config.NumberColumn("Green Candles"),
                        "Momentum_Start_Date": st.column_config.TextColumn("Momentum Start"),
                        "Momentum_End_Date": st.column_config.TextColumn("Momentum End"),
                        "Strike_Price": st.column_config.NumberColumn("Suggested Strike", format="%.2f"),
                        "Status": st.column_config.TextColumn("Status"),
                        "Last_Scanned": st.column_config.TextColumn("Last Scanned"),
                        "Red_Candle_High": st.column_config.NumberColumn("Red Candle High", format="%.2f")
                    },
                    use_container_width=True,
                    height=400
                )
            else:
                st.info("No momentum loss recovery results found.")

            selected_ticker = st.selectbox("Select Ticker for Candlestick Chart", screening_df['Ticker'].unique())
            if 'Strike_Price' in screening_df.columns and not screening_df[screening_df['Ticker'] == selected_ticker].empty:
                strike_prices = screening_df[screening_df['Ticker'] == selected_ticker]['Strike_Price'].unique()
                if strike_prices.size > 0:
                    selected_strike = st.selectbox("Select Strike Price", strike_prices)
                    chart = generate_option_candlestick(selected_ticker, selected_strike, datetime.now().date() - timedelta(days=30), datetime.now().date())
                    if chart:
                        st.plotly_chart(chart, use_container_width=True)
                    else:
                        st.warning(f"No chart data available for {selected_ticker} at strike {selected_strike}.")
                else:
                    st.warning(f"No strike prices available for {selected_ticker}.")
            else:
                st.warning(f"No data available for {selected_ticker}.")
            if st.button("Clear Screening Results"):
                st.session_state['screening_data'] = []
                save_screening_data(st.session_state['screening_data'])
                if os.path.exists(NOTIFIED_TICKERS_FILE):
                    os.remove(NOTIFIED_TICKERS_FILE)
                st.rerun()
        else:
            st.info("No results found. Run a scan to check for stocks.")
    
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Script error: {str(e)}")
        st.error(f"Script error: {str(e)}")