import streamlit as st
import requests
import pandas as pd
import time
import json
import os
import cloudscraper
import aiohttp
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date
import yfinance as yf
import numpy as np

# Create a cloudscraper session
scraper = cloudscraper.create_scraper()

# Constants
BASE_URL = "https://www.nseindia.com"
STORED_TICKERS_PATH = "tickers.csv"
CONFIG_FILE = "config.json"
TEMP_TABLE_DATA_FILE = "temp_table_data.json"
ALERTS_DATA_FILE = "alerts_data.json"
HISTORICAL_DATA_FILE = "historical_data.json"
CALL_SUGGESTIONS_FILE = "call_suggestions.json"

# Headers mimicking your browser
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/market-data/equity-derivatives-watch",
}

# Initial cookie setup
print("Visiting homepage...")
response = scraper.get("https://www.nseindia.com/", headers=headers)
if response.status_code != 200:
    print(f"Failed to load homepage: {response.status_code}")
    exit()

print("Visiting derivatives page...")
scraper.get("https://www.nseindia.com/market-data/equity-derivatives-watch", headers=headers)
time.sleep(2)

# Cache for performance
cache = {}

# Load/Save Telegram Config
def load_config() -> Dict:
    default_config = {
        "telegram_bot_token": "",
        "telegram_chat_id": "",
        "auto_scan_interval": 5,
        "proximity_to_resistance": 0.5,
        "auto_running": False
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

# Load/Save Table Data to JSON
def load_table_data() -> List[Dict]:
    if os.path.exists(TEMP_TABLE_DATA_FILE):
        with open(TEMP_TABLE_DATA_FILE, 'r') as f:
            return json.load(f)
    return []

def save_table_data(data: List[Dict]):
    with open(TEMP_TABLE_DATA_FILE, 'w') as f:
        json.dump(data, f, indent=4)

# Load/Save Alerts Data to JSON
def load_alerts_data() -> List[Dict]:
    if os.path.exists(ALERTS_DATA_FILE):
        with open(ALERTS_DATA_FILE, 'r') as f:
            return json.load(f)
    return []

def save_alerts_data(data: List[Dict]):
    with open(ALERTS_DATA_FILE, 'w') as f:
        json.dump(data, f, indent=4)

# Load/Save Historical Data to JSON
def load_historical_data() -> List[Dict]:
    if os.path.exists(HISTORICAL_DATA_FILE):
        with open(HISTORICAL_DATA_FILE, 'r') as f:
            data = json.load(f)
            print(f"Loaded {len(data)} historical entries from {HISTORICAL_DATA_FILE}")
            return data
    print(f"No historical data file found at {HISTORICAL_DATA_FILE}")
    return []

def save_historical_data(data: List[Dict]):
    with open(HISTORICAL_DATA_FILE, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Saved {len(data)} historical entries to {HISTORICAL_DATA_FILE}")

# Load/Save Call Suggestions to JSON
def load_call_suggestions() -> List[Dict]:
    if os.path.exists(CALL_SUGGESTIONS_FILE):
        with open(CALL_SUGGESTIONS_FILE, 'r') as f:
            return json.load(f)
    return []

def save_call_suggestions(data: List[Dict]):
    with open(CALL_SUGGESTIONS_FILE, 'w') as f:
        json.dump(data, f, indent=4)

# Telegram Integration
async def send_telegram_message(bot_token: str, chat_id: str, message: str):
    if not bot_token or not chat_id:
        print("Telegram Bot Token or Chat ID is missing.")
        return
    
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                print(f"Failed to send Telegram message: {await response.text()}")
            else:
                print(f"Telegram message sent: {message}")

# Fetch Options Data with Last Price as Underlying
def fetch_options_data(symbol: str, refresh_key: float) -> Optional[Dict]:
    cache_key = f"{symbol}_{refresh_key}"
    if cache_key in cache and (time.time() - cache.get(f"{cache_key}_timestamp", 0)) < 300:  # Cache for 5 minutes
        return cache[cache_key]

    url = f"{BASE_URL}/api/option-chain-equities?symbol={symbol}"
    quote_url = f"https://www.nseindia.com/api/quote-equity?symbol={symbol}"
    print(f"Fetching data for {symbol} from: {url} and {quote_url}")
    
    response = scraper.get(url, headers=headers)
    if response.status_code != 200:
        print(f"Failed to load options chain for {symbol}: {response.status_code}")
        return None
    
    data = response.json()
    
    quote_response = scraper.get(quote_url, headers=headers)
    if quote_response.status_code == 200:
        quote_data = quote_response.json()
        last_price = quote_data.get('priceInfo', {}).get('lastPrice', 0)
        if last_price > 0 and 'records' in data:
            data['records']['underlyingValue'] = last_price
        else:
            print(f"No valid last price found for {symbol}, using default underlying.")
    else:
        print(f"Failed to load quote data for {symbol}: {quote_response.status_code}")
    
    cache[cache_key] = data
    cache[f"{cache_key}_timestamp"] = time.time()
    return data

# Process Option Data
def process_option_data(data: Dict, expiry: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not data or 'records' not in data or 'data' not in data['records']:
        print("Invalid data")
        return pd.DataFrame(), pd.DataFrame()
    
    options = [item for item in data['records']['data'] if item.get('expiryDate') == expiry]
    if not options:
        print(f"No options found for expiry {expiry}")
        return pd.DataFrame(), pd.DataFrame()
    
    strikes = sorted({item['strikePrice'] for item in options})
    call_data = {s: {'OI': 0, 'Volume': 0, 'Last Price': 0} for s in strikes}
    put_data = {s: {'OI': 0, 'Volume': 0} for s in strikes}
    
    for item in options:
        strike = item['strikePrice']
        if 'CE' in item:
            call_data[strike] = {
                'OI': item['CE']['openInterest'],
                'Volume': item['CE']['totalTradedVolume'],
                'Last Price': item['CE'].get('lastPrice', 0)
            }
        if 'PE' in item:
            put_data[strike] = {'OI': item['PE']['openInterest'], 'Volume': item['PE']['totalTradedVolume']}
    
    call_df = pd.DataFrame([{'Strike': k, **v} for k, v in call_data.items()])
    put_df = pd.DataFrame([{'Strike': k, **v} for k, v in put_data.items()])
    
    return call_df, put_df

# Suggest Call Options
def suggest_call_options(data: Dict, expiry: str, underlying: float, ticker: str) -> Optional[Dict]:
    if not data or 'records' not in data or 'data' not in data['records']:
        print(f"No valid data for {ticker} to suggest call options")
        return None
    
    call_df, _ = process_option_data(data, expiry)
    if call_df.empty or underlying <= 0:
        print(f"No call data or invalid underlying for {ticker}")
        return None
    
    min_otm = underlying * 1.05  # 5% OTM
    max_otm = underlying * 1.10  # 10% OTM
    slightly_otm_calls = call_df[(call_df['Strike'] >= min_otm) & (call_df['Strike'] <= max_otm)].sort_values('Last Price', ascending=False)
    
    if slightly_otm_calls.empty:
        print(f"No slightly OTM calls found for {ticker}, falling back to most OTM")
        most_otm_calls = call_df[call_df['Strike'] > underlying].sort_values('Strike', ascending=False)
        if most_otm_calls.empty:
            return None
        most_otm_strike = most_otm_calls.iloc[0]
        last_price = most_otm_strike['Last Price']
        potential_gain_percent = ((most_otm_strike['Strike'] - underlying) / last_price * 100) if last_price > 0 else 0
    else:
        best_strike = slightly_otm_calls.iloc[0]
        last_price = best_strike['Last Price']
        potential_gain_percent = ((best_strike['Strike'] - underlying) / last_price * 100) if last_price > 0 else 0
    
    return {
        "Ticker": ticker,
        "Underlying": underlying,
        "Suggested_Call_Strike": best_strike['Strike'] if not slightly_otm_calls.empty else most_otm_strike['Strike'],
        "Call_Last_Price": last_price,
        "Potential_Gain_%": potential_gain_percent,
        "Expiry": expiry,
        "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

# Identify Support and Resistance
def identify_support_resistance(call_df: pd.DataFrame, put_df: pd.DataFrame, top_n: int = 3) -> Tuple[Optional[float], Optional[float]]:
    resistance_strike = None
    support_strike = None
    
    if not call_df.empty and call_df['OI'].sum() > 0 and call_df['Volume'].sum() > 0:
        call_df['Weighted_Score'] = call_df['OI'] * call_df['Volume']
        top_calls = call_df.nlargest(top_n, 'Weighted_Score')
        resistance_strike = top_calls['Strike'].mean()
    
    if not put_df.empty and put_df['OI'].sum() > 0 and put_df['Volume'].sum() > 0:
        put_df['Weighted_Score'] = put_df['OI'] * put_df['Volume']
        top_puts = put_df.nlargest(top_n, 'Weighted_Score')
        support_strike = top_puts['Strike'].mean()
    
    return support_strike, resistance_strike

# Load Tickers
def load_tickers() -> List[str]:
    try:
        if os.path.exists(STORED_TICKERS_PATH):
            df = pd.read_csv(STORED_TICKERS_PATH)
            if 'SYMBOL' not in df.columns:
                st.error("Stored CSV file must contain 'SYMBOL' column")
                return ["HDFCBANK"]
            return df['SYMBOL'].dropna().tolist()
        return ["HDFCBANK"]
    except Exception as e:
        st.error(f"Error loading tickers: {e}")
        return ["HDFCBANK"]

# Process Ticker Group for Parallel Execution
def process_ticker_group(ticker_group: List[str], expiry: str, bot_token: str, chat_id: str, proximity_percent: float, refresh_key: float) -> Tuple[List[Dict], List[Dict]]:
    suggestions = []
    call_suggestions = []

    for ticker in ticker_group:
        data = fetch_options_data(ticker, refresh_key)
        if not data or 'records' not in data:
            print(f"Failed to fetch data for {ticker}")
            continue

        call_df, put_df = process_option_data(data, expiry)
        underlying = data['records'].get('underlyingValue', 0)
        resistance_strike = identify_support_resistance(call_df, put_df)[1]

        if resistance_strike is None:
            continue

        proximity_threshold = abs(proximity_percent) / 100.0

        if proximity_percent >= 0:
            distance_to_resistance = resistance_strike - underlying
            if 0 <= distance_to_resistance <= (resistance_strike * proximity_threshold):
                message = (
                    f"*Resistance Alert*\n"
                    f"Stock: *{ticker}*\n"
                    f"Underlying: *₹{underlying:.2f}*\n"
                    f"Resistance: *₹{resistance_strike:.2f}*\n"
                    f"Distance: *{distance_to_resistance:.2f}*\n"
                    f"Reason: *Within {proximity_percent}% of strong resistance*"
                )
                asyncio.run(send_telegram_message(bot_token, chat_id, message))
                suggestions.append({
                    "Ticker": ticker,
                    "Underlying": underlying,
                    "Resistance": resistance_strike,
                    "Distance_to_Resistance": distance_to_resistance,
                    "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                })
        else:
            distance_to_resistance = underlying - resistance_strike
            if distance_to_resistance > 0 and distance_to_resistance <= (resistance_strike * abs(proximity_threshold)):
                message = (
                    f"*Resistance Crossed Alert*\n"
                    f"Stock: *{ticker}*\n"
                    f"Underlying: *₹{underlying:.2f}*\n"
                    f"Resistance: *₹{resistance_strike:.2f}*\n"
                    f"Crossed By: *{distance_to_resistance:.2f}*\n"
                    f"Reason: *Crossed resistance by {abs(proximity_percent)}%*"
                )
                asyncio.run(send_telegram_message(bot_token, chat_id, message))
                suggestions.append({
                    "Ticker": ticker,
                    "Underlying": underlying,
                    "Resistance": resistance_strike,
                    "Distance_to_Resistance": -distance_to_resistance,
                    "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                })

        if (proximity_percent >= 0 and 0 <= (resistance_strike - underlying) <= (resistance_strike * proximity_threshold)) or \
           (proximity_percent < 0 and (underlying - resistance_strike) > 0 and (underlying - resistance_strike) <= (resistance_strike * abs(proximity_threshold))):
            call_suggestion = suggest_call_options(data, expiry, underlying, ticker)
            if call_suggestion and call_suggestion['Potential_Gain_%'] > 0:
                call_message = (
                    f"*Call Option Suggestion (Most OTM)*\n"
                    f"Stock: *{ticker}*\n"
                    f"Underlying: *₹{underlying:.2f}*\n"
                    f"Suggested Call Strike: *₹{call_suggestion['Suggested_Call_Strike']:.2f}* (Most OTM)\n"
                    f"Call Last Price: *₹{call_suggestion['Call_Last_Price']:.2f}*\n"
                    f"Potential Gain: *{call_suggestion['Potential_Gain_%']:.2f}%*\n"
                    f"Expiry: *{expiry}*\n"
                    f"Timestamp: *{call_suggestion['Timestamp']}*"
                )
                asyncio.run(send_telegram_message(bot_token, chat_id, call_message))
                call_suggestions.append(call_suggestion)

    return suggestions, call_suggestions

# Perform Parallel Scan
def perform_parallel_scan(tickers: List[str], expiry: str, bot_token: str, chat_id: str, proximity_percent: float) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    refresh_key = time.time()  # Generate refresh_key here
    num_groups = 4
    group_size = max(1, len(tickers) // num_groups)
    ticker_groups = [tickers[i:i + group_size] for i in range(0, len(tickers), group_size)]

    all_suggestions = []
    all_call_suggestions = []
    scanned_stocks = []

    with ThreadPoolExecutor(max_workers=num_groups) as executor:
        future_to_group = {executor.submit(process_ticker_group, group, expiry, bot_token, chat_id, proximity_percent, refresh_key): group for group in ticker_groups}

        for future in future_to_group:
            try:
                suggestions, call_suggestions = future.result()
                all_suggestions.extend(suggestions)
                all_call_suggestions.extend(call_suggestions)
            except Exception as e:
                print(f"Error processing group: {e}")

    # Collect scanned stocks (post-processing)
    for ticker in tickers:
        data = fetch_options_data(ticker, refresh_key)
        if data and 'records' in data:
            call_df, put_df = process_option_data(data, expiry)
            underlying = data['records'].get('underlyingValue', 0)
            support_strike, resistance_strike = identify_support_resistance(call_df, put_df)

            distance_from_resistance = resistance_strike - underlying if resistance_strike else None
            distance_from_support = underlying - support_strike if support_strike else None
            distance_percent_from_resistance = (distance_from_resistance / resistance_strike * 100) if resistance_strike and distance_from_resistance is not None else None
            distance_percent_from_support = (distance_from_support / support_strike * 100) if support_strike and distance_from_support is not None else None

            total_volume = call_df['Volume'].sum() + put_df['Volume'].sum()
            high_volume_gainer = "Yes" if total_volume > 100000 else "No"

            scanned_stocks.append({
                "Ticker": ticker,
                "Underlying": underlying,
                "Resistance": resistance_strike,
                "Support": support_strike,
                "Distance_from_Resistance": distance_from_resistance,
                "Distance_from_Support": distance_from_support,
                "Distance_%_from_Resistance": distance_percent_from_resistance,
                "Distance_%_from_Support": distance_percent_from_support,
                "High_Volume_Gainer": high_volume_gainer,
                "Last_Scanned": time.strftime("%Y-%m-%d %H:%M:%S")
            })

    return all_suggestions, all_call_suggestions, scanned_stocks

# Perform Scan
def perform_scan(tickers_to_scan, expiry, telegram_bot_token, telegram_chat_id, proximity_percent):
    suggestions, call_suggestions, scanned_stocks = perform_parallel_scan(
        tickers_to_scan, expiry, telegram_bot_token, telegram_chat_id, proximity_percent
    )

    st.session_state['suggestions'].extend(suggestions)
    st.session_state['call_suggestions'].extend(call_suggestions)
    st.session_state['scanned_stocks'] = scanned_stocks

    save_alerts_data(st.session_state['suggestions'])
    save_call_suggestions(st.session_state['call_suggestions'])

# Generate Support/Resistance Table Data and Save to JSON
def generate_support_resistance_table(tickers: List[str], expiry: str) -> List[Dict]:
    refresh_key = time.time()
    table_data = []
    volume_threshold = 100000
    
    for ticker in tickers:
        data = fetch_options_data(ticker, refresh_key)
        if not data or 'records' not in data:
            print(f"Failed to fetch data for {ticker}")
            continue
        
        call_df, put_df = process_option_data(data, expiry)
        underlying = data['records'].get('underlyingValue', 0)
        support_strike, resistance_strike = identify_support_resistance(call_df, put_df)
        
        total_volume = call_df['Volume'].sum() + put_df['Volume'].sum()
        high_volume_gainer = "Yes" if total_volume > volume_threshold else "No"
        
        distance_from_resistance = resistance_strike - underlying if resistance_strike else None
        distance_from_support = underlying - support_strike if support_strike else None
        distance_percent_from_resistance = (distance_from_resistance / resistance_strike * 100) if resistance_strike and distance_from_resistance is not None else None
        distance_percent_from_support = (distance_from_support / support_strike * 100) if support_strike and distance_from_support is not None else None
        
        table_data.append({
            "Ticker": ticker,
            "Underlying": underlying,
            "Support": support_strike,
            "Resistance": resistance_strike,
            "Distance_from_Resistance": distance_from_resistance,
            "Distance_from_Support": distance_from_support,
            "Distance_%_from_Resistance": distance_percent_from_resistance,
            "Distance_%_from_Support": distance_percent_from_support,
            "High_Volume_Gainer": high_volume_gainer,
            "Last_Updated": time.strftime("%Y-%m-%d %H:%M:%S")
        })
    
    save_table_data(table_data)
    return table_data

# Main Application
def main():
    st.set_page_config(page_title="Resistance Screener", layout="wide")
    st.title("Real-Time Resistance Screener")

    # Load Telegram Config
    config = load_config()
    if 'telegram_config' not in st.session_state:
        st.session_state['telegram_config'] = config

    # Initialize Session State
    if 'last_scan_time' not in st.session_state:
        st.session_state['last_scan_time'] = time.time()
    if 'refresh_key' not in st.session_state:
        st.session_state['refresh_key'] = time.time()
    if 'suggestions' not in st.session_state:
        st.session_state['suggestions'] = load_alerts_data()
    if 'call_suggestions' not in st.session_state:
        st.session_state['call_suggestions'] = load_call_suggestions()
    if 'table_data' not in st.session_state:
        st.session_state['table_data'] = load_table_data()
    if 'historical_data' not in st.session_state:
        st.session_state['historical_data'] = load_historical_data()
    if 'auto_scan_triggered' not in st.session_state:
        st.session_state['auto_scan_triggered'] = False
    if 'scanned_stocks' not in st.session_state:
        st.session_state['scanned_stocks'] = []
    if 'auto_running' not in st.session_state:
        st.session_state['auto_running'] = st.session_state['telegram_config']['auto_running']
    if 'last_auto_time' not in st.session_state:
        st.session_state['last_auto_time'] = 0

    # Sidebar Configuration
    with st.sidebar:
        st.subheader("Auto Scan Control")
        if st.button("Toggle Auto (60s)", key="sidebar_auto_toggle"):
            st.session_state['auto_running'] = not st.session_state['auto_running']
            st.session_state['telegram_config']['auto_running'] = st.session_state['auto_running']
            save_config(st.session_state['telegram_config'])
            if st.session_state['auto_running']:
                st.session_state['last_auto_time'] = time.time()
            st.rerun()
        status = "Running" if st.session_state['auto_running'] else "Stopped"
        st.write(f"Auto Scan (60s): {status}")

        st.subheader("Telegram Integration")
        telegram_bot_token = st.text_input(
            "Telegram Bot Token:",
            value=st.session_state['telegram_config']['telegram_bot_token'],
            type="password",
            key="telegram_bot_token"
        )
        telegram_chat_id = st.text_input(
            "Telegram Chat ID:",
            value=st.session_state['telegram_config']['telegram_chat_id'],
            key="telegram_chat_id"
        )
        
        st.subheader("Scan Settings")
        auto_scan_interval = st.number_input(
            "Auto-Scan Interval (minutes):",
            value=st.session_state['telegram_config']['auto_scan_interval'],
            min_value=1, step=1, key="auto_scan_interval"
        )
        proximity_to_resistance = st.number_input(
            "Proximity to Resistance (%):",
            value=st.session_state['telegram_config']['proximity_to_resistance'],
            step=0.1,
            min_value=-5.0,
            max_value=5.0,
            key="proximity_to_resistance_input"
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

        st.subheader("Scan Specific Stocks")
        specific_tickers = st.text_input("Enter tickers (comma-separated, e.g., HDFCBANK,RELIANCE):", key="specific_tickers")

        # Update config if changed
        config_changed = False
        if st.session_state['telegram_config']['telegram_bot_token'] != telegram_bot_token:
            st.session_state['telegram_config']['telegram_bot_token'] = telegram_bot_token
            config_changed = True
        if st.session_state['telegram_config']['telegram_chat_id'] != telegram_chat_id:
            st.session_state['telegram_config']['telegram_chat_id'] = telegram_chat_id
            config_changed = True
        if st.session_state['telegram_config']['auto_scan_interval'] != auto_scan_interval:
            st.session_state['telegram_config']['auto_scan_interval'] = auto_scan_interval
            config_changed = True
        if st.session_state['telegram_config']['proximity_to_resistance'] != proximity_to_resistance:
            st.session_state['telegram_config']['proximity_to_resistance'] = proximity_to_resistance
            config_changed = True
        if config_changed:
            save_config(st.session_state['telegram_config'])

        if not telegram_bot_token or not telegram_chat_id:
            st.warning("Please configure Telegram Bot Token and Chat ID.")

    # Fetch initial data to get expiry
    data = fetch_options_data("HDFCBANK", st.session_state['refresh_key'])
    if not data or 'records' not in data:
        st.error("Failed to load initial data!")
        return
    expiry = data['records']['expiryDates'][0]

    # Define tabs
    tabs = st.tabs(["Real-Time Resistance Alerts", "Support & Resistance Table"])

    # Real-Time Resistance Alerts Tab
    with tabs[0]:
        st.subheader("Real-Time Resistance Alerts")
        current_time = time.time()
        auto_scan_interval_seconds = st.session_state['telegram_config']['auto_scan_interval'] * 60
        time_since_last_scan = current_time - st.session_state['last_scan_time']
        time_to_next_scan = auto_scan_interval_seconds - time_since_last_scan
        
        st.write(f"Last Scan: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(st.session_state['last_scan_time']))}")
        minutes_to_next_scan = int(time_to_next_scan // 60)
        seconds_to_next_scan = int(time_to_next_scan % 60)
        st.write(f"Next Scan in: {minutes_to_next_scan} minutes {seconds_to_next_scan} seconds")

        # Automatic scanning logic (60s interval)
        if st.session_state['auto_running']:
            current_time = time.time()
            if current_time - st.session_state['last_auto_time'] >= 60:
                tickers = load_tickers() if not specific_tickers else [t.strip() for t in specific_tickers.split(',')]
                perform_scan(tickers, expiry, telegram_bot_token, telegram_chat_id, proximity_to_resistance)
                st.session_state['last_scan_time'] = current_time
                st.session_state['last_auto_time'] = current_time
                st.rerun()

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Scan All Tickers"):
                tickers = load_tickers()
                perform_scan(tickers, expiry, telegram_bot_token, telegram_chat_id, proximity_to_resistance)
                st.session_state['last_scan_time'] = time.time()
                st.session_state['auto_scan_triggered'] = False
                st.rerun()
        with col2:
            if st.button("Scan Specific Tickers") and specific_tickers:
                tickers = [t.strip() for t in specific_tickers.split(',')]
                perform_scan(tickers, expiry, telegram_bot_token, telegram_chat_id, proximity_to_resistance)
                st.session_state['last_scan_time'] = time.time()
                st.session_state['auto_scan_triggered'] = False
                st.rerun()

        status = "Running" if st.session_state['auto_running'] else "Stopped"
        st.write(f"Auto Scan (60s): {status}")

        if st.session_state['scanned_stocks']:
            st.write("### Scanned Stocks")
            st.dataframe(pd.DataFrame(st.session_state['scanned_stocks']))

        if st.session_state['suggestions']:
            st.write("### Stocks Near Resistance (Alerts)")
            st.table(pd.DataFrame(st.session_state['suggestions']))

        if st.session_state['call_suggestions']:
            st.write("### Suggested Call Options")
            st.table(pd.DataFrame(st.session_state['call_suggestions']))

    # Support & Resistance Table Tab
    with tabs[1]:
        st.subheader("Support & Resistance Levels for All Stocks")
        if st.button("Refresh Table"):
            tickers = load_tickers()
            table_data = generate_support_resistance_table(tickers, expiry)
            st.session_state['table_data'] = table_data
            st.rerun()

        table_data = st.session_state['table_data']
        if table_data:
            st.write("### Support & Resistance Table")
            st.dataframe(pd.DataFrame(table_data))
        else:
            st.info("No data available. Click 'Refresh Table' to load data.")

if __name__ == "__main__":
    main()