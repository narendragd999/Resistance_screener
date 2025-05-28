import streamlit as st
import requests
import pandas as pd
import time
import json
import os
import cloudscraper
import aiohttp
import asyncio
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date
import yfinance as yf
import numpy as np

# Create a cloudscraper session
scraper = cloudscraper.create_scraper()

# Constants
BASE_URL = "https://www.nseindia.com"
STORED_TICKERS_PATH = "tickers-test.csv"
CONFIG_FILE = "config.json"
TEMP_TABLE_DATA_FILE = "temp_table_data.json"
HISTORICAL_DATA_FILE = "historical_resistance_data.json"
ALERTS_DATA_FILE = "alerts_data.json"

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

# Load/Save Config, Table Data, Historical Data, and Alerts Data
def load_config() -> Dict:
    default_config = {
        "telegram_bot_token": "",
        "telegram_chat_id": "",
        "auto_scan_interval": 5,
        "proximity_to_resistance": 0.5,
        "price_gain_threshold": 0.5  # New config for price gain threshold
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

def load_table_data() -> List[Dict]:
    if os.path.exists(TEMP_TABLE_DATA_FILE):
        with open(TEMP_TABLE_DATA_FILE, 'r') as f:
            return json.load(f)
    return []

def save_table_data(data: List[Dict]):
    with open(TEMP_TABLE_DATA_FILE, 'w') as f:
        json.dump(data, f, indent=4)

def load_historical_data() -> List[Dict]:
    if os.path.exists(HISTORICAL_DATA_FILE):
        try:
            with open(HISTORICAL_DATA_FILE, 'r') as f:
                data = json.load(f)
                converted_data = []
                for item in data:
                    converted_item = {}
                    for key, value in item.items():
                        if isinstance(value, (np.int64, np.float64)):
                            converted_item[key] = float(value) if isinstance(value, np.float64) else int(value)
                        else:
                            converted_item[key] = value
                    converted_data.append(converted_item)
                return converted_data
        except json.JSONDecodeError as e:
            st.error(f"Error decoding JSON file {HISTORICAL_DATA_FILE}: {str(e)}. File may be corrupted or empty. Creating new empty list.")
            return []
        except Exception as e:
            st.error(f"Unexpected error loading historical data: {str(e)}")
            return []
    return []

def save_historical_data(data: List[Dict]):
    converted_data = []
    for item in data:
        converted_item = {}
        for key, value in item.items():
            if isinstance(value, (pd.Series, pd.DataFrame)):
                converted_item[key] = value.to_list() if isinstance(value, pd.Series) else value.to_dict()
            elif isinstance(value, (np.int64, np.float64)):
                converted_item[key] = int(value) if isinstance(value, np.int64) else float(value)
            else:
                converted_item[key] = value
        converted_data.append(converted_item)
    with open(HISTORICAL_DATA_FILE, 'w') as f:
        json.dump(converted_data, f, indent=4)

def load_alerts_data() -> List[Dict]:
    if os.path.exists(ALERTS_DATA_FILE):
        try:
            with open(ALERTS_DATA_FILE, 'r') as f:
                data = json.load(f)
                converted_data = []
                for item in data:
                    converted_item = {}
                    for key, value in item.items():
                        if isinstance(value, (np.int64, np.float64)):
                            converted_item[key] = float(value) if isinstance(value, np.float64) else int(value)
                        else:
                            converted_item[key] = value
                    converted_data.append(converted_item)
                return converted_data
        except json.JSONDecodeError as e:
            st.error(f"Error decoding JSON file {ALERTS_DATA_FILE}: {str(e)}. File may be corrupted or empty. Creating new empty list.")
            return []
        except Exception as e:
            st.error(f"Unexpected error loading alerts data: {str(e)}")
            return []
    return []

def save_alerts_data(data: List[Dict]):
    converted_data = []
    for item in data:
        converted_item = {}
        for key, value in item.items():
            if isinstance(value, (pd.Series, pd.DataFrame)):
                converted_item[key] = value.to_list() if isinstance(value, pd.Series) else value.to_dict()
            elif isinstance(value, (np.int64, np.float64)):
                converted_item[key] = int(value) if isinstance(value, np.int64) else float(value)
            else:
                converted_item[key] = value
        converted_data.append(converted_item)
    with open(ALERTS_DATA_FILE, 'w') as f:
        json.dump(converted_data, f, indent=4)

# Telegram Integration
async def send_telegram_message(bot_token: str, chat_id: str, message: str):
    if not bot_token or not chat_id:
        st.error("Telegram Bot Token or Chat ID is missing. Please configure them in the sidebar.")
        return
    
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                st.error(f"Failed to send Telegram message: {await response.text()}")
            else:
                st.success(f"Telegram message sent: {message}")

# Fetch Options Data with Last Price as Underlying
def fetch_options_data(symbol: str, _refresh_key: float) -> Optional[Dict]:
    url = f"{BASE_URL}/api/option-chain-equities?symbol={symbol}"
    quote_url = f"https://www.nseindia.com/api/quote-equity?symbol={symbol}"
    #print(f"Fetching data from: {url} and {quote_url}")
    
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
    
    return data

# Process Option Data and Identify Support/Resistance
def process_option_data(data: Dict, expiry: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not data or 'records' not in data or 'data' not in data['records']:
        print("Invalid data")
        return pd.DataFrame(), pd.DataFrame()
    
    options = [item for item in data['records']['data'] if item.get('expiryDate') == expiry]
    if not options:
        print(f"No options found for expiry {expiry}")
        return pd.DataFrame(), pd.DataFrame()
    
    strikes = sorted({item['strikePrice'] for item in options})
    call_data = {s: {'OI': 0, 'Volume': 0} for s in strikes}
    put_data = {s: {'OI': 0, 'Volume': 0} for s in strikes}
    
    for item in options:
        strike = item['strikePrice']
        if 'CE' in item:
            call_data[strike] = {'OI': float(item['CE']['openInterest']), 'Volume': float(item['CE']['totalTradedVolume'])}
        if 'PE' in item:
            put_data[strike] = {'OI': float(item['PE']['openInterest']), 'Volume': float(item['PE']['totalTradedVolume'])}
    
    call_df = pd.DataFrame([{'Strike': k, **v} for k, v in call_data.items()])
    put_df = pd.DataFrame([{'Strike': k, **v} for k, v in put_data.items()])
    
    return call_df, put_df

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

# New Function: Analyze Option Chain for Price Gains
def analyze_option_chain_price_gain(symbol: str, resistance_strike: float, expiry: str, gain_threshold: float = 50.0) -> Dict:
    url = f"{BASE_URL}/api/option-chain-equities?symbol={symbol}"
    
    response = scraper.get(url, headers=headers)
    if response.status_code != 200:
        print(f"Failed to load option chain for {symbol}: {response.status_code}")
        return {"gain_percent": None, "ltp": None, "low_price": None}
    
    data = response.json()
    if not data or 'records' not in data or 'data' not in data['records']:
        return {"gain_percent": None, "ltp": None, "low_price": None}
    
    options = [item for item in data['records']['data'] if item.get('expiryDate') == expiry and 'CE' in item]
    for item in options:
        if item['strikePrice'] == resistance_strike:
            ce_data = item['CE']            
            ltp = float(ce_data.get('lastPrice', 0))
            low_price = float(ce_data.get('lowPrice', ltp))  # Use lowPrice from option chain, fallback to LTP if unavailable
            if low_price > 0 and ltp > low_price:
                gain_percent = ((ltp - low_price) / low_price) * 100
                if gain_percent >= gain_threshold:
                    return {"gain_percent": gain_percent, "ltp": ltp, "low_price": low_price}
            break
    
    return {"gain_percent": None, "ltp": None, "low_price": None}

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

# Check Resistance and Send Notification (Updated)
def check_resistance_and_notify(tickers: List[str], expiry: str, bot_token: str, chat_id: str, proximity_percent: float, gain_threshold: float = 0.5):
    refresh_key = time.time()
    suggestions = []
    
    for ticker in tickers:
        with st.spinner(f"Fetching data for {ticker}..."):
            data = fetch_options_data(ticker, refresh_key)
            if not data or 'records' not in data:
                print(f"Failed to fetch data for {ticker}")
                continue
            
            call_df, put_df = process_option_data(data, expiry)
            underlying = data['records'].get('underlyingValue', 0)
            resistance_strike = identify_support_resistance(call_df, put_df)[1]
            
            if resistance_strike is None:
                continue
            
            proximity_threshold = resistance_strike * (abs(proximity_percent) / 100)
            distance_to_resistance = resistance_strike - underlying
            
            if 0 <= distance_to_resistance <= proximity_threshold:
                # Analyze option chain for price gain at resistance strike
                price_gain_data = analyze_option_chain_price_gain(ticker, resistance_strike, expiry, gain_threshold)
                gain_percent = price_gain_data["gain_percent"]
                ltp = price_gain_data["ltp"]
                low_price = price_gain_data["low_price"]
                
                # Prepare Telegram message
                message = (
                    f"*Resistance Alert*\n"
                    f"Stock: *{ticker}*\n"
                    f"Underlying: *₹{underlying:.2f}*\n"
                    f"Resistance: *₹{resistance_strike:.2f}*\n"
                    f"Distance: *{distance_to_resistance:.2f}*\n"
                    f"Reason: *Within {proximity_percent}% of strong resistance*"
                )
                if gain_percent is not None:
                    message += (
                        f"\n*Option Chain Alert*\n"
                        f"Call LTP: *₹{ltp:.2f}*\n"
                        f"Low Price: *₹{low_price:.2f}*\n"
                        f"Gain: *{gain_percent:.2f}%*"
                    )
                
                asyncio.run(send_telegram_message(bot_token, chat_id, message))
                suggestion = {
                    "Ticker": ticker,
                    "Underlying": underlying,
                    "Resistance": resistance_strike,
                    "Distance_to_Resistance": distance_to_resistance,
                    "Call_LTP": ltp if ltp is not None else None,
                    "Call_Low_Price": low_price if low_price is not None else None,
                    "Gain_Percent": gain_percent if gain_percent is not None else None
                }
                suggestions.append(suggestion)
    
    return suggestions

# Fetch historical prices using yfinance
def get_historical_price(ticker: str, target_date: date) -> Optional[Tuple[float, float]]:
    try:
        stock = yf.Ticker(ticker + ".NS")
        hist = stock.history(start=target_date, end=target_date + pd.Timedelta(days=1))
        if not hist.empty:
            high_price = hist['High'].iloc[0]
            close_price = hist['Close'].iloc[0]
            return high_price, close_price
        return None, None
    except Exception as e:
        st.error(f"Error fetching historical price for {ticker}: {e}")
        return None, None

# Check historical resistance
def check_historical_resistance(tickers: List[str], target_date: date, expiry: str, proximity_percent: float) -> List[Dict]:
    date_str = target_date.strftime("%Y-%m-%d")
    refresh_key = time.time()
    results = []
    
    for ticker in tickers:
        with st.spinner(f"Checking historical data for {ticker} on {date_str}..."):
            data = fetch_options_data(ticker, refresh_key)
            if not data or 'records' not in data:
                continue
            
            high_price, close_price = get_historical_price(ticker, target_date)
            if high_price is None or isinstance(high_price, (np.int64, np.float64)):
                high_price = float(high_price) if high_price is not None else 0.0
            if close_price is None or isinstance(close_price, (np.int64, np.float64)):
                close_price = float(close_price) if close_price is not None else 0.0

            call_df, put_df = process_option_data(data, expiry)
            resistance_strike = identify_support_resistance(call_df, put_df)[1]
            
            if resistance_strike is None or isinstance(resistance_strike, (np.int64, np.float64)):
                resistance_strike = float(resistance_strike) if resistance_strike is not None else 0.0
            
            touched_resistance = high_price >= resistance_strike
            volume = call_df['Volume'].sum() + put_df['Volume'].sum()
            if isinstance(volume, (np.int64, np.float64)):
                volume = float(volume)
            
            distance_to_resistance = resistance_strike - close_price

            if touched_resistance:
                results.append({
                    "Date": date_str,
                    "Time": "End of Day",
                    "Ticker": ticker,
                    "High_Price": float(high_price) if high_price is not None else 0.0,
                    "Close_Price": float(close_price) if close_price is not None else 0.0,
                    "Resistance_Price": float(resistance_strike) if resistance_strike is not None else 0.0,
                    "Distance_to_Resistance": float(distance_to_resistance) if distance_to_resistance is not None else 0.0,
                    "Volume": float(volume) if volume is not None else 0.0,
                    "Touched_Resistance": "Yes"
                })

    save_historical_data(results)
    return results

# Download data as CSV
def download_csv(data: List[Dict], filename: str):
    df = pd.DataFrame(data)
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name=filename,
        mime="text/csv",
    )

# Generate Support/Resistance Table Data and Save to JSON (Updated)
def generate_support_resistance_table(tickers: List[str], expiry: str, gain_threshold: float = 0.5) -> List[Dict]:
    refresh_key = time.time()
    table_data = []
    volume_threshold = 100000
    
    for ticker in tickers:
        with st.spinner(f"Fetching data for {ticker}..."):
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
            
            # Check option chain for price gain at resistance strike
            price_gain_data = analyze_option_chain_price_gain(ticker, resistance_strike, expiry, gain_threshold)
            
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
                "Call_LTP": price_gain_data["ltp"],
                "Call_Low_Price": price_gain_data["low_price"],
                "Gain_Percent": price_gain_data["gain_percent"]
            })
    
    save_table_data(table_data)
    return table_data

# Main Application
def main():
    st.set_page_config(page_title="Resistance Screener", layout="wide")
    st.title("Real-Time Resistance Screener")

    # Initialize all required session state variables
    if 'telegram_config' not in st.session_state:
        st.session_state['telegram_config'] = load_config()
    if 'last_scan_time' not in st.session_state:
        st.session_state['last_scan_time'] = time.time()
    if 'refresh_key' not in st.session_state:
        st.session_state['refresh_key'] = time.time()
    if 'suggestions' not in st.session_state:
        st.session_state['suggestions'] = load_alerts_data()
    if 'table_data' not in st.session_state:
        st.session_state['table_data'] = load_table_data()
    if 'historical_data' not in st.session_state:
        st.session_state['historical_data'] = load_historical_data()
    if 'auto_scan_triggered' not in st.session_state:
        st.session_state['auto_scan_triggered'] = False

    # Sidebar Configuration
    with st.sidebar:
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
            min_value=0.0,
            max_value=5.0,
            key="proximity_to_resistance_input"
        )
        price_gain_threshold = st.number_input(
            "Price Gain Threshold (%):",
            value=st.session_state['telegram_config']['price_gain_threshold'],
            step=10.0,
            min_value=0.0,
            max_value=1000.0,
            key="price_gain_threshold_input"
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
        if st.session_state['telegram_config']['price_gain_threshold'] != price_gain_threshold:
            st.session_state['telegram_config']['price_gain_threshold'] = price_gain_threshold
            config_changed = True
        if config_changed:
            save_config(st.session_state['telegram_config'])

        if not telegram_bot_token or not telegram_chat_id:
            st.warning("Please configure Telegram Bot Token and Chat ID.")

    # Fetch initial data to get expiry
    data = fetch_options_data("HDFCBANK", time.time())
    if not data or 'records' not in data:
        st.error("Failed to load initial data!")
        return
    expiry = data['records']['expiryDates'][0]

    # Tabs
    tabs = st.tabs(["Resistance Alerts", "Support & Resistance Table", "Historical Resistance Tracker"])

    # Inside the "Real-Time Resistance Alerts" tab (around line 614 in your traceback)
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

        if time_to_next_scan <= 0 and not st.session_state['auto_scan_triggered']:
            tickers = load_tickers()
            new_suggestions = check_resistance_and_notify(
                tickers, expiry, telegram_bot_token, telegram_chat_id,
                st.session_state['telegram_config']['proximity_to_resistance'],
                st.session_state['telegram_config']['price_gain_threshold']
            )
            st.session_state['suggestions'] = new_suggestions
            save_alerts_data(st.session_state['suggestions'])
            st.session_state['last_scan_time'] = current_time
            st.session_state['auto_scan_triggered'] = True
            st.rerun()
        elif time_to_next_scan > 0:
            st.session_state['auto_scan_triggered'] = False

        if st.button("Scan Now"):
            tickers = load_tickers()
            new_suggestions = check_resistance_and_notify(
                tickers, expiry, telegram_bot_token, telegram_chat_id,
                st.session_state['telegram_config']['proximity_to_resistance'],
                st.session_state['telegram_config']['price_gain_threshold']
            )
            st.session_state['suggestions'] = new_suggestions
            save_alerts_data(st.session_state['suggestions'])
            st.session_state['last_scan_time'] = time.time()
            st.session_state['auto_scan_triggered'] = False
            st.rerun()

        if st.session_state['suggestions']:
            st.write("### Stocks Near Resistance (Real-Time Alerts)")
            search_query = st.text_input("Search Alerts by Ticker", key="alerts_search")
            suggestions_df = pd.DataFrame(st.session_state['suggestions'])
            
            if search_query:
                suggestions_df = suggestions_df[suggestions_df['Ticker'].str.contains(search_query, case=False, na=False)]
            
            # Replace None with NaN to avoid formatting issues
            suggestions_df = suggestions_df.fillna(pd.NA)
            
            # Define a custom formatter to handle None/NaN gracefully
            def format_float(x):
                return '{:.2f}'.format(x) if pd.notna(x) else '-'

            styled_df = suggestions_df.style.format({
                'Underlying': format_float,
                'Resistance': format_float,
                'Distance_to_Resistance': format_float,
                'Call_LTP': format_float,
                'Call_Low_Price': format_float,
                'Gain_Percent': format_float
            })
            st.table(styled_df)
            
            if st.button("Clear Alerts"):
                st.session_state['suggestions'] = []
                save_alerts_data([])
                st.rerun()
        else:
            st.info("No stocks currently near strong resistance. Click 'Scan Now' or wait for auto-scan.")

    # Support & Resistance Table Tab
    with tabs[1]:
        st.subheader("Support & Resistance Levels for All Stocks")

        if st.button("Refresh Table"):
            tickers = load_tickers()
            table_data = generate_support_resistance_table(
                tickers, expiry, st.session_state['telegram_config']['price_gain_threshold']
            )
            st.session_state['table_data'] = table_data
            save_table_data(table_data)
            st.rerun()

        if st.session_state['table_data']:
            st.write("### Support & Resistance Table (Sortable & Searchable)")
            search_query = st.text_input("Search Table by Ticker", key="table_search")
            table_df = pd.DataFrame(st.session_state['table_data'])
            
            if search_query:
                table_df = table_df[table_df['Ticker'].str.contains(search_query, case=False, na=False)]
            
            st.dataframe(
                table_df,
                column_config={
                    "Ticker": st.column_config.TextColumn("Ticker"),
                    "Underlying": st.column_config.NumberColumn("Underlying", format="%.2f"),
                    "Support": st.column_config.NumberColumn("Support", format="%.2f"),
                    "Resistance": st.column_config.NumberColumn("Resistance", format="%.2f"),
                    "Distance_from_Resistance": st.column_config.NumberColumn("Distance from Resistance", format="%.2f"),
                    "Distance_from_Support": st.column_config.NumberColumn("Distance from Support", format="%.2f"),
                    "Distance_%_from_Resistance": st.column_config.NumberColumn("Distance % from Resistance", format="%.2f"),
                    "Distance_%_from_Support": st.column_config.NumberColumn("Distance % from Support", format="%.2f"),
                    "High_Volume_Gainer": st.column_config.TextColumn("High Volume Gainer"),
                    "Call_LTP": st.column_config.NumberColumn("Call LTP", format="%.2f"),
                    "Call_Low_Price": st.column_config.NumberColumn("Call Low Price", format="%.2f"),
                    "Gain_Percent": st.column_config.NumberColumn("Gain %", format="%.2f")
                },
                use_container_width=True,
                height=400
            )
        else:
            st.info("No data available. Click 'Refresh Table' to load support and resistance data.")

    # Historical Resistance Tracker Tab
    with tabs[2]:
        st.subheader("Historical Resistance Tracker")

        selected_date = st.date_input("Select Date", value=date.today() - pd.Timedelta(days=1))

        if st.button("Check Historical Resistance"):
            tickers = load_tickers()
            proximity_percent = st.session_state['telegram_config']['proximity_to_resistance']
            
            with st.spinner("Checking historical resistance touches..."):
                historical_results = check_historical_resistance(
                    tickers, selected_date, expiry, proximity_percent
                )
                st.session_state['historical_data'] = historical_results
                save_historical_data(historical_results)

            if st.session_state['historical_data']:
                st.write("### Stocks that touched resistance on selected date")
                historical_df = pd.DataFrame(st.session_state['historical_data'])
                st.dataframe(
                    historical_df,
                    column_config={
                        "Date": st.column_config.TextColumn("Date"),
                        "Time": st.column_config.TextColumn("Time"),
                        "Ticker": st.column_config.TextColumn("Ticker"),
                        "High_Price": st.column_config.NumberColumn("High Price", format="%.2f"),
                        "Close_Price": st.column_config.NumberColumn("Close Price", format="%.2f"),
                        "Resistance_Price": st.column_config.NumberColumn("Resistance Price", format="%.2f"),
                        "Distance_to_Resistance": st.column_config.NumberColumn("Distance to Resistance (Close)", format="%.2f"),
                        "Volume": st.column_config.NumberColumn("Volume", format="%.0f"),
                        "Touched_Resistance": st.column_config.TextColumn("Touched Resistance")
                    },
                    use_container_width=True
                )
                download_csv(st.session_state['historical_data'], f"historical_resistance_{selected_date}.csv")
            else:
                st.info("No stocks touched resistance on the selected date.")
        else:
            if st.session_state['historical_data']:
                st.write("### Last Historical Scan Results")
                historical_df = pd.DataFrame(st.session_state['historical_data'])
                st.dataframe(
                    historical_df,
                    column_config={
                        "Date": st.column_config.TextColumn("Date"),
                        "Time": st.column_config.TextColumn("Time"),
                        "Ticker": st.column_config.TextColumn("Ticker"),
                        "High_Price": st.column_config.NumberColumn("High Price", format="%.2f"),
                        "Close_Price": st.column_config.NumberColumn("Close Price", format="%.2f"),
                        "Resistance_Price": st.column_config.NumberColumn("Resistance Price", format="%.2f"),
                        "Distance_to_Resistance": st.column_config.NumberColumn("Distance to Resistance (Close)", format="%.2f"),
                        "Volume": st.column_config.NumberColumn("Volume", format="%.0f"),
                        "Touched_Resistance": st.column_config.TextColumn("Touched Resistance")
                    },
                    use_container_width=True
                )
                download_csv(st.session_state['historical_data'], f"historical_resistance_{st.session_state['historical_data'][0]['Date'] if st.session_state['historical_data'] else 'last_scan'}.csv")
            else:
                st.info("No historical data available. Select a date and click 'Check Historical Resistance'.")

if __name__ == "__main__":
    main()