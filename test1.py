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
from datetime import datetime

# Create a cloudscraper session
scraper = cloudscraper.create_scraper()

# Constants
BASE_URL = "https://www.nseindia.com"
STORED_TICKERS_PATH = "tickers-test.csv"
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
    print(f"Fetching data from: {url} and {quote_url}")
    
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

# Suggest Call Options (OTM 3% above spot price)
def suggest_call_options(data: Dict, expiry: str, underlying: float, ticker: str) -> Optional[Dict]:
    call_df, _ = process_option_data(data, expiry)
    if call_df.empty or underlying <= 0:
        return None
    
    target_strike = underlying * 1.03
    call_df = call_df[call_df['Strike'] > underlying].sort_values('Strike')
    
    if not call_df.empty:
        suggested_strike = call_df.iloc[0]['Strike']
        last_price = call_df.iloc[0]['Last Price']
        potential_gain_percent = ((suggested_strike - underlying) / last_price * 100) if last_price > 0 else 0
        
        return {
            "Ticker": ticker,
            "Underlying": underlying,
            "Suggested_Call_Strike": suggested_strike,
            "Call_Last_Price": last_price,
            "Potential_Gain_%": potential_gain_percent,
            "Expiry": expiry,
            "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    return None

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

# Check Resistance and Send Notification with Call Suggestions (for Tab 1)
def check_resistance_and_notify(tickers: List[str], expiry: str, bot_token: str, chat_id: str, proximity_percent: float):
    refresh_key = time.time()
    suggestions = []
    call_suggestions = []
    
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

                call_suggestion = suggest_call_options(data, expiry, underlying, ticker)
                if call_suggestion:
                    call_message = (
                        f"*Call Option Suggestion*\n"
                        f"Stock: *{ticker}*\n"
                        f"Underlying: *₹{underlying:.2f}*\n"
                        f"Suggested Call Strike: *₹{call_suggestion['Suggested_Call_Strike']:.2f}* (OTM 3% above spot)\n"
                        f"Call Last Price: *₹{call_suggestion['Call_Last_Price']:.2f}*\n"
                        f"Potential Gain: *{call_suggestion['Potential_Gain_%']:.2f}%*\n"
                        f"Expiry: *{expiry}*\n"
                        f"Timestamp: *{call_suggestion['Timestamp']}*"
                    )
                    asyncio.run(send_telegram_message(bot_token, chat_id, call_message))
                    call_suggestions.append(call_suggestion)

    return suggestions, call_suggestions

# Scan for Historical Data based on Proximity Criteria (for Tab 3)
def scan_historical_data(tickers: List[str], expiry: str, proximity_percent: float) -> List[Dict]:
    refresh_key = time.time()
    historical_data = []
    
    for ticker in tickers:
        with st.spinner(f"Scanning {ticker} for historical data..."):
            data = fetch_options_data(ticker, refresh_key)
            if not data or 'records' not in data:
                print(f"Failed to fetch data for {ticker}")
                continue
            
            call_df, put_df = process_option_data(data, expiry)
            underlying = data['records'].get('underlyingValue', 0)
            support_strike, resistance_strike = identify_support_resistance(call_df, put_df)
            
            if resistance_strike is None:
                continue
            
            proximity_threshold = resistance_strike * (abs(proximity_percent) / 100)
            distance_to_resistance = resistance_strike - underlying
            
            if 0 <= distance_to_resistance <= proximity_threshold:
                historical_data.append({
                    "Ticker": ticker,
                    "Underlying": underlying,
                    "Resistance": resistance_strike,
                    "Support": support_strike,
                    "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                })
                print(f"Added {ticker} to historical data: Within {proximity_percent}% of resistance")
    
    return historical_data

# Generate Support/Resistance Table Data and Save to JSON
def generate_support_resistance_table(tickers: List[str], expiry: str) -> List[Dict]:
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

    # Initialize auto_running from config if not in session state
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
            min_value=0.0,
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

    # Define three tabs
    tabs = st.tabs(["Real-Time Resistance Alerts", "Support & Resistance Table", "Historical Scan Data"])

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

        def perform_scan(tickers_to_scan):
            new_suggestions, new_call_suggestions = check_resistance_and_notify(
                tickers_to_scan, expiry, telegram_bot_token, telegram_chat_id,
                st.session_state['telegram_config']['proximity_to_resistance']
            )
            st.session_state['suggestions'].extend(new_suggestions)
            st.session_state['call_suggestions'].extend(new_call_suggestions)
            save_alerts_data(st.session_state['suggestions'])
            save_call_suggestions(st.session_state['call_suggestions'])

            # Update scanned stocks (Tab 1 only)
            refresh_key = time.time()
            scanned_data = []
            for ticker in tickers_to_scan:
                data = fetch_options_data(ticker, refresh_key)
                if data and 'records' in data:
                    call_df, put_df = process_option_data(data, expiry)
                    underlying = data['records'].get('underlyingValue', 0)
                    support_strike, resistance_strike = identify_support_resistance(call_df, put_df)
                    
                    scanned_data.append({
                        "Ticker": ticker,
                        "Underlying": underlying,
                        "Resistance": resistance_strike,
                        "Support": support_strike,
                        "Last_Scanned": time.strftime("%Y-%m-%d %H:%M:%S")
                    })
                else:
                    print(f"No data fetched for ticker: {ticker}")
            st.session_state['scanned_stocks'] = scanned_data

        # Automatic scanning logic (60s interval)
        if st.session_state['auto_running']:
            current_time = time.time()
            if current_time - st.session_state['last_auto_time'] >= 60:
                tickers = load_tickers() if not specific_tickers else [t.strip() for t in specific_tickers.split(',')]
                perform_scan(tickers)
                st.session_state['last_scan_time'] = current_time
                st.session_state['last_auto_time'] = current_time
                st.rerun()

        # Button columns
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Scan All Tickers"):
                tickers = load_tickers()
                perform_scan(tickers)
                st.session_state['last_scan_time'] = time.time()
                st.session_state['auto_scan_triggered'] = False
                st.rerun()
        with col2:
            if st.button("Scan Specific Tickers") and specific_tickers:
                tickers = [t.strip() for t in specific_tickers.split(',')]
                perform_scan(tickers)
                st.session_state['last_scan_time'] = time.time()
                st.session_state['auto_scan_triggered'] = False
                st.rerun()

        # Display auto status
        status = "Running" if st.session_state['auto_running'] else "Stopped"
        st.write(f"Auto Scan (60s): {status}")

        if st.session_state['scanned_stocks']:
            st.write("### Scanned Stocks (Sortable & Searchable)")
            search_query = st.text_input("Search Scanned Stocks by Ticker", key="scanned_search")
            scanned_df = pd.DataFrame(st.session_state['scanned_stocks'])
            
            if search_query:
                scanned_df = scanned_df[scanned_df['Ticker'].str.contains(search_query, case=False, na=False)]
            
            st.dataframe(
                scanned_df,
                column_config={
                    "Ticker": st.column_config.TextColumn("Ticker"),
                    "Underlying": st.column_config.NumberColumn("Underlying", format="%.2f"),
                    "Resistance": st.column_config.NumberColumn("Resistance", format="%.2f"),
                    "Support": st.column_config.NumberColumn("Support", format="%.2f"),
                    "Last_Scanned": st.column_config.TextColumn("Last Scanned")
                },
                use_container_width=True,
                height=300
            )

        if st.session_state['suggestions']:
            st.write("### Stocks Near Resistance (Alerts)")
            alert_search_query = st.text_input("Search Alerts by Ticker", key="alerts_search")
            suggestions_df = pd.DataFrame(st.session_state['suggestions'])
            
            if alert_search_query:
                suggestions_df = suggestions_df[suggestions_df['Ticker'].str.contains(alert_search_query, case=False, na=False)]
            
            styled_df = suggestions_df.style.format({
                'Underlying': '{:.2f}',
                'Resistance': '{:.2f}',
                'Distance_to_Resistance': '{:.2f}'
            })
            st.table(styled_df)
            
            if st.button("Clear Alerts"):
                st.session_state['suggestions'] = []
                save_alerts_data(st.session_state['suggestions'])
                st.rerun()
        else:
            st.info("No stocks currently near strong resistance.")

        if st.session_state['call_suggestions']:
            st.write("### Suggested Call Options (OTM 3% Above Spot)")
            call_search_query = st.text_input("Search Call Suggestions by Ticker", key="call_search")
            call_suggestions_df = pd.DataFrame(st.session_state['call_suggestions'])
            
            if call_search_query:
                call_suggestions_df = call_suggestions_df[call_suggestions_df['Ticker'].str.contains(call_search_query, case=False, na=False)]
            
            styled_call_df = call_suggestions_df.style.format({
                'Underlying': '{:.2f}',
                'Suggested_Call_Strike': '{:.2f}',
                'Call_Last_Price': '{:.2f}',
                'Potential_Gain_%': '{:.2f}'
            })
            st.table(styled_call_df)
            
            if st.button("Clear Call Suggestions"):
                st.session_state['call_suggestions'] = []
                save_call_suggestions(st.session_state['call_suggestions'])
                st.rerun()
        else:
            st.info("No call option suggestions available.")

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
            st.write("### Support & Resistance Table (Sortable & Searchable)")
            search_query = st.text_input("Search Table by Ticker", key="table_search")
            table_df = pd.DataFrame(table_data)
            
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
                    "Last_Updated": st.column_config.TextColumn("Last Updated")
                },
                use_container_width=True,
                height=400
            )
        else:
            st.info("No data available. Click 'Refresh Table' to load support and resistance data.")

    # Historical Scan Data Tab (Tab 3)
    with tabs[2]:
        st.subheader("Historical Scan Data")

        # Date picker for filtering
        selected_date = st.date_input("Select Date to Filter Historical Data", value=None)

        # Buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Refresh Historical Data"):
                tickers = load_tickers()
                new_historical_data = scan_historical_data(
                    tickers, expiry, st.session_state['telegram_config']['proximity_to_resistance']
                )
                st.session_state['historical_data'] = new_historical_data
                save_historical_data(st.session_state['historical_data'])
                st.rerun()
        with col2:
            if st.button("Clear Historical Data"):
                st.session_state['historical_data'] = []
                save_historical_data(st.session_state['historical_data'])
                print("Historical data cleared")
                st.rerun()

        # Display historical data
        if st.session_state['historical_data']:
            st.write("### Historical Data (Sortable & Searchable)")
            hist_search_query = st.text_input("Search Historical Data by Ticker", key="hist_search")
            hist_df = pd.DataFrame(st.session_state['historical_data'])
            print(f"Raw historical data: {len(hist_df)} entries")

            # Apply filters
            filtered_df = hist_df.copy()
            if selected_date:
                filtered_df['Date'] = pd.to_datetime(filtered_df['Timestamp'], errors='coerce').dt.date
                filtered_df = filtered_df[filtered_df['Date'] == selected_date]
                print(f"Filtered by date {selected_date}: {len(filtered_df)} entries")
            if hist_search_query:
                filtered_df = filtered_df[filtered_df['Ticker'].str.contains(hist_search_query, case=False, na=False)]
                print(f"Filtered by search '{hist_search_query}': {len(filtered_df)} entries")

            if not filtered_df.empty:
                st.dataframe(
                    filtered_df,
                    column_config={
                        "Ticker": st.column_config.TextColumn("Ticker"),
                        "Underlying": st.column_config.NumberColumn("Underlying", format="%.2f"),
                        "Resistance": st.column_config.NumberColumn("Resistance", format="%.2f"),
                        "Support": st.column_config.NumberColumn("Support", format="%.2f"),
                        "Timestamp": st.column_config.TextColumn("Timestamp")
                    },
                    use_container_width=True,
                    height=400
                )
            else:
                st.warning("No data matches the selected filters. Try adjusting the date or search query.")
        else:
            st.warning("No historical data available. Click 'Refresh Historical Data' to scan for stocks near resistance.")

if __name__ == "__main__":
    main()