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

# Create a cloudscraper session
scraper = cloudscraper.create_scraper()

# Constants
BASE_URL = "https://www.nseindia.com"
STORED_TICKERS_PATH = "tickers.csv"
CONFIG_FILE = "config.json"
TEMP_TABLE_DATA_FILE = "temp_table_data.json"  # Temporary JSON file for table data

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
        "proximity_to_resistance": 0.5
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
    call_data = {s: {'OI': 0, 'Volume': 0} for s in strikes}
    put_data = {s: {'OI': 0, 'Volume': 0} for s in strikes}
    
    for item in options:
        strike = item['strikePrice']
        if 'CE' in item:
            call_data[strike] = {'OI': item['CE']['openInterest'], 'Volume': item['CE']['totalTradedVolume']}
        if 'PE' in item:
            put_data[strike] = {'OI': item['PE']['openInterest'], 'Volume': item['PE']['totalTradedVolume']}
    
    call_df = pd.DataFrame([{'Strike': k, **v} for k, v in call_data.items()])
    put_df = pd.DataFrame([{'Strike': k, **v} for k, v in put_data.items()])
    
    return call_df, put_df

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

# Check Resistance and Send Notification
def check_resistance_and_notify(tickers: List[str], expiry: str, bot_token: str, chat_id: str, proximity_percent: float):
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
                    "Distance_to_Resistance": distance_to_resistance
                })
    
    return suggestions

# Generate Support/Resistance Table Data and Save to JSON
def generate_support_resistance_table(tickers: List[str], expiry: str) -> List[Dict]:
    refresh_key = time.time()
    table_data = []
    volume_threshold = 100000  # Define a threshold for high volume gainer
    
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
                "High_Volume_Gainer": high_volume_gainer
            })
    
    save_table_data(table_data)  # Save to JSON
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
        st.session_state['suggestions'] = None

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
    expiry = data['records']['expiryDates'][0]  # Use first expiry for simplicity

    # Tabs
    tabs = st.tabs(["Resistance Alerts", "Support & Resistance Table"])

    # Resistance Alerts Tab
    with tabs[0]:
        st.subheader("Real-Time Resistance Alerts")

        # Auto-Scan Logic
        current_time = time.time()
        auto_scan_interval_seconds = st.session_state['telegram_config']['auto_scan_interval'] * 60
        time_since_last_scan = current_time - st.session_state['last_scan_time']
        
        print(f"time_since_last_scan-{time_since_last_scan}")
        print(f"auto_scan_interval_seconds-{auto_scan_interval_seconds}")
        if time_since_last_scan >= auto_scan_interval_seconds:
            tickers = load_tickers()
            suggestions = check_resistance_and_notify(
                tickers, expiry, telegram_bot_token, telegram_chat_id,
                st.session_state['telegram_config']['proximity_to_resistance']
            )
            st.session_state['suggestions'] = suggestions
            st.session_state['last_scan_time'] = current_time
            st.rerun()

        # Manual Scan Button
        if st.button("Scan Now"):
            tickers = load_tickers()
            suggestions = check_resistance_and_notify(
                tickers, expiry, telegram_bot_token, telegram_chat_id,
                st.session_state['telegram_config']['proximity_to_resistance']
            )
            st.session_state['suggestions'] = suggestions

        # Display Results
        if st.session_state['suggestions'] is not None:
            if st.session_state['suggestions']:
                st.write("### Stocks Near Resistance")
                suggestions_df = pd.DataFrame(st.session_state['suggestions'])
                styled_df = suggestions_df.style.format({
                    'Underlying': '{:.2f}',
                    'Resistance': '{:.2f}',
                    'Distance_to_Resistance': '{:.2f}'
                })
                st.table(styled_df)
            else:
                st.info("No stocks currently near strong resistance.")
        else:
            st.info("Click 'Scan Now' or wait for auto-scan to check for stocks near resistance.")

    # Support & Resistance Table Tab
    with tabs[1]:
        st.subheader("Support & Resistance Levels for All Stocks")

        if st.button("Refresh Table"):
            tickers = load_tickers()
            table_data = generate_support_resistance_table(tickers, expiry)
            # Data is saved to JSON in generate_support_resistance_table

        # Always display table from JSON with sortable columns
        table_data = load_table_data()
        if table_data:
            st.write("### Support & Resistance Table (Sortable)")
            table_df = pd.DataFrame(table_data)
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
                    "High_Volume_Gainer": st.column_config.TextColumn("High Volume Gainer")
                },
                use_container_width=True
            )
        else:
            st.info("No data available. Click 'Refresh Table' to load support and resistance data.")

if __name__ == "__main__":
    main()