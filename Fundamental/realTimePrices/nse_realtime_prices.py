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
from datetime import datetime, date, timedelta
import yfinance as yf
import numpy as np
import logging
import threading
import queue

# Queue to handle async tasks
async_queue = queue.Queue()

def run_async_task(coro):
    """Run an async coroutine in a separate thread and return the result."""
    result = []
    def run_in_thread():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result.append(loop.run_until_complete(coro))
        finally:
            loop.close()
    thread = threading.Thread(target=run_in_thread)
    thread.start()
    thread.join()
    return result[0] if result else None

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('resistance_screener.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
SELL_SUGGESTIONS_FILE = "sell_suggestions.json"
OPTIONS_DATA_FILE = "options_data.csv"

# Headers mimicking a browser
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/market-data/equity-derivatives-watch",
}

# Initial cookie setup
logger.info("Visiting homepage...")
response = scraper.get("https://www.nseindia.com/", headers=headers)
if response.status_code != 200:
    logger.error(f"Failed to load homepage: {response.status_code}")
    st.error(f"Failed to load NSE homepage: {response.status_code}")
    exit()

logger.info("Visiting derivatives page...")
scraper.get("https://www.nseindia.com/market-data/equity-derivatives-watch", headers=headers)
time.sleep(2)

# Initialize JSON files
def initialize_json_files():
    for file_path in [ALERTS_DATA_FILE, TEMP_TABLE_DATA_FILE, HISTORICAL_DATA_FILE, CALL_SUGGESTIONS_FILE, SELL_SUGGESTIONS_FILE]:
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                json.dump([], f)
            logger.info(f"Initialized empty {file_path}")

# Load/Save Telegram Config
def load_config() -> Dict:
    default_config = {
        "telegram_bot_token": "",
        "telegram_chat_id": "",
        "auto_scan_interval": 5,
        "proximity_to_resistance": 0.5,
        "premium_change_threshold": 100.0,
        "auto_running": False,
        "notification_types": [
            "Resistance Alert",
            "Resistance Crossed Alert",
            "Sell CE Option Suggestion Premium Threshold",
            "Call Option Suggestion (Most OTM)"
        ]
    }
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                content = f.read().strip()
                if content:
                    config = json.loads(content)
                    for key, value in default_config.items():
                        if key not in config:
                            config[key] = value
                    return config
                else:
                    logger.warning(f"{CONFIG_FILE} is empty, returning default config")
                    return default_config
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding {CONFIG_FILE}: {e}, returning default config")
            return default_config
        except Exception as e:
            logger.error(f"Error loading {CONFIG_FILE}: {e}, returning default config")
            return default_config
    logger.info(f"No config file found at {CONFIG_FILE}")
    return default_config

def save_config(config: Dict):
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)
        logger.info(f"Saved config to {CONFIG_FILE}")
    except Exception as e:
        logger.error(f"Error saving {CONFIG_FILE}: {e}")

# Load/Save Table Data to JSON
def load_table_data() -> List[Dict]:
    if os.path.exists(TEMP_TABLE_DATA_FILE):
        try:
            with open(TEMP_TABLE_DATA_FILE, 'r') as f:
                content = f.read().strip()
                if content:
                    return json.loads(content)
                else:
                    logger.warning(f"{TEMP_TABLE_DATA_FILE} is empty, returning empty list")
                    return []
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding {TEMP_TABLE_DATA_FILE}: {e}, returning empty list")
            return []
        except Exception as e:
            logger.error(f"Error loading {TEMP_TABLE_DATA_FILE}: {e}, returning empty list")
            return []
    logger.info(f"No table data file found at {TEMP_TABLE_DATA_FILE}")
    return []

def save_table_data(data: List[Dict]):
    try:
        with open(TEMP_TABLE_DATA_FILE, 'w') as f:
            json.dump(data, f, indent=4)
        logger.info(f"Saved {len(data)} table entries to {TEMP_TABLE_DATA_FILE}")
    except Exception as e:
        logger.error(f"Error saving {TEMP_TABLE_DATA_FILE}: {e}")

# Load/Save Alerts Data to JSON
def load_alerts_data() -> List[Dict]:
    if os.path.exists(ALERTS_DATA_FILE):
        try:
            with open(ALERTS_DATA_FILE, 'r') as f:
                content = f.read().strip()
                if content:
                    return json.loads(content)
                else:
                    logger.warning(f"{ALERTS_DATA_FILE} is empty, returning empty list")
                    return []
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding {ALERTS_DATA_FILE}: {e}, returning empty list")
            return []
        except Exception as e:
            logger.error(f"Error loading {ALERTS_DATA_FILE}: {e}, returning empty list")
            return []
    logger.info(f"No alerts data file found at {ALERTS_DATA_FILE}")
    return []

def save_alerts_data(data: List[Dict]):
    try:
        with open(ALERTS_DATA_FILE, 'w') as f:
            json.dump(data, f, indent=4)
        logger.info(f"Saved {len(data)} alert entries to {ALERTS_DATA_FILE}")
    except Exception as e:
        logger.error(f"Error saving {ALERTS_DATA_FILE}: {e}")

# Load/Save Historical Data to JSON
def load_historical_data() -> List[Dict]:
    if os.path.exists(HISTORICAL_DATA_FILE):
        try:
            with open(HISTORICAL_DATA_FILE, 'r') as f:
                content = f.read().strip()
                if content:
                    data = json.loads(content)
                    logger.info(f"Loaded {len(data)} historical entries from {HISTORICAL_DATA_FILE}")
                    return data
                else:
                    logger.warning(f"{HISTORICAL_DATA_FILE} is empty, returning empty list")
                    return []
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding {HISTORICAL_DATA_FILE}: {e}, returning empty list")
            return []
        except Exception as e:
            logger.error(f"Error loading {HISTORICAL_DATA_FILE}: {e}, returning empty list")
            return []
    logger.info(f"No historical data file found at {HISTORICAL_DATA_FILE}")
    return []

def save_historical_data(data: List[Dict]):
    try:
        with open(HISTORICAL_DATA_FILE, 'w') as f:
            json.dump(data, f, indent=4)
        logger.info(f"Saved {len(data)} historical entries to {HISTORICAL_DATA_FILE}")
    except Exception as e:
        logger.error(f"Error saving {HISTORICAL_DATA_FILE}: {e}")

# Load/Save Call Suggestions to JSON
def load_call_suggestions() -> List[Dict]:
    if os.path.exists(CALL_SUGGESTIONS_FILE):
        try:
            with open(CALL_SUGGESTIONS_FILE, 'r') as f:
                content = f.read().strip()
                if content:
                    return json.loads(content)
                else:
                    logger.warning(f"{CALL_SUGGESTIONS_FILE} is empty, returning empty list")
                    return []
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding {CALL_SUGGESTIONS_FILE}: {e}, returning empty list")
            return []
        except Exception as e:
            logger.error(f"Error loading {CALL_SUGGESTIONS_FILE}: {e}, returning empty list")
            return []
    logger.info(f"No call suggestions file found at {CALL_SUGGESTIONS_FILE}")
    return []

def save_call_suggestions(data: List[Dict]):
    try:
        with open(CALL_SUGGESTIONS_FILE, 'w') as f:
            json.dump(data, f, indent=4)
        logger.info(f"Saved {len(data)} call suggestions to {CALL_SUGGESTIONS_FILE}")
    except Exception as e:
        logger.error(f"Error saving {CALL_SUGGESTIONS_FILE}: {e}")

# Load/Save Sell Suggestions to JSON
def load_sell_suggestions() -> List[Dict]:
    if os.path.exists(SELL_SUGGESTIONS_FILE):
        try:
            with open(SELL_SUGGESTIONS_FILE, 'r') as f:
                content = f.read().strip()
                if content:
                    return json.loads(content)
                else:
                    logger.warning(f"{SELL_SUGGESTIONS_FILE} is empty, returning empty list")
                    return []
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding {SELL_SUGGESTIONS_FILE}: {e}, returning empty list")
            return []
        except Exception as e:
            logger.error(f"Error loading {SELL_SUGGESTIONS_FILE}: {e}, returning empty list")
            return []
    logger.info(f"No sell suggestions file found at {SELL_SUGGESTIONS_FILE}")
    return []

def save_sell_suggestions(data: List[Dict]):
    try:
        with open(SELL_SUGGESTIONS_FILE, 'w') as f:
            json.dump(data, f, indent=4)
        logger.info(f"Saved {len(data)} sell suggestions to {SELL_SUGGESTIONS_FILE}")
    except Exception as e:
        logger.error(f"Error saving {SELL_SUGGESTIONS_FILE}: {e}")

# Load Options Data
def load_options_data() -> pd.DataFrame:
    if os.path.exists(OPTIONS_DATA_FILE):
        try:
            df = pd.read_csv(OPTIONS_DATA_FILE)
            required_columns = ['TICKER', 'EXPIRY', 'CALL TYPE', 'STRIKE_PRICE', 'PREVIOUS_PRICE']
            if all(col in df.columns for col in required_columns):
                df['CALL TYPE'] = df['CALL TYPE'].str.upper()
                df = df[df['CALL TYPE'].isin(['CE', 'PE'])]
                logger.info(f"Loaded options data from {OPTIONS_DATA_FILE} with {len(df)} rows")
                return df
            else:
                logger.warning(f"Options data at {OPTIONS_DATA_FILE} missing required columns, returning empty DataFrame")
                return pd.DataFrame(columns=required_columns)
        except Exception as e:
            logger.error(f"Error loading {OPTIONS_DATA_FILE}: {e}, returning empty DataFrame")
            return pd.DataFrame(columns=['TICKER', 'EXPIRY', 'CALL TYPE', 'STRIKE_PRICE', 'PREVIOUS_PRICE'])
    logger.info(f"No options data file found at {OPTIONS_DATA_FILE}")
    return pd.DataFrame(columns=['TICKER', 'EXPIRY', 'CALL TYPE', 'STRIKE_PRICE', 'PREVIOUS_PRICE'])

# Telegram Integration
async def send_telegram_message(bot_token: str, chat_id: str, message: str):
    if not bot_token or not chat_id:
        st.error("Telegram Bot Token or Chat ID is missing. Please configure them in the sidebar.")
        logger.error("Telegram Bot Token or Chat ID is missing")
        return
    
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
    logger.info(f"Sending Telegram message to chat {chat_id}: {message[:100]}... (length: {len(message)})")
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as response:
            response_text = await response.text()
            if response.status != 200:
                logger.error(f"Failed to send Telegram message: {response.status} - {response_text}")
                st.error(f"Failed to send Telegram message: {response_text}")
            else:
                logger.info(f"Telegram response: {response.status} - {response_text}")
                st.success(f"Telegram message sent successfully (length: {len(message)})")

# Fetch options data from NSE
def fetch_options_data(symbol: str, _refresh_key: float) -> Optional[Dict]:
    url = f"{BASE_URL}/api/option-chain-equities?symbol={symbol}"
    logger.info(f"Fetching options data from: {url}")
    
    response = scraper.get(url, headers=headers)
    if response.status_code != 200:
        logger.error(f"Failed to load options chain for {symbol}: {response.status_code}")
        return None
    
    data = response.json()
    return data

# Process Option Data
def process_option_data(data: Dict, expiry: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not data or 'records' not in data or 'data' not in data['records']:
        logger.error("Invalid data structure received from API")
        return pd.DataFrame(columns=['Strike', 'OI', 'Volume', 'Last Price', 'Previous Price']), pd.DataFrame(columns=['Strike', 'OI', 'Volume'])
    
    options = [item for item in data['records']['data'] if item.get('expiryDate') == expiry]
    if not options:
        logger.warning(f"No options found for expiry {expiry}")
        return pd.DataFrame(columns=['Strike', 'OI', 'Volume', 'Last Price', 'Previous Price']), pd.DataFrame(columns=['Strike', 'OI', 'Volume'])
    
    strikes = sorted({item['strikePrice'] for item in options})
    call_data = {s: {'OI': 0, 'Volume': 0, 'Last Price': 0, 'Previous Price': 0} for s in strikes}
    put_data = {s: {'OI': 0, 'Volume': 0, 'Last Price': 0, 'Previous Price': 0} for s in strikes}
    
    for item in options:
        strike = item['strikePrice']        
        if 'CE' in item:
            ce = item['CE']
            call_data[strike] = {
                'OI': ce.get('openInterest', 0),
                'Volume': ce.get('totalTradedVolume', 0),
                'Last Price': ce.get('lastPrice', 0),
                'Previous Price': ce.get('pclose', 0)
            }
        if 'PE' in item:
            pe = item['PE']
            put_data[strike] = {
                'OI': pe.get('openInterest', 0),
                'Volume': pe.get('totalTradedVolume', 0),
                'Last Price': pe.get('lastPrice', 0),
                'Previous Price': pe.get('pclose', 0)
            }
    
    call_df = pd.DataFrame([{'Strike': k, **v} for k, v in call_data.items()])
    put_df = pd.DataFrame([{'Strike': k, **v} for k, v in put_data.items()])
    
    required_columns = ['Strike', 'OI', 'Volume', 'Last Price', 'Previous Price']
    for col in required_columns:
        if col not in call_df.columns:
            call_df[col] = 0
        if col not in put_df.columns:
            put_df[col] = 0
    
    return call_df, put_df

# Identify Support and Resistance
def identify_support_resistance(call_df: pd.DataFrame, put_df: pd.DataFrame, top_n: int = 3) -> Tuple[Optional[float], Optional[float]]:
    resistance_strike = None
    support_strike = None
    
    if not call_df.empty and call_df['OI'].sum() > 0 and call_df.get('Volume', pd.Series()).sum() > 0:
        call_df['Weighted_Score'] = call_df['OI'] * call_df.get('Volume', pd.Series(0))
        top_calls = call_df.nlargest(top_n, 'Weighted_Score')
        resistance_strike = top_calls['Strike'].mean()
    
    if not put_df.empty and put_df['OI'].sum() > 0 and put_df.get('Volume', pd.Series()).sum() > 0:
        put_df['Weighted_Score'] = put_df['OI'] * put_df.get('Volume', pd.Series(0))
        top_puts = put_df.nlargest(top_n, 'Weighted_Score')
        support_strike = top_puts['Strike'].mean()
    
    return support_strike, resistance_strike

# Fetch real-time prices using yfinance
def fetch_realtime_prices(tickers: List[str]) -> Dict[str, Optional[float]]:
    prices = {}
    try:
        yf_tickers = yf.Tickers([f"{t}.NS" for t in tickers])
        for ticker in tickers:
            try:
                stock = yf_tickers.tickers.get(f"{ticker}.NS")
                if stock is None:
                    logger.error(f"Could not fetch data for {ticker}.NS")
                    prices[ticker] = None
                    continue
                info = stock.info
                current_price = info.get('regularMarketPrice', None)
                prices[ticker] = current_price
                logger.info(f"Fetched price for {ticker}: {current_price}")
            except Exception as e:
                logger.error(f"Error fetching price for {ticker}: {e}")
                prices[ticker] = None
    except Exception as e:
        logger.error(f"Error fetching prices: {e}")
        for ticker in tickers:
            prices[ticker] = None
    return prices

# Check for resistance crossing and notify
def check_resistance_crossing(tickers: List[str], expiry: str, bot_token: str, chat_id: str, fetched_data: Dict[str, Optional[Dict]] = None) -> List[Dict]:
    logger.info(f"Checking resistance crossing for tickers: {tickers}")
    crossing_alerts = []
    
    # Fetch real-time prices
    prices = fetch_realtime_prices(tickers)
    
    for ticker in tickers:
        try:
            with st.spinner(f"Checking {ticker} for resistance crossing..."):
                # Fetch NSE data for support/resistance
                data = fetched_data.get(ticker) if fetched_data is not None else fetch_options_data(ticker, time.time())
                if not data or 'records' not in data:
                    logger.error(f"Failed to fetch NSE data for {ticker}")
                    continue
                
                call_df, put_df = process_option_data(data, expiry)
                support_strike, resistance_strike = identify_support_resistance(call_df, put_df)
                
                if resistance_strike is None:
                    logger.warning(f"No resistance strike identified for {ticker}")
                    continue
                
                current_price = prices.get(ticker)
                if current_price is None:
                    logger.warning(f"No valid price fetched for {ticker}")
                    continue
                
                # Check if current price crosses resistance
                if current_price > resistance_strike:
                    message = (
                        f"🚨 *Resistance Crossed Alert* 🚨\n"
                        f"Stock: *{ticker}*\n"
                        f"Current Price: *₹{current_price:.2f}*\n"
                        f"Resistance: *₹{resistance_strike:.2f}*\n"
                        f"Crossed By: *₹{(current_price - resistance_strike):.2f}*\n"
                        f"Timestamp: *{time.strftime('%Y-%m-%d %H:%M:%S')}*\n"
                        f"---"
                    )
                    crossing_alerts.append({
                        "Ticker": ticker,
                        "Current_Price": current_price,
                        "Resistance": resistance_strike,
                        "Crossed_By": current_price - resistance_strike,
                        "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    })
                    logger.info(f"Resistance crossed for {ticker}: Price={current_price}, Resistance={resistance_strike}")
                    if bot_token and chat_id:
                        run_async_task(send_telegram_message(bot_token, chat_id, message))
        
        except Exception as e:
            logger.error(f"Error checking resistance crossing for {ticker}: {e}")
            continue
    
    return crossing_alerts

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

# Generate Support/Resistance Table Data and Save to JSON
def generate_support_resistance_table(tickers: List[str], expiry: str) -> List[Dict]:
    refresh_key = time.time()
    table_data = []
    volume_threshold = 100000
    
    prices = fetch_realtime_prices(tickers)  # Use yfinance for real-time prices
    
    for ticker in tickers:
        with st.spinner(f"Fetching data for {ticker}..."):
            data = fetch_options_data(ticker, refresh_key)
            if not data or 'records' not in data:
                logger.error(f"Failed to fetch data for {ticker}")
                continue
            
            call_df, put_df = process_option_data(data, expiry)
            support_strike, resistance_strike = identify_support_resistance(call_df, put_df)
            underlying = prices.get(ticker, None)  # Use yfinance price
            
            total_volume = call_df.get('Volume', pd.Series(0)).sum() + put_df.get('Volume', pd.Series(0)).sum()
            high_volume_gainer = "Yes" if total_volume > volume_threshold else "No"
            
            distance_from_resistance = resistance_strike - underlying if resistance_strike and underlying is not None else None
            distance_from_support = underlying - support_strike if support_strike and underlying is not None else None
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

def check_resistance_and_notify(tickers: List[str], expiry: str, bot_token: str, chat_id: str, proximity_percent: float, premium_threshold: float, options_df: pd.DataFrame, notification_types: List[str], fetched_data: Dict[str, Optional[Dict]] = None):
    logger.info(f"Starting check_resistance_and_notify for tickers: {tickers}, expiry: {expiry}")
    suggestions = []
    call_suggestions = []
    sell_suggestions = []
    proximity_tickers = []
    resistance_alerts = []
    resistance_crossed_alerts = []
    sell_option_suggestions = []
    call_option_suggestions = []
    
    for ticker in tickers:
        try:
            with st.spinner(f"Processing {ticker}..."):
                data = fetched_data.get(ticker) if fetched_data is not None else fetch_options_data(ticker, time.time())
                if not data or 'records' not in data:
                    logger.error(f"Failed to fetch or retrieve data for {ticker}")
                    continue
                
                logger.info(f"Processing {ticker}: underlyingValue={data['records'].get('underlyingValue', 'N/A')}")
                call_df, put_df = process_option_data(data, expiry)
                underlying = data['records'].get('underlyingValue', 0)
                resistance_strike = identify_support_resistance(call_df, put_df)[1]
                
                if resistance_strike is None:
                    logger.warning(f"No resistance strike identified for {ticker}")
                    continue
                
                proximity_threshold = abs(proximity_percent) / 100.0
                logger.info(f"{ticker}: proximity_percent={proximity_percent}, threshold={proximity_threshold}")
                
                meets_proximity = False
                if proximity_percent >= 0:
                    distance_to_resistance = resistance_strike - underlying
                    if 0 <= distance_to_resistance <= (resistance_strike * proximity_threshold):
                        meets_proximity = True
                        if "Resistance Alert" in notification_types:
                            message = (
                                f"Stock: *{ticker}*\n"
                                f"Underlying: *₹{underlying:.2f}*\n"
                                f"Resistance: *₹{resistance_strike:.2f}*\n"
                                f"Distance: *{distance_to_resistance:.2f}*\n"
                                f"Reason: *Within {proximity_percent}% of strong resistance*\n"
                                f"---"
                            )
                            resistance_alerts.append(message)
                            logger.info(f"Resistance alert for {ticker}: distance={distance_to_resistance}")
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
                        meets_proximity = True
                        if "Resistance Crossed Alert" in notification_types:
                            message = (
                                f"Stock: *{ticker}*\n"
                                f"Underlying: *₹{underlying:.2f}*\n"
                                f"Resistance: *₹{resistance_strike:.2f}*\n"
                                f"Crossed By: *{distance_to_resistance:.2f}*\n"
                                f"Reason: *Crossed resistance by {abs(proximity_percent)}%*\n"
                                f"---"
                            )
                            resistance_crossed_alerts.append(message)
                            logger.info(f"Resistance crossed alert for {ticker}: crossed by={distance_to_resistance}")
                        suggestions.append({
                            "Ticker": ticker,
                            "Underlying": underlying,
                            "Resistance": resistance_strike,
                            "Distance_to_Resistance": -distance_to_resistance,
                            "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                        })

                if meets_proximity:
                    proximity_tickers.append(ticker)
                    if "Sell CE Option Suggestion Premium Threshold" in notification_types:
                        sell_suggestion_list = suggest_options_for_selling(data, expiry, underlying, ticker, premium_threshold, options_df)
                        for sell_suggestion in sell_suggestion_list:
                            sell_message = (
                                f"Stock: *{ticker}*\n"
                                f"Underlying: *₹{underlying:.2f}*\n"
                                f"Suggested Sell Strike: *₹{sell_suggestion['Suggested_Sell_Strike']:.2f}*\n"
                                f"Current Premium: *₹{sell_suggestion['Current_Premium']:.2f}*\n"
                                f"Previous Premium: *₹{sell_suggestion['Previous_Premium']:.2f}*\n"
                                f"Premium Change: *{sell_suggestion['Premium_Change_%']:.2f}%*\n"
                                f"Expiry: *{expiry}*\n"
                                f"---"
                            )
                            sell_option_suggestions.append(sell_message)
                            sell_suggestions.append(sell_suggestion)
                            logger.info(f"Sell suggestion for {ticker}: strike={sell_suggestion['Suggested_Sell_Strike']}, premium_change={sell_suggestion['Premium_Change_%']}%")

                if meets_proximity and (
                    (proximity_percent >= 0 and 0 <= (resistance_strike - underlying) <= (resistance_strike * proximity_threshold)) or
                    (proximity_percent < 0 and (underlying - resistance_strike) > 0 and (underlying - resistance_strike) <= (resistance_strike * abs(proximity_threshold)))
                ):
                    if "Call Option Suggestion (Most OTM)" in notification_types:
                        call_suggestion = suggest_call_options(data, expiry, underlying, ticker)
                        if call_suggestion and call_suggestion['Potential_Gain_%'] > 0:
                            call_message = (
                                f"Stock: *{ticker}*\n"
                                f"Underlying: *₹{underlying:.2f}*\n"
                            f"Suggested Call Strike: *₹{call_suggestion['Suggested_Call_Strike']:.2f}* (Most OTM)\n"
                                f"Call Last Price: *₹{call_suggestion['Call_Last_Price']:.2f}*\n"
                                f"Potential Gain: *{call_suggestion['Potential_Gain_%']:.2f}%*\n"
                                f"Expiry: *{expiry}*\n"
                                f"---"
                            )
                            call_option_suggestions.append(call_message)
                            call_suggestions.append(call_suggestion)
                            logger.info(f"Call suggestion for {ticker}: strike={call_suggestion['Suggested_Call_Strike']}, gain={call_suggestion['Potential_Gain_%']}%")
        except Exception as e:
            logger.error(f"Error processing ticker {ticker}: {e}")
            continue
    
    # Combine and send notifications
    combined_message = f"*Resistance Screener Update* - {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    has_content = False
    
    if resistance_alerts:
        combined_message += "*Resistance Alerts*\n"
        combined_message += "\n".join(resistance_alerts) + "\n"
        has_content = True
    
    if resistance_crossed_alerts:
        combined_message += "*Resistance Crossed Alerts*\n"
        combined_message += "\n".join(resistance_crossed_alerts) + "\n"
        has_content = True
    
    if sell_option_suggestions:
        combined_message += "*Sell CE Option Suggestions (Premium Threshold)*\n"
        combined_message += "\n".join(sell_option_suggestions) + "\n"
        has_content = True
    
    if call_option_suggestions:
        combined_message += "*Call Option Suggestions (Most OTM)*\n"
        combined_message += "\n".join(call_option_suggestions) + "\n"
        has_content = True
    
    if has_content:
        try:
            if len(combined_message) > 4096:
                logger.warning(f"Combined message too long ({len(combined_message)} chars), splitting...")
                chunks = []
                current_chunk = f"*Resistance Screener Update* - {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                current_length = len(current_chunk)
                
                for section in [resistance_alerts, resistance_crossed_alerts, sell_option_suggestions, call_option_suggestions]:
                    section_header = ""
                    if section == resistance_alerts and section:
                        section_header = "*Resistance Alerts*\n"
                    elif section == resistance_crossed_alerts and section:
                        section_header = "*Resistance Crossed Alerts*\n"
                    elif section == sell_option_suggestions and section:
                        section_header = "*Sell CE Option Suggestions (Premium Threshold)*\n"
                    elif section == call_option_suggestions and section:
                        section_header = "*Call Option Suggestions (Most OTM)*\n"
                    
                    if section_header:
                        if current_length + len(section_header) > 4000:
                            chunks.append(current_chunk)
                            current_chunk = f"*Resistance Screener Update (Part {len(chunks) + 1})* - {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                            current_length = len(current_chunk)
                        current_chunk += section_header
                        current_length += len(section_header)
                    
                    for msg in section:
                        if current_length + len(msg) > 4000:
                            chunks.append(current_chunk)
                            current_chunk = f"*Resistance Screener Update (Part {len(chunks) + 1})* - {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                            current_length = len(current_chunk)
                        current_chunk += msg + "\n"
                        current_length += len(msg) + 1
                
                if current_chunk.strip():
                    chunks.append(current_chunk)
                
                for i, chunk in enumerate(chunks, 1):
                    logger.info(f"Sending message chunk {i}/{len(chunks)} (length: {len(chunk)})")
                    run_async_task(send_telegram_message(bot_token, chat_id, chunk))
            else:
                logger.info(f"Sending combined message (length: {len(combined_message)})")
                run_async_task(send_telegram_message(bot_token, chat_id, combined_message))
        except Exception as e:
            logger.error(f"Error sending combined Telegram message: {e}")
            st.error(f"Failed to send combined Telegram message: {e}")
    else:
        logger.info("No alerts or suggestions to send in combined message")
    
    logger.info(f"Tickers meeting proximity criteria: {proximity_tickers}")
    return suggestions, call_suggestions, sell_suggestions

# Main Application
def main():
    initialize_json_files()
    st.set_page_config(page_title="Real-Time Resistance Screener", layout="wide")
    st.title("Real-Time NSE Resistance Screener with yfinance Prices")

    config = load_config()
    if 'telegram_config' not in st.session_state:
        st.session_state['telegram_config'] = config

    if 'last_scan_time' not in st.session_state:
        st.session_state['last_scan_time'] = time.time()
    if 'last_support_resistance_update' not in st.session_state:
        st.session_state['last_support_resistance_update'] = time.time()
    if 'refresh_key' not in st.session_state:
        st.session_state['refresh_key'] = time.time()
    if 'suggestions' not in st.session_state:
        st.session_state['suggestions'] = load_alerts_data()
    if 'call_suggestions' not in st.session_state:
        st.session_state['call_suggestions'] = load_call_suggestions()
    if 'sell_suggestions' not in st.session_state:
        st.session_state['sell_suggestions'] = load_sell_suggestions()
    if 'table_data' not in st.session_state:
        st.session_state['table_data'] = load_table_data()
    if 'historical_data' not in st.session_state:
        st.session_state['historical_data'] = load_historical_data()
    if 'crossing_alerts' not in st.session_state:
        st.session_state['crossing_alerts'] = []
    if 'auto_running' not in st.session_state:
        st.session_state['auto_running'] = st.session_state['telegram_config']['auto_running']
    if 'last_auto_time' not in st.session_state:
        st.session_state['last_auto_time'] = 0
    if 'options_data' not in st.session_state:  # Add this initialization
        st.session_state['options_data'] = load_options_data()


    with st.sidebar:
        st.subheader("Auto Scan Control")
        if st.button("Toggle Auto (60s for Prices, 15min for S/R)", key="sidebar_auto_toggle"):
            st.session_state['auto_running'] = not st.session_state['auto_running']
            st.session_state['telegram_config']['auto_running'] = st.session_state['auto_running']
            save_config(st.session_state['telegram_config'])
            if st.session_state['auto_running']:
                st.session_state['last_auto_time'] = time.time()
                st.session_state['last_support_resistance_update'] = time.time()
            st.rerun()
        status = "Running" if st.session_state['auto_running'] else "Stopped"
        st.write(f"Auto Scan (60s/15min): {status}")

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
        
        st.subheader("Notification Types")
        notification_types = st.multiselect(
            "Select Notification Types:",
            options=[
                "Resistance Alert",
                "Resistance Crossed Alert",
                "Sell CE Option Suggestion Premium Threshold",
                "Call Option Suggestion (Most OTM)"
            ],
            default=st.session_state['telegram_config']['notification_types'],
            key="notification_types"
        )
        
        st.subheader("Scan Settings")
        auto_scan_interval = st.number_input(
            "Auto-Scan Interval for Prices (seconds):",
            value=60,
            min_value=30,
            step=10,
            key="auto_scan_interval"
        )
        support_resistance_interval = st.number_input(
            "Support/Resistance Update Interval (minutes):",
            value=15,
            min_value=5,
            step=5,
            key="support_resistance_interval"
        )
        proximity_to_resistance = st.number_input(
            "Proximity to Resistance (%):",
            value=st.session_state['telegram_config']['proximity_to_resistance'],
            step=0.1,
            min_value=-5.0,
            max_value=10.0,
            key="proximity_to_resistance_input"
        )
        premium_change_threshold = st.number_input(
            "Premium Change Threshold (%):",
            value=st.session_state['telegram_config']['premium_change_threshold'],
            min_value=-1000.0,
            max_value=100000.0,
            step=10.0,
            key="premium_change_threshold_input"
        )

        st.subheader("Upload Tickers")
        uploaded_ticker_file = st.file_uploader("Upload CSV with 'SYMBOL' column", type=["csv"], key="ticker_upload")
        if uploaded_ticker_file:
            df = pd.read_csv(uploaded_ticker_file)
            if 'SYMBOL' in df.columns:
                df.to_csv(STORED_TICKERS_PATH, index=False)
                st.success(f"Tickers saved to {STORED_TICKERS_PATH}")
                st.session_state['refresh_key'] = time.time()
            else:
                st.error("CSV must contain 'SYMBOL' column")

        st.subheader("Upload Options Data")
        uploaded_options_file = st.file_uploader("Upload Options CSV (TICKER, EXPIRY, CALL TYPE, STRIKE_PRICE, PREVIOUS_PRICE)", type=["csv"], key="options_upload")
        if uploaded_options_file:
            df = pd.read_csv(uploaded_options_file)
            required_columns = ['TICKER', 'EXPIRY', 'CALL TYPE', 'STRIKE_PRICE', 'PREVIOUS_PRICE']
            if all(col in df.columns for col in required_columns):
                df['CALL TYPE'] = df['CALL TYPE'].str.upper()
                df = df[df['CALL TYPE'].isin(['CE', 'PE'])]
                df.to_csv(OPTIONS_DATA_FILE, index=False)
                st.session_state['options_data'] = df
                st.success(f"Options data saved to {OPTIONS_DATA_FILE}")
            else:
                st.error(f"Options CSV must contain columns: {', '.join(required_columns)}")
        elif st.session_state['options_data'].empty and os.path.exists(OPTIONS_DATA_FILE):
            st.session_state['options_data'] = load_options_data()

        st.subheader("Scan Specific Stocks")
        specific_tickers = st.text_input("Enter tickers (comma-separated, e.g., HDFCBANK,RELIANCE):", key="specific_tickers")

        config_changed = False
        if st.session_state['telegram_config']['telegram_bot_token'] != telegram_bot_token:
            st.session_state['telegram_config']['telegram_bot_token'] = telegram_bot_token
            config_changed = True
        if st.session_state['telegram_config']['telegram_chat_id'] != telegram_chat_id:
            st.session_state['telegram_config']['telegram_chat_id'] = telegram_chat_id
            config_changed = True
        if st.session_state['telegram_config']['notification_types'] != notification_types:
            st.session_state['telegram_config']['notification_types'] = notification_types
            config_changed = True
        if st.session_state['telegram_config']['proximity_to_resistance'] != proximity_to_resistance:
            st.session_state['telegram_config']['proximity_to_resistance'] = proximity_to_resistance
            config_changed = True
        if st.session_state['telegram_config']['premium_change_threshold'] != premium_change_threshold:
            st.session_state['telegram_config']['premium_change_threshold'] = premium_change_threshold
            config_changed = True
        if config_changed:
            save_config(st.session_state['telegram_config'])

        if not telegram_bot_token or not telegram_chat_id:
            st.warning("Please configure Telegram Bot Token and Chat ID.")

    data = fetch_options_data("HDFCBANK", st.session_state['refresh_key'])
    if not data or 'records' not in data:
        st.error("Failed to load initial data!")
        return
    expiry_dates = data['records']['expiryDates']
    current_date = datetime.now().date()
    
    current_weekday = current_date.weekday()
    days_to_sunday = 6 - current_weekday if current_weekday != 6 else 0
    current_week_sunday = current_date + timedelta(days=days_to_sunday)
    
    first_expiry = datetime.strptime(expiry_dates[0], "%d-%b-%Y").date()
    
    if first_expiry <= current_week_sunday and len(expiry_dates) > 1:
        expiry = expiry_dates[1]
        st.info(f"Current expiry ({expiry_dates[0]}) is within this week. Using next expiry: {expiry}")
    else:
        expiry = expiry_dates[0]

    tabs = st.tabs(["Real-Time Resistance Alerts", "Support & Resistance Table", "Historical Scan Data"])

    with tabs[0]:
        st.subheader("Real-Time Resistance Alerts")
        current_time = time.time()
        auto_scan_interval_seconds = auto_scan_interval
        support_resistance_interval_seconds = support_resistance_interval * 60
        time_since_last_scan = current_time - st.session_state['last_scan_time']
        time_since_last_sr_update = current_time - st.session_state['last_support_resistance_update']
        time_to_next_scan = auto_scan_interval_seconds - time_since_last_scan
        time_to_next_sr_update = support_resistance_interval_seconds - time_since_last_sr_update
        
        st.write(f"Last Price Scan: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(st.session_state['last_scan_time']))}")
        st.write(f"Last Support/Resistance Update: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(st.session_state['last_support_resistance_update']))}")
        minutes_to_next_scan = int(time_to_next_scan // 60)
        seconds_to_next_scan = int(time_to_next_scan % 60)
        minutes_to_next_sr_update = int(time_to_next_sr_update // 60)
        seconds_to_next_sr_update = int(time_to_next_sr_update % 60)
        st.write(f"Next Price Scan in: {minutes_to_next_scan} minutes {seconds_to_next_scan} seconds")
        st.write(f"Next Support/Resistance Update in: {minutes_to_next_sr_update} minutes {seconds_to_next_sr_update} seconds")

        def perform_scan(tickers_to_scan):
            logger.info(f"Starting perform_scan for tickers: {tickers_to_scan}")
            refresh_key = time.time()
            fetched_data = {}
            
            # Fetch NSE data for support/resistance
            for ticker in tickers_to_scan:
                with st.spinner(f"Fetching NSE data for {ticker}..."):
                    data = fetch_options_data(ticker, refresh_key)
                    fetched_data[ticker] = data if data and 'records' in data else None
                    if not data or 'records' not in data:
                        logger.error(f"No data fetched for ticker: {ticker}")

            # Update support/resistance table
            st.session_state['table_data'] = generate_support_resistance_table(tickers_to_scan, expiry)

            # Check resistance crossing
            new_crossing_alerts = check_resistance_crossing(
                tickers_to_scan, expiry, telegram_bot_token, telegram_chat_id, fetched_data
            )
            st.session_state['crossing_alerts'].extend(new_crossing_alerts)
            
            # Perform original resistance and notification checks
            new_suggestions, new_call_suggestions, new_sell_suggestions = check_resistance_and_notify(
                tickers_to_scan, expiry, telegram_bot_token, telegram_chat_id,
                st.session_state['telegram_config']['proximity_to_resistance'],
                st.session_state['telegram_config']['premium_change_threshold'],
                st.session_state['options_data'],
                st.session_state['telegram_config']['notification_types'],
                fetched_data
            )
            
            st.session_state['suggestions'].extend(new_suggestions)
            st.session_state['call_suggestions'].extend(new_call_suggestions)
            st.session_state['sell_suggestions'].extend(new_sell_suggestions)
            save_alerts_data(st.session_state['suggestions'])
            save_call_suggestions(st.session_state['call_suggestions'])
            save_sell_suggestions(st.session_state['sell_suggestions'])

            logger.info(f"Scan complete. Crossing alerts: {len(new_crossing_alerts)}, Resistance alerts: {len(new_suggestions)}")

        if 'scan_in_progress' not in st.session_state:
            st.session_state['scan_in_progress'] = False    
        
        if st.session_state['auto_running'] and not st.session_state['scan_in_progress']:
            current_time = time.time()
            if current_time - st.session_state['last_auto_time'] >= auto_scan_interval_seconds:
                tickers = load_tickers() if not specific_tickers else [t.strip() for t in specific_tickers.split(',')]
                st.session_state['scan_in_progress'] = True
                if current_time - st.session_state['last_support_resistance_update'] >= support_resistance_interval_seconds:
                    perform_scan(tickers)
                    st.session_state['last_support_resistance_update'] = current_time
                else:
                    # Only check prices and resistance crossing
                    new_crossing_alerts = check_resistance_crossing(
                        tickers, expiry, telegram_bot_token, telegram_chat_id
                    )
                    st.session_state['crossing_alerts'].extend(new_crossing_alerts)
                st.session_state['last_scan_time'] = current_time
                st.session_state['last_auto_time'] = current_time
                st.session_state['scan_in_progress'] = False
                st.rerun()

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Scan All Tickers") and not st.session_state['scan_in_progress']:
                tickers = load_tickers()
                st.session_state['scan_in_progress'] = True
                perform_scan(tickers)
                st.session_state['last_scan_time'] = time.time()
                st.session_state['last_support_resistance_update'] = time.time()
                st.session_state['scan_in_progress'] = False
                st.session_state['needs_rerun'] = True
        with col2:
            if st.button("Scan Specific Tickers") and specific_tickers and not st.session_state['scan_in_progress']:
                tickers = [t.strip() for t in specific_tickers.split(',')]
                st.session_state['scan_in_progress'] = True
                perform_scan(tickers)
                st.session_state['last_scan_time'] = time.time()
                st.session_state['last_support_resistance_update'] = time.time()
                st.session_state['scan_in_progress'] = False
                st.session_state['needs_rerun'] = True

        if 'needs_rerun' in st.session_state and st.session_state['needs_rerun']:
            st.session_state['needs_rerun'] = False
            st.rerun()

        status = "Running" if st.session_state['auto_running'] else "Stopped"
        st.write(f"Auto Scan (60s/15min): {status}")

        # Display resistance crossing alerts
        if st.session_state['crossing_alerts']:
            st.write("### Stocks Crossing Resistance (Real-Time)")
            crossing_search_query = st.text_input("Search Crossing Alerts by Ticker", key="crossing_search")
            crossing_df = pd.DataFrame(st.session_state['crossing_alerts'])
            
            if crossing_search_query:
                crossing_df = crossing_df[crossing_df['Ticker'].str.contains(crossing_search_query, case=False, na=False)]
            
            st.dataframe(
                crossing_df,
                column_config={
                    "Ticker": st.column_config.TextColumn("Ticker"),
                    "Current_Price": st.column_config.NumberColumn("Current Price", format="%.2f"),
                    "Resistance": st.column_config.NumberColumn("Resistance", format="%.2f"),
                    "Crossed_By": st.column_config.NumberColumn("Crossed By", format="%.2f"),
                    "Timestamp": st.column_config.TextColumn("Timestamp")
                },
                use_container_width=True,
                height=400
            )
            
            if st.button("Clear Crossing Alerts"):
                st.session_state['crossing_alerts'] = []
                st.session_state['needs_rerun'] = True
        else:
            st.info("No stocks have crossed resistance in real-time.")

        # Display scanned stocks
        if st.session_state['table_data']:
            st.write("### Scanned Stocks (Sortable & Searchable)")
            search_query = st.text_input("Search Scanned Stocks by Ticker", key="scanned_search")
            scanned_df = pd.DataFrame(st.session_state['table_data'])
            
            if search_query:
                scanned_df = scanned_df[scanned_df['Ticker'].str.contains(search_query, case=False, na=False)]
            
            st.dataframe(
                scanned_df,
                column_config={
                    "Ticker": st.column_config.TextColumn("Ticker"),
                    "Underlying": st.column_config.NumberColumn("Underlying", format="%.2f"),
                    "Resistance": st.column_config.NumberColumn("Resistance", format="%.2f"),
                    "Support": st.column_config.NumberColumn("Support", format="%.2f"),
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
            st.info("No scanned stocks available. Run a scan to populate data.")

        # Original alerts
        if st.session_state['suggestions']:
            st.write("### Stocks Near Resistance (Alerts)")
            alert_search_query = st.text_input("Search Alerts by Ticker", key="alerts_search")
            suggestions_df = pd.DataFrame(st.session_state['suggestions'])
            
            if alert_search_query:
                suggestions_df = suggestions_df[suggestions_df['Ticker'].str.contains(alert_search_query, case=False, na=False)]
            
            st.dataframe(
                suggestions_df,
                column_config={
                    "Ticker": st.column_config.TextColumn("Ticker"),
                    "Underlying": st.column_config.NumberColumn("Underlying", format="%.2f"),
                    "Resistance": st.column_config.NumberColumn("Resistance", format="%.2f"),
                    "Distance_to_Resistance": st.column_config.NumberColumn("Distance to Resistance", format="%.2f"),
                    "Timestamp": st.column_config.TextColumn("Timestamp")
                },
                use_container_width=True,
                height=400
            )
            
            if st.button("Clear Alerts"):
                st.session_state['suggestions'] = []
                save_alerts_data(st.session_state['suggestions'])
                st.session_state['needs_rerun'] = True

        # Call suggestions
        if st.session_state['call_suggestions']:
            st.write("### Suggested Call Options (OTM with maximum price Above Spot)")
            call_search_query = st.text_input("Search Call Suggestions by Ticker", key="call_search")
            call_suggestions_df = pd.DataFrame(st.session_state['call_suggestions'])
            
            if call_search_query:
                call_suggestions_df = call_suggestions_df[call_suggestions_df['Ticker'].str.contains(call_search_query, case=False, na=False)]
            
            st.dataframe(
                call_suggestions_df,
                column_config={
                    "Ticker": st.column_config.TextColumn("Ticker"),
                    "Underlying": st.column_config.NumberColumn("Underlying", format="%.2f"),
                    "Suggested_Call_Strike": st.column_config.NumberColumn("Suggested Call Strike", format="%.2f"),
                    "Call_Last_Price": st.column_config.NumberColumn("Call Last Price", format="%.2f"),
                    "Potential_Gain_%": st.column_config.NumberColumn("Potential Gain %", format="%.2f"),
                    "Expiry": st.column_config.TextColumn("Expiry"),
                    "Timestamp": st.column_config.TextColumn("Timestamp")
                },
                use_container_width=True,
                height=400
            )
            
            if st.button("Clear Call Suggestions"):
                st.session_state['call_suggestions'] = []
                save_call_suggestions(st.session_state['call_suggestions'])
                st.session_state['needs_rerun'] = True

        # Sell suggestions
        if st.session_state['sell_suggestions']:
            st.write("### Suggested Options for Selling (High Premium Change)")
            sell_search_query = st.text_input("Search Sell Suggestions by Ticker", key="sell_search")
            sell_suggestions_df = pd.DataFrame(st.session_state['sell_suggestions'])
            
            if sell_search_query:
                sell_suggestions_df = sell_suggestions_df[sell_suggestions_df['Ticker'].str.contains(sell_search_query, case=False, na=False)]
            
            st.dataframe(
                sell_suggestions_df,
                column_config={
                    "Ticker": st.column_config.TextColumn("Ticker"),
                    "Underlying": st.column_config.NumberColumn("Underlying", format="%.2f"),
                    "Option_Type": st.column_config.TextColumn("Option Type"),
                    "Resistance": st.column_config.NumberColumn("Resistance", format="%.2f"),
                    "Support": st.column_config.NumberColumn("Support", format="%.2f"),
                    "Suggested_Sell_Strike": st.column_config.NumberColumn("Suggested Sell Strike", format="%.2f"),
                    "Current_Premium": st.column_config.NumberColumn("Current Premium", format="%.2f"),
                    "Previous_Premium": st.column_config.NumberColumn("Previous Premium", format="%.2f"),
                    "Premium_Change_%": st.column_config.NumberColumn("Premium Change %", format="%.2f"),
                    "Expiry": st.column_config.TextColumn("Expiry"),
                    "Timestamp": st.column_config.TextColumn("Timestamp")
                },
                use_container_width=True,
                height=400
            )
            
            if st.button("Clear Sell Suggestions"):
                st.session_state['sell_suggestions'] = []
                save_sell_suggestions(st.session_state['sell_suggestions'])
                st.session_state['needs_rerun'] = True

    with tabs[1]:
        st.subheader("Support & Resistance Levels for All Stocks")
        if st.button("Refresh Table"):
            tickers = load_tickers()
            table_data = generate_support_resistance_table(tickers, expiry)
            st.session_state['table_data'] = table_data
            st.session_state['last_support_resistance_update'] = time.time()
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

    with tabs[2]:
        st.subheader("Historical Scan Data")
        selected_date = st.date_input("Select Date", value=date.today() - pd.Timedelta(days=1))
        date_str = selected_date.strftime("%Y-%m-%d")
        folder_path = "historical_data_date_wise"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        historical_file = os.path.join(folder_path, f"historical_data_{date_str}.json")

        def load_date_specific_historical_data(file_path: str) -> List[Dict]:
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        content = f.read().strip()
                        if content:
                            data = json.loads(content)
                            logger.info(f"Loaded {len(data)} historical entries from {file_path}")
                            return data
                        else:
                            logger.warning(f"{file_path} is empty, returning empty list")
                            return []
                except json.JSONDecodeError as e:
                    logger.error(f"Error decoding {file_path}: {e}, returning empty list")
                    return []
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}, returning empty list")
                    return []
            logger.info(f"No historical data file found at {file_path}")
            return []

        def save_date_specific_historical_data(file_path: str, data: List[Dict]):
            try:
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=4)
                logger.info(f"Saved {len(data)} historical entries to {file_path}")
            except Exception as e:
                logger.error(f"Error saving {file_path}: {e}")

        if 'historical_data_date' not in st.session_state or st.session_state['historical_data_date'] != date_str:
            st.session_state['historical_data'] = load_date_specific_historical_data(historical_file)
            st.session_state['historical_data_date'] = date_str

        if st.button("Check Historical Resistance"):
            if not st.session_state['historical_data']:
                tickers = load_tickers()
                proximity_percent = st.session_state['telegram_config']['proximity_to_resistance']
                with st.spinner(f"Fetching historical data for {date_str}..."):
                    historical_results = check_historical_resistance(
                        tickers, selected_date, expiry, proximity_percent
                    )
                st.session_state['historical_data'] = historical_results
                save_date_specific_historical_data(historical_file, historical_results)
            if st.session_state['historical_data']:
                st.write("### Stocks that touched resistance on selected date")
                historical_df = pd.DataFrame(st.session_state['historical_data'])
                column_order = [
                    "Date", "Time", "Ticker", "High_Price", "Close_Price",
                    "Resistance_Price", "Distance_to_Resistance", "Volume", "Touched_Resistance",
                    "Selected_Date_Resistance", "Touched_Selected_Date_Resistance"
                ]
                historical_df = historical_df[column_order]
                st.dataframe(
                    historical_df,
                    column_config={
                        "Date": st.column_config.TextColumn("Date"),
                        "Time": st.column_config.TextColumn("Time"),
                        "Ticker": st.column_config.TextColumn("Ticker"),
                        "High_Price": st.column_config.NumberColumn("High Price", format="%.2f"),
                        "Close_Price": st.column_config.NumberColumn("Close Price", format="%.2f"),
                        "Resistance_Price": st.column_config.NumberColumn("Resistance Price (Current)", format="%.2f"),
                        "Distance_to_Resistance": st.column_config.NumberColumn("Distance to Resistance (Close)", format="%.2f"),
                        "Volume": st.column_config.NumberColumn("Volume", format="%.0f"),
                        "Touched_Resistance": st.column_config.TextColumn("Touched Resistance"),
                        "Selected_Date_Resistance": st.column_config.NumberColumn("Selected Date Resistance", format="%.2f"),
                        "Touched_Selected_Date_Resistance": st.column_config.TextColumn("Touched Selected Date Resistance")
                    },
                    use_container_width=True,
                    height=400
                )
                if st.button("Clear Historical Data"):
                    st.session_state['historical_data'] = []
                    if os.path.exists(historical_file):
                        os.remove(historical_file)
                        logger.info(f"Deleted historical data file: {historical_file}")
                    st.session_state['historical_data_date'] = None
                    st.rerun()
                download_csv(st.session_state['historical_data'], f"historical_resistance_{selected_date}.csv")
            else:
                st.info("No stocks touched resistance on the selected date.")
        else:
            if st.session_state['historical_data']:
                st.write("### Last Historical Scan Results")
                historical_df = pd.DataFrame(st.session_state['historical_data'])
                column_order = [
                    "Date", "Time", "Ticker", "High_Price", "Close_Price",
                    "Resistance_Price", "Distance_to_Resistance", "Volume", "Touched_Resistance",
                    "Selected_Date_Resistance", "Touched_Selected_Date_Resistance"
                ]
                historical_df = historical_df[column_order]
                st.dataframe(
                    historical_df,
                    column_config={
                        "Date": st.column_config.TextColumn("Date"),
                        "Time": st.column_config.TextColumn("Time"),
                        "Ticker": st.column_config.TextColumn("Ticker"),
                        "High_Price": st.column_config.NumberColumn("High Price", format="%.2f"),
                        "Close_Price": st.column_config.NumberColumn("Close Price", format="%.2f"),
                        "Resistance_Price": st.column_config.NumberColumn("Resistance Price (Current)", format="%.2f"),
                        "Distance_to_Resistance": st.column_config.NumberColumn("Distance to Resistance (Close)", format="%.2f"),
                        "Volume": st.column_config.NumberColumn("Volume", format="%.0f"),
                        "Touched_Resistance": st.column_config.TextColumn("Touched Resistance"),
                        "Selected_Date_Resistance": st.column_config.NumberColumn("Selected Date Resistance", format="%.2f"),
                        "Touched_Selected_Date_Resistance": st.column_config.TextColumn("Touched Selected Date Resistance")
                    },
                    use_container_width=True,
                    height=400
                )
                if st.button("Clear Historical Data"):
                    st.session_state['historical_data'] = []
                    if os.path.exists(historical_file):
                        os.remove(historical_file)
                        logger.info(f"Deleted historical data file: {historical_file}")
                    st.session_state['historical_data_date'] = None
                    st.rerun()
                download_csv(st.session_state['historical_data'], f"historical_resistance_{st.session_state['historical_data'][0]['Date'] if st.session_state['historical_data'] else 'last_scan'}.csv")
            else:
                st.info("No historical data available. Select a date and click 'Check Historical Resistance'.")

if __name__ == "__main__":
    main()