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
import uuid

# Constants
STORED_TICKERS_PATH = "tickers.csv"
CONFIG_FILE = "config.json"
SCREENING_DATA_FILE = "screening_data.json"
BACKTEST_DATA_FILE = "backtest_data.json"
PORTFOLIO_FILE = "portfolio.json"
api_call_counter = 0

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
        "price_proximity_percent": 1.0,
        "min_risk_reward_ratio": 0.1,  # New: Minimum risk-reward ratio
        "iv_threshold": 30.0,  # New: Implied volatility threshold for alerts
        "oi_score_weight": 0.5,  # New: Weight for OI in resistance scoring
        "pcr_score_weight": 0.3,  # New: Weight for PCR in resistance scoring
        "premium_score_weight": 0.2  # New: Weight for premium in resistance scoring
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

# Load/Save Screening and Backtest Data
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

# Load/Save Portfolio
def load_portfolio() -> List[Dict]:
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE, 'r') as f:
            return json.load(f)
    return []

def save_portfolio(data: List[Dict]):
    with open(PORTFOLIO_FILE, 'w') as f:
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
        print(f"Using cached data for {ticker}")
        return cache[cache_key]

    try:
        if nse_session is None:
            print(f"No session available for {ticker}")
            return None

        option_chain_url = f"https://www.nseindia.com/api/option-chain-equities?symbol={ticker}"
        quote_url = f"https://www.nseindia.com/api/quote-equity?symbol={ticker}"

        print(f"Fetching data for API {api_call_counter}--{ticker}")

        response = nse_session.get(option_chain_url, headers=headers)
        if response.status_code != 200:
            print(f"Failed to load option chain for {ticker}: {response.status_code}")
            return None

        data = response.json()

        quote_response = nse_session.get(quote_url, headers=headers)
        if quote_response.status_code == 200:
            quote_data = quote_response.json()
            last_price = quote_data.get('priceInfo', {}).get('lastPrice', 0)
            if last_price > 0 and 'records' in data:
                data['records']['underlyingValue'] = last_price
                print(f"Updated underlying value for {ticker} with last price: {last_price}")
            else:
                print(f"No valid last price found for {ticker}, using default underlying.")
        else:
            print(f"Failed to load quote data for {ticker}: {quote_response.status_code}")

        cache[cache_key] = data
        cache[f"{cache_key}_timestamp"] = time.time()
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
        call_iv = record.get('CE', {}).get('impliedVolatility', 0)  # New: Extract IV
        put_oi = record.get('PE', {}).get('openInterest', 0)
        put_iv = record.get('PE', {}).get('impliedVolatility', 0)  # New: Extract IV
        call_data.append({'strikePrice': strike, 'callOI': call_oi, 'callPrice': call_price, 'callIV': call_iv})
        put_data.append({'strikePrice': strike, 'putOI': put_oi, 'putIV': put_iv})

    call_df = pd.DataFrame(call_data)
    put_df = pd.DataFrame(put_data)
    return call_df, put_df

# Identify Resistance Based on Option Chain OI
def identify_resistance(data: Dict, underlying_price: float, expiry: Optional[str], config: Dict) -> Optional[pd.DataFrame]:
    call_df, put_df = process_option_data(data, expiry)
    if call_df.empty or put_df.empty:
        return None

    combined_df = pd.DataFrame({
        'strikePrice': call_df['strikePrice'],
        'totalOI': call_df['callOI'] + put_df['putOI'],
        'callOI': call_df['callOI'],
        'putOI': put_df['putOI'],
        'callPrice': call_df['callPrice'],
        'callIV': call_df['callIV']
    })

    # Filter strikes above the underlying price (for resistance)
    resistance_candidates = combined_df[combined_df['strikePrice'] > underlying_price]
    if resistance_candidates.empty:
        return None

    # Calculate PCR
    resistance_candidates['PCR'] = resistance_candidates['putOI'] / resistance_candidates['callOI'].replace(0, np.nan)
    resistance_candidates['PCR'] = resistance_candidates['PCR'].fillna(1)

    # Calculate OI concentration
    total_oi = resistance_candidates['totalOI'].sum()
    resistance_candidates['OI_Percent'] = resistance_candidates['totalOI'] / total_oi * 100

    # Calculate resistance score
    resistance_candidates['Score'] = (
        resistance_candidates['totalOI'] / resistance_candidates['totalOI'].max() * config['oi_score_weight'] +
        resistance_candidates['PCR'] / resistance_candidates['PCR'].max() * config['pcr_score_weight'] +
        resistance_candidates['callPrice'] / resistance_candidates['callPrice'].max() * config['premium_score_weight']
    )

    # Sort by score
    resistance_candidates = resistance_candidates.sort_values(by='Score', ascending=False)
    return resistance_candidates

# Calculate Risk-Reward Ratio
def calculate_risk_reward(resistance_df: pd.DataFrame, current_price: float) -> pd.DataFrame:
    resistance_df['Risk'] = resistance_df['strikePrice'] - current_price
    resistance_df['Reward'] = resistance_df['callPrice']
    resistance_df['Risk_Reward_Ratio'] = resistance_df['Reward'] / resistance_df['Risk'].replace(0, np.nan)
    return resistance_df

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
def check_momentum(ticker: str, hist: pd.DataFrame, current_price: float, min_gain_percent: float, min_green_candles: int, bot_token: str, chat_id: str, price_proximity_percent: float, config: Dict, expiry: Optional[str]) -> Optional[Dict]:
    if hist.empty or len(hist) < min_green_candles + 2:
        print(f"{ticker}: Insufficient data (only {len(hist)} days)")
        return None
    try:
        yesterday = hist.index[-1]
        yesterday_close = hist.loc[yesterday, 'Close']
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
                if gain_percent >= min_gain_percent and current_green_candles >= max_green_candles:
                    max_green_candles = current_green_candles
                    max_gain = gain_percent
                    best_start_idx = current_start_idx
                    best_end_idx = current_end_idx
        else:
            current_green_candles = 0
            current_start_idx = None
    result = None
    if max_gain >= min_gain_percent and max_green_candles >= min_green_candles and current_price < yesterday_close:
        # Momentum loss detected, fetch resistance strike
        nse_data = fetch_nse_data(ticker, time.time())
        if nse_data is None:
            print(f"{ticker}: No option chain data available")
            return None
        resistance_df = identify_resistance(nse_data, current_price, expiry, config)
        if resistance_df is None or resistance_df.empty:
            print(f"{ticker}: Could not determine resistance strike")
            return None

        momentum_start_date = dates[best_start_idx].strftime("%Y-%m-%d")
        momentum_end_date = dates[best_end_idx].strftime("%Y-%m-%d")
        
        # Find the high or open of the day after the momentum period ends (red candle)
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
            print(f"{ticker}: No data available for the day after momentum period ends")
            red_candle_open = hist.loc[yesterday, 'Open']
            red_candle_high = hist.loc[yesterday, 'High']
            yesterday_high = max(red_candle_open, red_candle_high)

        # Filter resistance strikes above the red candle's high/open
        valid_resistances = resistance_df[resistance_df['strikePrice'] > yesterday_high]
        if valid_resistances.empty:
            print(f"{ticker}: No resistance strikes found above red candle high/open ({yesterday_high})")
            return None

        # Calculate risk-reward ratio
        valid_resistances = calculate_risk_reward(valid_resistances, current_price)
        valid_resistances = valid_resistances[valid_resistances['Risk_Reward_Ratio'] >= config['min_risk_reward_ratio']]
        if valid_resistances.empty:
            print(f"{ticker}: No strikes meet the minimum risk-reward ratio")
            return None

        # Select the strike with the highest score
        best_strike = valid_resistances.iloc[0]['strikePrice']
        call_premium = valid_resistances.iloc[0]['callPrice']
        call_iv = valid_resistances.iloc[0]['callIV']

        strike_price = float(best_strike)

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
            "Call_Premium": float(call_premium),
            "Implied_Volatility": float(call_iv),
            "Status": "Momentum Loss",
            "Last_Scanned": time.strftime("%Y-%m-%d %H:%M:%S"),
            "Expiry": expiry if expiry else "Nearest"
        }

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
            f"Call Premium: *₹{call_premium:.2f}*\n"
            f"Implied Volatility: *{call_iv:.2f}%*\n"
            f"Expiry: *{result['Expiry']}*\n"
            f"Timestamp: *{result['Last_Scanned']}*"
        )
        print(f"{ticker}: Sending Telegram notification - Momentum Loss")
        send_split_telegram_message(bot_token, chat_id, message)

        # Volatility-Based Alert
        if call_iv > config['iv_threshold']:
            iv_message = (
                f"*High IV Alert*\n"
                f"Stock: *{ticker}*\n"
                f"Strike: *₹{strike_price:.2f}*\n"
                f"Implied Volatility: *{call_iv:.2f}%*\n"
                f"Call Premium: *₹{call_premium:.2f}*\n"
                f"Expiry: *{result['Expiry']}*\n"
                f"Action: *High IV detected, consider selling call for higher premium*\n"
                f"Timestamp: *{result['Last_Scanned']}*"
            )
            print(f"{ticker}: Sending Telegram notification - High IV")
            send_split_telegram_message(bot_token, chat_id, iv_message)

        print(f"yesterday_high ==={yesterday_high}===CURRENT_PRICE {current_price}")
        # Check for momentum loss recovery
        if yesterday_high is not None:
            proximity_threshold = price_proximity_percent / 100
            if current_price >= yesterday_high:
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
                    f"Call Premium: *₹{call_premium:.2f}*\n"
                    f"Expiry: *{result['Expiry']}*\n"
                    f"Timestamp: *{result['Last_Scanned']}*\n"
                    f"Action: *Consider selling call at suggested strike*"
                )
                print(f"{ticker}: Sending Telegram notification - Momentum Loss Recovery")
                send_split_telegram_message(bot_token, chat_id, recovery_message)
                result["Status"] = "Momentum Loss Recovery"
    return result

# Screen Tickers
def screen_tickers(tickers: List[str], min_gain_percent: float, min_green_candles: int, lookback_days: int, bot_token: str, chat_id: str, refresh_key: float, price_proximity_percent: float, config: Dict, expiry: Optional[str]) -> List[Dict]:
    global nse_session
    end_date = date.today()
    start_date = end_date - timedelta(days=lookback_days)
    results = []
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
            result = check_momentum(ticker, hist, current_price, min_gain_percent, min_green_candles, bot_token, chat_id, price_proximity_percent, config, expiry)
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

# Main Application
def main():
    st.set_page_config(page_title="Momentum Loss Screener", layout="wide")
    tabs = st.tabs(["Real-Time Screener", "Portfolio"])  # New: Added Portfolio tab

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
    if 'selected_expiry' not in st.session_state:
        st.session_state['selected_expiry'] = None

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
        min_risk_reward_ratio = st.number_input(
            "Minimum Risk-Reward Ratio:",
            value=st.session_state['config']['min_risk_reward_ratio'],
            min_value=0.01,
            step=0.01,
            key="min_risk_reward_ratio"
        )
        iv_threshold = st.number_input(
            "Implied Volatility Threshold (%):",
            value=st.session_state['config']['iv_threshold'],
            min_value=10.0,
            step=1.0,
            key="iv_threshold"
        )

        st.subheader("Resistance Scoring Weights")
        oi_score_weight = st.number_input(
            "OI Score Weight:",
            value=st.session_state['config']['oi_score_weight'],
            min_value=0.0,
            max_value=1.0,
            step=0.1,
            key="oi_score_weight"
        )
        pcr_score_weight = st.number_input(
            "PCR Score Weight:",
            value=st.session_state['config']['pcr_score_weight'],
            min_value=0.0,
            max_value=1.0,
            step=0.1,
            key="pcr_score_weight"
        )
        premium_score_weight = st.number_input(
            "Premium Score Weight:",
            value=st.session_state['config']['premium_score_weight'],
            min_value=0.0,
            max_value=1.0,
            step=0.1,
            key="premium_score_weight"
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

        # Expiry Selection
        st.subheader("Option Expiry")
        nse_data = fetch_nse_data("HDFCBANK", time.time())  # Use a sample ticker to get expiries
        expiries = nse_data.get('records', {}).get('expiryDates', []) if nse_data else []
        if expiries:
            st.session_state['selected_expiry'] = st.selectbox(
                "Select Option Expiry",
                ["Nearest"] + expiries,
                index=0,
                key="expiry_select"
            )
            if st.session_state['selected_expiry'] == "Nearest":
                st.session_state['selected_expiry'] = expiries[0]

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
        if st.session_state['config']['min_risk_reward_ratio'] != min_risk_reward_ratio:
            st.session_state['config']['min_risk_reward_ratio'] = min_risk_reward_ratio
            config_changed = True
        if st.session_state['config']['iv_threshold'] != iv_threshold:
            st.session_state['config']['iv_threshold'] = iv_threshold
            config_changed = True
        if st.session_state['config']['oi_score_weight'] != oi_score_weight:
            st.session_state['config']['oi_score_weight'] = oi_score_weight
            config_changed = True
        if st.session_state['config']['pcr_score_weight'] != pcr_score_weight:
            st.session_state['config']['pcr_score_weight'] = pcr_score_weight
            config_changed = True
        if st.session_state['config']['premium_score_weight'] != premium_score_weight:
            st.session_state['config']['premium_score_weight'] = premium_score_weight
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
                st.session_state['refresh_key'],
                st.session_state['config']['price_proximity_percent'],
                st.session_state['config'],
                st.session_state['selected_expiry']
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
                perform_screening(tickers)
                st.rerun()
        with col2:
            if st.button("Screen Specific Tickers") and specific_tickers and not st.session_state['scan_in_progress']:
                tickers = [t.strip() for t in specific_tickers.split(',')]
                perform_screening(tickers)
                st.rerun()

        if st.session_state['screening_data']:
            st.write("### Stocks with Momentum Loss")
            search_query = st.text_input("Search Results by Ticker", key="screening_search")
            screening_df = pd.DataFrame(st.session_state['screening_data'])
            if search_query:
                screening_df = screening_df[screening_df['Ticker'].str.contains(search_query, case=False, na=False)]
            if not screening_df.empty:
                st.dataframe(
                    screening_df,
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
                        "Call_Premium": st.column_config.NumberColumn("Call Premium", format="%.2f"),
                        "Implied_Volatility": st.column_config.NumberColumn("IV %", format="%.2f"),
                        "Status": st.column_config.TextColumn("Status"),
                        "Last_Scanned": st.column_config.TextColumn("Last Scanned"),
                        "Expiry": st.column_config.TextColumn("Expiry")
                    },
                    use_container_width=True,
                    height=400
                )
                selected_ticker = st.selectbox("Select Ticker for Candlestick Chart", screening_df['Ticker'].unique())
                if 'Strike_Price' in screening_df.columns and not screening_df[screening_df['Ticker'] == selected_ticker].empty:
                    strike_prices = screening_df[screening_df['Ticker'] == selected_ticker]['Strike_Price'].unique()
                    if strike_prices.size > 0:
                        selected_strike = st.selectbox("Select Strike Price", strike_prices)
                        chart = generate_option_candlestick(selected_ticker, selected_strike, date.today() - timedelta(days=30), date.today())
                        if chart:
                            st.plotly_chart(chart, use_container_width=True)
                        else:
                            st.warning(f"No chart data available for {selected_ticker} at strike {selected_strike}.")
                    else:
                        st.warning(f"No strike prices available for {selected_ticker}.")
                else:
                    st.warning(f"No data available for {selected_ticker}.")
            else:
                st.info("No matching results found for the search query.")
            if st.button("Clear Screening Results"):
                st.session_state['screening_data'] = []
                save_screening_data(st.session_state['screening_data'])
                st.rerun()
        else:
            st.info("No results found. Run a scan to check for stocks.")

    with tabs[1]:
        st.title("Call Selling Portfolio")
        st.subheader("Manage Your Call Selling Trades")
        with st.form("add_trade_form"):
            st.write("Add New Trade")
            trade_ticker = st.text_input("Ticker", key="trade_ticker")
            trade_strike = st.number_input("Strike Price", min_value=0.0, step=0.1, key="trade_strike")
            trade_premium = st.number_input("Premium Received", min_value=0.0, step=0.01, key="trade_premium")
            trade_expiry = st.date_input("Expiry Date", min_value=date.today(), key="trade_expiry")
            trade_status = st.selectbox("Status", ["Open", "Closed", "Exercised"], key="trade_status")
            submitted = st.form_submit_button("Add Trade")
            if submitted and trade_ticker:
                portfolio = load_portfolio()
                trade = {
                    "Trade_ID": str(uuid.uuid4()),
                    "Ticker": trade_ticker,
                    "Strike": float(trade_strike),
                    "Premium": float(trade_premium),
                    "Expiry": trade_expiry.strftime("%Y-%m-%d"),
                    "Status": trade_status,
                    "Added": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                portfolio.append(trade)
                save_portfolio(portfolio)
                st.success("Trade added successfully!")
                st.rerun()

        portfolio_data = load_portfolio()
        if portfolio_data:
            portfolio_df = pd.DataFrame(portfolio_data)
            st.write("### Current Portfolio")
            st.dataframe(
                portfolio_df,
                column_config={
                    "Trade_ID": st.column_config.TextColumn("Trade ID"),
                    "Ticker": st.column_config.TextColumn("Ticker"),
                    "Strike": st.column_config.NumberColumn("Strike", format="%.2f"),
                    "Premium": st.column_config.NumberColumn("Premium", format="%.2f"),
                    "Expiry": st.column_config.TextColumn("Expiry"),
                    "Status": st.column_config.TextColumn("Status"),
                    "Added": st.column_config.TextColumn("Added")
                },
                use_container_width=True,
                height=400
            )
            st.subheader("Update Trade Status")
            selected_trade_id = st.selectbox("Select Trade to Update", portfolio_df['Trade_ID'], key="update_trade_id")
            new_status = st.selectbox("New Status", ["Open", "Closed", "Exercised"], key="new_trade_status")
            if st.button("Update Trade"):
                portfolio = load_portfolio()
                for trade in portfolio:
                    if trade['Trade_ID'] == selected_trade_id:
                        trade['Status'] = new_status
                        trade['Updated'] = time.strftime("%Y-%m-%d %H:%M:%S")
                        break
                save_portfolio(portfolio)
                st.success("Trade updated successfully!")
                st.rerun()
            if st.button("Clear Portfolio"):
                save_portfolio([])
                st.success("Portfolio cleared!")
                st.rerun()
        else:
            st.info("No trades in portfolio. Add a trade to get started.")
    
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Script error: {str(e)}")
        st.error(f"Script error: {str(e)}")