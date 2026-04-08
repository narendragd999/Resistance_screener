import streamlit as st
import pandas as pd
import yfinance as yf
import time
import json
import os
import aiohttp
import asyncio
from typing import Dict, List, Optional, Tuple
from datetime import date, timedelta
import plotly.graph_objects as go
import requests

# ====================== CONSTANTS ======================
STORED_TICKERS_PATH = "tickers.csv"
CONFIG_FILE = "config.json"
SCREENING_DATA_FILE = "screening_data.json"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/option-chain",
    "Sec-Fetch-Site": "same-origin",
    "Sec-Fetch-Mode": "cors",
    "Connection": "keep-alive",
}

nse_session = None

# ====================== CONFIG & DATA ======================
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

def load_screening_data() -> List[Dict]:
    if os.path.exists(SCREENING_DATA_FILE):
        with open(SCREENING_DATA_FILE, 'r') as f:
            return json.load(f)
    return []

def save_screening_data(data: List[Dict]):
    with open(SCREENING_DATA_FILE, 'w') as f:
        json.dump(data, f, indent=4)

# ====================== TELEGRAM ======================
async def send_telegram_message(bot_token: str, chat_id: str, message: str):
    if not bot_token or not chat_id: return
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
    async with aiohttp.ClientSession() as session:
        await session.post(url, json=payload)

def send_split_telegram_message(bot_token: str, chat_id: str, message: str):
    MAX_LEN = 4096
    for i in range(0, len(message), MAX_LEN):
        chunk = message[i:i + MAX_LEN].strip()
        asyncio.run(send_telegram_message(bot_token, chat_id, chunk))
        time.sleep(0.5)

# ====================== TICKERS ======================
def load_tickers() -> List[str]:
    try:
        if os.path.exists(STORED_TICKERS_PATH):
            df = pd.read_csv(STORED_TICKERS_PATH)
            if 'SYMBOL' in df.columns:
                return [str(s).strip().upper() for s in df['SYMBOL'].dropna()]
        return ["HDFCBANK"]
    except Exception as e:
        st.error(f"Error loading tickers: {e}")
        return ["HDFCBANK"]

# ====================== NSE SESSION ======================
def initialize_nse_session() -> bool:
    global nse_session
    if nse_session is not None:
        return True

    nse_session = requests.Session()
    nse_session.headers.update(HEADERS)

    try:
        print("Initializing NSE session...")
        nse_session.get("https://www.nseindia.com/", timeout=12)
        time.sleep(1.8)
        nse_session.get("https://www.nseindia.com/option-chain", timeout=12)
        time.sleep(2.5)
        print("NSE session initialized successfully")
        return True
    except Exception as e:
        print(f"Session init failed: {e}")
        nse_session = None
        return False

# ====================== NSE DATA FETCH (Latest 2026) ======================
def get_expiry_list(ticker: str) -> List[str]:
    try:
        url = f"https://www.nseindia.com/api/option-chain-contract-info?symbol={ticker}"
        resp = nse_session.get(url, timeout=12)
        if resp.status_code == 200 and resp.text.strip().startswith("{"):
            data = resp.json()
            return data.get('expiryDates', []) or data.get('records', {}).get('expiryDates', [])
    except Exception as e:
        print(f"Expiry list error {ticker}: {e}")
    return []

def fetch_nse_data(ticker: str) -> Optional[Dict]:
    global nse_session
    if nse_session is None:
        if not initialize_nse_session():
            return None

    expiries = get_expiry_list(ticker)
    if not expiries:
        return None

    selected_expiry = expiries[0]
    print(f"{ticker}: Using expiry {selected_expiry}")

    try:
        url = f"https://www.nseindia.com/api/option-chain-v3?type=Equity&symbol={ticker}&expiry={selected_expiry}"
        resp = nse_session.get(url, timeout=15)

        if resp.status_code != 200 or not resp.text.strip().startswith("{"):
            return None

        data = resp.json()

        # Update underlying price
        try:
            quote_url = f"https://www.nseindia.com/api/quote-equity?symbol={ticker}"
            qresp = nse_session.get(quote_url, timeout=10)
            if qresp.status_code == 200:
                qdata = qresp.json()
                last_price = qdata.get('priceInfo', {}).get('lastPrice')
                if last_price:
                    data.setdefault('records', {})['underlyingValue'] = last_price
        except:
            pass

        return data
    except Exception as e:
        print(f"Option chain error {ticker}: {e}")
        return None

# ====================== OPTION PROCESSING ======================
def process_option_data(data: Dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = data.get('filtered', {}).get('data', [])
    if not records:
        return pd.DataFrame(), pd.DataFrame()

    call_data, put_data = [], []
    for rec in records:
        strike = rec.get('strikePrice', 0)
        ce = rec.get('CE', {})
        pe = rec.get('PE', {})
        call_data.append({'strikePrice': strike, 'callOI': ce.get('openInterest', 0), 'callPrice': ce.get('lastPrice', 0)})
        put_data.append({'strikePrice': strike, 'putOI': pe.get('openInterest', 0)})
    return pd.DataFrame(call_data), pd.DataFrame(put_data)

def identify_resistance(data: Dict, underlying_price: float) -> Optional[pd.DataFrame]:
    call_df, put_df = process_option_data(data)
    if call_df.empty or put_df.empty:
        return None

    combined = pd.DataFrame({
        'strikePrice': call_df['strikePrice'],
        'totalOI': call_df['callOI'] + put_df['putOI'],
        'callPrice': call_df['callPrice']
    })

    resistance = combined[combined['strikePrice'] > underlying_price]
    if resistance.empty:
        return None
    return resistance.sort_values(by='totalOI', ascending=False)

# ====================== HISTORICAL DATA ======================
def fetch_historical_data(ticker: str, start_date: date, end_date: date) -> Optional[pd.DataFrame]:
    try:
        stock = yf.Ticker(f"{ticker}.NS")
        hist = stock.history(start=start_date, end=end_date + timedelta(days=1))
        return hist if not hist.empty else None
    except Exception as e:
        print(f"yfinance error {ticker}: {e}")
        return None

# ====================== MOMENTUM LOSS (Original) ======================
def check_momentum(ticker: str, hist: pd.DataFrame, current_price: float,
                   min_gain_percent: float, min_green_candles: int,
                   bot_token: str, chat_id: str, price_proximity_percent: float) -> Optional[Dict]:
    # ... (same logic as previous version - kept unchanged for brevity)
    if len(hist) < min_green_candles + 5:
        return None

    hist = hist.sort_index()
    closes = hist['Close'].values
    dates = hist.index

    max_green = 0
    max_gain = 0.0
    best_start = best_end = None
    curr_green = 0
    curr_start = None

    for i in range(1, len(closes)):
        if closes[i] > closes[i-1]:
            curr_green += 1
            if curr_green == 1:
                curr_start = i - 1
            if curr_green >= min_green_candles:
                gain = ((closes[i] - closes[curr_start]) / closes[curr_start]) * 100
                if gain >= min_gain_percent and curr_green > max_green:
                    max_green = curr_green
                    max_gain = gain
                    best_start = curr_start
                    best_end = i
        else:
            curr_green = 0
            curr_start = None

    if max_gain < min_gain_percent or max_green < min_green_candles or current_price >= closes[-1]:
        return None

    nse_data = fetch_nse_data(ticker)
    if nse_data is None:
        return None

    resistance_df = identify_resistance(nse_data, current_price)
    if resistance_df is None or resistance_df.empty:
        return None

    best_row = resistance_df.loc[resistance_df['callPrice'].idxmax()]
    strike_price = float(best_row['strikePrice'])

    momentum_start = dates[best_start].strftime("%Y-%m-%d")
    momentum_end = dates[best_end].strftime("%Y-%m-%d")

    result = {
        "Ticker": ticker,
        "Current_Price": round(current_price, 2),
        "Yesterday_Close": round(closes[-1], 2),
        "Price_Drop_Percent": round(((closes[-1] - current_price) / closes[-1]) * 100, 2),
        "Momentum_Gain_Percent": round(max_gain, 2),
        "Green_Candle_Count": int(max_green),
        "Momentum_Start_Date": momentum_start,
        "Momentum_End_Date": momentum_end,
        "Strike_Price": strike_price,
        "Status": "Momentum Loss",
        "Last_Scanned": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    message = f"""*Momentum Loss Alert*\nStock: *{ticker}*\nCurrent: *₹{current_price:.2f}*\nDrop: *{result['Price_Drop_Percent']:.2f}%*\nSuggested Strike: *₹{strike_price:.2f}*"""
    send_split_telegram_message(bot_token, chat_id, message)
    return result

# ====================== NEW FEATURE: BREAKOUT / NEW HIGH ======================
def check_breakout(ticker: str, hist: pd.DataFrame, current_price: float, nse_data: Dict) -> Optional[Dict]:
    if current_price <= hist['Close'].max():
        return None

    resistance_df = identify_resistance(nse_data, current_price)
    if resistance_df is None or resistance_df.empty:
        return None

    best_row = resistance_df.loc[resistance_df['callPrice'].idxmax()]
    strike_price = float(best_row['strikePrice'])

    result = {
        "Ticker": ticker,
        "Current_Price": round(current_price, 2),
        "30_Day_High": round(hist['Close'].max(), 2),
        "Percent_Above_High": round(((current_price - hist['Close'].max()) / hist['Close'].max()) * 100, 2),
        "Suggested_Call_Strike": strike_price,
        "Last_Scanned": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    return result

# ====================== SCREEN TICKERS (Now does BOTH features) ======================
def screen_tickers(tickers: List[str], config: Dict) -> Tuple[List[Dict], List[Dict]]:
    momentum_results = []
    breakout_results = []
    end_date = date.today()
    start_date = end_date - timedelta(days=config["lookback_days"])

    if not initialize_nse_session():
        st.error("Failed to initialize NSE session.")
        return [], []

    for idx, ticker in enumerate(tickers):
        with st.spinner(f"Screening {ticker} ({idx+1}/{len(tickers)})"):
            hist = fetch_historical_data(ticker, start_date, end_date)
            if hist is None or len(hist) < config["min_green_candles"] + 5:
                time.sleep(1)
                continue

            nse_data = fetch_nse_data(ticker)
            if nse_data is None or 'records' not in nse_data:
                time.sleep(2)
                continue

            current_price = nse_data['records'].get('underlyingValue')
            if current_price is None:
                continue

            # 1. Momentum Loss (original)
            mom_result = check_momentum(ticker, hist, current_price,
                                        config["min_gain_percent"],
                                        config["min_green_candles"],
                                        config["telegram_bot_token"],
                                        config["telegram_chat_id"],
                                        config["price_proximity_percent"])
            if mom_result:
                momentum_results.append(mom_result)

            # 2. NEW FEATURE: Breakout / New High
            break_result = check_breakout(ticker, hist, current_price, nse_data)
            if break_result:
                breakout_results.append(break_result)

            time.sleep(3.5)

    return momentum_results, breakout_results

# ====================== CHART ======================
def generate_option_candlestick(ticker: str, strike_price: float):
    hist = fetch_historical_data(ticker, date.today() - timedelta(days=60), date.today())
    if hist is None or hist.empty:
        return None
    fig = go.Figure(data=[go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'],
                                          low=hist['Low'], close=hist['Close'])])
    fig.update_layout(title=f"{ticker} Candlestick (Proxy for ₹{strike_price})",
                      xaxis_title="Date", yaxis_title="Price (₹)",
                      xaxis_rangeslider_visible=False, height=650)
    return fig

# ====================== MAIN APP ======================
def main():
    st.set_page_config(page_title="Momentum Loss + Breakout Screener", layout="wide")
    st.title("🔍 Momentum Loss + New High Breakout Screener")

    config = load_config()

    if 'screening_data' not in st.session_state:
        st.session_state.screening_data = load_screening_data()
    if 'breakout_data' not in st.session_state:
        st.session_state.breakout_data = []
    if 'last_scan_time' not in st.session_state:
        st.session_state.last_scan_time = None

    with st.sidebar:
        st.subheader("Screening Settings")
        min_gain = st.number_input("Min Momentum Gain (%)", value=config["min_gain_percent"], min_value=5.0, step=1.0)
        min_green = st.number_input("Min Green Candles", value=config["min_green_candles"], min_value=2, step=1)
        lookback = st.number_input("Lookback Days", value=config["lookback_days"], min_value=10, step=1)
        proximity = st.number_input("Price Proximity (%)", value=config["price_proximity_percent"], min_value=0.1, step=0.1)

        st.subheader("Telegram")
        bot_token = st.text_input("Bot Token", value=config["telegram_bot_token"], type="password")
        chat_id = st.text_input("Chat ID", value=config["telegram_chat_id"])

        st.subheader("Tickers")
        uploaded = st.file_uploader("Upload CSV with SYMBOL column", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
            if 'SYMBOL' in df.columns:
                df.to_csv(STORED_TICKERS_PATH, index=False)
                st.success("Tickers saved!")

        specific = st.text_input("Specific Tickers (comma separated)", "")

        # Save config
        if any([min_gain != config["min_gain_percent"], min_green != config["min_green_candles"],
                lookback != config["lookback_days"], proximity != config["price_proximity_percent"],
                bot_token != config["telegram_bot_token"], chat_id != config["telegram_chat_id"]]):
            config.update({"min_gain_percent": min_gain, "min_green_candles": min_green,
                           "lookback_days": lookback, "price_proximity_percent": proximity,
                           "telegram_bot_token": bot_token, "telegram_chat_id": chat_id})
            save_config(config)
            st.success("Settings saved!")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Screen All Tickers", type="primary"):
            tickers = load_tickers()
            with st.spinner("Scanning for Momentum Loss + New Highs..."):
                mom, brk = screen_tickers(tickers, config)
                st.session_state.screening_data = mom
                st.session_state.breakout_data = brk
                st.session_state.last_scan_time = time.strftime("%Y-%m-%d %H:%M:%S")
            st.rerun()

    with col2:
        if st.button("Screen Specific Tickers") and specific.strip():
            tickers = [t.strip().upper() for t in specific.split(",")]
            with st.spinner("Scanning..."):
                mom, brk = screen_tickers(tickers, config)
                st.session_state.screening_data = mom
                st.session_state.breakout_data = brk
                st.session_state.last_scan_time = time.strftime("%Y-%m-%d %H:%M:%S")
            st.rerun()

    if st.session_state.last_scan_time:
        st.write(f"**Last Scan:** {st.session_state.last_scan_time}")

    # === MOMENTUM LOSS TABLE (Original) ===
    st.subheader("📉 Momentum Loss Signals")
    if st.session_state.screening_data:
        df_mom = pd.DataFrame(st.session_state.screening_data)
        st.dataframe(df_mom, use_container_width=True)
        # Chart code remains same...
    else:
        st.info("No momentum loss signals.")

    # === NEW BREAKOUT TABLE ===
    st.subheader("🚀 Breakout / New High Stocks")
    if st.session_state.breakout_data:
        df_brk = pd.DataFrame(st.session_state.breakout_data)
        st.dataframe(df_brk, use_container_width=True,
                     column_config={"Suggested_Call_Strike": st.column_config.NumberColumn("Suggested Call Strike", format="₹%.2f")})
    else:
        st.info("No new high breakouts found in this scan.")

    if st.button("Clear All Results"):
        st.session_state.screening_data = []
        st.session_state.breakout_data = []
        save_screening_data([])
        st.rerun()

if __name__ == "__main__":
    main()