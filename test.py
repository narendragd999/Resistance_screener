import streamlit as st
import pandas as pd
import yfinance as yf
import time
import json
import os
import aiohttp
import asyncio
from typing import Dict, List, Optional
from datetime import date, timedelta
import requests

# ====================== CONSTANTS ======================
STORED_TICKERS_PATH = "tickers.csv"
CONFIG_FILE = "config.json"
SCREENING_DATA_FILE = "screening_data.json"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/option-chain",
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
    if not bot_token or not chat_id:
        return
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
        nse_session.get("https://www.nseindia.com/", timeout=12)
        time.sleep(1.8)
        nse_session.get("https://www.nseindia.com/option-chain", timeout=12)
        time.sleep(2.5)
        return True
    except Exception as e:
        print(f"Session init failed: {e}")
        nse_session = None
        return False


# ====================== LIVE OPTION CHAIN ======================
def get_expiry_list(ticker: str) -> List[str]:
    try:
        url = f"https://www.nseindia.com/api/option-chain-contract-info?symbol={ticker}"
        resp = nse_session.get(url, timeout=12)
        if resp.status_code == 200:
            return resp.json().get('expiryDates', [])
    except:
        pass
    return []


def fetch_nse_data(ticker: str) -> Optional[Dict]:
    if not initialize_nse_session():
        return None
    expiries = get_expiry_list(ticker)
    if not expiries:
        return None
    try:
        url = f"https://www.nseindia.com/api/option-chain-v3?type=Equity&symbol={ticker}&expiry={expiries[0]}"
        resp = nse_session.get(url, timeout=15)
        if resp.status_code == 200 and resp.text.strip().startswith("{"):
            data = resp.json()
            try:
                qresp = nse_session.get(f"https://www.nseindia.com/api/quote-equity?symbol={ticker}", timeout=10)
                if qresp.status_code == 200:
                    last_price = qresp.json().get('priceInfo', {}).get('lastPrice')
                    if last_price:
                        data.setdefault('records', {})['underlyingValue'] = last_price
            except:
                pass
            return data
    except:
        pass
    return None


def process_option_data(data: Dict):
    records = data.get('filtered', {}).get('data', [])
    call_data = []
    for rec in records:
        strike = rec.get('strikePrice', 0)
        ce = rec.get('CE', {})
        call_data.append({'strikePrice': strike, 'callPrice': ce.get('lastPrice', 0)})
    return pd.DataFrame(call_data)


def identify_resistance(data: Dict, underlying_price: float):
    call_df = process_option_data(data)
    if call_df.empty:
        return None
    combined = pd.DataFrame({'strikePrice': call_df['strikePrice'], 'callPrice': call_df['callPrice']})
    resistance = combined[combined['strikePrice'] > underlying_price]
    if resistance.empty:
        return None
    return resistance.sort_values(by='callPrice', ascending=False)


# ====================== HISTORICAL PREMIUM - MAX OF FH_TRADE_HIGH_PRICE & FH_CLOSING_PRICE ======================
def get_historical_max_premiums(ticker: str, expiry: str, strike: float) -> List[float]:
    try:
        from_date = (date.today() - timedelta(days=40)).strftime("%d-%m-%Y")
        to_date = date.today().strftime("%d-%m-%Y")

        url = f"https://www.nseindia.com/api/historicalOR/foCPV?from={from_date}&to={to_date}&instrumentType=OPTSTK&symbol={ticker}&year={date.today().year}&expiryDate={expiry}&optionType=CE&strikePrice={strike}"

        resp = nse_session.get(url, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            max_premiums = []
            for item in data.get('data', []):
                high = item.get('FH_TRADE_HIGH_PRICE')
                close = item.get('FH_CLOSING_PRICE')
                value = 0
                if high:
                    value = max(value, float(high))
                if close:
                    value = max(value, float(close))
                if value > 0:
                    max_premiums.append(value)
            return max_premiums
    except Exception as e:
        print(f"Historical error for {ticker}: {e}")
    return []


# ====================== CALL PREMIUM NEW HIGH ======================
def check_breakout(ticker: str, current_price: float, nse_data: Dict) -> Optional[Dict]:
    if not nse_data:
        return None

    call_df = process_option_data(nse_data)
    if call_df.empty:
        return None

    call_df['distance'] = (call_df['strikePrice'] - current_price).abs()
    nearest = call_df.loc[call_df['distance'].idxmin()]

    strike = nearest['strikePrice']
    current_premium = nearest['callPrice']

    if current_premium < 8:
        return None

    expiry = nse_data.get('records', {}).get('expiryDates', [None])[0]
    if not expiry:
        return None

    hist_max_premiums = get_historical_max_premiums(ticker, expiry, strike)
    if not hist_max_premiums:
        return None

    max_past = max(hist_max_premiums)

    if current_premium <= max_past * 1.02:   # 2% tolerance
        return None

    # Suggested resistance strike
    resistance_df = identify_resistance(nse_data, current_price)
    suggested_strike = float(resistance_df.loc[resistance_df['callPrice'].idxmax()]['strikePrice']) if not resistance_df.empty else strike + 50

    return {
        "Ticker": ticker,
        "Spot_Price": round(current_price, 2),
        "Call_Strike": int(strike),
        "Current_Call_Premium": round(current_premium, 2),
        "Max_Past_Premium": round(max_past, 2),
        "Suggested_Call_Strike": suggested_strike,
        "Last_Scanned": time.strftime("%Y-%m-%d %H:%M:%S")
    }


# ====================== MOMENTUM LOSS (Original) ======================
def check_momentum(ticker: str, hist: pd.DataFrame, current_price: float,
                   min_gain_percent: float, min_green_candles: int,
                   bot_token: str, chat_id: str, price_proximity_percent: float) -> Optional[Dict]:
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

    message = f"""*Momentum Loss Alert*
Stock: *{ticker}*
Current: *₹{current_price:.2f}*
Drop: *{result['Price_Drop_Percent']:.2f}%*
Gain: *{max_gain:.2f}%*
Strike: *₹{strike_price:.2f}*"""
    send_split_telegram_message(bot_token, chat_id, message)
    return result


def fetch_historical_data(ticker: str, start_date: date, end_date: date):
    try:
        stock = yf.Ticker(f"{ticker}.NS")
        hist = stock.history(start=start_date, end=end_date + timedelta(days=1))
        return hist if not hist.empty else None
    except:
        return None


# ====================== SCREEN TICKERS ======================
def screen_tickers(tickers: List[str], config: Dict):
    momentum_results = []
    breakout_results = []
    end_date = date.today()
    start_date = end_date - timedelta(days=config["lookback_days"])

    for idx, ticker in enumerate(tickers):
        with st.spinner(f"Screening {ticker} ({idx+1}/{len(tickers)})"):
            hist = fetch_historical_data(ticker, start_date, end_date)
            if hist is None or len(hist) < 10:
                time.sleep(1)
                continue

            nse_data = fetch_nse_data(ticker)
            if nse_data is None or 'records' not in nse_data:
                time.sleep(2)
                continue

            current_price = nse_data['records'].get('underlyingValue')
            if current_price is None:
                continue

            # Momentum Loss
            mom = check_momentum(ticker, hist, current_price,
                                 config["min_gain_percent"],
                                 config["min_green_candles"],
                                 config["telegram_bot_token"],
                                 config["telegram_chat_id"],
                                 config["price_proximity_percent"])
            if mom:
                momentum_results.append(mom)

            # Call Premium New High
            brk = check_breakout(ticker, current_price, nse_data)
            if brk:
                breakout_results.append(brk)

            time.sleep(2)

    return momentum_results, breakout_results


# ====================== MAIN APP ======================
def main():
    st.set_page_config(page_title="Momentum Loss + Call Premium New High", layout="wide")
    st.title("🔍 Momentum Loss + Call Premium New High Screener")

    config = load_config()

    if 'screening_data' not in st.session_state:
        st.session_state.screening_data = []
    if 'breakout_data' not in st.session_state:
        st.session_state.breakout_data = []
    if 'last_scan_time' not in st.session_state:
        st.session_state.last_scan_time = None

    with st.sidebar:
        st.subheader("Screening Settings")
        min_gain_percent = st.number_input("Minimum Momentum Gain (%)", value=config["min_gain_percent"], min_value=5.0, step=1.0)
        min_green_candles = st.number_input("Minimum Green Candles", value=config["min_green_candles"], min_value=2, step=1)
        lookback_days = st.number_input("Lookback Period (Days)", value=config["lookback_days"], min_value=10, step=1)
        price_proximity_percent = st.number_input("Price Proximity for Recovery (%)", value=config["price_proximity_percent"], min_value=0.1, step=0.1)

        st.subheader("Telegram Integration")
        telegram_bot_token = st.text_input("Telegram Bot Token", value=config["telegram_bot_token"], type="password")
        telegram_chat_id = st.text_input("Telegram Chat ID", value=config["telegram_chat_id"])

        st.subheader("Upload Tickers")
        uploaded_file = st.file_uploader("Upload CSV with 'SYMBOL' column", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            if 'SYMBOL' in df.columns:
                df.to_csv(STORED_TICKERS_PATH, index=False)
                st.success("✅ Tickers saved!")
            else:
                st.error("CSV must contain 'SYMBOL' column")

        specific_tickers = st.text_input("Screen Specific Tickers (comma-separated)", "")

        if any([min_gain_percent != config["min_gain_percent"], min_green_candles != config["min_green_candles"],
                lookback_days != config["lookback_days"], price_proximity_percent != config["price_proximity_percent"],
                telegram_bot_token != config["telegram_bot_token"], telegram_chat_id != config["telegram_chat_id"]]):
            config.update({
                "min_gain_percent": min_gain_percent,
                "min_green_candles": min_green_candles,
                "lookback_days": lookback_days,
                "price_proximity_percent": price_proximity_percent,
                "telegram_bot_token": telegram_bot_token,
                "telegram_chat_id": telegram_chat_id
            })
            save_config(config)
            st.success("✅ Settings saved!")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Screen All Tickers", type="primary"):
            tickers = load_tickers()
            with st.spinner("Scanning..."):
                mom, brk = screen_tickers(tickers, config)
                st.session_state.screening_data = mom
                st.session_state.breakout_data = brk
                st.session_state.last_scan_time = time.strftime("%Y-%m-%d %H:%M:%S")
            st.rerun()

    with col2:
        if st.button("Screen Specific Tickers") and specific_tickers.strip():
            tickers = [t.strip().upper() for t in specific_tickers.split(",")]
            with st.spinner("Scanning specific tickers..."):
                mom, brk = screen_tickers(tickers, config)
                st.session_state.screening_data = mom
                st.session_state.breakout_data = brk
                st.session_state.last_scan_time = time.strftime("%Y-%m-%d %H:%M:%S")
            st.rerun()

    if st.session_state.last_scan_time:
        st.write(f"**Last Scan:** {st.session_state.last_scan_time}")

    st.subheader("📉 Momentum Loss Signals")
    if st.session_state.screening_data:
        st.dataframe(pd.DataFrame(st.session_state.screening_data), use_container_width=True)
    else:
        st.info("No momentum loss signals found.")

    st.subheader("🚀 Call Premium New High (Max of FH_TRADE_HIGH_PRICE & FH_CLOSING_PRICE)")
    if st.session_state.breakout_data:
        st.dataframe(pd.DataFrame(st.session_state.breakout_data), use_container_width=True)
    else:
        st.info("No Call Premium New High detected in this scan.")

    if st.button("Clear All Results"):
        st.session_state.screening_data = []
        st.session_state.breakout_data = []
        save_screening_data([])
        st.rerun()


if __name__ == "__main__":
    main()