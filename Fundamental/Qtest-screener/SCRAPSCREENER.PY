import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import pyperclip

# Set page to wide mode for maximum column visibility
st.set_page_config(layout="wide")

# Function to scrape data from Screener.in
def scrape_screener_data(ticker):
    url = f"https://www.screener.in/company/{ticker}/consolidated/"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
    except requests.RequestException as e:
        st.error(f"Error fetching URL for ticker {ticker}: {e}")
        return None, None

    tables = soup.find_all('table', class_='data-table')
    quarterly_data = None
    yearly_data = None

    for table in tables:
        section = table.find_parent('section')
        if section:
            if section.get('id') == 'quarters':
                quarterly_data = parse_table(table)
            elif section.get('id') == 'profit-loss':
                yearly_data = parse_table(table)

    return quarterly_data, yearly_data

# Function to parse table data into a DataFrame
def parse_table(table):
    headers = [th.text.strip() for th in table.find('thead').find_all('th')]
    rows = []
    for tr in table.find('tbody').find_all('tr'):
        cells = [td.text.strip() for td in tr.find_all('td')]
        if cells:
            rows.append(cells)
    return pd.DataFrame(rows, columns=headers)

# Function to save tables to CSV
def save_to_csv(quarterly_data, yearly_data, ticker):
    output_dir = 'output_tables'
    os.makedirs(output_dir, exist_ok=True)

    if quarterly_data is not None:
        quarterly_data.to_csv(os.path.join(output_dir, f'{ticker}_quarterly_results.csv'), index=False)
        st.success(f"Quarterly Results saved to 'output_tables/{ticker}_quarterly_results.csv'")

    if yearly_data is not None:
        yearly_data.to_csv(os.path.join(output_dir, f'{ticker}_profit_loss.csv'), index=False)
        st.success(f"Profit & Loss saved to 'output_tables/{ticker}_profit_loss.csv'")

# Function to copy DataFrame to clipboard
def copy_to_clipboard(df, table_name):
    try:
        df_string = df.to_csv(sep='\t', index=False)
        pyperclip.copy(df_string)
        st.success(f"{table_name} copied to clipboard! Paste into Excel or Google Sheets.")
    except Exception as e:
        st.error(f"Error copying {table_name} to clipboard: {e}. Please try again or check clipboard permissions.")

# Streamlit app
def main():
    st.title("Screener.in Financial Data Scraper")
    st.write("Enter an NSE ticker to scrape raw Quarterly Results and Profit & Loss tables, "
             "download as CSV, and copy to clipboard with a single click. Tables are displayed in wide mode.")

    # Initialize session state
    if 'quarterly_data' not in st.session_state:
        st.session_state.quarterly_data = None
    if 'yearly_data' not in st.session_state:
        st.session_state.yearly_data = None
    if 'ticker' not in st.session_state:
        st.session_state.ticker = ""

    ticker = st.text_input("Enter NSE Ticker (e.g., RELIANCE):", value=st.session_state.ticker).strip().upper()

    if st.button("Scrape Data"):
        if not ticker:
            st.error("Please enter a valid NSE ticker.")
            return

        with st.spinner("Scraping data..."):
            quarterly_data, yearly_data = scrape_screener_data(ticker)
            if quarterly_data is None or yearly_data is None:
                st.error("Failed to scrape data. Please check the ticker or try again.")
                return

            # Store data in session state
            st.session_state.quarterly_data = quarterly_data
            st.session_state.yearly_data = yearly_data
            st.session_state.ticker = ticker

            save_to_csv(quarterly_data, yearly_data, ticker)

    # Display data if available
    if st.session_state.quarterly_data is not None:
        st.subheader("Raw Quarterly Results")
        st.dataframe(st.session_state.quarterly_data, use_container_width=True)
        if st.button("Copy Quarterly Results to Clipboard"):
            copy_to_clipboard(st.session_state.quarterly_data, "Quarterly Results")

    if st.session_state.yearly_data is not None:
        st.subheader("Raw Profit & Loss")
        st.dataframe(st.session_state.yearly_data, use_container_width=True)
        if st.button("Copy Profit & Loss to Clipboard"):
            copy_to_clipboard(st.session_state.yearly_data, "Profit & Loss")

if __name__ == "__main__":
    main()