import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
from datetime import datetime
import time
import uuid

# Set page to wide mode
st.set_page_config(layout="wide")

# Function to scrape all data tables from Screener.in for a ticker
@st.cache_data
def scrape_screener_data(ticker):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    urls = [
        f"https://www.screener.in/company/{ticker}/consolidated/",
        f"https://www.screener.in/company/{ticker}/"
    ]
    
    all_data = {}
    error = None
    
    for url in urls:
        try:
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find all sections containing tables
            sections = soup.find_all('section')
            data_found = False
            
            for section in sections:
                section_id = section.get('id')
                section_title = section.find('h2')
                section_name = section_title.text.strip().replace(' ', '').lower() if section_title else section_id or f"section{uuid.uuid4().hex[:8]}"
                
                tables = section.find_all('table', class_='data-table')
                for idx, table in enumerate(tables):
                    df = parse_table(table)
                    if df is not None and not df.empty and df.shape[1] >= 2 and df.shape[0] >= 1:
                        table_key = f"{section_name}table{idx + 1}" if len(tables) > 1 else section_name
                        all_data[table_key] = df
                        data_found = True
            
            # Scrape ratios section (usually in a div with class 'company-ratios')
            ratios_div = soup.find('ul', class_='company-ratios')
            if ratios_div:
                ratios_data = parse_ratios(ratios_div)
                if ratios_data is not None and not ratios_data.empty:
                    all_data['ratios'] = ratios_data
                    data_found = True
            
            if data_found:
                break  # Exit loop if valid data is found
            elif url == urls[0]:
                continue  # Try non-consolidated URL if no valid data found
                
        except requests.RequestException as e:
            error = f"Error fetching {url} for ticker {ticker}: {e}"
            if url == urls[1]:
                error = f"No valid data found for ticker {ticker}: {e}"
        
        if url == urls[1] and not all_data:
            error = f"No financial data tables or ratios found for ticker {ticker}"
    
    return all_data, error

# Function to parse table data into a DataFrame
def parse_table(table):
    headers = [th.text.strip() for th in table.find('thead').find_all('th')]
    if not headers or len(headers) < 2:  # Ensure at least 2 columns
        return None
    rows = []
    for tr in table.find('tbody').find_all('tr'):
        cells = [td.text.strip() for td in tr.find_all('td')]
        if cells and len(cells) == len(headers):  # Ensure row matches header length
            rows.append(cells)
    return pd.DataFrame(rows, columns=headers) if rows else None

# Function to parse ratios section into a DataFrame
def parse_ratios(ratios_div):
    rows = []
    headers = ['Metric', 'Value']
    for li in ratios_div.find_all('li'):
        name = li.find('span', class_='name')
        value = li.find('span', class_='value') or li.find('span', class_='number')
        if name and value:
            rows.append([name.text.strip(), value.text.strip()])
    return pd.DataFrame(rows, columns=headers) if rows else None

# Function to save DataFrames to CSV files
def save_to_csv(ticker, data_dict, folder="data"):
    os.makedirs(folder, exist_ok=True)
    for table_name, df in data_dict.items():
        # Normalize table name by replacing special characters and spaces with underscores
        normalized_table_name = table_name.lower().replace('&', '_').replace(' ', '_')
        
        # Map table names to desired file naming convention
        if 'profit' in normalized_table_name and 'loss' in normalized_table_name:
            file_key = f"{ticker}_profit_loss"
        elif 'quarterly' in normalized_table_name and 'results' in normalized_table_name:
            file_key = f"{ticker}_quarterly_results"
        else:
            file_key = f"{ticker}_{normalized_table_name}"
        
        filename = os.path.join(folder, f"{file_key}.csv")
        try:
            df.to_csv(filename, index=False)
            st.success(f"Saved {table_name} for {ticker} to {filename}")
        except Exception as e:
            st.error(f"Error saving {table_name} for {ticker}: {e}")

# Streamlit app
def main():
    st.title("Screener.in Data Scraper")
    st.write("Enter company tickers to scrape financial data and ratios from Screener.in")

    # Input for tickers
    ticker_input = st.text_area("Enter tickers (one per line or comma-separated):", placeholder="RELIANCE,INFY,TCS")
    ticker_list = [t.strip().upper() for t in ticker_input.replace(',', '\n').split('\n') if t.strip()]
    
    # Create a data folder
    data_folder = "data"
    
    if st.button("Scrape Data"):
        if not ticker_list:
            st.error("Please enter at least one ticker.")
            return
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, ticker in enumerate(ticker_list):
            status_text.text(f"Scraping data for {ticker} ({idx + 1}/{len(ticker_list)})...")
            data_dict, error = scrape_screener_data(ticker)
            
            if error:
                st.error(error)
            else:
                if data_dict:
                    save_to_csv(ticker, data_dict, data_folder)
                    for table_name, df in data_dict.items():
                        st.write(f"{ticker} - {table_name.replace('_', ' ').title()}")
                        st.dataframe(df)
                else:
                    st.warning(f"No data found for {ticker}")
            
            # Update progress
            progress_bar.progress((idx + 1) / len(ticker_list))
            time.sleep(2)  # Small delay to avoid overwhelming the server
        
        status_text.text("Scraping complete!")

if __name__ == "__main__":
    main()