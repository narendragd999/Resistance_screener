import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import pyperclip
import difflib

# Set page to wide mode
st.set_page_config(layout="wide")

# Function to scrape data from Screener.in
@st.cache_data
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

# Function to determine if company is finance or non-finance
def is_finance_company(quarterly_data):
    if quarterly_data is None:
        return False
    return "Financing Profit" in quarterly_data.iloc[:, 0].values

# Function to find row by partial, case-insensitive, or fuzzy match
def find_row(data, row_name, threshold=0.8):
    for index in data.index:
        if row_name.lower() in index.lower():
            return index
    matches = difflib.get_close_matches(row_name.lower(), [idx.lower() for idx in data.index], n=1, cutoff=threshold)
    return matches[0] if matches else None

# Function to clean numeric data
def clean_numeric(series):
    return pd.to_numeric(series.str.replace(',', ''), errors='coerce').fillna(0)

# Function to adjust Net Profit and Actual Income
def adjust_non_finance(data, is_finance):
    if is_finance:
        net_profit_row = find_row(data, "Net Profit") or find_row(data, "Profit after tax")
        actual_income_row = find_row(data, "Actual Income") or net_profit_row
        return (clean_numeric(data.loc[net_profit_row].iloc[1:]) if net_profit_row else None,
                clean_numeric(data.loc[actual_income_row].iloc[1:]) if actual_income_row else None)

    net_profit_row = find_row(data, "Net Profit") or find_row(data, "Profit after tax")
    actual_income_row = find_row(data, "Actual Income") or net_profit_row
    other_income_row = find_row(data, "Other Income")

    net_profit = clean_numeric(data.loc[net_profit_row].iloc[1:]) if net_profit_row else None
    actual_income = clean_numeric(data.loc[actual_income_row].iloc[1:]) if actual_income_row else net_profit
    other_income = clean_numeric(data.loc[other_income_row].iloc[1:]) if other_income_row else pd.Series(0, index=net_profit.index if net_profit is not None else [])

    adjusted_net_profit = net_profit - other_income if net_profit is not None else None
    adjusted_actual_income = actual_income - other_income if actual_income is not None else adjusted_net_profit

    return adjusted_net_profit, adjusted_actual_income

# Function to check if latest quarter/year is highest historically
def check_highest_historical(data, is_quarterly, is_finance):
    data = data.set_index('')
    adjusted_net_profit, adjusted_actual_income = adjust_non_finance(data, is_finance)
    raw_net_profit = clean_numeric(data.loc[find_row(data, "Net Profit") or find_row(data, "Profit after tax")].iloc[1:]) if find_row(data, "Net Profit") or find_row(data, "Profit after tax") else None
    raw_actual_income = clean_numeric(data.loc[find_row(data, "Actual Income") or find_row(data, "Net Profit") or find_row(data, "Profit after tax")].iloc[1:]) if find_row(data, "Actual Income") or find_row(data, "Net Profit") or find_row(data, "Profit after tax") else None

    results = {}
    for metric, values, prefix in [
        ("Net Profit (Adjusted)", adjusted_net_profit, ""),
        ("Actual Income (Adjusted)", adjusted_actual_income, ""),
        ("Net Profit (Raw)", raw_net_profit, "Raw "),
        ("Actual Income (Raw)", raw_actual_income, "Raw ")
    ]:
        if values is None or values.empty:
            results[f"{prefix}{metric}"] = "N/A"
            continue
        try:
            latest_value = values.iloc[-1]
            historical_values = values.iloc[:-1]
            if historical_values.empty:
                results[f"{prefix}{metric}"] = "N/A"
            else:
                is_highest = latest_value >= historical_values.max()
                results[f"{prefix}{metric}"] = "PASS" if is_highest else "FAIL"
        except Exception as e:
            st.error(f"Error checking {prefix}{metric} for {'quarterly' if is_quarterly else 'yearly'}: {e}")
            results[f"{prefix}{metric}"] = "N/A"
    return results

# Function to process quarterly data with Excel formulas
def process_quarterly_data(quarterly_data, is_finance):
    metrics = ["Revenue" if is_finance else "Sales", "Expenses", "Operating Profit", "Financing Profit",
               "OPM %", "Financing Margin %", "Other Income", "Interest", "Depreciation",
               "Profit before tax", "Tax %", "Net Profit", "Profit after tax", "EPS in Rs", "Actual Income"]

    data = quarterly_data[quarterly_data.iloc[:, 0].isin(metrics)].copy()
    data = data.set_index('')

    net_profit_row = find_row(data, "Net Profit") or find_row(data, "Profit after tax")
    actual_income_row = find_row(data, "Actual Income") or net_profit_row
    revenue_row = find_row(data, "Revenue" if is_finance else "Sales")

    if net_profit_row is None or revenue_row is None:
        st.warning(f"Missing required rows: Net Profit/Profit after tax ({net_profit_row}), "
                   f"{'Revenue' if is_finance else 'Sales'} ({revenue_row}). Skipping calculations.")
        return data.reset_index()

    try:
        adjusted_net_profit, adjusted_actual_income = adjust_non_finance(data, is_finance)
        revenue = clean_numeric(data.loc[revenue_row].iloc[1:])

        change_net_income = adjusted_net_profit.pct_change() * 100 if adjusted_net_profit is not None else pd.Series()
        change_revenue = revenue.pct_change() * 100
        change_actual_income = adjusted_actual_income.pct_change() * 100 if adjusted_actual_income is not None else change_net_income
        change_revenue_qoq = change_revenue

        change_net_income_row = pd.Series(['% Change Net Income'] + change_net_income.tolist(), index=data.columns)
        change_revenue_row = pd.Series(['% Change Revenue'] + change_revenue.tolist(), index=data.columns)
        change_revenue_qoq_row = pd.Series(['% Change Revenue QOQ'] + change_revenue_qoq.tolist(), index=data.columns)
        change_actual_income_row = pd.Series(['% Change QOQ INCOME'] + change_actual_income.tolist(), index=data.columns)

        data = data.reset_index()
        data = pd.concat([data, pd.DataFrame([change_net_income_row, change_revenue_row,
                                             change_revenue_qoq_row, change_actual_income_row])],
                         ignore_index=True)

        data = pd.concat([data, pd.DataFrame([['REVENUE', 'GREEN'] + [''] * (len(data.columns) - 2),
                                              ['NET INCOME', 'PINK'] + [''] * (len(data.columns) - 2)],
                                             columns=data.columns)],
                         ignore_index=True)
    except Exception as e:
        st.error(f"Error processing quarterly data: {e}")
        return data.reset_index()

    return data

# Function to process yearly data with Excel formulas
def process_yearly_data(yearly_data, is_finance):
    metrics = ["Revenue" if is_finance else "Sales", "Expenses", "Operating Profit", "Financing Profit",
               "OPM %", "Financing Margin %", "Other Income", "Interest", "Depreciation",
               "Profit before tax", "Tax %", "Net Profit", "Profit after tax", "EPS in Rs", "Actual Income"]

    data = yearly_data[yearly_data.iloc[:, 0].isin(metrics)].copy()
    data = data.set_index('')

    net_profit_row = find_row(data, "Net Profit") or find_row(data, "Profit after tax")
    actual_income_row = find_row(data, "Actual Income") or net_profit_row
    revenue_row = find_row(data, "Revenue" if is_finance else "Sales")

    if net_profit_row is None or revenue_row is None:
        st.warning(f"Missing required rows: Net Profit/Profit after tax ({net_profit_row}), "
                   f"{'Revenue' if is_finance else 'Sales'} ({revenue_row}). Skipping calculations.")
        return data.reset_index()

    try:
        adjusted_net_profit, adjusted_actual_income = adjust_non_finance(data, is_finance)
        revenue = clean_numeric(data.loc[revenue_row].iloc[1:])

        change_net_income = adjusted_net_profit.pct_change() * 100 if adjusted_net_profit is not None else pd.Series()
        change_revenue = revenue.pct_change() * 100
        change_actual_income = adjusted_actual_income.pct_change() * 100 if adjusted_actual_income is not None else change_net_income

        change_net_income_row = pd.Series(['% Change Net Income YOY'] + change_net_income.tolist(), index=data.columns)
        change_revenue_row = pd.Series(['% Change Revenue'] + change_revenue.tolist(), index=data.columns)
        change_actual_income_row = pd.Series(['% Change YOY INCOME'] + change_actual_income.tolist(), index=data.columns)

        data = data.reset_index()
        data = pd.concat([data, pd.DataFrame([change_net_income_row, change_revenue_row,
                                             change_actual_income_row])],
                         ignore_index=True)

        data = pd.concat([data, pd.DataFrame([['REVENUE', 'GREEN'] + [''] * (len(data.columns) - 2),
                                              ['NET INCOME', 'PINK'] + [''] * (len(data.columns) - 2)],
                                             columns=data.columns)],
                         ignore_index=True)
    except Exception as e:
        st.error(f"Error processing yearly data: {e}")
        return data.reset_index()

    return data

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

# Function to display PASS/FAIL results with color
def display_results(title, results):
    st.write(f"**{title}**")
    for metric, result in results.items():
        color = "green" if result == "PASS" else "red" if result == "FAIL" else "gray"
        st.markdown(f"- {metric}: <span style='color:{color}'>{result}</span>", unsafe_allow_html=True)

# Streamlit app
def main():
    st.title("Screener.in Financial Data Scraper")
    st.write("Enter an NSE ticker to scrape raw Quarterly Results and Profit & Loss tables, "
             "apply Excel formulas, check if latest results are historically highest, "
             "adjust Net Profit/Actual Income by discarding Other Income, "
             "download as CSV, and copy to clipboard. Tables are in wide mode.")

    # Initialize session state
    if 'quarterly_data' not in st.session_state:
        st.session_state.quarterly_data = None
    if 'yearly_data' not in st.session_state:
        st.session_state.yearly_data = None
    if 'ticker' not in st.session_state:
        st.session_state.ticker = ""

    ticker = st.text_input("Enter NSE Ticker (e.g., RELIANCE):", value=st.session_state.ticker).strip().upper()
    st.checkbox("Override: Treat as Finance Company", key="finance_override")

    if st.button("Scrape and Process Data"):
        if not ticker:
            st.error("Please enter a valid NSE ticker.")
            return

        with st.spinner("Scraping data..."):
            quarterly_data, yearly_data = scrape_screener_data(ticker)
            if quarterly_data is None or yearly_data is None:
                st.error("Failed to scrape data. Please check the ticker or try again.")
                return

            st.session_state.quarterly_data = quarterly_data
            st.session_state.yearly_data = yearly_data
            st.session_state.ticker = ticker

            save_to_csv(quarterly_data, yearly_data, ticker)

    # Display raw data and PASS/FAIL test
    if st.session_state.quarterly_data is not None:
        st.subheader("Raw Quarterly Results")
        st.dataframe(st.session_state.quarterly_data, use_container_width=True)
        if st.button("Copy Raw Quarterly Results to Clipboard"):
            copy_to_clipboard(st.session_state.quarterly_data, "Raw Quarterly Results")

        with st.spinner("Processing quarterly data..."):            
            is_finance = st.session_state.finance_override or is_finance_company(st.session_state.quarterly_data)
            processed_quarterly = process_quarterly_data(st.session_state.quarterly_data, is_finance)
            qoq_results = check_highest_historical(st.session_state.quarterly_data, True, is_finance)
            st.subheader(f"Processed Quarterly Data ({'Finance' if is_finance else 'Non-Finance'})")
            st.dataframe(processed_quarterly, use_container_width=True)
            display_results("QOQ Highest Historical Test", qoq_results)
            if st.button("Copy Processed Quarterly Data to Clipboard"):
                copy_to_clipboard(processed_quarterly, "Processed Quarterly Data")

    if st.session_state.yearly_data is not None:
        st.subheader("Raw Profit & Loss")
        st.dataframe(st.session_state.yearly_data, use_container_width=True)
        if st.button("Copy Raw Profit & Loss to Clipboard"):
            copy_to_clipboard(st.session_state.yearly_data, "Raw Profit & Loss")

        with st.spinner("Processing yearly data..."):
            is_finance = st.session_state.finance_override or is_finance_company(st.session_state.quarterly_data)
            processed_yearly = process_yearly_data(st.session_state.yearly_data, is_finance)
            yoy_results = check_highest_historical(st.session_state.yearly_data, False, is_finance)
            st.subheader(f"Processed Profit & Loss ({'Finance' if is_finance else 'Non-Finance'})")
            st.dataframe(processed_yearly, use_container_width=True)
            display_results("YOY Highest Historical Test", yoy_results)
            if st.button("Copy Processed Profit & Loss to Clipboard"):
                copy_to_clipboard(processed_yearly, "Processed Profit & Loss")

if __name__ == "__main__":
    main()