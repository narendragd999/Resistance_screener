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
        return None, None, f"Error fetching URL for ticker {ticker}: {e}"

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

    return quarterly_data, yearly_data, None

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
    # Convert series to string type safely, handling non-string values
    series = series.astype(str).str.replace(',', '', regex=False)
    # Convert to numeric, coercing errors to NaN, then fill NaN with 0
    return pd.to_numeric(series, errors='coerce').fillna(0)

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
        return data.reset_index()

    return data

# Function to save tables to CSV
def save_to_csv(quarterly_data, yearly_data, ticker):
    output_dir = 'output_tables'
    os.makedirs(output_dir, exist_ok=True)

    if quarterly_data is not None:
        quarterly_data.to_csv(os.path.join(output_dir, f'{ticker}_quarterly_results.csv'), index=False)

    if yearly_data is not None:
        yearly_data.to_csv(os.path.join(output_dir, f'{ticker}_profit_loss.csv'), index=False)

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

# Function to read tickers from CSV
def read_tickers_from_csv(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        if 'Ticker' in df.columns:
            tickers = df['Ticker'].dropna().str.strip().tolist()
        else:
            tickers = df.iloc[:, 0].dropna().str.strip().tolist()
        return tickers, None
    except Exception as e:
        return [], f"Error reading CSV file: {e}"

# Function to process multiple tickers and create screener
def process_multiple_tickers(tickers):
    screener_data = []
    for ticker in tickers:
        ticker = ticker.strip().upper()
        quarterly_data, yearly_data, error = scrape_screener_data(ticker)
        if error:
            screener_data.append({
                'Ticker': ticker,
                'Company Type': 'N/A',
                'QOQ Net Profit (Adjusted)': 'N/A',
                'QOQ Actual Income (Adjusted)': 'N/A',
                'QOQ Net Profit (Raw)': 'N/A',
                'QOQ Actual Income (Raw)': 'N/A',
                'YOY Net Profit (Adjusted)': 'N/A',
                'YOY Actual Income (Adjusted)': 'N/A',
                'YOY Net Profit (Raw)': 'N/A',
                'YOY Actual Income (Raw)': 'N/A',
                'Error': error
            })
            continue

        is_finance = st.session_state.finance_override or is_finance_company(quarterly_data)
        qoq_results = check_highest_historical(quarterly_data, True, is_finance) if quarterly_data is not None else {}
        yoy_results = check_highest_historical(yearly_data, False, is_finance) if yearly_data is not None else {}

        screener_data.append({
            'Ticker': ticker,
            'Company Type': 'Finance' if is_finance else 'Non-Finance',
            'QOQ Net Profit (Adjusted)': qoq_results.get('Net Profit (Adjusted)', 'N/A'),
            'QOQ Actual Income (Adjusted)': qoq_results.get('Actual Income (Adjusted)', 'N/A'),
            'QOQ Net Profit (Raw)': qoq_results.get('Raw Net Profit (Raw)', 'N/A'),
            'QOQ Actual Income (Raw)': qoq_results.get('Raw Actual Income (Raw)', 'N/A'),
            'YOY Net Profit (Adjusted)': yoy_results.get('Net Profit (Adjusted)', 'N/A'),
            'YOY Actual Income (Adjusted)': yoy_results.get('Actual Income (Adjusted)', 'N/A'),
            'YOY Net Profit (Raw)': yoy_results.get('Raw Net Profit (Raw)', 'N/A'),
            'YOY Actual Income (Raw)': yoy_results.get('Raw Actual Income (Raw)', 'N/A'),
            'Error': 'None'
        })

        save_to_csv(quarterly_data, yearly_data, ticker)

    return pd.DataFrame(screener_data)

# Streamlit app
def main():
    st.title("Screener.in Financial Data Screener")
    st.write("Enter NSE tickers (comma-separated, e.g., RELIANCE,HDFCBANK) or upload a CSV file with a 'Ticker' column or tickers in the first column. "
             "The app scrapes Quarterly Results and Profit & Loss tables, applies Excel formulas, checks if latest results are historically highest "
             "(both raw and adjusted), downloads as CSV, and copies to clipboard. View screener-like summary for all tickers.")

    # Initialize session state
    if 'quarterly_data' not in st.session_state:
        st.session_state.quarterly_data = {}
    if 'yearly_data' not in st.session_state:
        st.session_state.yearly_data = {}
    if 'tickers' not in st.session_state:
        st.session_state.tickers = ""
    if 'screener_df' not in st.session_state:
        st.session_state.screener_df = None
    if 'finance_override' not in st.session_state:
        st.session_state.finance_override = False

    # Input options
    st.subheader("Input Tickers")
    tickers_input = st.text_input("Enter NSE Tickers (comma-separated, e.g., RELIANCE,HDFCBANK):", value=st.session_state.tickers)
    uploaded_file = st.file_uploader("Or Upload CSV File with Tickers", type=["csv"])
    st.checkbox("Override: Treat as Finance Company", key="finance_override")

    if st.button("Scrape and Process Data"):
        tickers = []
        errors = []

        # Process text input
        if tickers_input:
            tickers.extend([ticker.strip() for ticker in tickers_input.split(',')])

        # Process CSV input
        if uploaded_file:
            csv_tickers, csv_error = read_tickers_from_csv(uploaded_file)
            if csv_error:
                st.error(csv_error)
                errors.append(csv_error)
            tickers.extend(csv_tickers)

        # Remove duplicates and empty tickers
        tickers = list(set([t for t in tickers if t]))

        if not tickers:
            st.error("Please provide at least one valid NSE ticker via text input or CSV file.")
            return

        st.session_state.tickers = tickers_input

        with st.spinner("Scraping and processing data for all tickers..."):
            st.session_state.screener_df = process_multiple_tickers(tickers)
            st.session_state.quarterly_data = {}
            st.session_state.yearly_data = {}
            for ticker in tickers:
                ticker = ticker.strip().upper()
                quarterly_data, yearly_data, error = scrape_screener_data(ticker)
                if not error:
                    st.session_state.quarterly_data[ticker] = quarterly_data
                    st.session_state.yearly_data[ticker] = yearly_data
                if error:
                    errors.append(error)

        if errors:
            st.warning("Some tickers encountered errors during processing. Check the 'Error' column in the screener summary.")

        if st.session_state.screener_df is not None:
            st.subheader("Screener Summary")
            st.dataframe(st.session_state.screener_df, use_container_width=True)
            if st.button("Copy Screener Summary to Clipboard"):
                copy_to_clipboard(st.session_state.screener_df, "Screener Summary")
            output_dir = 'output_tables'
            os.makedirs(output_dir, exist_ok=True)
            screener_csv = os.path.join(output_dir, 'screener_summary.csv')
            st.session_state.screener_df.to_csv(screener_csv, index=False)
            st.success(f"Screener Summary saved to '{screener_csv}'")

    # Display individual ticker data
    if st.session_state.quarterly_data:
        selected_ticker = st.selectbox("Select Ticker to View Details", list(st.session_state.quarterly_data.keys()))
        if selected_ticker:
            quarterly_data = st.session_state.quarterly_data.get(selected_ticker)
            yearly_data = st.session_state.yearly_data.get(selected_ticker)

            if quarterly_data is not None:
                st.subheader(f"Raw Quarterly Results - {selected_ticker}")
                st.dataframe(quarterly_data, use_container_width=True)
                if st.button(f"Copy Raw Quarterly Results - {selected_ticker}"):
                    copy_to_clipboard(quarterly_data, f"Raw Quarterly Results - {selected_ticker}")

                with st.spinner(f"Processing quarterly data for {selected_ticker}..."):
                    is_finance = st.session_state.finance_override or is_finance_company(quarterly_data)
                    processed_quarterly = process_quarterly_data(quarterly_data, is_finance)
                    qoq_results = check_highest_historical(quarterly_data, True, is_finance)
                    st.subheader(f"Processed Quarterly Data - {selected_ticker} ({'Finance' if is_finance else 'Non-Finance'})")
                    st.dataframe(processed_quarterly, use_container_width=True)
                    display_results(f"QOQ Highest Historical Test - {selected_ticker}", qoq_results)
                    if st.button(f"Copy Processed Quarterly Data - {selected_ticker}"):
                        copy_to_clipboard(processed_quarterly, f"Processed Quarterly Data - {selected_ticker}")

            if yearly_data is not None:
                st.subheader(f"Raw Profit & Loss - {selected_ticker}")
                st.dataframe(yearly_data, use_container_width=True)
                if st.button(f"Copy Raw Profit & Loss - {selected_ticker}"):
                    copy_to_clipboard(yearly_data, f"Raw Profit & Loss - {selected_ticker}")

                with st.spinner(f"Processing yearly data for {selected_ticker}..."):
                    is_finance = st.session_state.finance_override or is_finance_company(quarterly_data)
                    processed_yearly = process_yearly_data(yearly_data, is_finance)
                    yoy_results = check_highest_historical(yearly_data, False, is_finance)
                    st.subheader(f"Processed Profit & Loss - {selected_ticker} ({'Finance' if is_finance else 'Non-Finance'})")
                    st.dataframe(processed_yearly, use_container_width=True)
                    display_results(f"YOY Highest Historical Test - {selected_ticker}", yoy_results)
                    if st.button(f"Copy Processed Profit & Loss - {selected_ticker}"):
                        copy_to_clipboard(processed_yearly, f"Processed Profit & Loss - {selected_ticker}")

if __name__ == "__main__":
    main()