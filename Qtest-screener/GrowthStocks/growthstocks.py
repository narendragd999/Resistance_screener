import streamlit as st
import pandas as pd
import os
import pyperclip
import difflib
import re

# Set page to wide mode
st.set_page_config(layout="wide")

# Function to load data from CSV files
def load_from_csv(ticker):
    output_dir = 'output_tables'
    quarterly_file = os.path.join(output_dir, f'{ticker}_quarterly_results.csv')
    yearly_file = os.path.join(output_dir, f'{ticker}_profit_loss.csv')
    
    quarterly_data = None
    yearly_data = None
    error = None
    
    try:
        if os.path.exists(quarterly_file):
            quarterly_data = pd.read_csv(quarterly_file)
            if quarterly_data.empty or quarterly_data.shape[1] < 2:
                quarterly_data = None
                error = f"Invalid or empty quarterly CSV data for {ticker}"
            else:
                # Rename first column to '' if it exists
                if quarterly_data.columns[0] != '':
                    quarterly_data.columns = [''] + quarterly_data.columns[1:].tolist()
                print(f"Loaded quarterly data for {ticker} from {quarterly_file}, columns: {quarterly_data.columns.tolist()}")
        else:
            error = f"Quarterly CSV file not found for {ticker}"
    except Exception as e:
        error = f"Error loading quarterly CSV for {ticker}: {e}"
    
    try:
        if os.path.exists(yearly_file):
            yearly_data = pd.read_csv(yearly_file)
            if yearly_data.empty or yearly_data.shape[1] < 2:
                yearly_data = None
                error = f"Invalid or empty yearly CSV data for {ticker}" if not error else error
            else:
                # Rename first column to '' if it exists
                if yearly_data.columns[0] != '':
                    yearly_data.columns = [''] + yearly_data.columns[1:].tolist()
                print(f"Loaded yearly data for {ticker} from {yearly_file}, columns: {yearly_data.columns.tolist()}")
        else:
            error = f"Yearly CSV file not found for {ticker}" if not error else error
    except Exception as e:
        error = f"Error loading yearly CSV for {ticker}: {e}" if not error else error
    
    return quarterly_data, yearly_data, error

# Function to determine if company is finance or non-finance
def is_finance_company(quarterly_data):
    if quarterly_data is None or quarterly_data.empty:
        return False
    return "Financing Profit" in quarterly_data.iloc[:, 0].values

# Function to find row by partial, case-insensitive, or fuzzy match
def find_row(data, row_name, threshold=0.8):
    possible_names = [row_name, row_name.replace(" ", ""), "Consolidated " + row_name, row_name + " (Consolidated)"]
    for name in possible_names:
        for index in data.index:
            if name.lower() in index.lower():
                return index
    matches = difflib.get_close_matches(row_name.lower(), [idx.lower() for idx in data.index], n=1, cutoff=threshold)
    return matches[0] if matches else None

# Function to clean numeric data
def clean_numeric(series):
    if isinstance(series, pd.DataFrame):
        series = series.iloc[0]
    elif not isinstance(series, pd.Series):
        series = pd.Series(series)
    
    series = series.astype(str).str.replace(',', '', regex=False).str.replace('[^0-9.-]', '', regex=True)
    return pd.to_numeric(series, errors='coerce').fillna(0)

# Function to check if data is in ascending order
def is_ascending(series):
    if series is None or series.empty:
        return False
    return all(series[i] <= series[i + 1] for i in range(len(series) - 1))

# Function to process metrics for growth and ascending order
def process_metrics(data, is_finance, metrics, is_quarterly):
    if data is None or data.empty or '' not in data.columns:
        print(f"Error: Invalid data or no '' column, columns: {data.columns.tolist() if data is not None else 'None'}")
        return {metric: {'Growth (%)': 'N/A', 'Ascending': 'N/A'} for metric in metrics}

    data = data.set_index('')
    results = {}
    metric_names = {
        'Revenue': 'Revenue' if is_finance else 'Sales',
        'Profit before tax': 'Profit before tax',
        'Net Profit': 'Net Profit',
        'EPS in Rs': 'EPS in Rs',
        'Interest': 'Interest'
    }

    for metric in metrics:
        metric_name = metric_names.get(metric, metric)
        row_name = find_row(data, metric_name)
        if row_name is None:
            print(f"Metric {metric_name} not found for {'quarterly' if is_quarterly else 'yearly'} data")
            results[metric] = {'Growth (%)': 'N/A', 'Ascending': 'N/A'}
            continue

        try:
            values = clean_numeric(data.loc[row_name].iloc[1:])
            if values.empty or len(values) < (4 if is_quarterly else 3):
                print(f"Insufficient data for {metric_name}: {len(values)} periods")
                results[metric] = {'Growth (%)': 'N/A', 'Ascending': 'N/A'}
                continue

            # Calculate growth rates
            recent_values = values.iloc[-4:] if is_quarterly else values.iloc[-3:]
            growth_rates = recent_values.pct_change().dropna() * 100
            avg_growth = round(growth_rates.mean(), 2) if not growth_rates.empty else 'N/A'

            # Check if in ascending order
            is_asc = is_ascending(recent_values)
            results[metric] = {
                'Growth (%)': avg_growth,
                'Ascending': 'PASS' if is_asc else 'FAIL'
            }
            print(f"{'QOQ' if is_quarterly else 'YOY'} {metric}: Values={recent_values.tolist()}, Growth={avg_growth}, Ascending={is_asc}")
        except Exception as e:
            print(f"Error processing {metric_name} for {'quarterly' if is_quarterly else 'yearly'} data: {e}")
            results[metric] = {'Growth (%)': 'N/A', 'Ascending': 'N/A'}

    return results

# Function to screen tickers for growth and ascending order
def screen_tickers(tickers, selected_metrics):
    screener_data = []
    output_dir = 'output_tables'

    for ticker in tickers:
        ticker = ticker.strip().upper()
        quarterly_data, yearly_data, error = load_from_csv(ticker)

        if error:
            print(f"Error for {ticker}: {error}")
            screener_data.append({
                'Ticker': ticker,
                'Company Type': 'N/A',
                **{f"QOQ {metric} Growth (%)": 'N/A' for metric in selected_metrics},
                **{f"QOQ {metric} Ascending": 'N/A' for metric in selected_metrics},
                **{f"YOY {metric} Growth (%)": 'N/A' for metric in selected_metrics},
                **{f"YOY {metric} Ascending": 'N/A' for metric in selected_metrics},
                'Error': error
            })
            continue

        is_finance = st.session_state.get('finance_override', False) or is_finance_company(quarterly_data)
        qoq_results = process_metrics(quarterly_data, is_finance, selected_metrics, is_quarterly=True)
        yoy_results = process_metrics(yearly_data, is_finance, selected_metrics, is_quarterly=False)

        row_data = {
            'Ticker': ticker,
            'Company Type': 'Finance' if is_finance else 'Non-Finance',
            'Error': 'None'
        }

        for metric in selected_metrics:
            row_data[f"QOQ {metric} Growth (%)"] = qoq_results.get(metric, {}).get('Growth (%)', 'N/A')
            row_data[f"QOQ {metric} Ascending"] = qoq_results.get(metric, {}).get('Ascending', 'N/A')
            row_data[f"YOY {metric} Growth (%)"] = yoy_results.get(metric, {}).get('Growth (%)', 'N/A')
            row_data[f"YOY {metric} Ascending"] = yoy_results.get(metric, {}).get('Ascending', 'N/A')

        screener_data.append(row_data)

        # Update session state
        st.session_state.quarterly_data[ticker] = quarterly_data
        st.session_state.yearly_data[ticker] = yearly_data

    return pd.DataFrame(screener_data)

# Function to copy DataFrame to clipboard
def copy_to_clipboard(df, table_name):
    try:
        df_string = df.to_csv(sep='\t', index=False)
        pyperclip.copy(df_string)
        st.success(f"{table_name} copied to clipboard! Paste into Excel or Google Sheets.")
    except Exception as e:
        st.error(f"Error copying {table_name} to clipboard: {e}. Please try again or check clipboard permissions.")

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

# Streamlit app
def main():
    st.title("Stock Screener for QoQ and YoY Growth")
    st.write("Enter NSE tickers (comma-separated, e.g., RELIANCE,HDFCBANK) or upload a CSV file with a 'Ticker' column or tickers in the first column. "
             "The app loads Quarterly Results and Profit & Loss tables from CSVs in the 'output_tables' folder. "
             "Select metrics (Revenue, Profit before tax, Net Profit, EPS in Rs, Interest for finance companies) to screen for QoQ (last 4 quarters) and YoY (last 3 years) growth in ascending order. "
             "Results are displayed in a summary table, saved as CSV, and copied to the clipboard. Tickers meeting all PASS criteria (ascending order for all selected metrics) are shown separately. "
             "Debug logs are printed for TATAELXSI.")

    # Initialize session state
    if 'quarterly_data' not in st.session_state:
        st.session_state.quarterly_data = {}
    if 'yearly_data' not in st.session_state:
        st.session_state.yearly_data = {}
    if 'tickers' not in st.session_state:
        st.session_state.tickers = ""
    if 'screener_df' not in st.session_state:
        st.session_state.screener_df = None

    # Input options
    st.subheader("Input Tickers and Metrics")
    tickers_input = st.text_input("Enter NSE Tickers (comma-separated, e.g., RELIANCE,HDFCBANK):", value=st.session_state.tickers)
    uploaded_file = st.file_uploader("Or Upload CSV File with Tickers", type=["csv"])
    st.checkbox("Override: Treat as Finance Company", key="finance_override")

    st.subheader("Select Metrics to Screen")
    metrics = ['Revenue', 'Profit before tax', 'Net Profit', 'EPS in Rs']
    if st.session_state.get('finance_override', False):
        metrics.append('Interest')
    selected_metrics = []
    for metric in metrics:
        if st.checkbox(f"Screen for {metric}", value=(metric == 'Revenue')):
            selected_metrics.append(metric)

    if not selected_metrics:
        st.error("Please select at least one metric to screen.")
        return

    # Process tickers
    if st.button("Screen Tickers"):
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

        with st.spinner("Screening tickers..."):
            st.session_state.screener_df = screen_tickers(tickers, selected_metrics)

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

            # Create filtered table for tickers with all PASS criteria
            pass_columns = [f"QOQ {metric} Ascending" for metric in selected_metrics] + \
                           [f"YOY {metric} Ascending" for metric in selected_metrics]
            pass_df = st.session_state.screener_df[st.session_state.screener_df[pass_columns].eq('PASS').all(axis=1)]
            if not pass_df.empty:
                st.subheader("Tickers with All PASS Criteria (Ascending for All Metrics)")
                display_columns = ['Ticker', 'Company Type'] + \
                                 [f"QOQ {metric} Growth (%)" for metric in selected_metrics] + \
                                 [f"QOQ {metric} Ascending" for metric in selected_metrics] + \
                                 [f"YOY {metric} Growth (%)" for metric in selected_metrics] + \
                                 [f"YOY {metric} Ascending" for metric in selected_metrics]
                st.dataframe(pass_df[display_columns], use_container_width=True)
                if st.button("Copy All PASS Tickers to Clipboard"):
                    copy_to_clipboard(pass_df[display_columns], "All PASS Tickers")
                pass_csv = os.path.join(output_dir, 'all_pass_tickers.csv')
                pass_df.to_csv(pass_csv, index=False)
                st.success(f"All PASS Tickers saved to '{pass_csv}'")
            else:
                st.info("No tickers meet all PASS criteria for ascending order of selected metrics.")

    # Display individual ticker data
    if st.session_state.quarterly_data:
        selected_ticker = st.selectbox("Select Ticker to View Details", list(st.session_state.quarterly_data.keys()))
        if selected_ticker:
            quarterly_data = st.session_state.quarterly_data.get(selected_ticker)
            yearly_data = st.session_state.yearly_data.get(selected_ticker)

            if quarterly_data is not None and not quarterly_data.empty:
                st.subheader(f"Raw Quarterly Results - {selected_ticker}")
                st.dataframe(quarterly_data, use_container_width=True)
                if st.button(f"Copy Raw Quarterly Results - {selected_ticker}"):
                    copy_to_clipboard(quarterly_data, f"Raw Quarterly Results - {selected_ticker}")

            if yearly_data is not None and not yearly_data.empty:
                st.subheader(f"Raw Profit & Loss - {selected_ticker}")
                st.dataframe(yearly_data, use_container_width=True)
                if st.button(f"Copy Raw Profit & Loss - {selected_ticker}"):
                    copy_to_clipboard(yearly_data, f"Raw Profit & Loss - {selected_ticker}")

if __name__ == "__main__":
    main()