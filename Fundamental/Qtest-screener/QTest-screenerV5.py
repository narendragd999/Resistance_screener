import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import pyperclip
import difflib
import re

# Set page to wide mode
st.set_page_config(layout="wide")

# Function to scrape data from Screener.in
@st.cache_data
def scrape_screener_data(ticker):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    urls = [
        f"https://www.screener.in/company/{ticker}/consolidated/",
        f"https://www.screener.in/company/{ticker}/"
    ]
    
    quarterly_data = None
    yearly_data = None
    error = None
    
    for url in urls:
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            tables = soup.find_all('table', class_='data-table')
            data_found = False
            
            for table in tables:
                tbody = table.find('tbody')
                if tbody and any(tr.find_all('td') for tr in tbody.find_all('tr')):
                    section = table.find_parent('section')
                    if section:
                        if section.get('id') == 'quarters':
                            quarterly_data = parse_table(table)
                            if (quarterly_data is not None and 
                                not quarterly_data.empty and
                                quarterly_data.shape[1] >= 2 and 
                                quarterly_data.shape[0] >= 1 and 
                                str(quarterly_data.iloc[:, 1].values[0]).strip() != ""):
                                data_found = True
                        elif section.get('id') == 'profit-loss':
                            yearly_data = parse_table(table)
                            if (yearly_data is not None and 
                                not yearly_data.empty and
                                yearly_data.shape[0] >= 1):
                                data_found = True
            
            if data_found:
                break
            elif url == urls[0]:
                continue
           
        except requests.RequestException as e:
            error = f"Error fetching {url} for ticker {ticker}: {e}"
        
        if url == urls[1] and quarterly_data is None and yearly_data is None:
            error = f"No financial data tables with valid data found for ticker {ticker}"
    print(f"Scraped data from {url} for {ticker}") 
    return quarterly_data, yearly_data, error

# Function to parse table data into a DataFrame
def parse_table(table):
    headers = [th.text.strip() for th in table.find('thead').find_all('th')]
    if not headers or len(headers) < 2:
        return None
    headers[0] = ''
    rows = []
    for tr in table.find('tbody').find_all('tr'):
        cells = [td.text.strip() for td in tr.find_all('td')]
        if cells and len(cells) == len(headers):
            rows.append(cells)
    df = pd.DataFrame(rows, columns=headers) if rows else None
    print(f"Parsed table columns: {df.columns.tolist() if df is not None else 'None'}")
    return df

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
                if quarterly_data.columns[0] != '':
                    quarterly_data.columns = [''] + quarterly_data.columns[1:].tolist()
                print(f"Loaded quarterly data from {quarterly_file}, columns: {quarterly_data.columns.tolist()}")
        else:
            error = f"Quarterly CSV file not found for {ticker}"
    except Exception as e:
        error = f"Error loading quarterly CSV for {ticker}: {e}"
    
    try:
        if os.path.exists(yearly_file):
            yearly_data = pd.read_csv(yearly_file)
            if yearly_data.empty or yearly_data.shape[1] < 2:
                yearly_data = None
                error = f"Invalid or empty yearly CSV data for {ticker}"
            else:
                if yearly_data.columns[0] != '':
                    yearly_data.columns = [''] + yearly_data.columns[1:].tolist()
                print(f"Loaded yearly data from {yearly_file}, columns: {yearly_data.columns.tolist()}")
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
    series = series.astype(str).str.replace(',', '', regex=False).str.replace('[^0-9.-]', '', regex=True)
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

    print(f"Net Profit: {net_profit}\nOther Income: {other_income}\nAdjusted Net Profit: {adjusted_net_profit}")
    return adjusted_net_profit, adjusted_actual_income

# Function to check if data is in ascending order
def is_ascending(series):
    if series is None or series.empty:
        return False
    return all(series[i] <= series[i + 1] for i in range(len(series) - 1))

# Function to extract quarter and year from column name
def extract_quarter_year(column):
    patterns = [
        r'(\w+)\s+(\d{4})',  # e.g., "Mar 2025"
        r'(\w+)-(\d{2})',    # e.g., "Mar-25"
        r'(\w+)\s*\'(\d{2})' # e.g., "Mar'25"
    ]
    column = column.strip()
    for pattern in patterns:
        match = re.match(pattern, column)
        if match:
            quarter, year = match.groups()
            year = int(year) if len(year) == 4 else int("20" + year)
            return quarter, year
    return None, None

# Function to suggest tickers with improving net profit and revenue
def suggest_improving_tickers(tickers, force_refresh, min_qoq_quarters, min_yoy_years):
    suggestions = []
    output_dir = 'output_tables'
    
    for ticker in tickers:
        ticker = ticker.strip().upper()
        quarterly_data, yearly_data, error = None, None, None

        if not force_refresh:
            quarterly_data, yearly_data, error = load_from_csv(ticker)
        else:
            quarterly_data, yearly_data, error = None, None, None

        if error or quarterly_data is None or yearly_data is None:
            quarterly_data, yearly_data, error = scrape_screener_data(ticker)
            if not error:
                save_to_csv(quarterly_data, yearly_data, ticker)
            else:
                print(f"Error processing {ticker}: {error}")
                suggestions.append({
                    'Ticker': ticker,
                    'QoQ Net Profit Growth (%)': 'N/A',
                    'QoQ Compared Quarters': 'N/A',
                    'YoY Net Profit Growth (%)': 'N/A',
                    'YoY Compared Years': 'N/A',
                    'QoQ Revenue Growth (%)': 'N/A',
                    'YoY Revenue Growth (%)': 'N/A',
                    'Recovered from Harsh Condition': 'N/A',
                    'Error': error
                })
                continue

        is_finance = st.session_state.get('finance_override', False) or is_finance_company(quarterly_data)
        
        qoq_compared_quarters = 'N/A'
        yoy_compared_years = 'N/A'
        qoq_np_growth = 'N/A'
        qoq_rev_growth = 'N/A'
        qoq_harsh = False
        
        # Process quarterly data
        if quarterly_data is not None and not quarterly_data.empty and '' in quarterly_data.columns:
            try:
                quarterly_data = quarterly_data.set_index('')
                adjusted_net_profit, _ = adjust_non_finance(quarterly_data, is_finance)
                revenue_row = find_row(quarterly_data, "Revenue" if is_finance else "Sales")
                revenue = clean_numeric(quarterly_data.loc[revenue_row].iloc[1:]) if revenue_row else None

                if adjusted_net_profit is not None and len(adjusted_net_profit) >= min_qoq_quarters:
                    recent_quarters = quarterly_data.columns[-min_qoq_quarters:]
                    qoq_compared_quarters = ', '.join(recent_quarters)
                    print(f"{ticker} Recent Net Profits for QoQ: {adjusted_net_profit.tolist()} for quarters {qoq_compared_quarters}")
                    recent_profits = adjusted_net_profit.iloc[-min_qoq_quarters:]
                    if is_ascending(recent_profits):
                        if len(recent_profits) >= 2:
                            latest_profit = recent_profits.iloc[-1]
                            prev_profit = recent_profits.iloc[-2]
                            qoq_np_growth = round(((latest_profit - prev_profit) / prev_profit) * 100, 2) if prev_profit != 0 else 'N/A'
                            print(f"{ticker} QoQ Growth Calculation: ({latest_profit} - {prev_profit}) / {prev_profit} * 100 = {qoq_np_growth}%")
                        else:
                            qoq_np_growth = 'Need at least 2 quarters'
                        all_profits = adjusted_net_profit
                        if any(all_profits.iloc[:-min_qoq_quarters] <= 0) or any(all_profits.iloc[:-min_qoq_quarters] < 0.1 * recent_profits.iloc[-1]):
                            qoq_harsh = True
                    else:
                        qoq_np_growth = 'Not ascending'
                else:
                    qoq_np_growth = f'Need {min_qoq_quarters} quarters'

                if revenue is not None and len(revenue) >= min_qoq_quarters:
                    recent_revenue = revenue.iloc[-min_qoq_quarters:]
                    if is_ascending(recent_revenue):
                        if len(recent_revenue) >= 2:
                            latest_revenue = recent_revenue.iloc[-1]
                            prev_revenue = recent_revenue.iloc[-2]
                            qoq_rev_growth = round(((latest_revenue - prev_revenue) / prev_revenue) * 100, 2) if prev_revenue != 0 else 'N/A'
                        else:
                            qoq_rev_growth = 'Need at least 2 quarters'
                    else:
                        qoq_rev_growth = 'Not ascending'
                else:
                    qoq_rev_growth = f'Need {min_qoq_quarters} quarters'

                print(f"{ticker} QoQ Net Profit: {adjusted_net_profit.tolist() if adjusted_net_profit is not None else 'None'}, Growth: {qoq_np_growth}, Harsh: {qoq_harsh}, Quarters: {qoq_compared_quarters}")
                print(f"{ticker} QoQ Revenue: {revenue.tolist() if revenue is not None else 'None'}, Growth: {qoq_rev_growth}")
            except Exception as e:
                print(f"Error processing quarterly data for {ticker}: {e}")
                qoq_np_growth = 'Error'
                qoq_rev_growth = 'Error'

        # Process yearly data
        yoy_np_growth = 'N/A'
        yoy_rev_growth = 'N/A'
        yoy_harsh = False
        if yearly_data is not None and not yearly_data.empty and '' in yearly_data.columns:  # Fixed: Check yearly_data.columns
            try:
                yearly_data = yearly_data.set_index('')
                adjusted_net_profit, _ = adjust_non_finance(yearly_data, is_finance)
                revenue_row = find_row(yearly_data, "Revenue" if is_finance else "Sales")
                revenue = clean_numeric(yearly_data.loc[revenue_row].iloc[1:]) if revenue_row else None

                if adjusted_net_profit is not None and len(adjusted_net_profit) >= min_yoy_years:
                    recent_years = yearly_data.columns[-min_yoy_years:]
                    yoy_compared_years = ', '.join(recent_years)
                    recent_profits = adjusted_net_profit.iloc[-min_yoy_years:]
                    if is_ascending(recent_profits):
                        growth_rates = recent_profits.pct_change().dropna() * 100
                        yoy_np_growth = round(growth_rates.mean(), 2)
                        all_profits = adjusted_net_profit
                        if any(all_profits.iloc[:-min_yoy_years] <= 0) or any(all_profits.iloc[:-min_yoy_years] < 0.1 * recent_profits.iloc[-1]):
                            yoy_harsh = True
                    else:
                        yoy_np_growth = 'Not ascending'
                else:
                    yoy_np_growth = f'Need {min_yoy_years} years'

                if revenue is not None and len(revenue) >= min_yoy_years:
                    recent_revenue = revenue.iloc[-min_yoy_years:]
                    if is_ascending(recent_revenue):
                        growth_rates = recent_revenue.pct_change().dropna() * 100
                        yoy_rev_growth = round(growth_rates.mean(), 2)
                    else:
                        yoy_rev_growth = 'Not ascending'
                else:
                    yoy_rev_growth = f'Need {min_yoy_years} years'

                print(f"{ticker} YoY Net Profit: {adjusted_net_profit.tolist() if adjusted_net_profit is not None else 'None'}, Growth: {yoy_np_growth}, Harsh: {yoy_harsh}, Years: {yoy_compared_years}")
                print(f"{ticker} YoY Revenue: {revenue.tolist() if revenue is not None else 'None'}, Growth: {yoy_rev_growth}")
            except Exception as e:
                print(f"Error processing yearly data for {ticker}: {e}")
                yoy_np_growth = 'Error'
                yoy_rev_growth = 'Error'

        # Add to suggestions
        suggestion = {
            'Ticker': ticker,
            'QoQ Net Profit Growth (%)': qoq_np_growth,
            'QoQ Compared Quarters': qoq_compared_quarters,
            'YoY Net Profit Growth (%)': yoy_np_growth,
            'YoY Compared Years': yoy_compared_years,
            'QoQ Revenue Growth (%)': qoq_rev_growth,
            'YoY Revenue Growth (%)': yoy_rev_growth,
            'Recovered from Harsh Condition': 'Yes' if (isinstance(qoq_np_growth, (int, float)) and
                                                       isinstance(yoy_np_growth, (int, float)) and
                                                       isinstance(qoq_rev_growth, (int, float)) and
                                                       isinstance(yoy_rev_growth, (int, float)) and
                                                       qoq_harsh and yoy_harsh) else 'No',
            'Error': 'None' if (isinstance(qoq_np_growth, (int, float)) and
                                isinstance(yoy_np_growth, (int, float)) and
                                isinstance(qoq_rev_growth, (int, float)) and
                                isinstance(yoy_rev_growth, (int, float)) and
                                qoq_harsh and yoy_harsh) else ('Processing error' if qoq_np_growth == 'Error' or yoy_np_growth == 'Error' or qoq_rev_growth == 'Error' or yoy_rev_growth == 'Error' else 'Did not meet improvement criteria' if qoq_harsh and yoy_harsh else 'No harsh condition recovery')
        }
        print(f"{ticker} Suggestion Entry: {suggestion}")
        suggestions.append(suggestion)

        st.session_state.quarterly_data[ticker] = quarterly_data
        st.session_state.yearly_data[ticker] = yearly_data

    return pd.DataFrame(suggestions)
# Function to check same-quarter comparison
def check_same_quarter_comparison(data, enable_same_quarter):
    if not enable_same_quarter or data is None or data.empty:
        return {
            'Same Quarter Net Profit (Adjusted)': 'N/A',
            'Same Quarter Net Profit (Raw)': 'N/A'
        }
    
    if '' not in data.columns:
        print(f"Error: No '' column in data, columns: {data.columns.tolist()}")
        return {
            'Same Quarter Net Profit (Adjusted)': 'N/A',
            'Same Quarter Net Profit (Raw)': 'N/A'
        }
    
    data = data.set_index('')
    adjusted_net_profit, _ = adjust_non_finance(data, is_finance_company(data))
    raw_net_profit = clean_numeric(data.loc[find_row(data, "Net Profit") or find_row(data, "Profit after tax")].iloc[1:]) if find_row(data, "Net Profit") or find_row(data, "Profit after tax") else None

    results = {
        'Same Quarter Net Profit (Adjusted)': 'N/A',
        'Same Quarter Net Profit (Raw)': 'N/A'
    }

    if adjusted_net_profit is None or raw_net_profit is None:
        return results

    try:
        latest_column = data.columns[-1]
        latest_quarter, latest_year = extract_quarter_year(latest_column)
        if latest_quarter is None or latest_year is None:
            return results

        prev_year_column = None
        for col in data.columns[:-1]:
            quarter, year = extract_quarter_year(col)
            if quarter == latest_quarter and year == latest_year - 1:
                prev_year_column = col
                break

        if prev_year_column:
            latest_adj_np = adjusted_net_profit[latest_column]
            prev_adj_np = adjusted_net_profit[prev_year_column]
            latest_raw_np = raw_net_profit[latest_column]
            prev_raw_np = raw_net_profit[prev_year_column]

            results['Same Quarter Net Profit (Adjusted)'] = 'PASS' if latest_adj_np >= prev_adj_np else 'FAIL'
            results['Same Quarter Net Profit (Raw)'] = 'PASS' if latest_raw_np >= prev_raw_np else 'FAIL'
            print(f"Same Quarter Comparison: {latest_column} ({latest_adj_np}, {latest_raw_np}) vs {prev_year_column} ({prev_adj_np}, {prev_raw_np})")

    except Exception as e:
        print(f"Error in same-quarter comparison: {e}")
        pass

    return results

# Function to check if latest quarter/year is highest and in ascending order
def check_highest_historical(data, is_quarterly, is_finance):
    if data is None or data.empty:
        return {}
    
    print(f"Checking historical data, columns: {data.columns.tolist()}")
    if '' not in data.columns:
        print(f"Error: No '' column in data, cannot set index")
        return {
            'Net Profit (Adjusted)': 'N/A',
            'Actual Income (Adjusted)': 'N/A',
            'Raw Net Profit (Raw)': 'N/A',
            'Raw Actual Income (Raw)': 'N/A',
            'Net Profit (Adjusted) Ascending': 'N/A',
            'Actual Income (Adjusted) Ascending': 'N/A',
            'Raw Net Profit (Raw) Ascending': 'N/A',
            'Raw Actual Income (Raw) Ascending': 'N/A'
        }
    
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
            results[f"{prefix}{metric} Ascending"] = "N/A"
            continue
        try:
            latest_value = values.iloc[-1]
            historical_values = values.iloc[:-1]
            if historical_values.empty:
                results[f"{prefix}{metric}"] = "N/A"
                results[f"{prefix}{metric} Ascending"] = "N/A"
            else:
                is_highest = latest_value >= historical_values.max()
                results[f"{prefix}{metric}"] = "PASS" if is_highest else "FAIL"
                is_asc = is_ascending(values)
                results[f"{prefix}{metric} Ascending"] = "PASS" if is_asc else "FAIL"
                print(f"{prefix}{metric}: Latest={latest_value}, Historical Max={historical_values.max()}, Ascending={is_asc}")
        except Exception as e:
            print(f"Error in highest historical check for {prefix}{metric}: {e}")
            results[f"{prefix}{metric}"] = "N/A"
            results[f"{prefix}{metric} Ascending"] = "N/A"
    return results

# Function to process quarterly data with Excel formulas
def process_quarterly_data(quarterly_data, is_finance):
    if quarterly_data is None or quarterly_data.empty:
        return None
    
    metrics = ["Revenue" if is_finance else "Sales", "Expenses", "Operating Profit", "Financing Profit",
               "OPM %", "Financing Margin %", "Other Income", "Interest", "Depreciation",
               "Profit before tax", "Tax %", "Net Profit", "Profit after tax", "EPS in Rs", "Actual Income"]

    data = quarterly_data[quarterly_data.iloc[:, 0].isin(metrics)].copy()
    if data.empty:
        return None
    if '' not in data.columns:
        print(f"Error in process_quarterly_data: No '' column, columns: {data.columns.tolist()}")
        return data
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
    except Exception:
        return data.reset_index()

    return data

# Function to process yearly data with Excel formulas
def process_yearly_data(yearly_data, is_finance):
    if yearly_data is None or yearly_data.empty:
        return None
    
    metrics = ["Revenue" if is_finance else "Sales", "Expenses", "Operating Profit", "Financing Profit",
               "OPM %", "Financing Margin %", "Other Income", "Interest", "Depreciation",
               "Profit before tax", "Tax %", "Net Profit", "Profit after tax", "EPS in Rs", "Actual Income"]

    data = yearly_data[yearly_data.iloc[:, 0].isin(metrics)].copy()
    if data.empty:
        return None
    if '' not in data.columns:
        print(f"Error in process_yearly_data: No '' column, columns: {data.columns.tolist()}")
        return data
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
    except Exception:
        return data.reset_index()

    return data

# Function to save tables to CSV
def save_to_csv(quarterly_data, yearly_data, ticker):
    output_dir = 'output_tables'
    os.makedirs(output_dir, exist_ok=True)

    if quarterly_data is not None and not quarterly_data.empty:
        quarterly_data.to_csv(os.path.join(output_dir, f'{ticker}_quarterly_results.csv'), index=False)

    if yearly_data is not None and not yearly_data.empty:
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
def process_multiple_tickers(tickers, enable_same_quarter, force_refresh):
    screener_data = []
    for ticker in tickers:
        ticker = ticker.strip().upper()
        
        if not force_refresh:
            quarterly_data, yearly_data, error = load_from_csv(ticker)
        else:
            quarterly_data, yearly_data, error = None, None, None

        if error or quarterly_data is None or yearly_data is None:
            quarterly_data, yearly_data, error = scrape_screener_data(ticker)
            if not error:
                save_to_csv(quarterly_data, yearly_data, ticker)
                st.session_state.quarterly_data[ticker] = quarterly_data
                st.session_state.yearly_data[ticker] = yearly_data
            else:
                print(f"Error scraping {ticker}: {error}")

        if error:
            screener_data.append({
                'Ticker': ticker,
                'Company Type': 'N/A',
                'QOQ Net Profit (Adjusted)': 'N/A',
                'QOQ Actual Income (Adjusted)': 'N/A',
                'QOQ Net Profit (Raw)': 'N/A',
                'QOQ Actual Income (Raw)': 'N/A',
                'QOQ Net Profit Ascending (Adjusted)': 'N/A',
                'QOQ Actual Income Ascending (Adjusted)': 'N/A',
                'QOQ Net Profit Ascending (Raw)': 'N/A',
                'QOQ Actual Income Ascending (Raw)': 'N/A',
                'Same Quarter Net Profit (Adjusted)': 'N/A',
                'Same Quarter Net Profit (Raw)': 'N/A',
                'YOY Net Profit (Adjusted)': 'N/A',
                'YOY Actual Income (Adjusted)': 'N/A',
                'YOY Net Profit (Raw)': 'N/A',
                'YOY Actual Income (Raw)': 'N/A',
                'YOY Net Profit Ascending (Adjusted)': 'N/A',
                'YOY Actual Income Ascending (Adjusted)': 'N/A',
                'YOY Net Profit Ascending (Raw)': 'N/A',
                'YOY Actual Income Ascending (Raw)': 'N/A',
                'Error': error
            })
            continue

        is_finance = st.session_state.finance_override or is_finance_company(quarterly_data)
        qoq_results = check_highest_historical(quarterly_data, True, is_finance)
        same_quarter_results = check_same_quarter_comparison(quarterly_data, enable_same_quarter)
        yoy_results = check_highest_historical(yearly_data, False, is_finance)

        screener_data.append({
            'Ticker': ticker,
            'Company Type': 'Finance' if is_finance else 'Non-Finance',
            'QOQ Net Profit (Adjusted)': qoq_results.get('Net Profit (Adjusted)', 'N/A'),
            'QOQ Actual Income (Adjusted)': qoq_results.get('Actual Income (Adjusted)', 'N/A'),
            'QOQ Net Profit (Raw)': qoq_results.get('Raw Net Profit (Raw)', 'N/A'),
            'QOQ Actual Income (Raw)': qoq_results.get('Raw Actual Income (Raw)', 'N/A'),
            'QOQ Net Profit Ascending (Adjusted)': qoq_results.get('Net Profit (Adjusted) Ascending', 'N/A'),
            'QOQ Actual Income Ascending (Adjusted)': qoq_results.get('Actual Income (Adjusted) Ascending', 'N/A'),
            'QOQ Net Profit Ascending (Raw)': qoq_results.get('Raw Net Profit (Raw) Ascending', 'N/A'),
            'QOQ Actual Income Ascending (Raw)': qoq_results.get('Raw Actual Income (Raw) Ascending', 'N/A'),
            'Same Quarter Net Profit (Adjusted)': same_quarter_results.get('Same Quarter Net Profit (Adjusted)', 'N/A'),
            'Same Quarter Net Profit (Raw)': same_quarter_results.get('Same Quarter Net Profit (Raw)', 'N/A'),
            'YOY Net Profit (Adjusted)': yoy_results.get('Net Profit (Adjusted)', 'N/A'),
            'YOY Actual Income (Adjusted)': yoy_results.get('Actual Income (Adjusted)', 'N/A'),
            'YOY Net Profit (Raw)': yoy_results.get('Raw Net Profit (Raw)', 'N/A'),
            'YOY Actual Income (Raw)': yoy_results.get('Raw Actual Income (Raw)', 'N/A'),
            'YOY Net Profit Ascending (Adjusted)': yoy_results.get('Net Profit (Adjusted) Ascending', 'N/A'),
            'YOY Actual Income Ascending (Adjusted)': yoy_results.get('Actual Income (Adjusted) Ascending', 'N/A'),
            'YOY Net Profit Ascending (Raw)': yoy_results.get('Raw Net Profit (Raw) Ascending', 'N/A'),
            'YOY Actual Income Ascending (Raw)': yoy_results.get('Raw Actual Income (Raw) Ascending', 'N/A'),
            'Error': 'None'
        })

        st.session_state.quarterly_data[ticker] = quarterly_data
        st.session_state.yearly_data[ticker] = yearly_data

    return pd.DataFrame(screener_data)

# Streamlit app
# Streamlit app
def main():
    st.title("Financial Data Screener")
    st.write("Enter NSE tickers (comma-separated, e.g., RELIANCE,HDFCBANK) or upload a CSV file with a 'Ticker' column or tickers in the first column. "
             "The app processes Quarterly Results and Profit & Loss tables from previously saved CSVs in 'output_tables' if available, or scrapes from Screener.in if CSVs are missing or 'Force Refresh Data' is checked. "
             "It applies Excel formulas, checks if latest results are historically highest and in ascending order (both raw and adjusted), and optionally compares Net Profit of the same quarter year-over-year (e.g., Mar 2025 vs. Mar 2024). "
             "The 'Suggest Improving Tickers' feature identifies tickers with consistent QoQ and YoY net profit and revenue growth (user-specified quarters/years) from harsh conditions (negative or low past profits). "
             "It also displays the specific quarters and years compared for QoQ and YoY analyses. "
             "Results are downloaded as CSV and copied to clipboard. View screener summary, PASS criteria tickers, and suggested tickers. Debug logs are printed for TATAELXSI.")

    # Initialize session state
    if 'quarterly_data' not in st.session_state:
        st.session_state.quarterly_data = {}
    if 'yearly_data' not in st.session_state:
        st.session_state.yearly_data = {}
    if 'tickers' not in st.session_state:
        st.session_state.tickers = ""
    if 'screener_df' not in st.session_state:
        st.session_state.screener_df = None
    if 'suggested_df' not in st.session_state:
        st.session_state.suggested_df = None
    if 'finance_override' not in st.session_state:
        st.session_state.finance_override = False
    if 'enable_same_quarter' not in st.session_state:
        st.session_state.enable_same_quarter = False
    if 'force_refresh' not in st.session_state:
        st.session_state.force_refresh = False
    if 'min_qoq_quarters' not in st.session_state:
        st.session_state.min_qoq_quarters = 2  # Default to 2 for KAYNES
    if 'min_yoy_years' not in st.session_state:
        st.session_state.min_yoy_years = 2

    # Input options
    st.subheader("Input Tickers")
    tickers_input = st.text_input("Enter NSE Tickers (comma-separated, e.g., RELIANCE,HDFCBANK):", value=st.session_state.tickers)
    uploaded_file = st.file_uploader("Or Upload CSV File with Tickers", type=["csv"])
    st.checkbox("Override: Treat as Finance Company", key="finance_override")
    st.checkbox("Enable Same Quarter Year-over-Year Net Profit Comparison (e.g., Mar 2025 vs. Mar 2024)", key="enable_same_quarter")
    st.checkbox("Force Refresh Data (re-scrape all tickers instead of using CSVs)", key="force_refresh")

    # Process tickers (unchanged)
    if st.button("Scrape and Process Data"):
        tickers = []
        errors = []

        if tickers_input:
            tickers.extend([ticker.strip() for ticker in tickers_input.split(',')])
        if uploaded_file:
            csv_tickers, csv_error = read_tickers_from_csv(uploaded_file)
            if csv_error:
                st.error(csv_error)
                errors.append(csv_error)
            tickers.extend(csv_tickers)

        tickers = list(set([t for t in tickers if t]))

        if not tickers:
            st.error("Please provide at least one valid NSE ticker via text input or CSV file.")
            return

        st.session_state.tickers = tickers_input

        with st.spinner("Processing data for all tickers..."):
            st.session_state.screener_df = process_multiple_tickers(tickers, st.session_state.enable_same_quarter, st.session_state.force_refresh)

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

            pass_columns = [
                'QOQ Net Profit (Adjusted)', 'QOQ Actual Income (Adjusted)', 'QOQ Net Profit (Raw)', 'QOQ Actual Income (Raw)',
                'QOQ Net Profit Ascending (Adjusted)', 'QOQ Actual Income Ascending (Adjusted)', 'QOQ Net Profit Ascending (Raw)', 'QOQ Actual Income Ascending (Raw)',
                'YOY Net Profit (Adjusted)', 'YOY Actual Income (Adjusted)', 'YOY Net Profit (Raw)', 'YOY Actual Income (Raw)',
                'YOY Net Profit Ascending (Adjusted)', 'YOY Actual Income Ascending (Adjusted)', 'YOY Net Profit Ascending (Raw)', 'YOY Actual Income Ascending (Raw)'
            ]
            if st.session_state.enable_same_quarter:
                pass_columns.extend(['Same Quarter Net Profit (Adjusted)', 'Same Quarter Net Profit (Raw)'])
            pass_df = st.session_state.screener_df[st.session_state.screener_df[pass_columns].eq('PASS').all(axis=1)]
            if not pass_df.empty:
                st.subheader("Tickers with All PASS Criteria")
                st.dataframe(pass_df[['Ticker', 'Company Type'] + pass_columns], use_container_width=True)
                if st.button("Copy All PASS Tickers to Clipboard"):
                    copy_to_clipboard(pass_df[['Ticker', 'Company Type'] + pass_columns], "All PASS Tickers")
                pass_csv = os.path.join(output_dir, 'all_pass_tickers.csv')
                pass_df.to_csv(pass_csv, index=False)
                st.success(f"All PASS Tickers saved to '{pass_csv}'")
            else:
                st.info("No tickers meet all PASS criteria.")

    # Suggest improving tickers
    st.subheader("Suggest Improving Tickers")
    st.session_state.min_qoq_quarters = st.slider("Minimum Quarters for QoQ Net Profit and Revenue Growth", min_value=1, max_value=8, value=st.session_state.min_qoq_quarters)
    st.session_state.min_yoy_years = st.slider("Minimum Years for YoY Net Profit and Revenue Growth", min_value=1, max_value=5, value=st.session_state.min_yoy_years)
    
    if st.button("Analyze Improving Tickers"):
        # Clear cache to ensure fresh data
        st.cache_data.clear()
        tickers = []
        errors = []

        if tickers_input:
            tickers.extend([ticker.strip() for ticker in tickers_input.split(',')])
        if uploaded_file:
            csv_tickers, csv_error = read_tickers_from_csv(uploaded_file)
            if csv_error:
                st.error(csv_error)
                errors.append(csv_error)
            tickers.extend(csv_tickers)

        tickers = list(set([t for t in tickers if t]))

        if not tickers:
            st.error("Please provide at least one valid NSE ticker via text input or CSV file.")
            return

        with st.spinner(f"Analyzing tickers for improving net profit and revenue over {st.session_state.min_qoq_quarters} quarters and {st.session_state.min_yoy_years} years..."):
            st.session_state.suggested_df = suggest_improving_tickers(
                tickers, 
                st.session_state.force_refresh, 
                st.session_state.min_qoq_quarters, 
                st.session_state.min_yoy_years
            )

        if st.session_state.suggested_df is not None:
            st.subheader("Suggested Tickers with Improving Net Profit and Revenue")
            suggested_df = st.session_state.suggested_df[st.session_state.suggested_df['Recovered from Harsh Condition'] == 'Yes']
            if not suggested_df.empty:
                st.dataframe(suggested_df[['Ticker', 'QoQ Net Profit Growth (%)', 'QoQ Compared Quarters', 'YoY Net Profit Growth (%)', 'YoY Compared Years', 'QoQ Revenue Growth (%)', 'YoY Revenue Growth (%)', 'Recovered from Harsh Condition', 'Error']], use_container_width=True)
                if st.button("Copy Suggested Tickers to Clipboard"):
                    copy_to_clipboard(suggested_df[['Ticker', 'QoQ Net Profit Growth (%)', 'QoQ Compared Quarters', 'YoY Net Profit Growth (%)', 'YoY Compared Years', 'QoQ Revenue Growth (%)', 'YoY Revenue Growth (%)', 'Recovered from Harsh Condition', 'Error']], "Suggested Tickers")
                output_dir = 'output_tables'
                os.makedirs(output_dir, exist_ok=True)
                suggested_csv = os.path.join(output_dir, 'suggested_tickers.csv')
                suggested_df.to_csv(suggested_csv, index=False)
                st.success(f"Suggested Tickers saved to '{suggested_csv}'")
            else:
                st.info(f"No tickers meet the improving net profit and revenue criteria ({st.session_state.min_qoq_quarters} quarters QoQ and {st.session_state.min_yoy_years} years YoY growth from harsh conditions).")
            # Show all tickers for debugging
            st.subheader("All Tickers Analysis")
            st.dataframe(st.session_state.suggested_df[['Ticker', 'QoQ Net Profit Growth (%)', 'QoQ Compared Quarters', 'YoY Net Profit Growth (%)', 'YoY Compared Years', 'QoQ Revenue Growth (%)', 'YoY Revenue Growth (%)', 'Recovered from Harsh Condition', 'Error']], use_container_width=True)
            if st.button("Copy All Tickers Analysis to Clipboard"):
                copy_to_clipboard(st.session_state.suggested_df[['Ticker', 'QoQ Net Profit Growth (%)', 'QoQ Compared Quarters', 'YoY Net Profit Growth (%)', 'YoY Compared Years', 'QoQ Revenue Growth (%)', 'YoY Revenue Growth (%)', 'Recovered from Harsh Condition', 'Error']], "All Tickers Analysis")
            all_tickers_csv = os.path.join(output_dir, 'all_tickers_analysis.csv')
            st.session_state.suggested_df.to_csv(all_tickers_csv, index=False)
            st.success(f"All Tickers Analysis saved to '{all_tickers_csv}'")

    # Individual ticker data (unchanged)
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

                with st.spinner(f"Processing quarterly data for {selected_ticker}..."):
                    is_finance = st.session_state.finance_override or is_finance_company(quarterly_data)
                    processed_quarterly = process_quarterly_data(quarterly_data, is_finance)
                    qoq_results = check_highest_historical(quarterly_data, True, is_finance)
                    same_quarter_results = check_same_quarter_comparison(quarterly_data, st.session_state.enable_same_quarter)
                    st.subheader(f"Processed Quarterly Data - {selected_ticker} ({'Finance' if is_finance else 'Non-Finance'})")
                    if processed_quarterly is not None:
                        st.dataframe(processed_quarterly, use_container_width=True)
                    display_results(f"QoQ Highest Historical, Ascending, and Same Quarter - {selected_ticker}", {**qoq_results, **same_quarter_results})
                    if processed_quarterly is not None and st.button(f"Copy Processed Quarterly Data - {selected_ticker}"):
                        copy_to_clipboard(processed_quarterly, f"Processed Quarterly Data - {selected_ticker}")

            if yearly_data is not None and not yearly_data.empty:
                st.subheader(f"Raw Profit & Loss - {selected_ticker}")
                st.dataframe(yearly_data, use_container_width=True)
                if st.button(f"Copy Raw Profit & Loss - {selected_ticker}"):
                    copy_to_clipboard(yearly_data, f"Raw Profit & Loss - {selected_ticker}")

                with st.spinner(f"Processing yearly data for {selected_ticker}..."):
                    is_finance = st.session_state.finance_override or is_finance_company(quarterly_data)
                    processed_yearly = process_yearly_data(yearly_data, is_finance)
                    yoy_results = check_highest_historical(yearly_data, False, is_finance)
                    st.subheader(f"Processed Profit & Loss - {selected_ticker} ({'Finance' if is_finance else 'Non-Finance'})")
                    if processed_yearly is not None:
                        st.dataframe(processed_yearly, use_container_width=True)
                    display_results(f"YOY Highest Historical - {selected_ticker}", yoy_results)
                    if processed_yearly is not None and st.button(f"Copy Processed Profit & Loss - {selected_ticker}"):
                        copy_to_clipboard(processed_yearly, f"Processed Profit & Loss - {selected_ticker}")

if __name__ == "__main__":
    main()