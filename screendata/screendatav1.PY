import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
from datetime import datetime
import time
import uuid
import numpy as np
import re

# Set page to wide mode
st.set_page_config(layout="wide")

# Function to load data from CSV files if available
def load_data_from_csv(ticker, folder="data"):
    all_data = {}
    data_found = False
    for filename in os.listdir(folder):
        if filename.startswith(f"{ticker}_") and filename.endswith(".csv"):
            table_name = filename.replace(f"{ticker}_", "").replace(".csv", "")
            try:
                df = pd.read_csv(os.path.join(folder, filename))
                all_data[table_name] = df
                data_found = True
            except Exception as e:
                st.warning(f"Error loading {filename}: {e}")
    return all_data, None if data_found else "No data found in folder."

# Function to scrape all data from Screener.in for a ticker
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
                section_name = section_title.text.strip().replace(' ', '_').lower() if section_title else section_id or f"section_{uuid.uuid4().hex[:8]}"
                
                tables = section.find_all('table', class_='data-table')
                for idx, table in enumerate(tables):
                    df = parse_table(table)
                    if df is not None and not df.empty and df.shape[1] >= 2 and df.shape[0] >= 1:
                        table_key = f"{section_name}_table_{idx + 1}" if len(tables) > 1 else section_name
                        # Map specific section names to expected CSV filenames
                        if section_name == "quarterly_results":
                            table_key = "quarterly_results"
                        elif section_name == "profit_and_loss":
                            table_key = "profit_loss"
                        elif section_name == "balance_sheet":
                            table_key = "balance_sheet"
                        elif section_name == "cash_flows":
                            table_key = "cash_flows"
                        all_data[table_key] = df
                        data_found = True
            
            # Scrape existing company ratios/summary data
            ratios_div = soup.find('ul', class_='company-ratios')
            if ratios_div:
                ratios_data = parse_ratios(ratios_div)
                if ratios_data is not None and not ratios_data.empty:
                    all_data['company_summary'] = ratios_data
                    data_found = True
            
            # Scrape About and Key Points sections (fallback)
            about_data, key_points_data = parse_about_and_key_points(soup)
            if about_data:
                all_data['about'] = pd.DataFrame({'About': [about_data]})
                data_found = True
            if key_points_data:
                key_points_df = pd.DataFrame(list(key_points_data.items()), columns=['Key Point', 'Description'])
                all_data['key_points'] = key_points_df
                data_found = True
            
            # Scrape Company Profile and Financial Ratios from company-info section
            company_info_data = parse_company_info(soup)
            if company_info_data:
                all_data['company_info'] = pd.DataFrame([company_info_data])
                data_found = True
            else:
                st.warning(f"No company-info data found for {ticker} in {url}")

            if data_found:
                break
            elif url == urls[0]:
                continue
                
        except requests.RequestException as e:
            error = f"Error fetching {url} for ticker {ticker}: {e}"
            if url == urls[1]:
                error = f"No valid data found for ticker {ticker}: {e}"
        
        if url == urls[1] and not all_data:
            error = f"No financial data tables, ratios, or about/key points found for ticker {ticker}"
    
    return all_data, error

# Function to parse table data into a DataFrame
def parse_table(table):
    headers = [th.text.strip() for th in table.find('thead').find_all('th')]
    if not headers or len(headers) < 2:
        return None
    rows = []
    for tr in table.find('tbody').find_all('tr'):
        cells = [td.text.strip() for td in tr.find_all('td')]
        if cells and len(cells) == len(headers):
            rows.append(cells)
    return pd.DataFrame(rows, columns=headers) if rows else None

# Function to parse existing company ratios/summary data into a DataFrame
def parse_ratios(ratios_div):
    rows = []
    headers = ['Metric', 'Value']
    for li in ratios_div.find_all('li'):
        name = li.find('span', class_='name')
        value = li.find('span', class_='value') or li.find('span', class_='number')
        if name and value:
            metric = name.text.strip()
            val = value.text.strip()
            rows.append([metric, val])
    return pd.DataFrame(rows, columns=headers) if rows else None

# Function to parse About and Key Points sections (fallback)
def parse_about_and_key_points(soup):
    about_text = None
    key_points = {}
    
    about_div = soup.find('div', class_='about')
    if about_div:
        about_p = about_div.find('p')
        if about_p:
            about_text = about_p.text.strip()
    
    key_points_div = soup.find('div', class_='commentary')
    if key_points_div:
        for p in key_points_div.find_all('p'):
            text = p.text.strip()
            if 'OIL-TO-CHEMICALS' in text:
                key_points['Oil-to-Chemicals Segment'] = text
    
    return about_text, key_points

# Function to parse the company-info section with improved error handling
def parse_company_info(soup):
    company_data = {}
    company_info_div = soup.find('div', class_='company-info')
    if not company_info_div:
        return None

    company_profile_div = company_info_div.find('div', class_='company-profile')
    if company_profile_div:
        about_div = company_profile_div.find('div', class_='sub')
        if about_div:
            about_p = about_div.find('p')
            if about_p:
                about_text = about_p.text.strip()
                # Robust parsing for About section
                if "founded by" in about_text:
                    try:
                        company_data['Founder'] = about_text.split('founded by ')[1].split(' and')[0].strip()
                    except IndexError:
                        company_data['Founder'] = "Unknown"
                else:
                    company_data['Founder'] = "Not specified"

                if "promoted and managed by" in about_text:
                    try:
                        company_data['Current Leader'] = about_text.split('promoted and managed by ')[1].split('.')[0].strip()
                    except IndexError:
                        company_data['Current Leader'] = "Unknown"
                else:
                    company_data['Current Leader'] = "Not specified"

                if "about " in about_text and "%" in about_text:
                    try:
                        percentage_text = about_text.split('about ')[1].split('%')[0].strip()
                        company_data['Family Shareholding (%)'] = float(percentage_text)
                    except (IndexError, ValueError):
                        company_data['Family Shareholding (%)'] = 0.0
                else:
                    company_data['Family Shareholding (%)'] = 0.0

        key_points_div = company_profile_div.find('div', class_='commentary')
        if key_points_div:
            key_points_text = key_points_div.find('p')
            if key_points_text:
                key_points_text = key_points_text.text.strip()
                # Robust parsing for Key Points
                if "~" in key_points_text and "%" in key_points_text:
                    try:
                        company_data['Oil-to-Chemicals Segment (% of revenues)'] = float(key_points_text.split('~')[1].split('%')[0].strip())
                    except (IndexError, ValueError):
                        company_data['Oil-to-Chemicals Segment (% of revenues)'] = 0.0
                else:
                    company_data['Oil-to-Chemicals Segment (% of revenues)'] = 0.0

                if "crude refining capacity of " in key_points_text:
                    try:
                        company_data['Crude Refining Capacity (million barrels per day)'] = float(key_points_text.split('crude refining capacity of ')[1].split(' million')[0].strip())
                    except (IndexError, ValueError):
                        company_data['Crude Refining Capacity (million barrels per day)'] = 0.0
                else:
                    company_data['Crude Refining Capacity (million barrels per day)'] = 0.0

                company_data['Largest Single Site Refinery'] = "Yes" if "largest single site refinery complex globally" in key_points_text else "No"

                if "throughput of ~" in key_points_text:
                    try:
                        company_data['FY22 Throughput (million metric tonnes)'] = float(key_points_text.split('throughput of ~')[1].split(' million')[0].strip())
                    except (IndexError, ValueError):
                        company_data['FY22 Throughput (million metric tonnes)'] = 0.0
                else:
                    company_data['FY22 Throughput (million metric tonnes)'] = 0.0

                if "out of which ~" in key_points_text and "%" in key_points_text:
                    try:
                        company_data['FY22 Throughput for Sale (%)'] = float(key_points_text.split('out of which ~')[1].split('%')[0].strip())
                    except (IndexError, ValueError):
                        company_data['FY22 Throughput for Sale (%)'] = 0.0
                else:
                    company_data['FY22 Throughput for Sale (%)'] = 0.0

    ratios_div = company_info_div.find('ul', id='top-ratios')
    if ratios_div:
        for li in ratios_div.find_all('li'):
            name = li.find('span', class_='name')
            value_span = li.find('span', class_='value')
            if name and value_span:
                metric = name.text.strip()
                value_text = value_span.text.strip()
                try:
                    # Clean the value_text and handle units like "Cr."
                    cleaned_value = value_text.replace('₹', '').replace(',', '').strip()
                    # Remove any non-numeric characters except for decimal points
                    if 'Cr.' in cleaned_value:
                        # Convert Crores to a numeric value (1 Cr = 10,000,000)
                        numeric_part = re.sub(r'[^0-9.]', '', cleaned_value)
                        if numeric_part:
                            value = float(numeric_part) * 1e7  # Multiply by 10,000,000 for Crores
                        else:
                            value = 0.0
                    elif '%' in cleaned_value:
                        numeric_part = re.sub(r'[^0-9.]', '', cleaned_value)
                        value = float(numeric_part) if numeric_part else 0.0
                    else:
                        numeric_part = re.sub(r'[^0-9.]', '', cleaned_value)
                        value = float(numeric_part) if numeric_part else 0.0
                    company_data[metric] = value
                except ValueError as e:
                    st.warning(f"Error parsing metric {metric}: {value_text} ({e})")
                    company_data[metric] = value_text  # Store the raw text if parsing fails

    return company_data if company_data else None

# Function: Perform Comprehensive Fundamental Analysis for a Specific Ticker
def perform_comprehensive_fundamental_analysis(ticker, data_dict):
    analysis_results = []
    consolidated_summary = []

    # Check if company_info data is available
    if 'company_info' not in data_dict:
        return None, "No company-info data available for analysis."

    company_info_df = data_dict['company_info']
    if company_info_df.empty:
        return None, "Company-info data is empty."

    try:
        # 1. Company Information and Stability
        section = {"title": "Company Information and Stability", "details": [], "pros": [], "cons": []}
        founder = company_info_df.get('Founder', [None])[0]
        leader = company_info_df.get('Current Leader', [None])[0]
        family_shareholding = company_info_df.get('Family Shareholding (%)', [None])[0]
        largest_refinery = company_info_df.get('Largest Single Site Refinery', [None])[0]
        
        if founder and leader and family_shareholding:
            stability_status = "Stable" if float(family_shareholding) > 30 else "Potentially Unstable"
            section["details"].append(f"Founder: {founder}")
            section["details"].append(f"Current Leader: {leader}")
            section["details"].append(f"Family Shareholding: {family_shareholding}% → {stability_status}")
            section["pros"].append("Strong leadership continuity") if stability_status == "Stable" else section["cons"].append("Potential governance risks due to lower family control")
            if largest_refinery == "Yes":
                section["pros"].append("Largest single-site refinery globally")
            analysis_results.append(section)
            consolidated_summary.append(f"- **Company Stability**: {stability_status} (Family Shareholding: {family_shareholding}%)")
        else:
            section["details"].append("Data unavailable")
            analysis_results.append(section)

        # 2. P/E Analysis
        section = {"title": "P/E Analysis", "details": [], "pros": [], "cons": []}
        pe_ratio = company_info_df.get('Stock P/E', [None])[0]
        industry_pe = company_info_df.get('Industry PE', [None])[0]
        if pe_ratio and industry_pe:
            pe_ratio = float(pe_ratio)
            industry_pe = float(industry_pe)
            valuation_status = "Overvalued" if pe_ratio > industry_pe else "Undervalued"
            section["details"].append(f"P/E Ratio: {pe_ratio} vs Industry P/E: {industry_pe} → {valuation_status}")
            section["pros"].append("Attractive valuation") if valuation_status == "Undervalued" else section["cons"].append("Potentially overpriced")
            consolidated_summary.append(f"- **Valuation**: Stock P/E ({pe_ratio}) vs Industry P/E ({industry_pe}): {valuation_status}")
        else:
            section["details"].append("Data unavailable")
        analysis_results.append(section)

        # 3. Intrinsic Valuation
        section = {"title": "Intrinsic Valuation", "details": [], "pros": [], "cons": []}
        current_price = company_info_df.get('Current Price', [None])[0]
        intrinsic_value = company_info_df.get('Intrinsic Value', [None])[0]
        if current_price and intrinsic_value:
            current_price = float(current_price)
            intrinsic_value = float(intrinsic_value)
            price_status = "Overpriced" if current_price > intrinsic_value else "Underpriced"
            section["details"].append(f"Current Price: ₹{current_price}, Intrinsic Value: ₹{intrinsic_value} → {price_status}")
            section["pros"].append("Potential upside") if price_status == "Underpriced" else section["cons"].append("Limited upside potential")
            consolidated_summary.append(f"- **Intrinsic Valuation**: Current Price (₹{current_price}) vs Intrinsic Value (₹{intrinsic_value}): {price_status}")
        else:
            section["details"].append("Data unavailable")
        analysis_results.append(section)

        # 4. Quarterly Results (Pros and Cons) - Use TICKER_quarterly_results.csv
        section = {"title": "Quarterly Results", "details": [], "pros": [], "cons": []}
        quarterly_df = data_dict.get('quarterly_results', pd.DataFrame())
        if not quarterly_df.empty and 'Net Profit' in quarterly_df.index:
            latest_quarter = quarterly_df.iloc[:, -1]
            prev_quarter = quarterly_df.iloc[:, -2] if quarterly_df.shape[1] > 1 else None
            if prev_quarter is not None:
                latest_profit = float(latest_quarter[quarterly_df.index.get_loc('Net Profit')].replace(',', ''))
                prev_profit = float(prev_quarter[quarterly_df.index.get_loc('Net Profit')].replace(',', ''))
                profit_growth = ((latest_profit - prev_profit) / prev_profit * 100) if prev_profit != 0 else 0
                section["details"].append(f"Latest Net Profit: ₹{latest_profit:,}, Previous: ₹{prev_profit:,}, Growth: {profit_growth:.2f}%")
                section["pros"].append("Profit growth") if profit_growth > 0 else section["cons"].append("Profit decline")
                consolidated_summary.append(f"- **Quarterly Results**: Net Profit Growth: {profit_growth:.2f}% ({'Positive' if profit_growth > 0 else 'Negative'})")
            else:
                section["details"].append("Insufficient data for comparison")
        else:
            section["details"].append("Data unavailable")
        analysis_results.append(section)

        # 5. Profit and Loss (Pros and Cons) - Use TICKER_profit_&_loss.csv
        section = {"title": "Profit and Loss", "details": [], "pros": [], "cons": []}
        profit_loss_df = data_dict.get('profit_&_loss', pd.DataFrame())
        if not profit_loss_df.empty and 'Net Profit' in profit_loss_df.index:
            latest_year = profit_loss_df.iloc[:, -1]
            prev_year = profit_loss_df.iloc[:, -2] if profit_loss_df.shape[1] > 1 else None
            if prev_year is not None:
                latest_profit = float(latest_year[profit_loss_df.index.get_loc('Net Profit')].replace(',', ''))
                prev_profit = float(prev_year[profit_loss_df.index.get_loc('Net Profit')].replace(',', ''))
                profit_growth = ((latest_profit - prev_profit) / prev_profit * 100) if prev_profit != 0 else 0
                section["details"].append(f"Latest Net Profit: ₹{latest_profit:,}, Previous: ₹{prev_profit:,}, Growth: {profit_growth:.2f}%")
                section["pros"].append("Consistent profit growth") if profit_growth > 0 else section["cons"].append("Profit decline")
                consolidated_summary.append(f"- **Profit and Loss**: Net Profit Growth: {profit_growth:.2f}% ({'Positive' if profit_growth > 0 else 'Negative'})")
            else:
                section["details"].append("Insufficient data for comparison")
        else:
            section["details"].append("Data unavailable")
        analysis_results.append(section)

        # 6. Balance Sheet Analysis (Positive and Negative Points) - Use TICKER_balance_sheet.csv
        section = {"title": "Balance Sheet Analysis", "details": [], "pros": [], "cons": []}
        balance_df = data_dict.get('balance_sheet', pd.DataFrame())
        if not balance_df.empty and 'Total Assets' in balance_df.index and 'Total Liabilities' in balance_df.index:
            latest_year = balance_df.iloc[:, -1]
            total_assets = float(latest_year[balance_df.index.get_loc('Total Assets')].replace(',', ''))
            total_liabilities = float(latest_year[balance_df.index.get_loc('Total Liabilities')].replace(',', ''))
            net_worth = total_assets - total_liabilities
            section["details"].append(f"Total Assets: ₹{total_assets:,}, Total Liabilities: ₹{total_liabilities:,}, Net Worth: ₹{net_worth:,}")
            section["pros"].append("Strong net worth") if net_worth > 0 else section["cons"].append("Negative net worth")
            consolidated_summary.append(f"- **Balance Sheet**: Net Worth: ₹{net_worth:,} ({'Positive' if net_worth > 0 else 'Negative'})")
        else:
            section["details"].append("Data unavailable")
        analysis_results.append(section)

        # 7. Cash Flow (Positive and Negative Points) - Use TICKER_cash_flows.csv
        section = {"title": "Cash Flow Analysis", "details": [], "pros": [], "cons": []}
        cash_flow_df = data_dict.get('cash_flows', pd.DataFrame())
        if not cash_flow_df.empty and 'Cash from Operating Activity' in cash_flow_df.index:
            latest_year = cash_flow_df.iloc[:, -1]
            operating_cash_flow = float(latest_year[cash_flow_df.index.get_loc('Cash from Operating Activity')].replace(',', ''))
            section["details"].append(f"Operating Cash Flow: ₹{operating_cash_flow:,}")
            section["pros"].append("Positive operating cash flow") if operating_cash_flow > 0 else section["cons"].append("Negative operating cash flow")
            consolidated_summary.append(f"- **Cash Flow**: Operating Cash Flow: ₹{operating_cash_flow:,} ({'Positive' if operating_cash_flow > 0 else 'Negative'})")
        else:
            section["details"].append("Data unavailable")
        analysis_results.append(section)

        # 8. Ratios Analysis (Positive and Negative Points) - Use TICKER_company_info.csv
        section = {"title": "Ratios Analysis", "details": [], "pros": [], "cons": []}
        debt_to_equity = company_info_df.get('Debt to equity', [None])[0]
        roe = company_info_df.get('ROE', [None])[0]
        roce = company_info_df.get('ROCE', [None])[0]
        if debt_to_equity and roe and roce:
            debt_to_equity = float(debt_to_equity)
            roe = float(roe)
            roce = float(roce)
            section["details"].append(f"Debt to Equity: {debt_to_equity}, ROE: {roe}%, ROCE: {roce}%")
            section["pros"].append("Low debt burden") if debt_to_equity < 1 else section["cons"].append("High debt burden")
            if roe > 15 and roce > 15:
                section["pros"].append("Strong profitability")
            else:
                section["cons"].append("Weak profitability")
            consolidated_summary.append(f"- **Ratios**: Debt to Equity: {debt_to_equity}, ROE: {roe}%, ROCE: {roce}%")
        else:
            section["details"].append("Data unavailable")
        analysis_results.append(section)

        # 9. Shareholding Analysis (Positive and Negative Points) - Use TICKER_company_info.csv
        section = {"title": "Shareholding Analysis", "details": [], "pros": [], "cons": []}
        promoter_holding = company_info_df.get('Promoter holding', [None])[0]
        pledged_percentage = company_info_df.get('Pledged percentage', [None])[0]
        if promoter_holding and pledged_percentage:
            promoter_holding = float(promoter_holding)
            pledged_percentage = float(pledged_percentage)
            section["details"].append(f"Promoter Holding: {promoter_holding}%, Pledged Percentage: {pledged_percentage}%")
            section["pros"].append("High promoter confidence") if promoter_holding > 50 else section["cons"].append("Low promoter confidence")
            section["pros"].append("No pledged shares") if pledged_percentage == 0 else section["cons"].append("Promoter shares pledged")
            consolidated_summary.append(f"- **Shareholding**: Promoter Holding: {promoter_holding}%, Pledged: {pledged_percentage}%")
        else:
            section["details"].append("Data unavailable")
        analysis_results.append(section)

        # 10. Documents with Sensitive Information
        section = {"title": "Documents with Sensitive Information", "details": [], "pros": [], "cons": []}
        key_points_df = data_dict.get('key_points', pd.DataFrame())
        sensitive_info = "No sensitive documents or references found."
        if not key_points_df.empty:
            for _, row in key_points_df.iterrows():
                if 'http' in row['Description']:
                    sensitive_info = "References to annual reports or filings found in Key Points section."
                    break
        section["details"].append(sensitive_info)
        analysis_results.append(section)
        consolidated_summary.append(f"- **Documents**: {sensitive_info}")

        # 11. 200-Day Moving Average
        section = {"title": "200-Day Moving Average", "details": [], "pros": [], "cons": []}
        current_price = company_info_df.get('Current Price', [None])[0]
        high_price = company_info_df.get('High Price', [None])[0]
        low_price = company_info_df.get('Low Price', [None])[0]
        if current_price and high_price and low_price:
            current_price = float(current_price)
            ma_200 = (float(high_price) + float(low_price)) / 2  # Simplified estimate
            ma_status = "Above" if current_price > ma_200 else "Below"
            section["details"].append(f"Current Price: ₹{current_price}, 200-Day MA (estimated): ₹{ma_200:.2f} → {ma_status}")
            consolidated_summary.append(f"- **200-Day MA**: Current Price (₹{current_price}) is {ma_status} 200-Day MA (₹{ma_200:.2f})")
        else:
            section["details"].append("Data unavailable")
        analysis_results.append(section)

    except Exception as e:
        return None, f"Error during fundamental analysis: {e}"

    return analysis_results, "\n".join(consolidated_summary)

# Function to save DataFrames to CSV files with timestamp
def save_to_csv(ticker, data_dict, folder="data"):
    os.makedirs(folder, exist_ok=True)
    timestamp = "08:57 AM IST, Tuesday, May 27, 2025"
    
    for table_name, df in data_dict.items():
        df['Data Captured At'] = timestamp
        filename = os.path.join(folder, f"{ticker}_{table_name}.csv")
        try:
            if os.path.exists(filename):
                existing_df = pd.read_csv(filename)
                combined_df = pd.concat([existing_df, df], ignore_index=True)
                combined_df.to_csv(filename, index=False)
            else:
                df.to_csv(filename, index=False)
            st.success(f"Saved {table_name} for {ticker} to {filename}")
        except Exception as e:
            st.error(f"Error saving {table_name} for {ticker}: {e}")

# Streamlit app
def main():
    st.title("Screener.in Data Scraper and Fundamental Analysis")
    st.write("Load or scrape financial data from Screener.in and perform comprehensive fundamental analysis on specific tickers.")

    # Input for tickers
    ticker_input = st.text_area("Enter tickers (one per line or comma-separated):", placeholder="RELIANCE,INFY,TCS")
    ticker_list = [t.strip().upper() for t in ticker_input.replace(',', '\n').split('\n') if t.strip()]
    
    # Create a data folder
    data_folder = "data"
    
    # Dictionary to store data for all tickers
    if 'scraped_data' not in st.session_state:
        st.session_state.scraped_data = {}

    # Load or Scrape Data Section
    if st.button("Load or Scrape Data"):
        if not ticker_list:
            st.error("Please enter at least one ticker.")
            return
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, ticker in enumerate(ticker_list):
            status_text.text(f"Processing data for {ticker} ({idx + 1}/{len(ticker_list)})...")
            
            # Try to load data from CSV first
            data_dict, error = load_data_from_csv(ticker, data_folder)
            
            if error:  # Data not found in folder, so scrape
                status_text.text(f"Scraping data for {ticker} ({idx + 1}/{len(ticker_list)})...")
                data_dict, error = scrape_screener_data(ticker)
                if error:
                    st.error(error)
                elif data_dict:
                    save_to_csv(ticker, data_dict, data_folder)
            else:
                st.success(f"Loaded existing data for {ticker} from {data_folder}")
            
            if data_dict:
                st.session_state.scraped_data[ticker] = data_dict
                for table_name, df in data_dict.items():
                    st.write(f"**{ticker} - {table_name.replace('_', ' ').title()}**")
                    st.dataframe(df)
            else:
                st.warning(f"No data found for {ticker}")
            
            progress_bar.progress((idx + 1) / len(ticker_list))
            time.sleep(1) if error else time.sleep(0.1)
        
        status_text.text("Data processing complete!")

    # Comprehensive Fundamental Analysis Section
    st.header("Comprehensive Fundamental Analysis of a Specific Ticker")
    selected_ticker = st.selectbox("Select a ticker for analysis:", list(st.session_state.scraped_data.keys()) if st.session_state.scraped_data else ["No tickers available"])
    
    if st.button("Perform Comprehensive Analysis"):
        if selected_ticker == "No tickers available":
            st.error("No tickers available for analysis. Please load or scrape data first.")
        else:
            analysis_results, summary = perform_comprehensive_fundamental_analysis(selected_ticker, st.session_state.scraped_data[selected_ticker])
            
            if analysis_results:
                st.subheader(f"Comprehensive Fundamental Analysis for {selected_ticker}")
                for section in analysis_results:
                    st.markdown(f"#### {section['title']}")
                    for detail in section['details']:
                        st.write(f"- {detail}")
                    if section['pros']:
                        st.write("**Pros:**")
                        for pro in section['pros']:
                            st.write(f"  - {pro}")
                    if section['cons']:
                        st.write("**Cons:**")
                        for con in section['cons']:
                            st.write(f"  - {con}")
                    st.write("")  # Add spacing between sections
                
                st.subheader("Consolidated Summary")
                st.markdown(summary)
            else:
                st.error(summary)

if __name__ == "__main__":
    main()