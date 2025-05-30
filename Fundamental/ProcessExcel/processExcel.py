import streamlit as st
import pandas as pd
import re

# Setting page configuration
st.set_page_config(page_title="Options Data Processor", layout="wide")

# Adding a title
st.title("Options Data Processor")

# Defining expected output columns with exact header names
OUTPUT_COLUMNS = ['TICKER', 'EXPIRY', 'CALL TYPE', 'STRIEK PRICE', 'CLOSE_PRIC']

# Uploading the CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# Adding a radio button for selecting CALL TYPE (CE or PE)
st.subheader("Select Option Type")
call_type_option = st.radio("Choose which CALL TYPE to include in the output:", ("CE", "PE"))

# Adding an input field for minimum CLOSE_PRIC
st.subheader("Minimum Close Price")
min_close_price = st.number_input("Enter the minimum CLOSE_PRIC to include:", min_value=0.0, value=0.0, step=0.1)

if uploaded_file is not None:
    # Reading the CSV file
    try:
        df = pd.read_csv(uploaded_file)
        
        # Validating required input columns
        required_columns = ['CONTRACT_D', 'CLOSE_PRIC']
        if not all(col in df.columns for col in required_columns):
            st.error("CSV file must contain 'CONTRACT_D' and 'CLOSE_PRIC' columns.")
        else:
            # Defining a function to parse CONTRACT_D
            def parse_contract_d(contract):
                try:
                    # Using regex to extract components
                    # Expected format: OPTSTK<STOCK><DATE><CE/PE><STRIKE>
                    match = re.match(r"OPTSTK([A-Z]+)([0-9]{2}-[A-Z]{3}-[0-9]{4})(CE|PE)(\d+)", contract)
                    if match:
                        ticker = match.group(1)
                        expiry = match.group(2)
                        call_type = match.group(3)
                        striek_price = float(match.group(4))
                        return ticker, expiry, call_type, striek_price
                    else:
                        return None, None, None, None
                except Exception:
                    return None, None, None, None

            # Applying parsing to CONTRACT_D column
            parsed_data = df['CONTRACT_D'].apply(parse_contract_d)
            df[['TICKER', 'EXPIRY', 'CALL TYPE', 'STRIEK PRICE']] = pd.DataFrame(parsed_data.tolist(), index=df.index)

            # Filtering out rows where parsing failed
            df = df.dropna(subset=['TICKER', 'EXPIRY', 'CALL TYPE', 'STRIEK PRICE'])

            # Removing rows where CLOSE_PRIC is blank (NaN or empty string)
            df = df.dropna(subset=['CLOSE_PRIC'])
            df = df[df['CLOSE_PRIC'].astype(str).str.strip() != '']

            # Filtering by selected CALL TYPE (CE or PE)
            df = df[df['CALL TYPE'] == call_type_option]

            # Filtering by minimum CLOSE_PRIC
            df = df[df['CLOSE_PRIC'] >= min_close_price]

            # Adding a selectbox for EXPIRY
            st.subheader("Select Expiry Date")
            expiry_options = sorted(df['EXPIRY'].unique())
            selected_expiry = st.selectbox("Choose an EXPIRY date to include in the output:", expiry_options)

            # Filtering by selected EXPIRY
            df = df[df['EXPIRY'] == selected_expiry]

            # Selecting only the required output columns
            final_df = df[OUTPUT_COLUMNS]

            # Verifying that the output contains exactly the required columns
            if list(final_df.columns) == OUTPUT_COLUMNS:
                st.subheader("Processed Data")
                st.write(f"Showing data for CALL TYPE: {call_type_option}, EXPIRY: {selected_expiry}, CLOSE_PRIC >= {min_close_price}")
                st.write(f"The output contains the following columns: TICKER, EXPIRY, CALL TYPE, STRIEK PRICE, CLOSE_PRIC")
                st.write(f"Number of rows after filtering: {len(final_df)}")
                st.dataframe(final_df)

                # Providing a download button for the processed CSV
                csv = final_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label=f"Download Processed CSV ({call_type_option}, {selected_expiry}, CLOSE_PRIC >= {min_close_price})",
                    data=csv,
                    file_name=f"processed_options_data_{call_type_option}_{selected_expiry.replace('-', '')}.csv",
                    mime="text/csv"
                )
            else:
                st.error("Error: The processed data does not match the required columns: TICKER, EXPIRY, CALL TYPE, STRIEK PRICE, CLOSE_PRIC")
    except Exception as e:
        st.error(f"Error processing the file: {str(e)}")
else:
    st.info("Please upload a CSV file to process.")