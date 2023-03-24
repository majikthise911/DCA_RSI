import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
import plotly.express as px
import streamlit as st


# Hide Streamlit Menu and Footer
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

rsi_window = st.sidebar.slider('RSI Window', 1, 200, 14)


# Define function to calculate RSI for a given DataFrame
def get_rsi(data):
    delta = data['Adj Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=rsi_window).mean()
    avg_loss = loss.rolling(window=rsi_window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Get list of tickers from user input
# Import tickers from SessionState
if 'tickers' not in st.session_state:
    st.warning('Please enter tickers on the previous page.')
else:
    tickers = st.session_state.tickers

# Get start date from user input
# start_date = st.date_input("Start Date", dt.date(2021, 1, 1))

if 'start_date' not in st.session_state:
    st.warning('Please enter start date on the previous page.')
else:
    start_date = st.session_state.start_date

# Create empty dataframe to store RSI data
rsi_df = pd.DataFrame(columns=['Ticker', 'RSI'])

# Loop through tickers and calculate RSI
for ticker in tickers:
    # Get historical price data
    stock_data = yf.download(ticker, start=start_date, end=dt.datetime.now().strftime('%Y-%m-%d'))
    # Calculate RSI
    rsi = get_rsi(stock_data)
    last_rsi = rsi.iloc[-1]

    # Add RSI to dataframe
    rsi_df = rsi_df.append({'Ticker': ticker, 'RSI': last_rsi}, ignore_index=True)

###########################################################################################################################
# Save tickers to SessionState
if 'tickers' not in st.session_state:
    st.session_state.tickers = rsi_df
###########################################################################################################################

# Display RSI dataframe at the top of the page
st.markdown('## RSI')
st.write(rsi_df)

# Loop through tickers and plot RSI
for ticker in tickers:
    # Get historical price data
    stock_data = yf.download(ticker, start=start_date, end=dt.datetime.now().strftime('%Y-%m-%d'))
    # Calculate RSI
    rsi = get_rsi(stock_data)

    # Define oversold and overbought RSI ranges
    oversold = 30
    overbought = 70

    # Plot RSI over time using Plotly Express
    fig = px.line(x=rsi.index, y=rsi.values, title=ticker + ' RSI')
    fig.update_xaxes(title_text='Date')
    fig.update_yaxes(title_text='RSI')

    # Add lower boundary line for oversold RSI range
    fig.add_hline(y=oversold, line_dash="dash", annotation_text="Oversold", 
                  annotation_position="bottom right", line_color="red")

    # Add upper boundary line for overbought RSI range
    fig.add_hline(y=overbought, line_dash="dash", annotation_text="Overbought", 
                  annotation_position="top right", line_color="red")

    st.plotly_chart(fig)
