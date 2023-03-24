import yfinance as yf
import pandas as pd
import numpy as np
import talib
import datetime as dt
import plotly.express as px
import streamlit as st

# Define function to get RSI for a given ticker
def get_rsi(ticker, start_date):
    # Get historical price data
    stock_data = yf.download(ticker, start=start_date, end=dt.datetime.now().strftime('%Y-%m-%d'))
    # Calculate RSI
    stock_data['RSI'] = talib.RSI(stock_data['Adj Close'], timeperiod=14)

    # Return last RSI value
    return stock_data['RSI'][-1]

# Get list of tickers from user input
tickers_string = st.text_input('Tickers', 'TSLA,ETH-USD,BTC-USD,AVAX-USD,OCEAN-USD,DOT-USD,MATIC-USD').upper()
tickers = tickers_string.split(',')

# Get start date from user input
start_date = st.date_input("Start Date", dt.date(2021, 1, 1))

# Create empty dataframe to store RSI data
rsi_df = pd.DataFrame(columns=['Ticker', 'RSI'])

# Loop through tickers and calculate RSI
for ticker in tickers:
    # Calculate RSI
    rsi = get_rsi(ticker, start_date)

    # Add RSI to dataframe
    rsi_df = rsi_df.append({'Ticker': ticker, 'RSI': rsi}, ignore_index=True)

# Display RSI dataframe at the top of the page
st.write(rsi_df)


# Loop through tickers and calculate RSI
for ticker in tickers:


    # Get historical price data
    stock_data = yf.download(ticker, start=start_date, end=dt.datetime.now().strftime('%Y-%m-%d'))

    # Calculate RSI
    stock_data['RSI'] = talib.RSI(stock_data['Adj Close'], timeperiod=14)

    # Plot RSI over time using Plotly Express
    fig = px.line(stock_data, x=stock_data.index, y='RSI', title=ticker + ' RSI')
    fig.update_xaxes(title_text='Date')
    fig.update_yaxes(title_text='RSI')
    st.plotly_chart(fig)


