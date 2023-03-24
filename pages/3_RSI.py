import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
import plotly.express as px
import streamlit as st

# Define function to calculate RSI for a given DataFrame
def get_rsi(data):
    delta = data['Adj Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Get list of tickers from user input
tickers_string = st.text_input('Tickers', 'TSLA,ETH-USD,BTC-USD,AVAX-USD,OCEAN-USD,DOT-USD,MATIC-USD').upper()
tickers = tickers_string.split(',')

# Get start date from user input
start_date = st.date_input("Start Date", dt.date(2021, 1, 1))

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

# Display RSI dataframe at the top of the page
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
