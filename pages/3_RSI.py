import yfinance as yf
import pandas as pd
import numpy as np
import talib
import datetime as dt
import plotly.express as px
import streamlit as st

####################################
import streamlit as st
import requests
import os
import sys
import subprocess

# check if the library folder already exists, to avoid building everytime you load the pahe
if not os.path.isdir("/tmp/ta-lib"):

    # Download ta-lib to disk
    with open("/tmp/ta-lib-0.4.0-src.tar.gz", "wb") as file:
        response = requests.get(
            "http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz"
        )
        file.write(response.content)
    # get our current dir, to configure it back again. Just house keeping
    default_cwd = os.getcwd()
    os.chdir("/tmp")
    # untar
    os.system("tar -zxvf ta-lib-0.4.0-src.tar.gz")
    os.chdir("/tmp/ta-lib")
    # build
    os.system("./configure --prefix=/home/appuser/venv/")
    os.system("make")
    # install
    os.system("mkdir -p /home/appuser/venv/")
    os.system("make install")
    os.system("ls -la /home/appuser/venv/")
    # back to the cwd
    os.chdir(default_cwd)
    sys.stdout.flush()

# add the library to our current environment
from ctypes import *

lib = CDLL("/home/appuser/venv/lib/libta_lib.so.0.0.0")
# import library
try:
    import talib
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--global-option=build_ext", "--global-option=-L/home/appuser/venv/lib/", "--global-option=-I/home/appuser/venv/include/", "ta-lib==0.4.24"])
finally:
    import talib
######################################

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

    # Define oversold and overbought RSI ranges
    oversold = 30
    overbought = 70

    # Plot RSI over time using Plotly Express
    fig = px.line(stock_data, x=stock_data.index, y='RSI', title=ticker + ' RSI')
    fig.update_xaxes(title_text='Date')
    fig.update_yaxes(title_text='RSI')

    # Add lower boundary line for oversold RSI range
    fig.add_hline(y=oversold, line_dash="dash", annotation_text="Oversold", 
                  annotation_position="bottom right", line_color="red")

    # Add upper boundary line for overbought RSI range
    fig.add_hline(y=overbought, line_dash="dash", annotation_text="Overbought", 
                  annotation_position="top right", line_color="red")

    st.plotly_chart(fig)

