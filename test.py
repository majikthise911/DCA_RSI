# Imports

import streamlit as st
import pypfopt	
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices# pip install PyPortfolioOpt
from streamlit_option_menu import option_menu # pip install streamlit-option-menu
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import plotting
from pypfopt import objective_functions
import copy	# for deepcopy
import numpy as np
import pandas as pd
import datetime as dt
import yfinance as yf # pip install yfinance
import plotly.express as px	# pip install plotly
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import seaborn as sns	# pip install seaborn
import datetime
from datetime import datetime, timedelta
from io import BytesIO # for downloading files
import logging	# for logging
import pickle # pip install pickle5
from sklearn.linear_model import LinearRegression
from scipy.stats import norm
from scipy.interpolate import interp1d
from xgboost import XGBRegressor  
import yfinance as yf

# Model fitting
# Get data  
start_date = '2020-01-01'
end_date = '2023-01-01'
tickers = ['AAPL', 'TSLA']  

prices = yf.download(tickers, start=start_date, end=end_date)['Close']   

filled_prices = prices.fillna(method='ffill').fillna(method='bfill')

model = XGBRegressor()
model.fit(filled_prices.shift(1), filled_prices)

# Monte Carlo simulation
sims = 1000
horizons = 252

sim_paths = np.zeros((sims, len(tickers), horizons+1))

for i in range(sims):
  path = filled_prices.iloc[-1].values
  for j in range(horizons):
    noise = norm.rvs(0, filled_prices.std(), size=len(tickers))
    predictions = model.predict(path.reshape(1,-1))[0] 
    path = predictions + noise
    sim_paths[i, :, j+1] = path
    
projections = sim_paths.mean(axis=0)

# Plot 
import plotly.express as px
fig = px.line(projections)
fig.add_scatter(x=filled_prices.index, y=filled_prices, mode='lines')



# # Display in Streamlit
# import streamlit as st 
# st.header('Monte Carlo Projections')
# st.plotly_chart(fig)