# TODO: add future projections via monte carlo or something to get an estimate for future performance 

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

# from dotenv import load_dotenv # comment out for deployment 
# load_dotenv() # comment out for deployment 

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


# -------------- PAGE CONFIG --------------
page_title = "Financial Portfolio Optimizer"
page_icon = ":zap:"  # emojis: https://www.webfx.com/tools/emoji-cheat-sheet/
layout = "centered"

st.set_page_config(page_title = page_title, layout = layout, page_icon = page_icon)
st.title(page_title + " " + page_icon)

# Streamlit sidebar table of contents
st.sidebar.markdown('''
# Sections
- [Optimized Max Sharpe Portfolio Weights](#optimized-max-sharpe-portfolio-weights)
- [Performance Expectations](#performance-expectations)
- [Correlation Matrix](#correlation-matrix)
- [Individual Stock Prices and Cumulative Returns](#individual-stock-prices-and-cumulative-returns)
- [AI Generated Report](#summary)
''', unsafe_allow_html=True)


st.markdown('### 1. How Much?')
# 1. AMOUNT
# Enter investment amount and display it. It must be an integer not a string
if 'amount' not in st.session_state:
    st.session_state.amount = 100000

amount = st.number_input('Investment Amount $', min_value=0, max_value=1000000, value=st.session_state.amount, step=100)
# Save amount to SessionState
st.session_state.amount = amount
# st.write('You have entered: ', amount)
st.markdown("""---""")


# 2. TICKERS
# Display tickers in a table
st.markdown('''### 2. What? Enter assets you would like to test as a portfolio''')

tickers_string = st.text_input('Tickers', 'TSLA,ETH-USD,BTC-USD,AVAX-USD,DOT-USD,COIN,XRP-USD,ARKK,AMZN,USD,COIN,NVDA,SQ,SHOP,PLTR').upper()

tickers = tickers_string.split(',')

with st.expander("Tickers"):
     ticker_df = pd.DataFrame(tickers, columns=['Tickers'])   
     st.table(ticker_df)


#################################################3/24/23##########################################################################
# Save tickers to SessionState
st.session_state.tickers = tickers
#################################################3/24/23##########################################################################

st.markdown("""---""")


st.markdown('''### 3. How Long?
Enter start and end dates for backtesting your portfolio. 
''')

st.caption('''*Rule of thumb is to use 5 years of data for backtesting __OR__ the number of years you plan on holding the portfolio minus today's date.*''') 
col1, col2 = st.columns(2)  # split the screen into two columns. columns(2) says split the screen into two columns
							# if said columns(1,2) then the first column would be 1/3 of the screen and the second column would be 2/3 of the screen
with col1:
	if 'start_date' not in st.session_state:
		st.session_state.start_date = datetime(2020, 1, 1)

	start_date = st.date_input("Start Date", st.session_state.start_date)


with col2:
    if 'end_date' not in st.session_state:
        st.session_state.end_date = datetime.today()

    end_date = st.date_input("End Date", st.session_state.end_date)

st.markdown("""---""")

risk_free_rate = 0.02

# Hide Streamlit Menu and Footer
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)



# # Efficient frontier
# def plot_efficient_frontier_and_max_sharpe(mu, S, risk_free_rate, n_samples=1000):
#     ef = EfficientFrontier(mu, S)
#     fig, ax = plt.subplots(figsize=(6, 4))
#     ef_max_sharpe = pickle.loads(pickle.dumps(ef))
#     plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False)

#     # Find the max sharpe portfolio
#     ef_max_sharpe.max_sharpe(risk_free_rate)
#     ret_tangent, std_tangent, _ = ef_max_sharpe.portfolio_performance()
#     ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Max Sharpe")

#     # Generate random portfolios
#     w = np.random.dirichlet(np.ones(ef.n_assets), n_samples)
#     rets = w.dot(ef.expected_returns)
#     stds = np.sqrt(np.diag(w @ ef.cov_matrix @ w.T))
#     sharpes = rets / stds
#     ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")

#     ax.legend()
#     return fig

def plot_efficient_frontier_and_max_sharpe(mu, S, risk_free_rate, n_samples=1000):
    # Make sure the covariance matrix is symmetric
    S_symmetric = (S + S.T) / 2

    ef = EfficientFrontier(mu, S_symmetric)
    fig, ax = plt.subplots(figsize=(6, 4))
    ef_max_sharpe = pickle.loads(pickle.dumps(ef))
    plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False)

    # Find the max sharpe portfolio
    ef_max_sharpe.max_sharpe(risk_free_rate)
    ret_tangent, std_tangent, _ = ef_max_sharpe.portfolio_performance()
    ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Max Sharpe")

    # Generate random portfolios
    w = np.random.dirichlet(np.ones(ef.n_assets), n_samples)
    rets = w.dot(ef.expected_returns)
    stds = np.sqrt(np.diag(w @ ef.cov_matrix @ w.T))
    sharpes = rets / stds
    ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")

    ax.legend()
    return fig



weights_df = {}
weights = {}

#THIS IS WHERE OLD TRY STARTED 

# Add risk_free_rate as an argument - this fixed the error that did not let me remove the try and except block 
risk_free_rate = 0.02  # Assuming a risk-free rate of 2%, you can adjust this value as needed

# Download data
stocks_df = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

# Plot Individual Stock Prices
fig_price = px.line(stocks_df, title='Price of Individual Stocks')

# Plot cumulative returns
def plot_cum_returns(data, title):    
    daily_cum_returns = (1 + data.fillna(0).pct_change()).cumprod()
    fig = px.line(daily_cum_returns, title=title)
    return fig

# Plot Individual Cumulative Returns
fig_cum_returns = plot_cum_returns(stocks_df, 'Cumulative Returns of Individual Stocks Starting with $100')

# Calculate and Plot Correlation Matrix between Stocks
corr_df = stocks_df.corr().round(2) # round to 2 decimal places
fig_corr = px.imshow(corr_df, text_auto=True, title = 'Correlation between Stocks', width=600, height=600)
avg_corr = corr_df.mean().mean()

# Calculate expected returns and sample covariance matrix for portfolio optimization later
mu = expected_returns.mean_historical_return(stocks_df)
S = risk_models.sample_cov(stocks_df)

# Enforce the covariance matrix to be symmetric
S = (S + S.T) / 2

# Plot efficient frontier curve
fig = plot_efficient_frontier_and_max_sharpe(mu, S, risk_free_rate)  # Add risk_free_rate argument here
fig_efficient_frontier = BytesIO()
fig.savefig(fig_efficient_frontier, format="png")

# Get optimized weights
ef = EfficientFrontier(mu, S)
ef.add_objective(objective_functions.L2_reg, gamma=5)
# ef.add_constraint(lambda w: all(wi >= 0.05 for wi in w)) # delete
# ef.add_constraint(lambda w: sum(w) == 1) # delete
ef.max_sharpe(risk_free_rate)  # Add risk_free_rate argument here
weights = ef.clean_weights()

expected_annual_return, annual_volatility, sharpe_ratio = ef.portfolio_performance()
weights_df = pd.DataFrame.from_dict(weights, orient = 'index')
weights_df.columns = ['weights']
    
stocks_df['Optimized Portfolio'] = 0
for ticker in weights.keys():
    weight = weights[ticker]
    stocks_df['Optimized Portfolio'] = stocks_df[ticker]*weight



stocks_df['Optimized Portfolio Amounts'] = 0 # create a new column in the stocks_df dataframe called "Optimized Portfolio Amounts" and set it equal to 0
stocks_df2 = stocks_df # create a new dataframe called stocks_df2 and set it equal to stocks_df
stocks_df2['Time'] = stocks_df2.index

for ticker, weight in weights.items(): # for each ticker and weight in the weights dictionary, .items() returns a list of tuples. 
	stocks_df2['Optimized Portfolio Amounts'] += stocks_df2[ticker]*(weight/100)*amount
        
last_index = len(stocks_df2) - 1
for i in range(last_index, -1, -1):
      if not pd.isna(stocks_df2["Optimized Portfolio Amounts"].iloc[i]):
        previous_date_value = stocks_df2["Optimized Portfolio Amounts"].iloc[i]
        break
      
weights_df = weights_df.sort_values(by=['weights'], ascending=False)
# display the weights_df dataframe multiplied by the amount of money invested
amounts = weights_df*amount
amounts_sorted=amounts.sort_values(by=['weights'], ascending=False)
# rename the weights column to amounts
amounts_sorted.columns = ['$ amounts']

st.header('Results')

fig = px.pie(amounts_sorted, values='$ amounts', names=amounts_sorted.index)
st.plotly_chart(fig)
st.markdown("""---""")

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
    # sort the dataframe by RSI
    rsi_df = rsi_df.sort_values(by=['RSI'], ascending=True)

# Save tickers to SessionState
if 'tickers' not in st.session_state:
    st.session_state.tickers = rsi_df

st.caption(':point_down: Check any of the boxes below to see more details about the portfolio.')


with st.expander("Weights & RSI"):

    # Display pie chart 
    
    col1, col2, col3 = st.columns(3)

    with col1:
        # display the weights_df dataframe
        st.markdown('''#### WEIGHTS (must add up to 1) ''')  
        st.dataframe(weights_df)

    with col2:
        st.markdown(f'''#### BUY THIS AMOUNT (must add up to $ {amount}) ''')
        st.dataframe(amounts_sorted)

    with col3:
        st.markdown(f'''<h3 style="text-align:center;">RSI</h3> 
        <p style="text-align:center;">Buy <= 30, Sell >= 70</p>''', unsafe_allow_html=True)
        st.dataframe(rsi_df.style.hide_index())

        
with st.expander("Performance Expectations"):
    col1, col2 = st.columns(2)
    with col1: 
        st.markdown("Optimized Portfolio Expectations")
        st.markdown("""---""")
        st.markdown(''' Expected annual return: :green[{}%] '''.format((expected_annual_return*100).round(2)))
        st.markdown(''' Annual volatility: :green[{}%] '''.format((annual_volatility*100).round(2)))
        st.markdown(''' Sharpe Ratio: :green[{}] '''.format(sharpe_ratio.round(2)))
        st.markdown(''' Portfolio Correlation: :green[{}] '''.format(avg_corr))

    with col2: 
        st.markdown("SP500 Average Historical Performance")
        st.markdown("""---""")
        st.markdown(''' Expected annual return: :green[{}%] '''.format(10))
        st.markdown(''' Annual volatility: :green[{}%] '''.format(15))
        st.markdown(''' Sharpe Ratio: :green[{}] '''.format(0.5))
        st.markdown(''' Portfolio Correlation: :green[{}] '''.format(0.7))

    
    # Optimized Portfolio: Cumulative Returns 
    fig = px.line(stocks_df2, x='Time', y='Optimized Portfolio Amounts', title= 'Optimized Portfolio: Cumulative Returns')
    fig.update_yaxes(title_text='$ Amount')
    st.plotly_chart(fig)
    
    st.caption('Click and drag a box on the graph to zoom in on a specific time period.:point_up:')

##############################################8/1/23####################################################################################

# Extract metrics into variables
ear = ((expected_annual_return*100).round(2)) 
av = ((annual_volatility*100).round(2))
sr = (sharpe_ratio.round(2))
corr = avg_corr

# Create dataframe 
data = {
  'Metric': ['Expected Return', 'Volatility', 'Sharpe Ratio', 'Correlation'],
  'Value': [ear, av, sr, corr]
}

df = pd.DataFrame(data)

# st.dataframe(df)
###################################################8/1/23###############################################################################
with st.expander("Correlation & Efficient Frontier"):

    st.header('Correlation Matrix')

    st.markdown('''[Correlation Info](https://www.investopedia.com/terms/c/correlationcoefficient.asp)''')
    
    st.plotly_chart(fig_corr)

    st.caption(''' Portfolio Correlation: {} '''.format(avg_corr))

    st.subheader("Optimized MaxSharpe Portfolio Performance")

    st.markdown('''[Efficient Frontier Info](https://www.investopedia.com/terms/e/efficientfrontier.asp)''')

    st.image(fig_efficient_frontier)
        


with st.expander("Individual Stock Prices and Cumulative Returns"):

    st.header('Individual Stock Prices and Cumulative Returns')
    
    st.plotly_chart(fig_price)
    
    st.plotly_chart(fig_cum_returns)
    
    st.markdown("""---""")

######################################################8/2/23 EMPYRIAL ############################################################################
# import streamlit as st
# from empyrial import empyrial, Engine
# import matplotlib.pyplot as plt

# # Disable the PyplotGlobalUseWarning
# st.set_option('deprecation.showPyplotGlobalUse', False)

# # Build empyrial plot 
# portfolio = Engine(
#     start_date="2018-01-01",
#     benchmark=["SPY"],
#     portfolio=['TSLA', 'ETH-USD', 'BTC-USD', 'AVAX-USD', 'DOT-USD', 'COIN', 'XRP-USD', 'ARKK', 'AMZN', 'USD', 'NVDA', 'SQ', 'SHOP', 'PLTR'],
#     optimizer="EF"
# )

# # Get the empyrial plot and performance metrics
# eplot, performance_metrics = empyrial(portfolio)

# # Check if eplot is not None
# if eplot is not None:
#     # Display the plots using matplotlib in Streamlit
#     for fig in eplot:
#         # Pass the figure explicitly to st.pyplot()
#         st.pyplot(fig)

#     # Display the performance metrics table
#     st.write("Backtest")
#     st.write("Annual return:", performance_metrics["Annual return"])
#     st.write("Cumulative return:", performance_metrics["Cumulative return"])
#     st.write("Annual volatility:", performance_metrics["Annual volatility"])
#     # Add other performance metrics as needed

# else:
#     st.write("No data available for the plot.")



######################################################8/2/23 EMPYRIAL ############################################################################

######################################################8/1/23############################################################################
st.markdown("## GPT-4 Analysis (takes about 30 seconds to load)")
from dotenv import load_dotenv
load_dotenv()

import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

# Update imports
from openai.error import InvalidRequestError

# New function to get portfolio analysis  
def get_portfolio_analysis(weights_df):

  # Craft prompt with portfolio weights
  prompt = f'''write a report on the dataframe. 
    Please provide your information using markdown labeled sub-headings and bullet points for each block of text.
    Discuss the Sharpe ratio and correlation of the optimized portfolio.
    Compare the optimized portfolio to the S&P 500 and discuss the differences.
    Explain the expected returns, volatility, and best time horizon for the portfolio.
    Offer suggestions to better optimize and diversify the portfolio and provide logic for your suggestions. : {rsi_df.to_string()}{df.to_string()}{weights_df.to_string()}
    Offer an example portfolio with allocations that incorporates your suggestions to further optimize and 
    output this suggested portfolio in a table with one column for the new suggested allocations and another for the original optimized portfolio'''
  
  try:
    # Use Chat Completions endpoint
    response = openai.ChatCompletion.create(
      model="gpt-4",  
      messages=[
        {"role": "system", "content": "You are an investment advisor giving portfolio analysis"},
        {"role": "user", "content": prompt}
      ]
    )
  except InvalidRequestError as e:
    # Handle incorrect API endpoint error
    print(e)
    return None
  
  return response.choices[0].message.content
portfolio_analysis = get_portfolio_analysis(weights_df)


with st.expander("Portfolio Analysis"):
    st.markdown(portfolio_analysis)

######################################################8/1/23############################################################################
######################################################8/8/23CHATBOT############################################################################
with st.expander("Chatbot"):

    import openai
    import toml
    import os
    import streamlit as st
    
    from dotenv import load_dotenv
    load_dotenv()

    if "messages" not in st.session_state:
        st.session_state["messages"] = []


    def show_messages(text):
        messages_str = [
            f"{_['role']}: {_['content']}" for _ in st.session_state["messages"][1:]  
        ]
        text.text_area("Messages", value=str("\n".join(messages_str)), height=400)

    openai.api_key = os.environ.get("OPENAI_API_KEY")  
    
    BASE_PROMPT = [{...}]

    prompt = st.text_input("Prompt", value="Enter your message here...")
    
    if st.button("Send"):
        with st.spinner("Generating response..."):
            st.session_state["messages"] += [{"role": "user", "content": prompt}] 
            response = openai.ChatCompletion.create(
                model=model, messages=st.session_state["messages"]
            )
            message_response = response["choices"][0]["message"]["content"]
            st.session_state["messages"] += [{"role": "system", "content": message_response}]
            show_messages(text)

    if st.button("Clear"):
        st.session_state["messages"] = BASE_PROMPT
        show_messages(text)
        
    text = st.empty()
    show_messages(text)
######################################################8/8/23CHATBOT############################################################################


######################################################8/2/23############################################################################
# # Imports
# from xgboost import XGBRegressor  

# # Model fitting  
# # Get data
# start_date = '2020-01-01'
# end_date = '2023-01-01' 
# tickers = ['AAPL', 'TSLA']

# prices = yf.download(tickers, start=start_date, end=end_date)['Close']   

# filled_prices = prices.fillna(method='ffill').fillna(method='bfill')

# model = XGBRegressor()
# model.fit(filled_prices.shift(1), filled_prices)

# # Monte Carlo simulation
# sims = 1000
# horizons = 252

# # Pre-allocate 3D numpy array instead of DataFrame 
# sim_paths = np.zeros((sims, len(tickers), horizons+1))

# for i in range(sims):
#   path = filled_prices.iloc[-1].values

#   for j in range(horizons):
#     noise = norm.rvs(0, filled_prices.std(), size=len(tickers))
#     predictions = model.predict(path.reshape(1,-1))[0]  
#     path = predictions + noise

#     sim_paths[i, :, j+1] = path

# # Take mean across sims  
# projections = sim_paths.mean(axis=0)

# # Plot 
# fig = px.line(projections)
# fig.add_scatter(x=filled_prices.index, y=filled_prices, mode='lines')

# # Display in Streamlit
# st.header('Monte Carlo Projections')
# st.plotly_chart(fig)
# ######################################################8/2/23############################################################################
# footer 
st.markdown("""---""")
st.markdown('Made with :heart: by [Jordan Clayton](https://dca-rsi.streamlit.app/Contact_Me)')
st.markdown("""---""")
