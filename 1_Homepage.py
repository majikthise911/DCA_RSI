import streamlit as st
import pypfopt	
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices# pip install PyPortfolioOpt
from streamlit_option_menu import option_menu # pip install streamlit-option-menu
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import plotting
from pypfopt import objective_functions
import os	# for os.path.join
import copy	# for deepcopy
import openai # pip install openai
import numpy as np
import pandas as pd
import datetime as dt
import yfinance as yf # pip install yfinance
import plotly.express as px	# pip install plotly
import matplotlib.pyplot as plt
import seaborn as sns	# pip install seaborn
import datetime
from datetime import datetime, timedelta
from io import BytesIO # for downloading files
import logging	# for logging
import pickle # pip install pickle5

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
amount = st.number_input('Investment Amount $', min_value=0, max_value=1000000, value=1000, step=100)
# st.write('You have entered: ', amount)
st.markdown("""---""")

# 2. TICKERS
st.markdown('''### 2. What?
Enter assets you would like to test as a portfolio''')
st.caption(''' Enter tickers separated by commas WITHOUT spaces, e.g. "TSLA,ETH-USD,BTC-USD,AVAX-USD,OCEAN-USD,DOT-USD,MATIC-USD" ''')
tickers_string = st.text_input('Tickers', 'TSLA,ETH-USD,BTC-USD,AVAX-USD,OCEAN-USD,DOT-USD,MATIC-USD').upper()
tickers = tickers_string.split(',')

########################################################################################################################################################
# Save tickers to SessionState
if 'tickers' not in st.session_state:
    st.session_state.tickers = tickers
########################################################################################################################################################
st.markdown("""---""")

# 3. How Long?
st.markdown('''### 3. How Long?
Enter start and end dates for backtesting your portfolio. 
''')

st.caption('''*Rule of thumb is to use 5 years of data for backtesting __OR__ the number of years you plan on holding the portfolio minus today's date.*''') 
col1, col2 = st.columns(2)  # split the screen into two columns. columns(2) says split the screen into two columns
							# if said columns(1,2) then the first column would be 1/3 of the screen and the second column would be 2/3 of the screen
with col1:
	start_date = st.date_input("Start Date",datetime(2020, 1, 1))
	
with col2:
	end_date = st.date_input("End Date") # it defaults to current date
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

# -------------- FUNCTIONS ----------------
# Plot cumulative returns
def plot_cum_returns(data, title):    
	daily_cum_returns = 1 + data.dropna().pct_change()
	daily_cum_returns = daily_cum_returns.cumprod()*100 ### is this 100 for percentage or does it represent the initial investment amount? Answer: 100 for percentage
	fig = px.line(daily_cum_returns, title=title)
	return fig
# Efficient frontier
def plot_efficient_frontier_and_max_sharpe(mu, S): # mu is expected returns, S is covariance matrix. So we are defining a function that takes in these two parameters
	# Optimize portfolio for max Sharpe ratio and plot it out with efficient frontier curve
	ef = EfficientFrontier(mu, S) # the efficient frontier object is 
	fig, ax = plt.subplots(figsize=(6,4)) # fig, ax = plt.subplots() is the same as fig = plt.figure() and ax = fig.add_subplot(111)
	ef_max_sharpe = pickle.loads(pickle.dumps(ef)) 			
		# 1. Import the "pickle" module.
		# 2. Serialize the "ef" object using "pickle.dumps".
		# 3. Deserialize the serialized object using "pickle.loads".
		# 4. Create a new object called "ef_max_sharpe" from the deserialized object.
		# This method is useful when you want to make a duplicate of an object without 
		# modifying the original object. It is also useful when you want to store an object in a file or send it over a network.
		# original method was to use copy.deepcopy(ef) but this breaks the code on cloud deployment. Cloud does not support deepcopy of CVXPY expression 
		# however we need to use deepcopy because the original object is modified when we call ef.max_sharpe() and we need to keep the original object intact
		# so we use pickle.loads(pickle.dumps(ef)) to make a copy of the object
	plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False)
	# Find the max sharpe portfolio
	ef_max_sharpe.max_sharpe(risk_free_rate) # risk_free_rate is the risk-free rate of return which is the return you would get if you invested in a risk-free asset like a US Treasury Bill
	ret_tangent, std_tangent, _ = ef_max_sharpe.portfolio_performance()
	ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Max Sharpe") # s is size of marker, c is color of marker
	# Generate random portfolios
	n_samples = 1000
	w = np.random.dirichlet(np.ones(ef.n_assets), n_samples)
	rets = w.dot(ef.expected_returns)
	stds = np.sqrt(np.diag(w @ ef.cov_matrix @ w.T))
	sharpes = rets / stds
	ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")
	# Output
	ax.legend()
	return fig

# The code to get stock prices using yfinance is below and in a try/except block because it sometimes fails and we need to catch the error
# the try block will try to run the code in the try block. If it fails, it will run the code in the except block
# the except block will run if the code in the try block fails
try:
	# Get Stock Prices using pandas_datareader Library	
	stocks_df = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
	stocks_df = stocks_df
	# # Plot Individual Stock Prices
	fig_price = px.line(stocks_df, title='Price of Individual Stocks')
	# # Plot Individual Cumulative Returns
	fig_cum_returns = plot_cum_returns(stocks_df, 'Cumulative Returns of Individual Stocks Starting with $100')
	# # Calculatge and Plot Correlation Matrix between Stocks
	corr_df = stocks_df.corr().round(2) # round to 2 decimal places
	fig_corr = px.imshow(corr_df, text_auto=True, title = 'Correlation between Stocks', width=600, height=600)
	avg_corr = corr_df.mean().mean()

	# Calculate expected returns and sample covariance matrix for portfolio optimization later
	mu = expected_returns.mean_historical_return(stocks_df)
	S = risk_models.sample_cov(stocks_df)

	# Plot efficient frontier curve
	fig = plot_efficient_frontier_and_max_sharpe(mu, S)
	fig_efficient_frontier = BytesIO()
	fig.savefig(fig_efficient_frontier, format="png")

	# Get optimized weights
	ef = EfficientFrontier(mu, S)
	ef.add_objective(objective_functions.L2_reg, gamma=5)
	# ef.add_constraint(lambda w: all(wi >= 0.05 for wi in w)) # delete
	# ef.add_constraint(lambda w: sum(w) == 1) # delete
	ef.max_sharpe(risk_free_rate)
	weights = ef.clean_weights()


	expected_annual_return, annual_volatility, sharpe_ratio = ef.portfolio_performance()
	weights_df = pd.DataFrame.from_dict(weights, orient = 'index')
	weights_df.columns = ['weights']

	# Calculate returns of portfolio with optimized weights by multiplying the 
	# weights by the returns of each stock and saving it in a new column of the stocks_df dataframe
	stocks_df['Optimized Portfolio'] = 0
	for ticker, weight in weights.items():
		stocks_df['Optimized Portfolio'] += stocks_df[ticker]*weight

except Exception as e:
    logging.exception('An error occurred: %s', str(e))
#___________________________________________2/2/23 _________________________________________________________

stocks_df['Optimized Portfolio Amounts'] = 0
stocks_df2 = stocks_df
stocks_df2['Time'] = stocks_df2.index
for ticker, weight in weights.items():
	stocks_df2['Optimized Portfolio Amounts'] += stocks_df2[ticker]*(weight/100)*amount

# # This code is to display how much the initial investment would be worth today
last_index = len(stocks_df2) - 1
for i in range(last_index, -1, -1):
    if not pd.isna(stocks_df2["Optimized Portfolio Amounts"].iloc[i]):
        previous_date_value = stocks_df2["Optimized Portfolio Amounts"].iloc[i]
        break

weights_df['weights'] = (weights_df['weights']).round(2)
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

st.markdown(f''' ####  If you would have invested :green[$ *{amount:,.2f}*] in the Optimized Portfolio on :blue[*{start_date}*]
#### it would be worth :green[$ *{previous_date_value:,.2f}*] today. :eyes: ''')
st.markdown("""---""")

#####################################################################################################################################################

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

###########################################################################################################################
# Save tickers to SessionState
if 'tickers' not in st.session_state:
    st.session_state.tickers = rsi_df
###########################################################################################################################

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
    figg = px.line(x=rsi.index, y=rsi.values, title=ticker + ' RSI')
    figg.update_xaxes(title_text='Date')
    figg.update_yaxes(title_text='RSI')

    # Add lower boundary line for oversold RSI range
    figg.add_hline(y=oversold, line_dash="dash", annotation_text="Oversold", 
                  annotation_position="bottom right", line_color="red")

    # Add upper boundary line for overbought RSI range
    figg.add_hline(y=overbought, line_dash="dash", annotation_text="Overbought", 
                  annotation_position="top right", line_color="red")

#####################################################################################################################################################

st.caption(':point_down: Check any of the boxes below to see more details about the portfolio.')

results = st.checkbox('Detailed Results')
if results:
	# Display pie chart created below 

	# Tables of weights and amounts
    col1, col2 = st.columns(2)
    with col1:
		# display the weights_df dataframe
        st.markdown('''#### WEIGHTS 
					(must add up to 1) ''')
        st.dataframe(weights_df)
		
    with col2:
        st.markdown(f'''#### BUY THIS AMOUNT 
					(must add up to $ {amount}) ''')
        st.dataframe(amounts_sorted)
    
	# Create a pie chart of the amounts_sorted dataframe

    st.header('Performance Expectations:')

	# st.subheader('Expected annual return: {}%'.format((expected_annual_return*100).round(2)))
    st.markdown(''' ### Expected annual return: :green[{}%] '''.format((expected_annual_return*100).round(2)))
    big_wrds = st.expander("big wrds??")
    big_wrds.caption("Annual expected return shows how much money you can get from a thing you put your eggs in. Big number good, little number bad.")

	# st.subheader('Annual volatility: {}%'.format((annual_volatility*100).round(2)))
    st.markdown(''' ### Annual volatility: :green[{}%] '''.format((annual_volatility*100).round(2)))
    so_wat = st.expander("so wat??")
    so_wat.caption("Volatility mean big ups and big downs for investment. More ups and downs mean more danger, but also maybe more meat for hunt.")

	# st.subheader('Sharpe Ratio: {}'.format(sharpe_ratio.round(2)))
    st.markdown(''' ### Sharpe Ratio: :green[{}] '''.format(sharpe_ratio.round(2)))
    wat_dis = st.expander("wat dis??")
    wat_dis.caption("Bigger number good, smaller number bad. Sharpe big, risk go away.")
    wat_dis.caption(r"""
			$$
		\frac{R_p - R_f}{\sigma_p}
		$$
		where:
		- $R_p$ is the average return of the portfolio
		- $R_f$ is the risk-free rate
		- $\sigma_p$ is the standard deviation of the portfolio returns
		""")
	
    st.markdown(''' ### Portfolio Correlation: :green[{}] '''.format(avg_corr.round(2)))
    hmm = st.expander("hmm??")
    hmm.caption('''Correlation shows how stocks move together, more together
	*(larger correlation number)* means more risk. Spread out stocks *(smaller correlation number)*
	less risk, like having different tools for different jobs.''')
    hmm.caption('''
	- A correlation of 1 = you are comparing the same stock to itself, so it is 100% correlated.
	- A correlation of 0 = you are comparing two stocks that are not correlated at all.
	- A correlation of -1 = you are comparing two stocks that are negatively correlated, meaning that when one goes up, the other goes down.
	''')
	# #Combine the 4 performance expectations into a table to pass through to chat gpt
    performance = pd.DataFrame({
		'Expected Annual Return': [expected_annual_return*100], 
		'Annual Volatility': [annual_volatility*100], 
		'Sharpe Ratio': [sharpe_ratio], 
		'Portfolio Correlation': [avg_corr]
		})
	# st.dataframe(performance)__this code used for testing

	# Optimized Portfolio: Cumulative Returns
    fig = px.line(stocks_df2, x='Time', y='Optimized Portfolio Amounts', title= 'Optimized Portfolio: Cumulative Returns')
    fig.update_yaxes(title_text='$ Amount')
    st.plotly_chart(fig)
    st.caption('Click and drag a box on the graph to zoom in on a specific time period.:point_up:')
    st.markdown("""---""")

show_more = st.checkbox('More Deeeetzz')
if show_more:

	# add link to Correlation Matrix header that takes you to investopedia article on correlation
	st.header('Correlation Matrix') # https://www.investopedia.com/terms/c/correlationcoefficient.asp
	st.markdown('''[Correlation Info](https://www.investopedia.com/terms/c/correlationcoefficient.asp)''')
	st.plotly_chart(fig_corr) # fig_corr is not a plotly chart
	st.caption(''' Portfolio Correlation: {} '''.format(avg_corr.round(2)))

	st.subheader("Optimized MaxSharpe Portfolio Performance")
	st.markdown('''[Efficient Frontier Info](https://www.investopedia.com/terms/e/efficientfrontier.asp)''')
	st.image(fig_efficient_frontier)
	st.markdown("""---""")

	st.header('Individual Stock Prices and Cumulative Returns')
	st.plotly_chart(fig_price)
	st.markdown("""---""")

	st.plotly_chart(fig_cum_returns)
	st.write(logging.exception(''))
	st.markdown("""---""")

# #################################################################################################################################

# GPT_says = st.checkbox('AI, WHAT DOES ALL THIS MEAN? :crystal_ball: ')
# if GPT_says:
# # Display the weights_df dataframe
# 	report_title = " AI Generated Report  "  
# 	# Specify font to be inconsolata
# 	report_icon = ":crystal_ball:" # emojis: https://www.webfx.com/tools/emoji-cheat-sheet/
# 	st.header('Summary')
# 	st.title(report_icon + " " + report_title + " " + report_icon)
# 	st.write('(give it a moment to generate response)')

# 	# Get the api key from the .env file
# 	openai.api_key = os.getenv("OPENAI_API_KEY")

# 	# Call GPT-3 to generate summary
# 	prompt = f'''write a report on the {weights_df} dataframe and the {performance} metrics of the dataframe. 
# 	Please provide your information using markdown labeled sub-headings and bullet points for each block of text.
# 	Explain the weights assigned to each asset and why they were assigned those weights. 
# 	Discuss the Sharpe ratio and correlation in {performance}.
# 	Compare the optimized portfolio to the S&P 500 and discuss the differences.
# 	Explain the expected returns, volatility in {performance}, and best time horizon for the portfolio.
# 	'''
# 	response = openai.Completion.create(
# 		model="text-davinci-003", 
# 		prompt= prompt,
# 		temperature=.7,
# 		max_tokens=1000, # the tokens are the max number of words. 
# 		top_p=1.0,
# 		frequency_penalty=0.0,
# 		presence_penalty=0.0
# 	)
# 	resp = (f"\n {response['choices'][0]['text']}")
# 	st.markdown(resp)
# #################################################################################################################################


