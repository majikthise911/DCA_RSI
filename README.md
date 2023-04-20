# SLPortOpt
README for Financial Portfolio Optimizer App

## Introduction:
The Financial Portfolio Optimizer app allows users to optimize their portfolio by calculating the optimal allocation of assets based on their desired risk-return profile. The app uses historical asset price data to calculate expected returns, sample covariance matrix, and efficient frontier to optimize the portfolio. The user can enter an investment amount, tickers of the assets they want to include in their portfolio, and the start and end dates for backtesting their portfolio. The app also includes features such as a correlation matrix, individual stock prices and cumulative returns, and an AI-generated report.

To use the app, simply enter the investment amount, tickers of the assets you want to include in your portfolio, and the start and end dates for backtesting your portfolio. The app will then display a pie chart showing the optimized allocation of assets based on the desired risk-return profile. The user can also view the correlation matrix, individual stock prices and cumulative returns, and an AI-generated report.

## Technologies used:

- Python

- Streamlit: a Python library for creating web apps

- PyPortfolioOpt: a Python library for portfolio optimization

- yfinance: a Python library for downloading historical asset price data from Yahoo Finance

- Plotly: a Python library for creating interactive data visualizations

- Matplotlib: a Python library for creating static data visualizations

- Plotly Express & Seaborn: For creating data visualizations

- OpenAI: an AI research laboratory consisting of the for-profit corporation OpenAI LP and its parent company, the non-profit OpenAI Inc.

Features:

# Investment Amount: 
Allows the user to enter the investment amount, which is used to calculate the optimal allocation of assets based on the desired risk-return profile.
Tickers: Allows the user to enter the tickers of the assets they want to include in their portfolio.
Backtesting: Allows the user to enter the start and end dates for backtesting their portfolio.
Correlation Matrix: Displays the correlation matrix between the assets in the portfolio.
Individual Stock Prices and Cumulative Returns: Displays the individual stock prices and cumulative returns of the assets in the portfolio.
AI-generated Report: Generates an AI-generated report summarizing the portfolio's performance.
Limitations:

# Data Availability: 
The app relies on the availability of historical asset price data, which may not be available for all assets.
Accuracy: The app relies on historical data to predict future returns, which may not be accurate due to unforeseen events or changes in market conditions.
Risk: The app does not guarantee a profit and involves risk, which is inherent in any investment.
Conclusion:
The Financial Portfolio Optimizer app is a useful tool for investors looking to optimize their portfolio based on their desired risk-return profile. The app is easy to use and provides valuable insights into the performance of the portfolio. However, users should be aware of the limitations of the app and the inherent risks involved in any investment.



TODO: add future price predictions - look at code in ml dir
TODO: embed a loom video 
TODO: Add a button to refresh the page+
TODO: add microcap stocks and cryptos section that are most likely to succeed in the future
TODO: add daily top 10 stocks and cryptos with lowest rsi and highest rsi
TODO: add section where you can input trades and get a report on good or badness of trade? 
TODO: *********find a way to create support and resistance indicators with ai - ability to put in an asset and have gpt draw support and resistance lines 
TODO: ADD wale watchers section
TODO: Add a button to download the optimized portfolio weights - Completed
TODO: Change how the optimized portfolio list is sorted. Sort by weights instead of alphabetically - Completed
TODO: Add section to have GPT-3 generate a stock portfolio
TODO: Add section to have GPT-3 generate a report on the portfolio
TODO: Add button to each graph that shows the code/math to generate the graph
Maybe find a way to integrate chat bot to explain the math/code with the chat bot that I already built in streamlit
