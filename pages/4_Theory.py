import streamlit as st

# Hide Streamlit Menu and Footer
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """

# Streamlit sidebar table of contents
st.sidebar.markdown('''
# Sections
- [Background](#background)
- [Cumulative Returns](#cumulative-returns)
- [Optimized Max Sharpe Portfolio Weights](#optimized-max-sharpe-portfolio-weights)
- [Sharpe Ratio](#sharpe-ratio)
- [Efficient Frontier](#efficient-frontier)
- [Correlation Matrix](#correlation-matrix)
- [Individual Stocks Graph](#individual-stocks-graph)
- [Cumulative Returns of Individual Stocks](#cumulative-returns-of-individual-stocks)
''', unsafe_allow_html=True)


st.markdown(hide_st_style, unsafe_allow_html=True)
pg_icon = ":scroll:"
st.title('Theory' + " " + pg_icon)
st.header('Background')
st.markdown('''

	#### How do financial advisors manage portfolios?


	Once upon a time, in the early 1950s, a brilliant economist named Harry Markowitz had an idea that would change the way we 
	think about investing forever. He had been struggling to find a way to balance the trade-off between risk and return in his 
	own portfolio, and he knew there had to be a better way.

	Markowitz's breakthrough was the realization that diversifying your investments across different asset classes could help to 
	reduce risk. He called this idea "Modern Portfolio Theory" (MPT) and it was quickly adopted by the likes of Warren Buffet, 
	who is known for using the principles of MPT to build his portfolio. Today, MPT is used by investors, portfolio managers, 
	and financial advisors around the world to help them make better investment decisions.
	
	Modern Portfolio Theory (MPT) is a framework for constructing investment portfolios that aims to maximize expected return 
	for a given level of risk. It is based on the idea that investors are risk-averse, meaning that they prefer a certain level 
	of return to a higher level of risk. MPT helps investors balance their desire for high returns with their need to minimize risk 
	by diversifying their investments across multiple asset classes.

	The core concept of MPT is the [Efficient Frontier](#efficient-frontier), which is a curve that represents the highest expected return for a given level 
	of risk. The efficient frontier is constructed by plotting the returns and standard deviations of all possible portfolios, and 
	selecting those portfolios that lie on the curve. These portfolios are considered efficient because they offer the highest expected 
	return for a given level of risk.

	### Assumptions
	MPT is based on several key assumptions:	

	1. Investors are rational and seek to maximize their expected utility. 
	2. Investors are risk-averse, meaning that they prefer a certain level of return to a higher level of risk.
	3. Markets are efficient, meaning that asset prices reflect all available information.
	4. There is a positive correlation between risk and return.
	5. Returns are normally distributed.

	### Methods
	And To achieve this goal, MPT uses several key tools and methods, including:

	1. [Efficient Frontier](#efficient-frontier) 
	
	2. [Mean-variance analysis](https://www.investopedia.com/terms/m/meanvariance-analysis.asp): This is a mathematical method used to calculate the expected return and risk of a portfolio. It involves calculating the expected returns and variances of the individual assets in the portfolio, and then combining them to find the expected return and variance of the portfolio as a whole.
	
	3. Diversification: MPT emphasizes the importance of diversifying investments across different asset classes, such as stocks, bonds, and real estate. Diversification helps to reduce risk by spreading investments across different types of assets, which can have different levels of risk and return.
	
	4. Risk-adjusted performance measures: MPT uses various performance measures to evaluate the risk-adjusted performance of a portfolio. One of the most popular measure is the [Sharpe Ratio](#sharpe-ratio), which measures the risk-adjusted return of a portfolio by dividing the excess return over the risk-free rate by the standard deviation of the portfolio returns.
	
	5. Portfolio optimization: MPT uses optimization techniques to find the optimal portfolio that lies on the efficient frontier. This involves using mathematical algorithms to find the portfolio with the highest expected return for a given level of risk, or the lowest risk for a given level of expected return.
	
	6. Asset allocation: MPT encourages investors to allocate their assets among different asset classes, based on their risk tolerance, investment horizon and financial goals. This allocation can be done using a variety of techniques such as strategic, tactical, or dynamic asset allocation.


	MPT is often used by financial advisors and investors to construct portfolios that are tailored to meet specific investment 
	goals. It is important to note, however, that MPT is based on several assumptions that may not hold true in all circumstances. 
	In practice, investors may need to consider other factors such as taxes, transaction costs, and personal preferences when constructing 
	their portfolios.

	In summary, Modern Portfolio Theory is a framework for constructing investment portfolios that aims to maximize expected return for a 
	given level of risk. It is based on the assumption that investors are risk-averse and seek to maximize their expected utility, and that 
	markets are efficient and prices reflect all available information. While MPT can be a useful tool for investors, it is important to consider 
	a range of factors when constructing a portfolio.

''')

st.header('Cumulative Returns')
st.markdown(''' 
	The cumulative return of an optimized portfolio is the total return that the portfolio has achieved over a certain period of time, taking into 
	account the effects of compound interest. It is a measure of the overall performance of the portfolio and reflects the combined effects of all 
	the individual returns of the assets that make up the portfolio.

	Cumulative return is an important measure in portfolio analysis because it helps investors evaluate the performance of their portfolio over 
	time and compare it to other portfolios or benchmarks. It can also provide insight into the risk-return trade-off of the portfolio, as portfolios 
	with higher cumulative returns may also carry higher levels of risk.

	Optimized portfolios are portfolios that have been constructed to maximize a specific objective, such as maximizing return or minimizing risk. 
	The cumulative return of an optimized portfolio can be used to assess the effectiveness of the optimization process and determine whether the 
	portfolio is achieving its intended objective.

''')

st.header('Optimized Max Sharpe Portfolio Weights')
st.markdown('This is simply the weights of the assets in the portfolio that maximize the Sharpe ratio.')


st.header('Sharpe Ratio')
st.markdown('''
The Sharpe Ratio is a popular risk-adjusted performance measure that is widely used to evaluate the performance of an investment relative to its risk. It was developed by William Sharpe, a Nobel laureate in economics, in the 1960s as a way to compare the performance of different investments.

#### Calculation
The Sharpe ratio is calculated by dividing the excess return of an investment over the risk-free rate by the standard deviation of the investment's returns.

The excess return is the difference between the investment's return and the risk-free rate, which is typically the return on a Treasury bill.
The standard deviation is a measure of the volatility or risk of the investment's returns.

#### Interpretation
The Sharpe ratio is a dimensionless measure, which means that it is not affected by the units of measurement used to express returns or risk. This makes it a useful tool for comparing the performance of investments with different expected returns and levels of risk.

The higher the Sharpe ratio, the better the risk-adjusted performance of an investment. A Sharpe ratio of 1 or higher is generally considered to be good, while a ratio below 1 is considered to be poor. A ratio of 0 means that the investment's returns are equal to the risk-free rate.

#### Applications
The Sharpe ratio is used to compare the performance of different investments, such as stocks, bonds, and mutual funds. It can also be used to compare the performance of different portfolios, as well as to evaluate the performance of a portfolio over time.

In addition to evaluating the performance of individual investments, the Sharpe ratio can also be used as a benchmark for evaluating the performance of portfolio managers. A portfolio manager with a high Sharpe ratio is considered to have generated good returns relative to the level of risk taken.

''')

st.header('Efficient Frontier')
st.markdown(''' 
	Efficient frontier is a graphical representation of the possible portfolios that can be created using a given set of assets. It shows the 
	trade-off between risk and return for different portfolio combinations, and helps investors identify the portfolios with the highest expected 
	return for a given level of risk, or the lowest level of risk for a given expected return.

	#### Plotting the efficient frontier
	This is the aim of going through all the topics above, to plot the efficient frontier. Efficient frontier is a graph with 
	‘returns’ on the Y-axis and ‘volatility’ on the X-axis. It shows us the maximum return we can get for a set level of volatility, 
	or conversely, the volatility that we need to accept for certain level of returns.

	A loop is necessary. In each iteration, the loop considers different weights for assets and calculates the return and volatility of that particular portfolio combination.
	We run this loop a 10000 times.
	To get random numbers for weights, we use the np.random.random() function. But remember that the sum of weights must be 1, so we divide those weights by their cumulative sum.
	
	The maximum Sharpe ratio is the portfolio on the efficient frontier that has the highest Sharpe ratio. 

	Efficient frontier and maximum Sharpe ratio are important tools in portfolio analysis because they help investors identify portfolios that offer 
	the highest expected return for a given level of risk, or the lowest level of risk for a given expected return. This can help investors make informed 
	decisions about how to allocate their investments and manage risk in their portfolio.

''')

st.header('Correlation Matrix')
st.markdown(''' 
	Correlation between assets in a portfolio is important to understand because it can affect the risk and return characteristics of the portfolio.

	If the assets in a portfolio are perfectly positively correlated, they will move in the same direction and by the same amount. This means that the 
	portfolio's risk and return will be largely determined by the individual risk and return characteristics of the assets.

	On the other hand, if the assets in a portfolio are perfectly negatively correlated, they will move in opposite directions and by the same amount. 
	This can help to diversify risk and reduce the overall risk of the portfolio.

	In reality, most assets have some level of correlation, which can be positive, negative, or zero. Understanding the correlation between the assets 
	in a portfolio can help investors to identify diversification opportunities and make informed decisions about how to allocate their investments. 
	It can also help investors to understand the potential risk and return trade-offs of different portfolio configurations.

''')


st.header('Individual Stocks Graph')
st.markdown(''' 
	This is simply a graph of the individual stocks in the portfolio plotted on the same graph.
''')


st.header('Cumulative Returns of Individual Stocks ')
st.markdown(''' 
	This is a graph of the cumulative returns of the individual stocks in the portfolio plotted on the same graph.
''')


