import streamlit as st

st.header('How to use this app:')
st.markdown('''

    1. Choose a start date and end date for your analysis. The time interval between the start and end date will be your time horizon that we will use to calculate the expected return and volatility of your portfolio.
   
    2. Choose the assets you want to include in your portfolio. 
        - There are some assets that are populated by default just to demonstrate the app. You may add any asset you are interested in as long as it is in the correct format.
        - The correct format for stocks is the ticker symbol in all caps separated by a comma and no spaces. 
            - For example, if you want to include Apple, Microsoft, and Amazon in your portfolio, you would enter: AAPL,MSFT,AMZN
        - The correct format for cryptocurrencies is token symbol followed by a - and the currency it is represented in such as USD. This should be written in all caps and separated by a comma and no spaces. 
            - For example, if you want to include Bitcoin, Ethereum, and Dogecoin in your portfolio, you would enter: BTC-USD,ETH-USD,DOGE-USD
    
    3. Once you have entered the dates and assets, just click enter or click out of the text box and the app will answer all the above questions above about your portfolio.

''')

# Hide Streamlit Menu and Footer
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)