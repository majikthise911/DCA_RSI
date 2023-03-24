import streamlit as st
import openai
import os


# Hide Streamlit Menu and Footer
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

report_icon = ":crystal_ball:" # emojis: https://www.webfx.com/tools/emoji-cheat-sheet/
st.title(report_icon + " " + 'Discover' + " " + report_icon)
report_title = " Top 10s  "  
st.header(report_title)
# specify font to be inconsolata
st.write('(give it a few seconds to load)')

# # get the api key from the .env file
openai.api_key = os.getenv("OPENAI_API_KEY")

# Create the dataframe
# fig_cum_returns_optimized = st.sidebar.selectbox("Dataframe", fig_cum_returns_optimized)
# Call GPT-3 to generate summary
prompt = f''' First: List the top most interesting things you have discovered about the stock market, cryptocurrencies, or anything else you are interested in.
Next: Create 3 tables with the following questions label them top 10 cryptocurrencies by market cap with the lowest hourly rsi, top 10 stocks by market cap with the lowest hourly rsi, and top 10 cryptocurrencies that have a market cap of $100,000,000 or below right now that are most likely to succeed in the long term and why.
1. What are the top 10 cryptocurrencies by market cap with the lowest hourly rsi currently? 
2. What are the top 10 stocks by market cap with the lowest hourly rsi currently? 
3. What are the top 10 cryptocurrencies that have a market cap of $100,000,000 or below right now that are most likely to succeed in the long term and why?
'''
response = openai.Completion.create(
	model="text-davinci-003", 
	prompt= prompt,
	temperature=.7,
	max_tokens=1000, # the tokens are the max number of words. 
	top_p=1.0,
	frequency_penalty=0.0,
	presence_penalty=0.0
)
resp = (f"\n {response['choices'][0]['text']}")
#st.markdown(f"## Summary of Optimized Portfolio \n {response['choices'][0]['text']}")
st.markdown(resp)

# TODO: have to feed data to gpt 3 to generate the tables. currently it is using old data 