import streamlit as st

st.header('You May Have Some Questions:')
st.markdown(''' 
    You have 100 USD. You are interested in investing it and have 4 assets you like but you have some questions: 
    
    1. How much should you invest in each asset? 

    2. How long should you invest for **(time horizon)**? 
    
    3. What is the expected return? 
    
    4. How risky is it **(volatility)**? 
    
    5. How similar is the behavior of the assets to each other **(correlation)**?


    We aim to show you how to answer these questions using the tools of portfolio optimization.
    Click the **```MPT Analysis```** button in the navigation pane on the left to go to the next page where we will add our inputs and get the answers to all these questions.

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