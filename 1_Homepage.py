import requests
import streamlit as st

page_title = "WealthWyze"
page_icon = ":zap:"  # emojis: https://www.webfx.com/tools/emoji-cheat-sheet/
layout = "centered"
st.set_page_config(page_title = page_title, layout = "centered", page_icon = page_icon)
st.title(page_title + " " + page_icon)

# # Streamlit sidebar table of contents
# st.sidebar.markdown('''
# # Sections
# - [Welcome](#welcome)
# - [Questions](#you-may-have-some-questions)
# - [How to use this app](#how-to-use-this-app)
# ''', unsafe_allow_html=True)

st.markdown(''' # Why does your portfolio suck?

### You haven't optimized it yet! We have a solution.

### Enter, WealthWyze.

### We calculate and optimize your portfolio weights and show you how diversified your selections are.

''')

# add image from "image" folder in the root directory
st.image("image/newplot.png", use_column_width=True)

# add custom css to the button that links to the next page
button_style = st.markdown("""
<style>
   div.stButton    > button:first-child {
        top: 50%;
        left: 50%;
        background-image: linear-gradient(#b6007a, #522886);
        color: white;
        padding: 20px 50px;
        text-align: center;
        font-size: 60px;
        border-radius: 20px;
        width: 20%;
        height: 60%;
        }
</style>
""", unsafe_allow_html=True)
link = "<a href='https://wealthwyze.streamlit.app/Optimize'>Next</a>"
if st.button("Next"):
    st.markdown(link, unsafe_allow_html=True)

# Hide Streamlit Menu and Footer
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)