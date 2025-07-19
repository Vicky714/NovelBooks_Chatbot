import streamlit as st
from backend import book_bot

st.header('NOVELBOOKS_BOT')
user_input = st.text_input('Enter your question here...')

if st.button('start'):
    st.write(book_bot(user_input))

# st.rerun()

