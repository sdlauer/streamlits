import streamlit as st
primaryColor="#F63366"
backgroundColor="#000000"
secondaryBackgroundColor="#F0F2F6"
textColor="#262730"
font="times"
age = st.slider('How old are you?', 0, 130, 25)
st.write("I'm ", age, 'years old')
