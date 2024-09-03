import WebApp_Final
import app2
import about
import streamlit as st

PAGES = {
    "About": about,
    "Sales Prediction": WebApp_Final,
    "Game Recommendation System": app2
}
st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()
