import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="RAG Institute", page_icon=":book:", layout="wide")


st.title("Welcome to RAG Institute!")
st.write("Please select a page from the sidebar to get started.")


st.image("static/image.png", use_column_width=True)