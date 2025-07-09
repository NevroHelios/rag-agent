import streamlit as st
from dotenv import load_dotenv

from chain import build_chain

load_dotenv()

st.title("Hello from rag-institute!")

with st.sidebar:
    st.header("Additional Context")
    st.info("Add any additional context or information that might help the model answer questions more accurately.")
    st.warning("Currently supports only vector retrieval for additional context.")
    uploaded_file = st.file_uploader(label="Upload a file", type=["pdf"]) # TODO: add more file types

chain = build_chain()

# init history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("How to detect CME from particle activity in L1?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            response = chain.invoke(prompt)
            st.markdown(response)
    
    # add response to history
    st.session_state.messages.append({"role": "assistant", "content": response})