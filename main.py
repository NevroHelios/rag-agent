import streamlit as st
from dotenv import load_dotenv

from chain import build_chain

load_dotenv()
st.set_page_config(page_title="RAG Institute", page_icon=":book:", layout="wide")


def main():
    st.title("Hello from rag-institute!")
    chain = build_chain()
    
    # Initialize chat history
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
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                response = chain.invoke(prompt)
                st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
