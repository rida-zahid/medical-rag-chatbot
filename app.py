import streamlit as st
from query import ask

st.set_page_config(page_title="Medical RAG Assistant")
st.title(" Medical RAG Chatbot")
st.write("Ask questions based on skin cancer research papers.")

question = st.text_input("Ask a question:")

if st.button("Search"):
    if question.strip():
        with st.spinner("Thinking..."):
            answer = ask(question)
        st.write("### Answer:")
        st.write(answer)
    else:
        st.warning("Please enter a question")