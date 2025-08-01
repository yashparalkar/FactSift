import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from rag_engine.rag_engine import initialize_rag_pipeline, process_query
from rag_engine.pdf_qa import PDFContextRetriever
import tempfile

st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("📰📄 News & PDF Q&A Chatbot")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pipeline" not in st.session_state:
    st.session_state.pipeline = initialize_rag_pipeline()

# Tabs for News & PDF
tab1, tab2 = st.tabs(["🗞️ News Chat", "📄 PDF QA"])

# ----------------------------
# TAB 1: News Chat Interface
# ----------------------------
with tab1:
    st.subheader("Chat About Current News")

    query = st.chat_input("Ask something about current news...")

    if query:
        with st.spinner("Thinking..."):
            response = process_query(query, st.session_state.pipeline, st.session_state.chat_history)
            st.session_state.chat_history = response["chat_history"]
            st.session_state.chat_history.append(("bot", response["answer"]))


        st.session_state.chat_history.append(("user", query))
        st.session_state.chat_history.append(("bot", response["answer"]))

    for role, message in st.session_state.chat_history:
        if role == "user":
            st.chat_message("user").write(message)
        else:
            st.chat_message("assistant").write(message)

# ----------------------------
# TAB 2: PDF Question Answering
# ----------------------------
with tab2:
    st.subheader("Ask Questions About a PDF")

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            pdf_path = tmp_file.name

        question_pdf = st.text_input("Ask a question about the uploaded PDF")

        if question_pdf:
            with st.spinner("Analyzing PDF..."):
                pdf_retriever = PDFContextRetriever(file_path=pdf_path)
                context = pdf_retriever.retrieve_context(question_pdf)
                pdf_answer = pdf_retriever.generate(question_pdf, context)

            st.markdown("**Answer from PDF:**")
            st.write(pdf_answer)
