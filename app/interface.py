import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
from rag_engine.rag_engine import initialize_rag_pipeline, process_query
from rag_engine.pdf_qa import PDFContextRetriever
import tempfile
import time
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="FactSift Chatbot", 
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 3rem;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    .chat-container {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        max-height: 400px;
        overflow-y: auto;
    }
    
    .source-info {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 0.5rem;
        margin: 0.5rem 0;
        border-radius: 4px;
        font-size: 0.9rem;
    }
    
    .error-message {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 0.5rem;
        margin: 0.5rem 0;
        border-radius: 4px;
        color: #c62828;
    }
    
    .success-message {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
        padding: 0.5rem;
        margin: 0.5rem 0;
        border-radius: 4px;
        color: #2e7d2e;
    }
    
    .stTab {
        background-color: white;
        border-radius: 10px;
        padding: 0.5rem;
    }
    
    .upload-section {
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        margin: 0.5rem 0;
        background-color: #fafafa;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">📰📄 FactSift</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Your AI-powered fact-checking and document analysis assistant</p>', unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "pipeline" not in st.session_state:
        with st.spinner("🔧 Initializing FactSift engine..."):
            st.session_state.pipeline = initialize_rag_pipeline()
    if "pdf_history" not in st.session_state:
        st.session_state.pdf_history = []
    if "current_pdf" not in st.session_state:
        st.session_state.current_pdf = None

initialize_session_state()

# Sidebar with information
with st.sidebar:
    st.markdown("### 📊 Chat Statistics")
    # Safe counting with error handling
    news_queries = len([msg for msg in st.session_state.chat_history if isinstance(msg, (list, tuple)) and len(msg) >= 2 and msg[0] == "user"])
    pdf_queries = len([msg for msg in st.session_state.pdf_history if isinstance(msg, (list, tuple)) and len(msg) >= 2 and msg[0] == "user"])
    
    st.metric("News Queries", news_queries)
    st.metric("PDF Questions", pdf_queries)
    
    st.markdown("### 🔧 Controls")
    if st.button("🗑️ Clear News Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
    
    if st.button("🗑️ Clear PDF Chat", use_container_width=True):
        st.session_state.pdf_history = []
        st.session_state.current_pdf = None
        st.rerun()
    
    st.markdown("### ℹ️ About FactSift")
    st.info("""
    **News Chat**: Ask questions about current events and news. 
    FactSift will search and analyze recent information.
    
    **PDF QA**: Upload any PDF document and ask specific questions 
    about its content.
    """)

# Main tabs
tab1, tab2 = st.tabs(["🗞️ News Chat", "📄 PDF Analysis"])

# ----------------------------
# TAB 1: Enhanced News Chat Interface
# ----------------------------
with tab1:
    st.markdown("### 💬 Chat About Current News")
    # Chat history display
    if st.session_state.chat_history:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for msg in st.session_state.chat_history:
            # Ensure message is in correct format
            if isinstance(msg, (list, tuple)) and len(msg) >= 2:
                role, message = msg[0], msg[1]
                if role == "user":
                    st.chat_message("user").write(message)
                elif role == "bot" or role == "assistant":
                    st.chat_message("assistant").write(message)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("👋 Welcome! Ask me anything about current news and events. I'll search for the latest information to give you accurate, up-to-date answers.")
    
    # Chat input with better UX
    col1, col2 = st.columns([6, 1])
    with col1:
        query = st.chat_input("Ask something about current news... (e.g., 'What's happening with AI regulation?')")
    
    # Example questions
    st.markdown("**💡 Try asking about:**")
    example_cols = st.columns(3)
    with example_cols[0]:
        if st.button("🌍 Global Events", use_container_width=True):
            st.session_state.example_query = "What are the major global events happening this week?"
    with example_cols[1]:
        if st.button("💼 Business News", use_container_width=True):
            st.session_state.example_query = "What are the latest developments in the tech industry?"
    with example_cols[2]:
        if st.button("🔬 Science & Tech", use_container_width=True):
            st.session_state.example_query = "What are the recent breakthroughs in AI research?"
    
    # Handle example queries
    if hasattr(st.session_state, 'example_query'):
        query = st.session_state.example_query
        delattr(st.session_state, 'example_query')
    
    # Process query
    if query:
        # Add user message immediately
        st.session_state.chat_history.append(("user", query))
        
        try:
            with st.spinner("🔍 Searching for latest information..."):
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                response = process_query(query, st.session_state.pipeline, st.session_state.chat_history)
                
                # Fix the duplicate entries issue
                st.session_state.chat_history = response.get("chat_history", st.session_state.chat_history)
                st.session_state.chat_history.append(("bot", response["answer"]))
                
                progress_bar.empty()
                
                # Show sources if available
                if "sources" in response and response["sources"]:
                    st.markdown('<div class="source-info">📚 <strong>Sources:</strong><br>' + 
                              '<br>'.join([f"• {source}" for source in response["sources"][:3]]) + 
                              '</div>', unsafe_allow_html=True)
                
        except Exception as e:
            st.markdown(f'<div class="error-message">❌ <strong>Error:</strong> {str(e)}</div>', 
                       unsafe_allow_html=True)
        
        st.rerun()

# ----------------------------
# TAB 2: Enhanced PDF Question Answering
# ----------------------------
with tab2:
    st.markdown("### 📄 Document Analysis & Q&A")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "📁 Upload your PDF document", 
            type=["pdf"],
            help="Upload any PDF document to analyze and ask questions about its content"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded_file:
            st.markdown(f'<div class="success-message">✅ <strong>Uploaded:</strong> {uploaded_file.name}</div>', 
                       unsafe_allow_html=True)
            
            # Store PDF info
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                st.session_state.current_pdf = {
                    "name": uploaded_file.name,
                    "path": tmp_file.name,
                    "size": len(uploaded_file.getvalue())
                }
    
    with col2:
        if st.session_state.current_pdf:
            st.markdown("**📊 Document Info:**")
            st.write(f"**Name:** {st.session_state.current_pdf['name']}")
            st.write(f"**Size:** {st.session_state.current_pdf['size'] / 1024:.1f} KB")
            
            # Example questions for PDF
            st.markdown("**💡 Example questions:**")
            if st.button("📝 Summarize this document", key="pdf_summary"):
                st.session_state.pdf_example_query = "Can you provide a comprehensive summary of this document?"
            if st.button("🔍 Key findings", key="pdf_findings"):
                st.session_state.pdf_example_query = "What are the main findings or conclusions in this document?"
            if st.button("📊 Important data", key="pdf_data"):
                st.session_state.pdf_example_query = "What are the most important statistics or data points mentioned?"
    
    # PDF Chat History
    if st.session_state.pdf_history and st.session_state.current_pdf:
        st.markdown("### 💬 Document Q&A History")
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for msg in st.session_state.pdf_history:
            # Ensure message is in correct format
            if isinstance(msg, (list, tuple)) and len(msg) >= 2:
                role, message = msg[0], msg[1]
                if role == "user":
                    st.chat_message("user").write(message)
                elif role == "bot" or role == "assistant":
                    st.chat_message("assistant").write(message)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # PDF Question Input
    if st.session_state.current_pdf:
        question_pdf = st.chat_input("Ask a question about the uploaded PDF...")
        
        # Handle example queries
        if hasattr(st.session_state, 'pdf_example_query'):
            question_pdf = st.session_state.pdf_example_query
            delattr(st.session_state, 'pdf_example_query')
        
        if question_pdf:
            # Add question to history
            st.session_state.pdf_history.append(("user", question_pdf))
            
            try:
                with st.spinner("📖 Analyzing document..."):
                    progress_bar = st.progress(0)
                    
                    # Initialize PDF retriever
                    progress_bar.progress(25)
                    pdf_retriever = PDFContextRetriever(file_path=st.session_state.current_pdf["path"])
                    
                    # Retrieve context
                    progress_bar.progress(50)
                    context = pdf_retriever.retrieve_context(question_pdf)
                    
                    # Generate answer
                    progress_bar.progress(75)
                    pdf_answer = pdf_retriever.generate(question_pdf, context)
                    progress_bar.progress(100)
                    
                    # Add answer to history
                    st.session_state.pdf_history.append(("bot", pdf_answer))
                    
                    progress_bar.empty()
                    
                    # Show context info
                    if context:
                        st.markdown('<div class="source-info">📄 <strong>Analyzed sections:</strong> ' + 
                                  f'Found relevant content from the document</div>', 
                                  unsafe_allow_html=True)
                
            except Exception as e:
                st.markdown(f'<div class="error-message">❌ <strong>Error analyzing PDF:</strong> {str(e)}</div>', 
                           unsafe_allow_html=True)
            
            st.rerun()
    else:
        st.info("📁 Please upload a PDF document to start asking questions about it.")

# Footer
st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #666; font-size: 0.9rem;">',
    unsafe_allow_html=True
)
st.markdown("🤖 **FactSift** - Powered by Advanced RAG Technology | Built with Streamlit")
st.markdown('</div>', unsafe_allow_html=True)