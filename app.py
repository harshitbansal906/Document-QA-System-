import os
import shutil
import logging
import streamlit as st
from utils.rag import RAGSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
VECTOR_DIR = os.path.join(BASE_DIR, "vector_store")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Page configuration
st.set_page_config(
    page_title="Offline Document QA System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session States
if "rag_system" not in st.session_state:
    st.session_state.rag_system = RAGSystem(
        embedding_model_name="all-MiniLM-L6-v2",
        llm_model_name="google/flan-t5-base",
        models_dir=MODELS_DIR,
        vector_store_dir=VECTOR_DIR
    )

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "ingested_stats" not in st.session_state:
    # Try to load existing index if present on disk
    loaded = st.session_state.rag_system.vector_store.load()
    if loaded and st.session_state.rag_system.vector_store.chunks:
        chunks = st.session_state.rag_system.vector_store.chunks
        distinct_sources = list(set(c["metadata"]["source"] for c in chunks))
        st.session_state.ingested_stats = {
            "documents_ingested": len(distinct_sources),
            "total_chunks": len(chunks),
            "status": "ready"
        }
    else:
        st.session_state.ingested_stats = None


# Helper to clear vectors and local uploaded files
def reset_system():
    # Reset RAG Vector store files
    st.session_state.rag_system.reset_database()
    # Wipe uploads folder
    if os.path.exists(UPLOADS_DIR):
        shutil.rmtree(UPLOADS_DIR)
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    # Clear states
    st.session_state.ingested_stats = None
    st.session_state.chat_history = []
    st.success("System database reset successfully!")
    st.rerun()


# Sidebar Layout
with st.sidebar:
    st.title("Document-QA-System")
    
    st.subheader("Project Info")
    st.info(
        "A production-ready offline Document Question-Answering system. "
        "It parses PDFs, splits text, creates embeddings, and performs semantic search "
        "using FAISS and a local Flan-T5 model, guaranteeing data privacy."
    )
    
    st.divider()
    
    st.subheader("Upload PDFs")
    uploaded_files = st.file_uploader(
        "Select PDF documents to ingest:", 
        type=["pdf"], 
        accept_multiple_files=True,
        help="You can upload multiple files."
    )
    
    # Process files button
    if uploaded_files:
        if st.button("Process & Index PDFs", type="primary", use_container_width=True):
            # Save uploaded files to the uploads directory
            saved_files_to_process = []
            for file in uploaded_files:
                file_path = os.path.join(UPLOADS_DIR, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getvalue())
                saved_files_to_process.append({
                    "source": file_path,
                    "name": file.name
                })
            
            # Run Ingestion
            with st.spinner("Processing documents (downloading models on first-run)..."):
                try:
                    stats = st.session_state.rag_system.ingest_files(
                        files=saved_files_to_process,
                        chunk_size=500,
                        chunk_overlap=100
                    )
                    
                    if stats.get("status") == "success":
                        st.session_state.ingested_stats = stats
                        st.success("Ingestion completed successfully!")
                        st.rerun()
                    else:
                        st.error(stats.get("message", "Ingestion failed."))
                except Exception as ex:
                    st.error(f"Error during ingestion: {ex}")
                    logger.exception("Ingestion failed")
                    
    st.divider()
    
    # System resets
    if st.button("Reset Knowledge Base", type="secondary", use_container_width=True):
        reset_system()

# Main Panel Layout
st.title("Offline Q&A Assistant")
st.caption("Ask questions about your uploaded documents locally.")

# Ingestion Stats Display using native st.metric
if st.session_state.ingested_stats:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label="Documents Indexed", 
            value=st.session_state.ingested_stats.get("documents_ingested", 0)
        )
    with col2:
        st.metric(
            label="Text Chunks", 
            value=st.session_state.ingested_stats.get("total_chunks", 0)
        )
    with col3:
        st.metric(
            label="Environment Security", 
            value="Offline & Local"
        )
else:
    st.info("Please upload PDF files in the sidebar and click 'Process & Index PDFs' to begin.")

st.divider()

# Chat History Container
st.subheader("Chat History")
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        
        # Display sources in a native expander if available
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Sources Cited"):
                for src in msg["sources"]:
                    meta = src.get("metadata", {})
                    st.write(f"📄 **{meta.get('source', 'Unknown')}** - Page {meta.get('page', '?')} (Score: {src.get('score', 0.0):.4f})")
                    st.write(src.get("text", ""))

st.divider()

# Question Box and Ask Button using a native form for side-by-side alignment
st.subheader("Ask a Question")
with st.form("query_form", clear_on_submit=False):
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        user_query = st.text_input(
            label="Question input",
            placeholder="Ask a question about your documents...",
            label_visibility="collapsed"
        )
    with col_btn:
        ask_submitted = st.form_submit_button("Ask", use_container_width=True)

# Process Submitted Question
if ask_submitted and user_query.strip():
    if not st.session_state.ingested_stats:
        st.error("Please ingest documents before asking questions.")
    else:
        # Display user query in chat
        with st.chat_message("user"):
            st.write(user_query)
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        # In progress message and loader spinner
        with st.spinner("Searching database and generating answer..."):
            try:
                answer, sources = st.session_state.rag_system.query(
                    question=user_query,
                    top_k=4,
                    temperature=0.0
                )
                
                # Display assistant message in chat
                with st.chat_message("assistant"):
                    st.write(answer)
                    if sources:
                        with st.expander("Sources Cited"):
                            for src in sources:
                                meta = src.get("metadata", {})
                                st.write(f"📄 **{meta.get('source', 'Unknown')}** - Page {meta.get('page', '?')} (Score: {src.get('score', 0.0):.4f})")
                                st.write(src.get("text", ""))
                                
                # Save message and sources to history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
                
                st.success("Answer generated successfully!")
                
            except Exception as e:
                st.error(f"Error during retrieval or answer generation: {e}")
                logger.exception("Query generation failed")
