import sys
import os
from typing import List

# -------------------------------------------------
# Fix Python path to access project modules
# -------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import streamlit as st

# Backend modules
from modules.data_input import load_document
from modules.data_preprocessing import preprocess_text
from modules.chunking import chunk_text
from modules.embedding import EmbeddingModel
from modules.vector_store import VectorStore
from modules.retrieval import Retriever
from modules.generation import ResponseGenerator

# Frontend helpers
from frontend.state_manager import (
    initialize_state,
    add_uploaded_file,
    add_query_response,
    reset_state,
    has_documents,
)

from frontend.ui_components import (
    render_upload_box,
    render_uploaded_file_list,
    render_query_input,
    render_response,
    render_reset_button,
)

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(page_title="Multi-Model RAG System", layout="wide")

# -------------------------------------------------
# Custom CSS
# -------------------------------------------------
st.markdown(
    """
    <style>
    .upload-box {
        border: 2px dashed #4A90E2;
        padding: 30px;
        border-radius: 12px;
        background-color: #F8FBFF;
        text-align: center;
    }
    .response-box {
        border: 1px solid #E0E0E0;
        padding: 20px;
        border-radius: 12px;
        background-color: gray;
    }
    .file-item {
        font-size: 14px;
        margin-bottom: 6px;
    }
    .small-text {
        font-size: 12px;
        color: gray;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------
# Initialize Session State
# -------------------------------------------------
initialize_state()

# -------------------------------------------------
# Title
# -------------------------------------------------
st.title("📄 Multi-Model RAG System for Document Intelligence")

# -------------------------------------------------
# Layout
# -------------------------------------------------
left_col, right_col = st.columns([1, 2])

# =================================================
# LEFT PANEL — Document Upload & Processing
# =================================================
with left_col:
    st.subheader("📂 Upload Documents")

    uploaded_files = render_upload_box(
        accepted_types=["pdf", "docx", "csv", "txt", "png", "jpg", "jpeg"]
    )

    if uploaded_files:
        if st.button("⚙️ Process Documents"):
            with st.spinner("Processing documents..."):
                for file in uploaded_files:
                    # Absolute upload directory
                    upload_dir = os.path.join(PROJECT_ROOT, "data", "sample_docs")
                    os.makedirs(upload_dir, exist_ok=True)

                    # Windows-safe filename
                    safe_filename = file.name.replace(" ", "_")

                    save_path = os.path.join(upload_dir, safe_filename)

                    # Save uploaded file
                    with open(save_path, "wb") as f:
                        f.write(file.getbuffer())

                    # Pipeline
                    raw_text = load_document(save_path)
                    clean_text = preprocess_text(raw_text)
                    chunks = chunk_text(clean_text)

                    embeddings = st.session_state.embedding_model.embed_texts(chunks)
                    st.session_state.vector_store.add_documents(chunks, embeddings)

                    add_uploaded_file(safe_filename)

            st.success("✅ Documents processed successfully!")

    # Show uploaded files
    render_uploaded_file_list(st.session_state.uploaded_files)

    # Reset
    if has_documents():
        if render_reset_button():
            reset_state()
            st.rerun()

# =================================================
# RIGHT PANEL — Query & Response
# =================================================
with right_col:
    st.subheader("🔍 Ask a Question")

    query = render_query_input()

    if st.button("Search"):
        if not has_documents():
            st.warning("Please upload and process documents first.")
        elif not query.strip():
            st.warning("Please enter a query.")
        else:
            retriever = Retriever(
                st.session_state.vector_store,
                st.session_state.embedding_model,
                top_k=3,
            )

            with st.spinner("Generating response..."):
                retrieved_chunks: List[str] = retriever.retrieve(query)
                generator = ResponseGenerator()
                answer = generator.generate_answer(query, retrieved_chunks)

            add_query_response(query, answer)

            render_response(answer, retrieved_chunks)

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("---")
st.caption("Multi-Model RAG System for Document Intelligence")
