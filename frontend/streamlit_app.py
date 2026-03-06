import sys
import os
from typing import List
import streamlit as st
import streamlit.components.v1 as components

# -------------------------------------------------
# Fix Python path to access project modules
# -------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

# Backend & Frontend imports
from modules.data_input import load_document
from modules.data_preprocessing import preprocess_text
from modules.chunking import chunk_text
from modules.embedding import EmbeddingModel
from modules.vector_store import VectorStore
from modules.retrieval import Retriever
from modules.generation import ResponseGenerator
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
# Professional Dark Theme with Animated Grid Background
# -------------------------------------------------
def inject_dynamic_background():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Sora:wght@300;400;600;700&display=swap');

        /* ── Root Variables ── */
        :root {
            --bg-base:      #080d14;
            --bg-surface:   #0e1620;
            --bg-elevated:  #141e2d;
            --accent-blue:  #2d7bf4;
            --accent-cyan:  #00d4ff;
            --accent-teal:  #00b894;
            --text-primary: #e8edf5;
            --text-muted:   #7a8fa8;
            --border:       rgba(45, 123, 244, 0.18);
            --glow:         rgba(45, 123, 244, 0.12);
        }

        /* ── Full App Reset ── */
        html, body, .stApp {
            background-color: var(--bg-base) !important;
            font-family: 'Sora', sans-serif !important;
            color: var(--text-primary) !important;
        }

        /* ── Animated Grid Background ── */
        #dynamic-bg {
            position: fixed;
            inset: -10%;
            width: 120%;
            height: 120%;
            z-index: -2;
            background-color: var(--bg-base);
            background-image:
                linear-gradient(rgba(45, 123, 244, 0.06) 1px, transparent 1px),
                linear-gradient(90deg, rgba(45, 123, 244, 0.06) 1px, transparent 1px);
            background-size: 48px 48px;
            transition: transform 0.15s ease-out;
        }

        /* Radial glow overlay */
        #dynamic-bg::after {
            content: '';
            position: absolute;
            inset: 0;
            background: radial-gradient(ellipse 60% 50% at 50% 40%,
                rgba(45, 123, 244, 0.08) 0%,
                transparent 70%);
        }

        /* Scanline overlay for depth */
        #scanlines {
            position: fixed;
            inset: 0;
            z-index: -1;
            background: repeating-linear-gradient(
                0deg,
                transparent,
                transparent 2px,
                rgba(0, 0, 0, 0.04) 2px,
                rgba(0, 0, 0, 0.04) 4px
            );
            pointer-events: none;
        }

        /* ── Content Panel ── */
        .main .block-container {
            background: rgba(14, 22, 32, 0.85) !important;
            padding: 2.5rem 3rem !important;
            border-radius: 16px !important;
            box-shadow:
                0 0 0 1px var(--border),
                0 24px 64px rgba(0, 0, 0, 0.6),
                inset 0 1px 0 rgba(255,255,255,0.04) !important;
            backdrop-filter: blur(20px) !important;
            margin-top: 24px !important;
        }

        /* ── Typography ── */
        h1, h2, h3 {
            font-family: 'Sora', sans-serif !important;
            letter-spacing: -0.02em !important;
        }

        h1 {
            font-size: 1.9rem !important;
            font-weight: 700 !important;
            color: var(--text-primary) !important;
            background: linear-gradient(135deg, #e8edf5 30%, var(--accent-cyan)) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            background-clip: text !important;
        }

        h2, h3 {
            font-size: 1rem !important;
            font-weight: 600 !important;
            color: var(--text-muted) !important;
            text-transform: uppercase !important;
            letter-spacing: 0.1em !important;
        }

        p, .stMarkdown, label, span {
            color: var(--text-muted) !important;
            font-size: 0.88rem !important;
        }

        /* ── Sidebar-style left column header ── */
        .stSubheader {
            border-bottom: 1px solid var(--border) !important;
            padding-bottom: 0.5rem !important;
            margin-bottom: 1rem !important;
        }

        /* ── File Uploader ── */
        [data-testid="stFileUploader"] {
            background: var(--bg-elevated) !important;
            border: 1px dashed rgba(45, 123, 244, 0.35) !important;
            border-radius: 12px !important;
            transition: border-color 0.3s ease !important;
        }
        [data-testid="stFileUploader"]:hover {
            border-color: var(--accent-blue) !important;
            box-shadow: 0 0 16px var(--glow) !important;
        }
        [data-testid="stFileUploader"] * {
            color: var(--text-muted) !important;
        }

        /* ── Text Input / Text Area ── */
        .stTextInput input, .stTextArea textarea {
            background: var(--bg-elevated) !important;
            border: 1px solid var(--border) !important;
            border-radius: 10px !important;
            color: var(--text-primary) !important;
            font-family: 'IBM Plex Mono', monospace !important;
            font-size: 0.875rem !important;
            transition: border-color 0.25s ease, box-shadow 0.25s ease !important;
        }
        .stTextInput input:focus, .stTextArea textarea:focus {
            border-color: var(--accent-blue) !important;
            box-shadow: 0 0 0 3px rgba(45, 123, 244, 0.15) !important;
            outline: none !important;
        }
        .stTextInput input::placeholder, .stTextArea textarea::placeholder {
            color: rgba(122, 143, 168, 0.5) !important;
        }

        /* ── Primary Buttons ── */
        .stButton > button {
            background: linear-gradient(135deg, var(--accent-blue), #1a5fd4) !important;
            color: #fff !important;
            border: none !important;
            border-radius: 8px !important;
            font-family: 'Sora', sans-serif !important;
            font-weight: 600 !important;
            font-size: 0.82rem !important;
            letter-spacing: 0.04em !important;
            padding: 0.55rem 1.4rem !important;
            transition: all 0.25s ease !important;
            box-shadow: 0 4px 14px rgba(45, 123, 244, 0.3) !important;
        }
        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 24px rgba(45, 123, 244, 0.45) !important;
            background: linear-gradient(135deg, #3d8bf6, var(--accent-blue)) !important;
        }
        .stButton > button:active {
            transform: translateY(0) !important;
        }

        /* ── Spinner ── */
        .stSpinner > div {
            border-top-color: var(--accent-cyan) !important;
        }

        /* ── Success / Warning / Info boxes ── */
        .stSuccess {
            background: rgba(0, 184, 148, 0.1) !important;
            border: 1px solid rgba(0, 184, 148, 0.3) !important;
            border-radius: 8px !important;
            color: var(--accent-teal) !important;
        }
        .stWarning {
            background: rgba(253, 203, 110, 0.08) !important;
            border: 1px solid rgba(253, 203, 110, 0.25) !important;
            border-radius: 8px !important;
        }
        .stInfo {
            background: rgba(45, 123, 244, 0.08) !important;
            border: 1px solid rgba(45, 123, 244, 0.25) !important;
            border-radius: 8px !important;
        }

        /* ── Divider ── */
        hr {
            border-color: var(--border) !important;
            margin: 2rem 0 !important;
        }

        /* ── Caption / Footer ── */
        .stCaption, [data-testid="stCaptionContainer"] {
            color: rgba(122, 143, 168, 0.5) !important;
            font-family: 'IBM Plex Mono', monospace !important;
            font-size: 0.72rem !important;
            letter-spacing: 0.06em !important;
        }

        /* ── Scrollbar ── */
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: var(--bg-base); }
        ::-webkit-scrollbar-thumb {
            background: rgba(45, 123, 244, 0.35);
            border-radius: 3px;
        }
        ::-webkit-scrollbar-thumb:hover { background: var(--accent-blue); }

        /* ── Column separator ── */
        [data-testid="column"]:first-child {
            border-right: 1px solid var(--border);
            padding-right: 2rem !important;
        }
        [data-testid="column"]:last-child {
            padding-left: 2rem !important;
        }

        /* ── Status badge (mono tag line under title) ── */
        .rag-tagline {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.72rem;
            color: var(--accent-cyan);
            opacity: 0.75;
            letter-spacing: 0.1em;
            margin-top: -0.75rem;
            margin-bottom: 1.5rem;
        }
        </style>

        <!-- Grid + scanline layers -->
        <div id="dynamic-bg"></div>
        <div id="scanlines"></div>

        <script>
        const bg = document.getElementById('dynamic-bg');
        let ticking = false;
        window.addEventListener('mousemove', (e) => {
            if (!ticking) {
                requestAnimationFrame(() => {
                    const x = (window.innerWidth  / 2 - e.pageX) / 80;
                    const y = (window.innerHeight / 2 - e.pageY) / 80;
                    bg.style.transform = `translate(${x}px, ${y}px)`;
                    ticking = false;
                });
                ticking = true;
            }
        });
        </script>
        """,
        unsafe_allow_html=True,
    )


inject_dynamic_background()

# -------------------------------------------------
# Initialize Session State
# -------------------------------------------------
initialize_state()

# -------------------------------------------------
# Main UI Layout
# -------------------------------------------------
st.title("Multi-Model RAG System")
st.markdown(
    '<p class="rag-tagline">RETRIEVAL-AUGMENTED GENERATION · DOCUMENT INTELLIGENCE PLATFORM</p>',
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([1, 2], gap="large")

with left_col:
    st.subheader("📂 Document Ingestion")
    uploaded_files = render_upload_box(
        accepted_types=["pdf", "docx", "csv", "txt", "png", "jpg", "jpeg"]
    )

    if uploaded_files:
        if st.button("⬆ Index Documents"):
            with st.spinner("Processing..."):
                for file in uploaded_files:
                    upload_dir = os.path.join(PROJECT_ROOT, "data", "sample_docs")
                    os.makedirs(upload_dir, exist_ok=True)
                    safe_filename = file.name.replace(" ", "_")
                    save_path = os.path.join(upload_dir, safe_filename)

                    with open(save_path, "wb") as f:
                        f.write(file.getbuffer())

                    raw_text = load_document(save_path)

                    if not safe_filename.lower().endswith(".csv"):
                        clean_text = preprocess_text(raw_text)
                        chunks = chunk_text(clean_text)
                        embeddings = st.session_state.embedding_model.embed_texts(
                            chunks
                        )
                        st.session_state.vector_store.add_documents(chunks, embeddings)

                    add_uploaded_file(safe_filename)
            st.success("Vector index updated successfully.")

    render_uploaded_file_list(st.session_state.uploaded_files)
    if st.session_state.uploaded_files:
        if render_reset_button():
            reset_state()
            st.rerun()

with right_col:
    st.subheader("🔎 Query Interface")
    query = render_query_input()

    if st.button("Run Query"):
        if not st.session_state.uploaded_files:
            st.warning("No documents indexed. Please upload files first.")
        elif not query.strip():
            st.warning("Query cannot be empty.")
        else:
            with st.spinner("Retrieving and generating response..."):
                csv_files = [
                    f
                    for f in st.session_state.uploaded_files
                    if f.lower().endswith(".csv")
                ]
                if csv_files:
                    upload_dir = os.path.join(PROJECT_ROOT, "data", "sample_docs")
                    retrieved_chunks = [
                        load_document(os.path.join(upload_dir, f)) for f in csv_files
                    ]
                else:
                    retriever = Retriever(
                        st.session_state.vector_store,
                        st.session_state.embedding_model,
                        top_k=3,
                    )
                    retrieved_chunks = retriever.retrieve(query)

                generator = ResponseGenerator()
                answer = generator.generate_answer(query, retrieved_chunks)

            add_query_response(query, answer)
            render_response(answer, retrieved_chunks)

st.markdown("---")
st.caption(
    """MULTI-MODEL RAG SYSTEM  ·  DOCUMENT INTELLIGENCE PLATFORM  ·  v1.0
          \n @ Developed by Shaik Mohammad Kaif, Gobburi Prabhavi and Bokka Tranush Reddy
           """
)
