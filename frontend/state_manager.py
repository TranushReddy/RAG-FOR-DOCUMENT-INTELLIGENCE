"""
state_manager.py
----------------
This module manages Streamlit session state for the
Multi-Model RAG System for Document Intelligence.

It initializes and stores shared objects such as:
- Vector Store
- Embedding Model
- Uploaded documents
- Query history
"""

import streamlit as st
from typing import List

from modules.embedding import EmbeddingModel
from modules.vector_store import VectorStore


def initialize_state():
    """
    Initialize required session state variables.
    This function should be called once at the start
    of the Streamlit application.
    """

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = VectorStore()

    if "embedding_model" not in st.session_state:
        st.session_state.embedding_model = EmbeddingModel()

    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files: List[str] = []

    if "query_history" not in st.session_state:
        st.session_state.query_history: List[str] = []

    if "response_history" not in st.session_state:
        st.session_state.response_history: List[str] = []


def add_uploaded_file(file_name: str):
    """
    Track uploaded document names.
    """

    if file_name not in st.session_state.uploaded_files:
        st.session_state.uploaded_files.append(file_name)


def add_query_response(query: str, response: str):
    """
    Store query and corresponding response.
    """

    st.session_state.query_history.append(query)
    st.session_state.response_history.append(response)


def reset_state():
    """
    Reset the entire session state.
    Useful for 'Clear / Reset' functionality.
    """

    st.session_state.vector_store = VectorStore()
    st.session_state.uploaded_files = []
    st.session_state.query_history = []
    st.session_state.response_history = []


def has_documents() -> bool:
    """
    Check if documents are already uploaded.
    """

    return (
        "vector_store" in st.session_state
        and st.session_state.vector_store.embeddings is not None
    )
