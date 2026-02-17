import streamlit as st
from typing import List


# -------------------------------------------------
# Upload Box
# -------------------------------------------------
def render_upload_box(accepted_types: List[str]):
    st.markdown(
        '<div class="upload-box">📂 Drag and drop documents here or click to upload</div>',
        unsafe_allow_html=True,
    )
    uploaded_files = st.file_uploader(
        "Upload documents",
        type=accepted_types,
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    return uploaded_files


# -------------------------------------------------
# Uploaded File List
# -------------------------------------------------
def render_uploaded_file_list(files: List[str]):
    if files:
        st.markdown("### 📄 Uploaded Documents")
        for file in files:
            st.markdown(f"- {file}", unsafe_allow_html=True)


# -------------------------------------------------
# Query Input
# -------------------------------------------------
def render_query_input():
    return st.text_input(
        "Enter your question",
        placeholder="e.g. What does the termination clause say?",
    )


# -------------------------------------------------
# Response Display (CLEAN RAG OUTPUT)
# -------------------------------------------------
def render_response(answer: str, retrieved_chunks: List[str]):
    st.markdown("### ✅ Answer")
    st.markdown(
        f"""
        <div class="response-box">
        {answer}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Optional source transparency
    with st.expander("📄 View source excerpts"):
        for i, chunk in enumerate(retrieved_chunks, 1):
            st.markdown(f"**Source {i}:** {chunk}")


# -------------------------------------------------
# Reset Button
# -------------------------------------------------
def render_reset_button():
    return st.button("🔄 Reset System")
