"""
chunking.py
------------
This module splits preprocessed document text into smaller,
overlapping chunks to preserve semantic context for embedding
generation and retrieval in the RAG pipeline.
"""

from typing import List


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> List[str]:
    """
    Splits text into overlapping chunks.

    Args:
        text (str): Preprocessed document text
        chunk_size (int): Number of words per chunk
        chunk_overlap (int): Number of overlapping words between chunks

    Returns:
        List[str]: List of text chunks
    """

    if not text:
        return []

    words = text.split()
    chunks = []

    start = 0
    text_length = len(words)

    while start < text_length:
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk = " ".join(chunk_words)
        chunks.append(chunk)

        # Move start forward with overlap
        start = end - chunk_overlap

        if start < 0:
            start = 0

    return chunks


# -------------------------
# Local Testing
# -------------------------
if __name__ == "__main__":
    sample_text = (
        "This agreement is made between the company and the client. "
        "The agreement includes terms related to payment, termination, "
        "confidentiality, and liability. The termination clause defines "
        "the conditions under which the agreement can be ended."
    )

    chunks = chunk_text(sample_text, chunk_size=10, chunk_overlap=3)

    print("Generated Chunks:\n")
    for i, c in enumerate(chunks, 1):
        print(f"Chunk {i}: {c}\n")
