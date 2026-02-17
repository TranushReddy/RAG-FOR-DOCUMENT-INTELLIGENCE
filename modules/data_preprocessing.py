"""
data_preprocessing.py
----------------------
This module performs Natural Language Processing (NLP) based
preprocessing on raw document text and user queries.

It prepares the text for chunking, embedding generation,
and semantic retrieval in the RAG pipeline.
"""


import re


def preprocess_text(text: str) -> str:
    """
    Preprocess document text.

    Steps:
    1. Convert text to lowercase
    2. Remove special characters and punctuation
    3. Remove numbers
    4. Normalize whitespace

    Args:
        text (str): Raw extracted document text

    Returns:
        str: Cleaned and normalized text
    """

    if text is None or len(text.strip()) == 0:
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove special characters and punctuation
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove numbers
    text = re.sub(r"\d+", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_query(query: str) -> str:
    """
    Preprocess user query (lighter preprocessing than document text).

    Args:
        query (str): Raw user query

    Returns:
        str: Cleaned query
    """

    if query is None or len(query.strip()) == 0:
        return ""

    query = query.lower()
    query = re.sub(r"[^a-zA-Z\s]", " ", query)
    query = re.sub(r"\s+", " ", query).strip()

    return query


# -------------------------
# Local Testing
# -------------------------
if __name__ == "__main__":
    sample_text = "  TERMINATION Clause!!! Valid till 2026   "
    print("Original Text:", sample_text)
    print("Processed Text:", preprocess_text(sample_text))

    sample_query = "What is the termination clause?"
    print("Original Query:", sample_query)
    print("Processed Query:", preprocess_query(sample_query))
