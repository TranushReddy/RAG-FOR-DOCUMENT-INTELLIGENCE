"""
embedding.py
-------------
This module generates semantic vector embeddings for document chunks
and user queries using a transformer-based embedding model.
"""

from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingModel:
    """
    Wrapper class for transformer-based embedding generation.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding model.

        Args:
            model_name (str): Name of the sentence-transformer model
        """
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts (List[str]): List of document chunks

        Returns:
            np.ndarray: Array of embeddings
        """
        if not texts:
            return np.array([])

        embeddings = self.model.encode(
            texts, convert_to_numpy=True, show_progress_bar=False
        )
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single user query.

        Args:
            query (str): User query

        Returns:
            np.ndarray: Query embedding
        """
        embedding = self.model.encode(
            query, convert_to_numpy=True, show_progress_bar=False
        )
        return embedding


# -------------------------
# Local Testing
# -------------------------
if __name__ == "__main__":
    sample_chunks = [
        "This document explains the termination clause.",
        "Payment terms are discussed in this section.",
        "Confidentiality obligations are described here.",
    ]

    model = EmbeddingModel()
    chunk_embeddings = model.embed_texts(sample_chunks)

    print("Number of chunks:", len(sample_chunks))
    print("Embedding shape:", chunk_embeddings.shape)

    query = "What is the termination clause?"
    query_embedding = model.embed_query(query)
    print("Query embedding shape:", query_embedding.shape)
