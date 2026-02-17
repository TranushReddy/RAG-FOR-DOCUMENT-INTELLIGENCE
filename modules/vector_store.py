"""
vector_store.py
----------------
This module implements a simple in-memory vector store
to store embeddings and perform similarity search.
"""

from typing import List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class VectorStore:
    """
    A simple vector database for storing text chunks and their embeddings.
    """

    def __init__(self):
        self.text_chunks: List[str] = []
        self.embeddings: np.ndarray = None

    def add_documents(self, chunks: List[str], embeddings: np.ndarray):
        """
        Add document chunks and their embeddings to the vector store.

        Args:
            chunks (List[str]): List of document text chunks
            embeddings (np.ndarray): Corresponding embeddings
        """

        if len(chunks) == 0:
            return

        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack((self.embeddings, embeddings))

        self.text_chunks.extend(chunks)

    def similarity_search(
        self, query_embedding: np.ndarray, top_k: int = 3
    ) -> List[str]:
        """
        Retrieve Top-K most similar text chunks using cosine similarity.

        Args:
            query_embedding (np.ndarray): Embedding of the user query
            top_k (int): Number of relevant chunks to retrieve

        Returns:
            List[str]: Top-K relevant document chunks
        """

        if self.embeddings is None or len(self.text_chunks) == 0:
            return []

        # Reshape query embedding
        query_embedding = query_embedding.reshape(1, -1)

        # Compute cosine similarity
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]

        # Get indices of top-k similarities
        top_indices = similarities.argsort()[-top_k:][::-1]

        return [self.text_chunks[i] for i in top_indices]


# -------------------------
# Local Testing
# -------------------------
if __name__ == "__main__":
    from modules.embedding import EmbeddingModel

    chunks = [
        "This document explains the termination clause.",
        "Payment terms are discussed in this section.",
        "Confidentiality obligations are described here.",
    ]

    embedding_model = EmbeddingModel()
    embeddings = embedding_model.embed_texts(chunks)

    vector_store = VectorStore()
    vector_store.add_documents(chunks, embeddings)

    query = "What is the termination clause?"
    query_embedding = embedding_model.embed_query(query)

    results = vector_store.similarity_search(query_embedding, top_k=2)

    print("Top Relevant Chunks:\n")
    for r in results:
        print("-", r)
