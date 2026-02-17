"""
retrieval.py
-------------
This module handles semantic retrieval of document chunks
based on similarity between query embedding and stored embeddings.
"""

from typing import List
from modules.embedding import EmbeddingModel
from modules.vector_store import VectorStore
from modules.data_preprocessing import preprocess_query


class Retriever:
    """
    Retriever class to fetch relevant document chunks
    using semantic similarity.
    """

    def __init__(
        self, vector_store: VectorStore, embedding_model: EmbeddingModel, top_k: int = 3
    ):
        """
        Initialize the retriever.

        Args:
            vector_store (VectorStore): Vector database
            embedding_model (EmbeddingModel): Embedding model instance
            top_k (int): Number of chunks to retrieve
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.top_k = top_k

    def retrieve(self, query: str) -> List[str]:
        """
        Retrieve top-k relevant chunks for a user query.

        Args:
            query (str): Raw user query

        Returns:
            List[str]: Relevant document chunks
        """

        # Preprocess the query
        clean_query = preprocess_query(query)

        # Generate query embedding
        query_embedding = self.embedding_model.embed_query(clean_query)

        # Perform similarity search
        results = self.vector_store.similarity_search(
            query_embedding=query_embedding, top_k=self.top_k
        )

        return results


# -------------------------
# Local Testing
# -------------------------
if __name__ == "__main__":
    from modules.chunking import chunk_text

    # Sample document
    document_text = (
        "This agreement includes payment terms and termination clauses. "
        "The termination clause explains how the contract can be ended. "
        "Confidentiality is mandatory for both parties."
    )

    # Chunk document
    chunks = chunk_text(document_text, chunk_size=15, chunk_overlap=5)

    # Initialize embedding model and vector store
    embedding_model = EmbeddingModel()
    vector_store = VectorStore()

    # Generate embeddings and store them
    embeddings = embedding_model.embed_texts(chunks)
    vector_store.add_documents(chunks, embeddings)

    # Create retriever
    retriever = Retriever(vector_store, embedding_model, top_k=2)

    # Query
    query = "Explain the termination clause"
    retrieved_chunks = retriever.retrieve(query)

    print("Retrieved Chunks:\n")
    for chunk in retrieved_chunks:
        print("-", chunk)
