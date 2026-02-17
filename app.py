"""
app.py
-------
Main entry point for the Multi-Model RAG System for Document Intelligence.
This script integrates all modules and executes the complete RAG pipeline.
"""

import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.data_input import load_document
from modules.data_preprocessing import preprocess_text
from modules.chunking import chunk_text
from modules.embedding import EmbeddingModel
from modules.vector_store import VectorStore
from modules.retrieval import Retriever
from modules.generation import ResponseGenerator


def run_rag_pipeline(document_path: str, query: str):
    """
    Runs the complete RAG pipeline for a given document and query.
    """

    print("\n--- Multi-Model RAG System for Document Intelligence ---\n")

    # Step 1: Load document
    print("[1] Loading document...")
    raw_text = load_document(document_path)

    # Step 2: Preprocess document text
    print("[2] Preprocessing document...")
    clean_text = preprocess_text(raw_text)

    # Step 3: Chunk document
    print("[3] Chunking document...")
    chunks = chunk_text(clean_text)

    if not chunks:
        print("No content available after chunking.")
        return

    # Step 4: Generate embeddings
    print("[4] Generating embeddings...")
    embedding_model = EmbeddingModel()
    chunk_embeddings = embedding_model.embed_texts(chunks)

    # Step 5: Store embeddings
    print("[5] Storing embeddings in vector store...")
    vector_store = VectorStore()
    vector_store.add_documents(chunks, chunk_embeddings)

    # Step 6: Retrieve relevant chunks
    print("[6] Retrieving relevant document chunks...")
    retriever = Retriever(vector_store, embedding_model, top_k=3)
    relevant_chunks = retriever.retrieve(query)

    if not relevant_chunks:
        print("No relevant chunks found.")
        return

    # Step 7: Generate answer
    print("[7] Generating response...\n")
    generator = ResponseGenerator()
    answer = generator.generate_answer(query, relevant_chunks)

    # Output result
    print("🔍 Query:")
    print(query)

    print("\n📄 Retrieved Context:")
    for i, chunk in enumerate(relevant_chunks, 1):
        print(f"\nChunk {i}: {chunk}")

    print("\n✅ Final Answer:")
    print(answer)


# -------------------------
# Local Execution
# -------------------------
if __name__ == "__main__":

    # Example document path (update as needed)
    document_path = "data/sample_docs/sample.txt"

    # Example query
    query = "What does the termination clause say?"

    run_rag_pipeline(document_path, query)
