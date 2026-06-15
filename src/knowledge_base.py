# src/knowledge_base.py
# FlowSync Knowledge Base — ChromaDB RAG
# Ingests data/faqs.txt and exposes retrieve() for semantic search

import os
import chromadb
from chromadb.utils import embedding_functions

# ─────────────────────────────────────────────
# BLOCK 1 — CLIENT + COLLECTION SETUP
# ─────────────────────────────────────────────
# PersistentClient saves to disk — survives restarts
# Path is relative to project root
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "flowsync_faqs"

def get_collection():
    """
    Returns ChromaDB collection.
    Creates it if it doesn't exist.
    Uses cosine similarity — better than L2 for text.
    """
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    
    # Default embedding function uses sentence-transformers locally
    # No API key needed — runs on CPU
    ef = embedding_functions.DefaultEmbeddingFunction()
    
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
        embedding_function=ef
    )
    return collection

# ─────────────────────────────────────────────
# BLOCK 2 — INGEST FAQs
# ─────────────────────────────────────────────
def ingest_faqs(faqs_path: str = "./data/faqs.txt") -> int:
    """
    Loads data/faqs.txt, splits into Q&A chunks, upserts into ChromaDB.
    Safe to run multiple times — upsert won't duplicate.
    Returns number of chunks ingested.
    """
    collection = get_collection()
    
    # Check if already ingested
    if collection.count() > 0:
        print(f"[KB] Already ingested {collection.count()} chunks. Skipping.")
        return collection.count()
    
    # Read the FAQ file
    with open(faqs_path, "r", encoding="utf-8") as f:
        raw = f.read()
    
    # Split by blank lines into chunks
    # Each chunk = one Q&A pair
    chunks = [c.strip() for c in raw.split("\n\n") if c.strip()]
    
    # Filter out section headers (BILLING, TECHNICAL, etc.)
    qa_chunks = []
    for c in chunks:
        if c.startswith("Q:"):
            qa_chunks.append(c)
        elif "\nQ:" in c:
            # Header + Q&A combined — strip the header line
            qa_only = c[c.index("\nQ:") + 1:]
            qa_chunks.append(qa_only)

    if qa_chunks:
        collection.upsert(
            documents=qa_chunks,
            ids=[f"faq_{i}" for i in range(len(qa_chunks))]
        )
        print(f"[KB] Ingested {len(qa_chunks)} FAQ chunks into ChromaDB.")
    else:
        print("[KB] No FAQ chunks found to ingest.")
    return len(qa_chunks)

# ─────────────────────────────────────────────
# BLOCK 3 — RETRIEVE
# ─────────────────────────────────────────────
def retrieve(query: str, n_results: int = 3) -> str:
    """
    Semantic search over FAQ knowledge base.
    Returns top n_results chunks as a single string.
    Called by lookup_knowledge_base tool in tools.py.
    """
    collection = get_collection()
    
    if collection.count() == 0:
        ingest_faqs()
    
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    
    # Extract the matched documents
    docs = results["documents"][0]  # list of matching chunks
    
    if not docs:
        return "No relevant information found in knowledge base."
    
    # Format as clean context string
    context = "\n\n---\n\n".join(docs)
    return context

# ─────────────────────────────────────────────
# BLOCK 4 — TEST / STANDALONE RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("Ingesting FAQs...")
    count = ingest_faqs()
    print(f"Total chunks: {count}\n")
    
    # Test 5 queries
    test_queries = [
        "How do I cancel my subscription?",
        "I am getting a 401 error on the API",
        "What plans do you offer?",
        "Where is my data stored?",
        "My invoice is wrong"
    ]
    
    for q in test_queries:
        print(f"Query: {q}")
        result = retrieve(q)
        print(f"Top match:\n{result[:200]}...")
        print("─" * 50)