# src/knowledge_base.py
# FlowSync Knowledge Base — HYBRID SEARCH (Pinecone dense + BM25 sparse)
#
# Upgraded Week 5 Wednesday from ChromaDB-only similarity search.
# Public contract UNCHANGED from the Chroma version:
#   retrieve(query: str, n_results: int = 3) -> str
# tools.py's lookup_knowledge_base needs no changes because of this.

import os
import sys
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever  # moved here in LangChain 1.0+
from langchain_core.documents import Document

load_dotenv()  # expects PINECONE_API_KEY and OPENAI_API_KEY in .env (never committed)

FAQS_PATH = "./data/faqs.txt"
INDEX_NAME = "flowsync-kb"
EMBEDDING_MODEL = "text-embedding-3-small"  # 1536 dims

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

# ─────────────────────────────────────────────
# BLOCK 1 — CHUNK PARSING (unchanged logic from Chroma version)
# ─────────────────────────────────────────────
def _parse_faq_chunks(faqs_path: str = FAQS_PATH) -> list[str]:
    """
    Reads data/faqs.txt, splits on blank lines, strips section headers
    (BILLING, TECHNICAL, etc.) that prefix some Q: blocks.
    Same parsing rules as the original Chroma ingest_faqs() — carried
    over exactly so chunk content doesn't silently change between versions.
    """
    with open(faqs_path, "r", encoding="utf-8") as f:
        raw = f.read()

    chunks = [c.strip() for c in raw.split("\n\n") if c.strip()]

    qa_chunks = []
    for c in chunks:
        if c.startswith("Q:"):
            qa_chunks.append(c)
        elif "\nQ:" in c:
            # Header + Q&A combined — strip the header line
            qa_only = c[c.index("\nQ:") + 1:]
            qa_chunks.append(qa_only)

    return qa_chunks


# ─────────────────────────────────────────────
# BLOCK 2 — INDEX SETUP + INGEST
# ─────────────────────────────────────────────
def _ensure_index():
    """Create the Pinecone index once, if it doesn't already exist."""
    existing = [i.name for i in pc.list_indexes()]
    if INDEX_NAME not in existing:
        pc.create_index(
            name=INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print(f"[KB] Created Pinecone index '{INDEX_NAME}'")


def ingest_faqs(faqs_path: str = FAQS_PATH) -> int:
    """
    Populates the Pinecone index from faqs.txt.
    Idempotent — same behavior as the Chroma version: if the index
    already has vectors, skip re-ingesting rather than duplicating.
    Returns number of chunks ingested (or already present).
    """
    _ensure_index()
    index = pc.Index(INDEX_NAME)
    stats = index.describe_index_stats()

    if stats.total_vector_count > 0:
        print(f"[KB] Already ingested {stats.total_vector_count} chunks. Skipping.")
        return stats.total_vector_count

    qa_chunks = _parse_faq_chunks(faqs_path)

    if not qa_chunks:
        print("[KB] No FAQ chunks found to ingest.")
        return 0

    docs = [Document(page_content=c, metadata={"chunk_id": i}) for i, c in enumerate(qa_chunks)]
    PineconeVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
        index_name=INDEX_NAME,
    )
    print(f"[KB] Ingested {len(qa_chunks)} FAQ chunks into Pinecone index '{INDEX_NAME}'.")
    return len(qa_chunks)


# ─────────────────────────────────────────────
# BLOCK 3 — HYBRID RETRIEVER (dense + sparse)
# ─────────────────────────────────────────────
def _build_hybrid_retriever(faqs_path: str = FAQS_PATH):
    """
    Dense retriever: Pinecone vector search (semantic similarity).
    Sparse retriever: BM25 (exact keyword / lexical match).
    Combined via EnsembleRetriever, weighted 60% dense / 40% sparse —
    verified this week in rag-learning/hybrid_test.py.
    """
    qa_chunks = _parse_faq_chunks(faqs_path)
    docs = [Document(page_content=c, metadata={"chunk_id": i}) for i, c in enumerate(qa_chunks)]

    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 5

    return EnsembleRetriever(
        retrievers=[dense_retriever, bm25_retriever],
        weights=[0.6, 0.4],
    )


_hybrid_retriever = None  # built lazily so importing this module doesn't require the index to exist yet


def _get_retriever():
    global _hybrid_retriever
    if _hybrid_retriever is None:
        _hybrid_retriever = _build_hybrid_retriever()
    return _hybrid_retriever


# ─────────────────────────────────────────────
# BLOCK 4 — RETRIEVE (public contract — UNCHANGED signature and return type)
# ─────────────────────────────────────────────
def retrieve(query: str, n_results: int = 3) -> str:
    """
    Hybrid search over FAQ knowledge base.
    Returns top n_results chunks as a single formatted string — SAME
    contract as the Chroma version. Called by lookup_knowledge_base
    tool in tools.py; tools.py needs no changes for this upgrade.
    """
    index = pc.Index(INDEX_NAME)
    if index.describe_index_stats().total_vector_count == 0:
        ingest_faqs()

    results = _get_retriever().invoke(query)
    docs = [doc.page_content for doc in results[:n_results]]

    if not docs:
        return "No relevant information found in knowledge base."

    context = "\n\n---\n\n".join(docs)
    return context


# ─────────────────────────────────────────────
# BLOCK 5 — TEST / STANDALONE RUN (same 5 queries as the Chroma version)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    if sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    print("Ingesting FAQs...")
    count = ingest_faqs()
    print(f"Total chunks: {count}\n")

    test_queries = [
        "How do I cancel my subscription?",
        "I am getting a 401 error on the API",
        "What plans do you offer?",
        "Where is my data stored?",
        "My invoice is wrong",
    ]

    for q in test_queries:
        print(f"Query: {q}")
        result = retrieve(q)
        print(f"Top match:\n{result[:200]}...")
        print("─" * 50)
