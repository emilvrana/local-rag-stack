"""
Hybrid search for local-rag-stack.

Combines vector similarity with keyword (BM25-like) matching using
PostgreSQL full-text search. Returns results ranked by a weighted
blend of both signals, which significantly improves retrieval quality
for queries with specific terms, names, or IDs.

Usage:
    from hybrid_search import hybrid_query

    answer = hybrid_query("What is pgvector?", alpha=0.7)
    # alpha: weight for vector score (0 = keyword only, 1 = vector only)

Requires: pg_trgm extension (auto-created by init_hybrid_tables)
"""

import os
import re
from dotenv import load_dotenv
import psycopg2
from openai import OpenAI

load_dotenv()

EMBEDDING_URL = os.getenv("EMBEDDING_URL", "http://localhost:8081/embed")
LLM_URL = os.getenv("LLM_URL", "http://localhost:8080/v1")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5433"))
DB_NAME = os.getenv("DB_NAME", "ragdb")
DB_USER = os.getenv("DB_USER", "raguser")
DB_PASS = os.getenv("DB_PASS", "RagPass2025")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "local-model")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "3"))
DEFAULT_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.7"))


def _get_embedding(text: str) -> list:
    import requests
    response = requests.post(
        EMBEDDING_URL,
        json={"inputs": text},
        headers={"Content-Type": "application/json"}
    )
    response.raise_for_status()
    return response.json()[0]


def init_hybrid_tables():
    """Enable full-text search on the documents table.

    Adds a tsvector column and GIN index for keyword matching.
    Safe to run multiple times — uses IF NOT EXISTS.
    """
    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
        user=DB_USER, password=DB_PASS
    )
    cur = conn.cursor()

    # pg_trgm for trigram similarity (fuzzy matching)
    cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")

    # Full-text search column + index
    cur.execute("""
        ALTER TABLE documents ADD COLUMN IF NOT EXISTS
        search_vector tsvector
        GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS documents_search_idx
        ON documents USING GIN (search_vector);
    """)

    # Trigram index for fuzzy keyword matching
    cur.execute("""
        CREATE INDEX IF NOT EXISTS documents_content_trgm_idx
        ON documents USING GIN (content gin_trgm_ops);
    """)

    conn.commit()
    cur.close()
    conn.close()


def hybrid_query(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    alpha: float = DEFAULT_ALPHA,
) -> str:
    """Retrieve relevant chunks using hybrid vector + keyword search.

    Uses Reciprocal Rank Fusion (RRF) to merge results from both
    retrieval methods. This handles cases where:
    - Vector search misses exact term matches (names, IDs, codes)
    - Keyword search misses semantic matches (paraphrases, synonyms)

    Args:
        question: User question.
        top_k: Number of results to return.
        alpha: Weight for vector similarity (0.0 = keyword only,
               1.0 = vector only). Default 0.7 favors semantic matching
               while still catching exact terms.

    Returns:
        Generated answer string.
    """
    q_embedding = _get_embedding(question)

    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
        user=DB_USER, password=DB_PASS
    )
    cur = conn.cursor()

    # Vector similarity scores (cosine distance → similarity)
    cur.execute("""
        SELECT id, content, source,
               1 - (embedding <=> %s::vector) AS vec_score
        FROM documents
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """, (q_embedding, q_embedding, top_k * 3))

    vec_results = {row[0]: {
        "content": row[1],
        "source": row[2],
        "vec_score": float(row[3]),
    } for row in cur.fetchall()}

    # Keyword search using tsquery + trigram similarity
    # Build tsquery from question terms
    terms = re.findall(r'\w+', question.lower())
    ts_query = " & ".join(terms) if terms else ""

    if ts_query:
        cur.execute("""
            SELECT id, content, source,
                   ts_rank_cd(search_vector, query) AS kw_score
            FROM documents, plainto_tsquery('english', %s) query
            WHERE search_vector @@ query
            ORDER BY kw_score DESC
            LIMIT %s
        """, (question, top_k * 3))

        kw_results = {row[0]: {
            "content": row[1],
            "source": row[2],
            "kw_score": float(row[3]),
        } for row in cur.fetchall()}

        # Fallback: trigram similarity for terms not in dictionary
        if len(kw_results) < top_k:
            cur.execute("""
                SELECT id, content, source,
                       similarity(content, %s) AS trgm_score
                FROM documents
                WHERE similarity(content, %s) > 0.1
                ORDER BY trgm_score DESC
                LIMIT %s
            """, (question, question, top_k))

            for row in cur.fetchall():
                if row[0] not in kw_results:
                    kw_results[row[0]] = {
                        "content": row[1],
                        "source": row[2],
                        "kw_score": float(row[3]) * 10,  # scale up
                    }
    else:
        kw_results = {}

    cur.close()
    conn.close()

    # Reciprocal Rank Fusion (RRF) to combine both result sets
    # Pre-compute ranks from sorted order (O(n log n) instead of O(n²))
    k = 60  # RRF constant

    vec_ranks = {doc_id: rank for rank, doc_id in enumerate(
        sorted(vec_results, key=lambda d: vec_results[d].get("vec_score", 0), reverse=True), 1
    )}
    kw_ranks = {doc_id: rank for rank, doc_id in enumerate(
        sorted(kw_results, key=lambda d: kw_results[d].get("kw_score", 0), reverse=True), 1
    )}

    all_ids = set(vec_ranks) | set(kw_ranks)
    scored = []
    for doc_id in all_ids:
        rrf_score = 0.0
        if doc_id in vec_ranks:
            rrf_score += alpha / (k + vec_ranks[doc_id])
        if doc_id in kw_ranks:
            rrf_score += (1 - alpha) / (k + kw_ranks[doc_id])

        doc_data = vec_results.get(doc_id) or kw_results.get(doc_id)
        scored.append((rrf_score, doc_data["content"], doc_data["source"]))

    # Sort by combined score, take top_k
    scored.sort(key=lambda x: x[0], reverse=True)
    top_results = scored[:top_k]

    if not top_results:
        return "No relevant documents found."

    # Build context and query LLM — include source numbers for attribution
    context = "\n\n".join([f"[Source {i+1} — {r[2]}]: {r[1]}" for i, r in enumerate(top_results)])

    client = OpenAI(base_url=LLM_URL, api_key="not-needed")
    prompt = f"""Answer the question using only the provided context. If the context doesn't contain enough information, say so. Cite sources by number when referencing specific information.

Context:
{context}

Question: {question}

Answer:"""

    response = client.chat.completions.create(
        model=LLM_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=LLM_TEMPERATURE
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    from example_rag import init_db, add_document

    # Setup: ensure tables and hybrid indexes exist
    init_db()
    init_hybrid_tables()

    # Add sample docs if needed
    sample_docs = [
        ("PostgreSQL is a powerful, open-source object-relational database system. "
         "It has more than 30 years of active development and a proven architecture that "
         "has earned it a strong reputation for reliability, feature robustness, and performance.",
         "postgresql-overview"),
        ("pgvector is an open-source vector similarity search for PostgreSQL. "
         "It allows you to store, query, and index vectors alongside your regular data. "
         "Supports L2 distance, inner product, and cosine distance operations.",
         "pgvector-docs"),
        ("llama.cpp is a port of Facebook's LLaMA model in C/C++. "
         "It enables running LLMs on modest hardware, including CPUs, with quantization "
         "techniques that reduce memory usage while maintaining reasonable quality.",
         "llama.cpp-readme"),
    ]

    for content, source in sample_docs:
        add_document(content, source)

    # Compare: semantic vs hybrid search
    print("=== Semantic-only (alpha=1.0) ===")
    print(hybrid_query("What is pgvector?", alpha=1.0))
    print("\n=== Keyword-only (alpha=0.0) ===")
    print(hybrid_query("What is pgvector?", alpha=0.0))
    print("\n=== Hybrid (alpha=0.7, default) ===")
    print(hybrid_query("What is pgvector?"))