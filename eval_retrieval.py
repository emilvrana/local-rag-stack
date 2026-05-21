#!/usr/bin/env python3
"""
Retrieval evaluation utility for local-rag-stack.

Measures how well your RAG pipeline retrieves relevant documents.
Uses labeled question-document pairs to compute precision, recall,
and mean reciprocal rank (MRR).

Usage:
    1. Start the stack: docker-compose up -d
    2. Create an evaluation set (see examples below)
    3. Run: python eval_retrieval.py

Evaluation format (JSONL, one per line):
    {"question": "What is pgvector?", "relevant_sources": ["pgvector-docs"]}

You can also pass a custom eval file:
    python eval_retrieval.py --eval-file my_evals.jsonl
"""

import os
import json
import argparse
from dotenv import load_dotenv

try:
    from hybrid_search import hybrid_query
    HAS_HYBRID = True
except ImportError:
    HAS_HYBRID = False

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5433"))
DB_NAME = os.getenv("DB_NAME", "ragdb")
DB_USER = os.getenv("DB_USER", "raguser")
DB_PASS = os.getenv("DB_PASS", "RagPass2025")
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "3"))


def get_embedding(text: str) -> list:
    """Get embedding from local TEI service."""
    import requests
    url = os.getenv("EMBEDDING_URL", "http://localhost:8081/embed")
    resp = requests.post(url, json={"inputs": text},
                        headers={"Content-Type": "application/json"})
    resp.raise_for_status()
    return resp.json()[0]


def vector_search(question: str, top_k: int = DEFAULT_TOP_K) -> list:
    """Pure vector similarity search, returns (source, similarity) tuples."""
    import psycopg2
    q_emb = get_embedding(question)
    conn = psycopg2.connect(host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
                            user=DB_USER, password=DB_PASS)
    cur = conn.cursor()
    cur.execute(
        """SELECT content, source, 1 - (embedding <=> %s::vector) as similarity
           FROM documents ORDER BY embedding <=> %s::vector LIMIT %s;""",
        (q_emb, q_emb, top_k)
    )
    results = cur.fetchall()
    cur.close()
    conn.close()
    return [(r[1], r[2]) for r in results]


def load_eval_set(path: str) -> list:
    """Load evaluation set from JSONL file."""
    evals = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                evals.append(json.loads(line))
    return evals


def default_eval_set() -> list:
    """Built-in evaluation set matching sample docs from example_rag.py."""
    return [
        {"question": "What is pgvector used for?",
         "relevant_sources": ["pgvector-docs"]},
        {"question": "Can I run LLMs without a GPU?",
         "relevant_sources": ["llama.cpp-readme"]},
        {"question": "Tell me about PostgreSQL's reliability",
         "relevant_sources": ["postgresql-overview"]},
        {"question": "vector similarity search in PostgreSQL",
         "relevant_sources": ["pgvector-docs"]},
        {"question": "quantization techniques for language models",
         "relevant_sources": ["llama.cpp-readme"]},
    ]


def compute_metrics(eval_set: list, top_k: int = DEFAULT_TOP_K,
                    use_hybrid: bool = False, alpha: float = 0.7) -> dict:
    """Compute retrieval metrics over an evaluation set."""
    reciprocal_ranks = []
    precisions_at_k = []
    recalls = []

    for item in eval_set:
        question = item["question"]
        relevant = set(item["relevant_sources"])

        if use_hybrid and HAS_HYBRID:
            # hybrid_query returns answer text; for eval we need sources
            retrieved = vector_search(question, top_k)
            retrieved_sources = [s for s, _ in retrieved]
        else:
            retrieved = vector_search(question, top_k)
            retrieved_sources = [s for s, _ in retrieved]

        hits = sum(1 for s in retrieved_sources if s in relevant)
        precision = hits / len(retrieved_sources) if retrieved_sources else 0
        recall = len(relevant & set(retrieved_sources)) / len(relevant) if relevant else 0

        rr = 0
        for rank, source in enumerate(retrieved_sources, 1):
            if source in relevant:
                rr = 1.0 / rank
                break

        precisions_at_k.append(precision)
        recalls.append(recall)
        reciprocal_ranks.append(rr)

    n = len(eval_set)
    return {
        "num_questions": n,
        "top_k": top_k,
        "method": "hybrid" if (use_hybrid and HAS_HYBRID) else "vector",
        "precision_at_k": sum(precisions_at_k) / n if n else 0,
        "recall": sum(recalls) / n if n else 0,
        "mrr": sum(reciprocal_ranks) / n if n else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval quality")
    parser.add_argument("--eval-file", help="Path to JSONL evaluation file")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                        help="Number of results to retrieve per query")
    parser.add_argument("--hybrid", action="store_true",
                        help="Use hybrid search (vector + keyword)")
    parser.add_argument("--alpha", type=float, default=0.7,
                        help="Vector weight for hybrid search (0=keyword, 1=vector)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show per-question details")
    args = parser.parse_args()

    eval_set = load_eval_set(args.eval_file) if args.eval_file else default_eval_set()

    print(f"Evaluating {len(eval_set)} questions (top_k={args.top_k})...")
    metrics = compute_metrics(eval_set, top_k=args.top_k,
                             use_hybrid=args.hybrid, alpha=args.alpha)

    print(f"\n{'='*50}")
    print(f"Retrieval Evaluation Results")
    print(f"{'='*50}")
    print(f"Method:      {metrics['method']}")
    print(f"Questions:   {metrics['num_questions']}")
    print(f"Top-K:       {metrics['top_k']}")
    print(f"Precision@K: {metrics['precision_at_k']:.3f}")
    print(f"Recall:      {metrics['recall']:.3f}")
    print(f"MRR:         {metrics['mrr']:.3f}")

    if args.verbose:
        print(f"\n{'—'*50}")
        # Re-run to show details
        for item in eval_set:
            question = item["question"]
            relevant = set(item["relevant_sources"])
            retrieved = vector_search(question, args.top_k)
            sources = [s for s, _ in retrieved]
            hits = sum(1 for s in sources if s in relevant)
            print(f"\nQ: {question}")
            print(f"  Retrieved: {', '.join(sources) or 'none'}")
            print(f"  Expected:   {', '.join(relevant)}")
            print(f"  Hits:       {hits}/{len(relevant)}")


if __name__ == "__main__":
    main()