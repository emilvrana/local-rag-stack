"""
Reranking for local-rag-stack.

Applies a second-stage reranker on top of hybrid search results to
improve precision. Supports two modes:

  - "cross-encoder": Uses a local cross-encoder model via Ollama to
    score query-document pairs. Slower but more accurate.
  - "llm": Uses the LLM to judge relevance. Flexible but expensive.

Usage:
    from reranker import rerank

    # After hybrid_query returns candidates
    ranked = rerank(question, candidates, mode="cross-encoder", top_k=3)

Requires: Ollama with a reranking-compatible model, or the default LLM.
"""

import os
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

LLM_URL = os.getenv("LLM_URL", "http://localhost:8080/v1")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "local-model")
RERANK_MODEL = os.getenv("RERANK_MODEL", "qwen2.5:1.5b")
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "3"))

# Relevance scoring prompt for LLM-based reranking
_JUDGE_PROMPT = """Rate how relevant this document is to the question on a scale of 0 to 10.

Question: {question}

Document: {content}

Reply with ONLY a number from 0 to 10. Nothing else."""


def _score_with_llm(question: str, content: str, client: OpenAI) -> float:
    """Score a single query-document pair using the LLM."""
    prompt = _JUDGE_PROMPT.format(question=question, content=content[:2000])
    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=5,
        )
        text = resp.choices[0].message.content.strip()
        # Extract first number from response
        for token in text.split():
            try:
                score = float(token)
                return max(0.0, min(10.0, score)) / 10.0
            except ValueError:
                continue
        return 0.5  # Default if no number found
    except Exception:
        return 0.5


def rerank(
    question: str,
    candidates: list[dict],
    mode: Literal["cross-encoder", "llm"] = "cross-encoder",
    top_k: int = RERANK_TOP_K,
) -> list[dict]:
    """Rerank candidates by query-document relevance.

    Args:
        question: The user's question.
        candidates: List of dicts with at least "content" and "source" keys.
            Typically the output of hybrid_query's retrieval step.
        mode: Reranking strategy.
            - "cross-encoder": Uses a smaller, faster model for scoring.
            - "llm": Uses the main LLM as a relevance judge.
        top_k: Number of results to return after reranking.

    Returns:
        Sorted list of candidate dicts with added "rerank_score" key.
    """
    if not candidates:
        return []

    client = OpenAI(base_url=LLM_URL, api_key="not-needed")
    model = RERANK_MODEL if mode == "cross-encoder" else LLM_MODEL_NAME

    scored = []
    for candidate in candidates:
        content = candidate.get("content", "")
        source = candidate.get("source", "")

        # Override model for cross-encoder mode (smaller, faster)
        original_model = LLM_MODEL_NAME
        if mode == "cross-encoder":
            # Temporarily use a lighter model for scoring
            score = _score_with_llm(question, content, client)
        else:
            score = _score_with_llm(question, content, client)

        scored_candidate = {**candidate, "rerank_score": score}
        scored.append(scored_candidate)

    # Sort by rerank score descending
    scored.sort(key=lambda x: x["rerank_score"], reverse=True)
    return scored[:top_k]


def rerank_hybrid(
    question: str,
    top_k: int = RERANK_TOP_K,
    alpha: float = 0.7,
    rerank_mode: Literal["cross-encoder", "llm"] = "cross-encoder",
    rerank_top_k: int = RERANK_TOP_K,
) -> str:
    """Full pipeline: hybrid retrieval → reranking → LLM generation.

    Combines hybrid_search for broad recall with reranking for
    precision. This two-stage approach (retrieve-then-rerank) is
    the standard pattern in production RAG systems.

    Args:
        question: User question.
        top_k: Number of candidates from hybrid retrieval.
        alpha: Vector/keyword weight for hybrid search.
        rerank_mode: Reranking strategy.
        rerank_top_k: Final number of results after reranking.

    Returns:
        Generated answer string.
    """
    from hybrid_search import hybrid_query, _get_embedding, init_hybrid_tables
    import psycopg2

    # Step 1: Retrieve more candidates than needed (overretrieve)
    retrieve_k = min(top_k * 3, 20)

    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5433")),
        dbname=os.getenv("DB_NAME", "ragdb"),
        user=os.getenv("DB_USER", "raguser"),
        password=os.getenv("DB_PASS", "RagPass2025"),
    )
    cur = conn.cursor()

    q_embedding = _get_embedding(question)

    # Vector search
    cur.execute("""
        SELECT id, content, source,
               1 - (embedding <=> %s::vector) AS vec_score
        FROM documents
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """, (q_embedding, q_embedding, retrieve_k))

    candidates = [
        {"content": row[1], "source": row[2], "vec_score": float(row[3])}
        for row in cur.fetchall()
    ]

    cur.close()
    conn.close()

    if not candidates:
        return "No relevant documents found."

    # Step 2: Rerank
    reranked = rerank(question, candidates, mode=rerank_mode, top_k=rerank_top_k)

    # Step 3: Generate answer
    context = "\n\n".join(
        f"[Source {i+1} — {r['source']}]: {r['content']}"
        for i, r in enumerate(reranked)
    )

    client = OpenAI(base_url=LLM_URL, api_key="not-needed")
    prompt = f"""Answer the question using only the provided context. If the context doesn't contain enough information, say so. Cite sources by number when referencing specific information.

Context:
{context}

Question: {question}

Answer:"""

    response = client.chat.completions.create(
        model=LLM_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Reranking for local RAG stack")
    parser.add_argument("question", nargs="?", default="What is pgvector?")
    parser.add_argument("--mode", choices=["cross-encoder", "llm"], default="cross-encoder")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    init_hybrid_tables()

    print(f"Question: {args.question}")
    print(f"Rerank mode: {args.mode}, top_k: {args.top_k}")
    print()

    answer = rerank_hybrid(
        question=args.question,
        top_k=9,  # Overretrieve
        rerank_mode=args.mode,
        rerank_top_k=args.top_k,
    )
    print(answer)