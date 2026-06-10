# Local RAG Stack with Docker

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Contributing](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

This repository provides a simple `docker-compose.yml` to run the core components of a Retrieval-Augmented Generation (RAG) pipeline on your own server or local machine, even without a GPU.

> **Update (April 2026):** Now using [Ollama](https://ollama.com/) for LLM serving — automatic model management, broader model support, and simpler configuration. The previous llama.cpp setup remains available as an alternative.

## Components

This stack includes:

1.  **`postgres`**: A PostgreSQL database with the `pgvector` extension for storing vector embeddings.
2.  **`embeddings`**: A Hugging Face text embeddings model (`bge-base-en-v1.5`) served via the Text Embeddings Inference container.
3.  **`llm`**: Local LLM via [Ollama](https://ollama.com/). Pulls and caches models automatically. Supports Llama, Qwen, Mistral, and many others.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/) are installed.
- ~2 GB disk space for the embeddings model; 4–8 GB for the LLM (varies by model).

## How to Use

1.  **Clone this repository:**
    ```bash
    git clone https://github.com/emilvrana/local-rag-stack.git
    cd local-rag-stack
    ```

2.  **Configure your model:**
    - Copy the example environment file: `cp .env.example .env`
    - Edit `.env` and set `OLLAMA_MODEL` to your preferred model:
        - `qwen2.5:7b` — fast, good for most tasks (default)
        - `qwen2.5:14b` — better quality, slower
        - `llama3.2` — compact, good for constrained environments
        - See [ollama.com/library](https://ollama.com/library) for all options

3.  **Start the services:**
    ```bash
    docker-compose up -d
    ```
    The model will download automatically on first startup (this may take a few minutes).

4.  **Verify:**
    ```bash
    # Quick health check (all services)
    ./healthcheck.sh
    
    # Or wait for services to become ready (useful in CI or scripts)
    python wait_for_services.py
    
    # Or manually:
    curl http://localhost:8080/api/tags    # LLM
    curl http://localhost:8081/embed -X POST \
      -H "Content-Type: application/json" \
      -d '{"inputs": "Hello world"}'      # Embeddings
    ```

You should have:
- PostgreSQL with pgvector on `localhost:5433`
- Embeddings API at `http://localhost:8081`
- LLM API (OpenAI-compatible) at `http://localhost:8080`

## Python Example

A complete working example is provided in [`example_rag.py`](example_rag.py). It demonstrates:
- Database initialization with pgvector
- Document chunking with sliding windows
- Embedding via the local TEI service
- Similarity search using cosine distance
- Answer generation via the local LLM

```bash
# Install dependencies
pip install -r requirements.txt

# Configure
# Edit .env with your settings (if you changed defaults in docker-compose.yml)

# Run the example
python example_rag.py
```

## Model Recommendations

| Model | Use Case | Notes |
|-------|----------|-------|
| `qwen2.5:7b` | General purpose (default) | Good balance of speed and quality |
| `qwen2.5:14b` | Complex reasoning | Noticeably slower, better answers |
| `llama3.2` | Constrained environments | Fastest, sufficient for simple tasks |
| `mistral:7b` | Coding tasks | Good code understanding |

## Semantic Chunking

The default chunking now uses **sentence-aware semantic chunking** instead of naive sliding windows. This keeps sentences intact, producing more coherent chunks that work better for RAG retrieval.

```python
from semantic_chunker import semantic_chunk

# Sentence-aware (default): groups sentences into chunks
chunks = semantic_chunk(text, chunk_size=500, strategy="sentence")

# Paragraph-aware: splits at paragraph boundaries, subdivides oversized paragraphs
chunks = semantic_chunk(text, chunk_size=500, strategy="paragraph")
```

The `example_rag.py` uses semantic chunking automatically. If you need the old naive chunker, just remove the `semantic_chunker` import — it falls back gracefully.

## Hybrid Search

Vector similarity alone misses exact term matches. The new `hybrid_search.py` module combines vector search with PostgreSQL full-text search and trigram matching, using Reciprocal Rank Fusion (RRF) to merge results.

```python
from hybrid_search import hybrid_query, init_hybrid_tables

# Run once to add full-text indexes
init_hybrid_tables()

# Alpha controls vector vs keyword weight (0.0–1.0, default 0.7)
answer = hybrid_query("What is pgvector?", alpha=0.7)
```

Why it matters: queries with specific names, error codes, or IDs often fail on pure vector search. Keyword-only search misses paraphrases. Hybrid catches both.

## Reranking

Hybrid search retrieves broad candidates. Reranking applies a second-stage relevance model to improve precision — the standard pattern in production RAG systems.

```python
from reranker import rerank_hybrid

# Full pipeline: hybrid retrieval → rerank → generate
answer = rerank_hybrid("What is pgvector?", rerank_mode="cross-encoder")

# Or rerank existing candidates
from reranker import rerank
ranked = rerank(question, candidates, mode="llm", top_k=3)
```

Two modes:
- **`cross-encoder`** (default): Uses a smaller model for fast scoring. Lower latency, good accuracy.
- **`llm`**: Uses the main LLM as a relevance judge. More thorough, higher cost per query.

The `rerank_hybrid()` function overretrieves (3× top_k), reranks, and returns only the most relevant chunks to the generator.

## Streaming Responses

Waiting for a full LLM response feels slow. Use `query_stream()` to see answers token-by-token:

```python
from example_rag import query_stream

for token in query_stream("What is pgvector?"):
    print(token, end="", flush=True)
# Tokens appear in real time instead of all at once
```

Works with any Ollama model — just set `OLLAMA_MODEL` in `.env`.

## Evaluating Retrieval Quality

How do you know if your RAG pipeline actually retrieves the right documents? The `eval_retrieval.py` script measures precision, recall, and mean reciprocal rank (MRR) against a labeled evaluation set.

```bash
# Quick evaluation with built-in sample questions
python eval_retrieval.py

# Custom evaluation set (JSONL: one {"question": "...", "relevant_sources": [...]} per line)
python eval_retrieval.py --eval-file my_evals.jsonl --top-k 5 --verbose

# Compare hybrid vs pure vector
python eval_retrieval.py --hybrid --alpha 0.7
```

Output:
```
==================================================
Retrieval Evaluation Results
==================================================
Method:      vector
Questions:   5
Top-K:       3
Precision@K: 0.800
Recall:      0.800
MRR:         0.900
```

Create your own eval set to measure retrieval quality on your actual data — evaluation isn't overhead, it's the specification of what "working" means.

## Waiting for Services

Docker Compose starts containers quickly, but PostgreSQL, TEI, and Ollama need time to initialize. Use `wait_for_services.py` to poll until everything is ready:

```bash
# Wait for all services (default 120s timeout)
python wait_for_services.py

# Custom timeout (useful for slow hardware or large models)
python wait_for_services.py --timeout 300

# Wait for a single service
python wait_for_services.py --service postgres

# Use in scripts — exits 0 on success, 1 on failure
python wait_for_services.py && echo "Ready!" || echo "Still starting..."
```

This is especially useful in CI pipelines or when pulling a large LLM model for the first time.

## Index Maintenance

After bulk document loads or if query performance degrades, rebuild indexes:

```bash
# Reindex everything + update statistics
make reindex

# Or selectively:
python3 reindex.py --vector    # Only vector (HNSW/IVFFlat) indexes
python3 reindex.py --fts       # Only full-text search indexes
python3 reindex.py --analyze   # Reindex + ANALYZE for query planner
```

Uses `REINDEX CONCURRENTLY` — non-blocking for reads.

## What's Next?

Extend the example for your use case:
- Add document loaders (PDF, web scraping, APIs)
- Try paragraph-aware chunking for structured documents
- Build a web interface (FastAPI, Streamlit)
- Add caching and rate limiting
- Deploy to your own infrastructure

## Alternative: Direct llama.cpp

If you prefer direct GGUF model serving without Ollama, the `docker-compose.yml` includes a commented `llm` service using llama.cpp. Uncomment that block and comment out the Ollama service to switch.
