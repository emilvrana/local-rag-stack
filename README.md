# Local RAG Stack with Docker

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

## What's Next?

Extend the example for your use case:
- Add document loaders (PDF, web scraping, APIs)
- Try paragraph-aware chunking for structured documents
- Build a web interface (FastAPI, Streamlit)
- Add caching and rate limiting
- Deploy to your own infrastructure

## Alternative: Direct llama.cpp

If you prefer direct GGUF model serving without Ollama, the `docker-compose.yml` includes a commented `llm` service using llama.cpp. Uncomment that block and comment out the Ollama service to switch.
