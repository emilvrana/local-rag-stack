# Local RAG Stack with Docker

This repository provides a simple `docker-compose.yml` to run the core components of a Retrieval-Augmented Generation (RAG) pipeline on your own server or local machine, even without a GPU.

It is based on the setup described in my blog post: [Running a Local LLM on Your Own Server](https://emil.aiadoption.cz/posts/running-a-local-llm-on-your-own-server).

## Components

This stack includes:

1.  **`postgres`**: A PostgreSQL database with the `pgvector` extension for storing vector embeddings.
2.  **`embeddings`**: A Hugging Face text embeddings model (`bge-base-en-v1.5`) served via the Text Embeddings Inference container. This service turns your documents into vector embeddings.
3.  **`llm`**: A GGUF-compatible Large Language Model (like Llama 3, Mistral, etc.) served via `llama.cpp`. This service provides the generative AI capabilities.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/) are installed.
- You have a GGUF-format language model file. You can download one from the [Hugging Face Hub](https://huggingface.co/models?search=gguf). A good starting point is a quantized 7B or 8B model.

## How to Use

1.  **Clone this repository:**
    ```bash
    git clone https://github.com/emilvrana/local-rag-stack.git
    cd local-rag-stack
    ```

2.  **Download a model:**
    - Create a `models` directory: `mkdir models`
    - Download your chosen GGUF model file and place it inside the `models` directory.

3.  **Update the `docker-compose.yml`:**
    - In the `llm` service definition, change the command from `your-model.gguf` to the actual filename of your model.
    ```yaml
    command: -m /models/your-model.gguf -c 4096 --host 0.0.0.0 --port 8080
    # Change to something like:
    # command: -m /models/Llama-3-8B-Instruct.Q4_K_M.gguf -c 4096 --host 0.0.0.0 --port 8080
    ```

4.  **Start the services:**
    ```bash
    docker-compose up -d
    ```

5.  **Verify:**
    - Check that the containers are running: `docker-compose ps`
    - You should now have:
        - A PostgreSQL database running on `localhost:5433`.
        - A text embeddings API endpoint at `http://localhost:8081`.
        - An LLM API endpoint at `http://localhost:8080` (compatible with the OpenAI API format).

## Python Example

A complete working example is provided in [`example_rag.py`](example_rag.py). It demonstrates:
- Database initialization with pgvector
- Document chunking with sliding windows
- Embedding via the local TEI service
- Similarity search using cosine distance
- Answer generation via the local LLM

```bash
pip install openai psycopg2-binary requests
python example_rag.py
```

## What's Next?

Extend the example for your use case:
- Add document loaders (PDF, web scraping, APIs)
- Implement smarter chunking (semantic splits, overlap tuning)
- Build a web interface (FastAPI, Streamlit)
- Add caching and rate limiting
- Deploy to your own infrastructure
