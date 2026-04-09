#!/usr/bin/env python3
"""
Simple RAG example for local-rag-stack.
Demonstrates: document loading → chunking → embedding → storage → retrieval → generation

Requires:
    pip install openai psycopg2-binary python-dotenv

Setup:
    1. Copy .env.example to .env: cp .env.example .env
    2. Adjust .env if you changed docker-compose.yml defaults
    3. Start services: docker-compose up -d
    4. Run: python example_rag.py
"""

import os
import hashlib
import psycopg2
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Configuration (override via .env file or environment variables)
EMBEDDING_URL = os.getenv("EMBEDDING_URL", "http://localhost:8081/embed")
LLM_URL = os.getenv("LLM_URL", "http://localhost:8080/v1")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5433"))
DB_NAME = os.getenv("DB_NAME", "ragdb")
DB_USER = os.getenv("DB_USER", "raguser")
DB_PASS = os.getenv("DB_PASS", "RagPass2025")

# LLM settings
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "local-model")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))

# Chunking settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "3"))


def get_embedding(text: str) -> list:
    """Get embedding from local text embeddings inference service."""
    import requests
    response = requests.post(
        EMBEDDING_URL,
        json={"inputs": text},
        headers={"Content-Type": "application/json"}
    )
    response.raise_for_status()
    return response.json()[0]


def init_db():
    """Ensure the vector table exists."""
    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
        user=DB_USER, password=DB_PASS
    )
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            content TEXT NOT NULL,
            embedding vector(768),
            source TEXT,
            chunk_id TEXT UNIQUE
        );
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS documents_embedding_idx 
        ON documents USING ivfflat (embedding vector_cosine_ops);
    """)
    conn.commit()
    cur.close()
    conn.close()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
    """Simple sliding window chunking."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def add_document(content: str, source: str = "unknown"):
    """Chunk, embed, and store a document."""
    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
        user=DB_USER, password=DB_PASS
    )
    cur = conn.cursor()
    
    chunks = chunk_text(content)
    for i, chunk in enumerate(chunks):
        chunk_id = hashlib.md5(f"{source}:{i}:{chunk[:100]}".encode()).hexdigest()
        
        # Check if already exists
        cur.execute("SELECT 1 FROM documents WHERE chunk_id = %s", (chunk_id,))
        if cur.fetchone():
            continue
        
        embedding = get_embedding(chunk)
        cur.execute(
            "INSERT INTO documents (content, embedding, source, chunk_id) VALUES (%s, %s, %s, %s)",
            (chunk, embedding, source, chunk_id)
        )
    
    conn.commit()
    cur.close()
    conn.close()
    print(f"Added {len(chunks)} chunks from {source}")


def query(question: str, top_k: int = DEFAULT_TOP_K) -> str:
    """Retrieve relevant chunks and generate an answer."""
    # Get query embedding
    q_embedding = get_embedding(question)
    
    # Retrieve similar chunks
    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
        user=DB_USER, password=DB_PASS
    )
    cur = conn.cursor()
    cur.execute(
        """SELECT content, source, 1 - (embedding <=> %s::vector) as similarity
           FROM documents
           ORDER BY embedding <=> %s::vector
           LIMIT %s;""",
        (q_embedding, q_embedding, top_k)
    )
    results = cur.fetchall()
    cur.close()
    conn.close()
    
    if not results:
        return "No relevant documents found."
    
    # Build context
    context = "\n\n".join([f"[{r[1]}]: {r[0]}" for r in results])
    
    # Query local LLM
    client = OpenAI(base_url=LLM_URL, api_key="not-needed")
    prompt = f"""Answer the question using only the provided context.

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


def main():
    """Example usage."""
    init_db()
    
    # Example: add some sample documents
    sample_docs = [
        ("""PostgreSQL is a powerful, open-source object-relational database system. 
        It has more than 30 years of active development and a proven architecture that 
        has earned it a strong reputation for reliability, feature robustness, and performance.""", "postgresql-overview"),
        
        ("""pgvector is an open-source vector similarity search for PostgreSQL. 
        It allows you to store, query, and index vectors alongside your regular data. 
        Supports L2 distance, inner product, and cosine distance operations.""", "pgvector-docs"),
        
        ("""llama.cpp is a port of Facebook's LLaMA model in C/C++. 
        It enables running LLMs on modest hardware, including CPUs, with quantization 
        techniques that reduce memory usage while maintaining reasonable quality.""", "llama.cpp-readme"),
    ]
    
    for content, source in sample_docs:
        add_document(content, source)
    
    # Example queries
    questions = [
        "What is pgvector used for?",
        "Can I run LLMs without a GPU?",
        "Tell me about PostgreSQL's reliability"
    ]
    
    for q in questions:
        print(f"\nQ: {q}")
        print(f"A: {query(q)}")


if __name__ == "__main__":
    main()
