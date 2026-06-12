"""
Batch document ingestion for local-rag-stack.

Loads documents from files (TXT, MD, CSV), directories, or URLs.
Handles chunking, embedding, and insertion into pgvector.

Usage:
    # Ingest a single file
    python batch_ingest.py ./docs/readme.md

    # Ingest all .txt and .md files in a directory
    python batch_ingest.py ./docs/ --recursive

    # Ingest from URLs
    python batch_ingest.py --urls https://example.com/page1 https://example.com/page2

    # Combine sources
    python batch_ingest.py ./docs/ --urls https://example.com/api-docs --recursive
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

from semantic_chunker import semantic_chunk

load_dotenv()

SUPPORTED_EXTENSIONS = {".txt", ".md", ".csv", ".html", ".rst", ".json"}


def load_file(filepath: str) -> tuple[str, str]:
    """Load a file and return (content, source_name).

    Handles encoding gracefully — tries UTF-8 first, then latin-1.
    """
    path = Path(filepath)
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        content = path.read_text(encoding="latin-1")
    source = path.name
    return content, source


def load_url(url: str) -> tuple[str, str]:
    """Fetch a URL and extract text content.

    Strips HTML tags if the response is HTML. Returns (text, url).
    """
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    text = resp.text

    # Basic HTML stripping — extract visible text
    if "<html" in text.lower() or "<body" in text.lower():
        import re
        # Remove scripts and styles
        text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", text, flags=re.DOTALL | re.IGNORECASE)
        # Remove tags
        text = re.sub(r"<[^>]+>", " ", text)
        # Clean up whitespace
        text = re.sub(r"\s+", " ", text).strip()

    return text, url


def collect_files(path: str, recursive: bool = False) -> list[str]:
    """Collect all supported files from a path.

    If path is a file, return it (if supported).
    If path is a directory, find all supported files (optionally recursive).
    """
    path = Path(path)

    if path.is_file():
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            return [str(path)]
        print(f"Skipping unsupported file: {path} (extension: {path.suffix})")
        return []

    if path.is_dir():
        pattern = "**/*" if recursive else "*"
        files = []
        for f in path.glob(pattern):
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(str(f))
        return sorted(files)

    print(f"Path not found: {path}")
    return []


def ingest_documents(
    file_paths: list[str] | None = None,
    urls: list[str] | None = None,
    chunk_size: int = 500,
    chunk_strategy: str = "sentence",
    batch_size: int = 50,
):
    """Ingest documents from files and/or URLs into the RAG database.

    Args:
        file_paths: List of file paths to ingest.
        urls: List of URLs to ingest.
        chunk_size: Maximum chunk size in characters.
        chunk_strategy: Chunking strategy ('sentence' or 'paragraph').
        batch_size: Number of chunks to process before committing.
    """
    from example_rag import init_db, add_document
    init_db()
    total_chunks = 0
    total_docs = 0

    # Process files
    if file_paths:
        for filepath in file_paths:
            try:
                content, source = load_file(filepath)
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
                continue

            chunks = semantic_chunk(content, chunk_size=chunk_size, strategy=chunk_strategy)
            for i, chunk in enumerate(chunks):
                add_document(chunk, source=f"{source}#{i+1}")
                total_chunks += 1
                if total_chunks % batch_size == 0:
                    print(f"  Processed {total_chunks} chunks from {total_docs} documents...")

            total_docs += 1
            print(f"  Ingested {len(chunks)} chunks from: {filepath}")

    # Process URLs
    if urls:
        for url in urls:
            try:
                content, source = load_url(url)
            except Exception as e:
                print(f"Error fetching {url}: {e}")
                continue

            chunks = semantic_chunk(content, chunk_size=chunk_size, strategy=chunk_strategy)
            for i, chunk in enumerate(chunks):
                add_document(chunk, source=f"{source}#{i+1}")
                total_chunks += 1

            total_docs += 1
            print(f"  Ingested {len(chunks)} chunks from URL: {url}")

    print(f"\nDone. {total_chunks} chunks from {total_docs} documents ingested.")
    return total_chunks, total_docs


def main():
    parser = argparse.ArgumentParser(
        description="Batch document ingestion for local-rag-stack"
    )
    parser.add_argument(
        "paths", nargs="*",
        help="Files or directories to ingest"
    )
    parser.add_argument(
        "--urls", nargs="+", default=[],
        help="URLs to ingest"
    )
    parser.add_argument(
        "--recursive", "-r", action="store_true",
        help="Recursively search directories"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=500,
        help="Max chunk size in characters (default: 500)"
    )
    parser.add_argument(
        "--chunk-strategy", choices=["sentence", "paragraph"], default="sentence",
        help="Chunking strategy (default: sentence)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=50,
        help="Progress reporting batch size (default: 50)"
    )

    args = parser.parse_args()

    if not args.paths and not args.urls:
        parser.error("Provide at least one file, directory, or --url")

    # Collect all files from paths
    all_files = []
    for path in args.paths:
        all_files.extend(collect_files(path, recursive=args.recursive))

    if all_files:
        print(f"Found {len(all_files)} file(s) to ingest")
    if args.urls:
        print(f"Found {len(args.urls)} URL(s) to ingest")

    if not all_files and not args.urls:
        print("No documents to ingest.")
        return

    ingest_documents(
        file_paths=all_files or None,
        urls=args.urls or None,
        chunk_size=args.chunk_size,
        chunk_strategy=args.chunk_strategy,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()