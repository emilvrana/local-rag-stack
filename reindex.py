#!/usr/bin/env python3
"""Reindex PostgreSQL vector and full-text search indexes.

Use after bulk document loads, index corruption, or when query
performance degrades. Rebuilds all indexes concurrently (non-blocking
for normal reads).

Usage:
    python3 reindex.py              # Reindex all indexes
    python3 reindex.py --vector     # Only vector (HNSW/IVFFlat) indexes
    python3 reindex.py --fts        # Only full-text search indexes
    python3 reindex.py --analyze    # Also run ANALYZE after reindex
"""

import argparse
import os
import sys

import psycopg2
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5433"))
DB_NAME = os.getenv("DB_NAME", "ragdb")
DB_USER = os.getenv("DB_USER", "raguser")
DB_PASS = os.getenv("DB_PASS", "RagPass2025")


def get_connection():
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
        user=DB_USER, password=DB_PASS,
    )


def list_indexes(cur, table="documents"):
    """Return list of index names on the documents table."""
    cur.execute("""
        SELECT indexname FROM pg_indexes
        WHERE tablename = %s
    """, (table,))
    return [row[0] for row in cur.fetchall()]


def reindex_vector(cur):
    """Rebuild vector similarity indexes (HNSW, IVFFlat)."""
    print("Reindexing vector indexes...")
    indexes = list_indexes(cur)
    vector_indexes = [i for i in indexes if "embedding" in i.lower()]

    if not vector_indexes:
        print("  No vector indexes found. Creating default HNSW index...")
        cur.execute("""
            CREATE INDEX IF NOT EXISTS documents_embedding_hnsw_idx
            ON documents USING hnsw (embedding vector_cosine_ops)
        """)
        print("  Created documents_embedding_hnsw_idx")
    else:
        for idx in vector_indexes:
            cur.execute(f"REINDEX INDEX CONCURRENTLY {idx}")
            print(f"  Reindexed {idx}")


def reindex_fts(cur):
    """Rebuild full-text search and trigram indexes."""
    print("Reindexing FTS/trigram indexes...")

    # Ensure extensions exist
    cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

    # Ensure FTS column exists
    try:
        cur.execute("""
            ALTER TABLE documents ADD COLUMN IF NOT EXISTS
            search_vector tsvector
            GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
        """)
    except psycopg2.errors.DuplicateColumn:
        pass

    indexes = list_indexes(cur)
    fts_indexes = [i for i in indexes if "search" in i.lower() or "trgm" in i.lower()]

    if not fts_indexes:
        print("  No FTS indexes found. Creating defaults...")
        cur.execute("""
            CREATE INDEX IF NOT EXISTS documents_search_idx
            ON documents USING GIN (search_vector)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS documents_content_trgm_idx
            ON documents USING GIN (content gin_trgm_ops)
        """)
        print("  Created documents_search_idx, documents_content_trgm_idx")
    else:
        for idx in fts_indexes:
            cur.execute(f"REINDEX INDEX CONCURRENTLY {idx}")
            print(f"  Reindexed {idx}")


def run_analyze(cur):
    """Update table statistics for the query planner."""
    print("Running ANALYZE on documents table...")
    cur.execute("ANALYZE documents")
    print("  Statistics updated")


def main():
    parser = argparse.ArgumentParser(
        description="Reindex PostgreSQL vector and FTS indexes for the local RAG stack"
    )
    parser.add_argument("--vector", action="store_true", help="Only reindex vector indexes")
    parser.add_argument("--fts", action="store_true", help="Only reindex FTS/trigram indexes")
    parser.add_argument("--analyze", action="store_true", help="Run ANALYZE after reindexing")
    args = parser.parse_args()

    do_all = not args.vector and not args.fts

    conn = get_connection()
    conn.autocommit = True  # Required for CONCURRENTLY
    cur = conn.cursor()

    # Check table exists
    cur.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_name = 'documents'
        )
    """)
    if not cur.fetchone()[0]:
        print("documents table not found. Run the stack first (make up && python3 example_rag.py)")
        sys.exit(1)

    try:
        if do_all or args.vector:
            reindex_vector(cur)
        if do_all or args.fts:
            reindex_fts(cur)
        if args.analyze:
            run_analyze(cur)

        print("\n✓ Done")
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    main()