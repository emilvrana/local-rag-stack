"""
Semantic chunking for local-rag-stack.

Splits text into chunks that respect sentence boundaries, keeping
sentences intact instead of cutting mid-word like naive sliding windows.

Two strategies:
  - "sentence": Split at sentence boundaries, group sentences into chunks.
  - "paragraph": Split at paragraph boundaries (double newlines), then
    subdivide paragraphs that exceed chunk_size.

Usage:
    from semantic_chunker import semantic_chunk

    chunks = semantic_chunk(text, chunk_size=500, strategy="sentence")
"""

import re
from typing import Literal

# Sentence boundary pattern: period/exclamation/question followed by space or end
_SENTENCE_RE = re.compile(r'(?<=[.!?])\s+')
_PARAGRAPH_RE = re.compile(r'\n\s*\n')


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using punctuation heuristics."""
    parts = _SENTENCE_RE.split(text.strip())
    return [s.strip() for s in parts if s.strip()]


def _split_paragraphs(text: str) -> list[str]:
    """Split text at paragraph boundaries."""
    parts = _PARAGRAPH_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def _group_by_size(units: list[str], chunk_size: int, overlap: int) -> list[str]:
    """Group text units (sentences or paragraphs) into chunks by word count.

    Args:
        units: List of text units to group.
        chunk_size: Target chunk size in words.
        overlap: Number of words to overlap between chunks.

    Returns:
        List of chunk strings.
    """
    chunks = []
    current: list[str] = []
    current_len = 0

    for unit in units:
        unit_len = len(unit.split())

        if current_len + unit_len > chunk_size and current:
            # Finalize current chunk
            chunks.append(" ".join(current))

            # Build overlap: keep trailing units that fit within overlap
            overlap_units: list[str] = []
            overlap_len = 0
            for u in reversed(current):
                u_len = len(u.split())
                if overlap_len + u_len > overlap:
                    break
                overlap_units.insert(0, u)
                overlap_len += u_len

            current = overlap_units
            current_len = overlap_len

        current.append(unit)
        current_len += unit_len

    if current:
        chunks.append(" ".join(current))

    return chunks


def semantic_chunk(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
    strategy: Literal["sentence", "paragraph"] = "sentence",
) -> list[str]:
    """Split text into semantically coherent chunks.

    Unlike naive sliding-window chunking (which splits on word count regardless
    of meaning), this keeps sentences and paragraphs intact.

    Args:
        text: Input text to chunk.
        chunk_size: Target chunk size in words.
        overlap: Number of overlapping words between consecutive chunks.
        strategy: "sentence" splits at sentence boundaries, "paragraph" at
            paragraph boundaries first, then subdivides oversized paragraphs.

    Returns:
        List of chunk strings.
    """
    if not text or not text.strip():
        return []

    if strategy == "paragraph":
        paragraphs = _split_paragraphs(text)
        chunks = []
        for para in paragraphs:
            para_words = len(para.split())
            if para_words <= chunk_size:
                chunks.append(para)
            else:
                # Subdivide oversized paragraph by sentences
                sentences = _split_sentences(para)
                chunks.extend(_group_by_size(sentences, chunk_size, overlap))
        return chunks

    # Default: sentence strategy
    sentences = _split_sentences(text)
    return _group_by_size(sentences, chunk_size, overlap)


def chunk_text_semantic(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[str]:
    """Drop-in replacement for the original chunk_text function.

    Uses sentence-aware chunking instead of naive word sliding window.
    Same interface, better results.

    Args:
        text: Input text to chunk.
        chunk_size: Target chunk size in words.
        overlap: Number of overlapping words between consecutive chunks.

    Returns:
        List of chunk strings.
    """
    return semantic_chunk(text, chunk_size, overlap, strategy="sentence")