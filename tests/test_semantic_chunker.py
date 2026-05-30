"""Tests for semantic_chunker module.

Covers sentence splitting, paragraph splitting, grouping by size,
and both chunking strategies with edge cases.
"""

import pytest
from semantic_chunker import (
    _split_sentences,
    _split_paragraphs,
    _group_by_size,
    semantic_chunk,
    chunk_text_semantic,
)


class TestSplitSentences:
    def test_basic(self):
        text = "Hello world. How are you? I am fine!"
        result = _split_sentences(text)
        assert result == ["Hello world.", "How are you?", "I am fine!"]

    def test_single_sentence(self):
        assert _split_sentences("One sentence only.") == ["One sentence only."]

    def test_empty(self):
        assert _split_sentences("") == []

    def test_whitespace_only(self):
        assert _split_sentences("   ") == []

    def test_no_punctuation(self):
        # No sentence boundaries — returns the whole text as one unit
        result = _split_sentences("just words no punctuation")
        assert len(result) == 1
        assert result[0] == "just words no punctuation"


class TestSplitParagraphs:
    def test_basic(self):
        text = "First paragraph.\n\nSecond paragraph.\n\nThird."
        result = _split_paragraphs(text)
        assert len(result) == 3

    def test_single_paragraph(self):
        result = _split_paragraphs("Just one paragraph.")
        assert len(result) == 1

    def test_empty(self):
        assert _split_paragraphs("") == []

    def test_extra_newlines(self):
        text = "Para one.\n\n\n\nPara two."
        result = _split_paragraphs(text)
        assert len(result) == 2


class TestGroupBySize:
    def test_single_unit_fits(self):
        result = _group_by_size(["short text"], chunk_size=100, overlap=0)
        assert len(result) == 1

    def test_multiple_units_merged(self):
        units = ["one two", "three four", "five six"]
        result = _group_by_size(units, chunk_size=10, overlap=0)
        # All fit in one chunk
        assert len(result) == 1

    def test_overflow_creates_chunks(self):
        units = [f"word {i}" for i in range(20)]
        result = _group_by_size(units, chunk_size=5, overlap=0)
        assert len(result) > 1

    def test_overlap(self):
        units = [f"sentence number {i} here" for i in range(10)]
        result = _group_by_size(units, chunk_size=6, overlap=2)
        # With overlap, chunks share trailing units from previous chunk
        assert len(result) >= 2

    def test_empty_units(self):
        assert _group_by_size([], chunk_size=100, overlap=0) == []


class TestSemanticChunk:
    def test_sentence_strategy_basic(self):
        text = "First sentence here. Second sentence here. Third sentence here. Fourth sentence here. Fifth sentence here."
        chunks = semantic_chunk(text, chunk_size=6, strategy="sentence")
        assert len(chunks) >= 2
        # No sentence should be cut in the middle
        for chunk in chunks:
            assert chunk.strip()  # no empty chunks

    def test_paragraph_strategy_basic(self):
        text = "Short paragraph one.\n\nParagraph two has more words in it to make it a bit longer than the first one.\n\nParagraph three."
        chunks = semantic_chunk(text, chunk_size=20, strategy="paragraph")
        assert len(chunks) >= 1

    def test_empty_input(self):
        assert semantic_chunk("") == []

    def test_whitespace_input(self):
        assert semantic_chunk("   \n\n  ") == []

    def test_text_shorter_than_chunk_size(self):
        text = "This is short."
        result = semantic_chunk(text, chunk_size=100)
        assert len(result) == 1
        assert result[0] == "This is short."

    def test_paragraph_oversized_subdivides(self):
        # A single paragraph longer than chunk_size should be split by sentences
        long_para = ". ".join([f"Sentence number {i} continues here" for i in range(20)])
        text = long_para  # no double-newlines = one paragraph
        chunks = semantic_chunk(text, chunk_size=15, strategy="paragraph")
        assert len(chunks) > 1

    def test_default_strategy_is_sentence(self):
        text = "First. Second. Third. Fourth. Fifth."
        result_sentence = semantic_chunk(text, chunk_size=5, strategy="sentence")
        result_default = semantic_chunk(text, chunk_size=5)
        assert result_sentence == result_default


class TestChunkTextSemanticBackwardCompat:
    """chunk_text_semantic should be a drop-in replacement for naive chunking."""

    def test_basic(self):
        text = "This is some text. It has multiple sentences. Some more content here."
        chunks = chunk_text_semantic(text, chunk_size=10, overlap=2)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.strip()

    def test_delegates_to_semantic_chunk(self):
        text = "Test sentence one. Test sentence two."
        result_a = chunk_text_semantic(text, chunk_size=100)
        result_b = semantic_chunk(text, chunk_size=100, strategy="sentence")
        assert result_a == result_b