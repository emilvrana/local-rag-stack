# Contributing to Local RAG Stack

Thanks for your interest! This is a small, focused project — contributions welcome.

## Quick Start

1. Fork & clone
2. `cp .env.example .env`
3. `docker-compose up -d`
4. `pip install -r requirements.txt`
5. `python example_rag.py` — verify everything works

## What to Contribute

- **Bug fixes** — open an issue first, then PR
- **Document loaders** — PDF, web scraping, Markdown parsers
- **New chunking strategies** — see `semantic_chunker.py` for the interface
- **Retrieval improvements** — better hybrid search, reranking modes
- **Documentation** — examples, guides, clarifications

## Style

- Python 3.10+
- Functions with docstrings, type hints where practical
- Keep dependencies minimal — this project runs on modest hardware
- Run existing tests before submitting: `python -m pytest tests/`

## PR Process

1. Small, focused changes preferred over large refactors
2. Add tests for new functionality
3. Update README if you add user-facing features
4. One PR per feature/fix

## Reporting Issues

Include: Python version, OS, Docker version, error output, and steps to reproduce.