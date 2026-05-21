# Changelog

## 2026-05-21

### Added
- `eval_retrieval.py`: retrieval evaluation utility measuring Precision@K, Recall, and MRR
- Built-in default evaluation set matching sample docs
- Support for custom JSONL evaluation files (`--eval-file`)
- Hybrid search evaluation mode (`--hybrid --alpha`)
- Verbose per-question output (`-v`)

## Previous

## 2026-04-20

### Added

- `hybrid_search.py`: combines vector similarity with PostgreSQL full-text search and trigram matching using Reciprocal Rank Fusion (RRF)
- Configurable `alpha` parameter to weight vector vs keyword results (default 0.7)
- `init_hybrid_tables()` auto-creates tsvector column, GIN index, and trigram index
- Fallback trigram search for terms not in the text search dictionary

## 2026-04-14

### Added

- `semantic_chunker.py`: sentence-aware and paragraph-aware chunking
- `example_rag.py` now uses semantic chunking by default (falls back to naive sliding window)
- Updated README with semantic chunking section

## 2026-04-11

### Changed
- Switched default LLM service from llama.cpp to Ollama
- Added automatic model pulling on container startup
- Updated TEI image to v1.5 (cpu-1.5)
- Added healthcheck for Ollama service
- Moved model configuration to `.env` file

### Added
- `ollama-entrypoint.sh`: handles automatic model download
- `CHANGELOG.md`: track project changes
- Ollama volume for persistent model storage

### Deprecated
- Direct llama.cpp service (still available as commented option)

## Previous
- Initial release with PostgreSQL + pgvector, TEI embeddings, llama.cpp serving
