# Changelog

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
