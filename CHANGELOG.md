# Changelog

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
