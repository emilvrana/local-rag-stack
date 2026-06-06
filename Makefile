.PHONY: up down restart status health test reindex clean help

# Local RAG Stack — common operations
# Configure: cp .env.example .env && edit .env

DOCKER := docker-compose
MODEL  := $(shell grep OLLAMA_MODEL .env 2>/dev/null | cut -d= -f2 || echo qwen2.5:7b)

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

up: ## Start all services
	$(DOCKER) up -d
	@echo "⏳ Waiting for model '$(MODEL)' to download on first run..."
	@./healthcheck.sh --wait

down: ## Stop all services
	$(DOCKER) down

restart: ## Restart all services
	$(DOCKER) restart

status: ## Show container status
	@$(DOCKER) ps
	@echo ""
	@./healthcheck.sh --verbose

health: ## Run health check
	@./healthcheck.sh --verbose

test: ## Run retrieval evaluation
	python3 eval_retrieval.py --verbose

clean: ## Stop services and remove data volumes
	$(DOCKER) down -v
	@echo "🗑️  Volumes removed. Run 'make up' for a fresh start."

shell: ## Open psql session
	PGPASSWORD=$$(grep POSTGRES_PASSWORD .env 2>/dev/null | cut -d= -f2 || echo RagPass2025) \
		psql -h localhost -p $$(grep PG_PORT .env 2>/dev/null | cut -d= -f2 || echo 5433) \
		-U $$(grep POSTGRES_USER .env 2>/dev/null | cut -d= -f2 || echo raguser) \
		-d $$(grep POSTGRES_DB .env 2>/dev/null | cut -d= -f2 || echo ragdb)

reindex: ## Rebuild vector and FTS indexes (non-blocking)
	python3 reindex.py --analyze

query: ## Quick RAG query (usage: make query Q="your question")
	@test -n "$(Q)" || (echo "Usage: make query Q=\"your question\"" && exit 1)
	python3 -c "from example_rag import query; print(query('$(Q)'))"