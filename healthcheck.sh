#!/usr/bin/env bash
# Health check for Local RAG Stack
# Usage: ./healthcheck.sh

set -euo pipefail

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

check() {
  local name=$1 url=$2
  if curl -sf -o /dev/null -m 5 "$url"; then
    echo -e "  ${GREEN}✓${NC} $name ($url)"
    return 0
  else
    echo -e "  ${RED}✗${NC} $name ($url)"
    return 1
  fi
}

echo "Local RAG Stack Health Check"
echo "============================"

failed=0

# Check Docker containers
echo ""
echo "Containers:"
if docker compose ps --format "table {{.Name}}\t{{.Status}}" 2>/dev/null || docker-compose ps 2>/dev/null; then
  :
else
  echo "  Could not list containers"
fi

# Check services
echo ""
echo "Services:"
check "PostgreSQL (pgvector)" "http://localhost:5433" || ((failed++)) || true
check "Embeddings (TEI)" "http://localhost:8081" || ((failed++)) || true
check "LLM (Ollama)" "http://localhost:8080/api/tags" || ((failed++)) || true

# Test embedding endpoint
echo ""
echo "Functional test:"
if curl -sf -m 10 http://localhost:8081/embed -X POST \
  -H "Content-Type: application/json" \
  -d '{"inputs": "health check"}' | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'  Embedding dim: {len(d[0]) if isinstance(d,list) and len(d)>0 else \"?\"}')" 2>/dev/null; then
  echo -e "  ${GREEN}✓${NC} Embedding pipeline works"
else
  echo -e "  ${RED}✗${NC} Embedding pipeline failed"
  ((failed)) || true
fi

echo ""
if [ "$failed" -eq 0 ]; then
  echo -e "${GREEN}All checks passed${NC}"
else
  echo -e "${RED}$failed check(s) failed${NC}"
  exit 1
fi