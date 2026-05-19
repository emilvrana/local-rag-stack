#!/usr/bin/env bash
# Health check for local-rag-stack services
# Usage: ./healthcheck.sh [--verbose] [--wait]

set -euo pipefail

VERBOSE=false
WAIT=false
TIMEOUT=30

for arg in "$@"; do
  case "$arg" in
    --verbose|-v) VERBOSE=true ;;
    --wait|-w) WAIT=true ;;
    --help|-h)
      echo "Usage: ./healthcheck.sh [--verbose] [--wait]"
      echo ""
      echo "  --verbose, -v  Show full response bodies"
      echo "  --wait, -w     Wait for all services to become healthy (up to 30s)"
      echo "  --help, -h     Show this help"
      exit 0
      ;;
  esac
done

# Service endpoints
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5433}"
DB_NAME="${DB_NAME:-ragdb}"
DB_USER="${DB_USER:-raguser}"
DB_PASS="${DB_PASS:-RagPass2025}"
EMBEDDINGS_URL="${EMBEDDING_URL:-http://localhost:8081/embed}"
LLM_URL="${LLM_URL:-http://localhost:8080/api/tags}"

check_postgres() {
  if PGPASSWORD="$DB_PASS" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "SELECT 1;" &>/dev/null; then
    return 0
  else
    # Fallback: check if port is listening
    if nc -z "$DB_HOST" "$DB_PORT" 2>/dev/null; then
      return 0
    fi
    return 1
  fi
}

check_embeddings() {
  local response
  response=$(curl -sf -X POST "$EMBEDDINGS_URL" \
    -H "Content-Type: application/json" \
    -d '{"inputs": "health check"}' 2>/dev/null)
  
  if [ $? -eq 0 ] && [ -n "$response" ]; then
    if [ "$VERBOSE" = true ]; then
      echo "  Embeddings response: $response"
    fi
    return 0
  fi
  return 1
}

check_llm() {
  local response
  response=$(curl -sf "$LLM_URL" 2>/dev/null)
  
  if [ $? -eq 0 ] && [ -n "$response" ]; then
    if [ "$VERBOSE" = true ]; then
      echo "  LLM models: $(echo "$response" | python3 -c "import sys,json; models=json.load(sys.stdin).get('models',[]); print(', '.join(m['name'] for m in models))" 2>/dev/null || echo "parse error")"
    fi
    return 0
  fi
  return 1
}

check_docker() {
  if ! command -v docker &>/dev/null; then
    echo "⚠️  Docker not found"
    return 1
  fi
  
  local services=("rag_postgres" "rag_embeddings" "rag_llm")
  local all_running=true
  
  for svc in "${services[@]}"; do
    if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^${svc}$"; then
      :
    else
      all_running=false
    fi
  done
  
  if [ "$all_running" = true ]; then
    return 0
  else
    echo "⚠️  Not all containers running. Check: docker-compose ps"
    return 1
  fi
}

# Print status
echo "🔍 local-rag-stack health check"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

failed=0

# Docker containers
printf "Docker containers  "
if check_docker; then
  echo "✅"
else
  echo "❌"
  failed=$((failed + 1))
fi

# PostgreSQL
printf "PostgreSQL (%s:%s)  " "$DB_HOST" "$DB_PORT"
if check_postgres; then
  echo "✅"
else
  echo "❌  (may still be starting)"
  failed=$((failed + 1))
fi

# Embeddings
printf "Embeddings API     "
if check_embeddings; then
  echo "✅"
else
  echo "❌  (may still be loading model)"
  failed=$((failed + 1))
fi

# LLM
printf "LLM (Ollama)       "
if check_llm; then
  echo "✅"
else
  echo "❌  (may still be downloading model)"
  failed=$((failed + 1))
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ "$failed" -eq 0 ]; then
  echo "✅ All services healthy"
  exit 0
else
  echo "❌ $failed service(s) not ready"
  if [ "$WAIT" = true ]; then
    echo "   Waiting up to ${TIMEOUT}s for services..."
    elapsed=0
    while [ $elapsed -lt $TIMEOUT ]; do
      sleep 5
      elapsed=$((elapsed + 5))
      if check_postgres && check_embeddings && check_llm; then
        echo "✅ All services healthy after ${elapsed}s"
        exit 0
      fi
      echo "   Still waiting... (${elapsed}s)"
    done
    echo "❌ Timed out after ${TIMEOUT}s"
  fi
  exit 1
fi