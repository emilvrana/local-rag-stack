#!/bin/sh
# Ollama startup with automatic model pull

# Start ollama in background
ollama serve &

# Wait for ollama to be ready
sleep 5

# Pull the configured model if not already present
if ! ollama list | grep -q "$OLLAMA_MODEL"; then
    echo "Pulling model: $OLLAMA_MODEL"
    ollama pull "$OLLAMA_MODEL"
fi

# Keep the container running
wait
