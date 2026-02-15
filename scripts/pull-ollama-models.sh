#!/bin/bash

echo "=========================================="
echo "Ollama Model Downloader"
echo "=========================================="
echo ""

# Wait for Ollama service to be fully ready
echo "⏳ Waiting for Ollama service to start..."
max_attempts=30
attempt=0

while ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; do
    attempt=$((attempt + 1))
    if [ $attempt -ge $max_attempts ]; then
        echo "❌ Failed to connect to Ollama service after $max_attempts attempts"
        exit 1
    fi
    echo "   Attempt $attempt/$max_attempts - waiting..."
    sleep 2
done

echo "✅ Ollama service is ready!"
echo ""

# Check if models already exist
echo "📋 Checking existing models..."
ollama list
echo ""

# Pull llama3.2 model if not exists
if ! ollama list | grep -q "llama3.2"; then
    echo "📥 Downloading llama3.2 model..."
    ollama pull llama3.2
    echo "✅ llama3.2 model downloaded successfully!"
else
    echo "✅ llama3.2 model already exists (skipping download)"
fi

echo ""

# Pull nomic-embed-text model if not exists
if ! ollama list | grep -q "nomic-embed-text"; then
    echo "📥 Downloading nomic-embed-text model for embeddings..."
    ollama pull nomic-embed-text
    echo "✅ nomic-embed-text model downloaded successfully!"
else
    echo "✅ nomic-embed-text model already exists (skipping download)"
fi

echo ""
echo "=========================================="
echo "✅ All required models are ready!"
echo "=========================================="
echo ""
echo "Available models:"
ollama list
