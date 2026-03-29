#!/usr/bin/env bash
set -euo pipefail

export BBE_BASE_URL="${BBE_BASE_URL:-http://127.0.0.1:11434}"
export BBE_MODEL="${BBE_MODEL:-tinyllama:latest}"
export BBE_TEMPERATURE="${BBE_TEMPERATURE:-0.1}"
export BBE_MAX_TOKENS="${BBE_MAX_TOKENS:-120}"
export BBE_TIMEOUT_SECONDS="${BBE_TIMEOUT_SECONDS:-60}"
export BBE_CRITIQUE_PASS="${BBE_CRITIQUE_PASS:-false}"
export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
export OLLAMA_MODELS="${OLLAMA_MODELS:-/var/lib/ollama}"

echo "Starting Ollama on ${OLLAMA_HOST}..."
ollama serve > /tmp/ollama.log 2>&1 &
OLLAMA_PID=$!

cleanup() {
  if kill -0 "${OLLAMA_PID}" >/dev/null 2>&1; then
    kill "${OLLAMA_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

echo "Waiting for Ollama to become ready..."
for attempt in $(seq 1 60); do
  if curl -fsS "http://${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
    break
  fi

  if ! kill -0 "${OLLAMA_PID}" >/dev/null 2>&1; then
    echo "Ollama exited during startup."
    cat /tmp/ollama.log || true
    exit 1
  fi

  sleep 2

  if [ "${attempt}" -eq 60 ]; then
    echo "Ollama did not become ready in time."
    cat /tmp/ollama.log || true
    exit 1
  fi
done

echo "Ensuring model ${BBE_MODEL} is available..."
ollama pull "${BBE_MODEL}"

echo "Starting FastAPI backend on port ${PORT:-8000}..."
exec uvicorn main:app --host 0.0.0.0 --port "${PORT:-8000}"
