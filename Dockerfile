FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OLLAMA_HOST=127.0.0.1:11434 \
    OLLAMA_MODELS=/var/lib/ollama

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates bash zstd \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://ollama.com/install.sh | sh

COPY backend/requirements.txt /tmp/backend-requirements.txt
RUN pip install -r /tmp/backend-requirements.txt

COPY backend /app/backend
COPY explain /app/explain
COPY llm /app/llm
COPY utils /app/utils
COPY config.py /app/config.py
COPY docker-start.sh /app/docker-start.sh

RUN chmod +x /app/docker-start.sh \
    && mkdir -p /var/lib/ollama

WORKDIR /app/backend

EXPOSE 8000

CMD ["/app/docker-start.sh"]
