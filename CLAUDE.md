# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Single-file FastAPI server (`embed-pro.py`) providing OpenAI-compatible embedding and reranking endpoints using BGE models on Apple MPS (Metal Performance Shaders).

- **Embedding model**: bge-m3 via `SentenceTransformer`
- **Reranking model**: bge-reranker-v2-m3 via `CrossEncoder`
- Both from `sentence-transformers` library

## Running

```bash
# Setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Start server (default port 8020)
python embed-pro.py

# Custom port
PORT=9000 python embed-pro.py

# Custom model paths (defaults to /Users/adam/models/bge-m3 and /Users/adam/models/bge-reranker-v2-m3)
BGE_M3_PATH=/path/to/model BGE_RERANKER_PATH=/path/to/reranker python embed-pro.py
```

## API Endpoints

- `POST /v1/embeddings` — OpenAI-compatible embeddings endpoint
- `POST /v1/rerank` — Reranking endpoint (query + documents)
- `GET /health` — Health check showing model load status

## Dependencies

Python packages: `fastapi`, `uvicorn`, `torch`, `sentence-transformers`, `pydantic`

## Architecture Notes

- Models are lazy-loaded on first request (not at startup)
- Single-file server: `embed-pro.py`
