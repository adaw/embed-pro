# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Single-file FastAPI server (`embed-pro.py`) providing OpenAI-compatible embedding and reranking endpoints using BGE models. Supports Apple MPS, CUDA, CPU, and ONNX backends.

- **Embedding model**: bge-m3 via `SentenceTransformer` (torch or ONNX)
- **Reranking model**: bge-reranker-v2-m3 via `CrossEncoder`
- Both from `sentence-transformers` library

## Running

```bash
# Setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Start server (default: MPS, port 8020)
python embed-pro.py

# With preload + warmup
PRELOAD=true python embed-pro.py

# With auth
API_KEY=my-secret PRELOAD=true python embed-pro.py

# ONNX backend (faster inference, lower memory)
pip install optimum[onnxruntime]
EMBED_BACKEND=onnx PRELOAD=true python embed-pro.py

# CUDA device
DEVICE=cuda python embed-pro.py

# OpenTelemetry
pip install opentelemetry-api opentelemetry-sdk opentelemetry-instrumentation-fastapi opentelemetry-exporter-otlp
OTEL_ENABLED=true python embed-pro.py

# Docker
docker build -t embed-pro .
docker run -p 8020:8020 -v /path/to/models:/models \
  -e BGE_M3_PATH=/models/bge-m3 \
  -e BGE_RERANKER_PATH=/models/bge-reranker-v2-m3 \
  embed-pro
```

## Testing

```bash
pytest test_embed_pro.py -v
```

Tests use mock models — no GPU or real model files needed. 51 tests covering all endpoints, auth, WebSocket, admin, backpressure.

## API Endpoints

| Endpoint | Method | Auth | Description |
|---|---|---|---|
| `/v1/embeddings` | POST | Yes | OpenAI-compatible embeddings (`truncate`, `dimensions`, `encoding_format`, `stream`) |
| `/v1/embeddings/ws` | WS | Yes | WebSocket persistent connection for high-frequency embedding |
| `/v1/rerank` | POST | Yes | Reranking (`truncate`, `top_n`) |
| `/v1/similarity` | POST | Yes | Server-side cosine similarity between text sets |
| `/v1/tokenize` | POST | Yes | Token count estimation |
| `/v1/models` | GET | No | OpenAI-compatible model list |
| `/config` | GET | No | Runtime config introspection |
| `/health` | GET | No | Health check (uptime, memory, counters, cache, backend) |
| `/ready` | GET | No | Readiness probe (model loaded + not draining) |
| `/metrics` | GET | No | Prometheus metrics |
| `/admin/offload/{model}` | POST | Yes | Manually offload embed/rerank/all |
| `/admin/reload/{model}` | POST | Yes | Force reload embed/rerank/all |
| `/admin/cache/clear` | POST | Yes | Clear embedding LRU cache |

## Environment Variables

See `GET /config` for full list with descriptions. Key ones:

| Variable | Default | Description |
|---|---|---|
| `PORT` | 8020 | Server port |
| `DEVICE` | mps | Inference device: mps, cuda, cpu |
| `EMBED_BACKEND` | torch | Embedding backend: torch or onnx |
| `BGE_M3_PATH` | `./models/bge-m3` | Embedding model path |
| `BGE_RERANKER_PATH` | `./models/bge-reranker-v2-m3` | Reranker model path |
| `API_KEY` | (empty) | Bearer token auth (empty = disabled) |
| `PRELOAD` | false | Preload + warmup models at startup |
| `OFFLOAD_SECONDS` | 300 | Idle seconds before model offload |
| `MAX_TEXTS` | 512 | Max texts per embedding request |
| `MAX_DOCUMENTS` | 512 | Max documents per rerank request |
| `MAX_CHARS` | 32768 | Max chars per text |
| `MAX_BODY_BYTES` | 50MB | Max request body size |
| `BATCH_SIZE` | 64 | Base inference batch size |
| `ADAPTIVE_BATCH` | true | Auto-tune batch size by text length |
| `INFERENCE_TIMEOUT` | 120 | Max inference time (seconds) |
| `MAX_CONCURRENT` | 8 | Max concurrent inference (429 after) |
| `CACHE_SIZE` | 1024 | LRU embedding cache capacity (0=disabled) |
| `STREAM_THRESHOLD` | 64 | Auto-stream NDJSON when batch >= this |
| `DRAIN_TIMEOUT` | 30 | Graceful shutdown drain timeout |
| `OTEL_ENABLED` | false | Enable OpenTelemetry tracing |

## Architecture Notes

- Single-file server: `embed-pro.py`
- **Multi-backend**: torch (MPS/CUDA/CPU) or ONNX via `sentence-transformers` backend param
- Models lazy-loaded on first request (or preloaded with `PRELOAD=true`)
- MPS shader warmup on preload (eliminates cold-start latency)
- Auto-offload after idle period to free GPU memory
- Admin endpoints for manual offload/reload/cache management
- LRU cache for repeated embedding texts
- In-request dedup (identical texts computed once)
- Adaptive batch sizing (short texts -> bigger batches)
- Backpressure via semaphore (429 when full)
- NDJSON streaming for large batches
- WebSocket endpoint for persistent high-frequency connections
- ETag support for HTTP-level caching
- Graceful drain on SIGTERM (finish in-flight, reject new)
- Structured JSON logging with rate-limited error output
- Prometheus metrics for observability
- Optional OpenTelemetry distributed tracing
- Request body size limit (413 on oversized payloads)

## Dependencies

Core: `fastapi`, `uvicorn[standard]`, `torch`, `sentence-transformers`, `pydantic`, `prometheus-client`, `numpy`

Optional:
- ONNX backend: `optimum[onnxruntime]`
- OpenTelemetry: `opentelemetry-api`, `opentelemetry-sdk`, `opentelemetry-instrumentation-fastapi`, `opentelemetry-exporter-otlp`

Dev: `pytest`, `httpx`
