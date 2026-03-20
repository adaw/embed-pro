# embed-pro

Production-grade embedding + reranking server. OpenAI-compatible API, single-file, multi-backend.

**Models**: [bge-m3](https://huggingface.co/BAAI/bge-m3) (1024-dim embeddings) + [bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)

**Backends**: Apple MPS, CUDA, CPU, ONNX

## Quick Start

```bash
git clone https://github.com/adaw/embed-pro.git
cd embed-pro
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Place models (or symlink) into ./models/
mkdir -p models
# e.g. huggingface-cli download BAAI/bge-m3 --local-dir models/bge-m3
# e.g. huggingface-cli download BAAI/bge-reranker-v2-m3 --local-dir models/bge-reranker-v2-m3

python embed-pro.py
```

Server starts on `http://localhost:8020`. Models lazy-load on first request.

## Usage

```bash
# Embeddings (OpenAI-compatible)
curl http://localhost:8020/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": ["hello world", "how are you"]}'

# Rerank
curl http://localhost:8020/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{"query": "what is AI", "documents": ["AI is...", "cooking recipe"], "top_n": 1}'

# Similarity
curl http://localhost:8020/v1/similarity \
  -H "Content-Type: application/json" \
  -d '{"input_a": "hello", "input_b": "world"}'

# Token count
curl http://localhost:8020/v1/tokenize \
  -H "Content-Type: application/json" \
  -d '{"input": "hello world"}'
```

## Configuration

Copy `.env.example` to `.env`, or use environment variables:

```bash
# Auth
API_KEY=my-secret python embed-pro.py

# Preload models at startup (recommended for production)
PRELOAD=true python embed-pro.py

# ONNX backend (3x faster single requests)
pip install optimum[onnxruntime]
EMBED_BACKEND=onnx python embed-pro.py

# CUDA
DEVICE=cuda python embed-pro.py

# Dev mode (auto-reload on code change)
RELOAD=true python embed-pro.py
```

See all options: `curl http://localhost:8020/config`

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/v1/embeddings` | POST | Embeddings (`truncate`, `dimensions`, `encoding_format`, `stream`) |
| `/v1/embeddings/ws` | WS | WebSocket for high-frequency embedding |
| `/v1/rerank` | POST | Reranking (`truncate`, `top_n`) |
| `/v1/similarity` | POST | Cosine similarity between text sets |
| `/v1/tokenize` | POST | Token count estimation |
| `/v1/models` | GET | Model list |
| `/health` | GET | Health + stats |
| `/ready` | GET | Readiness probe |
| `/metrics` | GET | Prometheus metrics |
| `/config` | GET | Runtime config |
| `/admin/offload/{model}` | POST | Offload model from memory |
| `/admin/reload/{model}` | POST | Force reload model |
| `/admin/cache/clear` | POST | Clear embedding cache |
| `/admin/restart` | POST | Graceful restart |

## Performance

Tested on M1 Ultra (MPS):

| Metric | Torch/MPS | ONNX/CPU |
|---|---|---|
| Single request p50 | 71ms | 22ms |
| Batch 100 | 313ms (320/s) | 1347ms (71/s) |

## Dashboard

```bash
make dash
```

Live terminal dashboard with RPS sparklines, memory/cache graphs, model status, error counters.

## Testing

```bash
make test   # 51 tests, no GPU needed
```

## Docker

```bash
docker build -t embed-pro .
docker run -p 8020:8020 -v /path/to/models:/app/models embed-pro
```

## macOS Service

```bash
# Install (auto-start on login, auto-restart on crash)
cp com.studio.embed-pro.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.studio.embed-pro.plist

# Restart
launchctl kickstart -k gui/$(id -u)/com.studio.embed-pro

# Logs
tail -f /tmp/embed-pro.log
```

## Architecture

Single-file server with: lazy model loading, auto-offload after idle, LRU cache with auto-clear, in-request dedup, adaptive batch sizing, backpressure (429), NDJSON streaming, ETag caching, graceful drain on SIGTERM, structured JSON logging, rate-limited errors, Prometheus metrics, optional OpenTelemetry tracing.
