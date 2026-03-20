# embed-pro

Local embedding + reranking server for Apple Silicon (MPS). OpenAI-compatible API.

## Models

- **bge-m3** — multilingual embedding model (1024-dim vectors)
- **bge-reranker-v2-m3** — multilingual reranker

Both run on Apple MPS via `sentence-transformers`. Models are lazy-loaded on first request and automatically offloaded after 5 minutes of inactivity to free memory.

## Setup

```bash
pip install -r requirements.txt
python embed-pro.py
```

Server starts on port **8020** by default.

## API

### Embeddings

```bash
curl http://localhost:8020/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": ["hello world", "how are you"], "model": "bge-m3"}'
```

### Rerank

```bash
curl http://localhost:8020/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{"query": "what is AI", "documents": ["AI is...", "cooking recipe", "machine learning"], "top_n": 2}'
```

### Health

```bash
curl http://localhost:8020/health
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PORT` | `8020` | Server port |
| `BGE_M3_PATH` | `/Users/adam/models/bge-m3` | Path to embedding model |
| `BGE_RERANKER_PATH` | `/Users/adam/models/bge-reranker-v2-m3` | Path to reranker model |
| `OFFLOAD_SECONDS` | `300` | Idle time before model offload |
| `MAX_TEXTS` | `512` | Max texts per embedding request |
| `MAX_DOCUMENTS` | `512` | Max documents per rerank request |
| `BATCH_SIZE` | `64` | Inference batch size |
| `PRELOAD` | `false` | Load models at startup instead of first request |

## Run as macOS service

```bash
# Install
cp com.studio.embed-pro.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.studio.embed-pro.plist

# Uninstall
launchctl unload ~/Library/LaunchAgents/com.studio.embed-pro.plist

# Logs
tail -f /tmp/embed-pro.log
```
