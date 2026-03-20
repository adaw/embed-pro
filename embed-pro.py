#!/usr/bin/env python3
"""
Studio Embedding + Reranker Server
bge-m3 (embed) + bge-reranker-v2-m3 (rerank) on Apple MPS
FastAPI, OpenAI-compatible /v1/embeddings + /v1/rerank
"""
import os, gc, time, logging, threading, signal, sys
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Union
from sentence_transformers import SentenceTransformer, CrossEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("embed-pro")

OFFLOAD_SECONDS = int(os.environ.get("OFFLOAD_SECONDS", "300"))
MAX_TEXTS = int(os.environ.get("MAX_TEXTS", "512"))
MAX_DOCUMENTS = int(os.environ.get("MAX_DOCUMENTS", "512"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "64"))

PRELOAD = os.environ.get("PRELOAD", "").lower() in ("1", "true", "yes")


def _shutdown():
    for t in (_embed_timer, _rerank_timer):
        if t is not None:
            t.cancel()
    log.info("Timers cancelled, shutting down")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if PRELOAD:
        log.info("Preloading models...")
        get_embed_model()
        get_rerank_model()
    yield
    _shutdown()


app = FastAPI(title="Studio Embed Server", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Model manager with auto-offload ---

_embed_model = None
_rerank_model = None
_embed_lock = threading.Lock()
_rerank_lock = threading.Lock()
_embed_timer: Optional[threading.Timer] = None
_rerank_timer: Optional[threading.Timer] = None


def _offload_embed():
    global _embed_model
    with _embed_lock:
        if _embed_model is not None:
            log.info("Offloading bge-m3 (idle %ds)", OFFLOAD_SECONDS)
            del _embed_model
            _embed_model = None
            gc.collect()


def _offload_rerank():
    global _rerank_model
    with _rerank_lock:
        if _rerank_model is not None:
            log.info("Offloading bge-reranker-v2-m3 (idle %ds)", OFFLOAD_SECONDS)
            del _rerank_model
            _rerank_model = None
            gc.collect()


def _reset_timer(kind: str):
    global _embed_timer, _rerank_timer
    if kind == "embed":
        if _embed_timer is not None:
            _embed_timer.cancel()
        _embed_timer = threading.Timer(OFFLOAD_SECONDS, _offload_embed)
        _embed_timer.daemon = True
        _embed_timer.start()
    else:
        if _rerank_timer is not None:
            _rerank_timer.cancel()
        _rerank_timer = threading.Timer(OFFLOAD_SECONDS, _offload_rerank)
        _rerank_timer.daemon = True
        _rerank_timer.start()


def get_embed_model():
    global _embed_model
    with _embed_lock:
        if _embed_model is None:
            log.info("Loading bge-m3 on MPS...")
            t0 = time.time()
            _embed_model = SentenceTransformer(
                os.environ.get("BGE_M3_PATH", "/Users/adam/models/bge-m3"),
                device="mps",
            )
            log.info("bge-m3 loaded in %.1fs", time.time() - t0)
        _reset_timer("embed")
        return _embed_model


def get_rerank_model():
    global _rerank_model
    with _rerank_lock:
        if _rerank_model is None:
            log.info("Loading bge-reranker-v2-m3 on MPS...")
            t0 = time.time()
            _rerank_model = CrossEncoder(
                os.environ.get("BGE_RERANKER_PATH", "/Users/adam/models/bge-reranker-v2-m3"),
                device="mps",
            )
            log.info("bge-reranker-v2-m3 loaded in %.1fs", time.time() - t0)
        _reset_timer("rerank")
        return _rerank_model


# --- Embeddings (OpenAI compatible) ---

class EmbedRequest(BaseModel):
    input: Union[List[str], str]
    model: str = "bge-m3"


@app.post("/v1/embeddings")
def embeddings(req: EmbedRequest):
    texts = req.input if isinstance(req.input, list) else [req.input]
    if len(texts) > MAX_TEXTS:
        raise HTTPException(400, f"Max {MAX_TEXTS} texts per request")
    model = get_embed_model()
    t0 = time.time()
    embs = model.encode(texts, normalize_embeddings=True, batch_size=BATCH_SIZE)
    elapsed = time.time() - t0
    data = [
        {"object": "embedding", "index": i, "embedding": emb.tolist()}
        for i, emb in enumerate(embs)
    ]
    return {
        "object": "list",
        "data": data,
        "model": "bge-m3",
        "usage": {"prompt_tokens": sum(len(t.split()) for t in texts), "total_tokens": 0},
        "_timing_ms": round(elapsed * 1000),
        "_throughput": round(len(texts) / elapsed, 1),
    }


# --- Reranking ---

class RerankRequest(BaseModel):
    query: str
    documents: List[str]
    model: str = "bge-reranker-v2-m3"
    top_n: Optional[int] = None


@app.post("/v1/rerank")
def rerank(req: RerankRequest):
    if len(req.documents) > MAX_DOCUMENTS:
        raise HTTPException(400, f"Max {MAX_DOCUMENTS} documents per request")
    model = get_rerank_model()
    pairs = [[req.query, doc] for doc in req.documents]
    t0 = time.time()
    scores = model.predict(pairs, batch_size=BATCH_SIZE)
    elapsed = time.time() - t0
    results = [
        {"index": i, "relevance_score": float(score)}
        for i, score in enumerate(scores)
    ]
    results.sort(key=lambda x: x["relevance_score"], reverse=True)
    if req.top_n is not None:
        results = results[: req.top_n]
    return {
        "results": results,
        "model": "bge-reranker-v2-m3",
        "_timing_ms": round(elapsed * 1000),
        "_pairs": len(pairs),
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "embed_loaded": _embed_model is not None,
        "rerank_loaded": _rerank_model is not None,
        "offload_seconds": OFFLOAD_SECONDS,
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8020"))
    uvicorn.run(app, host="0.0.0.0", port=port)
