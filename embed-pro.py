#!/usr/bin/env python3
"""
Studio Embedding + Reranker Server
bge-m3 (embed) + bge-reranker-v2-m3 (rerank) on Apple MPS
FastAPI, OpenAI-compatible /v1/embeddings + /v1/rerank
"""
import os, time, torch
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Union
from sentence_transformers import SentenceTransformer, CrossEncoder

app = FastAPI(title="Studio Embed Server")

# Lazy-load models
_embed_model = None
_rerank_model = None

def get_embed_model():
    global _embed_model
    if _embed_model is None:
        print("Loading bge-m3 on MPS...")
        t0 = time.time()
        _embed_model = SentenceTransformer(os.environ.get("BGE_M3_PATH", "/Users/adam/models/bge-m3"), device="mps")
        print(f"bge-m3 loaded in {time.time()-t0:.1f}s")
    return _embed_model

def get_rerank_model():
    global _rerank_model
    if _rerank_model is None:
        print("Loading bge-reranker-v2-m3 on MPS...")
        t0 = time.time()
        _rerank_model = CrossEncoder(os.environ.get("BGE_RERANKER_PATH", "/Users/adam/models/bge-reranker-v2-m3"), device="mps")
        print(f"bge-reranker-v2-m3 loaded in {time.time()-t0:.1f}s")
    return _rerank_model

# --- Embeddings (OpenAI compatible) ---
class EmbedRequest(BaseModel):
    input: Union[List[str], str]
    model: str = "bge-m3"

@app.post("/v1/embeddings")
async def embeddings(req: EmbedRequest):
    texts = req.input if isinstance(req.input, list) else [req.input]
    model = get_embed_model()
    t0 = time.time()
    embs = model.encode(texts, normalize_embeddings=True)
    elapsed = time.time() - t0
    data = []
    for i, emb in enumerate(embs):
        data.append({
            "object": "embedding",
            "index": i,
            "embedding": emb.tolist()
        })
    return {
        "object": "list",
        "data": data,
        "model": "bge-m3",
        "usage": {"prompt_tokens": sum(len(t.split()) for t in texts), "total_tokens": 0},
        "_timing_ms": round(elapsed * 1000),
        "_throughput": round(len(texts) / elapsed, 1)
    }

# --- Reranking ---
class RerankPair(BaseModel):
    query: str
    document: str

class RerankRequest(BaseModel):
    query: str
    documents: List[str]
    model: str = "bge-reranker-v2-m3"
    top_n: Optional[int] = None

@app.post("/v1/rerank")
async def rerank(req: RerankRequest):
    model = get_rerank_model()
    pairs = [[req.query, doc] for doc in req.documents]
    t0 = time.time()
    scores = model.predict(pairs)
    elapsed = time.time() - t0
    results = []
    for i, score in enumerate(scores):
        results.append({
            "index": i,
            "relevance_score": float(score),
            "document": req.documents[i][:100]
        })
    results.sort(key=lambda x: x["relevance_score"], reverse=True)
    if req.top_n:
        results = results[:req.top_n]
    return {
        "results": results,
        "model": "bge-reranker-v2-m3",
        "_timing_ms": round(elapsed * 1000),
        "_pairs": len(pairs)
    }

@app.get("/health")
async def health():
    return {"status": "ok", "embed_loaded": _embed_model is not None, "rerank_loaded": _rerank_model is not None}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8020"))
    uvicorn.run(app, host="0.0.0.0", port=port)
