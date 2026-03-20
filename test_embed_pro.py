"""
Tests for embed-pro server.
Run: pytest test_embed_pro.py -v
Requires models at configured paths or PRELOAD=false (default).
"""
import json, hashlib, threading
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import numpy as np


# Patch models before import so tests don't need real GPU/models
class FakeTokenizer:
    model_max_length = 8192
    def encode(self, text, add_special_tokens=False):
        return list(range(len(text.split())))
    def decode(self, ids):
        return " ".join(f"tok{i}" for i in ids)


class FakeEmbedModel:
    tokenizer = FakeTokenizer()
    def encode(self, texts, normalize_embeddings=True, batch_size=64):
        dim = 1024
        embs = np.random.randn(len(texts), dim).astype(np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        return embs / norms


class FakeRerankModel:
    tokenizer = FakeTokenizer()
    def predict(self, pairs, batch_size=64):
        return np.random.rand(len(pairs)).astype(np.float32)


@pytest.fixture(autouse=True)
def mock_models(monkeypatch):
    """Skip model path check and inject fake models."""
    import embed_pro
    monkeypatch.setattr(embed_pro, "_check_model_paths", lambda: None)
    monkeypatch.setattr(embed_pro, "_warmup_models", lambda: None)
    monkeypatch.setattr(embed_pro, "_install_drain_handler", lambda: None)

    fake_embed = FakeEmbedModel()
    fake_rerank = FakeRerankModel()
    monkeypatch.setattr(embed_pro, "_embed_model", fake_embed)
    monkeypatch.setattr(embed_pro, "_rerank_model", fake_rerank)

    def get_embed():
        embed_pro._reset_timer("embed")
        return fake_embed
    def get_rerank():
        embed_pro._reset_timer("rerank")
        return fake_rerank

    monkeypatch.setattr(embed_pro, "get_embed_model", get_embed)
    monkeypatch.setattr(embed_pro, "get_rerank_model", get_rerank)


@pytest.fixture
def client():
    from embed_pro import app
    return TestClient(app)


# --- Health & readiness ---

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "uptime_seconds" in data
    assert "memory_mb" in data

def test_ready(client):
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json()["ready"] is True

def test_config(client):
    r = client.get("/config")
    assert r.status_code == 200
    data = r.json()
    assert "BATCH_SIZE" in data
    assert "API_KEY" in data
    assert data["API_KEY"]["value"] != "actual_secret"  # masked

def test_models(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    data = r.json()
    assert len(data["data"]) == 2
    ids = {m["id"] for m in data["data"]}
    assert "bge-m3" in ids
    assert "bge-reranker-v2-m3" in ids

def test_metrics(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert b"embed_requests_total" in r.content


# --- Embeddings ---

def test_embed_single(client):
    r = client.post("/v1/embeddings", json={"input": "hello world"})
    assert r.status_code == 200
    data = r.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 1
    assert data["data"][0]["object"] == "embedding"
    assert len(data["data"][0]["embedding"]) == 1024

def test_embed_batch(client):
    texts = [f"text {i}" for i in range(10)]
    r = client.post("/v1/embeddings", json={"input": texts})
    assert r.status_code == 200
    assert len(r.json()["data"]) == 10

def test_embed_empty_rejected(client):
    r = client.post("/v1/embeddings", json={"input": ""})
    assert r.status_code == 400

def test_embed_too_many(client):
    import embed_pro
    texts = ["x"] * (embed_pro.MAX_TEXTS + 1)
    r = client.post("/v1/embeddings", json={"input": texts})
    assert r.status_code == 400

def test_embed_char_limit(client):
    import embed_pro
    long_text = "x" * (embed_pro.MAX_CHARS + 1)
    r = client.post("/v1/embeddings", json={"input": long_text})
    assert r.status_code == 400
    assert "char limit" in r.json()["detail"]

def test_embed_truncate(client):
    r = client.post("/v1/embeddings", json={"input": "hello world this is a test", "truncate": True})
    assert r.status_code == 200

def test_embed_dimensions(client):
    r = client.post("/v1/embeddings", json={"input": "hello", "dimensions": 256})
    assert r.status_code == 200
    assert len(r.json()["data"][0]["embedding"]) == 256
    assert r.json()["_dimensions"] == 256

def test_embed_dimensions_invalid(client):
    r = client.post("/v1/embeddings", json={"input": "hello", "dimensions": 99999})
    assert r.status_code == 400

def test_embed_base64(client):
    r = client.post("/v1/embeddings", json={"input": "hello", "encoding_format": "base64"})
    assert r.status_code == 200
    emb = r.json()["data"][0]["embedding"]
    assert isinstance(emb, str)
    import base64
    raw = base64.b64decode(emb)
    assert len(raw) == 1024 * 4  # float32

def test_embed_dedup(client):
    texts = ["same text", "same text", "same text", "different"]
    r = client.post("/v1/embeddings", json={"input": texts})
    assert r.status_code == 200
    data = r.json()
    assert data["_unique"] <= 2
    # Deduped texts should have identical embeddings
    assert data["data"][0]["embedding"] == data["data"][1]["embedding"]
    assert data["data"][0]["embedding"] == data["data"][2]["embedding"]

def test_embed_cache(client):
    text = "cache me please"
    r1 = client.post("/v1/embeddings", json={"input": text})
    assert r1.status_code == 200
    r2 = client.post("/v1/embeddings", json={"input": text})
    assert r2.status_code == 200
    assert r2.json()["_cache_hits"] == 1

def test_embed_etag(client):
    r1 = client.post("/v1/embeddings", json={"input": "etag test"})
    assert r1.status_code == 200
    etag = r1.headers.get("etag")
    assert etag
    r2 = client.post("/v1/embeddings", json={"input": "etag test"}, headers={"If-None-Match": etag})
    assert r2.status_code == 304

def test_embed_stream(client):
    texts = [f"text {i}" for i in range(5)]
    r = client.post("/v1/embeddings", json={"input": texts, "stream": True})
    assert r.status_code == 200
    assert "ndjson" in r.headers.get("content-type", "")
    lines = [json.loads(line) for line in r.text.strip().split("\n")]
    assert len(lines) == 6  # 5 embeddings + 1 meta
    assert lines[-1]["_done"] is True
    assert lines[0]["object"] == "embedding"

def test_embed_request_id(client):
    r = client.post("/v1/embeddings", json={"input": "hello"}, headers={"X-Request-ID": "test-123"})
    assert r.headers.get("x-request-id") == "test-123"

def test_embed_generated_request_id(client):
    r = client.post("/v1/embeddings", json={"input": "hello"})
    assert r.headers.get("x-request-id")


# --- Reranking ---

def test_rerank(client):
    r = client.post("/v1/rerank", json={"query": "hello", "documents": ["doc1", "doc2", "doc3"]})
    assert r.status_code == 200
    data = r.json()
    assert len(data["results"]) == 3
    # Should be sorted by relevance_score descending
    scores = [x["relevance_score"] for x in data["results"]]
    assert scores == sorted(scores, reverse=True)

def test_rerank_top_n(client):
    r = client.post("/v1/rerank", json={"query": "hello", "documents": ["a", "b", "c"], "top_n": 2})
    assert r.status_code == 200
    assert len(r.json()["results"]) == 2

def test_rerank_empty_query(client):
    r = client.post("/v1/rerank", json={"query": "  ", "documents": ["a"]})
    assert r.status_code == 400

def test_rerank_empty_docs(client):
    r = client.post("/v1/rerank", json={"query": "hello", "documents": []})
    # pydantic will catch this or our validation
    assert r.status_code in (400, 422)

def test_rerank_truncate(client):
    r = client.post("/v1/rerank", json={"query": "hello", "documents": ["doc"], "truncate": True})
    assert r.status_code == 200


# --- Similarity ---

def test_similarity_single(client):
    r = client.post("/v1/similarity", json={"input_a": "hello", "input_b": "world"})
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data["similarities"], float)
    assert -1.1 <= data["similarities"] <= 1.1

def test_similarity_matrix(client):
    r = client.post("/v1/similarity", json={"input_a": ["a", "b"], "input_b": ["c", "d", "e"]})
    assert r.status_code == 200
    data = r.json()
    assert len(data["similarities"]) == 2
    assert len(data["similarities"][0]) == 3

def test_similarity_dimensions(client):
    r = client.post("/v1/similarity", json={"input_a": "hello", "input_b": "world", "dimensions": 128})
    assert r.status_code == 200
    assert r.json()["_dimensions"] == 128


# --- Tokenize ---

def test_tokenize(client):
    r = client.post("/v1/tokenize", json={"input": "hello world"})
    assert r.status_code == 200
    data = r.json()
    assert data["data"][0]["token_count"] >= 1
    assert data["data"][0]["max_tokens"] == 8192

def test_tokenize_batch(client):
    r = client.post("/v1/tokenize", json={"input": ["hello", "world"]})
    assert r.status_code == 200
    assert len(r.json()["data"]) == 2


# --- Auth ---

def test_auth_required(client, monkeypatch):
    import embed_pro
    monkeypatch.setattr(embed_pro, "API_KEY", "test-secret")
    r = client.post("/v1/embeddings", json={"input": "hello"})
    assert r.status_code == 401

def test_auth_valid(client, monkeypatch):
    import embed_pro
    monkeypatch.setattr(embed_pro, "API_KEY", "test-secret")
    r = client.post(
        "/v1/embeddings",
        json={"input": "hello"},
        headers={"Authorization": "Bearer test-secret"},
    )
    assert r.status_code == 200

def test_auth_invalid(client, monkeypatch):
    import embed_pro
    monkeypatch.setattr(embed_pro, "API_KEY", "test-secret")
    r = client.post(
        "/v1/embeddings",
        json={"input": "hello"},
        headers={"Authorization": "Bearer wrong-key"},
    )
    assert r.status_code == 401

def test_health_no_auth(client, monkeypatch):
    import embed_pro
    monkeypatch.setattr(embed_pro, "API_KEY", "test-secret")
    r = client.get("/health")
    assert r.status_code == 200


# --- Backpressure ---

def test_backpressure(client, monkeypatch):
    import embed_pro
    monkeypatch.setattr(embed_pro, "MAX_CONCURRENT", 1)
    monkeypatch.setattr(embed_pro, "_inference_sem", threading.Semaphore(0))  # all slots taken
    monkeypatch.setattr(embed_pro, "CACHE_SIZE", 0)  # disable cache so it hits inference path
    # Use unique text to avoid any prior cache entries
    r = client.post("/v1/embeddings", json={"input": "unique_backpressure_test_text_xyz"})
    assert r.status_code == 429


# --- Admin endpoints ---

def test_admin_offload(client):
    r = client.post("/admin/offload/embed")
    assert r.status_code == 200
    assert r.json()["offloaded"] == "bge-m3"

def test_admin_offload_all(client):
    r = client.post("/admin/offload/all")
    assert r.status_code == 200

def test_admin_offload_invalid(client):
    r = client.post("/admin/offload/invalid")
    assert r.status_code == 400

def test_admin_cache_clear(client):
    # Populate cache first
    client.post("/v1/embeddings", json={"input": "cache_clear_test"})
    r = client.post("/admin/cache/clear")
    assert r.status_code == 200
    assert r.json()["cleared"] >= 0

def test_admin_reload(client, monkeypatch):
    import embed_pro
    # Mock _force_reload to avoid actual model loading
    reloaded = []
    monkeypatch.setattr(embed_pro, "_force_reload_embed", lambda: reloaded.append("embed"))
    monkeypatch.setattr(embed_pro, "_force_reload_rerank", lambda: reloaded.append("rerank"))
    r = client.post("/admin/reload/all")
    assert r.status_code == 200
    assert "embed" in reloaded
    assert "rerank" in reloaded


# --- WebSocket ---

def test_ws_embed(client):
    with client.websocket_connect("/v1/embeddings/ws") as ws:
        ws.send_text(json.dumps({"id": "test1", "input": "hello world"}))
        resp = json.loads(ws.receive_text())
        assert resp["id"] == "test1"
        assert len(resp["data"]) == 1
        assert len(resp["data"][0]["embedding"]) == 1024

def test_ws_batch(client):
    with client.websocket_connect("/v1/embeddings/ws") as ws:
        ws.send_text(json.dumps({"id": "batch", "input": ["a", "b", "c"]}))
        resp = json.loads(ws.receive_text())
        assert len(resp["data"]) == 3

def test_ws_invalid_json(client):
    with client.websocket_connect("/v1/embeddings/ws") as ws:
        ws.send_text("not json")
        resp = json.loads(ws.receive_text())
        assert "error" in resp

def test_ws_empty_input(client):
    with client.websocket_connect("/v1/embeddings/ws") as ws:
        ws.send_text(json.dumps({"id": "empty", "input": []}))
        resp = json.loads(ws.receive_text())
        assert "error" in resp

def test_ws_dimensions(client):
    with client.websocket_connect("/v1/embeddings/ws") as ws:
        ws.send_text(json.dumps({"id": "dim", "input": "hello", "dimensions": 128}))
        resp = json.loads(ws.receive_text())
        assert len(resp["data"][0]["embedding"]) == 128

def test_ws_multiple_messages(client):
    with client.websocket_connect("/v1/embeddings/ws") as ws:
        for i in range(3):
            ws.send_text(json.dumps({"id": f"msg{i}", "input": f"text {i}"}))
            resp = json.loads(ws.receive_text())
            assert resp["id"] == f"msg{i}"

def test_ws_auth_rejected(client, monkeypatch):
    import embed_pro
    monkeypatch.setattr(embed_pro, "API_KEY", "secret123")
    with pytest.raises(Exception):
        with client.websocket_connect("/v1/embeddings/ws?token=wrong") as ws:
            ws.receive_text()

def test_ws_auth_valid(client, monkeypatch):
    import embed_pro
    monkeypatch.setattr(embed_pro, "API_KEY", "secret123")
    with client.websocket_connect("/v1/embeddings/ws?token=secret123") as ws:
        ws.send_text(json.dumps({"id": "auth", "input": "hello"}))
        resp = json.loads(ws.receive_text())
        assert resp["id"] == "auth"


# --- Backend/device info in response ---

def test_backend_info(client):
    r = client.post("/v1/embeddings", json={"input": "hello"})
    assert r.status_code == 200
    data = r.json()
    assert "_backend" in data
    assert "_device" in data

def test_models_show_backend(client):
    r = client.get("/v1/models")
    data = r.json()
    assert data["data"][0]["backend"] in ("torch", "onnx")

def test_health_shows_backend(client):
    r = client.get("/health")
    data = r.json()
    assert "backend" in data
    assert "device" in data
    assert "otel" in data
    assert "ws_connections" in data
