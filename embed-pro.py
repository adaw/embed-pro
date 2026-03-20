#!/usr/bin/env python3
"""
embed-pro — Production embedding + reranking server
bge-m3 (embed) + bge-reranker-v2-m3 (rerank) on Apple MPS / ONNX / CUDA / CPU
FastAPI, OpenAI-compatible /v1/embeddings + /v1/rerank + WebSocket
"""
import os, gc, time, logging, threading, signal, sys, uuid, resource, json, asyncio
import hashlib, base64
import numpy as np
from collections import OrderedDict
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from fastapi import FastAPI, HTTPException, Request, Response, Security, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Optional, Union, Literal
from sentence_transformers import SentenceTransformer, CrossEncoder
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# Optional: OpenTelemetry
_otel_enabled = False
try:
    if os.environ.get("OTEL_ENABLED", "").lower() in ("1", "true", "yes"):
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        # Try OTLP exporter first, fall back to console
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            exporter = OTLPSpanExporter()
        except ImportError:
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter
            exporter = ConsoleSpanExporter()
        resource = Resource.create({"service.name": "embed-pro"})
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(BatchSpanExporter(exporter))
        trace.set_tracer_provider(provider)
        _otel_tracer = trace.get_tracer("embed-pro")
        _otel_enabled = True
except ImportError:
    pass

# --- Structured JSON logging ---

class JSONFormatter(logging.Formatter):
    def format(self, record):
        entry = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0]:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False)


handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logging.root.handlers = [handler]
logging.root.setLevel(logging.INFO)
log = logging.getLogger("embed-pro")

# Rate-limited error logging
ERROR_LOG_INTERVAL = 10
_error_log_last: dict[str, float] = {}
_error_log_suppressed: dict[str, int] = {}


def log_error_ratelimited(key: str, msg: str, exc: bool = True):
    now = time.time()
    last = _error_log_last.get(key, 0)
    if now - last >= ERROR_LOG_INTERVAL:
        suppressed = _error_log_suppressed.pop(key, 0)
        suffix = f" (suppressed {suppressed}x)" if suppressed else ""
        if exc:
            log.exception("%s%s", msg, suffix)
        else:
            log.error("%s%s", msg, suffix)
        _error_log_last[key] = now
    else:
        _error_log_suppressed[key] = _error_log_suppressed.get(key, 0) + 1

# --- Config ---

_CONFIG = {
    "OFFLOAD_SECONDS": ("300", "Idle seconds before model offload"),
    "MAX_TEXTS": ("512", "Max texts per embedding request"),
    "MAX_DOCUMENTS": ("512", "Max documents per rerank request"),
    "MAX_CHARS": ("32768", "Max characters per text/document"),
    "MAX_BODY_BYTES": (str(50 * 1024 * 1024), "Max request body size in bytes"),
    "BATCH_SIZE": ("64", "Inference batch size"),
    "INFERENCE_TIMEOUT": ("120", "Max inference time in seconds"),
    "DRAIN_TIMEOUT": ("30", "Max drain time on SIGTERM in seconds"),
    "MAX_CONCURRENT": ("8", "Max concurrent inference requests"),
    "CACHE_SIZE": ("1024", "LRU embedding cache capacity (0=disabled)"),
    "ADAPTIVE_BATCH": ("1", "Auto-tune batch size by text length"),
    "STREAM_THRESHOLD": ("64", "Stream NDJSON when batch >= this size"),
    "API_KEY": ("", "Bearer token (empty=auth disabled)"),
    "PRELOAD": ("", "Preload models on startup (1/true/yes)"),
    "DEVICE": ("mps", "Inference device: mps, cuda, cpu"),
    "EMBED_BACKEND": ("torch", "Embedding backend: torch or onnx"),
    "BGE_M3_PATH": ("/Users/adam/models/bge-m3", "Embedding model path"),
    "BGE_RERANKER_PATH": ("/Users/adam/models/bge-reranker-v2-m3", "Reranker model path"),
    "OTEL_ENABLED": ("", "Enable OpenTelemetry tracing (1/true/yes)"),
    "PORT": ("8020", "Server port"),
}

def _cfg(name: str) -> str:
    return os.environ.get(name, _CONFIG[name][0])

OFFLOAD_SECONDS = int(_cfg("OFFLOAD_SECONDS"))
MAX_TEXTS = int(_cfg("MAX_TEXTS"))
MAX_DOCUMENTS = int(_cfg("MAX_DOCUMENTS"))
MAX_CHARS = int(_cfg("MAX_CHARS"))
MAX_BODY_BYTES = int(_cfg("MAX_BODY_BYTES"))
BATCH_SIZE = int(_cfg("BATCH_SIZE"))
INFERENCE_TIMEOUT = int(_cfg("INFERENCE_TIMEOUT"))
DRAIN_TIMEOUT = int(_cfg("DRAIN_TIMEOUT"))
MAX_CONCURRENT = int(_cfg("MAX_CONCURRENT"))
CACHE_SIZE = int(_cfg("CACHE_SIZE"))
ADAPTIVE_BATCH = _cfg("ADAPTIVE_BATCH").lower() in ("1", "true", "yes")
STREAM_THRESHOLD = int(_cfg("STREAM_THRESHOLD"))
API_KEY = _cfg("API_KEY")
PRELOAD = _cfg("PRELOAD").lower() in ("1", "true", "yes")
DEVICE = _cfg("DEVICE")
EMBED_BACKEND = _cfg("EMBED_BACKEND").lower()
BGE_M3_PATH = _cfg("BGE_M3_PATH")
BGE_RERANKER_PATH = _cfg("BGE_RERANKER_PATH")

# --- Prometheus metrics ---

EMBED_REQUESTS = Counter("embed_requests_total", "Total embedding requests")
EMBED_ERRORS = Counter("embed_errors_total", "Total embedding errors", ["status"])
EMBED_LATENCY = Histogram("embed_latency_seconds", "Embedding latency", buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30])
EMBED_TEXTS_HIST = Histogram("embed_texts_per_request", "Texts per embedding request", buckets=[1, 5, 10, 25, 50, 100, 250, 512])
EMBED_CACHE_HITS = Counter("embed_cache_hits_total", "Embedding cache hits")
EMBED_CACHE_MISSES = Counter("embed_cache_misses_total", "Embedding cache misses")
EMBED_DEDUP = Counter("embed_dedup_saved_total", "Texts skipped via in-request dedup")

RERANK_REQUESTS = Counter("rerank_requests_total", "Total rerank requests")
RERANK_ERRORS = Counter("rerank_errors_total", "Total rerank errors", ["status"])
RERANK_LATENCY = Histogram("rerank_latency_seconds", "Rerank latency", buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30])
RERANK_PAIRS_HIST = Histogram("rerank_pairs_per_request", "Pairs per rerank request", buckets=[1, 5, 10, 25, 50, 100, 250, 512])

WS_CONNECTIONS = Gauge("ws_connections_active", "Active WebSocket connections")
REJECTED_REQUESTS = Counter("rejected_requests_total", "Requests rejected", ["reason"])
MODELS_LOADED = Gauge("models_loaded", "Number of models currently loaded")
INFLIGHT = Gauge("inflight_requests", "Requests currently being processed")

# --- State ---

_start_time = time.time()
_embed_count = 0
_rerank_count = 0
_stats_lock = threading.Lock()
_draining = False

_inference_pool = ThreadPoolExecutor(max_workers=2)
_inference_sem = threading.Semaphore(MAX_CONCURRENT)

# --- Embedding cache (LRU) ---

_embed_cache: OrderedDict[str, list] = OrderedDict()
_cache_lock = threading.Lock()


def _cache_key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _cache_get(text: str) -> Optional[list]:
    key = _cache_key(text)
    with _cache_lock:
        if key in _embed_cache:
            _embed_cache.move_to_end(key)
            EMBED_CACHE_HITS.inc()
            return _embed_cache[key]
    EMBED_CACHE_MISSES.inc()
    return None


def _cache_put(text: str, embedding: list):
    key = _cache_key(text)
    with _cache_lock:
        _embed_cache[key] = embedding
        _embed_cache.move_to_end(key)
        while len(_embed_cache) > CACHE_SIZE:
            _embed_cache.popitem(last=False)


def _cache_clear():
    with _cache_lock:
        count = len(_embed_cache)
        _embed_cache.clear()
    return count


# --- Auth ---

_bearer = HTTPBearer(auto_error=False)


async def verify_api_key(creds: Optional[HTTPAuthorizationCredentials] = Security(_bearer)):
    if not API_KEY:
        return
    if creds is None or creds.credentials != API_KEY:
        raise HTTPException(401, "Invalid or missing API key")


# --- Lifecycle ---

def _shutdown():
    for t in (_embed_timer, _rerank_timer):
        if t is not None:
            t.cancel()
    _inference_pool.shutdown(wait=False)
    log.info("Timers cancelled, shutting down")


def _check_model_paths():
    ok = True
    for name, path in [("bge-m3", BGE_M3_PATH), ("bge-reranker-v2-m3", BGE_RERANKER_PATH)]:
        if not os.path.isdir(path):
            log.error("Model path not found: %s -> %s", name, path)
            ok = False
        else:
            log.info("Model path OK: %s -> %s", name, path)
    if not ok:
        log.error("Fix model paths or set BGE_M3_PATH / BGE_RERANKER_PATH env vars")
        sys.exit(1)


def _warmup_models():
    log.info("Warming up embedding model...")
    t0 = time.time()
    m = get_embed_model()
    m.encode(["warmup"], normalize_embeddings=True)
    log.info("Embedding warmup done in %.1fs", time.time() - t0)

    log.info("Warming up reranker model...")
    t0 = time.time()
    r = get_rerank_model()
    r.predict([["warmup query", "warmup document"]])
    log.info("Reranker warmup done in %.1fs", time.time() - t0)


def _install_drain_handler():
    loop = asyncio.get_event_loop()

    def _sigterm(signum, frame):
        global _draining
        _draining = True
        log.info("SIGTERM received — draining (max %ds)...", DRAIN_TIMEOUT)

        async def _drain_and_exit():
            deadline = time.time() + DRAIN_TIMEOUT
            while INFLIGHT._value._value > 0 and time.time() < deadline:
                await asyncio.sleep(0.5)
            remaining = INFLIGHT._value._value
            if remaining > 0:
                log.warning("Drain timeout, %d requests still in-flight", remaining)
            else:
                log.info("All requests drained, exiting cleanly")
            _shutdown()
            sys.exit(0)

        loop.call_soon_threadsafe(lambda: asyncio.ensure_future(_drain_and_exit()))

    signal.signal(signal.SIGTERM, _sigterm)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _check_model_paths()
    _install_drain_handler()
    if PRELOAD:
        log.info("Preloading models...")
        get_embed_model()
        get_rerank_model()
        _warmup_models()
    log.info("Server ready (device=%s, backend=%s, auth=%s, offload=%ds, timeout=%ds, concurrent=%d, cache=%d, otel=%s)",
             DEVICE, EMBED_BACKEND, "on" if API_KEY else "off", OFFLOAD_SECONDS, INFERENCE_TIMEOUT,
             MAX_CONCURRENT, CACHE_SIZE, _otel_enabled)
    yield
    _shutdown()


app = FastAPI(title="embed-pro", version="3.0", lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenTelemetry auto-instrumentation
if _otel_enabled:
    FastAPIInstrumentor.instrument_app(app)


# --- Middleware ---

@app.middleware("http")
async def request_middleware(request: Request, call_next):
    if _draining and request.url.path not in ("/health", "/ready", "/metrics"):
        return Response(
            content=json.dumps({"detail": "Server is shutting down"}),
            status_code=503, media_type="application/json",
            headers={"Retry-After": "5"},
        )
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_BODY_BYTES:
        REJECTED_REQUESTS.labels(reason="body_too_large").inc()
        return Response(
            content=json.dumps({"detail": f"Request body exceeds {MAX_BODY_BYTES} bytes"}),
            status_code=413, media_type="application/json",
        )
    rid = request.headers.get("x-request-id", str(uuid.uuid4())[:8])
    request.state.request_id = rid
    INFLIGHT.inc()
    t0 = time.time()
    try:
        response = await call_next(request)
    finally:
        INFLIGHT.dec()
    response.headers["X-Request-ID"] = rid
    response.headers["X-Timing-Ms"] = str(round((time.time() - t0) * 1000))
    return response


# --- Model manager with auto-offload ---

_embed_model = None
_rerank_model = None
_embed_lock = threading.Lock()
_rerank_lock = threading.Lock()
_embed_timer: Optional[threading.Timer] = None
_rerank_timer: Optional[threading.Timer] = None


def _update_models_gauge():
    loaded = (_embed_model is not None) + (_rerank_model is not None)
    MODELS_LOADED.set(loaded)


def _offload_embed():
    global _embed_model
    with _embed_lock:
        if _embed_model is not None:
            log.info("Offloading bge-m3 (idle %ds)", OFFLOAD_SECONDS)
            del _embed_model
            _embed_model = None
            gc.collect()
            _update_models_gauge()


def _offload_rerank():
    global _rerank_model
    with _rerank_lock:
        if _rerank_model is not None:
            log.info("Offloading bge-reranker-v2-m3 (idle %ds)", OFFLOAD_SECONDS)
            del _rerank_model
            _rerank_model = None
            gc.collect()
            _update_models_gauge()


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


def _load_embed_model():
    """Load embedding model with configured backend and device."""
    if EMBED_BACKEND == "onnx":
        log.info("Loading bge-m3 with ONNX backend...")
        # sentence-transformers >= 3.0 supports backend="onnx"
        model = SentenceTransformer(BGE_M3_PATH, backend="onnx")
        log.info("ONNX providers: %s", getattr(model, '_onnx_providers', 'default'))
    else:
        log.info("Loading bge-m3 on %s (torch)...", DEVICE)
        model = SentenceTransformer(BGE_M3_PATH, device=DEVICE)
    return model


def get_embed_model():
    global _embed_model
    with _embed_lock:
        if _embed_model is None:
            t0 = time.time()
            _embed_model = _load_embed_model()
            log.info("bge-m3 loaded in %.1fs (backend=%s, device=%s)", time.time() - t0, EMBED_BACKEND, DEVICE)
            _update_models_gauge()
        _reset_timer("embed")
        return _embed_model


def get_rerank_model():
    global _rerank_model
    with _rerank_lock:
        if _rerank_model is None:
            log.info("Loading bge-reranker-v2-m3 on %s...", DEVICE)
            t0 = time.time()
            _rerank_model = CrossEncoder(BGE_RERANKER_PATH, device=DEVICE)
            log.info("bge-reranker-v2-m3 loaded in %.1fs", time.time() - t0)
            _update_models_gauge()
        _reset_timer("rerank")
        return _rerank_model


def _force_reload_embed():
    global _embed_model
    with _embed_lock:
        if _embed_model is not None:
            del _embed_model
            _embed_model = None
            gc.collect()
        t0 = time.time()
        _embed_model = _load_embed_model()
        log.info("bge-m3 reloaded in %.1fs", time.time() - t0)
        _update_models_gauge()
        _reset_timer("embed")


def _force_reload_rerank():
    global _rerank_model
    with _rerank_lock:
        if _rerank_model is not None:
            del _rerank_model
            _rerank_model = None
            gc.collect()
        log.info("Reloading bge-reranker-v2-m3 on %s...", DEVICE)
        t0 = time.time()
        _rerank_model = CrossEncoder(BGE_RERANKER_PATH, device=DEVICE)
        log.info("bge-reranker-v2-m3 reloaded in %.1fs", time.time() - t0)
        _update_models_gauge()
        _reset_timer("rerank")


def _acquire_inference_slot():
    if not _inference_sem.acquire(blocking=False):
        REJECTED_REQUESTS.labels(reason="backpressure").inc()
        raise HTTPException(429, f"Server busy — max {MAX_CONCURRENT} concurrent inference requests")


def _release_inference_slot():
    _inference_sem.release()


def _run_with_timeout(fn, timeout, error_msg):
    future = _inference_pool.submit(fn)
    try:
        return future.result(timeout=timeout)
    except FuturesTimeout:
        log_error_ratelimited("timeout", f"{error_msg} (timeout {timeout}s)", exc=False)
        raise HTTPException(504, f"{error_msg}: timed out after {timeout}s")


def _get_tokenizer():
    model = get_embed_model()
    return model.tokenizer


def _adaptive_batch_size(texts: list[str]) -> int:
    if not ADAPTIVE_BATCH or not texts:
        return BATCH_SIZE
    avg_len = sum(len(t) for t in texts) / len(texts)
    if avg_len < 200:
        return min(BATCH_SIZE * 2, 256)
    elif avg_len < 1000:
        return BATCH_SIZE
    elif avg_len < 5000:
        return max(BATCH_SIZE // 2, 8)
    else:
        return max(BATCH_SIZE // 4, 4)


def _dedup_texts(texts: list[str]) -> tuple[list[str], list[int], dict[int, int]]:
    seen: dict[str, int] = {}
    unique_texts: list[str] = []
    unique_indices: list[int] = []
    mapping: dict[int, int] = {}
    for i, t in enumerate(texts):
        if t in seen:
            mapping[i] = seen[t]
        else:
            uid = len(unique_texts)
            seen[t] = uid
            unique_texts.append(t)
            unique_indices.append(i)
            mapping[i] = uid
    deduped = len(texts) - len(unique_texts)
    if deduped > 0:
        EMBED_DEDUP.inc(deduped)
    return unique_texts, unique_indices, mapping


def _format_embedding(emb: list, fmt: str) -> Union[list, str]:
    if fmt == "base64":
        return base64.b64encode(np.array(emb, dtype=np.float32).tobytes()).decode("ascii")
    return emb


def _apply_dimensions(embeddings: list[list], dimensions: Optional[int]) -> list[list]:
    if dimensions is None:
        return embeddings
    full_dim = len(embeddings[0])
    if dimensions < 1 or dimensions > full_dim:
        raise HTTPException(400, f"dimensions must be 1-{full_dim}, got {dimensions}")
    result = []
    for emb in embeddings:
        arr = np.array(emb[:dimensions], dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        result.append(arr.tolist())
    return result


def _etag_for_texts(texts: list[str], dimensions: Optional[int], fmt: str) -> str:
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode())
    h.update(f"|d={dimensions}|f={fmt}".encode())
    return f'"{h.hexdigest()[:16]}"'


def _embed_texts_core(texts: list[str]) -> tuple[list[list], float, int, int]:
    """Core embedding logic shared by HTTP and WebSocket. Returns (embeddings, elapsed, unique_count, cache_hits)."""
    unique_texts, unique_indices, dedup_map = _dedup_texts(texts)

    cached: dict[int, list] = {}
    uncached_uids: list[int] = []
    uncached_texts: list[str] = []
    if CACHE_SIZE > 0:
        for uid, t in enumerate(unique_texts):
            hit = _cache_get(t)
            if hit is not None:
                cached[uid] = hit
            else:
                uncached_uids.append(uid)
                uncached_texts.append(t)
    else:
        uncached_uids = list(range(len(unique_texts)))
        uncached_texts = list(unique_texts)

    elapsed = 0.0
    if uncached_texts:
        model = get_embed_model()
        bs = _adaptive_batch_size(uncached_texts)
        t0 = time.time()
        embs = model.encode(uncached_texts, normalize_embeddings=True, batch_size=bs)
        elapsed = time.time() - t0
        for uid, emb in zip(uncached_uids, embs):
            vec = emb.tolist()
            cached[uid] = vec
            if CACHE_SIZE > 0:
                _cache_put(unique_texts[uid], vec)

    all_embeddings = [cached[dedup_map[i]] for i in range(len(texts))]
    cache_hits = len(unique_texts) - len(uncached_texts)
    return all_embeddings, elapsed, len(unique_texts), cache_hits


# --- Embeddings (OpenAI compatible) ---

class EmbedRequest(BaseModel):
    input: Union[List[str], str]
    model: str = "bge-m3"
    truncate: bool = False
    dimensions: Optional[int] = None
    encoding_format: Literal["float", "base64"] = "float"
    stream: bool = False


@app.post("/v1/embeddings", dependencies=[Depends(verify_api_key)])
def embeddings(request: Request, req: EmbedRequest):
    global _embed_count
    texts = req.input if isinstance(req.input, list) else [req.input]
    if not texts or all(t == "" for t in texts):
        raise HTTPException(400, "Input must contain at least one non-empty string")
    if len(texts) > MAX_TEXTS:
        raise HTTPException(400, f"Max {MAX_TEXTS} texts per request")

    etag = _etag_for_texts(texts, req.dimensions, req.encoding_format)
    if_none_match = request.headers.get("if-none-match")
    if if_none_match == etag:
        return Response(status_code=304, headers={"ETag": etag})

    truncated_indices = []
    if req.truncate:
        tokenizer = _get_tokenizer()
        max_tokens = tokenizer.model_max_length
        for i, t in enumerate(texts):
            encoded = tokenizer.encode(t, add_special_tokens=False)
            if len(encoded) > max_tokens:
                texts[i] = tokenizer.decode(encoded[:max_tokens])
                truncated_indices.append(i)
    else:
        for i, t in enumerate(texts):
            if len(t) > MAX_CHARS:
                raise HTTPException(400, f"Text at index {i} exceeds {MAX_CHARS} char limit ({len(t)} chars)")

    EMBED_REQUESTS.inc()
    EMBED_TEXTS_HIST.observe(len(texts))

    _acquire_inference_slot()
    try:
        try:
            all_embeddings, elapsed, unique_count, cache_hits = _embed_texts_core(texts)
        except Exception as e:
            if isinstance(e, HTTPException):
                raise
            log_error_ratelimited("embed_infer", "Embedding inference failed")
            EMBED_ERRORS.labels(status="500").inc()
            raise HTTPException(500, f"Inference error: {e}")
    finally:
        _release_inference_slot()

    if elapsed > 0:
        EMBED_LATENCY.observe(elapsed)
    with _stats_lock:
        _embed_count += 1

    all_embeddings = _apply_dimensions(all_embeddings, req.dimensions)

    # Streaming NDJSON
    use_stream = req.stream or (len(texts) >= STREAM_THRESHOLD)
    if use_stream:
        def _generate_ndjson():
            for i, emb in enumerate(all_embeddings):
                line = {"object": "embedding", "index": i, "embedding": _format_embedding(emb, req.encoding_format)}
                yield json.dumps(line, ensure_ascii=False) + "\n"
            meta = {
                "_done": True, "model": "bge-m3",
                "_timing_ms": round(elapsed * 1000), "_count": len(texts),
                "_unique": unique_count, "_cache_hits": cache_hits,
            }
            if truncated_indices:
                meta["_truncated"] = truncated_indices
            yield json.dumps(meta, ensure_ascii=False) + "\n"
        return StreamingResponse(
            _generate_ndjson(), media_type="application/x-ndjson",
            headers={"ETag": etag, "X-Request-ID": getattr(request.state, "request_id", "")},
        )

    data = [
        {"object": "embedding", "index": i, "embedding": _format_embedding(emb, req.encoding_format)}
        for i, emb in enumerate(all_embeddings)
    ]
    result = {
        "object": "list", "data": data, "model": "bge-m3",
        "usage": {"prompt_tokens": sum(len(t.split()) for t in texts), "total_tokens": 0},
        "_timing_ms": round(elapsed * 1000),
        "_throughput": round(len(texts) / elapsed, 1) if elapsed > 0 else None,
        "_unique": unique_count, "_cache_hits": cache_hits,
        "_backend": EMBED_BACKEND, "_device": DEVICE,
    }
    if truncated_indices:
        result["_truncated"] = truncated_indices
    if req.dimensions:
        result["_dimensions"] = req.dimensions
    return Response(
        content=json.dumps(result, ensure_ascii=False),
        media_type="application/json", headers={"ETag": etag},
    )


# --- WebSocket embeddings ---

@app.websocket("/v1/embeddings/ws")
async def ws_embeddings(ws: WebSocket):
    # Auth check for WebSocket
    if API_KEY:
        token = ws.query_params.get("token", "")
        if token != API_KEY:
            await ws.close(code=4001, reason="Invalid API key")
            return

    await ws.accept()
    WS_CONNECTIONS.inc()
    log.info("WebSocket connected")
    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send_text(json.dumps({"error": "Invalid JSON"}))
                continue

            msg_id = msg.get("id", str(uuid.uuid4())[:8])
            texts = msg.get("input", [])
            if isinstance(texts, str):
                texts = [texts]
            dimensions = msg.get("dimensions")
            encoding_format = msg.get("encoding_format", "float")

            if not texts:
                await ws.send_text(json.dumps({"id": msg_id, "error": "Empty input"}))
                continue
            if len(texts) > MAX_TEXTS:
                await ws.send_text(json.dumps({"id": msg_id, "error": f"Max {MAX_TEXTS} texts"}))
                continue

            try:
                all_embeddings, elapsed, unique_count, cache_hits = await asyncio.get_event_loop().run_in_executor(
                    _inference_pool, lambda: _embed_texts_core(texts)
                )
                all_embeddings = _apply_dimensions(all_embeddings, dimensions)
                data = [
                    {"index": i, "embedding": _format_embedding(emb, encoding_format)}
                    for i, emb in enumerate(all_embeddings)
                ]
                response = {
                    "id": msg_id, "data": data, "model": "bge-m3",
                    "_timing_ms": round(elapsed * 1000), "_unique": unique_count,
                    "_cache_hits": cache_hits,
                }
                await ws.send_text(json.dumps(response, ensure_ascii=False))
            except Exception as e:
                log_error_ratelimited("ws_embed", "WebSocket embedding error")
                await ws.send_text(json.dumps({"id": msg_id, "error": str(e)}))
    except WebSocketDisconnect:
        pass
    finally:
        WS_CONNECTIONS.dec()
        log.info("WebSocket disconnected")


# --- Reranking ---

class RerankRequest(BaseModel):
    query: str
    documents: List[str]
    model: str = "bge-reranker-v2-m3"
    top_n: Optional[int] = None
    truncate: bool = False


@app.post("/v1/rerank", dependencies=[Depends(verify_api_key)])
def rerank(req: RerankRequest):
    global _rerank_count
    if not req.query.strip():
        raise HTTPException(400, "Query must be a non-empty string")
    if not req.documents or all(d == "" for d in req.documents):
        raise HTTPException(400, "Documents must contain at least one non-empty string")
    if len(req.documents) > MAX_DOCUMENTS:
        raise HTTPException(400, f"Max {MAX_DOCUMENTS} documents per request")

    if not req.truncate:
        if len(req.query) > MAX_CHARS:
            raise HTTPException(400, f"Query exceeds {MAX_CHARS} char limit ({len(req.query)} chars)")
        for i, d in enumerate(req.documents):
            if len(d) > MAX_CHARS:
                raise HTTPException(400, f"Document at index {i} exceeds {MAX_CHARS} char limit ({len(d)} chars)")

    RERANK_REQUESTS.inc()
    RERANK_PAIRS_HIST.observe(len(req.documents))

    _acquire_inference_slot()
    try:
        try:
            model = get_rerank_model()
        except Exception as e:
            log_error_ratelimited("rerank_load", "Failed to load reranker model")
            RERANK_ERRORS.labels(status="503").inc()
            raise HTTPException(503, f"Reranker model unavailable: {e}")

        query = req.query
        documents = list(req.documents)
        truncated_indices = []
        if req.truncate:
            tokenizer = model.tokenizer
            max_len = tokenizer.model_max_length
            qenc = tokenizer.encode(query, add_special_tokens=False)
            if len(qenc) > max_len // 2:
                query = tokenizer.decode(qenc[: max_len // 2])
                truncated_indices.append("query")
            budget = max_len - min(len(qenc), max_len // 2) - 3
            for i, d in enumerate(documents):
                denc = tokenizer.encode(d, add_special_tokens=False)
                if len(denc) > budget:
                    documents[i] = tokenizer.decode(denc[:budget])
                    truncated_indices.append(i)

        pairs = [[query, doc] for doc in documents]
        bs = _adaptive_batch_size(documents)
        try:
            t0 = time.time()
            scores = _run_with_timeout(
                lambda: model.predict(pairs, batch_size=bs),
                INFERENCE_TIMEOUT, "Rerank inference timeout",
            )
            elapsed = time.time() - t0
        except HTTPException:
            RERANK_ERRORS.labels(status="504").inc()
            raise
        except Exception as e:
            log_error_ratelimited("rerank_infer", "Rerank inference failed")
            RERANK_ERRORS.labels(status="500").inc()
            raise HTTPException(500, f"Inference error: {e}")
    finally:
        _release_inference_slot()

    RERANK_LATENCY.observe(elapsed)
    with _stats_lock:
        _rerank_count += 1
    results = [{"index": i, "relevance_score": float(score)} for i, score in enumerate(scores)]
    results.sort(key=lambda x: x["relevance_score"], reverse=True)
    if req.top_n is not None:
        results = results[: req.top_n]
    result = {"results": results, "model": "bge-reranker-v2-m3", "_timing_ms": round(elapsed * 1000), "_pairs": len(pairs)}
    if truncated_indices:
        result["_truncated"] = truncated_indices
    return result


# --- Similarity ---

class SimilarityRequest(BaseModel):
    input_a: Union[List[str], str]
    input_b: Union[List[str], str]
    model: str = "bge-m3"
    dimensions: Optional[int] = None


@app.post("/v1/similarity", dependencies=[Depends(verify_api_key)])
def similarity(req: SimilarityRequest):
    texts_a = req.input_a if isinstance(req.input_a, list) else [req.input_a]
    texts_b = req.input_b if isinstance(req.input_b, list) else [req.input_b]
    if not texts_a or not texts_b:
        raise HTTPException(400, "Both input_a and input_b must be non-empty")
    total = len(texts_a) + len(texts_b)
    if total > MAX_TEXTS:
        raise HTTPException(400, f"Combined texts ({total}) exceed max {MAX_TEXTS}")
    for t in texts_a + texts_b:
        if len(t) > MAX_CHARS:
            raise HTTPException(400, f"Text exceeds {MAX_CHARS} char limit")

    all_texts = texts_a + texts_b
    _acquire_inference_slot()
    try:
        try:
            model = get_embed_model()
        except Exception as e:
            log_error_ratelimited("embed_load", "Failed to load embedding model")
            raise HTTPException(503, f"Embedding model unavailable: {e}")
        bs = _adaptive_batch_size(all_texts)
        try:
            t0 = time.time()
            embs = _run_with_timeout(
                lambda: model.encode(all_texts, normalize_embeddings=True, batch_size=bs),
                INFERENCE_TIMEOUT, "Similarity inference timeout",
            )
            elapsed = time.time() - t0
        except HTTPException:
            raise
        except Exception as e:
            log_error_ratelimited("embed_infer", "Similarity inference failed")
            raise HTTPException(500, f"Inference error: {e}")
    finally:
        _release_inference_slot()

    embs_a = embs[: len(texts_a)]
    embs_b = embs[len(texts_a):]

    if req.dimensions:
        dim = req.dimensions
        embs_a = embs_a[:, :dim]
        embs_b = embs_b[:, :dim]
        embs_a = embs_a / np.linalg.norm(embs_a, axis=1, keepdims=True)
        embs_b = embs_b / np.linalg.norm(embs_b, axis=1, keepdims=True)

    sim_matrix = (embs_a @ embs_b.T).tolist()
    if isinstance(req.input_a, str) and isinstance(req.input_b, str):
        sim_matrix = sim_matrix[0][0]

    return {
        "similarities": sim_matrix, "model": "bge-m3",
        "_timing_ms": round(elapsed * 1000),
        "_dimensions": req.dimensions or embs.shape[1],
    }


# --- Tokenize ---

class TokenizeRequest(BaseModel):
    input: Union[List[str], str]
    model: str = "bge-m3"


@app.post("/v1/tokenize", dependencies=[Depends(verify_api_key)])
def tokenize(req: TokenizeRequest):
    texts = req.input if isinstance(req.input, list) else [req.input]
    if not texts:
        raise HTTPException(400, "Input must contain at least one string")
    try:
        tokenizer = _get_tokenizer()
    except Exception as e:
        log_error_ratelimited("tokenizer", "Failed to get tokenizer")
        raise HTTPException(503, f"Tokenizer unavailable: {e}")
    results = []
    for t in texts:
        tokens = tokenizer.encode(t, add_special_tokens=False)
        results.append({"token_count": len(tokens), "max_tokens": tokenizer.model_max_length})
    return {"data": results, "model": "bge-m3"}


# --- Models list (OpenAI compatible) ---

@app.get("/v1/models")
def models():
    return {
        "object": "list",
        "data": [
            {
                "id": "bge-m3", "object": "model", "owned_by": "BAAI",
                "type": "embedding", "max_tokens": 8192,
                "backend": EMBED_BACKEND, "device": DEVICE,
                "ready": _embed_model is not None,
            },
            {
                "id": "bge-reranker-v2-m3", "object": "model", "owned_by": "BAAI",
                "type": "reranker", "max_tokens": 8192,
                "device": DEVICE,
                "ready": _rerank_model is not None,
            },
        ],
    }


# --- Admin endpoints ---

@app.post("/admin/offload/{model_name}", dependencies=[Depends(verify_api_key)])
def admin_offload(model_name: str):
    if model_name == "embed":
        _offload_embed()
        return {"offloaded": "bge-m3"}
    elif model_name == "rerank":
        _offload_rerank()
        return {"offloaded": "bge-reranker-v2-m3"}
    elif model_name == "all":
        _offload_embed()
        _offload_rerank()
        return {"offloaded": "all"}
    raise HTTPException(400, "model_name must be: embed, rerank, all")


@app.post("/admin/reload/{model_name}", dependencies=[Depends(verify_api_key)])
def admin_reload(model_name: str):
    if model_name == "embed":
        _force_reload_embed()
        return {"reloaded": "bge-m3"}
    elif model_name == "rerank":
        _force_reload_rerank()
        return {"reloaded": "bge-reranker-v2-m3"}
    elif model_name == "all":
        _force_reload_embed()
        _force_reload_rerank()
        return {"reloaded": "all"}
    raise HTTPException(400, "model_name must be: embed, rerank, all")


@app.post("/admin/cache/clear", dependencies=[Depends(verify_api_key)])
def admin_cache_clear():
    count = _cache_clear()
    return {"cleared": count}


# --- Config introspection ---

@app.get("/config")
def config():
    return {
        name: {
            "value": _cfg(name) if name != "API_KEY" else ("***" if API_KEY else ""),
            "description": desc,
        }
        for name, (_, desc) in _CONFIG.items()
    }


# --- Prometheus metrics ---

@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# --- Health + Readiness ---

@app.get("/health")
def health():
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    mem_mb = mem / (1024 * 1024) if sys.platform == "darwin" else mem / 1024
    return {
        "status": "draining" if _draining else "ok",
        "uptime_seconds": round(time.time() - _start_time),
        "embed_loaded": _embed_model is not None,
        "rerank_loaded": _rerank_model is not None,
        "embed_requests": _embed_count,
        "rerank_requests": _rerank_count,
        "inflight_requests": int(INFLIGHT._value._value),
        "ws_connections": int(WS_CONNECTIONS._value._value),
        "cache_size": len(_embed_cache),
        "cache_capacity": CACHE_SIZE,
        "memory_mb": round(mem_mb),
        "backend": EMBED_BACKEND,
        "device": DEVICE,
        "otel": _otel_enabled,
    }


@app.get("/ready")
def ready():
    if _draining:
        raise HTTPException(503, "Server is draining")
    if _embed_model is None and _rerank_model is None:
        raise HTTPException(503, "No models loaded")
    return {"ready": True}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8020"))
    reload = os.environ.get("RELOAD", "").lower() in ("1", "true", "yes")
    uvicorn.run(
        "embed-pro:app" if reload else app,
        host="0.0.0.0",
        port=port,
        reload=reload,
        reload_includes=["embed-pro.py"] if reload else None,
    )
