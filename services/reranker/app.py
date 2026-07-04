"""Reranker Service — CrossEncoder scoring, then forwards to LLM service.

Receives: POST /rerank  {"query", "chunks", "rerank_query", "lang_hint", "extra_docs", "temperature"}
Returns:  {"answer", "model", "sources"}  (passed through from LLM service)
"""
import os
import time
from typing import Optional

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sentence_transformers import CrossEncoder

load_dotenv(override=True)

app = FastAPI(title="Reranker Service")

_RERANKER_CACHE: dict = {}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class Chunk(BaseModel):
    content: str
    source: str


class RerankRequest(BaseModel):
    query: str
    chunks: list[Chunk]
    rerank_query: str = ""   # query used for scoring; falls back to query if empty
    lang_hint: str = ""
    extra_docs: Optional[list[dict]] = None
    temperature: Optional[float] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_reranker() -> CrossEncoder:
    model_name = os.environ["RERANKER_MODEL_ML"]
    if model_name not in _RERANKER_CACHE:
        _RERANKER_CACHE[model_name] = CrossEncoder(
            model_name,
            device=os.environ["RERANKER_DEVICE"],
            cache_folder=os.environ["HF_CACHE_DIR"],
            max_length=int(os.environ["RERANKER_MAX_LENGTH"]),
        )
    return _RERANKER_CACHE[model_name]


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
def warmup():
    _get_reranker()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/rerank")
def rerank(req: RerankRequest):
    """Score all chunks with the CrossEncoder, keep the top N, then call the LLM service."""
    t_start = time.time()

    top_n = int(os.environ["TOP_N_RERANK"])
    llm_url = os.environ["LLM_SERVICE_URL"]

    chunks = req.chunks
    scoring_query = req.rerank_query or req.query

    if chunks:
        reranker = _get_reranker()
        pairs = [[scoring_query, c.content] for c in chunks]
        batch_size = int(os.environ["RERANKER_BATCH_SIZE"])
        scores = reranker.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        chunks = [c for c, _ in ranked[:top_n]]

    t_rerank_done = time.time()
    print(f"[reranker] scoring {len(req.chunks)} chunks → top {len(chunks)}: {t_rerank_done - t_start:.2f}s")

    chunks_payload = [{"content": c.content, "source": c.source} for c in chunks]

    def proxy_stream():
        t_llm = time.time()
        try:
            with requests.post(
                f"{llm_url}/generate",
                json={
                    "query": req.query,
                    "chunks": chunks_payload,
                    "lang_hint": req.lang_hint,
                    "extra_docs": req.extra_docs,
                    "temperature": req.temperature,
                },
                stream=True,
                timeout=120,
            ) as resp:
                if resp.status_code != 200:
                    import json as _json
                    yield f"data: {_json.dumps({'type': 'error', 'error': f'LLM service returned HTTP {resp.status_code}'})}\n\n".encode()
                else:
                    for raw in resp.iter_content(chunk_size=None):
                        yield raw
        except Exception as exc:
            import json as _json
            yield f"data: {_json.dumps({'type': 'error', 'error': f'LLM service error: {exc}'})}\n\n".encode()
        print(f"[reranker] llm stream done: {time.time() - t_llm:.2f}s")
        print(f"[reranker] total: {time.time() - t_start:.2f}s")

    return StreamingResponse(proxy_stream(), media_type="text/event-stream")
