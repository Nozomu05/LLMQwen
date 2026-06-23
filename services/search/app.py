"""Search Service — entry point for the RAG pipeline.

Receives: POST /query  {"query", "documents"}
Flow:     language detection → translation → embedding → Chroma vector search
          → POST /rerank (Reranker Service) → POST /generate (LLM Service)
Returns:  {"answer", "model", "sources"}
"""
import os
import time
import warnings
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

warnings.filterwarnings(
    "ignore",
    message=".*multilingual-e5-large now uses mean pooling.*",
    category=UserWarning,
)

load_dotenv(override=True)

app = FastAPI(title="Search Service")

_EMBEDDINGS_CACHE: dict = {}
_VECTORSTORE_CACHE: dict = {}

_LANG_INSTRUCTIONS = {
    "fr": "Réponds en français.",
    "de": "Antworte auf Deutsch.",
    "es": "Responde en español.",
    "it": "Rispondi in italiano.",
    "pt": "Responda em português.",
    "nl": "Antwoord in het Nederlands.",
    "zh": "请用中文回答。",
    "ja": "日本語で答えてください。",
    "ko": "한국어로 답하세요.",
    "ar": "أجب باللغة العربية.",
}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class Document(BaseModel):
    filename: str
    text: str


class QueryRequest(BaseModel):
    query: str
    documents: Optional[list[Document]] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_embeddings():
    from langchain_community.embeddings import FastEmbedEmbeddings
    from langchain_huggingface import HuggingFaceEmbeddings

    provider = os.getenv("EMBEDDING_PROVIDER_ML", "fastembed").lower()
    model = os.getenv("EMBEDDING_MODEL_ML", "intfloat/multilingual-e5-large")
    key = (provider, model)
    if key in _EMBEDDINGS_CACHE:
        return _EMBEDDINGS_CACHE[key]

    if provider == "huggingface":
        emb = HuggingFaceEmbeddings(
            model_name=model,
            model_kwargs={"device": os.getenv("EMBEDDING_DEVICE", "cpu")},
            encode_kwargs={"normalize_embeddings": True},
        )
    else:
        emb = FastEmbedEmbeddings(
            model_name=model,
            max_length=512,
            additional_kwargs={"providers": ["CPUExecutionProvider"]},
        )
    _EMBEDDINGS_CACHE[key] = emb
    return emb


def _detect_language(text: str) -> tuple[str, str]:
    """Return (lang_code, response_hint)."""
    try:
        from langdetect import detect
        lang_code = detect(text)
        if lang_code == "en":
            return "en", ""
        hint = _LANG_INSTRUCTIONS.get(
            lang_code,
            f"Respond in the same language as the question ({lang_code}).",
        )
        return lang_code, f" ({hint})"
    except Exception:
        return "en", ""


def _translate_to_english(text: str) -> str:
    try:
        from deep_translator import GoogleTranslator
        return GoogleTranslator(source="auto", target="en").translate(text)
    except Exception:
        return text


def _get_vectorstore():
    from langchain_chroma import Chroma

    chroma_dir = Path(os.getenv("CHROMA_DIR", "storage/chroma")).resolve()
    if not chroma_dir.exists():
        raise FileNotFoundError(
            f"No vector store at {chroma_dir}. Run ingestion first: python rag/ingest.py"
        )
    vs_key = str(chroma_dir)
    if vs_key not in _VECTORSTORE_CACHE:
        _VECTORSTORE_CACHE[vs_key] = Chroma(
            persist_directory=str(chroma_dir),
            embedding_function=_get_embeddings(),
        )
    return _VECTORSTORE_CACHE[vs_key]


def _vector_search(retrieval_query: str) -> list[dict]:
    """Embed the retrieval query and search the Chroma index."""
    k_total = int(os.getenv("RETRIEVAL_CHUNKS", "300"))
    vs = _get_vectorstore()

    # Instruct-style models need a prefix on the query side
    embedding_model = os.getenv("EMBEDDING_MODEL_ML", "")
    if "instruct" in embedding_model.lower():
        retrieval_query = (
            f"Instruct: Retrieve relevant passages that answer the question\n"
            f"Query: {retrieval_query}"
        )

    docs = vs.as_retriever(search_kwargs={"k": k_total}).invoke(retrieval_query)
    return [
        {"content": d.page_content, "source": d.metadata.get("source", "unknown")}
        for d in docs
    ]


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
def warmup():
    _get_embeddings()
    try:
        _get_vectorstore()
    except FileNotFoundError:
        pass  # chroma index not ingested yet


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query")
def query(req: QueryRequest):
    """Full RAG entry point: search → rerank → generate."""
    t_start = time.time()

    reranker_url = os.getenv("RERANKER_SERVICE_URL", "http://localhost:8011")
    llm_url = os.getenv("LLM_SERVICE_URL", "http://localhost:8012")

    lang_code, lang_hint = _detect_language(req.query)

    # Determine the retrieval query and the reranking query
    if os.getenv("USE_HYDE", "false").lower() == "true":
        # HyDE: generate a hypothetical answer to use as the retrieval query.
        # The hypothesis is also used as the rerank query (matches original behaviour).
        try:
            resp = requests.post(
                f"{llm_url}/hyde",
                json={"query": req.query},
                timeout=60,
            )
            resp.raise_for_status()
            hypothesis = resp.json().get("hypothesis", req.query)
        except Exception:
            hypothesis = req.query
        retrieval_query = hypothesis
        rerank_query = hypothesis
    else:
        # Translate to English for better embedding similarity (all docs are in English).
        # Use the original (possibly non-English) query for reranking — the reranker is multilingual.
        if lang_code != "en":
            retrieval_query = _translate_to_english(req.query)
        else:
            retrieval_query = req.query
        rerank_query = req.query

    try:
        t_search = time.time()
        chunks = _vector_search(retrieval_query)
        print(f"[search] vector search: {time.time() - t_search:.2f}s — {len(chunks)} chunks retrieved")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    extra_docs = None
    if req.documents:
        extra_docs = [{"filename": d.filename, "text": d.text} for d in req.documents]

    rerank_payload = {
        "query": req.query,
        "chunks": chunks,
        "rerank_query": rerank_query,
        "lang_hint": lang_hint,
        "extra_docs": extra_docs,
    }

    def proxy_stream():
        t_reranker = time.time()
        try:
            with requests.post(
                f"{reranker_url}/rerank",
                json=rerank_payload,
                stream=True,
                timeout=180,
            ) as resp:
                if resp.status_code != 200:
                    import json as _json
                    yield f"data: {_json.dumps({'type': 'error', 'error': f'Reranker service returned HTTP {resp.status_code}'})}\n\n".encode()
                else:
                    for raw in resp.iter_content(chunk_size=None):
                        yield raw
        except Exception as exc:
            import json as _json
            yield f"data: {_json.dumps({'type': 'error', 'error': f'Reranker service error: {exc}'})}\n\n".encode()
        print(f"[search] rerank+generate: {time.time() - t_reranker:.2f}s")
        print(f"[search] total: {time.time() - t_start:.2f}s")

    return StreamingResponse(proxy_stream(), media_type="text/event-stream")
