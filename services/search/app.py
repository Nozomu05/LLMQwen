"""Search Service — entry point for the RAG pipeline.

Receives: POST /query  {"query", "documents", "temperature"}
Flow:     language detection → translation → embedding → Chroma vector search
          → POST /rerank (Reranker Service) → POST /generate (LLM Service)
Returns:  {"answer", "model", "sources"}
"""
import os
import time
import warnings
from io import BytesIO
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

warnings.filterwarnings(
    "ignore",
    message=".*multilingual-e5-large now uses mean pooling.*",
    category=UserWarning,
)

load_dotenv(override=True)

app = FastAPI(title="Search Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_EMBEDDINGS_CACHE: dict = {}
_VECTORSTORE_CACHE: dict = {}

MAX_EXTRACT_CHARS = 20_000

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
    temperature: Optional[float] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_embeddings():
    from langchain_community.embeddings import FastEmbedEmbeddings
    from langchain_huggingface import HuggingFaceEmbeddings

    provider = os.environ["EMBEDDING_PROVIDER_ML"].lower()
    model = os.environ["EMBEDDING_MODEL_ML"]
    key = (provider, model)
    if key in _EMBEDDINGS_CACHE:
        return _EMBEDDINGS_CACHE[key]

    normalize = os.environ["EMBEDDING_NORMALIZE"].lower() == "true"
    if provider == "huggingface":
        emb = HuggingFaceEmbeddings(
            model_name=model,
            model_kwargs={"device": os.environ["EMBEDDING_DEVICE"]},
            encode_kwargs={"normalize_embeddings": normalize},
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

    chroma_dir = Path(os.environ["CHROMA_DIR"]).resolve()
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


def _extract_text_from_bytes(filename: str, file_bytes: bytes) -> str:
    """Convert an uploaded office/PDF file to plain text for use as a query document."""
    ext = Path(filename).suffix.lower()
    try:
        if ext == ".pdf":
            import fitz
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            text = "\n".join(page.get_text() for page in doc)
        elif ext == ".docx":
            from docx import Document as DocxDoc
            doc = DocxDoc(BytesIO(file_bytes))
            text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        elif ext == ".pptx":
            from pptx import Presentation
            prs = Presentation(BytesIO(file_bytes))
            parts = []
            for slide in prs.slides:
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text.strip():
                        parts.append(shape.text)
            text = "\n".join(parts)
        elif ext in (".xlsx", ".xls", ".ods"):
            import openpyxl
            wb = openpyxl.load_workbook(BytesIO(file_bytes), read_only=True, data_only=True)
            rows = []
            for ws in wb.worksheets:
                rows.append(f"[Sheet: {ws.title}]")
                for row in ws.iter_rows(values_only=True):
                    row_str = "\t".join(str(c) if c is not None else "" for c in row)
                    if row_str.strip():
                        rows.append(row_str)
            text = "\n".join(rows)
        else:
            text = file_bytes.decode("utf-8", errors="replace")
    except Exception as e:
        text = f"[Error extracting text from {filename}: {e}]"

    if len(text) > MAX_EXTRACT_CHARS:
        text = text[:MAX_EXTRACT_CHARS] + f"\n\n[... truncated at {MAX_EXTRACT_CHARS:,} characters]"
    return text


def _vector_search(retrieval_query: str) -> list[dict]:
    """Embed the retrieval query and search the Chroma index."""
    k_total = int(os.environ["RETRIEVAL_CHUNKS"])
    vs = _get_vectorstore()

    # Instruct-style models need a prefix on the query side
    embedding_model = os.environ["EMBEDDING_MODEL_ML"]
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


@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    """Extract plain text from an uploaded PDF/DOCX/PPTX/XLSX file for use as a query document."""
    file_bytes = await file.read()
    text = _extract_text_from_bytes(file.filename or "document", file_bytes)
    return {"filename": file.filename, "text": text}


@app.post("/query")
def query(req: QueryRequest):
    """Full RAG entry point: search → rerank → generate."""
    t_start = time.time()

    reranker_url = os.environ["RERANKER_SERVICE_URL"]
    llm_url = os.environ["LLM_SERVICE_URL"]

    lang_code, lang_hint = _detect_language(req.query)

    # Determine the retrieval query and the reranking query
    if os.environ["USE_HYDE"].lower() == "true":
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
        "temperature": req.temperature,
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
