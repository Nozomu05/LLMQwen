import os
import re
import sys
import warnings
from pathlib import Path
from typing import List

# fastembed 0.5.2+ changed intfloat/multilingual-e5-large from CLS to mean pooling.
# Both ingest.py and query.py use the same fastembed version, so the behaviour is
# consistent. Suppress the warning since it is informational only.
warnings.filterwarnings(
    "ignore",
    message=".*multilingual-e5-large now uses mean pooling.*",
    category=UserWarning,
)

from dotenv import load_dotenv
load_dotenv(override=True)

from langchain_community.embeddings import FastEmbedEmbeddings  # type: ignore
from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
from langchain_chroma import Chroma  # type: ignore
from langchain_core.documents import Document  # type: ignore
from langchain_core.prompts import ChatPromptTemplate  # type: ignore
from langchain_openai import ChatOpenAI  # type: ignore
from sentence_transformers import CrossEncoder
import time


class Reranker:

    def __init__(self, model_name: str, top_n: int):
        self.model_name = model_name
        self.top_n = top_n
        print(f"Loading reranker: {model_name}...")
        cache_folder = os.getenv("HF_CACHE_DIR") or None
        self.reranker = CrossEncoder(model_name, device=os.getenv("RERANKER_DEVICE", "cpu"), cache_folder=cache_folder)

    def compress_documents(self, documents: List[Document], query: str) -> List[Document]:
        if not documents:
            return documents
        pairs = [[query, doc.page_content] for doc in documents]
        reranker_batch_size = int(os.getenv("RERANKER_BATCH_SIZE", "16"))
        scores = self.reranker.predict(pairs, batch_size=reranker_batch_size, show_progress_bar=False)
        doc_score_pairs = list(zip(documents, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in doc_score_pairs[:self.top_n]]


def _clean_chunk_text(text: str) -> str:
    # Remove any non-space run ≥18 chars — always an extraction artifact in
    # standardisation docs (covers pure-alpha, hyphenated, and mixed tokens).
    # 18 is the lowest safe threshold: legitimate hyphenated terms top out at
    # ~13 chars (e.g. "Dense-Dynamic"), while missing-space artifacts like
    # "onsStaticsequences" (18) and "Dense-Dynamicsequences" (22) are caught.
    # Apostrophes (straight and curly) are excluded so French contractions like
    # "d'expérimentations" (18 chars incl. apostrophe) are not wrongly removed.
    text = re.sub(r"[^\s'‘’]{18,}", '', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()


def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(
        f"[Source: {d.metadata.get('source', 'unknown')}]\n{_clean_chunk_text(d.page_content)}"
        for d in docs
    )


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


def detect_language(text: str) -> tuple[str, str]:
    """Return (index_lang, response_hint) where response_hint is an explicit
    instruction for the model to respond in the detected language (empty for English)."""
    try:
        from langdetect import detect
        lang_code = detect(text)
        if lang_code == "en":
            return "en", ""
        hint = _LANG_INSTRUCTIONS.get(lang_code, f"Respond in the same language as the question ({lang_code}).")
        return "other", f" ({hint})"
    except Exception:
        return "other", ""


def get_embeddings(lang: str = "en"):
    if lang == "en":
        embedding_provider = os.getenv("EMBEDDING_PROVIDER_EN", "fastembed").lower()
        embedding_model = os.getenv("EMBEDDING_MODEL_EN", "BAAI/bge-base-en-v1.5")
    else:
        embedding_provider = os.getenv("EMBEDDING_PROVIDER_ML", "fastembed").lower()
        embedding_model = os.getenv("EMBEDDING_MODEL_ML", "intfloat/multilingual-e5-large")

    cache_key = (embedding_provider, embedding_model)
    if cache_key in _EMBEDDINGS_CACHE:
        return _EMBEDDINGS_CACHE[cache_key]

    if embedding_provider == "huggingface":
        emb = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    else:
        # Force CPU execution so GPU memory stays free for vLLM.
        # fastembed uses ONNX Runtime; CPUExecutionProvider prevents it from
        # reserving CUDA memory that vLLM needs.
        emb = FastEmbedEmbeddings(
            model_name=embedding_model,
            max_length=512,
            additional_kwargs={"providers": ["CPUExecutionProvider"]},
        )

    _EMBEDDINGS_CACHE[cache_key] = emb
    return emb


_RERANKER_CACHE: dict = {}
_EMBEDDINGS_CACHE: dict = {}
_VECTORSTORE_CACHE: dict = {}
_LLM_CACHE: dict = {}


def generate_hyde_query(query: str, llm) -> str:
    """Generate a hypothetical answer (HyDE) to use as the retrieval query.

    The fake answer is never shown to the user — it is only embedded and used
    for vector similarity search so that results-table chunks (BD-rate values,
    sequence names, percentages) score higher than with the original vague query.
    The original query is still used for reranking and generation.
    """
    from langchain_core.messages import SystemMessage, HumanMessage
    try:
        result = llm.invoke([
            SystemMessage(content="You are a technical expert in video compression and MPEG standardization."),
            HumanMessage(content=(
                "Write exactly 2-3 sentences IN ENGLISH answering this question. "
                "Include specific metrics (BD-rate, PSNR), sequence names, and method names "
                "that would appear in MPEG documents. Make a reasonable technical guess.\n\n"
                f"Question: {query}"
            )),
        ])
        hypo = result.content
        sentences = re.split(r'(?<=[.!?])\s+', hypo.strip())
        return " ".join(sentences[:3])
    except Exception:
        return query


_RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a precise technical expert in video compression, 3D content coding, and international standards (MPEG, ISO/IEC). Answer strictly from the provided context.

Give a detailed, well-structured answer that covers all relevant findings: methodology, configurations, measurements (BD-rate, PSNR, dB gains, percentages), sequence names, method names, and session/meeting references. Group related findings with sub-headings when it helps clarity. End with 1–2 sentences summarising the key takeaways.

Rules:
- Only use information present in the context. Never fabricate data or numbers.
- Copy measurements verbatim — do not round or estimate.
- Omit [m12345], [1], [LuYu] and similar reference markers.
- Use acronyms exactly as written (VSS-PCC, TMAP, RAHT, V3C, SEI, …).
- Skip any garbled or concatenated text fragments from the context.
- Do not add follow-up questions or new Q:/A: turns after the answer.

Respond in the same language as the question."""),
    ("human", """Context:
{context}

Question: {question}

Answer:""")
])


def _build_answer_header(query: str, docs: List[Document]) -> str:
    sources = list(dict.fromkeys(d.metadata.get("source", "unknown") for d in docs))

    session_pat = re.compile(r'(?:session|meeting)\s+(\d{2,3})', re.IGNORECASE)
    seen_sessions: set = set()
    sessions = []
    for doc in docs:
        for m in session_pat.finditer(doc.page_content):
            n = m.group(1)
            if n not in seen_sessions:
                seen_sessions.add(n)
                sessions.append(n)

    sources_str = ", ".join(f"`{s}`" for s in sources)
    session_str = f"  \n**Sessions referenced:** {', '.join(sessions)}" if sessions else ""

    return (
        f"**Question:** {query}  \n"
        f"**Sources consulted:** {sources_str}{session_str}  \n\n"
        f"**Findings:**\n"
    )


def retrieve_from_all_indexes(query: str, verbose: bool = False, retrieval_query: str = None) -> List[Document]:
    """Query the unified Chroma index and rerank results.

    A single multilingual index (intfloat/multilingual-e5-large) and reranker
    (bge-reranker-v2-m3) are used for all queries so that English and French
    questions retrieve the same candidate chunks and receive consistent answers.

    retrieval_query: optional override for the embedding step (used by HyDE).
    If provided, this text is embedded instead of the original query.
    The original query is always used for reranking and answer generation.
    """
    k_total = int(os.getenv("RETRIEVAL_CHUNKS", "300"))
    use_reranking = os.getenv("USE_RERANKING", "true").lower() == "true"
    top_n = int(os.getenv("TOP_N_RERANK", "8"))

    chroma_dir = Path(os.getenv("CHROMA_DIR", "storage/chroma")).resolve()
    if not chroma_dir.exists():
        raise FileNotFoundError(
            f"No vector store found at {chroma_dir}. Run ingestion first: python rag/ingest.py"
        )

    if verbose:
        print(f"Loading Chroma from: {chroma_dir}")

    start = time.time()
    embeddings = get_embeddings("other")  # always use the multilingual model
    vs_key = str(chroma_dir)
    if vs_key not in _VECTORSTORE_CACHE:
        _VECTORSTORE_CACHE[vs_key] = Chroma(persist_directory=str(chroma_dir), embedding_function=embeddings)
    vs = _VECTORSTORE_CACHE[vs_key]

    if retrieval_query:
        # HyDE: caller already prepared the retrieval query (hypothetical answer).
        # Save the raw hypothetical for reranking (before adding the instruct prefix).
        rerank_query = retrieval_query
        embedding_model = os.getenv("EMBEDDING_MODEL_ML", "")
        if "instruct" in embedding_model.lower():
            retrieval_query = f"Instruct: Retrieve relevant passages that answer the question\nQuery: {retrieval_query}"
        if verbose:
            print(f"HyDE retrieval query: {retrieval_query[:120]}...")
    else:
        # Translate non-English queries to English before embedding so that French/German/etc.
        # questions produce the same embedding vector as their English equivalents.
        # All indexed documents are in English, so an English query always gives the best
        # cosine similarity match regardless of the original question language.
        query_lang, _ = detect_language(query)
        if query_lang != "en":
            try:
                from deep_translator import GoogleTranslator
                retrieval_query = GoogleTranslator(source="auto", target="en").translate(query)
                if verbose:
                    print(f"Query translated for retrieval: {retrieval_query}")
            except Exception:
                retrieval_query = query
        else:
            retrieval_query = query

        # intfloat/multilingual-e5-large-instruct requires an instruction prefix on the query.
        embedding_model = os.getenv("EMBEDDING_MODEL_ML", "")
        if "instruct" in embedding_model.lower():
            retrieval_query = f"Instruct: Retrieve relevant passages that answer the question\nQuery: {retrieval_query}"

        rerank_query = query

    docs = vs.as_retriever(search_kwargs={"k": k_total}).invoke(retrieval_query)

    if verbose:
        print(f"Loaded in {time.time() - start:.2f}s")

    reranker_model = os.getenv("RERANKER_MODEL_ML", "BAAI/bge-reranker-v2-m3")
    if use_reranking and docs:
        if verbose:
            print(f"Using reranking to select top {top_n} from {len(docs)} chunks...")
        compressor = _get_reranker(reranker_model, top_n)
        docs = compressor.compress_documents(docs, rerank_query)
        if verbose:
            print(f"Reranked to {len(docs)} most relevant chunks")
    else:
        docs = docs[:top_n]

    return docs


def _get_reranker(model_name: str, top_n: int) -> Reranker:
    key = (model_name, top_n)
    if key not in _RERANKER_CACHE:
        _RERANKER_CACHE[key] = Reranker(model_name=model_name, top_n=top_n)
    return _RERANKER_CACHE[key]


def warmup():
    """Preload the reranker and embedding model at startup so the first request isn't slow."""
    reranker_model = os.getenv("RERANKER_MODEL_ML", "BAAI/bge-reranker-v2-m3")
    top_n = int(os.getenv("TOP_N_RERANK", "8"))
    _get_reranker(reranker_model, top_n)
    get_embeddings("other")
    print("Warmup complete.")


def get_llm():
    model_name = os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-14B-Instruct-AWQ")
    base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000")
    temperature = float(os.getenv("TEMPERATURE", "0.1"))
    max_tokens = int(os.getenv("MAX_NEW_TOKENS", "2048"))
    cache_key = (model_name, base_url, temperature, max_tokens)
    if cache_key not in _LLM_CACHE:
        _LLM_CACHE[cache_key] = ChatOpenAI(
            base_url=f"{base_url}/v1",
            api_key="dummy",
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=True,
        )
        print(f"Using vLLM server: {base_url}")
    return _LLM_CACHE[cache_key], model_name


def _truncate_context(context_text: str, max_new_tokens: int, max_context_length: int) -> str:
    # ~800 tokens reserved for system prompt + question overhead; 1 token ≈ 4 chars
    overhead_tokens = int(os.getenv("PROMPT_OVERHEAD_TOKENS", "800"))
    available_tokens = max_context_length - max_new_tokens - overhead_tokens
    if available_tokens <= 0:
        available_tokens = 1000
    max_chars = available_tokens * 4
    if len(context_text) <= max_chars:
        return context_text
    truncated = context_text[:max_chars]
    last_para = truncated.rfind('\n\n')
    if last_para > max_chars * 0.8:
        truncated = truncated[:last_para]
    return truncated


def run_query_stream(query: str, extra_docs=None):
    """Generator that yields SSE event dicts for streaming responses.

    First yield is always {"type": "meta", ...} — it fires after retrieval
    completes so callers can commit to streaming only once retrieval succeeds.
    Subsequent yields are {"type": "chunk", "text": ...} and a final
    {"type": "done"}.
    """
    _, lang_hint = detect_language(query)
    llm, model_name = get_llm()

    hyde_query = None
    if os.getenv("USE_HYDE", "false").lower() == "true":
        hyde_query = generate_hyde_query(query, llm)
    docs = retrieve_from_all_indexes(query, retrieval_query=hyde_query)

    max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", "2048"))
    max_context_length = int(os.getenv("VLLM_MAX_CONTEXT", "8192"))

    context_text = format_docs(docs)
    if extra_docs:
        uploaded = "\n\n".join(
            f"[Uploaded document: {name}]\n{text}"
            for name, text in extra_docs
        )
        context_text = "## User-uploaded documents\n\n" + uploaded + "\n\n## Retrieved context\n\n" + context_text

    sources = list(dict.fromkeys(d.metadata.get("source", "unknown") for d in docs))
    if extra_docs:
        uploaded_names = [name for name, _ in extra_docs]
        sources = uploaded_names + [s for s in sources if s not in uploaded_names]

    context_text = _truncate_context(context_text, max_new_tokens, max_context_length)
    header = _build_answer_header(query, docs)

    yield {"type": "meta", "model": model_name, "sources": sources, "header": header}

    chain = _RAG_PROMPT | llm
    accumulated = ""
    _SUMMARY_MARKERS = (
        "**Summary**", "**Résumé**", "**Zusammenfassung**",
        "**Conclusión**", "**Conclusione**", "**Conclusão**",
        "**Samenvatting**",
    )

    for chunk in chain.stream({"question": query + lang_hint, "context": context_text}):
        text = chunk.content
        if not text:
            continue
        text = re.sub(r'\[m\d+\]|\[\d+\]|\[[A-Za-z][A-Za-z0-9]{1,}\]', '', text)
        if re.search(r"[^\s''']{18,}", text):
            break
        accumulated += text
        yield {"type": "chunk", "text": text}

        for _sm in _SUMMARY_MARKERS:
            _sp = accumulated.find(_sm)
            if _sp >= 0:
                _after = accumulated[_sp + 12:]
                if _after.find('\n\n') >= 50:
                    yield {"type": "done"}
                    return

        last_boundary = max(
            accumulated.rfind('.'), accumulated.rfind('!'),
            accumulated.rfind('?'), accumulated.rfind('\n')
        )
        if len(accumulated) - last_boundary > 400:
            break

    yield {"type": "done"}


def run_query_complete(query: str, extra_docs=None) -> tuple[str, str, list[str]]:
    """Run a complete RAG query and return (answer, model_name, sources)."""
    _, lang_hint = detect_language(query)
    llm, model_name = get_llm()

    hyde_query = None
    if os.getenv("USE_HYDE", "false").lower() == "true":
        hyde_query = generate_hyde_query(query, llm)
    docs = retrieve_from_all_indexes(query, retrieval_query=hyde_query)
    chain = _RAG_PROMPT | llm

    max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", "2048"))
    max_context_length = int(os.getenv("VLLM_MAX_CONTEXT", "8192"))

    context_text = format_docs(docs)
    if extra_docs:
        uploaded = "\n\n".join(
            f"[Uploaded document: {name}]\n{text}"
            for name, text in extra_docs
        )
        context_text = "## User-uploaded documents\n\n" + uploaded + "\n\n## Retrieved context\n\n" + context_text

    sources = list(dict.fromkeys(d.metadata.get("source", "unknown") for d in docs))
    if extra_docs:
        uploaded_names = [name for name, _ in extra_docs]
        sources = uploaded_names + [s for s in sources if s not in uploaded_names]

    context_text = _truncate_context(context_text, max_new_tokens, max_context_length)

    answer = ""
    for chunk in chain.stream({"question": query + lang_hint, "context": context_text}):
        answer += chunk.content

    answer = re.sub(r'\[m\d+\]|\[\d+\]|\[[A-Za-z][A-Za-z0-9]{1,}\]', '', answer)
    answer = re.sub(r'  +', ' ', answer).strip()

    _SUMMARY_MARKERS = (
        "**Summary**", "**Résumé**", "**Zusammenfassung**",
        "**Conclusión**", "**Conclusione**", "**Conclusão**",
        "**Samenvatting**",
    )
    for _sm in _SUMMARY_MARKERS:
        _sp = answer.find(_sm)
        if _sp >= 0:
            _after = answer[_sp + 12:]
            _blank = _after.find('\n\n')
            if _blank >= 50:
                answer = answer[:_sp + 12 + _blank].rstrip()
            break

    last_end = max(answer.rfind('.'), answer.rfind('!'), answer.rfind('?'))
    last_newline = answer.rfind('\n')
    if last_newline > last_end and re.search(r'[A-Za-z0-9%]$', answer[last_newline:].rstrip()):
        last_end = len(answer.rstrip()) - 1
    if last_end > 0:
        answer = answer[:last_end + 1].strip()

    header = _build_answer_header(query, docs)
    return header + answer, model_name, sources


def main() -> None:
    query = " ".join(sys.argv[1:]).strip()
    if not query:
        print('Usage: python rag/query.py "<your question>"')
        sys.exit(1)

    _, lang_hint = detect_language(query)
    llm, model_name = get_llm()

    hyde_query = None
    if os.getenv("USE_HYDE", "false").lower() == "true":
        hyde_query = generate_hyde_query(query, llm)
    try:
        docs = retrieve_from_all_indexes(query, verbose=True, retrieval_query=hyde_query)
    except FileNotFoundError as e:
        print(e)
        print("Run ingestion first: python rag/ingest.py")
        sys.exit(1)

    chain = _RAG_PROMPT | llm
    max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", "2048"))
    max_context_length = int(os.getenv("VLLM_MAX_CONTEXT", "8192"))
    context_text = _truncate_context(format_docs(docs), max_new_tokens, max_context_length)
    sources = list(dict.fromkeys(d.metadata.get("source", "unknown") for d in docs))

    print(f"Querying model: {model_name}\n")
    header = _build_answer_header(query, docs)
    print("=== Answer ===\n")
    print(header, end="", flush=True)
    query_start = time.time()

    accumulated = ""
    _stop_patterns = ["Human:", "\nQ:", "\nA:", "\nQuestion:"]
    _SUMMARY_MARKERS = (
        "**Summary**", "**Résumé**", "**Zusammenfassung**",
        "**Conclusión**", "**Conclusione**", "**Conclusão**",
        "**Samenvatting**", "**결론**", "**总结**", "**まとめ**",
    )
    for chunk in chain.stream({"question": query + lang_hint, "context": context_text}):
        text = chunk.content
        text = re.sub(r'\[m\d+\]|\[\d+\]|\[[A-Za-z][A-Za-z0-9]{1,}\]', '', text)
        accumulated += text
        print(text, end="", flush=True)

        if any(p in accumulated for p in _stop_patterns):
            break
        if accumulated.count("\n---") >= 2:
            break
        _summary_pos = next(
            (accumulated.find(m) for m in _SUMMARY_MARKERS if m in accumulated), -1
        )
        if _summary_pos >= 0:
            _after_summary = accumulated[_summary_pos + 12:]
            _blank = _after_summary.find('\n\n')
            if _blank >= 50:
                break
        if re.search(r"[^\s'‘’]{18,}", text):
            break
        last_boundary = max(
            accumulated.rfind('.'), accumulated.rfind('!'),
            accumulated.rfind('?'), accumulated.rfind('\n')
        )
        if len(accumulated) - last_boundary > 400:
            break

    print(f"\n\n[Query completed in {time.time() - query_start:.2f}s]")

    print("\n\n=== Sources ===")
    seen = set()
    i = 1
    for d in docs:
        src = d.metadata.get("source", "unknown")
        if src not in seen:
            print(f"{i}. {src}")
            seen.add(src)
            i += 1


if __name__ == "__main__":
    main()
