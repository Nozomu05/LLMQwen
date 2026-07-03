"""LLM Service — prompt assembly and answer generation via vLLM.

Receives: POST /generate  {"query", "chunks", "lang_hint", "extra_docs"}
Returns:  {"answer", "model", "sources"}

Receives: POST /hyde      {"query"}
Returns:  {"hypothesis"}
"""
import json
import os
import re
import time
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

load_dotenv(override=True)

app = FastAPI(title="LLM Service")

_LLM_CACHE: dict = {}
_TOKENIZER_CACHE: dict = {}

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

Question: {question} /no_think

Answer:"""),
])

_SUMMARY_MARKERS = (
    "**Summary**", "**Résumé**", "**Zusammenfassung**",
    "**Conclusión**", "**Conclusione**", "**Conclusão**",
    "**Samenvatting**", "**결론**", "**总结**", "**まとめ**",
)

_STOP_PATTERNS = ["Human:", "\nQ:", "\nA:", "\nQuestion:"]


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class Chunk(BaseModel):
    content: str
    source: str


class GenerateRequest(BaseModel):
    query: str
    chunks: list[Chunk]
    lang_hint: str = ""
    extra_docs: Optional[list[dict]] = None


class HydeRequest(BaseModel):
    query: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_llm():
    model_name = os.environ["VLLM_MODEL"]
    base_url = os.environ["VLLM_BASE_URL"]
    temperature = float(os.environ["TEMPERATURE"])
    max_tokens = int(os.environ["MAX_NEW_TOKENS"])
    top_p = float(os.environ["TOP_P"])
    frequency_penalty = float(os.environ["FREQUENCY_PENALTY"])
    key = (model_name, base_url, temperature, max_tokens, top_p, frequency_penalty)
    if key not in _LLM_CACHE:
        _LLM_CACHE[key] = ChatOpenAI(
            base_url=f"{base_url}/v1",
            api_key="dummy",
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            streaming=True,
        )
    return _LLM_CACHE[key], model_name


def _clean_chunk_text(text: str) -> str:
    text = re.sub(r"[^\s''']{18,}", "", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _format_chunks(chunks: list[Chunk]) -> str:
    return "\n\n".join(
        f"[Source: {c.source}]\n{_clean_chunk_text(c.content)}"
        for c in chunks
    )


def _get_tokenizer():
    model_name = os.environ["VLLM_MODEL"]
    if model_name not in _TOKENIZER_CACHE:
        from transformers import AutoTokenizer
        _TOKENIZER_CACHE[model_name] = AutoTokenizer.from_pretrained(model_name)
    return _TOKENIZER_CACHE[model_name]


def _count_tokens(text: str) -> int:
    return len(_get_tokenizer().encode(text, add_special_tokens=False))


def _truncate_context(context_text: str, question: str, max_new_tokens: int, max_context_length: int) -> str:
    """Truncate context_text so the full prompt fits the model's context window.

    Uses the real tokenizer instead of a chars-per-token estimate — with
    document uploads pushing prompts close to the limit, an approximate
    ratio can undercount and the request gets rejected by vLLM entirely.
    """
    # Small safety margin for chat-template special tokens (role markers, etc.)
    template_overhead_tokens = int(os.environ["PROMPT_OVERHEAD_TOKENS"])
    system_tokens = _count_tokens(_RAG_PROMPT.messages[0].prompt.template)
    fixed_tokens = system_tokens + _count_tokens(question) + template_overhead_tokens

    available_tokens = max_context_length - max_new_tokens - fixed_tokens
    if available_tokens <= 0:
        available_tokens = 500

    tokenizer = _get_tokenizer()
    context_token_ids = tokenizer.encode(context_text, add_special_tokens=False)
    if len(context_token_ids) <= available_tokens:
        return context_text

    truncated = tokenizer.decode(context_token_ids[:available_tokens])
    last_para = truncated.rfind("\n\n")
    if last_para > len(truncated) * 0.8:
        truncated = truncated[:last_para]
    return truncated


def _build_answer_header(query: str, chunks: list[Chunk]) -> str:
    sources = list(dict.fromkeys(c.source for c in chunks))
    session_pat = re.compile(r"(?:session|meeting)\s+(\d{2,3})", re.IGNORECASE)
    seen: set = set()
    sessions = []
    for c in chunks:
        for m in session_pat.finditer(c.content):
            n = m.group(1)
            if n not in seen:
                seen.add(n)
                sessions.append(n)
    sources_str = ", ".join(f"`{s}`" for s in sources)
    session_str = f"  \n**Sessions referenced:** {', '.join(sessions)}" if sessions else ""
    return (
        f"**Question:** {query}  \n"
        f"**Sources consulted:** {sources_str}{session_str}  \n\n"
        f"**Findings:**\n"
    )


def _clean_answer(answer: str) -> str:
    answer = re.sub(r"\[m\d+\]|\[\d+\]|\[[A-Za-z][A-Za-z0-9]{1,}\]", "", answer)
    answer = re.sub(r"  +", " ", answer).strip()
    for marker in _SUMMARY_MARKERS:
        pos = answer.find(marker)
        if pos >= 0:
            after = answer[pos + 12:]
            blank = after.find("\n\n")
            if blank >= 50:
                answer = answer[: pos + 12 + blank].rstrip()
            break
    last_end = max(answer.rfind("."), answer.rfind("!"), answer.rfind("?"))
    last_newline = answer.rfind("\n")
    if last_newline > last_end and re.search(r"[A-Za-z0-9%]$", answer[last_newline:].rstrip()):
        last_end = len(answer.rstrip()) - 1
    if last_end > 0:
        answer = answer[: last_end + 1].strip()
    return answer


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.on_event("startup")
def warmup():
    """Send a minimal dummy request to vLLM so CUDA kernels are compiled before the first real query."""
    try:
        llm, _ = _get_llm()
        llm.invoke([HumanMessage(content="hi /no_think")], max_tokens=1)
    except Exception:
        pass  # vLLM may not be ready yet — first real query will still benefit from the LLM object being cached


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/hyde")
def generate_hyde(req: HydeRequest):
    """Generate a hypothetical answer (HyDE) to improve retrieval quality."""
    llm, _ = _get_llm()
    try:
        result = llm.invoke([
            SystemMessage(content="You are a technical expert in video compression and MPEG standardization."),
            HumanMessage(content=(
                "Write exactly 2-3 sentences IN ENGLISH answering this question. "
                "Include specific metrics (BD-rate, PSNR), sequence names, and method names "
                "that would appear in MPEG documents. Make a reasonable technical guess.\n\n"
                f"Question: {req.query} /no_think"
            )),
        ])
        sentences = re.split(r"(?<=[.!?])\s+", result.content.strip())
        return {"hypothesis": " ".join(sentences[:3])}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/generate")
def generate(req: GenerateRequest):
    """Assemble the RAG prompt and stream the answer token-by-token as SSE."""
    llm, model_name = _get_llm()
    max_new_tokens = int(os.environ["MAX_NEW_TOKENS"])
    max_context_length = int(os.environ["VLLM_MAX_CONTEXT"])

    context_text = _format_chunks(req.chunks)
    if req.extra_docs:
        uploaded = "\n\n".join(
            f"[Uploaded document: {d['filename']}]\n{d['text']}"
            for d in req.extra_docs
        )
        context_text = (
            "## User-uploaded documents\n\n"
            + uploaded
            + "\n\n## Retrieved context\n\n"
            + context_text
        )

    sources = list(dict.fromkeys(c.source for c in req.chunks))
    if req.extra_docs:
        uploaded_names = [d["filename"] for d in req.extra_docs]
        sources = uploaded_names + [s for s in sources if s not in uploaded_names]

    context_text = _truncate_context(context_text, req.query + req.lang_hint, max_new_tokens, max_context_length)
    header = _build_answer_header(req.query, req.chunks)
    chain = _RAG_PROMPT | llm

    def stream_events():
        yield f"data: {json.dumps({'type': 'meta', 'model': model_name, 'sources': sources, 'header': header})}\n\n"

        accumulated = ""
        t_start = time.time()
        in_think = False
        think_buf = ""

        try:
            for chunk in chain.stream({"question": req.query + req.lang_hint, "context": context_text}):
                text = chunk.content
                text = re.sub(r"\[m\d+\]|\[\d+\]|\[[A-Za-z][A-Za-z0-9]{1,}\]", "", text)
                if re.search(r"[^\s''']{18,}", text):
                    break

                # Strip <think>...</think> blocks that Qwen3 emits (empty with /no_think)
                if in_think:
                    think_buf += text
                    if "</think>" in think_buf:
                        text = think_buf.split("</think>", 1)[1]
                        think_buf = ""
                        in_think = False
                    else:
                        accumulated += text
                        continue
                if "<think>" in text:
                    before, _, rest = text.partition("<think>")
                    if "</think>" in rest:
                        text = before + rest.split("</think>", 1)[1]
                    else:
                        in_think = True
                        think_buf = rest
                        text = before
                if not text:
                    continue

                accumulated += text
                yield f"data: {json.dumps({'type': 'chunk', 'text': text})}\n\n"
                if any(p in accumulated for p in _STOP_PATTERNS):
                    break
                if accumulated.count("\n---") >= 2:
                    break
                summary_pos = next((accumulated.find(m) for m in _SUMMARY_MARKERS if m in accumulated), -1)
                if summary_pos >= 0 and accumulated[summary_pos + 12:].find("\n\n") >= 50:
                    break
        except Exception as exc:
            print(f"[llm] generation failed after {time.time() - t_start:.2f}s: {exc}")
            yield f"data: {json.dumps({'type': 'error', 'error': f'LLM generation error: {exc}'})}\n\n"
            return

        print(f"[llm] generation: {time.time() - t_start:.2f}s — {len(accumulated.split())} words")
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(stream_events(), media_type="text/event-stream")
