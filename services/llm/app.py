"""LLM Service — prompt assembly and answer generation via vLLM.

Receives: POST /generate  {"query", "chunks", "lang_hint", "extra_docs", "temperature", "agent_id", "history"}
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
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from services.agents_config import DEFAULT_AGENT_ID, get_agent, reload_agent, reload_all_agents

load_dotenv(override=True)

app = FastAPI(title="LLM Service")

_LLM_CACHE: dict = {}
_TOKENIZER_CACHE: dict = {}

_HUMAN_TEMPLATE = """Context:
{context}

Question: {question} /no_think

Answer:"""


_SUMMARY_MARKERS = ("**Summary**", "**Résumé**")

_STOP_PATTERNS = ["Human:", "\nQ:", "\nA:", "\nQuestion:"]


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class Chunk(BaseModel):
    content: str
    source: str
    origin: str = "document"  # "document" (vector store) or "web" (SearXNG)


class HistoryTurn(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class GenerateRequest(BaseModel):
    query: str
    chunks: list[Chunk]
    lang_hint: str = ""
    extra_docs: Optional[list[dict]] = None
    temperature: Optional[float] = None
    agent_id: str = DEFAULT_AGENT_ID
    history: Optional[list[HistoryTurn]] = None


class HydeRequest(BaseModel):
    query: str


class ReloadAgentsRequest(BaseModel):
    agent_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_llm(temperature: Optional[float] = None):
    model_name = os.environ["VLLM_MODEL"]
    base_url = os.environ["VLLM_BASE_URL"]
    temperature = float(os.environ["TEMPERATURE"]) if temperature is None else temperature
    # Clamp regardless of what any caller requests — this is a RAG assistant,
    # not a creative-writing tool. High temperature demonstrably increases
    # the odds of the model drifting off the requested response language or
    # under-using retrieved context; it doesn't eliminate that risk even at
    # low temperature, but it measurably reduces how often it happens.
    max_temperature = float(os.environ["MAX_TEMPERATURE"])
    temperature = min(temperature, max_temperature)
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
    return _LLM_CACHE[key], model_name, temperature


def _style_hint(temperature: float, response_language: str) -> str:
    """Extra instruction appended to the question, scaled to the request's
    (clamped) temperature. Sampling temperature alone barely varies wording
    for RAG-grounded answers — the "never fabricate" system rule makes the
    model stick close to the source's literal phrasing for factual claims
    regardless of temperature, since paraphrasing risks subtly distorting a
    fact. An explicit style instruction is needed to get genuinely varied
    phrasing at higher temperature without weakening that accuracy rule.

    `response_language` must be spelled out concretely ("English"/"French")
    rather than referenced back to the system prompt — a vague "keep
    responding in the same language" reminder isn't a strong enough anchor
    for English questions specifically (lang_hint is empty for those, so
    there's nothing concrete for a vague back-reference to point at), and
    without a concrete name the model drifted to French once encouraged to
    paraphrase freely, since most of the source content is French.
    """
    language_reminder = f" You must still answer in {response_language}, no matter how much you vary your wording."
    if temperature <= 0.15:
        return ""
    if temperature <= 0.45:
        return (
            " (Feel free to vary your sentence structure and word choice somewhat "
            "from a plain restatement, as long as every fact, figure, and citation "
            "stays exactly accurate." + language_reminder + ")"
        )
    return (
        " (Actively vary your wording, sentence structure, and phrasing — use "
        "synonyms, reorder ideas, write with a more natural narrative flow — "
        "rather than closely mirroring the source's exact phrasing. This applies "
        "ONLY to phrasing: every fact, figure, and citation must still be exactly "
        "accurate." + language_reminder + ")"
    )


def _empty_context_hint(context_text: str) -> str:
    """Forceful instruction appended to the question when there are no
    retrieved chunks at all (e.g. an agent whose docs/ folder is still
    empty, with web search off or returning nothing).

    The system prompt already asks the model to admit when it doesn't know
    something — in practice that alone wasn't enough: with zero context the
    model still answered fluently from its own pretrained general knowledge
    about "how this sort of thing usually works" (even inventing fake
    "(Source: Official documents)" citations quoting the section headers
    themselves). A code-level check that can see the context is genuinely
    empty is more reliable than hoping the model self-polices that case.
    """
    if context_text.strip():
        return ""
    return (
        " (No information was found in the available documents or web search "
        "for this question — the context is empty. Unless the question is "
        "simply about who you are or what your own role/interests/personality "
        "are (answerable directly from your persona above, not from context), "
        "you must say plainly that you don't have this information. Do not "
        "answer a factual/external question from your own general knowledge, "
        "and do not invent a citation.)"
    )


def _build_messages(agent_id: str, history: Optional[list[HistoryTurn]], context: str, question: str) -> list[BaseMessage]:
    """Assemble the full message list for one turn: system persona, prior
    conversation turns (if any), then the current context + question.

    History is only ever used here, for conversational continuity — never
    for vector/web search, which stay scoped to the current question alone.
    """
    system_prompt = get_agent(agent_id)["system_prompt"]
    messages: list[BaseMessage] = [SystemMessage(content=system_prompt)]
    max_turns = int(os.environ.get("MAX_HISTORY_TURNS", "10"))
    for turn in (history or [])[-max_turns:]:
        cls = HumanMessage if turn.role == "user" else AIMessage
        messages.append(cls(content=turn.content))
    messages.append(HumanMessage(content=_HUMAN_TEMPLATE.format(context=context, question=question)))
    return messages


def _clean_chunk_text(text: str) -> str:
    text = re.sub(r"[^\s''']{18,}", "", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _format_chunk_group(chunks: list[Chunk]) -> str:
    return "\n\n".join(
        f"[Source: {c.source}]\n{_clean_chunk_text(c.content)}"
        for c in chunks
    )


def _format_chunks(chunks: list[Chunk]) -> str:
    """Render retrieved chunks as context, with document chunks (vector store)
    and web chunks (SearXNG) under separate headers so the model can tell them
    apart — the system prompt tells it to trust documents over web results
    when the two conflict (see services/agents_config.py)."""
    doc_chunks = [c for c in chunks if c.origin != "web"]
    web_chunks = [c for c in chunks if c.origin == "web"]

    sections = []
    if doc_chunks:
        sections.append("## Official documents\n\n" + _format_chunk_group(doc_chunks))
    if web_chunks:
        sections.append("## Web search results\n\n" + _format_chunk_group(web_chunks))
    return "\n\n".join(sections)


def _get_tokenizer():
    model_name = os.environ["VLLM_MODEL"]
    if model_name not in _TOKENIZER_CACHE:
        from transformers import AutoTokenizer
        _TOKENIZER_CACHE[model_name] = AutoTokenizer.from_pretrained(model_name)
    return _TOKENIZER_CACHE[model_name]


def _count_tokens(text: str) -> int:
    return len(_get_tokenizer().encode(text, add_special_tokens=False))


def _truncate_context(
    context_text: str,
    question: str,
    max_new_tokens: int,
    max_context_length: int,
    system_prompt: str,
    history: Optional[list[HistoryTurn]] = None,
) -> str:
    """Truncate context_text so the full prompt fits the model's context window.

    Uses the real tokenizer instead of a chars-per-token estimate — with
    document uploads pushing prompts close to the limit, an approximate
    ratio can undercount and the request gets rejected by vLLM entirely.
    """
    # Small safety margin for chat-template special tokens (role markers, etc.)
    template_overhead_tokens = int(os.environ["PROMPT_OVERHEAD_TOKENS"])
    system_tokens = _count_tokens(system_prompt)
    max_turns = int(os.environ.get("MAX_HISTORY_TURNS", "10"))
    history_tokens = sum(_count_tokens(t.content) for t in (history or [])[-max_turns:])
    fixed_tokens = system_tokens + history_tokens + _count_tokens(question) + template_overhead_tokens

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
    sources_str = ", ".join(f"`{s}`" for s in sources)
    return (
        f"**Question:** {query}  \n"
        f"**Sources consulted:** {sources_str}  \n\n"
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
        llm, _, _ = _get_llm()
        llm.invoke([HumanMessage(content="hi /no_think")], max_tokens=1)
    except Exception:
        pass  # vLLM may not be ready yet — first real query will still benefit from the LLM object being cached


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/agents/reload")
def reload_agents(req: ReloadAgentsRequest):
    """Drop cached agent persona(s) so the next request re-reads them from the database."""
    if req.agent_id:
        reload_agent(req.agent_id)
    else:
        reload_all_agents()
    return {"status": "ok"}


@app.post("/hyde")
def generate_hyde(req: HydeRequest):
    """Generate a hypothetical answer (HyDE) to improve retrieval quality."""
    llm, _, _ = _get_llm()
    try:
        result = llm.invoke([
            SystemMessage(content="You are an expert on Télécom SudParis, a French graduate engineering school."),
            HumanMessage(content=(
                "Write exactly 2-3 sentences IN ENGLISH answering this question. "
                "Include specific programme names, figures, and terminology "
                "that would appear on the school's website. Make a reasonable guess.\n\n"
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
    llm, model_name, effective_temperature = _get_llm(req.temperature)
    web_sources = [c.source for c in req.chunks if c.origin == "web"]
    doc_sources = [c.source for c in req.chunks if c.origin != "web"]
    print(
        f"[llm] /generate: requested_temperature={req.temperature!r} effective_temperature={effective_temperature!r} "
        f"agent_id={req.agent_id!r} query={req.query!r} "
        f"doc_chunks={len(doc_sources)} web_chunks={len(web_sources)} web_sources={web_sources}"
    )
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

    response_language = "French" if "français" in req.lang_hint.lower() else "English"
    question = (
        req.query + req.lang_hint
        + _style_hint(effective_temperature, response_language)
        + _empty_context_hint(context_text)
    )
    system_prompt = get_agent(req.agent_id)["system_prompt"]
    context_text = _truncate_context(
        context_text, question, max_new_tokens, max_context_length,
        system_prompt, req.history,
    )
    header = _build_answer_header(req.query, req.chunks)
    messages = _build_messages(req.agent_id, req.history, context_text, question)

    def stream_events():
        yield f"data: {json.dumps({'type': 'meta', 'model': model_name, 'sources': sources, 'header': header})}\n\n"

        accumulated = ""
        t_start = time.time()
        in_think = False
        think_buf = ""

        try:
            for chunk in llm.stream(messages):
                text = chunk.content
                text = re.sub(r"\[m\d+\]|\[\d+\]|\[[A-Za-z][A-Za-z0-9]{1,}\]", "", text)
                if re.search(r"[^\s''']{18,}", text) and "http" not in text.lower():
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
