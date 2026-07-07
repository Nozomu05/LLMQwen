"""Agent registry — persona + docs-folder config, backed by SQLite.

Each row in the `agents` table (in the file at AGENTS_DB_PATH) holds an
agent's name, profession, department, interests, behavior, the docs/
subfolder its retrieval is restricted to, whether it may also draw on live
web search, and (if so) a web_search_context phrase used to scope those web
searches to the right topic (see services/search/app.py's _web_search).
`get_agent()` fills the shared prompt template with a row's fields to build
that agent's system prompt.

The table and its seed rows are created automatically on first connection
(see _ensure_schema) — no separate init step needed. Edit rows directly in
the SQLite file, then call reload_agent(id) / reload_all_agents(), or hit
POST /agents/reload on the llm or search service, to pick up the change
(rows are cached in-memory per process after first lookup).
"""
import os
import sqlite3
from typing import Optional

DEFAULT_AGENT_ID = "TSP-0"

_DB_PATH = os.environ.get("AGENTS_DB_PATH", "storage/agents.db")

_PROMPT_TEMPLATE = """You are {name}, a {profession} in {department}. You are especially interested in {interests}. Your behavior: {behavior}.

Answer the user's question strictly using the information in the provided context — never fabricate facts, figures, or sources. Give a clear, well-structured answer, grouping related points with sub-headings when useful, and end with 1–2 summarising sentences. If a piece of context is tagged with its exact source, like "[Source: ...]", cite that source in parentheses after the relevant point, e.g. (Source: ...), copied verbatim from its tag. Never cite a source that is not present in a [Source: ...] tag.

If the context is empty, or doesn't actually contain information that answers the question, say plainly that you don't have that information — even if you personally could guess a plausible-sounding answer. Staying in character or being helpful never justifies inventing facts or a citation that isn't in the context.

The context may contain two sections: "Official documents" and "Web search results". If a topic appears in both and they disagree on a fact, trust the official documents and either ignore the conflicting web result or note that web sources differ. This priority rule only applies to actual disagreements — if the official documents are simply silent on the question (no conflict), answer normally from the web search results instead of treating their absence as a reason to withhold an answer.

Do not add follow-up questions or new Q:/A: turns after the answer. Skip any garbled or concatenated text fragments from the context.

Respond only in French or English: French if the question is in French, English for any other language."""

# Seed rows, inserted once when the `agents` table is first created.
# Columns: id, name, profession, department, interests, behavior, folder, web_search, web_search_context
_SEED_AGENTS = [
    (
        "TSP-0",
        "Télécom SudParis Assistant",
        "Virtual Assistant",
        "Télécom SudParis (school-wide)",
        "Helping students find accurate, well-sourced information about the school's "
        "programmes, admissions, campus life, and administrative practicalities",
        "Friendly, precise, and welcoming — makes navigating a French grande école's "
        "admissions and academic maze feel simple, without ever making things up",
        "TSP-0",
        1,
        "Télécom SudParis",
    ),
    (
        "TSP-1",
        "Antoine",
        "PhD Student",
        "Etoile Department",
        "Large Language Models (LLMs), RAG pipelines, and prompt engineering",
        "Direct yet a little shy, but still occupies space and is welcoming",
        "TSP-1",
        0,
        "",
    ),
    (
        "TSP-2",
        "Youssef",
        "Research Engineer",
        "Artemis Department",
        "3D avatars, motion generation, Gaussian avatar training, relightable avatars, "
        "and using generative AI to enhance Gaussian splats",
        "Kind of blunt, talks technicality, and makes jokes a lot of the time",
        "TSP-2",
        0,
        "",
    ),
    (
        "TSP-3",
        "Karen",
        "Professor Researcher",
        "Télécom SudParis campus Faculty",
        "3D digital avatars, real-time web-browser rendering, and pedagogy",
        "Pedagogic, acts like a supportive mentor, and makes dad jokes and stuff",
        "TSP-3",
        0,
        "",
    ),
    (
        "MPEG-0",
        "MPEG Technical Expert",
        "Standardization Engineer",
        "MPEG / ISO-IEC JTC 1/SC 29 Working Group",
        "Video compression, 3D content coding, and international coding standards "
        "(BD-rate and PSNR measurements, VSS-PCC, TMAP, RAHT, V3C, SEI, and similar)",
        "Rigorous, terse, and highly technical — no fluff, just precise engineering language",
        "MPEG-0",
        0,
        "",
    ),
]

_CACHE: dict = {}
_schema_ready = False


def _connect() -> sqlite3.Connection:
    global _schema_ready
    db_dir = os.path.dirname(_DB_PATH)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(_DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    if not _schema_ready:
        _ensure_schema(conn)
        _schema_ready = True
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS agents (
            id                  TEXT PRIMARY KEY,
            name                TEXT NOT NULL,
            profession          TEXT NOT NULL,
            department          TEXT NOT NULL,
            interests           TEXT NOT NULL,
            behavior            TEXT NOT NULL,
            folder              TEXT NOT NULL,
            web_search          INTEGER NOT NULL DEFAULT 0,
            web_search_context  TEXT NOT NULL DEFAULT ''
        )
    """)
    # Migrate tables created before these columns existed.
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(agents)")}
    if "web_search" not in existing_cols:
        conn.execute("ALTER TABLE agents ADD COLUMN web_search INTEGER NOT NULL DEFAULT 0")
    if "web_search_context" not in existing_cols:
        conn.execute("ALTER TABLE agents ADD COLUMN web_search_context TEXT NOT NULL DEFAULT ''")
    conn.executemany(
        "INSERT OR IGNORE INTO agents "
        "(id, name, profession, department, interests, behavior, folder, web_search, web_search_context) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        _SEED_AGENTS,
    )
    conn.commit()


def _fetch_agent_row(agent_id: str) -> Optional[dict]:
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT id, name, profession, department, interests, behavior, folder, "
            "web_search, web_search_context FROM agents WHERE id = ?",
            (agent_id,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_agent(agent_id: str) -> dict:
    """Return {"folder": ..., "system_prompt": ..., "web_search": ..., "web_search_context": ...}
    for `agent_id`.

    Falls back to DEFAULT_AGENT_ID if `agent_id` isn't in the database.
    Cached in-memory per process after first lookup.
    """
    if agent_id not in _CACHE:
        row = _fetch_agent_row(agent_id) or _fetch_agent_row(DEFAULT_AGENT_ID)
        if row is None:
            raise RuntimeError(
                f"No agents found in database (looked for '{agent_id}' and default '{DEFAULT_AGENT_ID}')"
            )
        _CACHE[agent_id] = {
            "folder": row["folder"],
            "web_search": bool(row["web_search"]),
            "web_search_context": row["web_search_context"],
            "system_prompt": _PROMPT_TEMPLATE.format(
                name=row["name"],
                profession=row["profession"],
                department=row["department"],
                interests=row["interests"],
                behavior=row["behavior"],
            ),
        }
    return _CACHE[agent_id]


def list_agents() -> list[dict]:
    """Return {"id", "name", "profession", "department", "web_search"} for every
    agent, ordered by id — for UIs to populate an agent selector dynamically
    instead of hardcoding the current set. Not cached (rarely called)."""
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT id, name, profession, department, web_search FROM agents ORDER BY id"
        ).fetchall()
        return [
            {
                "id": row["id"],
                "name": row["name"],
                "profession": row["profession"],
                "department": row["department"],
                "web_search": bool(row["web_search"]),
            }
            for row in rows
        ]
    finally:
        conn.close()


def reload_agent(agent_id: str) -> None:
    """Drop one agent's cached row so the next lookup re-reads it from the database."""
    _CACHE.pop(agent_id, None)


def reload_all_agents() -> None:
    """Drop every cached agent row so the next lookups re-read from the database."""
    _CACHE.clear()
