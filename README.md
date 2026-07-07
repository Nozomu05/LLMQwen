# LLMQwen

A chatbot that answers questions about your own documents — and, per agent, can also search the live web. You give it PDFs, Word files, PowerPoints, etc., pick (or build) an agent persona, and ask it anything: it finds the relevant parts and writes a precise, cited answer.

Everything runs **locally on your machine**. No data is sent to the cloud (web search, if enabled for an agent, is the one exception — it queries the public internet through a self-hosted search engine).

---

## How it works (simple version)

When you ask a question, the system does these things in order:

1. **Search** — it converts your question into numbers and uses those numbers to find the most relevant pieces of *that agent's* documents
2. **Web search** *(optional, per agent)* — if the agent has it enabled, it also searches the live web and merges those results in, tagged separately from your own documents
3. **Re-rank** — it scores every candidate piece more carefully to keep only the best ones
4. **Generate** — it feeds those pieces to the AI model, along with the agent's persona and any conversation history, and asks it to write an answer
5. **Display** — the answer streams back word by word, with citations

The AI model (Qwen) runs on your GPU via vLLM. The search, re-ranking, and web-search helpers are separate lightweight services.

### Agents

The chatbot isn't a single fixed persona — it's a small **registry of agents**, each with its own name, personality, job description, and its own restricted slice of `docs/`. Out of the box:

| Agent ID | Persona | Docs folder | Web search |
|---|---|---|---|
| `TSP-0` | Télécom SudParis Assistant (general school info) | `docs/TSP-0/` | ✅ on |
| `TSP-1` | Antoine — PhD student, Étoile Dept. | `docs/TSP-1/` | off |
| `TSP-2` | Youssef — Research Engineer, Artemis Dept. | `docs/TSP-2/` | off |
| `TSP-3` | Karen — Professor Researcher | `docs/TSP-3/` | off |
| `MPEG-0` | MPEG/video-compression technical expert | `docs/MPEG-0/` | off |

Personas live in a small SQLite database (`storage/agents.db`), auto-created on first run — see [Agents reference](#agents-reference) below for how to edit or add one.

---

## What you need before starting

### A Linux machine with an NVIDIA GPU

This project runs on **Linux only** (Ubuntu 22.04+, AlmaLinux/RHEL 9, or similar). It requires a powerful NVIDIA graphics card because the AI model is very large.

| What | Minimum | Why |
|---|---|---|
| GPU memory (VRAM) | 16 GB | The AI model needs this to fit in the GPU |
| Extra GPU memory | 4 GB | For the search and re-rank helpers |
| RAM | 32 GB | For loading documents and running everything |
| Disk space | 50 GB free | Models are large files |

> If you only have one GPU that's 24 GB or more (e.g. RTX 3090, RTX 4090), everything can share it — see the `cpu`/`cuda:0` device settings in Step 3.

### Software to install first

- **NVIDIA driver** (`nvidia-smi` should work)
- **Python 3.10 or higher** for the main environment (check with `python3 --version`)
- **tmux** (`sudo apt install tmux` / `sudo dnf install tmux`) — used to run all services in one manageable session
- **git**

You do **not** need Docker for the setup this README describes — everything runs directly on the host. (A `docker-compose.yml` also exists in this repo as an alternative path if you have working Docker access; see [Docker (alternative)](#docker-alternative) at the end.)

---

## Step 1 — Download the project

```bash
git clone git@github.com:Nozomu05/LLMQwen.git
cd LLMQwen
```

---

## Step 2 — Create the main Python environment

This one shared environment runs vLLM and all three backend services (`llm`, `reranker`, `search`) plus the document-ingestion script. SearXNG (web search) gets its **own separate** environment later — its pinned dependency versions would otherwise conflict with everything else.

```bash
python3 -m venv .venv
source .venv/bin/activate   # do this every time you open a new terminal

pip install vllm
pip install -r requirements.txt
pip install -r rag/requirements.txt
pip install -r services/llm/requirements.txt
pip install -r services/reranker/requirements.txt
pip install -r services/search/requirements.txt
```

---

## Step 3 — Configure the services

Each part of the project has a configuration file, created from a template:

```bash
cp services/llm/.env.example      services/llm/.env
cp services/reranker/.env.example services/reranker/.env
cp services/search/.env.example   services/search/.env
cp rag/.env.example               rag/.env
```

Now open each `.env` file and adjust the values below (everything else can stay at its default).

### `services/llm/.env`

```
VLLM_MODEL=Qwen/Qwen3-14B-AWQ       # must match what you start vLLM with in Step 6
MAX_TEMPERATURE=1.0                  # hard ceiling on requested temperature — see note in the file
AGENTS_DB_PATH=storage/agents.db     # auto-created; rarely needs changing
```

### `services/reranker/.env`

```
HF_CACHE_DIR=/home/yourname/.cache/huggingface
HF_HOME=/home/yourname/.cache/huggingface
RERANKER_DEVICE=cuda:0               # change to `cpu` if you only have one GPU
```

(Find your home folder with `echo $HOME`.)

### `services/search/.env`

```
HF_CACHE_DIR=/home/yourname/.cache/huggingface
HF_HOME=/home/yourname/.cache/huggingface
EMBEDDING_DEVICE=cuda:0              # change to `cpu` if you only have one GPU
SEARXNG_URL=http://localhost:8899    # set up in Step 4 — leave as-is unless you change SEARXNG_PORT
AGENTS_DB_PATH=storage/agents.db
```

### `rag/.env`

```
HF_CACHE_DIR=/home/yourname/.cache/huggingface
HF_HOME=/home/yourname/.cache/huggingface
EMBEDDING_DEVICE=cuda:0               # change to `cpu` if you only have one GPU
```

---

## Step 4 — Set up SearXNG (web search backend)

Some agents (like `TSP-0`) can supplement your documents with live web search, via a small self-hosted [SearXNG](https://github.com/searxng/searxng) instance — no API key, no cloud service, no cost. This is a one-time setup:

```bash
bash searxng/setup.sh
```

This script:
1. Clones SearXNG into `searxng/src/` (gitignored — it's vendored source, not something you edit).
2. Creates a **separate** virtual environment at `searxng/.venv/`, using whichever Python it finds that satisfies SearXNG's requirement of **Python ≥ 3.10**.
3. Works around one machine-specific SQLite compatibility issue (see below).

**About the Python version**: the script checks your system's default `python3` *first* — if it's already 3.10 or newer, that's what gets used directly, no extra step needed. It only searches for an alternate interpreter (`python3.10` through `python3.13`) if the default is too old. On the machine this project was originally built on, the system `python3` was 3.9, so it fell back to `/usr/bin/python3.11`; on a more up-to-date system, you likely won't need any extra Python install at all. If none of `python3`/`python3.10`–`python3.13` qualifies, the script exits with a clear error telling you to install one (e.g. `sudo apt install python3.11`).

**About the SQLite workaround**: SearXNG's internal cache requires SQLite ≥ 3.35. If your system's SQLite is older (common on RHEL/AlmaLinux-family systems) and you don't have root to upgrade it, `setup.sh` installs `pysqlite3-binary` (a self-contained, statically-linked modern SQLite) and points SearXNG at it automatically — this step runs unconditionally and is harmless even if your system SQLite was already new enough.

You'll also want to disable any search engines that turn out to be chronically unreliable for your network (see `searxng/settings.yml`, which already disables Google CSE by default — it ships with a hardcoded, shared CX id that gets rate-limited by Google almost immediately for everyone using it).

---

## Step 5 — Add your documents

Each agent only ever sees documents placed in its **own** subfolder of `docs/`:

```
docs/
├── TSP-0/     ← Télécom SudParis Assistant's documents
├── TSP-1/     ← Antoine's documents
├── TSP-2/     ← Youssef's documents
├── TSP-3/     ← Karen's documents
└── MPEG-0/    ← MPEG Technical Expert's documents
```

Put files into the matching agent's folder. An agent with an empty folder (and web search off) will correctly say "I don't have that information" rather than guessing — it won't invent an answer.

**Supported file types:** PDF, Word (DOCX), PowerPoint (PPTX), plain text, Markdown, Excel, CSV, OpenDocument (ODT), and ZIP archives (extracted automatically, including nested ZIPs).

---

## Step 6 — Process your documents

This reads every agent's documents, breaks them into pieces, and builds a searchable index. Run it once, and again whenever you add new documents:

```bash
source .venv/bin/activate
python rag/ingest.py
```

You'll see progress per file type, then a final summary. This can take anywhere from a minute (a handful of files) to much longer (hundreds of PDFs/Office documents) — the script prints live progress so you can gauge it.

---

## Step 7 — Start everything

```bash
bash serve.sh
```

This opens a tmux session named `rag` with one window per service:

| Window | Service | Port |
|---|---|---|
| `vllm` | The AI model itself | 8000 |
| `llm-svc` | Prompt assembly + generation | 8012 |
| `reranker-svc` | Cross-encoder re-ranking | 8011 |
| `search-svc` | Main entry point (vector + web search) | 8010 |
| `searxng-svc` | Web search backend | 8899 |
| `frontend-svc` | Test console (see Step 8) | 8013 |

Attach to watch logs: `tmux attach -t rag`, then `Ctrl-b` followed by a number to switch windows. The very first vLLM start downloads the model (~8 GB) and can take a while; wait for `Application startup complete` before querying.

To restart everything from scratch: `tmux kill-session -t rag && bash serve.sh`.

---

## Step 8 — Use it

### Option A — the test console

Open `http://localhost:8013/` in a browser. It's a plain HTML/JS page (no build step) that exercises every backend feature: agent selector, a raw temperature slider, streaming chat with citations, document attachment, conversation history, and an agent-cache reload button.

If you're on a remote machine over SSH/VS Code Remote, port-forwarding can remap ports unpredictably (VS Code sometimes forwards port `8010` to something like `localhost:65311` instead of `8010` itself, depending on what's already bound locally on your end). The console has a **Backend URL** field at the top for exactly this — just paste in whatever address your Ports panel actually shows for the search service, no code changes needed.

### Option B — the API directly

```bash
curl -N -X POST http://localhost:8010/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is Telecom SudParis?",
    "agent_id": "TSP-0",
    "temperature": 0.2,
    "history": []
  }'
```

The response streams back as Server-Sent Events (`meta` → `chunk`* → `done`, or `error`). Request fields:

| Field | Required | Notes |
|---|---|---|
| `query` | yes | The question |
| `agent_id` | no (defaults to `TSP-0`) | Which persona/doc-scope to use |
| `temperature` | no | Clamped server-side to `MAX_TEMPERATURE` regardless of what's requested |
| `history` | no | `[{"role": "user"/"assistant", "content": "..."}]` — prior turns, used for conversational context only, never for retrieval |
| `documents` | no | `[{"filename": "...", "text": "..."}]` — extra one-off text attached to this question (see `POST /extract` to convert an uploaded file first) |

Other useful endpoints on the search service (port 8010): `GET /agents` (list every agent), `POST /agents/reload` (drop cached personas so edits to `storage/agents.db` take effect without restarting), `POST /extract` (upload a file, get back plain text), `GET /health`.

---

## Agents reference

Agents live in the `agents` table of `storage/agents.db` (SQLite, auto-created with the 5 seed rows above on first run). Each row has:

| Column | Meaning |
|---|---|
| `id` | The `agent_id` used in API requests (e.g. `TSP-0`) |
| `name`, `profession`, `department`, `interests`, `behavior` | Filled into a shared prompt template to build that agent's persona |
| `folder` | The `docs/<folder>/` subfolder this agent's retrieval is restricted to |
| `web_search` | `0`/`1` — whether this agent may also draw on live web search |
| `web_search_context` | A short phrase (e.g. `"Télécom SudParis"`) appended to web search queries so results stay on-topic |

**To edit an existing agent** (change its personality, enable web search, etc.): edit the row directly in `storage/agents.db`, then call `POST /agents/reload` (on `search-svc`, which cascades to `llm-svc` automatically) — no restart needed.

**To add a new agent**: insert a new row (matching the schema in `services/agents_config.py`'s `_ensure_schema`), create its `docs/<new-folder>/` directory, reload the caches, and optionally re-run `python rag/ingest.py` if you added documents for it.

---

## Adding new documents later

1. Copy the new files into the right agent's `docs/<AGENT-ID>/` folder
2. Re-run: `python rag/ingest.py`
3. That's it — `search-svc` reads the updated index automatically, no restart needed

---

## Changing the AI model

1. Stop vLLM (`Ctrl-C` in the `vllm` tmux window)
2. Change `VLLM_MODEL` in `services/llm/.env`
3. Update the `--model` argument in `serve.sh` (or set the `VLLM_MODEL` environment variable before running it)
4. `bash serve.sh` again

No need to re-process your documents.

---

## Changing search quality settings

In `services/search/.env`, takes effect after restarting `search-svc`:

| Setting | What it does | Default |
|---|---|---|
| `RETRIEVAL_CHUNKS` | How many document pieces to consider before re-ranking | `50` |
| `WEB_SEARCH_RESULTS` | Max web results merged in, for agents with web search on | `5` |
| `USE_HYDE` | Generate a hypothetical answer first to improve search recall. Slower, better for vague questions. | `false` |

## Changing answer quality settings

In `services/llm/.env`, takes effect after restarting `llm-svc`:

| Setting | What it does | Default |
|---|---|---|
| `TEMPERATURE` | Default creativity if a request doesn't specify one. `0` = precise, `1` = creative | `0.1` |
| `MAX_TEMPERATURE` | Hard ceiling applied to *every* request, regardless of what's asked for | `1.0` |
| `MAX_NEW_TOKENS` | Maximum answer length | `1024` |
| `TOP_P` | Nucleus sampling cutoff. `1.0` = no cutoff | `0.95` |
| `FREQUENCY_PENALTY` | Reduces repetition within a single answer | `0.0` |
| `MAX_HISTORY_TURNS` | How many prior conversation turns to include | `10` |

---

## Something is not working

**"No vector store" error** — you haven't run ingestion yet, or it failed. Run `python rag/ingest.py` and check `ingestion_errors.log` for details.

**An agent always says "I don't have that information"** — its `docs/<folder>/` is empty and it has `web_search` off. Either add documents and re-ingest, or enable web search for it in `storage/agents.db`.

**The AI gives no answer or times out** — vLLM (the `vllm` tmux window) may have crashed or not finished loading. It should print `Application startup complete`; if it crashed, restart it.

**Out of GPU memory** — try, in order:
1. `services/reranker/.env`: `RERANKER_DEVICE=cpu`
2. `services/search/.env` and `rag/.env`: `EMBEDDING_DEVICE=cpu`
3. Restart the affected services

**SearXNG errors in its log** (`CAPTCHA`, `Too many request`, engine timeouts) — normal for unauthenticated scraping-based search engines under sustained use; SearXNG falls back to other engines automatically. Only worth acting on if it visibly degrades answer quality in practice — see `searxng/settings.yml` to disable a chronically unreliable engine (same pattern already used for Google CSE).

**Check if the services are running correctly:**
```bash
curl http://localhost:8010/health
curl http://localhost:8011/health
curl http://localhost:8012/health
curl "http://localhost:8899/search?q=test&format=json"
```
Each should respond successfully (the first three return `{"status":"ok"}`).

---

## Docker (alternative)

A `docker-compose.yml` also exists for a container-based deployment of the three backend services (`llm`, `reranker`, `search`) plus a one-shot `ingest` job — useful if you have working Docker with GPU support and prefer containers. It is **not** the path this README walks through above (which assumes bare-metal via `serve.sh`, and is what's actually been tested end-to-end for this project), and it does not include SearXNG or the frontend test console. If you want to use it:

```bash
docker compose run --rm ingest      # process documents
docker compose up                    # start llm, reranker, search
```

vLLM still needs to run directly on the host either way (see Step 7) — `services/llm/.env`'s `VLLM_BASE_URL` is overridden to `http://host.docker.internal:8000` automatically in the Docker Compose file so the containerized `llm` service can reach it.
