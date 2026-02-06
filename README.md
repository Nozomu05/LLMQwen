# Qwen RAG (CPU-friendly) - Multi-Model Support

This is a flexible local/cloud RAG setup supporting multiple LLM providers:
- **Ollama** for local models (Qwen, Mistral, Llama) - FREE & private
- **Mistral AI** cloud API - High quality, European
- **OpenAI** cloud API - GPT-4o, GPT-4o-mini
- LangChain + Chroma for retrieval
- FastEmbed embeddings (CPU-friendly, no PyTorch required)
- Nested ZIP extraction for complex document archives
- Smart caching and parallel processing

Works on Windows without GPU. Switch between providers easily via `.env` configuration.

## Prerequisites

- Python 3.9+
- Ollama installed and running (local server at `http://localhost:11434`)

### Install Ollama on Windows
If you're not sure Ollama is installed:
1) Install via Winget (requires admin approval on first use):
```powershell
winget install Ollama.Ollama -e
```
2) Start the Ollama daemon (it usually runs as a Windows service):
```powershell
ollama --version
ollama serve
```
Leave it running in a terminal, or rely on the service.

### Pull a small Qwen model for CPU
For better CPU performance, start with a smaller instruct model:
```powershell
ollama pull qwen2.5:3b-instruct
```
You can switch to larger models later (e.g., `qwen2.5:7b-instruct` or `qwen3:8b`) once you have a GPU.

### Install Pandoc (required for ODT files)
If you plan to use `.odt` files, install Pandoc:
```powershell
winget install --id JohnMacFarlane.Pandoc -e --accept-source-agreements --accept-package-agreements
```

## Setup Python environment
From the repo root (`qwen/` folder):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Note:** If you get an execution policy error when activating the venv, run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## Configure
Copy `.env.example` to `.env` and configure your settings:

```powershell
Copy-Item .env.example .env
```

### Key Configuration Options:
- `MODEL_PROVIDER` – Choose: `ollama`, `mistral`, or `openai`
- `OLLAMA_MODEL` – default is `qwen2.5:3b-instruct`
- `MISTRAL_API_KEY` / `OPENAI_API_KEY` – For cloud providers
- `DOCS_DIR` – folder with your documents (default: `docs`)
- `CHROMA_DIR` – vector DB storage (default: `storage/chroma`)
- `RETRIEVAL_CHUNKS` – how many chunks to retrieve (default: 12)
- `USE_RERANKING` – enable for better accuracy (default: true)

---

## Multi-Model Provider Setup

Your RAG system supports multiple LLM providers. Choose based on your needs:

### 🚀 Quick Start

Edit your `.env` file and set `MODEL_PROVIDER`:

```env
MODEL_PROVIDER=ollama    # Local (free, private)
MODEL_PROVIDER=mistral   # Cloud API (paid)
MODEL_PROVIDER=openai    # Cloud API (paid)
```

### Option 1: Ollama (Local - FREE) 🏠`.pdf`, `.odt` files and nested ZIP archives
- For `.odt` files, Pandoc must be installed (see Prerequisites above)
- FastEmbed uses ONNX under the hood and is lightweight for CPU
- Smart caching skips re-ingestion if documents haven't changed
- Parallel processing speeds up document loading
- Streaming responses provide immediate feedback
- Switch between Ollama/Mistral/OpenAI without code changes

**Setup:** Already configured! Just pull different models:

```powershell
# Fast & free models
ollama pull qwen2.5:3b-instruct     # Small, fast
ollama pull qwen2.5:7b-instruct     # Balanced
ollama pull mistral:7b-instruct-v0.3 # Alternative

# Larger models (need good CPU/GPU)
ollama pull qwen2.5:14b-instruct
ollama pull mixtral:8x7b
```

**Configuration:**
```env
MODEL_PROVIDER=ollama
OLLAMA_MODEL=qwen2.5:7b-instruct
OLLAMA_BASE_URL=http://localhost:11434
```

**No API key needed!**

### Option 2: Mistral AI (Cloud API) ☁️

**Best for:** High quality, faster than local large models, European company

**Setup:**
1. Get API key: https://console.mistral.ai/
2. Install package: `pip install langchain-mistralai`

**Configuration:**
```env
MODEL_PROVIDER=mistral
MISTRAL_API_KEY=your_actual_api_key_here
MISTRAL_MODEL=mistral-large-latest
```

**Model Options:**
- `mistral-large-latest` - Most capable (expensive)
- `mistral-medium-latest` - Balanced
- `mistral-small-latest` - Fast & cheap

**Pricing:** ~$2-8 per 1M tokens

### Option 3: OpenAI (Cloud API) 🤖

**Best for:** Highest quality (GPT-4), well-tested, most features

**Setup:**
1. Get API key: https://platform.openai.com/api-keys
2. Install package: `pip install langchain-openai`

**Configuration:**
```env
MODEL_PROVIDER=openai
OPENAI_API_KEY=your_actual_api_key_here
OPENAI_MODEL=gpt-4o-mini
```

**Model Options:**
- `gpt-4o` - Most capable (expensive)
- `gpt-4o-mini` - Great balance (recommended)
- `gpt-3.5-turbo` - Fast & cheap

**Pricing:** ~$0.15-15 per 1M tokens

### Provider Comparison

| Feature | Ollama | Mistral AI | OpenAI |
|---------|--------|------------|--------|
| **Cost** | Free | ~$2-8/1M tokens | ~$0.15-15/1M tokens |
| **Privacy** | ✅ 100% local | ❌ Cloud | ❌ Cloud |
| **Speed (small)** | ~15s | ~3-5s | ~3-5s |
| **Speed (large)** | ~30-60s | ~5-10s | ~5-10s |
| **Quality (small)** | Good | Excellent | Excellent |
| **Quality (large)** | Very Good | Excellent | Outstanding |
| **Setup** | Easy | API key | API key |
| **Internet** | ❌ No | ✅ Yes | ✅ Yes |

### Recommendations

**For Development/Testing:** ✅ Ollama (free, private, no limits)

**For Production:**
- ✅ Mistral AI for good quality + reasonable cost
- ✅ OpenAI GPT-4o-mini for best balance
- ✅ OpenAI GPT-4o for highest quality

**For Maximum Privacy:** ✅ Ollama only (everything local)

### Switching Between Providers

No code changes needed! Just edit `.env`:

```powershell
# Try different providers
python .\rag\query.py "test question"

# Check active provider
Get-Content .env | Select-String "MODEL_PROVIDER"
```

---

## Ingest documents
Put `.md`, `.txt`, `.docx`, `.pptx`, `.pdf`, `.odt` files or **ZIP archives** (including nested ZIPs) in the `docs/` folder:

```powershell
python .\rag\ingest.py
```

This will:
- Extract nested ZIP files automatically
- Load all supported document types
- Build a Chroma vector store under `storage/chroma`
- Cache results to skip re-ingestion if files unchanged
- Use parallel processing for faster PDF loading

**Supported formats:** PDF, Word (.docx), PowerPoint (.pptx), Markdown (.md), Text (.txt), ODT (.odt)

## Ask questions (RAG)
```powershell
python .\rag\query.py "What does this project do?"
```

The script:
- Retrieves relevant chunks from your documents
- Uses streaming responses (answer appears immediately)
- Shows query completion time
- Cites sources from your documents

## Upgrading to Qwen 8B later
When you have a GPU, pull and use a larger model:
```powershell
ollama pull qwen3:8b
# then set in .env
OLLAMA_MODEL=qwen3:8b
```

## Notes
- Supports `.md`, `.txt`, `.docx`, `.pptx`, and `.odt` files
- For `.odt` files, Pandoc must be installed (see Prerequisites above)
- FastEmbed uses ONNX under the hood and is light-weight for CPU