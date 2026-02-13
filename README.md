# RAG System with Qwen

A Retrieval-Augmented Generation (RAG) system that lets you query your documents using Ollama and Qwen models locally.

---

## Installation and Setup

### Step 1: Install Ollama

**Windows (via Winget):**
```powershell
winget install Ollama.Ollama -e
```

Verify installation:
```powershell
ollama --version
```

Ollama runs as a Windows service automatically. If not running:
```powershell
ollama serve
```

**macOS (via Homebrew):**
```bash
brew install ollama
```

Verify installation:
```bash
ollama --version
```

Start Ollama service:
```bash
ollama serve
```

**macOS (Manual Download):**
Download from [https://ollama.ai/download](https://ollama.ai/download) and install the .dmg file.

### Step 2: Pull Required Ollama Models

**LLM Model (for answering queries):**
```bash
ollama pull qwen2.5:14b-instruct
```

**Embedding Model (for semantic search):**
```bash
ollama pull mxbai-embed-large
```

**Note for low-end computers:** The 14b model requires ~16GB RAM. If you have less RAM, use:
```bash
ollama pull qwen2.5:7b-instruct  # Requires ~8GB RAM
```

### Step 3: Enable CPU-Only Mode (For Low-End Computers)

**If you have a low-end computer or insufficient GPU memory**, force Ollama to run on CPU only:

**Windows:**
```powershell
[System.Environment]::SetEnvironmentVariable('OLLAMA_NUM_GPU', '0', 'User')
$env:OLLAMA_NUM_GPU = '0'
```

**macOS/Linux:**
```bash
echo 'export OLLAMA_NUM_GPU=0' >> ~/.bashrc  # or ~/.zshrc for zsh
source ~/.bashrc  # or source ~/.zshrc
```

Restart your terminal after setting this. The model will run slower but work on any computer.

**To re-enable GPU later (if you upgrade hardware):**
```powershell
# Windows
[System.Environment]::SetEnvironmentVariable('OLLAMA_NUM_GPU', '1', 'User')
```
```bash
# macOS/Linux - remove the line from ~/.bashrc or ~/.zshrc
```

### Step 4: Install Pandoc (Optional)

Only needed if you have OpenDocument (.odt) files:

**Windows:**
```powershell
winget install --id JohnMacFarlane.Pandoc -e
```

**macOS:**
```bash
brew install pandoc
```

### Step 5: Setup Python Environment

**Windows:**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Note:** If you get an execution policy error:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Install dependencies (all platforms):**
```bash
pip install -r requirements.txt
```

### Step 6: Configure Environment

**Windows:**
```powershell
Copy-Item .env.example .env
```

**macOS/Linux:**
```bash
cp .env.example .env
```

Edit `.env` with your settings:
```env
# Model Configuration
OLLAMA_MODEL=qwen2.5:14b-instruct
OLLAMA_BASE_URL=http://localhost:11434
EMBEDDING_MODEL=mxbai-embed-large

# Retrieval Settings
RETRIEVAL_CHUNKS=100
TOP_N_RERANK=15
USE_RERANKING=true

# Document Processing
CHUNK_SIZE=800
CHUNK_OVERLAP=160
```

**Note:** If using 7b model on low-end computer, change to `OLLAMA_MODEL=qwen2.5:7b-instruct`

### Step 7: Add Your Documents

Place your documents (Word, PDF, PowerPoint, Text, Markdown, etc.) in the `docs/` folder.

### Step 8: Ingest Documents

Run the ingestion script to process your documents:

**Windows:**
```powershell
python rag\ingest.py
```

**macOS/Linux:**
```bash
python rag/ingest.py
```

This will:
- Extract ZIP files automatically
- Load and process all documents
- Generate embeddings
- Store vectors in the database

### Step 9: Start the Frontend

Start the web interface:

**Windows:**
```powershell
python frontend\app.py
```

**macOS/Linux:**
```bash
python frontend/app.py
```

The server will start at: **http://127.0.0.1:8000**

Open this URL in your browser to start querying your documents!

---

## Command-Line Query (Optional)

You can also run queries directly from the command line:

**Windows:**
```powershell
python rag\query.py "Your question here"
```

**macOS/Linux:**
```bash
python rag/query.py "Your question here"
```

---

## Performance Notes

### CPU vs GPU Mode
- **GPU Mode (default):** Fast responses (1-2 seconds with 14b model)
- **CPU-Only Mode:** Slower responses (8-15 seconds with 14b model) but works on any computer

### Model Recommendations by Hardware

| RAM Available | Recommended Model | CPU Query Time | Quality |
|---------------|-------------------|----------------|----------|
| 8GB | qwen2.5:7b-instruct | 3-5 seconds | Good |
| 16GB+ | qwen2.5:14b-instruct | 8-15 seconds | Excellent |
| 32GB+ | qwen2.5:32b | 30-60 seconds | Best |

**Note:** These times are for CPU-only mode. GPU mode is 6-10x faster.
