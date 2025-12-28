# Qwen RAG (CPU-friendly)

This is a minimal local RAG setup using:
- Ollama for a local Qwen chat model
- LangChain + Chroma for retrieval
- FastEmbed embeddings (CPU-friendly, no PyTorch required)

Works on Windows without GPU. You can upgrade the model later (e.g., Qwen 8B) when you have GPU.

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
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Configure
Copy `.env.example` to `.env` and adjust if needed:
- `OLLAMA_MODEL` – default is `qwen2.5:3b-instruct`
- `DOCS_DIR` – folder with your documents (default: `docs`)
- `CHROMA_DIR` – vector DB storage (default: `storage/chroma`)

```powershell
Copy-Item .env.example .env
```

## Ingest documents
Put `.md`, `.txt`, `.docx`, `.pptx`, or `.odt` files in the `docs/` folder, then run:
```powershell
python .\rag\ingest.py
```
This will build a Chroma vector store under `storage/chroma`.

## Ask questions (RAG)
```powershell
python .\rag\query.py "What does this project do?"
```
The script retrieves relevant chunks and asks the local Qwen model to answer using that context.

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