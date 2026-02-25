# RAG System with Qwen

A Retrieval-Augmented Generation (RAG) system that lets you query your documents using Qwen models from HuggingFace Transformers locally.

---

## Table of Contents

- [Installation and Setup](#installation-and-setup)
  - [Linux Prerequisites](#linux-prerequisites)
  - [Step 1: Setup Python Environment](#step-1-setup-python-environment)
  - [Step 2: Install Pandoc (Optional)](#step-2-install-pandoc-optional)
  - [Step 3: Configure Environment](#step-3-configure-environment)
  - [Step 4: Add Your Documents](#step-4-add-your-documents)
  - [Step 5: Ingest Documents](#step-5-ingest-documents)
  - [Step 6: Start the Frontend](#step-6-start-the-frontend)
- [Command-Line Query (Optional)](#command-line-query-optional)
  - [Interactive Mode (Recommended)](#interactive-mode-recommended)
  - [Single Query Mode](#single-query-mode)
- [Performance Notes](#performance-notes)
  - [CPU vs GPU Mode](#cpu-vs-gpu-mode)
  - [Model Recommendations by Hardware](#model-recommendations-by-hardware)
- [Model Upgrade Guide](#model-upgrade-guide)
  - [Current Setup (Fast/Testing)](#current-setup-fasttesting)
  - [Upgrading to Production Quality](#upgrading-to-production-quality)
  - [🏆 Recommended Production Configurations](#-recommended-production-configurations)
  - [⚡ Performance Impact Summary](#-performance-impact-summary)
- [🌍 Multilingual Functionality Guide](#-multilingual-functionality-guide)
  - [How Each Component Affects Multilingual Support](#how-each-component-affects-multilingual-support)
  - [Current System Multilingual Capability](#current-system-multilingual-capability)
  - [Upgrading to Full Multilingual Support](#upgrading-to-full-multilingual-support)
  - [Testing Multilingual Functionality](#testing-multilingual-functionality)
- [Chunking Configuration Guide](#chunking-configuration-guide)
  - [What is Chunking?](#what-is-chunking)
  - [Current Default Settings](#current-default-settings)
  - [How Chunk Size Affects Quality](#how-chunk-size-affects-quality)
  - [Why Overlap Matters](#why-overlap-matters)
  - [Recommended Settings by Document Type](#recommended-settings-by-document-type)
  - [How to Adjust Chunking](#how-to-adjust-chunking)
  - [Chunk Size Impact on Your System](#chunk-size-impact-on-your-system)

---

## Installation and Setup

### Linux Prerequisites

**For Ubuntu/Debian-based distributions:**
```bash
# Update package list
sudo apt update

# Install Python 3.10+ and pip
sudo apt install python3 python3-pip python3-venv

# Install development tools (required for some Python packages)
sudo apt install build-essential python3-dev
```

**For Fedora/RHEL/CentOS:**
```bash
# Install Python 3.10+ and pip
sudo dnf install python3 python3-pip

# Install development tools
sudo dnf groupinstall "Development Tools"
sudo dnf install python3-devel
```

**For Arch Linux:**
```bash
# Install Python and pip
sudo pacman -S python python-pip

# Install base development tools
sudo pacman -S base-devel
```

**Verify Python installation:**
```bash
python3 --version  # Should be 3.10 or higher
pip3 --version
```

### Step 1: Setup Python Environment

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

**Note:** The first time you run a query, the Qwen model (~28GB for 14B model) will download automatically to `~/.cache/huggingface/`. This may take some time depending on your internet connection.

### Step 2: Install Pandoc (Optional)

Only needed if you have OpenDocument (.odt) files:

**Windows:**
```powershell
winget install --id JohnMacFarlane.Pandoc -e
```

**macOS:**
```bash
brew install pandoc
```

**Linux:**
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install pandoc

# Fedora/RHEL/CentOS
sudo dnf install pandoc

# Arch Linux
sudo pacman -S pandoc
```

### Step 3: Configure Environment

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
LLM_PROVIDER=transformers
TRANSFORMERS_MODEL=Qwen/Qwen2.5-14B-Instruct
MAX_NEW_TOKENS=4096
TEMPERATURE=0
LLM_SEED=42
QUANTIZATION=4bit

# Device Configuration (auto, cuda, or cpu)
LLM_DEVICE=auto
EMBEDDING_DEVICE=cuda
RERANKER_DEVICE=cuda

# Embedding Configuration
EMBEDDING_PROVIDER=huggingface
EMBEDDING_MODEL=intfloat/multilingual-e5-large-instruct

# Retrieval Settings
RETRIEVAL_CHUNKS=100
TOP_N_RERANK=8
USE_RERANKING=true

# Document Processing
CHUNK_SIZE=800
CHUNK_OVERLAP=100
```

**Note:** If using a lower-spec computer, change to `TRANSFORMERS_MODEL=Qwen/Qwen2.5-7B-Instruct` for faster performance. If you don't have a GPU, set all device settings to `cpu`.

### Step 4: Add Your Documents

Place your documents (Word, PDF, PowerPoint, Text, Markdown, etc.) in the `docs/` folder.

### Step 5: Ingest Documents

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

### Step 6: Start the Frontend

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

### Interactive Mode (Recommended)

For multiple queries without reloading the model each time:

**macOS/Linux:**
```bash
python rag/query_interactive.py
```

**Windows:**
```powershell
python rag\query_interactive.py
```

This loads the model **once** and keeps it in memory. You can then ask multiple questions without the 15-second checkpoint loading delay.

**Example session:**
```
Query: What is V-PCC?
[Answer streams in real-time...]

Query: How does it compare to G-PCC?
[Answer streams immediately - no reload!]

Query: quit
```

### Single Query Mode

For one-off queries from the command line:

**macOS/Linux:**
```bash
python rag/query.py "Your question here"
```

**Windows:**
```powershell
python rag\query.py "Your question here"
```

**Note:** This reloads the model each time (~15s startup)

---

## Performance Notes

### CPU vs GPU Mode

The system can run on either CPU or GPU for optimal performance. You can configure which device each component uses in your `.env` file:

```env
# Device configuration
# Options: auto (auto-detect GPU), cuda (force GPU), cpu (force CPU)
LLM_DEVICE=auto              # Qwen language model
EMBEDDING_DEVICE=cuda        # Document/query embeddings
RERANKER_DEVICE=cuda         # Re-ranking model
```

**Device Options:**
- `auto` - Automatically detects and uses GPU if available (recommended for LLM)
- `cuda` - Forces GPU usage (fastest, requires NVIDIA GPU with CUDA)
- `cpu` - Forces CPU usage (slower but works on any computer)

**Performance Comparison (14B model):**
- **GPU Mode (cuda):** Fast responses (1-2 seconds)
- **CPU-Only Mode (cpu):** Slower responses (8-15 seconds) but works on any computer
- **Auto Mode (auto):** Best of both worlds - uses GPU if available, falls back to CPU

**Recommended Configurations:**

*For systems with NVIDIA GPU:*
```env
LLM_DEVICE=auto              # Use GPU if available
EMBEDDING_DEVICE=cuda        # Embeddings are 10-50x faster on GPU
RERANKER_DEVICE=cuda         # Re-ranking is faster on GPU
```

*For CPU-only systems (no GPU):*
```env
LLM_DEVICE=cpu
EMBEDDING_DEVICE=cpu
RERANKER_DEVICE=cpu
```

*For systems with limited GPU memory:*
```env
LLM_DEVICE=cpu               # Save GPU memory
EMBEDDING_DEVICE=cuda        # Embeddings use less memory
RERANKER_DEVICE=cpu          # Only when needed
```

**Note:** After changing device settings, restart the application for changes to take effect. Re-ingestion is not required unless you change `EMBEDDING_DEVICE` after already ingesting documents.

### Model Recommendations by Hardware

| RAM Available | Recommended Model | CPU Query Time | Quality |
|---------------|-------------------|----------------|----------|
| 8GB | qwen2.5:7b-instruct | 3-5 seconds | Good |
| 16GB+ | qwen2.5:14b-instruct | 8-15 seconds | Excellent |
| 32GB+ | qwen2.5:32b | 30-60 seconds | Best |

**Note:** These times are for CPU-only mode. GPU mode is 6-10x faster.
---

## Model Upgrade Guide

### Current Setup (Fast/Testing)
Your system is currently configured for **speed and testing**:
- **Embedding:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim, very fast)
- **Reranker:** `BAAI/bge-reranker-base` (good quality)
- **LLM:** `qwen2.5:14b-instruct` (excellent balance)

### Upgrading to Production Quality

#### 🚀 **Embedding Model Upgrades**

**Current:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim)
- Speed: ⚡⚡⚡⚡⚡ Very Fast (5x faster than BGE-large)
- Quality: ⭐⭐⭐ Good
- Use case: Testing, prototyping, fast iterations

**Option 1 - Balanced:** `BAAI/bge-base-en-v1.5` (768-dim)
- Speed: ⚡⚡⚡⚡ Fast (2x faster than BGE-large)
- Quality: ⭐⭐⭐⭐ Very Good
- Use case: Production with good performance/quality balance
- **Recommended for most users**

**Option 2 - Best Quality:** `BAAI/bge-large-en-v1.5` (1024-dim)
- Speed: ⚡⚡⚡ Moderate
- Quality: ⭐⭐⭐⭐⭐ Excellent
- Use case: Production where quality is critical
- Trade-off: Slower ingestion (but queries remain fast)

**Option 3 - Multilingual:** `BAAI/bge-m3` (1024-dim)
- Speed: ⚡⚡⚡ Moderate  
- Quality: ⭐⭐⭐⭐⭐ Excellent
- Use case: Multi-language documents (100+ languages)
- Supports: Chinese, French, Spanish, German, etc.

**To upgrade embedding model:**
```env
# In .env file, change:
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5  # or bge-large-en-v1.5
```
Then re-run: `python rag/ingest.py`

#### 🎯 **Reranker Model Upgrades**

**Current:** `BAAI/bge-reranker-base` (278M params)
- Speed: ⚡⚡⚡⚡ Fast
- Quality: ⭐⭐⭐⭐ Very Good
- Already excellent for most use cases

**Option 1 - Higher Quality:** `BAAI/bge-reranker-large` (560M params)
- Speed: ⚡⚡⚡ Moderate
- Quality: ⭐⭐⭐⭐⭐ Excellent
- Use case: When answer quality is critical
- Trade-off: 2x slower reranking (still fast overall)

**Option 2 - Best Available:** `BAAI/bge-reranker-v2-m3` (568M params)
- Speed: ⚡⚡⚡ Moderate
- Quality: ⭐⭐⭐⭐⭐ State-of-the-art
- Use case: Maximum accuracy, multilingual support
- Supports: 100+ languages

**To upgrade reranker:**
```env
# In .env file, change:
RERANKER_MODEL=BAAI/bge-reranker-large  # or bge-reranker-v2-m3
```
No re-ingestion needed, changes apply immediately!

#### 🤖 **LLM Model Upgrades**

**Current:** `qwen2.5:14b-instruct` (14B params, 8GB VRAM/16GB RAM)
- Speed: ⚡⚡⚡⚡ Fast
- Quality: ⭐⭐⭐⭐ Excellent
- Already very good for most tasks

**Option 1 - More Capable:** `qwen2.5:32b-instruct` (32B params, 20GB VRAM/32GB RAM)
- Speed: ⚡⚡⚡ Moderate (2x slower)
- Quality: ⭐⭐⭐⭐⭐ Outstanding
- Use case: Complex reasoning, technical documents
- Requirements: 32GB+ RAM recommended

**Option 2 - Maximum Quality:** `Qwen/Qwen2.5-72B-Instruct` (72B params, 48GB VRAM/64GB RAM)
- Speed: ⚡⚡ Slow (5x slower)
- Quality: ⭐⭐⭐⭐⭐ Best available
- Use case: Research, critical analysis, highest accuracy
- Requirements: 64GB+ RAM, powerful hardware

**Option 3 - Faster Lightweight:** `Qwen/Qwen2.5-7B-Instruct` (7B params, 4GB VRAM/8GB RAM)
- Speed: ⚡⚡⚡⚡⚡ Very Fast (2x faster)
- Quality: ⭐⭐⭐ Good
- Use case: Low-end hardware, quick responses

**To upgrade LLM:**
```env
# Update .env
TRANSFORMERS_MODEL=Qwen/Qwen2.5-32B-Instruct
```
The new model will download automatically on first use. No re-ingestion needed!

### 🏆 **Recommended Production Configurations**

#### **Configuration 1: Balanced (Recommended)**
```env
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
RERANKER_MODEL=BAAI/bge-reranker-base
TRANSFORMERS_MODEL=Qwen/Qwen2.5-14B-Instruct
```
- **Speed:** Fast
- **Quality:** Very Good
- **Hardware:** 16GB RAM minimum
- **Best for:** Most production use cases

#### **Configuration 2: Maximum Quality**
```env
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
TRANSFORMERS_MODEL=Qwen/Qwen2.5-32B-Instruct
```
- **Speed:** Moderate
- **Quality:** Excellent
- **Hardware:** 32GB RAM minimum
- **Best for:** Critical applications, research

#### **Configuration 3: Fast & Efficient (Current)**
```env
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RERANKER_MODEL=BAAI/bge-reranker-base
TRANSFORMERS_MODEL=Qwen/Qwen2.5-14B-Instruct
```
- **Speed:** Very Fast
- **Quality:** Good
- **Hardware:** 16GB RAM minimum
- **Best for:** Testing, development, rapid iteration

#### **Configuration 4: Multilingual**
```env
EMBEDDING_MODEL=BAAI/bge-m3
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
TRANSFORMERS_MODEL=Qwen/Qwen2.5-14B-Instruct
```
- **Speed:** Moderate
- **Quality:** Excellent
- **Hardware:** 16GB RAM minimum
- **Best for:** Multi-language document collections

### ⚡ **Performance Impact Summary**

| Component | Affects | Re-ingestion Required? |
|-----------|---------|------------------------|
| Embedding Model | Ingestion speed, retrieval quality | ✅ Yes |
| Reranker Model | Query speed (minimal), answer quality | ❌ No |
| LLM Model | Response generation speed/quality | ❌ No |

**Note:** Upgrading embedding model requires re-running `python rag/ingest.py` to rebuild the vector database with new embeddings.

---

## 🌍 Multilingual Functionality Guide

The chatbot **automatically responds in the language you use** to ask questions (English, French, Spanish, etc.). However, **each model component affects multilingual quality differently**:

### How Each Component Affects Multilingual Support

#### **1. Embedding Model - CRITICAL for Multilingual Retrieval** 🔴

**Impact:** Determines if your question in ANY language can find relevant documents

**Current Model:** `sentence-transformers/all-MiniLM-L6-v2`
- ⚠️ **English-only optimized**
- Non-English queries will retrieve less relevant documents
- Works for English, poor for French/Spanish/other languages

**Recommended for Multilingual:**
```env
EMBEDDING_MODEL=BAAI/bge-m3
# or
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

**Why it matters:**
- French question → English-focused embeddings → retrieves wrong documents → LLM gets irrelevant context → poor answer **even if LLM speaks French**
- Multilingual embeddings → retrieves correct documents in any language → LLM gets relevant context → excellent answer

**⚠️ Requires re-ingestion:** YES - `python rag/ingest.py`

---

#### **2. Reranker Model - Important for Multilingual Precision** 🟡

**Impact:** Refines which documents are most relevant to your question

**Current Model:** `BAAI/bge-reranker-base`
- ⚠️ **English-focused**
- Can rerank, but less accurate for non-English queries

**Recommended for Multilingual:**
```env
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
```

**Why it matters:**
- Even if embeddings retrieve 10 good multilingual documents, English-only reranker might rank them poorly
- Multilingual reranker correctly identifies the most relevant chunks in any language

**⚠️ Requires re-ingestion:** NO - just update `.env` and restart

---

#### **3. LLM (Text Generation Model) - Determines Answer Language** 🟢

**Impact:** Generates the actual response in the target language

**Current Model:** `qwen2.5:14b-instruct`
- ✅ **Excellent multilingual support** (100+ languages)
- Strong in: English, Chinese, French, Spanish, German, Japanese, Korean, Arabic, and more
- The prompt automatically instructs it to respond in the question's language

**Alternative Multilingual LLMs:**
```env
# In .env file
TRANSFORMERS_MODEL=Qwen/Qwen2.5-14B-Instruct    # Excellent for 100+ languages
TRANSFORMERS_MODEL=Qwen/Qwen2.5-32B-Instruct    # Best multilingual quality
# Other alternatives:
# TRANSFORMERS_MODEL=meta-llama/Llama-3.1-8B-Instruct  # Good for European languages
```

**Why it matters:**
- Even with perfect retrieval, if LLM doesn't support the language, answers will be poor or in wrong language
- Qwen models are already excellent for multilingual - upgrading mainly improves reasoning depth

---

### Current System Multilingual Capability

| Component | Current Model | Multilingual? | Impact on Non-English |
|-----------|---------------|---------------|------------------------|
| **Embedding** | all-MiniLM-L6-v2 | ❌ English-only | 🔴 **Poor retrieval** for non-English questions |
| **Reranker** | bge-reranker-base | ⚠️ English-focused | 🟡 **Suboptimal ranking** for non-English |
| **LLM** | Qwen2.5-14B-Instruct | ✅ Excellent | ✅ **Perfect responses** in any language |

**Result:** The LLM **CAN respond** in French/Spanish/etc., but will work with **lower-quality context** retrieved by English-only embeddings.

---

### Upgrading to Full Multilingual Support

**Recommended Configuration:**

```env
# In .env file
EMBEDDING_MODEL=BAAI/bge-m3
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
TRANSFORMERS_MODEL=Qwen/Qwen2.5-14B-Instruct
```

**Steps:**
1. Update `.env` with multilingual models
2. Re-ingest documents: `python rag/ingest.py` (required for embedding change)
3. Restart frontend/queries

**Benefits:**
- ✅ Excellent retrieval for questions in **any language**
- ✅ Accurate reranking regardless of language
- ✅ High-quality answers in **100+ languages**

**Trade-offs:**
- Slightly slower (BGE-m3 is ~2x slower than all-MiniLM-L6-v2)
- Larger model downloads (~3GB vs 90MB)

---

### Testing Multilingual Functionality

```powershell
# English
python rag/query.py "What are the latest V-PCC compression results?"

# French
python rag/query.py "Quels sont les derniers résultats de compression V-PCC ?"

# Spanish
python rag/query.py "¿Cuáles son los últimos resultados de compresión V-PCC?"
```

**Expected behavior:**
- ✅ LLM responds in the correct language (works with current setup)
- ⚠️ Answer quality may be lower for non-English with current English-only embeddings
- ✅ Full quality in all languages after upgrading to multilingual embeddings

---

## Chunking Configuration Guide

### What is Chunking?

Chunking splits large documents into smaller pieces for better retrieval and processing. **Chunk settings significantly impact answer quality!**

### Current Default Settings:
```env
CHUNK_SIZE=800          # ~150-200 words, 2-3 paragraphs
CHUNK_OVERLAP=100       # 12.5% overlap between chunks
```

### How Chunk Size Affects Quality:

| Chunk Size | Best For | Pros | Cons |
|------------|----------|------|------|
| **300-600** | FAQs, snippets, Q&A | Precise retrieval, fast | May fragment ideas |
| **800-1000** | General technical docs | Balanced context/precision | Good all-around |
| **1200-1500** | Dense specs, standards | Complete explanations | Slower retrieval |
| **1500-2000** | Research papers, articles | Preserves narrative | May dilute relevance |

### Why Overlap Matters:

**Without overlap (0):**
```
Chunk 1: "...the solution requires three steps. First,"
Chunk 2: "Second, process the data. Third, validate..."
```
❌ Retrieving Chunk 2 misses "First" step!

**With overlap (10-20%):**
```
Chunk 1: "...the solution requires three steps. First, initialize."
Chunk 2: "...three steps. First, initialize. Second, process..."
```
✅ Important information appears in multiple chunks!

### Recommended Settings by Document Type:

#### **Dense Technical Specifications** (MPEG, ISO, IEEE standards)
```env
CHUNK_SIZE=1200
CHUNK_OVERLAP=200
```
- **Why:** Technical specs need complete multi-paragraph explanations
- **Example:** Algorithm descriptions, performance tables, conformance requirements
- **Impact:** Better context for complex technical questions

#### **Short FAQs / Knowledge Base**
```env
CHUNK_SIZE=500
CHUNK_OVERLAP=75
```
- **Why:** Quick, focused answers without excess context
- **Example:** Troubleshooting guides, quick reference docs
- **Impact:** Faster, more precise retrieval

#### **Long-Form Articles / Research Papers**
```env
CHUNK_SIZE=1500
CHUNK_OVERLAP=300
```
- **Why:** Preserves argument flow and narrative structure
- **Example:** White papers, academic articles, detailed reports
- **Impact:** Maintains logical connections between ideas

#### **Mixed Document Collection** (Recommended)
```env
CHUNK_SIZE=1000
CHUNK_OVERLAP=150
```
- **Why:** Good balance for varied content types
- **Example:** Mix of specs, guides, and reports
- **Impact:** Versatile performance across document types

### How to Adjust Chunking:

1. **Edit `.env` file:**
   ```env
   CHUNK_SIZE=1200
   CHUNK_OVERLAP=200
   ```

2. **Re-ingest your documents:**
   ```powershell
   python rag/ingest.py
   ```

3. **Test with same questions** to compare quality

### Chunk Size Impact on Your System:

| Setting | Total Chunks | Retrieval Speed | Context Quality |
|---------|--------------|-----------------|------------------|
| 500/75 | ~45,000 | Fastest | Fragmented |
| 800/100 | ~29,000 | Fast | Good |
| 1000/150 | ~23,000 | Medium | Better |
| 1500/300 | ~15,000 | Slower | Most Complete |

**Rule of Thumb:** Overlap should be 10-20% of chunk size for optimal results.

---
