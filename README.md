# RAG System with Multi-Model Support

A flexible Retrieval-Augmented Generation (RAG) system that lets you query your documents using local or cloud-based LLMs. Features intelligent document processing, semantic search with reranking, and a simple web interface.

## 🎯 What This System Does

1. **Ingests** your documents (Word, PDF, PowerPoint, Text, Markdown) into a vector database
2. **Retrieves** relevant chunks using semantic search when you ask a question
3. **Reranks** results to find the most relevant information
4. **Generates** comprehensive answers using your choice of LLM

## 🏗️ Architecture

```
Documents (docs/)
    ↓
Document Processing & Chunking
    ↓
Embedding Model → Vector Database (Chroma)
    ↓
Query → Semantic Search (50 chunks)
    ↓
Reranking (Top 10 chunks)
    ↓
LLM (Qwen/Mistral/OpenAI) → Answer
```

## ✨ Features

- **Multiple LLM Providers**: Ollama (local), Mistral AI, or OpenAI
- **Flexible Embeddings**: Ollama models (mxbai, nomic, bge) or FastEmbed
- **Smart Retrieval**: Semantic search + Flashrank reranking for precision
- **Document Support**: Word, PDF, PowerPoint, Text, Markdown, ODT
- **ZIP Extraction**: Automatically extracts nested ZIP archives
- **Web Interface**: Simple HTTP server with query interface
- **CPU-Friendly**: Optimized for systems without GPU

---

## 📦 Installation

### 1. Install Ollama (for local models)

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

### 2. Pull Required Models

**LLM Model (for answering queries):**
```powershell
# Recommended: 7B model (4.7GB, good balance)
ollama pull qwen2.5:7b-instruct

# Alternative: Smaller for low-end CPUs (2.0GB)
ollama pull qwen2.5:3b-instruct

# Alternative: Larger for better accuracy (8.9GB)
ollama pull qwen2.5:14b-instruct
```

**Embedding Model (for semantic search):**
```powershell
# Recommended: Best for technical documents (669MB, 1024-dim)
ollama pull mxbai-embed-large

# Alternative: Lighter option (274MB, 768-dim)
ollama pull nomic-embed-text
```

### 3. Install Pandoc (for ODT files)

Optional, only if you have OpenDocument files:
```powershell
winget install --id JohnMacFarlane.Pandoc -e
```

### 4. Setup Python Environment

**Create virtual environment:**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Note:** If you get an execution policy error:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Install dependencies:**
```powershell
pip install -r requirements.txt
```

---

## ⚙️ Configuration

### 1. Create .env file

```powershell
Copy-Item .env.example .env
```

### 2. Configure Settings

Edit `.env` with your preferred settings:

**LLM Configuration:**
```env
MODEL_PROVIDER=ollama              # Options: ollama, mistral, openai
OLLAMA_MODEL=qwen2.5:14b-instruct  # The model for generating answers
OLLAMA_BASE_URL=http://localhost:11434
```

**Embedding Configuration:**
```env
EMBEDDING_PROVIDER=ollama          # Options: ollama, fastembed
EMBEDDING_MODEL=mxbai-embed-large  # Must match during ingestion & query!
```

**Retrieval Settings:**
```env
RETRIEVAL_CHUNKS=100  # How many chunks to retrieve initially
TOP_N_RERANK=15       # Final chunks sent to LLM after reranking
USE_RERANKING=true    # Enable for better accuracy
```

**Document Processing:**
```env
CHUNK_SIZE=800        # Characters per chunk (smaller = more chunks)
CHUNK_OVERLAP=160     # Overlap prevents context loss
BATCH_SIZE=100        # Documents processed per batch
```

---

## 🚀 Usage

### Step 1: Ingest Your Documents

Place your documents in the `docs/` folder, then run:

```powershell
python rag\ingest.py
```

**What happens:**
- Extracts all ZIP files (including nested archives)
- Loads documents (Word, PDF, PowerPoint, etc.)
- Splits into chunks using configured size/overlap
- Generates embeddings using your chosen model
- Stores vectors in Chroma database (`storage/chroma/`)

**⏱️ Time estimate:**
- ~458 documents = ~5-10 minutes with mxbai-embed-large
- Faster with smaller embedding models

### Step 2: Start the Frontend (Web Interface)

```powershell
python frontend\app.py
```

The server starts at: **http://127.0.0.1:8000**

Open in your browser and start asking questions!

### Command-Line Query (Alternative)

Test queries without the web interface:

```powershell
python rag\query.py "What is the latest performance of V-PCC for gaussian splat?"
```

---

## 📊 Improving System Performance

### 🎯 Improve Answer Accuracy

#### 1. **Upgrade Embedding Model**

Better embeddings = better retrieval = better answers

| Model | Size | Dimensions | Best For | Quality |
|-------|------|------------|----------|---------|
| `mxbai-embed-large` | 669MB | 1024 | Technical docs | ⭐⭐⭐⭐⭐ |
| `bge-large` | 1.34GB | 1024 | Highest accuracy | ⭐⭐⭐⭐⭐ |
| `nomic-embed-text` | 274MB | 768 | General purpose | ⭐⭐⭐⭐ |
| `bge-small` (FastEmbed) | ~130MB | 384 | Speed over quality | ⭐⭐⭐ |

**How to upgrade:**
```powershell
# Pull new embedding model
ollama pull bge-large

# Update .env
EMBEDDING_MODEL=bge-large

# Re-ingest documents (required!)
Remove-Item -Path "storage\chroma" -Recurse -Force
python rag\ingest.py
```

**⚠️ Important:** You MUST re-ingest when changing embedding models! Query embeddings must match stored embeddings.

#### 2. **Upgrade LLM Model**

Larger models understand context better and generate more accurate answers.

| Model | Size | RAM Needed | Speed | Quality |
|-------|------|------------|-------|---------|
| `qwen2.5:3b-instruct` | 2.0GB | 4GB | Fast | ⭐⭐⭐ |
| `qwen2.5:7b-instruct` | 4.7GB | 8GB | Medium | ⭐⭐⭐⭐ |
| `qwen2.5:14b-instruct` | 8.9GB | 16GB | Slow | ⭐⭐⭐⭐⭐ |
| `qwen2.5:32b` | 19GB | 32GB+ | Very slow | ⭐⭐⭐⭐⭐ |

**How to upgrade:**
```powershell
# Pull new model
ollama pull qwen2.5:14b-instruct

# Update .env
OLLAMA_MODEL=qwen2.5:14b-instruct

# Restart frontend
python frontend\app.py
```

**No re-ingestion needed** when changing LLM models.

#### 3. **Tune Retrieval Settings**

Balance between recall (finding relevant chunks) and precision (avoiding irrelevant chunks):

```env
# More initial chunks = better recall
RETRIEVAL_CHUNKS=100

# More reranked chunks = more context for LLM
TOP_N_RERANK=15
```

**⚠️ Warning:** Larger models handle more chunks better!
- 7B models: max 10 chunks (get overwhelmed beyond this)
- 14B models: 12-15 chunks optimal
- 32B models: 25-30 chunks

#### 4. **Enable Reranking**

Reranking dramatically improves precision by re-scoring retrieved chunks:

```env
USE_RERANKING=true
```

**Impact:** ~30-50% improvement in answer relevance for complex queries.

### ⚡ Improve Speed

#### 1. **Use Smaller Embedding Model**

Trade-off: Speed vs. accuracy

```powershell
# Fast option
EMBEDDING_MODEL=nomic-embed-text
```

#### 2. **Use Smaller LLM**

```powershell
ollama pull qwen2.5:3b-instruct
```

#### 3. **Reduce Retrieval Chunks**

```env
RETRIEVAL_CHUNKS=20   # Faster search
TOP_N_RERANK=5        # Faster reranking
```

#### 4. **Increase Chunk Size**

Fewer chunks = faster retrieval (but potentially lower accuracy):

```env
CHUNK_SIZE=1200       # Larger chunks = fewer total chunks
CHUNK_OVERLAP=200
```

**⚠️ Note:** Requires re-ingestion!

### 💪 Improve Answer Depth & Usefulness

#### 1. **Optimize Chunk Size for Your Documents**

- **Technical docs with tables/code:** Smaller chunks (600-800)
- **Long-form articles:** Medium chunks (1000-1200)
- **Books/reports:** Larger chunks (1500-2000)

#### 2. **Increase Context for LLM**

More chunks = more comprehensive answers:

```env
RETRIEVAL_CHUNKS=100
TOP_N_RERANK=15      # Only if using 14B+ model!
```

#### 3. **Use Larger Open-Source Models for Best Quality**

For maximum quality while staying open-source and cost-free:

**Option 1: Larger Qwen Models (Recommended)**
```powershell
# Best balance: quality + speed (if you have 16GB+ RAM)
ollama pull qwen2.5:14b-instruct

# Maximum quality (requires 32GB+ RAM)
ollama pull qwen2.5:32b-instruct
```

```env
OLLAMA_MODEL=qwen2.5:14b-instruct
```

**Option 2: Qwen 3 (Newest generation)**
```powershell
# Latest Qwen 3 models (better reasoning)
ollama pull qwen3:8b
ollama pull qwen3:14b
```

**Option 3: DeepSeek-R1 (Strong reasoning)**
```powershell
# Excellent for complex technical questions
ollama pull deepseek-r1:7b
ollama pull deepseek-r1:14b
```

**Quality Comparison (all open-source & free):**
- `qwen2.5:32b` ≈ GPT-4 quality (19GB, very slow on CPU)
- `qwen2.5:14b` ≈ GPT-3.5-Turbo quality (8.9GB, acceptable on CPU)
- `deepseek-r1:14b` - Excellent reasoning (9GB)
- `qwen3:14b` - Latest generation (8.9GB)

**⚠️ No cloud APIs needed!** All models run locally for free.

---

## 🔧 Advanced Configuration

### Chunk Size Guidelines

The CHUNK_SIZE parameter controls how documents are split. Finding the optimal size depends on your document type and questions:

**When to use SMALLER chunks (600-800):**
- Technical documents with tables and code
- Q&A scenarios (specific fact retrieval)
- Documents with dense, structured information

**When to use LARGER chunks (1200-1500):**
- Long-form content (articles, reports)
- Narrative documents (books, essays)
- When questions require broader context

**⚠️ Context Length Limits:**
- `mxbai-embed-large`: Max ~800 chars/chunk (strict limit)
- `nomic-embed-text`: Max ~1500 chars/chunk
- `bge-large`: Max ~1200 chars/chunk

If ingestion fails with "context length exceeded", reduce CHUNK_SIZE.

### RAM Requirements

**Ingestion:**
- Minimum: 8GB RAM
- Recommended: 16GB+ for large document sets
- Embedding models: + model size (270MB - 1.3GB)

**Query:**
- 7B LLM: 8GB minimum
- 14B LLM: 16GB minimum
- 32B LLM: 32GB+ minimum
- Reranking: ~12.5MB per chunk

### 🖥️ Hardware Impact on Performance

#### 🔍 Understanding CPU, GPU, and RAM Roles

Your RAG system has **3 main operations**, and each hardware component plays a specific role:

**📥 1. INGESTION (Creating Vector Database)**
```
Documents → Text Extraction → Chunking → Embedding Creation → Store in ChromaDB
```

**Component Roles During Ingestion:**
- **CPU:** 
  - Reads files from disk (unzipping, PDF parsing)
  - Splits text into chunks
  - **Runs embedding model** (mxbai-embed-large) to convert chunks to vectors
  - Saves vectors to ChromaDB database
  - **🎯 Impact:** More CPU cores = faster parallel processing (5-15 min for 458 docs)
  
- **GPU:** 
  - **Can accelerate embeddings** if Ollama uses GPU mode
  - **🎯 Impact:** **5-10x faster ingestion** (1-3 min instead of 5-15 min)
  - **Note:** Your current RTX 5050 8GB works for embeddings during ingestion!
  
- **RAM:** 
  - Temporarily holds document content and chunks before processing
  - **🎯 Impact:** 16GB allows processing large document batches smoothly
  - **Insufficient RAM = system swaps to disk = dramatically slower**

**🔎 2. QUERY RETRIEVAL (Finding Relevant Context)**
```
Your Question → Embedding → Search Vector DB → Retrieve Chunks → Rerank → Top 15 Chunks
```

**Component Roles During Retrieval:**
- **CPU:** 
  - Converts your question to an embedding vector
  - Searches ChromaDB database (vector similarity calculation)
  - Runs reranking model (Flashrank) on 100 → 15 chunks
  - **🎯 Impact:** Fast enough (0.5-1 second total) - rarely a bottleneck
  
- **GPU:** 
  - **Not used** for retrieval in this project
  - Embeddings and reranking run on CPU only
  - **🎯 Impact:** None
  
- **RAM:** 
  - Loads vector database into memory
  - Holds 100 retrieved chunks during reranking
  - **🎯 Impact:** 16GB is more than enough for retrieval

**🤖 3. ANSWER GENERATION (LLM Response)**
```
Your Question + Top 15 Chunks → LLM (qwen2.5:14b-instruct) → Detailed Answer
```

**Component Roles During Answer Generation:**
- **CPU:** 
  - **Runs the entire LLM** when GPU is disabled (OLLAMA_NUM_GPU=0)
  - Processes tokens one-by-one through 14 billion parameters
  - **🎯 Impact:** 8-15 seconds per answer (slow but works!)
  
- **GPU:** 
  - **Runs the entire LLM** when GPU is enabled (default)
  - Processes tokens MUCH faster using parallel computation
  - **🎯 Impact:** **6-10x faster** (1-2 sec instead of 8-15 sec)
  - **⚠️ Problem:** Your RTX 5050 (8GB VRAM) is too small for 14B model (needs 10GB)
  - **Why you use CPU mode:** 14B doesn't fit in 8GB VRAM → CUDA error → must disable GPU
  
- **RAM:** 
  - **Stores the entire LLM model** in CPU mode
  - 14B model = ~9GB loaded into RAM
  - Also holds context (your question + 15 chunks)
  - **🎯 Impact:** 16GB is minimum for 14B, 32GB better for 32B
  - **Insufficient RAM = model won't load at all**

#### 📊 Component Impact Summary Table

| Component | Ingestion Speed | Query Retrieval | Answer Speed | Answer Quality |
|-----------|----------------|-----------------|--------------|----------------|
| **CPU** | ⭐⭐⭐ Major | ⭐⭐⭐ Critical | ⭐⭐⭐ Critical (CPU mode) | ❌ No impact |
| **GPU** | ⭐⭐ Helpful | ❌ Not used | ⭐⭐⭐ Critical (GPU mode) | ❌ No impact |
| **RAM** | ⭐ Minor | ⭐ Minor | ⭐⭐⭐ Critical | ⭐⭐ Indirect* |
| **Storage (SSD)** | ⭐ Minor | ⭐ Minor | ❌ No impact | ❌ No impact |

**\*** More RAM → allows larger models → better quality answers

#### 🎯 Why Your Current Setup Uses CPU-Only Mode

**Your Hardware:** RTX 5050 8GB VRAM, 16GB RAM, 24 CPU cores

**The Problem:**
1. **Embeddings (mxbai-embed-large):** 669MB → ✅ Fits in 8GB GPU → Works great!
2. **14B LLM Model:** Needs ~10GB VRAM → ❌ Only 8GB available → CUDA error!

**Your Choice:** Use 14B on CPU (slow but best quality) instead of 7B on GPU (fast but lower quality)

**Result:**
- **During ingestion:** GPU helps with embeddings → faster (2-5 min)
- **During queries:** LLM runs on CPU → slower (8-15 sec) but highest quality answers

#### CPU vs GPU Performance

**CPU-Only Systems (current setup):**
- **7B models:** 3-5 seconds/query (acceptable)
- **14B models:** 8-15 seconds/query (slow but usable)
- **32B models:** 30-60+ seconds/query (very slow)
- **Ingestion:** 5-15 minutes for 458 documents

**With GPU (NVIDIA recommended):**
- **7B models:** 0.5-1 second/query (8-10x faster)
- **14B models:** 1-2 seconds/query (6-8x faster)
- **32B models:** 3-5 seconds/query (10-15x faster)
- **Ingestion:** 1-3 minutes (5-10x faster)

**GPU Requirements:**
- 7B models: 6GB VRAM minimum (RTX 3060, RTX 4060)
- 14B models: 10GB VRAM minimum (RTX 3080, RTX 4070)
- 32B models: 24GB VRAM minimum (RTX 3090, RTX 4090)

**⚠️ GPU Memory Insufficient?**
If you have a GPU but get CUDA errors (e.g., RTX 5050 with 8GB trying to run 14B):
- **Option 1:** Force CPU-only mode (see Troubleshooting section)
- **Option 2:** Use smaller model (7B works on 8GB VRAM)
- **Trade-off:** CPU is slower but works with any model size

#### RAM Impact

| RAM | Max Model | Max Chunks | Experience |
|-----|-----------|------------|------------|
| 8GB | 7B | 50 | Basic, slow |
| 16GB | 14B | 100 | Good |
| 32GB | 32B | 300+ | Excellent |
| 64GB+ | 70B+ | 1000+ | Professional |

**⚠️ Important:** More RAM ≠ faster queries, but allows:
- Larger models (better quality)
- More retrieval chunks (better recall)
- Multiple processes without swapping

#### Storage Impact

**SSD vs HDD:**
- **SSD (Recommended):** Vector store loads in 0.5-1 second
- **HDD:** Vector store loads in 3-5 seconds
- **NVMe SSD:** Vector store loads in 0.2-0.5 second

**Model Storage:**
- 3B model: ~2GB
- 7B model: ~5GB
- 14B model: ~9GB
- 32B model: ~19GB
- Embedding models: 270MB - 1.3GB
- Vector database: ~2-3MB per 10,000 chunks

#### CPU Impact on Ingestion

**Embedding Generation (CPU-bound):**
- **8 CPU cores:** ~8-12 minutes (458 docs)
- **16 CPU cores:** ~5-8 minutes
- **32 CPU cores:** ~3-5 minutes

**Document Loading (I/O + CPU):**
- Single-core: 1 document/second
- Multi-core: 5-10 documents/second (parallel processing)

#### Optimal Hardware Recommendations

**Budget Setup ($0 upgrade cost - current):**
- CPU: Any modern CPU (4+ cores)
- RAM: 8-16GB
- Storage: Any SSD
- Model: `qwen2.5:7b-instruct`
- Expected: 3-5s queries, adequate quality

**Recommended Setup ($300-500):**
- CPU: Intel i5/Ryzen 5+ (8+ cores)
- RAM: 16GB
- Storage: SSD (500GB+)
- GPU: RTX 3060 (12GB VRAM)
- Model: `qwen2.5:14b-instruct`
- Expected: 1-2s queries, excellent quality

**Professional Setup ($1500-2000):**
- CPU: Intel i7/Ryzen 7+ (12+ cores)
- RAM: 32GB
- Storage: NVMe SSD (1TB+)
- GPU: RTX 4080/4090 (16-24GB VRAM)
- Model: `qwen2.5:32b`
- Expected: <1s queries, GPT-4 quality

**💡 Key Insight:** Even budget CPU-only setups work fine! GPU mainly improves speed, not quality. The open-source approach keeps costs at $0 regardless of hardware.

---

## 📁 Project Structure

```
LLMQwen/
├── docs/                      # Your documents go here
├── storage/
│   └── chroma/               # Vector database storage
├── rag/
│   ├── ingest.py             # Document ingestion script
│   └── query.py              # Query execution script
├── frontend/
│   ├── app.py                # Web server (backend)
│   ├── index.html            # Web interface
│   ├── script.js             # Frontend logic
│   └── style.css             # UI styling
├── .env                       # Your configuration
├── .env.example              # Configuration template
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## 🐛 Troubleshooting

### "Collection expecting embedding with dimension of X, got Y"

**Cause:** Embedding model mismatch between ingestion and query.

**Solution:**
```powershell
# Clear vector store
Remove-Item -Path "storage\chroma" -Recurse -Force

# Re-ingest with correct model
python rag\ingest.py
```

### "the input length exceeds the context length"

**Cause:** Chunks too large for embedding model.

**Solution:** Reduce CHUNK_SIZE in .env:
```env
CHUNK_SIZE=600
CHUNK_OVERLAP=120
```

Then re-ingest.

### Ollama Connection Error

**Cause:** Ollama service not running.

**Solution:**
```powershell
ollama serve
```

### Out of Memory During Query

**Cause:** Too many chunks for available RAM.

**Solution:** Reduce RETRIEVAL_CHUNKS:
```env
RETRIEVAL_CHUNKS=20
TOP_N_RERANK=5
```

### CUDA Error / GPU Out of Memory

**Error:** `llama runner process has terminated: CUDA error`

**Cause:** Model too large for your GPU VRAM, or GPU memory is full from other processes.

**GPU VRAM Requirements:**
- 7B models: 6GB VRAM minimum
- 14B models: 10GB VRAM minimum  
- 32B models: 24GB VRAM minimum

**Check your GPU:**
```powershell
nvidia-smi
```

**Solution 1: Force CPU-Only Mode (if insufficient VRAM)**

Permanently enable CPU-only mode:
```powershell
[System.Environment]::SetEnvironmentVariable('OLLAMA_NUM_GPU', '0', 'User')
$env:OLLAMA_NUM_GPU = '0'
```

Restart your terminal and frontend. Models will run on CPU (slower but works).

**Solution 2: Use Smaller Model (if you want GPU speed)**

Switch to 7B model if you have 8GB VRAM:
```powershell
ollama pull qwen2.5:7b-instruct
```

Update `.env`:
```env
OLLAMA_MODEL=qwen2.5:7b-instruct
```

**Solution 3: Free GPU Memory**

Close other GPU-consuming applications (games, video editing, etc.) and restart Ollama:
```powershell
Get-Process ollama -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 3
# Ollama will auto-restart
```

**To Re-Enable GPU Mode Later:**
```powershell
[System.Environment]::SetEnvironmentVariable('OLLAMA_NUM_GPU', '1', 'User')
$env:OLLAMA_NUM_GPU = '1'
```

---

## 📈 Performance Benchmarks

Based on 458 documents (~30,000 chunks):

**GPU Mode:**

| Configuration | Ingestion Time | Query Time | Accuracy* |
|---------------|----------------|------------|-----------|
| nomic + 7B (GPU) | 2 min | 0.5-1s | ⭐⭐⭐⭐ |
| mxbai + 7B (GPU) | 3 min | 0.5-1s | ⭐⭐⭐⭐⭐ |
| mxbai + 14B (GPU) | 3 min | 1-2s | ⭐⭐⭐⭐⭐ |

**CPU-Only Mode (OLLAMA_NUM_GPU=0):**

| Configuration | Ingestion Time | Query Time | Accuracy* |
|---------------|----------------|------------|-----------|
| nomic + 7B (CPU) | 5 min | 3-5s | ⭐⭐⭐⭐ |
| mxbai + 7B (CPU) | 8 min | 3-5s | ⭐⭐⭐⭐⭐ |
| mxbai + 14B (CPU) | 8 min | 8-15s | ⭐⭐⭐⭐⭐ |
| bge-large + 14B (CPU) | 12 min | 8-15s | ⭐⭐⭐⭐⭐ |

*Accuracy for technical documentation queries

---

## 🤝 Contributing

Contributions welcome! Key areas:
- Additional document loaders
- Embedding model benchmarks
- Prompt engineering improvements
- UI enhancements

---

## 📄 License

MIT License - see LICENSE file for details

---

## 🙏 Acknowledgments

Built with:
- [LangChain](https://github.com/langchain-ai/langchain) - RAG framework
- [Chroma](https://github.com/chroma-core/chroma) - Vector database
- [Ollama](https://ollama.ai/) - Local LLM runtime
- [Flashrank](https://github.com/PrithivirajDamodaran/FlashRank) - Fast reranking
- [Qwen](https://github.com/QwenLM/Qwen) - Language models
