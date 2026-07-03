# LLMQwen

A chatbot that answers questions about your own documents. You give it PDFs, Word files, PowerPoints, etc., and you can ask it anything — it finds the relevant parts and writes a precise answer.

Everything runs **locally on your machine**. No data is sent to the cloud.

---

## How it works (simple version)

When you ask a question, the system does four things in order:

1. **Search** — it converts your question into numbers and uses those numbers to find the most relevant pieces of your documents
2. **Re-rank** — it scores each piece more carefully to keep only the best ones
3. **Generate** — it feeds those pieces to the AI model and asks it to write an answer
4. **Display** — the answer streams back to your browser word by word

The AI model (Qwen) runs on your GPU. The search and re-ranking helpers run in Docker containers so they are easy to start and stop.

---

## What you need before starting

### A Linux machine with an NVIDIA GPU

This project runs on **Linux only** (Ubuntu 22.04+ or similar). It requires a powerful NVIDIA graphics card because the AI model is very large.

| What | Minimum | Why |
|---|---|---|
| GPU memory (VRAM) | 16 GB | The AI model needs this to fit in the GPU |
| Extra GPU memory | 4 GB | For the search and re-rank helpers |
| RAM | 32 GB | For loading documents and running everything |
| Disk space | 50 GB free | Models are large files |

> If you only have one GPU that's 24 GB or more (e.g. RTX 3090, RTX 4090), everything can share it.

### Software to install first

You need four things installed before you start. Follow the steps below in order.

---

## Step 0 — Install the required software

### 0a. Check that your NVIDIA driver is working

Open a terminal and run:
```bash
nvidia-smi
```
You should see a table showing your GPU. If you get "command not found", you need to install the NVIDIA driver for your GPU model from [nvidia.com](https://www.nvidia.com/Download/index.aspx).

### 0b. Install Docker

Docker is a tool that runs programs in isolated boxes so they don't interfere with each other. Think of it like a lightweight virtual machine.

```bash
# This command downloads and installs Docker automatically
curl -fsSL https://get.docker.com | sh

# This lets you run Docker without typing "sudo" every time
# You must log out and log back in after running this
sudo usermod -aG docker $USER
```

After logging back in, verify it works:
```bash
docker run hello-world
```
You should see a message saying "Hello from Docker!".

### 0c. Install the NVIDIA plugin for Docker

By default, Docker containers cannot see your GPU. This plugin adds that ability:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify it works (you should see the same GPU table as before):
```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### 0d. Install Python 3.11 or higher

```bash
python3 --version
```

If the version shown is below 3.11, install a newer version:
```bash
# Ubuntu / Debian
sudo apt install python3.11 python3.11-venv python3.11-pip
```

---

## Step 1 — Download the project

```bash
git clone https://github.com/Nozomu05/LLMQwen.git
cd LLMQwen
```

---

## Step 2 — Create a Python environment and install vLLM

vLLM is the program that actually runs the AI model on your GPU. It runs directly on your machine (not in Docker) so it can use the GPU as efficiently as possible.

```bash
# Create an isolated Python environment so packages don't conflict with anything else on your machine
python3 -m venv .venv

# Activate it — you must do this every time you open a new terminal
source .venv/bin/activate

# Install vLLM and its dependencies
pip install vllm
pip install -r requirements.txt
```

---

## Step 3 — Configure the project

Each part of the project has a configuration file. You need to create these files from the provided templates.

Run these four commands:

```bash
cp services/llm/.env.example     services/llm/.env
cp services/reranker/.env.example services/reranker/.env
cp services/search/.env.example   services/search/.env
cp rag/.env.example               rag/.env
```

Now open each `.env` file and change the values marked below.

---

### `services/llm/.env` — controls the AI model

The only thing you **must** change is the model name if you are using a different one:

```
VLLM_MODEL=Qwen/Qwen3-14B-AWQ
```

This must match exactly the model you will start in Step 5. Leave everything else as-is for now.

---

### `services/reranker/.env` — controls the re-ranking helper

Change `HF_CACHE_DIR` and `HF_HOME` to the folder where AI models are stored on your machine. This is usually `~/.cache/huggingface` but replace `~` with your actual home folder path:

```
HF_CACHE_DIR=/home/yourname/.cache/huggingface
HF_HOME=/home/yourname/.cache/huggingface
```

To find your home folder path, run `echo $HOME` in a terminal.

If you only have one GPU, change `cuda:0` to `cpu` so the re-ranker doesn't compete with the AI model for GPU memory:
```
RERANKER_DEVICE=cpu
```

---

### `services/search/.env` — controls the document search

Same as above — change `HF_CACHE_DIR` and `HF_HOME` to your home folder:

```
HF_CACHE_DIR=/home/yourname/.cache/huggingface
HF_HOME=/home/yourname/.cache/huggingface
```

If you only have one GPU, also change:
```
EMBEDDING_DEVICE=cpu
```

---

### `rag/.env` — controls how documents are processed

Same `HF_CACHE_DIR` change, and optionally switch the device to `cpu` if needed:

```
HF_CACHE_DIR=/home/yourname/.cache/huggingface
HF_HOME=/home/yourname/.cache/huggingface
EMBEDDING_DEVICE=cpu   # only if you have one GPU
```

---

## Step 4 — Copy your existing AI model files into Docker (optional, saves time)

When Docker containers start for the first time, they download the AI helper models (about 2–3 GB total). If you already downloaded these models before (they would be in `~/.cache/huggingface`), you can copy them into Docker's storage so they don't download again:

```bash
# Replace /home/yourname with your actual home folder path
docker run --rm \
  -v hf-cache:/dst \
  -v /home/yourname/.cache/huggingface:/src:ro \
  alpine sh -c "cp -r /src/. /dst/"
```

If you skip this step, the models will just download automatically when you first start the services.

---

## Step 5 — Add your documents

Put the files you want to ask questions about into the `docs/` folder.

**Supported file types:** PDF, Word (DOCX), PowerPoint (PPTX), plain text, Markdown, Excel, CSV, OpenDocument (ODT), and ZIP archives (which are extracted automatically).

---

## Step 6 — Process your documents

This step reads all your documents, breaks them into small pieces, and builds a searchable index. You only need to do this once, or again when you add new documents.

```bash
docker compose run --rm ingest
```

The first time this runs it will download the search helper model (~2 GB). You will see progress messages for each document. When it finishes, it will show a summary of how many documents were processed.

---

## Step 7 — Start the AI model

The AI model needs to be started separately. Open a terminal, activate the Python environment, and run:

```bash
source .venv/bin/activate

# If you have two GPUs, this uses the second one (GPU index 1)
# If you only have one GPU, remove "CUDA_VISIBLE_DEVICES=1"
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-14B-AWQ \
  --port 8000 \
  --quantization awq \
  --dtype float16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --trust-remote-code
```

The first time this runs, it downloads the Qwen model (~8 GB). This can take a while depending on your internet speed.

**Wait until you see this message before continuing:**
```
Application startup complete.
```

Keep this terminal open — the model stops if you close it.

---

## Step 8 — Start the three helper services

Open a **new terminal** (keep the one from Step 7 running) and run:

```bash
docker compose up
```

The first time, Docker needs to build the images — this takes a few minutes. After that it will be fast.

You will see log messages from three services starting up. Wait until all three show something like `Application startup complete`.

Keep this terminal open too.

---

## Step 9 — Query the API

Once Steps 7 and 8 are running, ask a question directly against the search service:

```bash
curl -N -X POST http://localhost:8010/query \
  -H "Content-Type: application/json" \
  -d '{"query": "your question here"}'
```

The response streams back as Server-Sent Events (SSE).

---

## Starting again after a reboot

After restarting your computer, do these steps each time:

**Terminal 1 — start the AI model:**
```bash
cd LLMQwen
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-14B-AWQ \
  --port 8000 --quantization awq --dtype float16 \
  --gpu-memory-utilization 0.90 --max-model-len 8192 --trust-remote-code
```

**Terminal 2 — start the helper services:**
```bash
cd LLMQwen
docker compose up
```

Then query the API as shown in Step 9.

---

## Alternative: start everything with one command

Instead of the three terminals above, you can use `serve.sh` which opens everything at once in a tmux session (a tool that manages multiple terminal panels):

```bash
# Install tmux if you don't have it
sudo apt install tmux

# Start everything
bash serve.sh
```

To see what's running, type `tmux attach -t rag`. Use `Ctrl-b` then a number (0–3) to switch between panels.

---

## Adding new documents

1. Copy the new files into `docs/`
2. Re-process them:
   ```bash
   docker compose run --rm ingest
   ```
3. That's it — the search service picks up the updated index automatically

---

## Changing the AI model

1. Stop the AI model (press `Ctrl-C` in terminal 1)
2. Change `VLLM_MODEL` in `services/llm/.env` to the new model name
3. Start the AI model again (Step 7) with the new model name in the command
4. Restart just the LLM service:
   ```bash
   docker compose restart llm
   ```

No need to re-process your documents.

---

## Changing search quality settings

These settings are in `services/search/.env` and take effect after restarting the search service (`docker compose restart search`):

| Setting | What it does | Default |
|---|---|---|
| `RETRIEVAL_CHUNKS` | How many document pieces to consider before filtering | `50` |
| `USE_HYDE` | Generate a fake answer first to improve search accuracy. Slower but better for vague questions. | `false` |

---

## Changing answer quality settings

These settings are in `services/llm/.env` and take effect after restarting (`docker compose restart llm`):

| Setting | What it does | Default |
|---|---|---|
| `TEMPERATURE` | How creative the answers are. `0` = very precise, `1` = very creative | `0.1` |
| `MAX_NEW_TOKENS` | Maximum length of the answer in word-pieces | `1024` |
| `TOP_P` | Cuts off unlikely word choices. `1.0` = no cut-off | `0.95` |
| `FREQUENCY_PENALTY` | Reduces repetition. `0` = none, higher = less repetition | `0.0` |

---

## Uploading images to Docker Hub (for sharing)

If you want to share your setup with someone else so they can pull your ready-made images instead of building from scratch:

```bash
# Log in to Docker Hub
docker login

# Build and upload the three services
docker compose build
docker compose push llm reranker search

# Build and upload the ingestion tool
docker compose --profile ingest build ingest
docker compose --profile ingest push ingest
```

On the other machine, instead of running `docker compose build`, they just run:
```bash
docker compose pull
docker compose --profile ingest pull ingest
```

---

## Something is not working

**"No vector store" error**
You skipped or the document processing failed. Re-run:
```bash
docker compose run --rm ingest
```

**The AI gives no answer or times out**
The AI model (Step 7) may have crashed or not finished loading. Check terminal 1 — it should say `Application startup complete`. If it crashed, restart it.

**Out of GPU memory error**
Your GPU doesn't have enough memory. Try these fixes in order:
1. In `services/reranker/.env`, change `RERANKER_DEVICE=cuda:0` to `RERANKER_DEVICE=cpu`
2. In `services/search/.env`, change `EMBEDDING_DEVICE=cuda:0` to `EMBEDDING_DEVICE=cpu`
3. After changing, run `docker compose restart reranker search`

**Check if the services are running correctly**
These commands should each return `{"status":"ok"}`:
```bash
curl http://localhost:8010/health
curl http://localhost:8011/health
curl http://localhost:8012/health
```
