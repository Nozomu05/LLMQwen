FROM python:3.11-slim

# System libraries required by document loaders
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    libmagic1 \
    pandoc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY rag/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY rag/ rag/

CMD ["python", "rag/ingest.py"]
