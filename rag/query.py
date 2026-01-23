import os
import sys
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

try:
    from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
    RERANKING_AVAILABLE = True
except ImportError:
    RERANKING_AVAILABLE = False


def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}" for d in docs)


def main() -> None:
    load_dotenv()

    chroma_dir = Path(os.getenv("CHROMA_DIR", "storage/chroma")).resolve()
    model_name = os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct")

    if not chroma_dir.exists():
        print(f"Vector store directory not found: {chroma_dir}")
        print("Run ingestion first: python .\\rag\\ingest.py")
        sys.exit(1)

    query = " ".join(sys.argv[1:]).strip()
    if not query:
        print("Usage: python .\\rag\\query.py \"<your question>\"")
        sys.exit(1)

    print(f"Loading Chroma from: {chroma_dir}")
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vectorstore = Chroma(persist_directory=str(chroma_dir), embedding_function=embeddings)
    
    k_chunks = int(os.getenv("RETRIEVAL_CHUNKS", "12"))
    top_n = int(os.getenv("TOP_N_RERANK", "6"))
    use_reranking = os.getenv("USE_RERANKING", "true").lower() == "true"
    
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": k_chunks})
    
    print("Retrieving context...")
    docs = base_retriever.invoke(query)
    
    if use_reranking and RERANKING_AVAILABLE and len(docs) > 0:
        print(f"Using reranking to select top {top_n} from {len(docs)} chunks...")
        try:
            compressor = FlashrankRerank(top_n=top_n, model="ms-marco-MiniLM-L-12-v2")
            docs = compressor.compress_documents(docs, query)
            print(f"Reranked to {len(docs)} most relevant chunks")
        except Exception as e:
            print(f"Reranking failed ({e}), using all retrieved chunks")

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert assistant that provides accurate, detailed answers based on the given context.

Instructions:
- Answer ONLY using information from the provided context
- Quote specific passages when making claims
- If the context doesn't contain enough information, clearly state what's missing
- Structure your answer with clear reasoning
- Include relevant numbers, dates, and specific details from the context
- If multiple sources provide different information, mention this"""),
        ("human", """Question: {question}

Context:
{context}

Provide a comprehensive answer based on the context above:""")
    ])

    llm = ChatOllama(model=model_name, base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    chain = prompt | llm

    print(f"Querying model: {model_name}\n")
    context_text = format_docs(docs)
    try:
        result = chain.invoke({"question": query, "context": context_text})
    except Exception as e:
        msg = str(e).lower()
        if "model" in msg and "not found" in msg:
            print(f"Model '{model_name}' not found in Ollama.")
            print("Pull it first (example):")
            print(f"  ollama pull {model_name}")
            print("CPU-friendly alternative:")
            print("  ollama pull qwen2.5:3b-instruct")
            print("Re-run the query after pulling.")
            sys.exit(1)
        raise

    print("=== Answer ===\n")
    print(result.content)

    print("\n=== Sources ===")
    for i, d in enumerate(docs, 1):
        print(f"{i}. {d.metadata.get('source', 'unknown')}")


if __name__ == "__main__":
    main()
