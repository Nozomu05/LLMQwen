import os
import sys
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
import time

try:
    from langchain_mistralai import ChatMistralAI
    MISTRAL_AVAILABLE = True
except ImportError:
    MISTRAL_AVAILABLE = False

try:
    from langchain_openai import ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
    RERANKING_AVAILABLE = True
except ImportError:
    RERANKING_AVAILABLE = False


def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}" for d in docs)


def get_llm(provider: str):
    if provider == "ollama":
        model_name = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(model=model_name, base_url=base_url), model_name
    
    elif provider == "mistral":
        if not MISTRAL_AVAILABLE:
            print("Error: Mistral support not available. Install with: pip install langchain-mistralai")
            sys.exit(1)
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key or api_key == "your_mistral_api_key_here":
            print("Error: MISTRAL_API_KEY not set in .env file")
            sys.exit(1)
        model_name = os.getenv("MISTRAL_MODEL", "mistral-large-latest")
        return ChatMistralAI(model=model_name, api_key=api_key, temperature=0), model_name
    
    elif provider == "openai":
        if not OPENAI_AVAILABLE:
            print("Error: OpenAI support not available. Install with: pip install langchain-openai")
            sys.exit(1)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "your_openai_api_key_here":
            print("Error: OPENAI_API_KEY not set in .env file")
            sys.exit(1)
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return ChatOpenAI(model=model_name, api_key=api_key, temperature=0), model_name
    
    else:
        print(f"Error: Unknown provider '{provider}'. Use: ollama, mistral, or openai")
        sys.exit(1)


def run_query_complete(question: str, provider: str = "ollama") -> tuple[str, str, List[str]]:
    load_dotenv()
    
    chroma_dir = Path(os.getenv("CHROMA_DIR", "storage/chroma")).resolve()
    
    if not chroma_dir.exists():
        raise FileNotFoundError(f"Vector store not found at {chroma_dir}. Run ingestion first.")
    
    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "fastembed").lower()
    
    if embedding_provider == "ollama":
        embedding_model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        embeddings = OllamaEmbeddings(model=embedding_model, base_url=base_url)
    else:  
        use_faster = os.getenv("USE_FASTER_EMBEDDINGS", "false").lower() == "true"
        embed_model = "BAAI/bge-small-en-v1.5" if not use_faster else "sentence-transformers/all-MiniLM-L6-v2"
        embeddings = FastEmbedEmbeddings(model_name=embed_model, max_length=512)
    
    vectorstore = Chroma(persist_directory=str(chroma_dir), embedding_function=embeddings)
    
    k_chunks = int(os.getenv("RETRIEVAL_CHUNKS", "100"))
    top_n = int(os.getenv("TOP_N_RERANK", "8"))
    use_reranking = os.getenv("USE_RERANKING", "true").lower() == "true"
    
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": k_chunks})
    docs = base_retriever.invoke(question)
    
    if use_reranking and RERANKING_AVAILABLE and len(docs) > 0:
        try:
            compressor = FlashrankRerank(top_n=top_n, model="ms-marco-MiniLM-L-12-v2")
            docs = compressor.compress_documents(docs, question)
        except Exception:
            pass
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert assistant that provides accurate, deeply reasoned answers based on the given context.

Instructions:
- Answer ONLY using information from the provided context
- Use step-by-step reasoning: analyze the question, identify relevant information, synthesize conclusions
- Compare and contrast different approaches or perspectives found in the context
- Identify implications, trade-offs, and relationships between concepts
- Quote specific passages when making claims and cite the source
- Break down complex topics into clear sections with logical flow
- Include ALL relevant numbers, dates, technical terms, and specific details
- Explain WHY things work the way they do, not just WHAT they are
- When evaluating suggestions or proposals: assess feasibility, identify gaps, compare with existing work
- If the context provides examples, analyze them and explain their significance
- If multiple sources provide different information, analyze the differences and explain why they might exist
- If the context doesn't contain enough information, explain what's missing and why it matters
- Use analytical frameworks: pros/cons, before/after, cause/effect"""),
        ("human", """Question: {question}

Context:
{context}

Provide a comprehensive, deeply reasoned answer:
1. First, analyze what the question is really asking
2. Then, examine the relevant information from the context
3. Finally, synthesize your conclusions with clear reasoning and evidence""")
    ])
    
    llm, model_name = get_llm(provider)
    chain = prompt | llm
    context_text = format_docs(docs)
    result = chain.invoke({"question": question, "context": context_text})
    
    sources = [d.metadata.get("source", "unknown") for d in docs]
    
    return result.content, model_name, sources


def main() -> None:
    load_dotenv()

    chroma_dir = Path(os.getenv("CHROMA_DIR", "storage/chroma")).resolve()
    provider = os.getenv("MODEL_PROVIDER", "ollama").lower()
    
    print(f"Using provider: {provider}")

    if not chroma_dir.exists():
        print(f"Vector store directory not found: {chroma_dir}")
        print("Run ingestion first: python .\\rag\\ingest.py")
        sys.exit(1)

    query = " ".join(sys.argv[1:]).strip()
    if not query:
        print("Usage: python .\\rag\\query.py \"<your question>\"")
        sys.exit(1)

    print(f"Loading Chroma from: {chroma_dir}")
    
    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "ollama").lower()
    
    start_time = time.time()
    if embedding_provider == "ollama":
        embedding_model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        embeddings = OllamaEmbeddings(model=embedding_model, base_url=base_url)
    else:
        use_faster = os.getenv("USE_FASTER_EMBEDDINGS", "false").lower() == "true"
        embed_model = "BAAI/bge-small-en-v1.5" if not use_faster else "sentence-transformers/all-MiniLM-L6-v2"
        embeddings = FastEmbedEmbeddings(model_name=embed_model, max_length=512)
    
    vectorstore = Chroma(persist_directory=str(chroma_dir), embedding_function=embeddings)
    print(f"Loaded in {time.time() - start_time:.2f}s")
    
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
        ("system", """You are an expert assistant that provides accurate, deeply reasoned answers based on the given context.

Instructions:
- Answer ONLY using information from the provided context
- Use step-by-step reasoning: analyze the question, identify relevant information, synthesize conclusions
- Compare and contrast different approaches or perspectives found in the context
- Identify implications, trade-offs, and relationships between concepts
- Quote specific passages when making claims and cite the source
- Break down complex topics into clear sections with logical flow
- Include ALL relevant numbers, dates, technical terms, and specific details
- Explain WHY things work the way they do, not just WHAT they are
- When evaluating suggestions or proposals: assess feasibility, identify gaps, compare with existing work
- If the context provides examples, analyze them and explain their significance
- If multiple sources provide different information, analyze the differences and explain why they might exist
- If the context doesn't contain enough information, explain what's missing and why it matters
- Use analytical frameworks: pros/cons, before/after, cause/effect"""),
        ("human", """Question: {question}

Context:
{context}

Provide a comprehensive, deeply reasoned answer:
1. First, analyze what the question is really asking
2. Then, examine the relevant information from the context
3. Finally, synthesize your conclusions with clear reasoning and evidence""")
    ])

    llm, model_name = get_llm(provider)
    chain = prompt | llm

    print(f"Querying model: {model_name}\n")
    context_text = format_docs(docs)
    
    print("=== Answer ===\n")
    query_start = time.time()
    
    try:
        for chunk in chain.stream({"question": query, "context": context_text}):
            print(chunk.content, end="", flush=True)
        print(f"\n\n[Query completed in {time.time() - query_start:.2f}s]")
        return
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

    print("\n=== Sources ===")
    for i, d in enumerate(docs, 1):
        print(f"{i}. {d.metadata.get('source', 'unknown')}")


if __name__ == "__main__":
    main()
