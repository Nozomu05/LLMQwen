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
    from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
    RERANKING_AVAILABLE = True
except ImportError:
    RERANKING_AVAILABLE = False


def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}" for d in docs)


def get_llm():
    model_name = os.getenv("OLLAMA_MODEL", "qwen2.5:14b-instruct")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    return ChatOllama(model=model_name, base_url=base_url), model_name


def run_query_complete(question: str) -> tuple[str, str, List[str]]:
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
        ("system", """You are an expert technical assistant that provides extremely detailed, comprehensive, and in-depth answers based on the given context.

CRITICAL INSTRUCTIONS - YOUR ANSWERS MUST BE DETAILED AND THOROUGH:

📝 LENGTH & DEPTH REQUIREMENTS:
- Write LONG, COMPREHENSIVE answers (minimum 300-500 words)
- Go into EXTENSIVE technical detail on every relevant point
- Provide COMPLETE explanations, not summaries
- Expand on ALL key concepts with thorough background information
- Include detailed methodology, implementation specifics, and technical reasoning

🔬 TECHNICAL DETAIL REQUIREMENTS:
- Include ALL relevant numbers, metrics, percentages, and quantitative data
- Explain technical terminology and concepts in depth
- Describe methodologies, algorithms, and approaches thoroughly
- Discuss implementation details, constraints, and trade-offs extensively
- Compare different approaches with detailed analysis
- Provide context for why specific choices or values were used

📊 STRUCTURE YOUR DETAILED ANSWER:
1. **Introduction/Overview**: Explain the context and scope (2-3 paragraphs)
2. **Main Analysis**: Deep dive into each aspect (multiple detailed paragraphs)
   - Break down complex topics into subsections
   - Provide step-by-step explanations
   - Include specific examples with full details
3. **Technical Details**: Explain HOW and WHY things work
   - Describe underlying mechanisms
   - Discuss implications and consequences
   - Compare alternatives when relevant
4. **Synthesis**: Connect different pieces of information
   - Identify patterns and relationships
   - Discuss broader implications
5. **Conclusion**: Summarize key insights comprehensively

💡 ANALYTICAL DEPTH:
- Don't just state facts - EXPLAIN THE REASONING behind them
- Analyze cause-and-effect relationships thoroughly
- Discuss trade-offs, advantages, and limitations in detail
- Compare and contrast different approaches extensively
- Identify implications for practice, implementation, or future work
- When citing numbers, explain their significance and context

📚 USE THE CONTEXT FULLY:
- Quote specific passages with proper citations [Source: filename]
- Reference multiple sources when they provide complementary information
- If sources disagree, explain the differences in detail
- Extract and explain ALL relevant technical details from the context

⚠️ QUALITY OVER BREVITY:
- NEVER give short, superficial answers
- Each paragraph should be substantial (5-7 sentences minimum)
- Elaborate on every important point
- Think "university lecture" not "quick summary"
- If the context is rich in detail, your answer should be too

Remember: The user wants DETAILED, IN-DEPTH, COMPREHENSIVE answers. More detail is ALWAYS better than less."""),
        ("human", """Question: {question}

Context:
{context}

⚡ IMPORTANT: Provide an EXTREMELY DETAILED, COMPREHENSIVE, and IN-DEPTH answer. Write at length with thorough explanations.

Follow this structure for maximum detail:
1. **Analyze the Question**: What exactly is being asked? What are the key components?
2. **Examine the Context**: What relevant information is available? What technical details are provided?
3. **Provide Comprehensive Answer**: Write a LONG, DETAILED response covering all aspects
4. **Technical Deep Dive**: Go into extensive technical specifics
5. **Synthesis & Conclusions**: Connect all information with thorough reasoning

Write your answer now (aim for 300-500+ words with extensive technical detail):""")
    ])
    
    llm, model_name = get_llm()
    chain = prompt | llm
    context_text = format_docs(docs)
    result = chain.invoke({"question": question, "context": context_text})
    
    sources = [d.metadata.get("source", "unknown") for d in docs]
    
    return result.content, model_name, sources


def main() -> None:
    load_dotenv()

    chroma_dir = Path(os.getenv("CHROMA_DIR", "storage/chroma")).resolve()

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
        ("system", """You are an expert technical assistant that provides extremely detailed, comprehensive, and in-depth answers based on the given context.

CRITICAL INSTRUCTIONS - YOUR ANSWERS MUST BE DETAILED AND THOROUGH:

📝 LENGTH & DEPTH REQUIREMENTS:
- Write LONG, COMPREHENSIVE answers (minimum 300-500 words)
- Go into EXTENSIVE technical detail on every relevant point
- Provide COMPLETE explanations, not summaries
- Expand on ALL key concepts with thorough background information
- Include detailed methodology, implementation specifics, and technical reasoning

🔬 TECHNICAL DETAIL REQUIREMENTS:
- Include ALL relevant numbers, metrics, percentages, and quantitative data
- Explain technical terminology and concepts in depth
- Describe methodologies, algorithms, and approaches thoroughly
- Discuss implementation details, constraints, and trade-offs extensively
- Compare different approaches with detailed analysis
- Provide context for why specific choices or values were used

📊 STRUCTURE YOUR DETAILED ANSWER:
1. **Introduction/Overview**: Explain the context and scope (2-3 paragraphs)
2. **Main Analysis**: Deep dive into each aspect (multiple detailed paragraphs)
   - Break down complex topics into subsections
   - Provide step-by-step explanations
   - Include specific examples with full details
3. **Technical Details**: Explain HOW and WHY things work
   - Describe underlying mechanisms
   - Discuss implications and consequences
   - Compare alternatives when relevant
4. **Synthesis**: Connect different pieces of information
   - Identify patterns and relationships
   - Discuss broader implications
5. **Conclusion**: Summarize key insights comprehensively

💡 ANALYTICAL DEPTH:
- Don't just state facts - EXPLAIN THE REASONING behind them
- Analyze cause-and-effect relationships thoroughly
- Discuss trade-offs, advantages, and limitations in detail
- Compare and contrast different approaches extensively
- Identify implications for practice, implementation, or future work
- When citing numbers, explain their significance and context

📚 USE THE CONTEXT FULLY:
- Quote specific passages with proper citations [Source: filename]
- Reference multiple sources when they provide complementary information
- If sources disagree, explain the differences in detail
- Extract and explain ALL relevant technical details from the context

⚠️ QUALITY OVER BREVITY:
- NEVER give short, superficial answers
- Each paragraph should be substantial (5-7 sentences minimum)
- Elaborate on every important point
- Think "university lecture" not "quick summary"
- If the context is rich in detail, your answer should be too

Remember: The user wants DETAILED, IN-DEPTH, COMPREHENSIVE answers. More detail is ALWAYS better than less."""),
        ("human", """Question: {question}

Context:
{context}

⚡ IMPORTANT: Provide an EXTREMELY DETAILED, COMPREHENSIVE, and IN-DEPTH answer. Write at length with thorough explanations.

Follow this structure for maximum detail:
1. **Analyze the Question**: What exactly is being asked? What are the key components?
2. **Examine the Context**: What relevant information is available? What technical details are provided?
3. **Provide Comprehensive Answer**: Write a LONG, DETAILED response covering all aspects
4. **Technical Deep Dive**: Go into extensive technical specifics
5. **Synthesis & Conclusions**: Connect all information with thorough reasoning

Write your answer now (aim for 300-500+ words with extensive technical detail):""")
    ])

    llm, model_name = get_llm()
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
