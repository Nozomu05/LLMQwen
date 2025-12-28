import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, TextLoader, UnstructuredPowerPointLoader, Docx2txtLoader, UnstructuredODTLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma


def main() -> None:
    load_dotenv()

    docs_dir = Path(os.getenv("DOCS_DIR", "docs")).resolve()
    chroma_dir = Path(os.getenv("CHROMA_DIR", "storage/chroma")).resolve()

    if not docs_dir.exists():
        docs_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created empty docs directory at: {docs_dir}")
        print("Add .md, .txt, .pptx, .docx, or .odt files and re-run this command.")
        return

    print(f"Loading documents from: {docs_dir}")

    # Load multiple file types
    all_docs = []
    
    # Markdown files
    md_loader = DirectoryLoader(str(docs_dir), glob="**/*.md", loader_cls=TextLoader, show_progress=True)
    all_docs.extend(md_loader.load())
    
    # Text files
    txt_loader = DirectoryLoader(str(docs_dir), glob="**/*.txt", loader_cls=TextLoader, show_progress=True)
    all_docs.extend(txt_loader.load())
    
    # PowerPoint files (.pptx)
    pptx_loader = DirectoryLoader(str(docs_dir), glob="**/*.pptx", loader_cls=UnstructuredPowerPointLoader, show_progress=True)
    all_docs.extend(pptx_loader.load())
    
    # Word files (.docx)
    docx_loader = DirectoryLoader(str(docs_dir), glob="**/*.docx", loader_cls=Docx2txtLoader, show_progress=True)
    all_docs.extend(docx_loader.load())
    
    # ODT files (.odt)
    odt_loader = DirectoryLoader(str(docs_dir), glob="**/*.odt", loader_cls=UnstructuredODTLoader, show_progress=True)
    all_docs.extend(odt_loader.load())

    if not all_docs:
        print("No documents found. Add .md, .txt, .pptx, .docx, or .odt files to the docs folder.")
        return
    
    docs = all_docs

    print(f"Loaded {len(docs)} documents. Splitting...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks.")

    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    chroma_dir.mkdir(parents=True, exist_ok=True)
    print(f"Building Chroma index at: {chroma_dir}")

    # Create or overwrite persistent Chroma store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(chroma_dir),
    )  # Persistence handled automatically in langchain-chroma

    print("Ingestion complete. Vector store is ready.")


if __name__ == "__main__":
    main()
