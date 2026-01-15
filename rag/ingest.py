import os
from pathlib import Path
from dotenv import load_dotenv
from typing import List

from langchain_community.document_loaders import (
    DirectoryLoader, 
    TextLoader, 
    UnstructuredPowerPointLoader, 
    Docx2txtLoader, 
    UnstructuredODTLoader,
    PyMuPDFLoader,
    UnstructuredPDFLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


def load_documents_batch(docs_dir: Path, batch_size: int = 50) -> List[Document]:
    """Load documents in batches to avoid memory issues with large files."""
    all_docs = []
    
    loaders_config = [
        ("**/*.md", TextLoader, "Markdown"),
        ("**/*.txt", TextLoader, "Text"),
        ("**/*.pdf", None, "PDF"),  # Special handling
        ("**/*.pptx", UnstructuredPowerPointLoader, "PowerPoint"),
        ("**/*.docx", Docx2txtLoader, "Word"),
        ("**/*.odt", UnstructuredODTLoader, "ODT"),
    ]
    
    for glob_pattern, loader_cls, doc_type in loaders_config:
        print(f"Loading {doc_type} files...")
        
        if doc_type == "PDF":
            # Use PyMuPDF for better table/image support
            pdf_files = list(docs_dir.glob(glob_pattern))
            for pdf_file in pdf_files:
                try:
                    loader = PyMuPDFLoader(str(pdf_file))
                    docs = loader.load()
                    all_docs.extend(docs)
                    print(f"  ✓ Loaded: {pdf_file.name} ({len(docs)} pages)")
                except Exception as e:
                    print(f"  ✗ Error loading {pdf_file.name}: {e}")
                    # Fallback to UnstructuredPDFLoader for complex PDFs
                    try:
                        print(f"  → Trying UnstructuredPDFLoader for {pdf_file.name}...")
                        loader = UnstructuredPDFLoader(
                            str(pdf_file),
                            mode="elements",  # Extract tables and images
                            strategy="hi_res"  # High resolution for tables
                        )
                        docs = loader.load()
                        all_docs.extend(docs)
                        print(f"  ✓ Loaded with fallback: {pdf_file.name} ({len(docs)} elements)")
                    except Exception as e2:
                        print(f"  ✗ Failed both methods for {pdf_file.name}: {e2}")
        else:
            try:
                loader = DirectoryLoader(
                    str(docs_dir), 
                    glob=glob_pattern, 
                    loader_cls=loader_cls,
                    show_progress=True
                )
                docs = loader.load()
                all_docs.extend(docs)
                if docs:
                    print(f"  ✓ Loaded {len(docs)} {doc_type} document(s)")
            except Exception as e:
                print(f"  ✗ Error loading {doc_type} files: {e}")
    
    return all_docs


def main() -> None:
    load_dotenv()

    docs_dir = Path(os.getenv("DOCS_DIR", "docs")).resolve()
    chroma_dir = Path(os.getenv("CHROMA_DIR", "storage/chroma")).resolve()
    batch_size = int(os.getenv("BATCH_SIZE", "100"))
    chunk_size = int(os.getenv("CHUNK_SIZE", "1500"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "300"))

    if not docs_dir.exists():
        docs_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created empty docs directory at: {docs_dir}")
        print("Add .md, .txt, .pdf, .pptx, .docx, or .odt files and re-run this command.")
        return

    print(f"Loading documents from: {docs_dir}")
    print(f"Batch size: {batch_size} | Chunk size: {chunk_size} | Overlap: {chunk_overlap}\n")

    # Load documents with batch processing
    all_docs = load_documents_batch(docs_dir, batch_size)

    if not all_docs:
        print("\nNo documents found. Add .md, .txt, .pdf, .pptx, .docx, or .odt files to the docs folder.")
        return

    print(f"\nLoaded {len(all_docs)} total documents. Splitting into chunks...")
    
    # Use larger chunks for better context preservation
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]  # Better splitting for structured content
    )
    chunks = splitter.split_documents(all_docs)
    print(f"Created {len(chunks)} chunks.")

    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    chroma_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nBuilding Chroma index at: {chroma_dir}")

    # Process in batches to avoid memory issues
    print(f"Processing {len(chunks)} chunks in batches of {batch_size}...")
    
    if len(chunks) <= batch_size:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(chroma_dir),
        )
    else:
        # First batch creates the vectorstore
        vectorstore = Chroma.from_documents(
            documents=chunks[:batch_size],
            embedding=embeddings,
            persist_directory=str(chroma_dir),
        )
        print(f"  ✓ Processed batch 1/{(len(chunks) + batch_size - 1) // batch_size}")
        
        # Remaining batches are added
        for i in range(batch_size, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            vectorstore.add_documents(batch)
            batch_num = (i // batch_size) + 1
            total_batches = (len(chunks) + batch_size - 1) // batch_size
            print(f"  ✓ Processed batch {batch_num}/{total_batches}")

    print("\n✓ Ingestion complete. Vector store is ready for querying.")
    print(f"Total chunks indexed: {len(chunks)}")


if __name__ == "__main__":
    main()
