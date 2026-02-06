import os
import zipfile
import shutil
import tempfile
import hashlib
import json
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
    UnstructuredPDFLoader,
    UnstructuredMarkdownLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from concurrent.futures import ThreadPoolExecutor, as_completed


def extract_zip_files(docs_dir: Path) -> None:
    """Extract zip files recursively, handling nested zips."""
    zip_files = list(docs_dir.glob("**/*.zip"))
    
    if not zip_files:
        return
    
    print(f"\nFound {len(zip_files)} ZIP file(s) to extract...\n")
    
    supported_extensions = {'.pptx', '.pdf', '.docx', '.md', '.odt', '.txt'}
    processed_zips = set()
    
    def extract_recursive(zip_path: Path, level: int = 0):
        if str(zip_path) in processed_zips:
            return
        processed_zips.add(str(zip_path))
        
        indent = "  " * level
        try:
            extract_dir = zip_path.parent / zip_path.stem
            
            print(f"{indent}Extracting: {zip_path.name}")
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                
                supported_files = [f for f in file_list 
                                 if any(f.lower().endswith(ext) for ext in supported_extensions)]
                nested_zips = [f for f in file_list if f.lower().endswith('.zip')]
                
                if supported_files or nested_zips:
                    zip_ref.extractall(extract_dir)
                    
                    if supported_files:
                        print(f"{indent}  → Found {len(supported_files)} supported document(s)")
                    
                    if nested_zips:
                        print(f"{indent}  → Found {len(nested_zips)} nested ZIP file(s)")
                        for nested_zip in nested_zips:
                            nested_zip_path = extract_dir / nested_zip
                            if nested_zip_path.exists():
                                extract_recursive(nested_zip_path, level + 1)
                    
                    print(f"{indent}  ✓ Extracted to: {extract_dir.name}/")
                else:
                    print(f"{indent}  ⚠ No supported documents found (skipping)")
                    
        except zipfile.BadZipFile:
            print(f"{indent}  ✗ Error: {zip_path.name} is not a valid ZIP file")
        except Exception as e:
            print(f"{indent}  ✗ Error extracting {zip_path.name}: {e}")
    
    for zip_path in zip_files:
        extract_recursive(zip_path)
    
    print()


def load_documents_batch(docs_dir: Path, batch_size: int = 50) -> List[Document]:
    all_docs = []
    
    loaders_config = [
        ("**/*.md", UnstructuredMarkdownLoader, "Markdown"),
        ("**/*.txt", TextLoader, "Text"),
        ("**/*.pdf", None, "PDF"),
        ("**/*.pptx", UnstructuredPowerPointLoader, "PowerPoint"),
        ("**/*.ppt", UnstructuredPowerPointLoader, "PowerPoint (legacy)"),
        ("**/*.docx", Docx2txtLoader, "Word"),
        ("**/*.doc", Docx2txtLoader, "Word (legacy)"),
        ("**/*.odt", UnstructuredODTLoader, "ODT"),
    ]
    
    def load_pdf(pdf_file):
        """Load single PDF with fallback."""
        try:
            loader = PyMuPDFLoader(str(pdf_file))
            docs = loader.load()
            return pdf_file.name, docs, None
        except Exception as e:
            try:
                loader = UnstructuredPDFLoader(str(pdf_file), mode="elements", strategy="hi_res")
                docs = loader.load()
                return pdf_file.name, docs, "fallback"
            except Exception as e2:
                return pdf_file.name, [], str(e2)
    
    for glob_pattern, loader_cls, doc_type in loaders_config:
        print(f"Loading {doc_type} files...")
        
        if doc_type == "PDF":
            pdf_files = list(docs_dir.glob(glob_pattern))
            if pdf_files:
                # Parallel PDF loading with more workers for large document sets
                max_workers = min(8, len(pdf_files))  # Scale workers based on file count
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {executor.submit(load_pdf, pdf_file): pdf_file for pdf_file in pdf_files}
                    for future in as_completed(futures):
                        filename, docs, error = future.result()
                        if error:
                            if error != "fallback":
                                print(f"  ✗ Error loading {filename}: {error}")
                        else:
                            all_docs.extend(docs)
                            fallback_msg = " (fallback)" if error == "fallback" else ""
                            print(f"  ✓ Loaded: {filename} ({len(docs)} pages){fallback_msg}")
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
    batch_size = int(os.getenv("BATCH_SIZE", "200"))
    chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
    cache_file = chroma_dir.parent / ".ingest_cache.json"

    if not docs_dir.exists():
        docs_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created empty docs directory at: {docs_dir}")
        print("Add .md, .txt, .pdf, .pptx, .docx, .odt files or .zip archives and re-run this command.")
        return

    # Check if we can skip ingestion based on cache
    current_hash = get_directory_hash(docs_dir)
    if cache_file.exists() and chroma_dir.exists():
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
                if cache_data.get('directory_hash') == current_hash:
                    print(f"✓ No changes detected. Skipping ingestion.")
                    print(f"Vector store is ready at: {chroma_dir}")
                    print(f"Total chunks: {cache_data.get('total_chunks', 'unknown')}")
                    return
        except Exception:
            pass

    print(f"Loading documents from: {docs_dir}")
    print(f"Batch size: {batch_size} | Chunk size: {chunk_size} | Overlap: {chunk_overlap}\n")

    extract_zip_files(docs_dir)

    all_docs = load_documents_batch(docs_dir, batch_size)

    if not all_docs:
        print("\nNo documents found. Add .md, .txt, .pdf, .pptx, .docx, .odt files or .zip archives to the docs folder.")
        return

    print(f"\nLoaded {len(all_docs)} total documents. Splitting into chunks...")
    
    import time
    start_split = time.time()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(all_docs)
    print(f"Created {len(chunks)} chunks in {time.time() - start_split:.2f}s")

    # Use faster embedding model if specified
    use_faster = os.getenv("USE_FASTER_EMBEDDINGS", "false").lower() == "true"
    embed_model = "BAAI/bge-small-en-v1.5" if not use_faster else "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = FastEmbedEmbeddings(model_name=embed_model, max_length=512)

    chroma_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nBuilding Chroma index at: {chroma_dir}")

    print(f"Processing {len(chunks)} chunks in batches of {batch_size}...")
    
    import time
    start_index = time.time()
    
    if len(chunks) <= batch_size:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(chroma_dir),
        )
    else:
        vectorstore = Chroma.from_documents(
            documents=chunks[:batch_size],
            embedding=embeddings,
            persist_directory=str(chroma_dir),
        )
        print(f"  ✓ Processed batch 1/{(len(chunks) + batch_size - 1) // batch_size}")
        
        for i in range(batch_size, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            vectorstore.add_documents(batch)
            batch_num = (i // batch_size) + 1
            total_batches = (len(chunks) + batch_size - 1) // batch_size
            print(f"  ✓ Processed batch {batch_num}/{total_batches}")

    print(f"\n✓ Ingestion complete in {time.time() - start_index:.2f}s. Vector store is ready for querying.")
    print(f"Total chunks indexed: {len(chunks)}")
    
    # Save cache
    try:
        with open(cache_file, 'w') as f:
            json.dump({
                'directory_hash': current_hash,
                'total_chunks': len(chunks),
                'timestamp': str(Path.cwd())
            }, f)
    except Exception:
        pass


def get_directory_hash(directory: Path) -> str:
    """Generate hash of all files in directory for caching."""
    hasher = hashlib.md5()
    try:
        for file_path in sorted(directory.rglob('*.zip')):
            hasher.update(str(file_path).encode())
            hasher.update(str(file_path.stat().st_mtime).encode())
    except Exception:
        pass
    return hasher.hexdigest()


if __name__ == "__main__":
    main()
