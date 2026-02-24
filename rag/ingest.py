import os
import zipfile
import shutil
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from typing import List
import re
import xml.etree.ElementTree as ET

from langchain_community.document_loaders import (  # type: ignore
    DirectoryLoader, 
    TextLoader, 
    UnstructuredPowerPointLoader, 
    Docx2txtLoader, 
    UnstructuredODTLoader,
    PyMuPDFLoader,
    UnstructuredPDFLoader,
    UnstructuredMarkdownLoader,
    UnstructuredWordDocumentLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore
from langchain_community.embeddings import FastEmbedEmbeddings  # type: ignore
from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
from langchain_chroma import Chroma  # type: ignore
from langchain_core.documents import Document  # type: ignore
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from datetime import datetime

try:
    from docx import Document as DocxDocument  # type: ignore
    PYTHON_DOCX_AVAILABLE = True
except ImportError:
    PYTHON_DOCX_AVAILABLE = False

try:
    from docx import Document as DocxDocument  # type: ignore
    PYTHON_DOCX_AVAILABLE = True
except ImportError:
    PYTHON_DOCX_AVAILABLE = False


def load_docx_with_python_docx(file_path: str) -> List[Document]:
    if not PYTHON_DOCX_AVAILABLE:
        raise ImportError("python-docx not available")
    doc = DocxDocument(file_path)
    text = "\n\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    if not text.strip():
        raise ValueError("No text extracted")
    return [Document(page_content=text, metadata={"source": Path(file_path).name})]


def load_docx_raw_xml(file_path: str) -> List[Document]:
    with zipfile.ZipFile(file_path, 'r') as docx_zip:
        try:
            xml_content = docx_zip.read('word/document.xml')
            root = ET.fromstring(xml_content)
            
            namespaces = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            paragraphs = root.findall('.//w:t', namespaces)
            text = "\n".join([p.text for p in paragraphs if p.text])
            
            if not text.strip():
                raise ValueError("No text extracted from XML")
            
            return [Document(page_content=text, metadata={"source": Path(file_path).name})]
        except Exception as e:
            raise ValueError(f"Failed to extract from XML: {e}")


def extract_zip_files(docs_dir: Path) -> None:
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
            
            if extract_dir.exists():
                print(f"{indent}Skipping (already extracted): {zip_path.name}")
                return
            
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
                    
                    print(f"{indent}  -> Extracted to: {extract_dir.name}/")
                else:
                    print(f"{indent}  ! No supported documents found (skipping)")
                    
        except zipfile.BadZipFile:
            print(f"{indent}  ✗ Error: {zip_path.name} is not a valid ZIP file")
        except Exception as e:
            print(f"{indent}  ✗ Error extracting {zip_path.name}: {e}")
    
    for zip_path in zip_files:
        extract_recursive(zip_path)
    
    print()


def load_documents_batch(docs_dir: Path, batch_size: int = 50) -> tuple[List[Document], dict]:
    all_docs = []
    stats = {
        'successful': [],
        'failed': [],
        'total_by_type': {}
    }

    log_file = Path("ingestion_errors.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting document ingestion from: {docs_dir}")
    
    def load_single_file(file_path: Path, loader_configs: list) -> tuple[str, list, str]:
        for config_name, loader_func in loader_configs:
            try:
                docs = loader_func(str(file_path))
                if isinstance(docs, list):
                    return file_path.name, docs, config_name
                else:
                    loaded_docs = docs.load()
                    return file_path.name, loaded_docs, config_name
            except Exception as e:
                logger.debug(f"  {config_name} failed for {file_path.name}: {str(e)}")
                continue
        
        return file_path.name, [], "FAILED"
    
    loaders_config = [
        ("**/*.md", [
            ("UnstructuredMarkdown", lambda p: UnstructuredMarkdownLoader(p)),
            ("TextLoader", lambda p: TextLoader(p))
        ], "Markdown"),
        
        ("**/*.txt", [
            ("TextLoader-utf8", lambda p: TextLoader(p, encoding='utf-8')),
            ("TextLoader-latin1", lambda p: TextLoader(p, encoding='latin-1')),
            ("TextLoader-auto", lambda p: TextLoader(p, autodetect_encoding=True))
        ], "Text"),
        
        ("**/*.pdf", [
            ("PyMuPDF", lambda p: PyMuPDFLoader(p)),
            ("Unstructured-Fast", lambda p: UnstructuredPDFLoader(p, mode="elements")),
            ("Unstructured-HiRes", lambda p: UnstructuredPDFLoader(p, mode="elements", strategy="hi_res"))
        ], "PDF"),
        
        ("**/*.pptx", [
            ("UnstructuredPowerPoint", lambda p: UnstructuredPowerPointLoader(p))
        ], "PowerPoint"),
        
        ("**/*.ppt", [
            ("UnstructuredPowerPoint", lambda p: UnstructuredPowerPointLoader(p))
        ], "PowerPoint (legacy)"),
        
        ("**/*.docx", [
            ("Docx2txt", lambda p: Docx2txtLoader(p)),
            ("PythonDocx", lambda p: load_docx_with_python_docx(p)),
            ("UnstructuredWord-Fast", lambda p: UnstructuredWordDocumentLoader(p, mode="single")),
            ("UnstructuredWord-Elements", lambda p: UnstructuredWordDocumentLoader(p, mode="elements")),
            ("RawXML", lambda p: load_docx_raw_xml(p))
        ], "Word"),
        
        ("**/*.doc", [
            ("UnstructuredWord", lambda p: UnstructuredWordDocumentLoader(p)),
            ("Docx2txt", lambda p: Docx2txtLoader(p))
        ], "Word (legacy)"),
        
        ("**/*.odt", [
            ("UnstructuredODT", lambda p: UnstructuredODTLoader(p))
        ], "ODT"),
    ]
    
    for glob_pattern, loader_strategies, doc_type in loaders_config:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {doc_type} files...")
        logger.info(f"{'='*60}")
        
        all_files = list(docs_dir.glob(glob_pattern))
        valid_files = [f for f in all_files if not f.name.startswith(('~$', '._'))]
        
        if not valid_files:
            logger.info(f"  No {doc_type} files found")
            continue
            
        if len(valid_files) < len(all_files):
            skipped = len(all_files) - len(valid_files)
            logger.info(f"  ! Skipping {skipped} temp file(s)")
        
        logger.info(f"  Found {len(valid_files)} {doc_type} file(s)")
        
        successful_count = 0
        failed_files = []
        
        for file_path in valid_files:
            filename, docs, method = load_single_file(file_path, loader_strategies)
            
            if method != "FAILED" and docs:
                all_docs.extend(docs)
                successful_count += 1
                stats['successful'].append({
                    'file': str(file_path),
                    'type': doc_type,
                    'chunks': len(docs),
                    'method': method
                })
                logger.info(f"  + {filename} ({len(docs)} chunks) [{method}]")
            else:
                failed_files.append(filename)
                stats['failed'].append({
                    'file': str(file_path),
                    'type': doc_type,
                    'error': 'All loading strategies failed'
                })
                logger.error(f"  ✗ {filename} - ALL STRATEGIES FAILED")
        
        stats['total_by_type'][doc_type] = {
            'total': len(valid_files),
            'successful': successful_count,
            'failed': len(failed_files)
        }
        
        logger.info(f"\n  Summary: {successful_count}/{len(valid_files)} {doc_type} files loaded successfully")
        if failed_files:
            logger.warning(f"  Failed files: {', '.join(failed_files)}")
    
    return all_docs, stats


def main() -> None:
    load_dotenv()

    docs_dir = Path(os.getenv("DOCS_DIR")).resolve()
    chroma_dir = Path(os.getenv("CHROMA_DIR")).resolve()
    batch_size = int(os.getenv("BATCH_SIZE"))
    chunk_size = int(os.getenv("CHUNK_SIZE"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP"))

    if not docs_dir.exists():
        docs_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created empty docs directory at: {docs_dir}")
        print("Add .md, .txt, .pdf, .pptx, .docx, .odt files or .zip archives and re-run this command.")
        return

    supported_extensions = ['.md', '.txt', '.pdf', '.pptx', '.ppt', '.docx', '.doc', '.odt']
    total_doc_count = sum(1 for ext in supported_extensions for _ in docs_dir.glob(f"**/*{ext}"))
    
    print(f"Loading documents from: {docs_dir}")
    print(f"Batch size: {batch_size} | Chunk size: {chunk_size} | Overlap: {chunk_overlap}")
    print(f"\n📁 Found {total_doc_count} total documents to process\n")

    extract_zip_files(docs_dir)

    all_docs, stats = load_documents_batch(docs_dir, batch_size)

    print("\n" + "="*70)
    print("INGESTION SUMMARY")
    print("="*70)
    
    total_files = len(stats['successful']) + len(stats['failed'])
    print(f"\nTotal files processed: {total_files}")
    print(f"✓ Successfully loaded: {len(stats['successful'])} ({len(stats['successful'])/total_files*100:.1f}%)")
    print(f"✗ Failed to load: {len(stats['failed'])} ({len(stats['failed'])/total_files*100:.1f}%)")
    
    print("\nBreakdown by file type:")
    for doc_type, counts in stats['total_by_type'].items():
        success_rate = counts['successful']/counts['total']*100 if counts['total'] > 0 else 0
        print(f"  {doc_type:20} {counts['successful']:3}/{counts['total']:3} ({success_rate:5.1f}%)")
    
    if stats['failed']:
        print(f"\n⚠ Failed files ({len(stats['failed'])}):")
        for failed in stats['failed'][:10]:
            print(f"    • {Path(failed['file']).name} ({failed['type']})")
        if len(stats['failed']) > 10:
            print(f"    ... and {len(stats['failed'])-10} more (see ingestion_errors.log)")
        print("\n→ Check 'ingestion_errors.log' for detailed error information")
    
    if not all_docs:
        print("\n❌ No documents were successfully loaded!")
        print("Check 'ingestion_errors.log' for details.")
        return

    print(f"\n✓ Loaded {len(all_docs)} total document chunks")
    print(f"  Splitting into optimized chunks...")
    
    import time
    start_split = time.time()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(all_docs)
    print(f"  Created {len(chunks)} chunks in {time.time() - start_split:.2f}s")

    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "fastembed").lower()
    embedding_model = os.getenv("EMBEDDING_MODEL")
    
    print(f"\nSetting up embeddings...")
    print(f"  Provider: {embedding_provider}")
    print(f"  Model: {embedding_model}")
    
    if embedding_provider == "huggingface":
        print("  Using HuggingFace embeddings (supports any model)")
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},  
            encode_kwargs={'normalize_embeddings': True}
        )
    else:  
        print("  Using FastEmbed embeddings (optimized, limited models)")
        embeddings = FastEmbedEmbeddings(
            model_name=embedding_model, 
            max_length=512,
            threads=4
        )

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
        print(f"  ✓ Batch 1/{(len(chunks) + batch_size - 1) // batch_size} ({batch_size} chunks)")
        
        for i in range(batch_size, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_start = time.time()
            vectorstore.add_documents(batch)
            batch_num = (i // batch_size) + 1
            total_batches = (len(chunks) + batch_size - 1) // batch_size
            elapsed = time.time() - batch_start
            chunks_per_sec = len(batch) / elapsed if elapsed > 0 else 0
            print(f"  ✓ Batch {batch_num}/{total_batches} ({len(batch)} chunks in {elapsed:.1f}s = {chunks_per_sec:.0f} chunks/s)")

    print(f"\n✓ Ingestion complete in {time.time() - start_index:.2f}s")
    
    print(f"\n{'='*70}")
    print("FINAL STATISTICS")
    print(f"{'='*70}")
    print(f"  📊 Total documents found:     {total_files}")
    print(f"  ✅ Successfully ingested:     {len(stats['successful'])} ({len(stats['successful'])/total_files*100:.1f}%)")
    print(f"  ❌ Failed to ingest:          {len(stats['failed'])} ({len(stats['failed'])/total_files*100:.1f}%)")
    print(f"  📦 Total chunks created:      {len(chunks):,}")
    print(f"  💾 Database location:         {chroma_dir}")
    print(f"{'='*70}")
    
    if stats['failed']:
        print(f"\n⚠️  {len(stats['failed'])} files could not be loaded")
        print(f"   See 'ingestion_errors.log' for details")
    else:
        print(f"\n🎉 All files successfully ingested!")
    
    print(f"\n✅ Vector store is ready for querying")


if __name__ == "__main__":
    main()
