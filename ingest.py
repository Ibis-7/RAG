import os
import shutil
import hashlib
from pathlib import Path
from typing import List, Optional, Iterable

from dotenv import load_dotenv
from tqdm.auto import tqdm

# loaders / splitters / embeddings / chroma imports your stack used
from langchain_community.document_loaders import TextLoader, PyPDFLoader, WebBaseLoader, YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv()

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
SUPPORTED_ENCODINGS = ["utf-8", "cp1252", "latin-1"]

def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def _ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def _try_read_text_file(path: str) -> str:
    """
    Read using multiple encodings. Return text or raise.
    """
    last_exc = None
    for enc in SUPPORTED_ENCODINGS:
        try:
            return Path(path).read_text(encoding=enc)
        except Exception as e:
            last_exc = e
    # if all encodings fail, raise the last exception
    raise last_exc

def _safe_remove_dir(p: str):
    """
    Remove directory safely (used for reset). Keeps parent if requested.
    """
    if not os.path.exists(p):
        return
    
    p = os.path.abspath(p)
    if p in ("/", os.path.expanduser("~"), ""):
        raise RuntimeError("Refusing to remove unsafe path: " + p)
    shutil.rmtree(p)

def _chunks(iterable: List, n: int):
    """Yield successive n-sized chunks from list."""
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]

def ingest(
    persist_directory: str = "./vectordb",
    pdf_files: Optional[List[str]] = None,
    text_files: Optional[List[str]] = None,
    urls: Optional[List[str]] = None,
    youtube_links: Optional[List[str]] = None,
    reset: bool = False,
    batch_size: int = 256,
    collection_name: str = "my_collection",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
):
    """
    Ingest files/urls/youtube into Chroma with:
      - reset option
      - batching
      - deduplication by chunk-hash
      - chunk metadata (source, page, chunk index)
      - encoding fallbacks for text files
    """
    _ensure_dir(persist_directory)


    if reset:
        print("Reset requested — removing persist directory:", persist_directory)
        _safe_remove_dir(persist_directory)
        _ensure_dir(persist_directory)

    docs = []


    # PDF files
    if pdf_files:
        for p in pdf_files:
            try:
                loader = PyPDFLoader(p)
                loaded = loader.load()
                print(f"Loaded PDF {p}, docs: {len(loaded)}")
                docs.extend(loaded)
            except Exception as e:
                print(f"PDF loader failed for {p}: {e}. Skipping or try alternate loader.")

    # Text files 
    if text_files:
        for p in text_files:
            try:
                try:
                    loader = TextLoader(p, encoding="utf-8")
                    loaded = loader.load()
                except Exception:
                    raw = _try_read_text_file(p)
                    loaded = [
                        Document(
                                page_content=raw,
                                metadata={"source": str(Path(p).absolute())}
                        )
                    ]
                print(f"Loaded text {p}, docs: {len(loaded)}")
                docs.extend(loaded)
            except Exception as e:
                print(f"Failed to load text file {p}: {e}")

    # Web pages
    if urls:
        try:
            loader = WebBaseLoader(urls if isinstance(urls, list) else [urls])
            loaded = loader.load()
            print(f"Loaded {len(loaded)} web documents.")
            docs.extend(loaded)
        except Exception as e:
            print("Web loader error:", e)

    # YouTube transcripts
    if youtube_links:
        for link in youtube_links:
            try:
                loader = YoutubeLoader.from_youtube_url(link)
                loaded = loader.load()
                print(f"Loaded youtube {link}, docs: {len(loaded)}")
                docs.extend(loaded)
            except Exception as e:
                print(f"YouTube loader failed for {link}: {e}")

    if not docs:
        print("No source docs found. Exiting.")
        return {"ingested": 0}

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)



    # Store in Chroma
    embedding = HuggingFaceEmbeddings(model_name=model_name)
    vectordb = Chroma(collection_name=collection_name, embedding_function=embedding, persist_directory=persist_directory)
    

    # Dedup: load seen hashes if present; we store a simple text file of hashes
    seen_hash_path = Path(persist_directory) / f"{collection_name}_seen_hashes.txt"
    seen_hashes = set()
    if seen_hash_path.exists():
        with open(seen_hash_path, "r", encoding="utf-8") as f:
            for line in f:
                seen_hashes.add(line.strip())

    # Build items to add: avoid duplicates by hash(text)
    items_to_add = []
    for c in chunks:
        text = c.page_content if hasattr(c, "page_content") else c["page_content"]
        h = _hash_text(text)
        if h in seen_hashes:
            continue
        # prepare minimal doc object expected by Chroma wrapper
        items_to_add.append((text, c.metadata, h))

    print(f"New chunks to add (deduped): {len(items_to_add)}")

    # Batch add: compute embeddings in batches (embedding model will accept list input)
    total_added = 0
    for batch in tqdm(list(_chunks(items_to_add, batch_size)), desc="Batches"):
        docs_batch = [
            Document(page_content=txt, metadata=md)
            for txt, md, h in batch
        ]

        vectordb.add_documents(docs_batch)
        

        # Save hashes
        with open(seen_hash_path, "a", encoding="utf-8") as f:
            for (_, _, h) in batch:
                f.write(h + "\n")
                seen_hashes.add(h)

        total_added += len(batch)
    # vectordb.persist()




    print(f"Ingestion complete. Added {total_added} new chunks.")
    return {"ingested": total_added}