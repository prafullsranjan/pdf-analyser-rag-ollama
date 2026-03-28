"""
RAG Pipeline for People Analytics
- PDF extraction with metadata tagging
- Chunking with overlap
- Ollama embeddings (nomic-embed-text)
- ChromaDB persistent vector store
- Gemma 3:4b for answer generation
"""

import os
import contextlib
import json
import hashlib
import logging
import datetime
import re
from pathlib import Path
from typing import Optional
import chromadb
from chromadb.config import Settings
import ollama
from docling.document_converter import DocumentConverter


# ── Minimal recursive text splitter (no langchain dependency) ─────────────────
class RecursiveCharacterTextSplitter:
    """
    Pure-Python recursive splitter that mirrors LangChain's behaviour.
    Tries each separator in order; falls back to the next if chunks are still too large.
    """

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        separators: Optional[list[str]] = None,
    ):
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators    = separators or ["\n\n", "\n", ". ", " ", ""]

    def _split(self, text: str, separators: list[str]) -> list[str]:
        sep = separators[0]
        next_seps = separators[1:]

        parts = text.split(sep) if sep else list(text)

        chunks: list[str] = []
        current = ""

        for part in parts:
            candidate = (current + sep + part).lstrip(sep) if current else part
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                # part itself too large → recurse with next separator
                if len(part) > self.chunk_size and next_seps:
                    chunks.extend(self._split(part, next_seps))
                    current = ""
                else:
                    current = part

        if current:
            chunks.append(current)

        return chunks

    def _merge_with_overlap(self, chunks: list[str]) -> list[str]:
        """Re-merge tiny chunks and add overlap between adjacent chunks."""
        merged: list[str] = []
        buf = ""

        for chunk in chunks:
            if not buf:
                buf = chunk
            elif len(buf) + 1 + len(chunk) <= self.chunk_size:
                buf = buf + " " + chunk
            else:
                merged.append(buf)
                # carry overlap from tail of previous chunk
                overlap_start = max(0, len(buf) - self.chunk_overlap)
                buf = buf[overlap_start:] + " " + chunk

        if buf:
            merged.append(buf)

        return merged

    def split_text(self, text: str) -> list[str]:
        raw    = self._split(text, self.separators)
        result = self._merge_with_overlap(raw)
        return [c.strip() for c in result if c.strip()]

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_DIR = Path("./logs")
LOG_DIR.mkdir(exist_ok=True)

log_filename = LOG_DIR / f"bls_analyser_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

log_format    = "%(asctime)s [%(levelname)-8s] [%(funcName)s:%(lineno)d] %(message)s"
log_formatter = logging.Formatter(log_format)

file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(log_formatter)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(log_formatter)

logger = logging.getLogger("rag_pipeline")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

logger.info("╔══════════════════════════════════════════════════════════════╗")
logger.info("║        BLS ANALYSER — RAG Pipeline Logger Started            ║")
logger.info(f"║        Log File: {str(log_filename):<44}║")
logger.info("╚══════════════════════════════════════════════════════════════╝")

# ── Constants ─────────────────────────────────────────────────────────────────
CHROMA_DIR        = "./chroma_store"
INDEX_MANIFEST    = "./index_manifest.json"
COLLECTION_NAME   = "people_analytics"
EMBED_MODEL       = "nomic-embed-text"
LLM_MODEL         = "gemma3:4b"
CHUNK_SIZE        = 800
CHUNK_OVERLAP     = 150
MAX_CHUNKS_RETURN = 6   # top-k for retrieval


# ── Helpers ───────────────────────────────────────────────────────────────────

def _file_hash(path: str) -> str:
    """SHA-256 of file contents for change detection."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_manifest() -> dict:
    if os.path.exists(INDEX_MANIFEST):
        with open(INDEX_MANIFEST, "r") as f:
            return json.load(f)
    return {}


def _save_manifest(manifest: dict):
    with open(INDEX_MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)


# ── ChromaDB client (singleton) ───────────────────────────────────────────────

_chroma_client: Optional[chromadb.PersistentClient] = None

def get_chroma_client() -> chromadb.PersistentClient:
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(
            path=CHROMA_DIR,
            settings=Settings(anonymized_telemetry=False)
        )
    return _chroma_client


def get_or_create_collection():
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch embed texts using Ollama nomic-embed-text."""
    embeddings = []
    for text in texts:
        response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
        embeddings.append(response["embedding"])
    return embeddings


# ── PDF Extraction ────────────────────────────────────────────────────────────

def extract_pdf_with_layout(pdf_path: str) -> list[dict]:
    """
    Uses Docling for layout-aware extraction. 
    Preserves table structures and column flows.
    """
    logger.debug(f"Starting Docling extraction for file: {pdf_path}")
    converter = DocumentConverter()
    
    # Suppress C-level pdfium stderr noise (cannot be caught by Python warnings)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    old_stderr_fd = os.dup(2)
    try:
        os.dup2(devnull_fd, 2)
        result = converter.convert(pdf_path)
    finally:
        os.dup2(old_stderr_fd, 2)   # always restore stderr
        os.close(old_stderr_fd)
        os.close(devnull_fd)
    
    doc = result.document
    pdf_name = Path(pdf_path).stem
    year_match = re.search(r'\d{4}', pdf_name)
    year = int(year_match.group()) if year_match else "Unknown"

    pages = []
    # Iterate through elements (paragraphs, tables, etc.)
    for item, level in doc.iterate_items():
        if hasattr(item, "text") and getattr(item, "text", ""):
            # We attach the year to the text itself to 'anchor' the embedding
            anchored_text = f"Year: {year} | Content: {item.text}"
            
            pages.append({
                "page": getattr(item, "page_no", 1) or 1,
                "text": anchored_text,
                "pdf_name": pdf_name,
                "pdf_path": pdf_path,
                "year": year
            })
    logger.debug(f"Docling conversion complete. {pdf_name} yielded {len(pages)} elements.")
    return pages


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_pages_enhanced(pages: list[dict]) -> list[dict]:
    """Split page texts into overlapping chunks, preserving metadata and temporal anchor."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for page in pages:
        parts = page["text"].split(" | Content: ", 1)
        if len(parts) == 2:
            prefix = parts[0] + " | Content: "
            raw_content = parts[1]
        else:
            prefix = ""
            raw_content = page["text"]
            
        splits = splitter.split_text(raw_content)
        for i, split in enumerate(splits):
            chunks.append({
                "text":      f"{prefix}{split}",
                "pdf_name":  page["pdf_name"],
                "pdf_path":  page["pdf_path"],
                "page":      page["page"],
                "chunk_idx": i,
                "year":      page.get("year", "Unknown")
            })
    return chunks


# ── Indexing ──────────────────────────────────────────────────────────────────

def index_folder(folder_path: str, force: bool = False, progress_callback=None) -> dict:
    """
    Scan folder for PDFs. Only index new/changed files (incremental).
    If force=True, re-index everything.
    Returns a status dict.
    """
    folder = Path(folder_path)
    if not folder.exists():
        return {"status": "error", "message": f"Folder not found: {folder_path}"}

    pdf_files = list(folder.rglob("*.pdf"))
    if not pdf_files:
        return {"status": "warning", "message": "No PDF files found in folder."}

    manifest    = _load_manifest()
    collection  = get_or_create_collection()

    indexed: int = 0
    skipped: int = 0
    removed: int = 0
    errors: list[str] = []
    
    total_found = len(pdf_files)
    total_extracted = 0
    total_chunks = 0
    total_embedded = 0

    def _report():
        if progress_callback:
            progress_callback({
                "found": total_found,
                "extracted": total_extracted,
                "chunks": total_chunks,
                "embedded": total_embedded,
            })
            
    _report()

    # ── Remove stale entries (deleted PDFs) ───────────────────────────────────
    current_paths = {str(p) for p in pdf_files}
    stale_paths   = [p for p in manifest if p not in current_paths]
    for stale in stale_paths:
        try:
            # Delete by pdf_path metadata filter
            ids_to_delete = collection.get(
                where={"pdf_path": stale}
            )["ids"]
            if ids_to_delete:
                collection.delete(ids=ids_to_delete)
            del manifest[stale]
            removed += 1
            logger.info(f"Removed stale index for: {stale}")
        except Exception as e:
            errors.append(f"Could not remove {stale}: {e}")

    # ── Index new / changed PDFs ──────────────────────────────────────────────
    for pdf_path in pdf_files:
        path_str = str(pdf_path)
        file_hash = _file_hash(path_str)

        if not force and manifest.get(path_str) == file_hash:
            skipped += 1
            continue

        # Remove old chunks for this file if re-indexing
        try:
            old_ids = collection.get(where={"pdf_path": path_str})["ids"]
            if old_ids:
                collection.delete(ids=old_ids)
        except Exception:
            pass

        # Extract → chunk → embed → store
        try:
            logger.debug(f"Initiating extraction for {pdf_path.name}")
            pages  = extract_pdf_with_layout(path_str)
            total_extracted += 1
            _report()
            
            chunks = chunk_pages_enhanced(pages)
            total_chunks += len(chunks)
            _report()
            logger.debug(f"Chunking finished. {pdf_path.name} split into {len(chunks)} temporal chunks.")

            if not chunks:
                logger.warning(f"No extractable text found in {pdf_path.name}")
                errors.append(f"No extractable text in {pdf_path.name}")
                continue

            logger.debug(f"Generating vectors for {len(chunks)} chunks using nomic-embed-text...")
            texts     = [c["text"]     for c in chunks]
            embeddings = embed_texts(texts)

            logger.debug(f"Writing {len(chunks)} vectors to ChromaDB for {pdf_path.name}...")

            ids       = [f"{file_hash}_{i}" for i in range(len(chunks))]
            metadatas = []
            for c in chunks:
                match = re.search(r'\d{4}', c["pdf_name"])
                metadatas.append({
                    "pdf_name":  c["pdf_name"],
                    "pdf_path":  c["pdf_path"],
                    "page":      c["page"],
                    "chunk_idx": c["chunk_idx"],
                    "year":      int(match.group()) if match else None
                })

            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )
            
            total_embedded += len(chunks)
            _report()

            manifest[path_str] = file_hash
            indexed += 1
            logger.info(f"Indexed {pdf_path.name} → {len(chunks)} chunks")

        except Exception as e:
            errors.append(f"{pdf_path.name}: {str(e)}")
            logger.error(f"Failed to index {pdf_path.name}: {e}")

    _save_manifest(manifest)

    return {
        "status":  "success",
        "indexed":  indexed,
        "skipped":  skipped,
        "removed":  removed,
        "total_pdfs": len(pdf_files),
        "errors":   errors,
        "collection_count": collection.count(),
    }

def clear_index() -> dict:
    """Clear all indexed data from ChromaDB and delete the manifest."""
    try:
        client = get_chroma_client()
        client.delete_collection(name=COLLECTION_NAME)
    except Exception as e:
        logger.warning(f"Could not delete collection: {e}")

    try:
        manifest_path = Path(INDEX_MANIFEST)
        if manifest_path.exists():
            manifest_path.unlink()
    except Exception as e:
        logger.warning(f"Could not delete manifest file: {e}")

    return {
        "status": "success",
        "message": "Index cleared successfully."
    }

# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve_chunks(query: str, n_results: int = MAX_CHUNKS_RETURN, years: Optional[list[int]] = None) -> list[dict]:
    """Embed query and return top-k relevant chunks with metadata."""
    collection = get_or_create_collection()
    if collection.count() == 0:
        return []

    where_clause = None
    if years:
        if len(years) == 1:
            where_clause = {"year": years[0]}
        else:
            where_clause = {"year": {"$in": years}}

    q_embed = embed_texts([query])[0]
    results = collection.query(
        query_embeddings=[q_embed],
        n_results=min(n_results, collection.count()),
        include=["documents", "metadatas", "distances"],
        where=where_clause
    )

    chunks = []
    if not results["documents"] or not results["documents"][0]:
        return chunks

    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text":      doc,
            "pdf_name":  meta["pdf_name"],
            "pdf_path":  meta["pdf_path"],
            "year":      meta.get("year", "N/A"),
            "page":      meta["page"],
            "score":     round(1 - dist, 4),   # cosine similarity
        })
    return chunks


# ── Answer Generation ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert labor historian and People Analytics assistant. 
Your role is to analyze Bureau of Labor Statistics Occupational Outlook Handbooks across different decades.
Focus on extracting insights regarding labor market evolution, skill gaps, diversity/inclusion terminology, wage trends, and technological transformations (e.g., automation).

Rules:
1. Only use information from the provided context chunks.
2. If the answer is not in the context, say so clearly — do not hallucinate.
3. Always cite the source PDF, the Year, and page number for every claim (e.g. OOH_1980.pdf, Year 1980, p. 45).
4. Outline your answer clearly with sections highlighting longitudinal patterns or decade-by-decade comparisons where appropriate.
5. Be concise, professional, and data-driven.
"""

def build_context_block(chunks: list[dict]) -> str:
    lines = []
    for i, c in enumerate(chunks, 1):
        lines.append(
            f"[Source {i}: {c['pdf_name']}, Year {c.get('year', 'N/A')}, Page {c['page']} | Relevance: {c['score']}]\n{c['text']}"
        )
    return "\n\n---\n\n".join(lines)


def generate_answer(
    query: str,
    output_format: str = "text",
    chunks: Optional[list[dict]] = None,
) -> dict:
    """
    Generate an answer using retrieved chunks.
    output_format: 'text' | 'word_cloud' | 'data_chart' | 'data_visualization'
    Returns dict with answer, sources, format_data.
    """
    if chunks is None:
        chunks = retrieve_chunks(query)

    if not chunks:
        return {
            "answer":      "⚠️ No indexed documents found. Please index a folder first.",
            "sources":     [],
            "format_data": None,
        }

    context = build_context_block(chunks)

    # ── Format-specific instructions ──────────────────────────────────────────
    format_instructions = {
        "text": (
            "Provide a detailed, well-structured text answer with clear headings "
            "and bullet points where needed. Cite sources inline as [PDF Name, p.X]."
        ),
        "word_cloud": (
            "Provide a structured text answer. ADDITIONALLY, at the very end, "
            "output a JSON block (wrapped in ```json ... ```) with key: 'word_frequencies' "
            "as a dict of {word: frequency} for the 30 most important domain-specific terms "
            "in the answer. Example: {\"engagement\": 15, \"turnover\": 12, ...}"
        ),
        "data_chart": (
            "Provide a structured text answer. ADDITIONALLY, at the very end, "
            "output a JSON block (wrapped in ```json ... ```) with key: 'chart_data' "
            "containing: 'title' (str), 'chart_type' (bar|line|pie), "
            "'labels' (list of str), 'values' (list of numbers), 'unit' (str). "
            "Extract or infer numeric data from the context for the chart."
        ),
        "data_visualization": (
            "Provide a structured text answer. ADDITIONALLY, at the very end, "
            "output a JSON block (wrapped in ```json ... ```) with key: 'viz_data' "
            "containing: 'title' (str), 'metrics' (list of {label, value, unit, trend}). "
            "trend must be one of: up | down | neutral. "
            "Extract key HR metrics/KPIs from the context."
        ),
    }

    user_message = f"""Question: {query}

Output Format Instruction: {format_instructions.get(output_format, format_instructions['text'])}

Context (from indexed PDFs):
{context}

Answer:"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        options={"temperature": 0.1, "num_ctx": 8192},
    )

    raw_answer = response["message"]["content"]

    # ── Parse JSON payload if present ─────────────────────────────────────────
    format_data = None
    clean_answer = raw_answer

    if output_format != "text":
        import re
        json_match = re.search(r"```json\s*([\s\S]+?)\s*```", raw_answer, re.IGNORECASE)
        if json_match:
            try:
                format_data  = json.loads(json_match.group(1))
                clean_answer = raw_answer[:json_match.start()].strip()
            except json.JSONDecodeError:
                pass

    # ── Deduplicate sources ───────────────────────────────────────────────────
    seen    = set()
    sources = []
    for c in chunks:
        key = (c["pdf_name"], c["page"])
        if key not in seen:
            seen.add(key)
            sources.append({
                "pdf_name": c["pdf_name"],
                "pdf_path": c["pdf_path"],
                "year":     c.get("year"),
                "page":     c["page"],
                "score":    c["score"],
            })

    return {
        "answer":      clean_answer,
        "sources":     sources,
        "format_data": format_data,
        "output_format": output_format,
    }


# ── Index status ──────────────────────────────────────────────────────────────

def get_index_status() -> dict:
    manifest   = _load_manifest()
    collection = get_or_create_collection()
    return {
        "indexed_files":    len(manifest),
        "total_chunks":     collection.count(),
        "files":            list(manifest.keys()),
    }