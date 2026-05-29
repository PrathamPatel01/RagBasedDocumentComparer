"""
rag.py — chunking, embedding, ChromaDB storage and retrieval.
No Streamlit imports here; pure data logic only.
"""

import streamlit as st
import fitz
import chromadb
from sentence_transformers import SentenceTransformer
from uuid import uuid4
import re 

from config import (
    EMBEDDING_MODEL,
    CHROMA_PATH,
    COLLECTION_NAME,
    MAX_PAGES,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBED_BATCH_SIZE,
)


# ─── Cached Resources ─────────────────────────────────────────────────────────

@st.cache_resource
def load_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL, device="cpu")


@st.cache_resource
def get_chroma_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=CHROMA_PATH)


def get_collection() -> chromadb.Collection:
    """Always returns the live collection — never a stale module-level reference."""
    return get_chroma_client().get_or_create_collection(COLLECTION_NAME)


# ─── PDF Extraction ───────────────────────────────────────────────────────────

def extract_text_from_pdf(uploaded_file) -> str:
    """
    Extract plain text from an uploaded PDF file.
    Returns empty string on failure; caller checks and shows errors via Streamlit.
    """
    try:
        pdf = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    except Exception as e:
        st.error(f"Could not open **'{uploaded_file.name}'**: {e}")
        return ""

    if len(pdf) > MAX_PAGES:
        st.error(
            f"**'{uploaded_file.name}'** has {len(pdf)} pages — "
            f"the limit is {MAX_PAGES}. Please split the PDF first."
        )
        return ""

    text = ""
    for page_num, page in enumerate(pdf):
        text += f"\n\n[Page {page_num + 1}]\n{page.get_text()}"

    return text


# ─── Chunking ─────────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Split text into overlapping chunks, snapping to sentence boundaries
    so chunks don't cut mid-sentence.
    """
    chunks: list[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        if end < text_len:
            for sep in (". ", ".\n", "\n\n", "\n"):
                boundary = text.rfind(sep, start + chunk_size // 2, end)
                if boundary != -1:
                    end = boundary + len(sep)
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        next_start = end - overlap
        if next_start <= start:  # overlap >= chunk_size would loop forever
            break
        start = next_start

    return chunks


# ─── Indexing ─────────────────────────────────────────────────────────────────

def is_already_indexed(filename: str) -> bool:
    results = get_collection().get(where={"filename": filename}, limit=1)
    return len(results["ids"]) > 0


def add_document_to_chroma(filename: str, text: str) -> int:
    """
    Chunk, embed, and store a document in ChromaDB.
    Returns the number of chunks stored (0 if nothing to store).
    """
    chunks = chunk_text(text)
    if not chunks:
        return 0

    model = load_embedding_model()
    embeddings = model.encode(
        chunks,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=False,
    ).tolist()

    ids = [str(uuid4()) for _ in chunks]
    metadatas = [{"filename": filename, "chunk": i} for i in range(len(chunks))]

    get_collection().add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )
    return len(chunks)


# ─── Query ────────────────────────────────────────────────────────────────────

def get_indexed_filenames() -> list[str]:
    collection = get_collection()
    if collection.count() == 0:
        return []
    results = collection.get(include=["metadatas"])
    filenames = sorted(set(meta["filename"] for meta in results["metadatas"]))
    # Mirror into session state so UI doesn't need to re-query
    st.session_state.indexed_files = filenames
    return filenames


def retrieve_context(
    query: str,
    top_k: int = 6,
    filename_filter: str | None = None,
) -> str:
    """
    Embed the query, search ChromaDB, and return a formatted context string
    with source file and chunk references.
    """
    collection = get_collection()
    if collection.count() == 0:
        return ""

    model = load_embedding_model()
    query_embedding = model.encode(query).tolist()
    filenames = [filename_filter] if filename_filter else get_indexed_filenames()
    context = ""

    for filename in filenames:
        file_items = collection.get(where={"filename": filename}, include=[])
        n = min(top_k, len(file_items["ids"]))
        if n == 0:
            continue

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n,
            where={"filename": filename},
        )

        context += f"\n\n===== DOCUMENT: {filename} =====\n"
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            context += f"\nSource: {meta['filename']} | Chunk {meta['chunk']}\n{doc}\n"

    return context


def clear_vector_db() -> None:
    """Delete and immediately recreate the ChromaDB collection."""
    try:
        get_chroma_client().delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    get_collection()  # recreates it empty


    import re

DATE_RE = re.compile(
    r"\b("
    r"\d{1,2}\s+[A-Z][a-z]+\s+\d{4}|"
    r"[A-Z][a-z]+\s+\d{1,2},\s+\d{4}|"
    r"\d{4}-\d{2}-\d{2}|"
    r"\b\d{4}\b"
    r")\b"
)


def retrieve_timeline_context(top_k_per_doc: int = 60) -> str:
    collection = get_collection()

    if collection.count() == 0:
        return ""

    context = ""

    for filename in get_indexed_filenames():
        results = collection.get(
            where={"filename": filename},
            include=["documents", "metadatas"],
        )

        dated_chunks = []

        for doc, meta in zip(results["documents"], results["metadatas"]):
            if DATE_RE.search(doc):
                dated_chunks.append((meta["chunk"], doc))

        dated_chunks = dated_chunks[:top_k_per_doc]

        if dated_chunks:
            context += f"\n\n===== DOCUMENT: {filename} =====\n"

            for chunk_num, doc in dated_chunks:
                context += f"\nSource: {filename} | Chunk {chunk_num}\n{doc}\n"

    return context