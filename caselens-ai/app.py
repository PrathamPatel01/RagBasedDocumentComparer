import streamlit as st
import fitz
import chromadb
import ollama
from sentence_transformers import SentenceTransformer
from uuid import uuid4

st.set_page_config(page_title="CaseLens AI", layout="wide")

st.title("CaseLens AI")
st.caption("Local-first RAG pipeline using ChromaDB + SentenceTransformers + Ollama")

MAX_PAGES = 200
MODEL_NAME = "mistral"


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def get_chroma_client():
    return chromadb.PersistentClient(path="./chroma_db")


embedding_model = load_embedding_model()
client = get_chroma_client()
collection = client.get_or_create_collection("documents")

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

if "full_docs" not in st.session_state:
    st.session_state.full_docs = {}


def extract_text_from_pdf(uploaded_file) -> str:
    text = ""
    pdf = fitz.open(stream=uploaded_file.read(), filetype="pdf")

    if len(pdf) > MAX_PAGES:
        st.error(
            f"'{uploaded_file.name}' has {len(pdf)} pages. "
            f"Maximum allowed is {MAX_PAGES} pages."
        )
        return ""

    for page_num, page in enumerate(pdf):
        page_text = page.get_text()
        text += f"\n\n[Page {page_num + 1}]\n{page_text}"

    return text


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 180) -> list[str]:
    chunks = []
    start = 0

    while start < len(text):
        chunk = text[start:start + chunk_size]

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def is_already_indexed(filename: str) -> bool:
    results = collection.get(where={"filename": filename}, limit=1)
    return len(results["ids"]) > 0


def add_document_to_chroma(filename: str, text: str) -> int:
    chunks = chunk_text(text)

    if not chunks:
        return 0

    embeddings = embedding_model.encode(
        chunks,
        show_progress_bar=False
    ).tolist()

    ids = [str(uuid4()) for _ in chunks]
    metadatas = [
        {
            "filename": filename,
            "chunk": i
        }
        for i in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas
    )

    return len(chunks)


def get_indexed_filenames():
    if st.session_state.indexed_files:
        return st.session_state.indexed_files

    if collection.count() == 0:
        return []

    results = collection.get(include=["metadatas"])
    filenames = sorted(set(meta["filename"] for meta in results["metadatas"]))

    st.session_state.indexed_files = filenames
    return filenames


def retrieve_context(query: str, top_k: int = 6) -> str:
    if collection.count() == 0:
        return ""

    query_embedding = embedding_model.encode(query).tolist()
    filenames = get_indexed_filenames()

    context = ""

    for filename in filenames:
        file_items = collection.get(
            where={"filename": filename},
            include=[]
        )

        n = min(top_k, len(file_items["ids"]))

        if n == 0:
            continue

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n,
            where={"filename": filename}
        )

        context += f"\n\n===== DOCUMENT: {filename} =====\n"

        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            context += (
                f"\nSource: {meta['filename']} | Chunk: {meta['chunk']}\n"
                f"{doc}\n"
            )

    return context


def get_full_document_context() -> str:
    context = ""

    for filename, text in st.session_state.full_docs.items():
        context += f"\n\n===== DOCUMENT: {filename} =====\n{text}\n"

    return context


SYSTEM_PROMPT = """
You are CaseLens AI, a document analysis assistant.

Rules:
- Use only the provided context.
- Do not invent facts.
- If something is missing, say it is not available in the documents.
- Include source filename and page/chunk references whenever possible.
- Be clear, structured, and concise.
"""


def ask_llm_stream(prompt: str):
    try:
        stream = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            stream=True
        )

        for chunk in stream:
            yield chunk["message"]["content"]

    except Exception as e:
        yield (
            f"\n\n⚠️ **LLM error:** {e}\n\n"
            f"Make sure Ollama is running and the `{MODEL_NAME}` model exists.\n\n"
            f"Run:\n\n```bash\nollama pull {MODEL_NAME}\n```"
        )


uploaded_files = st.file_uploader(
    f"Upload 2-3 case law, research, or resume PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

col1, col2 = st.columns([1, 1])

with col1:
    process_clicked = st.button(
        "Process Documents",
        disabled=not uploaded_files,
        type="primary"
    )

with col2:
    clear_clicked = st.button("Clear Vector DB")

if process_clicked:
    for file in uploaded_files:
        if is_already_indexed(file.name):
            st.info(f"'{file.name}' is already indexed — skipping.")
            continue

        with st.spinner(f"Processing {file.name}..."):
            text = extract_text_from_pdf(file)

            if not text.strip():
                st.error(f"No text found in '{file.name}'. It may be scanned/image-based.")
                continue

            st.session_state.full_docs[file.name] = text

            chunks_added = add_document_to_chroma(file.name, text)

            if file.name not in st.session_state.indexed_files:
                st.session_state.indexed_files.append(file.name)

            st.success(f"'{file.name}' processed successfully. Chunks stored: {chunks_added}")

if clear_clicked:
    try:
        client.delete_collection("documents")
    except Exception:
        pass

    collection = client.get_or_create_collection("documents")
    st.session_state.indexed_files = []
    st.session_state.full_docs = {}
    st.success("Vector database cleared.")
    st.rerun()

st.info(f"Current chunks in ChromaDB: {collection.count()}")

if get_indexed_filenames():
    st.caption("Indexed files: " + ", ".join(get_indexed_filenames()))

docs_ready = collection.count() > 0

if not docs_ready:
    st.warning("Upload PDFs and click Process Documents first.")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(
    ["Ask Questions", "Summaries", "Comparison Table", "Timeline"]
)

with tab1:
    question = st.text_input(
        "Ask a question about the uploaded documents",
        placeholder="Example: What are the main differences between these documents?",
        disabled=not docs_ready
    )

    if st.button("Ask", disabled=not docs_ready or not question.strip()):
        context = retrieve_context(question, top_k=8)

        if not context.strip():
            st.warning("No relevant context found.")
        else:
            prompt = f"""
Context:
{context}

Question:
{question}

Answer clearly.
Cite filename and chunk/page references.
"""

            with st.spinner("Thinking..."):
                st.write_stream(ask_llm_stream(prompt))

            with st.expander("Retrieved Context"):
                st.text(context)


with tab2:
    if st.button("Generate Summaries", disabled=not docs_ready):
        full_context = get_full_document_context()

        if full_context.strip():
            context = full_context
        else:
            context = retrieve_context(
                "main facts arguments findings conclusions summary",
                top_k=14
            )

        prompt = f"""
Create a separate summary for EACH document.

Context:
{context}

For each document, use this format:

## Document Name
- Main topic
- Purpose
- Key facts or ideas
- Important arguments or findings
- Conclusion
- What makes this document different from the others
- Source references

Do not mix documents together.
"""

        with st.spinner("Generating summaries..."):
            st.write_stream(ask_llm_stream(prompt))


with tab3:
    if st.button("Generate Comparison Table", disabled=not docs_ready):
        full_context = get_full_document_context()

        if full_context.strip():
            context = full_context
        else:
            context = retrieve_context(
                "compare documents issues facts arguments conclusions differences similarities",
                top_k=14
            )

        prompt = f"""
Compare the uploaded documents carefully.

Context:
{context}

Create a markdown table with these columns:

Category | Document 1 | Document 2 | Key Difference

Compare:
- Main purpose
- Target audience or use case
- Core topic
- Key facts
- Skills/methods/arguments
- Projects/examples/evidence
- Strengths
- Weaknesses
- Best use case

Do not invent information.
If a field is missing, write "Not available in document".
"""

        with st.spinner("Generating comparison table..."):
            st.write_stream(ask_llm_stream(prompt))


with tab4:
    if st.button("Generate Timeline", disabled=not docs_ready):
        full_context = get_full_document_context()

        if full_context.strip():
            context = full_context
        else:
            context = retrieve_context(
                "timeline chronology dates events sequence history",
                top_k=14
            )

        prompt = f"""
Create a combined chronological timeline from the documents.

Context:
{context}

Use this exact markdown table:

| Date / Time | Event | Source Document |
|---|---|---|

Rules:
- Only include dates or time periods found in the documents.
- Do not invent dates.
- If an event has no date, put "Not dated".
- Keep events in chronological order when dates are available.
- Mention the source document.
"""

        with st.spinner("Generating timeline..."):
            st.write_stream(ask_llm_stream(prompt))




#-------------
# import streamlit as st
# import fitz
# import chromadb
# import ollama
# from sentence_transformers import SentenceTransformer
# from uuid import uuid4

# st.set_page_config(page_title="CaseLens AI", layout="wide")

# st.title("CaseLens AI")
# st.caption("Local-first RAG pipeline using ChromaDB + SentenceTransformers + Ollama")


# # ── Cached resources ──────────────────────────────────────────────────────────

# @st.cache_resource
# def load_embedding_model():
#     return SentenceTransformer("all-MiniLM-L6-v2")


# @st.cache_resource
# def get_chroma_client():
#     return chromadb.PersistentClient(path="./chroma_db")


# embedding_model = load_embedding_model()
# client = get_chroma_client()
# collection = client.get_or_create_collection("documents")

# # Track indexed filenames in session state (avoids full metadata scan on every query)
# if "indexed_files" not in st.session_state:
#     st.session_state.indexed_files = []


# # ── PDF helpers ───────────────────────────────────────────────────────────────

# MAX_PAGES = 50


# def extract_text_from_pdf(uploaded_file) -> str:
#     """
#     Extract plain text from a PDF.
#     Returns empty string and shows an error if the file exceeds MAX_PAGES.
#     Page markers are intentionally omitted to keep chunks clean.
#     """
#     pdf = fitz.open(stream=uploaded_file.read(), filetype="pdf")

#     if len(pdf) > MAX_PAGES:
#         st.error(
#             f"'{uploaded_file.name}' has {len(pdf)} pages — maximum is {MAX_PAGES}. "
#             "Please upload a shorter document."
#         )
#         return ""

#     return "\n\n".join(page.get_text() for page in pdf)


# def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
#     """Split text into overlapping character-level chunks."""
#     chunks, start = [], 0
#     while start < len(text):
#         chunk = text[start : start + chunk_size]
#         if chunk.strip():
#             chunks.append(chunk)
#         start += chunk_size - overlap
#     return chunks


# # ── ChromaDB helpers ──────────────────────────────────────────────────────────

# def is_already_indexed(filename: str) -> bool:
#     """Check whether a filename has already been stored in the collection."""
#     results = collection.get(where={"filename": filename}, limit=1)
#     return len(results["ids"]) > 0


# def add_document_to_chroma(filename: str, text: str) -> int:
#     """
#     Chunk text, batch-encode embeddings, and store in ChromaDB.
#     Returns the number of chunks added.
#     """
#     chunks = chunk_text(text)
#     if not chunks:
#         return 0

#     # Batch encode — 5-20× faster than encoding one chunk at a time
#     embeddings = embedding_model.encode(chunks, show_progress_bar=False).tolist()

#     ids        = [str(uuid4()) for _ in chunks]
#     metadatas  = [{"filename": filename, "chunk": i} for i in range(len(chunks))]

#     collection.add(
#         ids=ids,
#         embeddings=embeddings,
#         documents=chunks,
#         metadatas=metadatas,
#     )
#     return len(chunks)


# def retrieve_context(query: str, top_k: int = 6) -> str:
#     """
#     Retrieve the most relevant chunks for each indexed file and
#     assemble them into a single context string.
#     """
#     if collection.count() == 0:
#         return ""

#     query_embedding = embedding_model.encode(query).tolist()

#     # Use session-state filenames (avoids scanning all metadata)
#     filenames = st.session_state.indexed_files or sorted(
#         set(m["filename"] for m in collection.get(include=["metadatas"])["metadatas"])
#     )

#     context = ""
#     for filename in filenames:
#         # Count chunks for this specific file so n_results never exceeds what exists
#         file_ids = collection.get(where={"filename": filename}, include=[])["ids"]
#         n = min(top_k, len(file_ids))
#         if n == 0:
#             continue

#         results = collection.query(
#             query_embeddings=[query_embedding],
#             n_results=n,
#             where={"filename": filename},
#         )

#         context += f"\n\n===== DOCUMENT: {filename} =====\n"
#         for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
#             context += f"\nSource: {meta['filename']} | Chunk: {meta['chunk']}\n{doc}\n"

#     return context


# # ── LLM helpers ───────────────────────────────────────────────────────────────

# SYSTEM_PROMPT = (
#     "You are a document analysis assistant. "
#     "Use only the provided context. "
#     "Always include source filename and chunk number in your answer."
# )


# def ask_llm_stream(prompt: str):
#     """
#     Stream tokens from Ollama one at a time.
#     Yields string chunks; yields an error message on failure.
#     """
#     try:
#         stream = ollama.chat(
#             model="mistral",
#             messages=[
#                 {"role": "system", "content": SYSTEM_PROMPT},
#                 {"role": "user",   "content": prompt},
#             ],
#             stream=True,
#         )
#         for chunk in stream:
#             yield chunk["message"]["content"]
#     except Exception as e:
#         yield f"\n\n⚠️ **LLM error:** {e}\n\nMake sure Ollama is running and the `mistral` model is pulled (`ollama pull mistral`)."


# # ── Sidebar: document management ──────────────────────────────────────────────

# with st.sidebar:
#     st.header("Documents")

#     uploaded_files = st.file_uploader(
#         "Upload PDFs (max 50 pages each)",
#         type=["pdf"],
#         accept_multiple_files=True,
#     )

#     if st.button("Process Documents", disabled=not uploaded_files, type="primary"):
#         for file in uploaded_files:
#             if is_already_indexed(file.name):
#                 st.info(f"'{file.name}' is already indexed — skipping.")
#                 continue

#             with st.spinner(f"Processing {file.name}…"):
#                 text = extract_text_from_pdf(file)
#                 if not text.strip():
#                     st.error(f"No text found in '{file.name}'.")
#                     continue

#                 n = add_document_to_chroma(file.name, text)
#                 if file.name not in st.session_state.indexed_files:
#                     st.session_state.indexed_files.append(file.name)
#                 st.success(f"'{file.name}' → {n} chunks stored.")

#     st.divider()

#     if st.button("Clear Vector DB", type="secondary"):
#         client.delete_collection("documents")
#         # Recreate so the rest of the app doesn't use a deleted handle
#         collection = client.get_or_create_collection("documents")
#         st.session_state.indexed_files = []
#         st.success("Vector database cleared.")
#         st.rerun()

#     total = collection.count()
#     st.caption(f"Total chunks in DB: **{total}**")

#     if st.session_state.indexed_files:
#         st.caption("Indexed files:")
#         for f in st.session_state.indexed_files:
#             st.caption(f"• {f}")


# # ── Main area: tabs ───────────────────────────────────────────────────────────

# docs_ready = collection.count() > 0

# if not docs_ready:
#     st.info("Upload PDFs in the sidebar and click **Process Documents** to get started.")

# tab1, tab2, tab3, tab4 = st.tabs(
#     ["Ask Questions", "Summaries", "Comparison Table", "Timeline"]
# )

# # ── Tab 1: Q&A ────────────────────────────────────────────────────────────────

# with tab1:
#     question = st.text_input(
#         "Ask a question about the uploaded documents",
#         placeholder="e.g. What was the court's ruling on admissibility?",
#         disabled=not docs_ready,
#     )

#     if st.button("Ask", disabled=not docs_ready or not question.strip()):
#         context = retrieve_context(question, top_k=8)
#         if not context.strip():
#             st.warning("No relevant context found. Try rephrasing your question.")
#         else:
#             prompt = (
#                 f"Context:\n{context}\n\n"
#                 f"Question:\n{question}\n\n"
#                 "Answer clearly, citing source filename and chunk number."
#             )
#             with st.spinner("Thinking…"):
#                 st.write_stream(ask_llm_stream(prompt))

# # ── Tab 2: Summaries ──────────────────────────────────────────────────────────

# with tab2:
#     if st.button("Generate Summaries", disabled=not docs_ready):
#         context = retrieve_context(
#             "main facts arguments findings conclusions summary", top_k=14
#         )
#         prompt = (
#             f"Create a concise summary for each document below.\n\n"
#             f"Context:\n{context}\n\n"
#             "Format each summary as:\n"
#             "## [Document name]\n"
#             "- **Main topic**\n"
#             "- **Key facts**\n"
#             "- **Main arguments**\n"
#             "- **Conclusion**\n"
#             "- **Source references**"
#         )
#         with st.spinner("Generating summaries…"):
#             st.write_stream(ask_llm_stream(prompt))

# # ── Tab 3: Comparison table ───────────────────────────────────────────────────

# with tab3:
#     if st.button("Generate Comparison Table", disabled=not docs_ready):
#         context = retrieve_context(
#             "compare documents issues facts arguments conclusions differences similarities",
#             top_k=14,
#         )
#         prompt = (
#             f"Compare the documents using this context:\n\n{context}\n\n"
#             "Produce a markdown table with these columns:\n"
#             "Document | Topic | Key Issue | Main Argument | Evidence | Conclusion | Similarities | Differences"
#         )
#         with st.spinner("Building comparison table…"):
#             st.write_stream(ask_llm_stream(prompt))

# # ── Tab 4: Timeline ───────────────────────────────────────────────────────────

# with tab4:
#     if st.button("Generate Timeline", disabled=not docs_ready):
#         context = retrieve_context(
#             "timeline chronology dates events sequence history", top_k=14
#         )
#         prompt = (
#             f"Extract a chronological timeline from the documents below.\n\n"
#             f"Context:\n{context}\n\n"
#             "Use this exact markdown table format:\n\n"
#             "| Date / Time | Event | Document Source |\n"
#             "|---|---|---|\n"
#             "List events in chronological order. If an exact date is missing, use the most specific time reference available."
#         )
#         with st.spinner("Building timeline…"):
#             st.write_stream(ask_llm_stream(prompt))