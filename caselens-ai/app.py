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


def retrieve_context(query: str, top_k: int = 6, filename_filter: str | None = None) -> str:
    if collection.count() == 0:
        return ""

    query_embedding = embedding_model.encode(query).tolist()

    if filename_filter:
        filenames = [filename_filter]
    else:
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


def collect_stream(prompt: str) -> str:
    output = ""

    for chunk in ask_llm_stream(prompt):
        output += chunk

    return output


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
    "Upload 2-3 case law, research, or resume PDFs",
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
    ["Ask Questions", "Summaries", "Document-wise Comparison", "Timeline"]
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
                st.code(context)


with tab2:
    if st.button("Generate Summaries", disabled=not docs_ready):
        filenames = get_indexed_filenames()

        for filename in filenames:
            context = retrieve_context(
                query="main topic purpose key issues arguments findings conclusion",
                top_k=20,
                filename_filter=filename
            )

            prompt = f"""
Summarize this ONE document only.

Document filename:
{filename}

Context:
{context}

Use this format:

## {filename}

- **Document type:**
- **Main purpose:**
- **Core topic:**
- **Key issues:**
- **Important facts:**
- **Main arguments/findings:**
- **Conclusion:**
- **Best use case:**
- **Source references:**

Rules:
- Do not compare with other documents.
- Do not invent facts.
- Use only the context from this document.
"""

            st.subheader(filename)

            with st.spinner(f"Generating summary for {filename}..."):
                st.write_stream(ask_llm_stream(prompt))


with tab3:
    if st.button("Generate Document-wise Comparison Table", disabled=not docs_ready):
        filenames = get_indexed_filenames()
        document_summaries = {}

        with st.spinner("Creating document-level summaries first..."):
            for filename in filenames:
                context = retrieve_context(
                    query="main topic purpose key issues arguments findings conclusion scope",
                    top_k=24,
                    filename_filter=filename
                )

                summary_prompt = f"""
You are summarizing ONE document only.

Document filename:
{filename}

Context:
{context}

Create a factual document-level summary.

Return this structure:

Document:
Document type:
Main purpose:
Core topic:
Legal or research focus:
Key issues:
Important facts:
Main arguments or findings:
Type of evidence or material:
Scope:
Strengths:
Limitations:
Best use case:
Source references:

Rules:
- Summarize only this document.
- Do not mention other documents.
- Do not invent facts.
- Include filename and chunk references.
"""

                summary_text = collect_stream(summary_prompt)
                document_summaries[filename] = summary_text

        combined_summary_context = ""

        for filename, summary in document_summaries.items():
            combined_summary_context += f"""

===== DOCUMENT SUMMARY: {filename} =====
{summary}
"""

        filename_columns = " | ".join(filenames)

        comparison_prompt = f"""
Compare the documents using ONLY these document-level summaries.

{combined_summary_context}

Create a document-wise markdown comparison table.

Columns:
Category | {filename_columns} | Key Difference

Rows:
- Document type
- Main purpose
- Core topic
- Legal/historical focus
- Main issues discussed
- Type of evidence/material
- Important findings
- Scope
- Strengths
- Limitations
- Best use case

Rules:
- Compare by filename.
- Do not treat sections inside one PDF as separate documents.
- Do not invent facts.
- If information is missing, write "Not available".
"""

        with st.spinner("Building document-wise comparison table..."):
            st.write_stream(ask_llm_stream(comparison_prompt))

        with st.expander("Intermediate document summaries"):
            for filename, summary in document_summaries.items():
                st.markdown(f"### {filename}")
                st.markdown(summary)


with tab4:
    if st.button("Generate Timeline", disabled=not docs_ready):
        filenames = get_indexed_filenames()
        timeline_context = ""

        for filename in filenames:
            context = retrieve_context(
                query="dates timeline chronology events publication year sequence history",
                top_k=20,
                filename_filter=filename
            )

            timeline_context += f"""

===== DOCUMENT: {filename} =====
{context}
"""

        prompt = f"""
Create a combined chronological timeline from the documents.

Context:
{timeline_context}

Use this exact markdown table:

| Date / Time | Event | Source Document |
|---|---|---|

Rules:
- Only include dates or time periods found in the documents.
- Do not invent dates.
- If an event has no date, put "Not dated".
- Keep events in chronological order when dates are available.
- Mention the source document filename.
- Do not treat sections inside one PDF as separate documents.
"""

        with st.spinner("Generating timeline..."):
            st.write_stream(ask_llm_stream(prompt))

