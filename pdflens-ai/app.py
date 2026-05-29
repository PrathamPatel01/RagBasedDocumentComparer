
"""
app.py — Streamlit UI only.
All RAG logic lives in rag.py, all LLM logic in llm.py, all constants in config.py.
"""

import streamlit as st

from config import MODEL_NAME, EMBEDDING_MODEL
from rag import (
    load_embedding_model,
    get_collection,
    get_chroma_client,
    extract_text_from_pdf,
    add_document_to_chroma,
    is_already_indexed,
    get_indexed_filenames,
    retrieve_context,
    clear_vector_db,
)
from llm import (
    ask_llm_stream,
    collect_stream,
    build_qa_prompt,
    build_summary_prompt,
    build_comparison_presummary_prompt,
    build_comparison_table_prompt,
    build_timeline_prompt,
)

st.set_page_config(page_title="PDFLens AI", layout="wide", page_icon="⚖️")

# Warm up the embedding model on first load
load_embedding_model()


# ─── Session State ────────────────────────────────────────────────────────────

for key, default in [
    ("indexed_files", []),
    ("full_docs", {}),
    ("chat_history", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("📂 Indexed Documents")

    chunk_count = get_collection().count()
    indexed = get_indexed_filenames()

    if indexed:
        st.success(f"{len(indexed)} doc(s) · {chunk_count} chunks in DB")
        for f in indexed:
            st.markdown(f"- 📄 `{f}`")
    else:
        st.info("No documents indexed yet.\nUpload PDFs above to get started.")

    st.divider()
    st.markdown(f"**Model:** `{MODEL_NAME}` via Ollama")
    st.markdown(f"**Embeddings:** `{EMBEDDING_MODEL}`")
    st.markdown("**Vector DB:** ChromaDB (local persistent)")


# ─── Header ───────────────────────────────────────────────────────────────────

st.title("⚖️ PDFLens AI")
st.caption("Local RAG pipeline · ChromaDB + SentenceTransformers + Ollama")


# ─── Upload & Controls ────────────────────────────────────────────────────────

uploaded_files = st.file_uploader(
    "Upload PDFs (case law, research papers— up to 200 pages each)",
    type=["pdf"],
    accept_multiple_files=True,
    help="Files are chunked and embedded into a local ChromaDB vector store.",
)

col1, col2 = st.columns(2)

with col1:
    process_clicked = st.button(
        "⚙️ Process Documents",
        disabled=not uploaded_files,
        type="primary",
        use_container_width=True,
    )

with col2:
    clear_clicked = st.button(
        "🗑️ Clear Vector DB",
        use_container_width=True,
        help="Permanently removes all indexed documents from ChromaDB.",
    )

if process_clicked:
    any_new = False

    for file in uploaded_files:
        if is_already_indexed(file.name):
            st.info(f"⏭️ **'{file.name}'** is already indexed — skipping.")
            continue

        with st.spinner(f"Reading and indexing **{file.name}**…"):
            text = extract_text_from_pdf(file)

            if not text.strip():
                st.error(
                    f"❌ No extractable text in **'{file.name}'**. "
                    "It may be a scanned or image-only PDF."
                )
                continue

            st.session_state.full_docs[file.name] = text
            chunks_added = add_document_to_chroma(file.name, text)

            if file.name not in st.session_state.indexed_files:
                st.session_state.indexed_files.append(file.name)

            st.success(f"✅ **'{file.name}'** indexed — {chunks_added} chunks stored.")
            any_new = True

    if any_new:
        st.rerun()

if clear_clicked:
    clear_vector_db()
    st.session_state.indexed_files = []
    st.session_state.full_docs = {}
    st.session_state.chat_history = []
    st.success("🗑️ Vector database cleared. All documents removed.")
    st.rerun()


# ─── Status ───────────────────────────────────────────────────────────────────

docs_ready = get_collection().count() > 0

if not docs_ready:
    st.warning("⬆️ Upload one or more PDFs and click **Process Documents** to get started.")

st.divider()


# ─── Tabs ─────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "💬 Ask Questions",
    "📝 Summaries",
    "📊 Comparison Table",
    "📅 Timeline",
])


# ── Tab 1: Ask Questions ──────────────────────────────────────────────────────

with tab1:
    st.subheader("Ask anything about your documents")

    q_col, filter_col = st.columns([3, 1])

    with filter_col:
        filter_options = ["All Documents"] + get_indexed_filenames()
        selected_filter = st.selectbox(
            "Scope",
            filter_options,
            disabled=not docs_ready,
            help="Narrow the search to a single document or query across all.",
        )
        filename_filter = None if selected_filter == "All Documents" else selected_filter

    with q_col:
        question = st.text_input(
            "Your question",
            placeholder="e.g. What are the main arguments made in this case?",
            disabled=not docs_ready,
            label_visibility="collapsed",
        )

    btn_col, clear_col = st.columns([3, 1])

    with btn_col:
        ask_clicked = st.button(
            "Ask",
            disabled=not docs_ready or not question.strip(),
            type="primary",
            use_container_width=True,
        )

    with clear_col:
        if st.button(
            "Clear Chat",
            disabled=not st.session_state.chat_history,
            use_container_width=True,
        ):
            st.session_state.chat_history = []
            st.rerun()

    # Render existing chat history
    for entry in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(entry["question"])
        with st.chat_message("assistant"):
            st.markdown(entry["answer"])
            if entry.get("context"):
                with st.expander("📎 View retrieved context"):
                    st.code(entry["context"])

    # Handle new question
    if ask_clicked and question.strip():
        context = retrieve_context(question, top_k=8, filename_filter=filename_filter)

        if not context.strip():
            st.warning(
                "No relevant chunks found for your query. "
                "Try rephrasing, or check that the right documents are indexed."
            )
        else:
            prompt = build_qa_prompt(context, question)

            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                response = st.write_stream(ask_llm_stream(prompt))
                with st.expander("📎 View retrieved context"):
                    st.code(context)

            st.session_state.chat_history.append({
                "question": question,
                "answer": response,
                "context": context,
            })


# ── Tab 2: Summaries ──────────────────────────────────────────────────────────

with tab2:
    st.subheader("Per-document summaries")
    st.caption(
        "Each document is summarized independently using its most relevant chunks. "
        "Summaries do not cross-reference other uploaded files."
    )

    if st.button("Generate Summaries", disabled=not docs_ready, type="primary"):
        for filename in get_indexed_filenames():
            context = retrieve_context(
                query="main topic purpose key issues arguments findings conclusion",
                top_k=20,
                filename_filter=filename,
            )

            st.subheader(f"📄 {filename}")
            with st.spinner(f"Summarizing '{filename}'…"):
                st.write_stream(ask_llm_stream(build_summary_prompt(filename, context)))
            st.divider()


# ── Tab 3: Comparison Table ───────────────────────────────────────────────────

with tab3:
    st.subheader("Document-wise comparison")
    st.caption(
        "Each document is summarized first, then compared side-by-side in a structured table. "
        "Requires at least 2 indexed documents."
    )

    if st.button("Generate Comparison Table", disabled=not docs_ready, type="primary"):
        filenames = get_indexed_filenames()

        if len(filenames) < 2:
            st.warning(
                "You need **at least 2 indexed documents** to run a comparison. "
                "Upload and process another PDF first."
            )
        else:
            document_summaries: dict[str, str] = {}

            for filename in filenames:
                with st.spinner(f"Pre-summarizing '{filename}' for comparison…"):
                    context = retrieve_context(
                        query="main topic purpose key issues arguments findings conclusion scope",
                        top_k=24,
                        filename_filter=filename,
                    )
                    prompt = build_comparison_presummary_prompt(filename, context)
                    document_summaries[filename] = collect_stream(prompt)

            combined = "\n\n".join(
                f"===== DOCUMENT SUMMARY: {fn} =====\n{s}"
                for fn, s in document_summaries.items()
            )

            st.markdown("### Comparison Table")
            with st.spinner("Building comparison table…"):
                st.write_stream(
                    ask_llm_stream(build_comparison_table_prompt(filenames, combined))
                )

            with st.expander("📋 Intermediate document summaries"):
                for fn, summary in document_summaries.items():
                    st.markdown(f"#### 📄 {fn}")
                    st.markdown(summary)
                    st.divider()


# ── Tab 4: Timeline ───────────────────────────────────────────────────────────

with tab4:
    st.subheader("Chronological timeline")
    st.caption(
        "Extracts all dates and time-referenced events found across your documents "
        "and orders them chronologically."
    )

    if st.button("Generate Timeline", disabled=not docs_ready, type="primary"):
        timeline_context = ""

        for filename in get_indexed_filenames():
            context = retrieve_context(
                query="dates timeline chronology events publication year sequence history",
                top_k=20,
                filename_filter=filename,
            )
            timeline_context += f"\n\n===== DOCUMENT: {filename} =====\n{context}"

        with st.spinner("Extracting timeline events…"):
            st.write_stream(ask_llm_stream(build_timeline_prompt(timeline_context)))