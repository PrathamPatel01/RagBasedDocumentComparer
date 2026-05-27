import streamlit as st
import fitz
import chromadb
import ollama
from sentence_transformers import SentenceTransformer
from uuid import uuid4

st.set_page_config(page_title="CaseLens AI", layout="wide")

st.title("CaseLens AI")
st.caption("Local-first RAG pipeline using ChromaDB + SentenceTransformers + Ollama")


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


embedding_model = load_embedding_model()

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("documents")


def extract_text_from_pdf(uploaded_file):
    text = ""
    pdf = fitz.open(stream=uploaded_file.read(), filetype="pdf")

    for page_num, page in enumerate(pdf):
        text += f"\n\n[Page {page_num + 1}]\n{page.get_text()}"

    return text


def chunk_text(text, chunk_size=900, overlap=150):
    chunks = []
    start = 0

    while start < len(text):
        chunk = text[start:start + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def add_document_to_chroma(filename, text):
    chunks = chunk_text(text)

    for i, chunk in enumerate(chunks):
        embedding = embedding_model.encode(chunk).tolist()

        collection.add(
            ids=[str(uuid4())],
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[{"filename": filename, "chunk": i}]
        )

    return len(chunks)


def retrieve_context(query, top_k=8):
    if collection.count() == 0:
        return ""

    query_embedding = embedding_model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count())
    )

    context = ""

    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        context += f"\nSource: {meta['filename']} | Chunk: {meta['chunk']}\n{doc}\n"

    return context


def ask_llm(prompt):
    response = ollama.chat(
        model="mistral",
        messages=[
            {
                "role": "system",
                "content": "You are a document analysis assistant. Use only the given context. Include source references."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


uploaded_files = st.file_uploader(
    "Upload 2-3 case law or research PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

col1, col2 = st.columns(2)

with col1:
    if st.button("Process Documents", disabled=not uploaded_files):
        total_chunks = 0

        with st.spinner("Processing PDFs..."):
            for file in uploaded_files:
                text = extract_text_from_pdf(file)

                if text.strip():
                    total_chunks += add_document_to_chroma(file.name, text)
                else:
                    st.error(f"No text found in {file.name}")

        st.success(f"Documents processed successfully. Total chunks stored: {collection.count()}")

with col2:
    if st.button("Clear Vector DB"):
        client.delete_collection("documents")
        collection = client.get_or_create_collection("documents")
        st.success("Vector database cleared. Refresh the page.")

st.info(f"Current chunks in ChromaDB: {collection.count()}")

docs_ready = collection.count() > 0

if not docs_ready:
    st.warning("Upload PDFs and click Process Documents first.")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(
    ["Ask Questions", "Summaries", "Comparison Table", "Timeline"]
)

with tab1:
    question = st.text_input("Ask a question about the uploaded documents")

    if st.button("Ask", disabled=not docs_ready):
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            with st.spinner("Thinking..."):
                context = retrieve_context(question, top_k=8)

                prompt = f"""
Context:
{context}

Question:
{question}

Answer clearly with source filename and chunk number.
"""

                answer = ask_llm(prompt)
                st.markdown(answer)

with tab2:
    if st.button("Generate Summaries", disabled=not docs_ready):
        with st.spinner("Generating summaries..."):
            context = retrieve_context(
                "main facts arguments findings conclusions summary",
                top_k=14
            )

            prompt = f"""
Create a summary for each document.

Context:
{context}

Format:

## Document Name
- Main topic
- Key facts
- Main arguments
- Conclusion
- Source references
"""

            answer = ask_llm(prompt)
            st.markdown(answer)

with tab3:
    if st.button("Generate Comparison Table", disabled=not docs_ready):
        with st.spinner("Generating comparison table..."):
            context = retrieve_context(
                "compare documents issues facts arguments conclusions differences similarities",
                top_k=14
            )

            prompt = f"""
Compare the documents using this context:

{context}

Create a markdown table with columns:

Document | Topic | Key Issue | Main Argument | Evidence | Conclusion | Similarities | Differences
"""

            answer = ask_llm(prompt)
            st.markdown(answer)

with tab4:
    if st.button("Generate Timeline", disabled=not docs_ready):
        with st.spinner("Generating timeline..."):
            context = retrieve_context(
                "timeline chronology dates events sequence history",
                top_k=14
            )

            prompt = f"""
Create a timeline from the documents.

Context:
{context}

Use this format:

| Time / Date | Event | Document Source |
"""

            answer = ask_llm(prompt)
            st.markdown(answer)