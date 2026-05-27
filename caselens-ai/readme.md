# ⚖️ CaseLens AI

CaseLens AI is a local-first RAG (Retrieval-Augmented Generation) application for analyzing and comparing PDFs such as legal case laws, research papers, resumes, and policy documents.

Built using:
- Streamlit
- ChromaDB
- SentenceTransformers
- Ollama
- Mistral

---

## 🚀 Features

- 📄 Upload multiple PDFs (up to 200 pages each)
- 💬 Ask questions about documents
- 📝 Generate document-wise summaries
- 📊 Create comparison tables
- 📅 Extract timelines and chronological events
- 🔍 Semantic search using vector embeddings
- ⚡ Fully local AI inference
- 💾 Persistent ChromaDB vector storage
- 🧵 Streaming LLM responses

---

## 🧠 What is RAG?

RAG = Retrieval-Augmented Generation

Instead of sending entire PDFs directly to the LLM:

```text
PDFs
↓
Chunking
↓
Embeddings
↓
Vector DB (ChromaDB)
↓
Semantic Retrieval
↓
LLM Response
```

The system retrieves only the most relevant chunks before generating answers.

This improves:
- accuracy
- grounding
- scalability
- hallucination reduction

---

## 🏗️ Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| PDF Parsing | PyMuPDF |
| Embeddings | SentenceTransformers |
| Embedding Model | all-MiniLM-L6-v2 |
| Vector DB | ChromaDB |
| LLM Runtime | Ollama |
| LLM | Mistral |
| Language | Python |

---

## 🤖 Why Mistral?

Mistral was chosen because it:
- runs locally
- provides strong reasoning
- is lightweight and fast
- performs well for summarization and document analysis

---

## ⚡ Installation

### 1. Install dependencies

```bash
pip install streamlit chromadb sentence-transformers pymupdf ollama
```

### 2. Pull Mistral model

```bash
ollama pull mistral
```

---

## ▶️ Run the App

```bash
streamlit run app.py --server.fileWatcherType none
```

Open:

```text
http://localhost:8501
```

---

## 📌 AI Concepts Demonstrated

- Retrieval-Augmented Generation (RAG)
- Semantic Search
- Dense Vector Embeddings
- Chunking & Overlap Strategy
- Metadata-filtered Retrieval
- Hierarchical Summarization
- Grounded AI Responses
- Local-first AI Systems

---

## 📂 Example Use Cases

- Legal document comparison
- Research paper summarization
- Policy document exploration

---