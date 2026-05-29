

# 📄 PDFLens AI

PDFLens AI is a local-first RAG (Retrieval-Augmented Generation) application for analyzing and comparing PDFs such as legal case laws, research papers,and policy documents.

The project uses semantic search, vector embeddings, and local LLM inference to generate grounded responses directly from uploaded documents.

Built using:
- Streamlit
- ChromaDB
- SentenceTransformers
- Ollama
- Llama 3.2
- Mistral

---

## 🚀 Features

- 📄 Upload multiple PDFs (up to 200 pages each)
- 💬 Ask questions about uploaded documents
- 📝 Generate document-wise summaries
- 📊 Create structured comparison tables
- 📅 Extract timelines and chronological events
- 🔍 Semantic search using vector embeddings
- ⚡ Fully local AI inference with Ollama
- 💾 Persistent ChromaDB vector storage
- 🧵 Streaming LLM responses
- 📂 Document-level retrieval filtering
- 🏗️ Modular architecture (`config.py`, `rag.py`, `llm.py`, `app.py`)

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
- grounding
- retrieval accuracy
- scalability
- hallucination reduction

---

## 🏗️ Project Architecture

```text
config.py  → settings & performance tuning
rag.py     → chunking, embeddings, ChromaDB retrieval
llm.py     → Ollama interaction & prompt builders
app.py     → Streamlit UI
```

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
| LLM | llama3.2 |
| Language | Python |

---

## 🤖 Why Llama 3.2?

Llama 3.2 was chosen because it:
- runs locally
- is lightweight and RAM-efficient
- works well on MacBook Air hardware
- provides fast inference
- performs well for summarization and Q&A tasks

---

## ⚡ Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd pdflens-ai
```

---

### 2. Create virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

---

### 3. Install dependencies

```bash
pip install streamlit chromadb sentence-transformers pymupdf ollama torch torchvision
```

---

### 4. Pull Llama 3.2 model

```bash
ollama pull llama3.2
```

---

## ▶️ Run the App

Recommended optimized command for low-memory systems:

```bash
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 streamlit run app.py --server.fileWatcherType none
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
- Streaming LLM Generation

---

## 📂 Example Use Cases

- Legal document comparison
- Human-rights case analysis
- Research paper summarization
- Policy document exploration

---