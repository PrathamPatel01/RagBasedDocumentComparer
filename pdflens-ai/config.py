import os
 
# ── Kill OOM crashes on MacBook Air before anything else loads ────────────────
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
 
# ── Model settings ────────────────────────────────────────────────────────────
MODEL_NAME = "llama3.2"          # swap to "llama3.2" etc. here only
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "documents"
 
# ── Document limits ───────────────────────────────────────────────────────────
MAX_PAGES = 200
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
EMBED_BATCH_SIZE = 16   


