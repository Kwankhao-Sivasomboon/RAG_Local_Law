import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATASETS_DIR = os.path.join(PROJECT_ROOT, "datasets")
DB_DIR = os.path.join(PROJECT_ROOT, "chroma_db")

# Files
CORE_LAW_FILE = os.path.join(DATASETS_DIR, "thailaw-v1.0.csv")
RECENT_LAW_DIR = os.path.join(DATASETS_DIR, "iapp_2025")

# Models
LLM_MODEL_NAME = "llama3.2"  # Ollama model name
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Vector DB
COLLECTION_CORE = "core_law"
COLLECTION_RECENT = "recent_law"

# Retrieval
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVAL_K = 5
