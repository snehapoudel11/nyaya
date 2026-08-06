import os
from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data", "raw")
CHROMA_DIR = os.path.join(ROOT_DIR, "chroma_db")
COLLECTION_NAME = "nyaya_docs"

# --- Embedding model (served locally via Ollama) ---
# Pull once with: ollama pull embeddinggemma:300m
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "embeddinggemma:300m")

# --- Groq LLM (cloud, does the actual generation) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# --- Chunking ---
CHUNK_SIZE = 800       # characters per chunk
CHUNK_OVERLAP = 150    # characters of overlap between chunks

# --- Retrieval ---
TOP_K = 4               # number of chunks to retrieve per query
