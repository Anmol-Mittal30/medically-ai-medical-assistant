import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

# Gemini configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
GEMINI_BASE_URL = os.getenv(
    "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Knowledge base configuration
DATA_PATH = BASE_DIR / os.getenv("DATA_PATH", "data")

FAISS_INDEX_PATH = BASE_DIR / os.getenv("FAISS_INDEX_PATH", "vectorstore/db_faiss")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Document processing
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))

# Retrieval configuration
TOP_K = int(os.getenv("TOP_K", "3"))

# LLM generation parameters
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))

MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "1024"))

REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "60"))
