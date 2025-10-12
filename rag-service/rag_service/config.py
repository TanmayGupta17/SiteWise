"""
Configuration settings for RAG service.
This file centralizes all settings and makes the system configurable.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base paths - where we store data
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
CRAWLED_DIR = DATA_DIR / "crawled"  # Raw crawled content
VECTORS_DIR = DATA_DIR / "vectors"  # Vector embeddings

# Create directories if they don't exist
for dir_path in [DATA_DIR, CRAWLED_DIR, VECTORS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Crawler settings - how we scrape websites
DEFAULT_CRAWL_DELAY_MS = 1000  # Wait 1 second between requests (be polite!)
DEFAULT_MAX_PAGES = 50         # Don't crawl too many pages
DEFAULT_MAX_DEPTH = 3          # Don't go too deep into site structure
USER_AGENT = "RAG-Service-Crawler/1.0 (+research-purpose)"  # Identify ourselves

# Text processing settings - how we chunk and embed text
DEFAULT_CHUNK_SIZE = 800       # Characters per chunk (good for context)
DEFAULT_CHUNK_OVERLAP = 100    # Overlap between chunks (prevents splitting sentences)
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast, good quality model

# Search settings - how we find relevant content
DEFAULT_TOP_K = 5              # How many chunks to retrieve per question
SIMILARITY_THRESHOLD = 0.7     # Minimum similarity score to consider relevant

# API settings
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))

# Safety settings - prevent bad behavior
MAX_ANSWER_LENGTH = 1000
REFUSAL_PHRASES = [
    "I don't have enough information",
    "The crawled content doesn't contain",
    "Not found in crawled content"
]