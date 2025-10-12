"""
Utility functions for RAG service.
These are helper functions used throughout the project.
"""
import re
import time
import json
import hashlib
from pathlib import Path
from urllib.parse import urlparse, urljoin
from typing import List, Dict, Any
import structlog

# Set up structured logging (better than print statements)
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
)

logger = structlog.get_logger()

def get_domain(url: str) -> str:
    """Extract the main domain from a URL."""
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    
    # Remove port numbers
    if ':' in domain:
        domain = domain.split(':')[0]
    
    # Remove www prefix
    if domain.startswith('www.'):
        domain = domain[4:]
    
    return domain

def is_same_domain(url1: str, url2: str) -> bool:
    """Check if two URLs are from the same domain."""
    return get_domain(url1) == get_domain(url2)

from typing import Optional

def normalize_url(url: str, base_url: Optional[str] = None) -> str:
    """Convert relative URLs to absolute and clean them."""
    if base_url:
        url = urljoin(base_url, url)
    
    parsed = urlparse(url)
    # Remove fragments (#section) but keep query parameters (?param=value)
    normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    if parsed.query:
        normalized += f"?{parsed.query}"
    
    return normalized

def clean_text(text: str) -> str:
    """Clean up text content by removing extra whitespace and boilerplate."""
    if not text:
        return ""
    
    # Remove multiple whitespaces and newlines
    text = re.sub(r'\s+', ' ', text)
    
    # Remove common boilerplate text patterns
    patterns_to_remove = [
        r'Cookie\s+Policy.*?Accept',
        r'Privacy\s+Policy.*?Terms',
        r'Subscribe\s+to.*?Newsletter',
        r'^(Home|About|Contact|Menu|Navigation)(\s|$)'
    ]
    
    for pattern in patterns_to_remove:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
    
    return text.strip()

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    """
    Split text into overlapping chunks.
    This is crucial for good retrieval - too big chunks lose precision,
    too small chunks lose context.
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at word boundaries (don't split words)
        if end < len(text):
            last_space = text.rfind(' ', start, end)
            if last_space > start:
                end = last_space
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start position with overlap
        start = end - overlap
        if start >= len(text):
            break
    
    return chunks

def url_to_filename(url: str) -> str:
    """Convert URL to a safe filename for storage."""
    # Create unique hash of URL
    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    
    # Extract meaningful parts from URL path
    parsed = urlparse(url)
    path_parts = [p for p in parsed.path.split('/') if p and p != 'index.html']
    
    if path_parts:
        filename = '_'.join(path_parts[-2:])  # Use last 2 path components
    else:
        filename = parsed.netloc.replace('.', '_')
    
    # Make filename safe for filesystem
    filename = re.sub(r'[^\w\-_]', '_', filename)
    filename = re.sub(r'_+', '_', filename).strip('_')
    
    return f"{filename}_{url_hash}.json"

def save_json(data: Any, filepath: Path) -> None:
    """Save data as JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_json(filepath: Path) -> Any:
    """Load data from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def measure_time(func):
    """Decorator to measure how long functions take to run."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_ms = (end_time - start_time) * 1000
        
        logger.info(f"{func.__name__} completed", duration_ms=elapsed_ms)
        return result, elapsed_ms
    
    return wrapper