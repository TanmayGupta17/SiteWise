"""
Text indexing system that converts text to searchable vectors.
This is the second step of our RAG pipeline - making data searchable.
"""
import numpy as np
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Import the ML libraries
from sentence_transformers import SentenceTransformer
import faiss

from .config import (
    DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, DEFAULT_EMBEDDING_MODEL,
    CRAWLED_DIR, VECTORS_DIR
)
from .utils import chunk_text, load_json, save_json

@dataclass
class IndexResult:
    """What we get back after indexing."""
    vector_count: int
    errors: List[str]

@dataclass
class DocumentChunk:
    """A piece of text with its metadata."""
    chunk_id: str          # Unique identifier
    url: str              # Source webpage
    title: str            # Page title
    content: str          # The actual text chunk
    chunk_index: int      # Which chunk number this is from the page
    start_char: int       # Where in the original text this chunk starts
    end_char: int         # Where it ends

class TextIndexer:
    """
    The indexer takes all our crawled text and converts it into
    a searchable format using AI embeddings.
    """
    
    def __init__(self, 
                 embedding_model: str = DEFAULT_EMBEDDING_MODEL,
                 chunk_size: int = DEFAULT_CHUNK_SIZE,
                 chunk_overlap: int = DEFAULT_CHUNK_OVERLAP):
        
        self.embedding_model_name = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.errors: List[str] = []
        
        print(f"[INFO] Loading embedding model: {embedding_model}")
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            print(f"[INFO] Model loaded (embedding_dim={self.embedding_dim})")
        except Exception as e:
            print(f"[ERROR] Failed to load embedding model: {e}")
            raise
        
        self.faiss_index = None
        self.document_chunks: List[DocumentChunk] = []
        
    def _load_crawled_documents(self) -> List[Dict[str, Any]]:
        """Load all the documents we crawled earlier."""
        documents = []
        
        if not CRAWLED_DIR.exists():
            print(f"[WARNING] No crawled data found at {CRAWLED_DIR}")
            return documents
        
        for file_path in CRAWLED_DIR.glob("*.json"):
            if file_path.name == "crawl_summary.json":
                continue
            
            try:
                doc_data = load_json(file_path)
                documents.append(doc_data)
            except Exception as e:
                self.errors.append(f"Error loading {file_path}: {str(e)}")
        
        print(f"[INFO] Loaded {len(documents)} documents")
        return documents
    
    def _create_chunks(self, documents: List[Dict[str, Any]]) -> List[DocumentChunk]:
        """Break documents into smaller chunks for better search."""
        all_chunks = []
        
        for doc in documents:
            url = doc['url']
            title = doc.get('title', '')
            content = doc.get('content', '')
            
            if not content or len(content.strip()) < 50:
                self.errors.append(f"Skipping document with no content: {url}")
                continue
            
            text_chunks = chunk_text(content, self.chunk_size, self.chunk_overlap)
            start_char = 0
            for i, chunk in enumerate(text_chunks):
                chunk_id = f"{url}#{i}"
                end_char = start_char + len(chunk)
                
                doc_chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    url=url,
                    title=title,
                    content=chunk,
                    chunk_index=i,
                    start_char=start_char,
                    end_char=end_char
                )
                
                all_chunks.append(doc_chunk)
                start_char = end_char - self.chunk_overlap
        
        avg_len = np.mean([len(c.content) for c in all_chunks]) if all_chunks else 0
        print(f"[INFO] Created {len(all_chunks)} chunks (avg length: {avg_len:.1f})")
        return all_chunks
    
    def _create_embeddings(self, chunks: List[DocumentChunk]) -> np.ndarray:
        """Convert text chunks into vector embeddings."""
        print(f"[INFO] Creating embeddings for {len(chunks)} chunks...")
        texts = [chunk.content for chunk in chunks]
        
        try:
            embeddings = self.embedding_model.encode(
                texts, 
                batch_size=32,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            print(f"[INFO] Embeddings created (shape={embeddings.shape}, dim={self.embedding_dim})")
            return embeddings
            
        except Exception as e:
            print(f"[ERROR] Error creating embeddings: {e}")
            raise
    
    def _build_faiss_index(self, embeddings: np.ndarray) -> faiss.IndexFlatIP:
        """Build a searchable index from the embeddings."""
        print(f"[INFO] Building FAISS index for {embeddings.shape[0]} vectors...")
        index = faiss.IndexFlatIP(self.embedding_dim)
        index.add(embeddings.astype(np.float32))
        print(f"[INFO] FAISS index built (total_vectors={index.ntotal})")
        return index
    
    def _save_index(self, embeddings: np.ndarray) -> None:
        """Save everything to disk so we can load it later."""
        faiss_path = VECTORS_DIR / "faiss_index.bin"
        embeddings_path = VECTORS_DIR / "embeddings.npy"
        chunks_path = VECTORS_DIR / "chunks_metadata.json"
        config_path = VECTORS_DIR / "index_config.json"
        
        faiss.write_index(self.faiss_index, str(faiss_path))
        np.save(embeddings_path, embeddings)
        
        chunks_data = []
        for chunk in self.document_chunks:
            chunks_data.append({
                'chunk_id': chunk.chunk_id,
                'url': chunk.url,
                'title': chunk.title,
                'content': chunk.content,
                'chunk_index': chunk.chunk_index,
                'start_char': chunk.start_char,
                'end_char': chunk.end_char
            })
        
        save_json(chunks_data, chunks_path)
        
        config_data = {
            'embedding_model': self.embedding_model_name,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'embedding_dim': self.embedding_dim,
            'total_chunks': len(self.document_chunks),
            'timestamp': time.time()
        }
        save_json(config_data, config_path)
        
        print(f"[INFO] Index saved to {VECTORS_DIR}")
        print(f"        • Chunks: {len(self.document_chunks)}")
        print(f"        • FAISS: {faiss_path}")
    
    def index_documents(self) -> IndexResult:
        """Main method that does the entire indexing process."""
        print("[INFO] Starting document indexing...")
        start_time = time.time()
        
        self.errors = []
        self.document_chunks = []
        
        try:
            documents = self._load_crawled_documents()
            if not documents:
                return IndexResult(vector_count=0, errors=["No documents found"])
            
            self.document_chunks = self._create_chunks(documents)
            if not self.document_chunks:
                return IndexResult(vector_count=0, errors=["No chunks created"])
            
            embeddings = self._create_embeddings(self.document_chunks)
            self.faiss_index = self._build_faiss_index(embeddings)
            self._save_index(embeddings)
            
            elapsed = time.time() - start_time
            print(f"[INFO] Indexing completed successfully in {elapsed:.2f}s ({len(self.document_chunks)} vectors)")
            
            return IndexResult(
                vector_count=len(self.document_chunks),
                errors=self.errors
            )
            
        except Exception as e:
            error_msg = f"Indexing failed: {str(e)}"
            print(f"[ERROR] {error_msg}")
            self.errors.append(error_msg)
            return IndexResult(vector_count=0, errors=self.errors)
    
    def load_existing_index(self) -> bool:
        """Load a previously created index from disk."""
        try:
            faiss_path = VECTORS_DIR / "faiss_index.bin"
            chunks_path = VECTORS_DIR / "chunks_metadata.json"
            config_path = VECTORS_DIR / "index_config.json"
            
            if not all(p.exists() for p in [faiss_path, chunks_path, config_path]):
                print("[WARNING] Index files not found.")
                return False
            
            self.faiss_index = faiss.read_index(str(faiss_path))
            chunks_data = load_json(chunks_path)
            self.document_chunks = [
                DocumentChunk(
                    chunk_id=c['chunk_id'],
                    url=c['url'],
                    title=c['title'],
                    content=c['content'],
                    chunk_index=c['chunk_index'],
                    start_char=c['start_char'],
                    end_char=c['end_char']
                )
                for c in chunks_data
            ]
            
            config_data = load_json(config_path)
            self.embedding_model_name = config_data['embedding_model']
            self.chunk_size = config_data['chunk_size']
            self.chunk_overlap = config_data['chunk_overlap']
            self.embedding_dim = config_data['embedding_dim']
            
            print(f"[INFO] Loaded existing index ({len(self.document_chunks)} chunks, model={self.embedding_model_name})")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to load existing index: {e}")
            return False
