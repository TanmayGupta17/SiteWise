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
from .utils import chunk_text, load_json, save_json, logger

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
        
        # Load the AI model for creating embeddings
        logger.info("Loading embedding model", model=embedding_model)
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            logger.info("Model loaded", embedding_dimension=self.embedding_dim)
        except Exception as e:
            logger.error("Failed to load embedding model", error=str(e))
            raise
        
        # These will hold our processed data
        self.faiss_index = None
        self.document_chunks: List[DocumentChunk] = []
        
    def _load_crawled_documents(self) -> List[Dict[str, Any]]:
        """Load all the documents we crawled earlier."""
        documents = []
        
        if not CRAWLED_DIR.exists():
            logger.warning("No crawled data found", path=str(CRAWLED_DIR))
            return documents
        
        # Load each JSON file
        for file_path in CRAWLED_DIR.glob("*.json"):
            if file_path.name == "crawl_summary.json":
                continue  # Skip the summary file
            
            try:
                doc_data = load_json(file_path)
                documents.append(doc_data)
            except Exception as e:
                self.errors.append(f"Error loading {file_path}: {str(e)}")
        
        logger.info("Loaded documents", count=len(documents))
        return documents
    
    def _create_chunks(self, documents: List[Dict[str, Any]]) -> List[DocumentChunk]:
        """Break documents into smaller chunks for better search."""
        all_chunks = []
        
        for doc in documents:
            url = doc['url']
            title = doc.get('title', '')
            content = doc.get('content', '')
            
            # Skip documents with no content
            if not content or len(content.strip()) < 50:
                self.errors.append(f"Skipping document with no content: {url}")
                continue
            
            # Split the content into chunks
            text_chunks = chunk_text(content, self.chunk_size, self.chunk_overlap)
            
            # Create DocumentChunk objects
            start_char = 0
            for i, chunk in enumerate(text_chunks):
                chunk_id = f"{url}#{i}"  # Unique ID for this chunk
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
        
        logger.info("Created chunks", 
                   total_chunks=len(all_chunks),
                   avg_length=np.mean([len(c.content) for c in all_chunks]) if all_chunks else 0)
        
        return all_chunks
    
    def _create_embeddings(self, chunks: List[DocumentChunk]) -> np.ndarray:
        """Convert text chunks into vector embeddings."""
        logger.info("Creating embeddings", chunk_count=len(chunks))
        
        # Extract just the text content
        texts = [chunk.content for chunk in chunks]
        
        try:
            # Use the AI model to create embeddings
            # This is where the magic happens - text becomes numbers!
            embeddings = self.embedding_model.encode(
                texts, 
                batch_size=32,              # Process 32 at a time
                show_progress_bar=True,     # Show progress
                convert_to_numpy=True,      # Return numpy arrays
                normalize_embeddings=True   # Normalize for better similarity search
            )
            
            logger.info("Embeddings created", 
                       shape=embeddings.shape,
                       embedding_dim=self.embedding_dim)
            
            return embeddings
            
        except Exception as e:
            logger.error("Error creating embeddings", error=str(e))
            raise
    
    def _build_faiss_index(self, embeddings: np.ndarray) -> faiss.IndexFlatIP:
        """Build a searchable index from the embeddings."""
        logger.info("Building FAISS index", vectors=embeddings.shape[0])
        
        # Create a FAISS index for similarity search
        # We use Inner Product (IP) because our embeddings are normalized
        # (equivalent to cosine similarity but faster)
        index = faiss.IndexFlatIP(self.embedding_dim)
        index.add(embeddings.astype(np.float32))
        
        logger.info("FAISS index built", 
                   total_vectors=index.ntotal,
                   index_type="FlatIP")
        
        return index
    
    def _save_index(self, embeddings: np.ndarray) -> None:
        """Save everything to disk so we can load it later."""
        # Save the FAISS index
        faiss_path = VECTORS_DIR / "faiss_index.bin"
        faiss.write_index(self.faiss_index, str(faiss_path))
        
        # Save embeddings as backup
        embeddings_path = VECTORS_DIR / "embeddings.npy"
        np.save(embeddings_path, embeddings)
        
        # Save chunk metadata
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
        
        chunks_path = VECTORS_DIR / "chunks_metadata.json"
        save_json(chunks_data, chunks_path)
        
        # Save configuration
        config_data = {
            'embedding_model': self.embedding_model_name,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'embedding_dim': self.embedding_dim,
            'total_chunks': len(self.document_chunks),
            'timestamp': time.time()
        }
        
        config_path = VECTORS_DIR / "index_config.json"
        save_json(config_data, config_path)
        
        logger.info("Index saved", 
                   faiss_path=str(faiss_path),
                   chunks_count=len(self.document_chunks))
    
    def index_documents(self) -> IndexResult:
        """
        Main method that does the entire indexing process.
        """
        logger.info("Starting document indexing")
        start_time = time.time()
        
        # Reset state
        self.errors = []
        self.document_chunks = []
        
        try:
            # Step 1: Load crawled documents
            documents = self._load_crawled_documents()
            if not documents:
                return IndexResult(vector_count=0, errors=["No documents found"])
            
            # Step 2: Create chunks
            self.document_chunks = self._create_chunks(documents)
            if not self.document_chunks:
                return IndexResult(vector_count=0, errors=["No chunks created"])
            
            # Step 3: Create embeddings
            embeddings = self._create_embeddings(self.document_chunks)
            
            # Step 4: Build search index
            self.faiss_index = self._build_faiss_index(embeddings)
            
            # Step 5: Save everything
            self._save_index(embeddings)
            
            elapsed_time = time.time() - start_time
            logger.info("Indexing completed",
                       vector_count=len(self.document_chunks),
                       elapsed_seconds=elapsed_time)
            
            return IndexResult(
                vector_count=len(self.document_chunks),
                errors=self.errors
            )
            
        except Exception as e:
            error_msg = f"Indexing failed: {str(e)}"
            logger.error(error_msg)
            self.errors.append(error_msg)
            return IndexResult(vector_count=0, errors=self.errors)
    
    def load_existing_index(self) -> bool:
        """Load a previously created index from disk."""
        try:
            # Check if all required files exist
            faiss_path = VECTORS_DIR / "faiss_index.bin"
            chunks_path = VECTORS_DIR / "chunks_metadata.json"
            config_path = VECTORS_DIR / "index_config.json"
            
            if not all(p.exists() for p in [faiss_path, chunks_path, config_path]):
                logger.warning("Index files not found")
                return False
            
            # Load FAISS index
            self.faiss_index = faiss.read_index(str(faiss_path))
            
            # Load chunks metadata
            chunks_data = load_json(chunks_path)
            self.document_chunks = []
            
            for chunk_data in chunks_data:
                chunk = DocumentChunk(
                    chunk_id=chunk_data['chunk_id'],
                    url=chunk_data['url'],
                    title=chunk_data['title'],
                    content=chunk_data['content'],
                    chunk_index=chunk_data['chunk_index'],
                    start_char=chunk_data['start_char'],
                    end_char=chunk_data['end_char']
                )
                self.document_chunks.append(chunk)
            
            # Load config
            config_data = load_json(config_path)
            self.embedding_model_name = config_data['embedding_model']
            self.chunk_size = config_data['chunk_size']
            self.chunk_overlap = config_data['chunk_overlap']
            self.embedding_dim = config_data['embedding_dim']
            
            logger.info("Loaded existing index",
                       vector_count=len(self.document_chunks),
                       model=self.embedding_model_name)
            
            return True
            
        except Exception as e:
            logger.error("Failed to load existing index", error=str(e))
            return False