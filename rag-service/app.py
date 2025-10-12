"""
FastAPI web service for the RAG system.
This provides HTTP endpoints so other applications can use our service.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import time

# Import our custom modules
from rag_service.crawler import WebCrawler
from rag_service.indexer import TextIndexer
from rag_service.retriever import GroundedQA
from rag_service.config import (
    DEFAULT_CRAWL_DELAY_MS, DEFAULT_MAX_PAGES, DEFAULT_MAX_DEPTH,
    DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, DEFAULT_EMBEDDING_MODEL,
    DEFAULT_TOP_K, API_HOST, API_PORT
)
from rag_service.utils import logger

# Create FastAPI app with metadata
app = FastAPI(
    title="RAG Service API",
    description="Retrieval-Augmented Generation service for web content",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI will be at /docs
    redoc_url="/redoc"  # ReDoc will be at /redoc
)

# Global variables to hold our system components
indexer = None
qa_system = None

# Define request/response models using Pydantic
# This ensures type safety and automatic API documentation

class CrawlRequest(BaseModel):
    start_url: str = Field(..., description="Starting URL to crawl from")
    max_pages: int = Field(DEFAULT_MAX_PAGES, description="Maximum number of pages to crawl")
    max_depth: int = Field(DEFAULT_MAX_DEPTH, description="Maximum crawl depth")
    crawl_delay_ms: int = Field(DEFAULT_CRAWL_DELAY_MS, description="Delay between requests in milliseconds")

class CrawlResponse(BaseModel):
    page_count: int
    skipped_count: int
    urls: List[str]
    errors: List[str]

class IndexRequest(BaseModel):
    chunk_size: int = Field(DEFAULT_CHUNK_SIZE, description="Size of text chunks in characters")
    chunk_overlap: int = Field(DEFAULT_CHUNK_OVERLAP, description="Overlap between chunks in characters")
    embedding_model: str = Field(DEFAULT_EMBEDDING_MODEL, description="Name of the embedding model")

class IndexResponse(BaseModel):
    vector_count: int
    errors: List[str]

class AskRequest(BaseModel):
    question: str = Field(..., description="Question to answer")
    top_k: int = Field(DEFAULT_TOP_K, description="Number of chunks to retrieve")

class SourceInfo(BaseModel):
    url: str
    snippet: str
    similarity_score: float

class TimingInfo(BaseModel):
    retrieval_ms: float
    generation_ms: float
    total_ms: float

class AskResponse(BaseModel):
    answer: str
    sources: List[SourceInfo]
    timings: TimingInfo
    is_refusal: bool = False

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "RAG Service API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "crawl": "POST /crawl",
            "index": "POST /index", 
            "ask": "POST /ask",
            "status": "GET /status"
        }
    }

@app.get("/status")
async def get_status():
    """Get service status and configuration."""
    global indexer, qa_system
    
    status = {
        "service": "running",
        "indexer_loaded": indexer is not None,
        "qa_system_ready": qa_system is not None,
        "timestamp": time.time()
    }
    
    if indexer:
        status["index_info"] = {
            "vector_count": len(indexer.document_chunks) if indexer.document_chunks else 0,
            "embedding_model": indexer.embedding_model_name,
            "chunk_size": indexer.chunk_size,
            "chunk_overlap": indexer.chunk_overlap
        }
    
    return status

@app.post("/crawl", response_model=CrawlResponse)
async def crawl_website(request: CrawlRequest):
    """
    Crawl a website starting from the given URL.
    This is the first step - getting the raw data.
    """
    logger.info("Crawl request received", 
               start_url=request.start_url,
               max_pages=request.max_pages)
    
    try:
        # Create crawler with user settings
        crawler = WebCrawler(
            crawl_delay_ms=request.crawl_delay_ms,
            max_pages=request.max_pages,
            max_depth=request.max_depth
        )
        
        # Do the actual crawling
        result = crawler.crawl(request.start_url)
        
        return CrawlResponse(
            page_count=result.page_count,
            skipped_count=result.skipped_count,
            urls=result.urls,
            errors=result.errors
        )
        
    except Exception as e:
        logger.error("Crawl error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Crawl failed: {str(e)}")

@app.post("/index", response_model=IndexResponse)
async def index_documents(request: IndexRequest):
    """
    Index crawled documents for search.
    This is the second step - making data searchable.
    """
    global indexer, qa_system
    
    logger.info("Index request received",
               chunk_size=request.chunk_size,
               embedding_model=request.embedding_model)
    
    try:
        # Create new indexer with user settings
        indexer = TextIndexer(
            embedding_model=request.embedding_model,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap
        )
        
        # Do the actual indexing
        result = indexer.index_documents()
        
        # Initialize QA system if indexing was successful
        if result.vector_count > 0:
            qa_system = GroundedQA(indexer)
            logger.info("QA system initialized")
        else:
            qa_system = None
            logger.warning("QA system not initialized - no vectors created")
        
        return IndexResponse(
            vector_count=result.vector_count,
            errors=result.errors
        )
        
    except Exception as e:
        logger.error("Index error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")

@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """
    Answer a question using the indexed content.
    This is the final step - getting answers from our data.
    """
    global qa_system, indexer
    
    # Try to load existing index if we don't have one
    if not qa_system:
        if not indexer:
            indexer = TextIndexer()
        
        if not indexer.load_existing_index():
            raise HTTPException(
                status_code=400, 
                detail="No index available. Please crawl and index documents first."
            )
        
        qa_system = GroundedQA(indexer)
        logger.info("Loaded existing index for QA")
    
    logger.info("Question received", question=request.question[:100])
    
    try:
        # Get the answer from our QA system
        response = qa_system.ask(request.question, request.top_k)
        
        # Convert to API response format
        return AskResponse(
            answer=response.answer,
            sources=[
                SourceInfo(
                    url=source.url,
                    snippet=source.snippet,
                    similarity_score=source.similarity_score
                ) for source in response.sources
            ],
            timings=TimingInfo(
                retrieval_ms=response.timings.retrieval_ms,
                generation_ms=response.timings.generation_ms,
                total_ms=response.timings.total_ms
            ),
            is_refusal=response.is_refusal
        )
        
    except Exception as e:
        logger.error("QA error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Question answering failed: {str(e)}")

# Add CORS middleware for development (allows web browsers to access our API)
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Run the server
if __name__ == "__main__":
    import uvicorn
    
    print(f"🚀 Starting RAG Service API on {API_HOST}:{API_PORT}")
    print(f"📖 API docs will be available at: http://{API_HOST}:{API_PORT}/docs")
    
    uvicorn.run(app, host=API_HOST, port=API_PORT)