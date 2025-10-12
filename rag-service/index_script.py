#!/usr/bin/env python3
"""
Command-line script for indexing crawled documents.
Usage: python index_script.py --chunk-size 800
"""
import click
from rag_service.indexer import TextIndexer
from rag_service.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, DEFAULT_EMBEDDING_MODEL

@click.command()
@click.option('--chunk-size', default=DEFAULT_CHUNK_SIZE, help='Size of text chunks in characters')
@click.option('--chunk-overlap', default=DEFAULT_CHUNK_OVERLAP, help='Overlap between chunks in characters')
@click.option('--embedding-model', default=DEFAULT_EMBEDDING_MODEL, help='Name of the embedding model')
def index(chunk_size, chunk_overlap, embedding_model):
    """Index crawled documents for search."""
    click.echo("🔍 Starting document indexing")
    click.echo(f"   Chunk size: {chunk_size} characters")
    click.echo(f"   Chunk overlap: {chunk_overlap} characters")
    click.echo(f"   Embedding model: {embedding_model}")
    click.echo()
    
    try:
        # Create indexer
        indexer = TextIndexer(
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        click.echo("📚 Loading and processing documents...")
        result = indexer.index_documents()
        
        click.echo()
        click.echo(f"✅ Indexing completed!")
        click.echo(f"   Vectors created: {result.vector_count}")
        click.echo(f"   Errors: {len(result.errors)}")
        
        if result.errors:
            click.echo("\n❌ Errors encountered:")
            for error in result.errors[:5]:
                click.echo(f"   • {error}")
            if len(result.errors) > 5:
                click.echo(f"   ... and {len(result.errors) - 5} more")
        
        click.echo(f"\n📁 Index saved to: data/vectors/")
        
    except Exception as e:
        click.echo(f"❌ Indexing failed: {e}", err=True)
        return 1
    
    return 0

if __name__ == '__main__':
    index()