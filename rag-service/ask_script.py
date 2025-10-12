#!/usr/bin/env python3
"""
Command-line script for asking questions.
Usage: python ask_script.py "What is this website about?"
"""
import click
import json
from rag_service.indexer import TextIndexer
from rag_service.retriever import GroundedQA
from rag_service.config import DEFAULT_TOP_K

@click.command()
@click.argument('question')
@click.option('--top-k', default=DEFAULT_TOP_K, help='Number of chunks to retrieve')
@click.option('--interactive', is_flag=True, help='Enter interactive mode')
def ask(question, top_k, interactive):
    """Ask a question using the indexed content."""
    
    # Load the system
    try:
        click.echo("🤔 Loading index...")
        indexer = TextIndexer()
        
        if not indexer.load_existing_index():
            click.echo("❌ No index found. Please crawl and index documents first.", err=True)
            click.echo("\nTo get started:")
            click.echo("1. python crawl_script.py https://example.com --max-pages 20")
            click.echo("2. python index_script.py")
            click.echo("3. python ask_script.py 'Your question here'")
            return 1
        
        qa_system = GroundedQA(indexer)
        click.echo(f"✅ Loaded index with {len(indexer.document_chunks)} chunks")
        click.echo()
        
    except Exception as e:
        click.echo(f"❌ Failed to load index: {e}", err=True)
        return 1
    
    def process_question(q):
        """Process a single question."""
        click.echo(f"❓ Question: {q}")
        click.echo("🔍 Searching...")
        
        try:
            response = qa_system.ask(q, top_k)
            
            click.echo()
            click.echo("📝 Answer:")
            click.echo(f"   {response.answer}")
            
            click.echo()
            if response.is_refusal:
                click.echo("⚠️  This is a refusal response (insufficient information)")
            else:
                click.echo("✅ Answer based on retrieved content")
            
            click.echo()
            click.echo(f"📊 Performance:")
            click.echo(f"   Retrieval: {response.timings.retrieval_ms:.1f}ms")
            click.echo(f"   Generation: {response.timings.generation_ms:.1f}ms")
            click.echo(f"   Total: {response.timings.total_ms:.1f}ms")
            
            if response.sources:
                click.echo()
                click.echo(f"🔗 Sources ({len(response.sources)}):")
                for i, source in enumerate(response.sources, 1):
                    click.echo(f"   {i}. {source.url}")
                    click.echo(f"      Similarity: {source.similarity_score:.3f}")
                    click.echo(f"      Snippet: {source.snippet}")
                    if i < len(response.sources):
                        click.echo()
            
            return response
            
        except Exception as e:
            click.echo(f"❌ Question processing failed: {e}", err=True)
            return None
    
    if interactive:
        click.echo("🚀 Interactive mode - type 'quit' to exit")
        click.echo()
        
        while True:
            try:
                q = click.prompt("Question", type=str)
                if q.lower() in ['quit', 'exit', 'q']:
                    break
                
                click.echo()
                process_question(q)
                click.echo("\n" + "="*50 + "\n")
                
            except (KeyboardInterrupt, EOFError):
                click.echo("\nGoodbye! 👋")
                break
    else:
        process_question(question)
    
    return 0

if __name__ == '__main__':
    ask()