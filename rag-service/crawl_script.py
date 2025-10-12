#!/usr/bin/env python3
"""
Command-line script for crawling websites.
Usage: python crawl_script.py https://example.com --max-pages 20
"""
import click
import json
from rag_service.crawler import WebCrawler
from rag_service.config import DEFAULT_CRAWL_DELAY_MS, DEFAULT_MAX_PAGES, DEFAULT_MAX_DEPTH

@click.command()
@click.argument('start_url')
@click.option('--max-pages', default=DEFAULT_MAX_PAGES, help='Maximum number of pages to crawl')
@click.option('--max-depth', default=DEFAULT_MAX_DEPTH, help='Maximum crawl depth')
@click.option('--crawl-delay-ms', default=DEFAULT_CRAWL_DELAY_MS, help='Delay between requests in milliseconds')
@click.option('--output', default=None, help='Output file for results (JSON)')
def crawl(start_url, max_pages, max_depth, crawl_delay_ms, output):
    """Crawl a website starting from START_URL."""
    click.echo(f"🕷️  Starting crawl of {start_url}")
    click.echo(f"   Max pages: {max_pages}")
    click.echo(f"   Max depth: {max_depth}")
    click.echo(f"   Crawl delay: {crawl_delay_ms}ms")
    click.echo()
    
    try:
        # Create and configure crawler
        crawler = WebCrawler(
            crawl_delay_ms=crawl_delay_ms,
            max_pages=max_pages,
            max_depth=max_depth
        )
        
        # Show progress
        with click.progressbar(length=max_pages, label='Crawling pages') as bar:
            result = crawler.crawl(start_url)
            bar.update(result.page_count)
        
        # Show results
        click.echo()
        click.echo(f"✅ Crawl completed!")
        click.echo(f"   Pages crawled: {result.page_count}")
        click.echo(f"   Pages skipped: {result.skipped_count}")
        click.echo(f"   Errors: {len(result.errors)}")
        
        if result.errors:
            click.echo("\n❌ Errors encountered:")
            for error in result.errors[:5]:  # Show first 5 errors
                click.echo(f"   • {error}")
            if len(result.errors) > 5:
                click.echo(f"   ... and {len(result.errors) - 5} more")
        
        # Save results if requested
        if output:
            results_data = {
                'start_url': start_url,
                'page_count': result.page_count,
                'skipped_count': result.skipped_count,
                'urls': result.urls,
                'errors': result.errors
            }
            
            with open(output, 'w') as f:
                json.dump(results_data, f, indent=2)
            
            click.echo(f"\n💾 Results saved to {output}")
        
        click.echo(f"\n📁 Crawled content saved to: data/crawled/")
        
    except Exception as e:
        click.echo(f"❌ Crawl failed: {e}", err=True)
        return 1
    
    return 0

if __name__ == '__main__':
    crawl()