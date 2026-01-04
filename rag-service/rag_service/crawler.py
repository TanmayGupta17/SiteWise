"""
Web crawler that respects robots.txt and extracts clean content.
This is the first step of our RAG pipeline - getting the data.
"""
import time
import requests
from urllib.robotparser import RobotFileParser
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from bs4.element import Comment
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass

from .config import (
    DEFAULT_CRAWL_DELAY_MS, DEFAULT_MAX_PAGES, DEFAULT_MAX_DEPTH,
    USER_AGENT, CRAWLED_DIR
)
from .utils import (
    get_domain, is_same_domain, normalize_url, clean_text,
    url_to_filename, save_json, logger
)

@dataclass
class CrawlResult:
    """What we get back after crawling."""
    page_count: int
    skipped_count: int
    urls: List[str]
    errors: List[str]

@dataclass
class PageContent:
    """Container for a scraped page."""
    url: str
    title: str
    content: str
    links: List[str]
    timestamp: float

class RobotChecker:
    """Checks if we're allowed to crawl URLs according to robots.txt."""
    
    def __init__(self, user_agent: str = USER_AGENT):
        self.user_agent = user_agent
        self._robots_cache: Dict[str, RobotFileParser] = {}
    
    def can_fetch(self, url: str) -> bool:
        """Check if we can crawl this URL."""
        try:
            parsed = urlparse(url)
            robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
            
            # Cache robots.txt files so we don't fetch them repeatedly
            if robots_url not in self._robots_cache:
                rp = RobotFileParser()
                rp.set_url(robots_url)
                rp.read()
                self._robots_cache[robots_url] = rp
            
            return self._robots_cache[robots_url].can_fetch(self.user_agent, url)
        except Exception as e:
            logger.warning("Error checking robots.txt", url=url, error=str(e))
            # If we can't check robots.txt, assume it's OK (be permissive)
            return True

class WebCrawler:
    """
    The main crawler class. This does the actual work of:
    1. Visiting web pages
    2. Extracting content
    3. Following links
    4. Staying within boundaries
    """
    
    def __init__(self, 
                 crawl_delay_ms: int = DEFAULT_CRAWL_DELAY_MS,
                 max_pages: int = DEFAULT_MAX_PAGES,
                 max_depth: int = DEFAULT_MAX_DEPTH,
                 respect_robots_txt: bool = True):
        self.crawl_delay_ms = crawl_delay_ms
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.respect_robots_txt = respect_robots_txt
        
        # Set up HTTP session with proper headers
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': USER_AGENT})
        
        self.robot_checker = RobotChecker()
        self.visited_urls: Set[str] = set()
        self.crawled_pages: List[PageContent] = []
        self.errors: List[str] = []
        
    def _extract_content(self, soup: BeautifulSoup) -> Tuple[str, str]:
        """Extract clean title and content from HTML."""
        # Get page title
        title_tag = soup.find('title')
        title = title_tag.get_text().strip() if title_tag else ""
        
        # Remove elements we don't want
        for element in soup(['script', 'style', 'nav', 'header', 'footer']):
            element.decompose()
        
        # Remove HTML comments
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()
        
        # Try to find main content first
        content_selectors = [
            'main', 'article', '[role="main"]',
            '.content', '.main-content', '.post-content',
            '#content', '#main-content'
        ]
        
        content_text = ""
        for selector in content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                content_text = content_elem.get_text()
                break
        
        # Fallback to body if no main content found
        if not content_text:
            body = soup.find('body')
            content_text = body.get_text() if body else soup.get_text()
        
        # Clean up the text
        content_text = clean_text(content_text)
        
        return title, content_text
    
    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Find all links on the page."""
        links = []
        
        for link_tag in soup.find_all('a', href=True):
            href = link_tag['href']
            absolute_url = normalize_url(href, base_url)
            
            # Skip non-HTTP links
            if not absolute_url.startswith(('http://', 'https://')):
                continue
            
            # Skip file downloads (we only want HTML pages)
            skip_extensions = {'.pdf', '.doc', '.docx', '.zip', '.jpg', '.png', '.gif'}
            if any(absolute_url.lower().endswith(ext) for ext in skip_extensions):
                continue
            
            links.append(absolute_url)
        
        return links
    
    def _fetch_page(self, url: str) -> Optional[PageContent]:
        """Download and parse a single page."""
        try:
            # Check if robots.txt allows this (if enabled)
            if self.respect_robots_txt and not self.robot_checker.can_fetch(url):
                self.errors.append(f"Blocked by robots.txt: {url}")
                return None
            
            # Be polite - wait between requests
            if self.crawl_delay_ms > 0:
                time.sleep(self.crawl_delay_ms / 1000.0)
            
            logger.info("Fetching page", url=url)
            
            # Download the page
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            # Make sure it's HTML
            content_type = response.headers.get('content-type', '').lower()
            if 'text/html' not in content_type:
                self.errors.append(f"Non-HTML content: {url}")
                return None
            
            # Parse HTML
            soup = BeautifulSoup(response.content.decode('utf-8', errors='ignore'), 'html.parser')
            title, content = self._extract_content(soup)
            links = self._extract_links(soup, url)
            
            return PageContent(
                url=url,
                title=title,
                content=content,
                links=links,
                timestamp=time.time()
            )
            
        except Exception as e:
            error_msg = f"Error fetching {url}: {str(e)}"
            self.errors.append(error_msg)
            logger.error("Fetch failed", url=url, error=str(e))
            return None
    
    def crawl(self, start_url: str) -> CrawlResult:
        """
        Main crawling method. This orchestrates the entire process.
        """
        logger.info("Starting crawl", start_url=start_url, max_pages=self.max_pages)
        
        # Reset state for new crawl
        self.visited_urls = set()
        self.crawled_pages = []
        self.errors = []
        
        # Use a queue for breadth-first crawling
        # Each item is (url, depth)
        url_queue = [(start_url, 0)]
        skipped_count = 0
        
        while url_queue and len(self.crawled_pages) < self.max_pages:
            url, depth = url_queue.pop(0)
            
            # Skip if already visited
            if url in self.visited_urls:
                continue
            
            # Skip if too deep
            if depth > self.max_depth:
                skipped_count += 1
                continue
            
            # Skip if different domain
            if not is_same_domain(url, start_url):
                skipped_count += 1
                continue
            
            self.visited_urls.add(url)
            
            # Download and process the page
            page_content = self._fetch_page(url)
            if page_content is None:
                continue
            
            # Skip pages with very little content (probably error pages)
            if len(page_content.content) < 100:
                skipped_count += 1
                continue
            
            self.crawled_pages.append(page_content)
            
            # Add new links to the queue
            for link in page_content.links:
                if (link not in self.visited_urls and 
                    is_same_domain(link, start_url)):
                    url_queue.append((link, depth + 1))
            
            logger.info("Page crawled", 
                       url=url, 
                       content_length=len(page_content.content),
                       total_pages=len(self.crawled_pages))
        
        # Save all the crawled data
        self._save_crawled_data()
        
        result = CrawlResult(
            page_count=len(self.crawled_pages),
            skipped_count=skipped_count,
            urls=[page.url for page in self.crawled_pages],
            errors=self.errors
        )
        
        logger.info("Crawl completed", 
                   pages_crawled=result.page_count,
                   pages_skipped=result.skipped_count)
        
        return result
    
    def _save_crawled_data(self) -> None:
        """Save all crawled pages to disk as JSON files."""
        for page in self.crawled_pages:
            filename = url_to_filename(page.url)
            filepath = CRAWLED_DIR / filename
            
            page_data = {
                'url': page.url,
                'title': page.title,
                'content': page.content,
                'timestamp': page.timestamp,
                'content_length': len(page.content)
            }
            
            save_json(page_data, filepath)
        
        # Also save a summary
        summary = {
            'crawl_timestamp': time.time(),
            'pages_count': len(self.crawled_pages),
            'urls': [page.url for page in self.crawled_pages],
            'errors': self.errors
        }
        
        save_json(summary, CRAWLED_DIR / 'crawl_summary.json')