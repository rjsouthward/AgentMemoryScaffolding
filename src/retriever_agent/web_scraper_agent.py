"""Web page scraping helpers used by retrievers.

This module provides:
- `sanitize_markdown`: simple cleanup for markdown content before chunking
- `WebPageScraper`: a small utility that crawls URLs and splits content into excerpts
"""

import re
from typing import Any, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
from langchain_text_splitters import RecursiveCharacterTextSplitter

from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CacheMode,
    CrawlerRunConfig,
    DefaultMarkdownGenerator,
    PruningContentFilter,
    RateLimiter,
    SemaphoreDispatcher,
)

from src.dataclass import RetrievedDocument

MAX_EXCERPT_PER_SOURCE = 30


def sanitize_markdown(markdown_content: str) -> str:
    """Remove noisy constructs and compress spacing to make chunking cleaner.

    Heuristics applied per line:
    - Remove image/link markdown, brackets and parentheses, digits, and whitespace
    - Keep the original line only if at least 10 non-whitespace, non-digit chars remain
    Then compress multiple blank lines and strip header spacing.
    """
    if not markdown_content:
        return ""

    # Split by double newlines to get blocks
    blocks = markdown_content.split('\n\n')
    sanitized_blocks = []

    for block in blocks:
        # Split block into lines
        lines = block.split('\n')
        sanitized_lines = []

        for line in lines:
            # Convert markdown bold headers (**text**) to markdown h2 headers (## text)
            if len(line.strip()) <= 50:
                if re.match(r'^\*\*(.*?)\*\*$', line.strip()):
                    line = re.sub(r'^\*\*(.*?)\*\*$', r'\n\n## \1', line.strip())

            # Create a test version of the line to check content length
            test_line = line

            # Remove image links ![text](url)
            test_line = re.sub(r'!\[.*?\]\(.*?\)', '', test_line)

            # Remove regular links [text](url)
            test_line = re.sub(r'\[.*?\]\(.*?\)', '', test_line)

            # Remove all kinds of parenthesis characters
            test_line = re.sub(r'[\(\)\[\]\{\}]', '', test_line)

            # Remove all numbers
            test_line = re.sub(r'\d', '', test_line)

            # Remove all whitespace
            test_line = re.sub(r'\s', '', test_line)

            # Check if remaining length is greater than 10
            if len(test_line) > 10:
                sanitized_lines.append(line)  # Keep the original line

        # Join lines back into a block
        if sanitized_lines:  # Only add non-empty blocks
            sanitized_blocks.append('\n'.join(sanitized_lines))

    # Join blocks back together
    merged_content = '\n\n'.join(sanitized_blocks)
    # Squeeze more than two consecutive newlines into exactly two newlines
    merged_content = re.sub(r'\n{3,}', '\n\n', merged_content)

    # Handle section lines - squeeze multiple newlines after headers to single newline
    merged_content = re.sub(r'(^#+[^#\n]*)\n+', r'\1\n', merged_content, flags=re.MULTILINE)

    merged_content = merged_content.replace("**", "")
    return merged_content


class WebPageScraper:
    """Scrape URLs and split content into text snippets.

    Acknowledgement: Parts adapted from the Stanford OVAL WikiChat project.
    """

    def __init__(
        self,
        min_char_count: int = 150,
        snippet_chunk_size: int = 2000,
    ):
        self.httpx_client = httpx.Client(verify=False)
        self.min_char_count = min_char_count
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=snippet_chunk_size,
            chunk_overlap=0,
            length_function=len,
            is_separator_regex=False,
            separators=[
                "###### ",
                "##### "
                "#### "
                "### "
                "## "
                "# ",
                "\n\n",
                "\n",
                ".",
                "\uff0e",  # Fullwidth full stop
                "\u3002",  # Ideographic full stop
                ",",
                "\uff0c",  # Fullwidth comma
                "\u3001",  # Ideographic comma
                " ",
                "\u200B",  # Zero-width space
                "",
            ],
        )

    def _get_crawl4ai_run_config(self) -> CrawlerRunConfig:
        """Create a Crawl4AI run configuration for general article-like pages."""
        return CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            markdown_generator=DefaultMarkdownGenerator(content_filter=PruningContentFilter(
                # Lower → more content retained, higher → more content pruned
                threshold=0.45,
                # "fixed" or "dynamic"
                threshold_type="dynamic",
                # Ignore nodes with <5 words
                min_word_threshold=5
            )),
            excluded_tags=['form', 'header', 'footer', 'nav'],
            exclude_social_media_links=True,
            page_timeout=600000,
            semaphore_count=10,
            # Add JavaScript to handle cookie popups
            js_code="""
            // Generic cookie acceptance script
            const acceptButtons = [
                'button[id*="accept"]',
                'button[class*="accept"]',
                'button[aria-label*="accept"]',
                'button[title*="accept"]',
                'a[id*="accept"]',
                'a[class*="consent"]',
                'button[class*="agree"]',
                'button[class*="consent"]',
                'button[class*="cookie"] button[class*="accept"]',
                'div[class*="cookie"] button:first-of-type',
                '.cookie-notice button',
                '#cookie-notice button',
                'button:has-text("Accept")',
                'button:has-text("I agree")',
                'button:has-text("Got it")',
                'button:has-text("OK")'
            ];
            
            for (const selector of acceptButtons) {
                try {
                    const elements = document.querySelectorAll(selector);
                    for (const el of elements) {
                        if (el && (el.offsetWidth > 0 || el.offsetHeight > 0)) {
                            el.click();
                            console.log('Clicked cookie accept button:', selector);
                            break;
                        }
                    }
                } catch (e) {
                    console.log('Error clicking:', selector, e);
                }
            }
            
            // Also try to remove cookie overlays directly
            const overlays = document.querySelectorAll('[class*="cookie"], [id*="cookie"], [class*="consent"], [id*="consent"], [class*="gdpr"], [class*="privacy-banner"]');
            overlays.forEach(el => {
                if (el && el.style) {
                    el.style.display = 'none';
                }
            });
            """,
        )

    def _get_crawl4ai_browser_config(self) -> BrowserConfig:
        """Create a headless browser configuration tuned for reliability."""
        return BrowserConfig(
            headless=True,
            verbose=False,
            # Use extra_args instead of browser_args
            extra_args=[
                '--disable-blink-features=AutomationControlled',
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-dev-shm-usage',
                '--window-size=1920,1080',
                '--start-maximized',
                # Additional args to help with cookie popups
                '--disable-notifications',
                '--disable-popup-blocking',
            ],
            # You can also use these existing parameters
            ignore_https_errors=True,
            java_script_enabled=True,
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )

    async def crawl_with_crawl4ai(self, urls: List[str]) -> Dict[str, Dict[str, str]]:
        """Crawl multiple URLs and return raw HTML and markdown content per URL."""

        dispatcher = SemaphoreDispatcher(
            semaphore_count=30,
            rate_limiter=RateLimiter(
                base_delay=(0.5, 1.0),
                max_delay=2.0
            )
        )

        async with AsyncWebCrawler(config=self._get_crawl4ai_browser_config()) as crawler:
            results = await crawler.arun_many(
                urls,
                config=self._get_crawl4ai_run_config(),
                dispatcher=dispatcher
            )
            formatted_results: Dict[str, Dict[str, str]] = {}
            for result in results:
                if result is not None and result.markdown is not None:
                    formatted_results[result.url] = {"html": result.html, "markdown": result.markdown.fit_markdown}
            return formatted_results
    
    async def _process_urls_async(self, urls: List[str]) -> Dict[str, Dict[str, Any]]:
        """Crawl URLs and produce sanitized snippets per URL."""
        # 1) Crawl the URLs with Crawl4AI
        crawl_results = await self.crawl_with_crawl4ai(urls)

        def process_url(url: str, page: Dict[str, str]) -> tuple[str, Dict[str, Any] | None]:
            """Sanitize and split crawled markdown into excerpts for a single URL."""
            markdown = page.get("markdown", "")
            if markdown and len(markdown) > self.min_char_count:
                sanitized = sanitize_markdown(markdown)
                snippets = [
                    s for s in self.text_splitter.split_text(sanitized)
                    if len(s) >= self.min_char_count
                ]
                snippets = snippets[: min(len(snippets), MAX_EXCERPT_PER_SOURCE)]
                return url, {"text": markdown, "snippets": snippets}
            return url, None

        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(process_url, url, page)
                for url, page in crawl_results.items()
            ]
            processed_results: Dict[str, Dict[str, Any]] = {}
            for future in as_completed(futures):
                url, processed = future.result()
                if processed:
                    processed_results[url] = processed
        return processed_results

    async def enrich_retrieved_document(self, retrieved_documents: List[RetrievedDocument]) -> List[RetrievedDocument]:
        """Populate `excerpts` for each document by crawling and chunking its URL."""
        urls = [d.url for d in retrieved_documents]
        url_to_page = await self._process_urls_async(urls)

        enriched: List[RetrievedDocument] = []
        for doc in retrieved_documents:
            snippets = url_to_page.get(doc.url, {}).get("snippets", [])
            doc.excerpts.extend(snippets)
            enriched.append(doc)
        return enriched
