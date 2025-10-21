"""Retriever agents for CS224V homework assignment.

This package provides various document retrieval implementations for the RAG system:

Modules:
- `retriever`: Abstract base class defining the retriever interface
- `internet_retriever`: Base class for web-based retrieval with scraping and filtering
- `serper_rm`: Serper.dev-backed Google search retriever implementation
- `web_scraper_agent`: Web page crawling and content extraction utilities

The retriever agents support different data sources and implement intelligent
filtering using embeddings to improve relevance of retrieved content.
"""

__all__ = [
    "internet_retriever",
    "serper_rm",
    "web_scraper_agent",
    "retriever",
]
