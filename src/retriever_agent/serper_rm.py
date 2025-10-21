"""Serper.dev-backed internet retriever.

This implementation uses the simplified `InternetRetriever` base and expects a
Serper API key directly. It returns candidate documents for scraping and (optionally)
embedding-based filtering.
"""

from typing import List, Dict, Any, Literal, Optional
import httpx

from src.retriever_agent.internet_retriever import InternetRetriever
from src.dataclass import RetrievedDocument, DocumentType
from src.encoder import Encoder


class SerperRM(InternetRetriever):
    """Google search retriever backed by Serper.dev (homework version).

    Provide your Serper API key directly to the constructor.
    """

    def __init__(
        self,
        api_key: str,
        encoder: Optional[Encoder] = None,
        top_k: int = 10,
        per_query_max_snippet_count: int = 10,
        per_source_max_snippet_count: int = 3,
        enable_embedding_filter: bool = True,
    ):
        super().__init__(
            top_k=top_k,
            encoder=encoder,
            per_query_max_snippet_count=per_query_max_snippet_count,
            per_source_max_snippet_count=per_source_max_snippet_count,
            enable_embedding_filter=enable_embedding_filter,
        )
        if not api_key:
            raise RuntimeError("You must provide a Serper API key.")
        self.api_key = api_key
        self.base_url = "https://google.serper.dev"

    async def _execute_search_async(self, query: str, query_params: Dict[str, Any]) -> List[RetrievedDocument]:
        """Call Serper.dev to obtain web search results for the query."""
        search_type = query_params.get("search_type", "search")
        url = f"{self.base_url}/{search_type}"
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
        }
        payload = {"q": query, "num": query_params.get("num", self.top_k)}
        # Optional time filter (e.g., "qdr:d") if provided
        if query_params.get("tbs"):
            payload["tbs"] = query_params["tbs"]

        async with httpx.AsyncClient() as client:
            resp = await client.post(url, headers=headers, json=payload)
        if resp.status_code != 200:
            raise RuntimeError(f"Serper search failed with status {resp.status_code}")

        data = resp.json()
        results = data.get("news", []) if search_type == "news" else data.get("organic", [])

        docs: List[RetrievedDocument] = []
        for item in results:
            link = item.get("link")
            title = item.get("title")
            if link:
                docs.append(
                    RetrievedDocument(
                        url=link,
                        title=title,
                        reason_for_retrieval=query,
                        document_type=DocumentType.DOCUMENT_TYPE_WEB_PAGE
                    )
                )
        return docs

    async def aretrieve(
        self,
        query: str,
        search_type: Literal["search", "news", "scholar"] = "search",
        tbs: Literal["qdr:h", "qdr:d", "qdr:w", "qdr:m", "qdr:y", ""] = "",
    ) -> List[RetrievedDocument]:
        """Run a Serper search then scrape and filter results via the base class."""
        return await super().aretrieve(query, search_type=search_type, tbs=tbs)
