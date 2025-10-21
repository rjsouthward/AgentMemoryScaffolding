from typing import List, Dict, Any, Tuple, Optional, Set
import numpy as np

from src.retriever_agent.web_scraper_agent import WebPageScraper
from src.dataclass import RetrievedDocument
from src.encoder import Encoder
from src.retriever_agent.retriever import Retriever


class InternetRetriever(Retriever):
    """Internet retriever with optional embedding-based filtering.

    Workflow:
    1) Subclasses perform a search and return candidate documents.
    2) This base scrapes each URL into sanitized text snippets (excerpts).
    3) Optionally, it filters excerpts using cosine similarity to the query,
       keeping top-k per source and top-k globally.
    """

    def __init__(
        self,
        top_k: int = 5,
        enable_embedding_filter: bool = True,
        encoder: Optional[Encoder] = None,
        per_query_max_snippet_count: int = 10,
        per_source_max_snippet_count: int = 3,
    ) -> None:
        self.top_k = top_k
        self.webpage_scraper = WebPageScraper(
            min_char_count=150,
            snippet_chunk_size=2000,
        )
        # Embedding filtering settings
        self.enable_embedding_filter = enable_embedding_filter
        self.per_query_max_snippet_count = max(0, per_query_max_snippet_count)
        self.per_source_max_snippet_count = max(0, per_source_max_snippet_count)
        self.encoder: Encoder = encoder
        if self.enable_embedding_filter and (
            self.per_query_max_snippet_count > 0 or self.per_source_max_snippet_count > 0
        ):
            assert encoder is not None, "Encoder must be provided if embedding filtering is enabled"

    async def _execute_search_async(self, query: str, query_params: Dict[str, Any]) -> List[RetrievedDocument]:
        """Run a specific search API and return candidate documents."""
        raise NotImplementedError

    async def aretrieve(self, query: str, **kwargs) -> List[RetrievedDocument]:
        """Search then enrich pages with simple scraping to produce excerpts."""
        # 1) Execute the specific search
        docs = await self._execute_search_async(query, dict(kwargs))

        # 2) Scrape pages and fill excerpts/timestamps
        await self.webpage_scraper.enrich_retrieved_document(docs)

        # 3) Optional embedding-based relevance filtering of excerpts
        if self.enable_embedding_filter and self.encoder:
            # Keep only documents with snippets/excerpts
            docs_with_excerpts = [d for d in docs if d.excerpts]
            if docs_with_excerpts:
                return await self._efficient_combined_filter(docs_with_excerpts, query)
        return docs

    async def _efficient_combined_filter(
        self,
        retrieved_documents: List[RetrievedDocument],
        query: str,
    ) -> List[RetrievedDocument]:
        """Apply per-source and per-query top-k filtering in one encode pass."""
        # Build mapping of all excerpts to their source documents
        excerpt_to_doc_map: List[Tuple[str, int, int]] = []  # (excerpt, doc_index, excerpt_index)
        all_excerpts: List[str] = []
        for doc_idx, doc in enumerate(retrieved_documents):
            for exc_idx, excerpt in enumerate(doc.excerpts):
                all_excerpts.append(excerpt)
                excerpt_to_doc_map.append((excerpt, doc_idx, exc_idx))

        if not all_excerpts:
            return retrieved_documents

        # Encode all excerpts and the query in one go
        all_texts_to_encode = all_excerpts + [query]
        all_embeddings = await self.encoder.aencode(all_texts_to_encode)  # type: ignore[arg-type]
        # Vectorized cosine similarity (mirrors original logic without sklearn)
        excerpt_embeddings = np.array(all_embeddings[:-1])  # shape: (N, D)
        query_embedding = np.array(all_embeddings[-1:])     # shape: (1, D)
        # dot products
        dots = (excerpt_embeddings @ query_embedding.T).flatten()  # shape: (N,)
        # norms
        ex_norms = np.linalg.norm(excerpt_embeddings, axis=1)
        q_norm = float(np.linalg.norm(query_embedding)) or 1.0
        denom = ex_norms * q_norm
        denom[denom == 0] = 1.0
        similarity_scores = dots / denom

        # Per-source filtering
        if self.per_source_max_snippet_count > 0:
            selected_indices = self._apply_per_source_filter(
                similarity_scores, excerpt_to_doc_map, len(retrieved_documents)
            )
        else:
            selected_indices = set(range(len(all_excerpts)))

        # Per-query filtering (top-k globally)
        if self.per_query_max_snippet_count > 0:
            filtered_scores = [(idx, similarity_scores[idx]) for idx in selected_indices]
            filtered_scores.sort(key=lambda x: x[1], reverse=True)
            top_k = min(self.per_query_max_snippet_count, len(filtered_scores))
            final_indices = {idx for idx, _ in filtered_scores[:top_k]}
        else:
            final_indices = selected_indices

        return self._build_filtered_documents(retrieved_documents, excerpt_to_doc_map, final_indices)

    def _apply_per_source_filter(
        self,
        similarity_scores: List[float] | np.ndarray,
        excerpt_to_doc_map: List[Tuple[str, int, int]],
        num_docs: int,
    ) -> Set[int]:
        """Select top-k excerpt indices per source based on similarity."""
        # Group scores by document
        doc_scores: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(num_docs)}
        for idx, (_, doc_idx, _) in enumerate(excerpt_to_doc_map):
            doc_scores[doc_idx].append((idx, similarity_scores[idx]))

        # Select top-k excerpts per document
        selected_indices: Set[int] = set()
        for doc_idx, scores in doc_scores.items():
            if scores:
                scores.sort(key=lambda x: x[1], reverse=True)
                k = min(self.per_source_max_snippet_count, len(scores))
                selected_indices.update(idx for idx, _ in scores[:k])
        return selected_indices

    @staticmethod
    def _build_filtered_documents(
        original_documents: List[RetrievedDocument],
        excerpt_to_doc_map: List[Tuple[str, int, int]],
        selected_indices: Set[int],
    ) -> List[RetrievedDocument]:
        """Construct new documents that keep only selected excerpts per source."""
        # Group selected excerpts by document index
        doc_excerpts: Dict[int, List[str]] = {i: [] for i in range(len(original_documents))}
        for idx in selected_indices:
            excerpt, doc_idx, _ = excerpt_to_doc_map[idx]
            doc_excerpts[doc_idx].append(excerpt)

        # Create shallow copies updating excerpts only
        result: List[RetrievedDocument] = []
        for doc_idx, excerpts in doc_excerpts.items():
            if excerpts:
                doc = RetrievedDocument(
                    url=original_documents[doc_idx].url,
                    title=original_documents[doc_idx].title,
                    reason_for_retrieval=original_documents[doc_idx].reason_for_retrieval,
                    timestamp=original_documents[doc_idx].timestamp,
                    excerpts=list(excerpts),
                )
                result.append(doc)
        return result
