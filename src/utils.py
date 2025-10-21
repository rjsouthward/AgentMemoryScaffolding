"""Utility functions for document processing and citation management.
"""

import re
from typing import List, Tuple, Dict
from src.dataclass import RetrievedDocument


def construct_sources_string(all_documents: List[RetrievedDocument]) -> str:
    """Construct a formatted string containing all source documents for LM input.

    This function creates a structured representation of retrieved documents
    that can be passed to language models for answer generation. Each document
    includes metadata (domain, retrieval reason) and its text excerpts.

    Args:
        all_documents: List of retrieved documents with excerpts and metadata

    Returns:
        Formatted string with all document content and metadata, or
        "No sources found" if the list is empty
    """
    if not all_documents:
        return "No sources found"

    source_strings = []
    for source in all_documents:
        # Combine all excerpts from this document
        content = "\n".join(source.excerpts) if source.excerpts else ""

        # Extract domain from URL for cleaner display
        url = source.url
        if url.startswith("http"):
            domain = url.split("/")[2]  # Extract domain from full URL
        else:
            domain = url  # Use as-is if not a standard URL

        retrieval_reason = source.reason_for_retrieval

        # Build metadata string with available information
        meta_strings = []
        if domain:
            meta_strings.append(f"Domain: {domain}")
        if retrieval_reason:
            meta_strings.append(f"Reason: {retrieval_reason}")

        # Format metadata in brackets, or empty if no metadata
        meta = f"[{', '.join(meta_strings)}]" if meta_strings else ""
        source_strings.append(f"{meta}\n{content}")

    # Join all sources with double newlines for clear separation
    result = "\n\n".join(source_strings)
    return result


def reset_citation_indices(answer: str) -> Tuple[str, Dict[str, str]]:
    """Normalize citation indices in generated text to be sequential.

    This function takes text with potentially non-sequential citation indices
    (e.g., [1], [5], [2]) and renumbers them to be sequential based on their
    order of first appearance (e.g., [1], [2], [3]).

    This is useful when combining multiple generated texts that each have
    their own citation numbering schemes.

    Args:
        answer: Text containing citation indices in [n] format

    Returns:
        Tuple containing:
        - Updated text with sequential citation indices
        - Mapping from old indices to new indices (as strings)

    Example:
        Input: "AI[5] is useful[1] for analysis[5]."
        Output: ("AI[1] is useful[2] for analysis[1].", {"5": "1", "1": "2"})
    """
    # Extract all citation indices using regex
    citation_pattern = r'\[(\d+)\]'
    citations = re.findall(citation_pattern, answer)

    # Get unique citations in order of first appearance
    seen = set()
    unique_citations = []
    for citation in citations:
        if citation not in seen:
            unique_citations.append(citation)
            seen.add(citation)

    # Create mapping from old indices to new sequential indices
    citation_mapping = {}
    for i, old_index in enumerate(unique_citations):
        citation_mapping[old_index] = str(i + 1)

    # Replace citations in the answer using the mapping
    def replace_citation(match):
        """Replace a single citation with its new index."""
        old_index = match.group(1)
        new_index = citation_mapping[old_index]
        return f'[{new_index}]'

    # Apply the replacement to the entire text
    result = re.sub(citation_pattern, replace_citation, answer)
    return result, citation_mapping
