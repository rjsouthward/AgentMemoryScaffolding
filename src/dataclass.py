"""Data classes and enums for the CS224V homework 1.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
import enum


class DocumentType(enum.Enum):
    """Enumeration of supported document types.

    This enum distinguishes between different sources of retrieved documents
    to enable specialized processing and rendering.
    """
    DOCUMENT_TYPE_WEB_PAGE = 1  # Standard web pages
    DOCUMENT_TYPE_DATATALK = 2  # DataTalk/tabular data sources


@dataclass
class RetrievedDocument:
    """Container for a retrieved document with metadata and content excerpts.

    This class represents a document that has been retrieved from some source
    (web page, database, etc.) along with its extracted content and metadata.
    The excerpts field contains the actual text content that will be used
    for generating responses.

    Attributes:
        url: Required document URL or identifier
        excerpts: List of text snippets extracted from the document
        title: Optional document title for display purposes
        timestamp: Optional publication or retrieval datetime
        reason_for_retrieval: Optional query or reason explaining why this was retrieved
        document_type: Type of document (web page, data table, etc.)
        metadata: Optional additional metadata as key-value pairs
    """
    url: str
    excerpts: List[str] = field(default_factory=list)
    title: Optional[str] = None
    timestamp: Optional[datetime] = None
    reason_for_retrieval: Optional[str] = None
    document_type: DocumentType = DocumentType.DOCUMENT_TYPE_WEB_PAGE
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert the document to a dictionary for JSON serialization.

        Returns:
            Dictionary representation with all fields, using enum values for document_type
        """
        return {
            "url": self.url,
            "excerpts": self.excerpts,
            "title": self.title,
            "timestamp": self.timestamp,
            "reason_for_retrieval": self.reason_for_retrieval,
            "document_type": self.document_type.value,  # Convert enum to int value
            "metadata": self.metadata
        }

    @classmethod
    def _parse_document_type(cls, doc_type_value: Any) -> DocumentType:
        """Parse document type from various formats (int, string name, or enum).
        
        Args:
            doc_type_value: Document type as int value, string name, or enum
            
        Returns:
            DocumentType enum instance
        """
        if isinstance(doc_type_value, DocumentType):
            return doc_type_value
        elif isinstance(doc_type_value, str):
            # Handle string enum names like "DOCUMENT_TYPE_DATATALK"
            try:
                return DocumentType[doc_type_value]
            except KeyError:
                return DocumentType.DOCUMENT_TYPE_WEB_PAGE
        elif isinstance(doc_type_value, int):
            # Handle integer values
            try:
                return DocumentType(doc_type_value)
            except ValueError:
                return DocumentType.DOCUMENT_TYPE_WEB_PAGE
        else:
            return DocumentType.DOCUMENT_TYPE_WEB_PAGE

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RetrievedDocument':
        """Create a RetrievedDocument from a dictionary representation.

        Args:
            data: Dictionary containing document data

        Returns:
            New RetrievedDocument instance
        """
        return cls(
            url=data['url'],
            excerpts=data.get('excerpts', []),
            title=data.get('title'),
            timestamp=data.get('timestamp'),
            reason_for_retrieval=data.get('reason_for_retrieval'),
            # Convert int value or string name back to enum, defaulting to web page type
            document_type=cls._parse_document_type(data.get('document_type', DocumentType.DOCUMENT_TYPE_WEB_PAGE.value)),
            metadata=data.get('metadata')
        )


@dataclass(init=False)
class RagRequest:
    """Request object for RAG (Retrieval-Augmented Generation) service.

    This class encapsulates all parameters needed to perform a RAG operation,
    including the question to answer, optional context, retrieval limits,
    and styling preferences for the generated response.

    The custom __init__ method allows for flexible parameter passing while
    maintaining type safety and default values.
    """
    question: str
    question_context: Optional[str] = None
    max_retriever_calls: Optional[int] = 3
    answer_style: str = """
    - Write in a clear, professional tone. Structure information logically with smooth transitions between concepts. 
    - Include all relevant details that is directly relevant to the question from sources while maintaining readability
    - Use precise, specific language rather than vague generalizations
    - Balance comprehensiveness with conciseness - avoid unnecessary verbosity
    - Maintain objective, factual presentation without speculation
    """.strip()
    # Container for any additional, optional parameters passed via **kwargs
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        question: str,
        question_context: Optional[str] = None,
        max_retriever_calls: Optional[int] = 3,
        answer_style: str = (
            """
            - Write in a clear, professional tone. Structure information logically with smooth transitions between concepts. 
            - Include all relevant details that is directly relevant to the question from sources while maintaining readability
            - Use precise, specific language rather than vague generalizations
            - Balance comprehensiveness with conciseness - avoid unnecessary verbosity
            - Maintain objective, factual presentation without speculation
            """.strip()
        ),
        **kwargs: Any,
    ) -> None:
        """Initialize a RAG request.

        Args:
            question: The main question to be answered
            question_context: Optional additional context about the question
            max_retriever_calls: Maximum number of retrieval operations to perform
            answer_style: Style guidelines for the generated answer
            **kwargs: Additional parameters stored in extra_kwargs
        """
        self.question = question
        self.question_context = question_context
        self.max_retriever_calls = max_retriever_calls
        self.answer_style = answer_style
        # Store any additional keyword arguments for downstream use
        self.extra_kwargs = dict(kwargs) if kwargs else {}


@dataclass
class RagResponse:
    """Response object from RAG (Retrieval-Augmented Generation) service.

    This class contains the complete results of a RAG operation, including
    the generated answer, source documents, and metadata about the retrieval process.

    Attributes:
        question: The original question that was answered
        answer: The generated answer with inline citations
        question_context: Optional context that was provided with the question
        cited_documents: Documents that were referenced in the answer
        uncited_documents: Documents that were retrieved but not used in the answer
        num_retriever_calls: Number of retrieval operations performed
    """
    question: str
    answer: str
    question_context: Optional[str] = None
    cited_documents: List[RetrievedDocument] = field(default_factory=list)
    uncited_documents: List[RetrievedDocument] = field(default_factory=list)
    key_insight: str = ""
    num_retriever_calls: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert the response to a dictionary for JSON serialization.

        Returns:
            Dictionary representation with all fields, converting documents to dicts
        """
        return {
            "question": self.question,
            "question_context": self.question_context,
            "answer": self.answer,
            "cited_documents": [doc.to_dict() for doc in self.cited_documents],
            "uncited_documents": [doc.to_dict() for doc in self.uncited_documents],
            "key_insight": self.key_insight,
            "num_retriever_calls": self.num_retriever_calls,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RagResponse':
        """Create a RagResponse from a dictionary representation.

        Args:
            data: Dictionary containing response data

        Returns:
            New RagResponse instance with documents reconstructed from dicts
        """
        rag_response = cls(
            question=data['question'],
            answer=data['answer'],
            question_context=data.get('question_context'),
            key_insight=data.get('key_insight', ''),
            num_retriever_calls=data.get('num_retriever_calls', 0),
        )
        # Reconstruct document objects from dictionary representations
        rag_response.cited_documents = [RetrievedDocument.from_dict(doc) for doc in data.get('cited_documents', [])]
        rag_response.uncited_documents = [RetrievedDocument.from_dict(
            doc) for doc in data.get('uncited_documents', [])]
        return rag_response


@dataclass
class LiteratureSearchAgentRequest:
    """Request object for Literature Search Agent service.

    This class encapsulates parameters for conducting a comprehensive literature search
    on a given topic using multiple RAG operations and synthesis.

    Attributes:
        topic: The main topic to search
        report_style: Style guidelines for the final report
        max_retriever_calls: Maximum number of retrieval operations across all sub-questions
        guideline: Instructions for determining when the literature search is complete
        with_synthesis: Whether to perform final synthesis of all findings
    """
    topic: str
    report_style: Optional[str] = "Comprehensive, highly accurate, and exhaustive; include every relevant detail and ensure no important information is omitted."
    max_retriever_calls: int = 15
    guideline: str = "Conduct a survey. Stop when information gain is low or hit the budget"
    with_synthesis: bool = True


@dataclass
class LiteratureSearchAgentResponse:
    """Response object from Literature Search Agent service.

    This class contains the complete results of a literature search operation, including
    the synthesized writeup, all source documents, and the individual RAG responses
    that contributed to the final literature search.

    Attributes:
        topic: The topic that was searched
        guideline: The guideline that was used to determine completeness
        writeup: The final synthesized literature search writeup
        cited_documents: All documents that were cited in the final writeup
        rag_responses: Individual RAG responses for each sub-question explored
    """
    topic: str
    guideline: str
    writeup: str
    cited_documents: List[RetrievedDocument] = field(default_factory=list)
    rag_responses: List[RagResponse] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the response to a dictionary for JSON serialization.

        Returns:
            Dictionary representation with all fields, converting nested objects to dicts
        """
        return {
            "topic": self.topic,
            "guideline": self.guideline,
            "writeup": self.writeup,
            "cited_documents": [doc.to_dict() for doc in self.cited_documents],
            "rag_responses": [rag_response.to_dict() for rag_response in self.rag_responses],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LiteratureSearchAgentResponse':
        """Reconstruct LiteratureSearchAgentResponse from dictionary format.

        Args:
            data: Dictionary containing literature search response data

        Returns:
            New LiteratureSearchAgentResponse instance with all nested objects reconstructed
        """
        # Reconstruct RAG responses from their dictionary representations
        rag_responses = []
        for rag_data in data.get('rag_responses', []):
            rag_response = RagResponse(
                question=rag_data['question'],
                answer=rag_data['answer'],
                cited_documents=rag_data.get('cited_documents', []),
                uncited_documents=rag_data.get('uncited_documents', []),
            )
            rag_responses.append(rag_response)

        return cls(
            topic=data['topic'],
            guideline=data.get('guideline', ''),  # Add missing guideline field
            writeup=data['writeup'],
            cited_documents=data.get('cited_documents', []),
            rag_responses=rag_responses
        )


