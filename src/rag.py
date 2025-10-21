"""Retrieval-Augmented Generation (RAG) implementation.
"""

import dspy
from typing import Union, List
from src.retriever_agent.retriever import Retriever
from src.dataclass import RagRequest, RagResponse
from src.utils import construct_sources_string, reset_citation_indices
import asyncio
from src.retriever_agent.internet_retriever import InternetRetriever


class RAGAnswerGeneration(dspy.Signature):
    """Generate a comprehensive, well-cited answer using only the provided information.

    CRITICAL PRINCIPLE: Work ONLY with the question asked and information provided.

    CITATION RULES:
    - Use [1] format for single citation and [1][2] format for multiple citations immediately after the specific fact/claim being cited
    - Cite individual facts, not entire sentences: "Machine learning[1] uses algorithms[2] to find patterns."
    - Each claim must be supported by at least one citation
    - Avoid redundant and multiple citations for the same fact
    - Incorrect citation format includes: [source [index]], [index, index, index]

    ANSWER REQUIREMENTS:
    - Base response ONLY on provided information - do not add external knowledge
    - If question cannot be answered from sources, respond exactly: "The question is not answerable based on the provided source documents"
    - Make every sentence informative and well-supported
    - Follow the specified answer_style format
    - No conclusion, summary, or reference at the end of the answer

    Today's date: {datetime.datetime.now().strftime("%Y-%m-%d")}
    """

    question: str = dspy.InputField(
        description="The original question that needs to be answered"
    )

    question_context: str = dspy.InputField(
        description="Additional context about the user's intent, constraints, or requirements"
    )

    gathered_information: str = dspy.InputField(
        description="All source information with document types and content from previous retrieval steps"
    )

    answer_style: str = dspy.InputField(
        description="The style of the answer"
    )

    answer: str = dspy.OutputField(
        description="""Final answer with inline citations using [1], [2], [n] format where a single integer is enclosed in square brackets. Do not modify the citation format.
        Requirements:
        - Cite immediately after each specific fact/claim, not at sentence end
        - Avoid redundant and multiple citations for the same fact
        - Use only information from gathered_information
        - Use question_context to understand the user's intent and constraints
        - If question cannot be answered from sources, respond: 'The question is not answerable based on the provided source documents because [reason]'
        - Ensure every factual claim has supporting citation
        - Prioritize user-uploaded documents in citation preference
        - No conclusion, summary, or reference at the end of the answer.
        """
    )


class QuestionToQuery(dspy.Signature):
    """Convert a question to search queries for internet retrieval. Use minimum number of queries to retrieve the information as each query is expensive."""

    question: str = dspy.InputField(
        description="The original question that needs to be converted to a query"
    )
    max_num_queries: int = dspy.InputField(
        description="The maximum number of queries to use to retrieve the information"
    )
    queries: List[str] = dspy.OutputField(
        description="The queries that are used to retrieve the information. At most 3 queries."
    )


class RagAgent(dspy.Module):
    def __init__(self, retriever: Retriever, rag_lm: dspy.LM):
        """Initialize the RAG agent with retriever and language model.

        Args:
            retriever: Service for retrieving relevant documents
            rag_lm: Language model for question processing and answer generation
        """
        self.retriever = retriever
        self.rag_lm = rag_lm
        # Initialize DSPy modules for the RAG pipeline
        self.convert_question_to_query = dspy.Predict(QuestionToQuery)
        self.answer_generation = dspy.Predict(RAGAnswerGeneration)

    async def aforward(self, rag_request: Union[RagRequest, str]) -> RagResponse:
        """Asynchronously process a RAG request and generate a cited response.

        This is the main entry point for the RAG system. It handles the complete
        pipeline from question to final answer with citations.

        Args:
            rag_request: Either a RagRequest object or a simple question string

        Returns:
            RagResponse containing the answer, citations, and metadata

        Raises:
            AssertionError: If rag_request is None
        """
        assert rag_request is not None, "RAG request cannot be None"

        # Handle string input by converting to RagRequest
        if isinstance(rag_request, str):
            rag_request = RagRequest(question=rag_request)

        # Convert question to optimized search queries
        # Note: The isinstance check seems incorrect - should check retriever type
        if isinstance(self.retriever, InternetRetriever) and rag_request.max_retriever_calls > 1:
            with dspy.context(lm=self.rag_lm):
                query_result = await self.convert_question_to_query.aforward(question=rag_request.question, max_num_queries=rag_request.max_retriever_calls)
                queries = query_result.queries
        else:
            # For non-internet retrievers, use the question directly
            queries = [rag_request.question]

        # Execute all retrieval operations in parallel for efficiency
        retrieval_tasks = [
            self.retriever.aretrieve(query=query)
            for query in queries
        ]

        # Gather all results and flatten into a single document list
        retrieved_documents = []
        results = await asyncio.gather(*retrieval_tasks)
        for retrieved_documents_for_query in results:
            retrieved_documents.extend(retrieved_documents_for_query)

        # Handle case where no documents were retrieved
        if not retrieved_documents:
            return RagResponse(
                question=rag_request.question,
                question_context=rag_request.question_context,
                answer="The question is not answerable based on the provided source documents",
                cited_documents=[],
                uncited_documents=[],
                num_retriever_calls=len(queries)
            )

        # Generate answer using the language model with retrieved sources
        with dspy.context(lm=self.rag_lm):
            rag_answer_synthesis_result = await self.answer_generation.aforward(
                question=rag_request.question,
                question_context=rag_request.question_context or "No question context provided",
                gathered_information=construct_sources_string(retrieved_documents),
                answer_style=rag_request.answer_style,
            )
        answer = rag_answer_synthesis_result.answer

        # Normalize citation indices and extract cited documents
        updated_answer, citation_mapping = reset_citation_indices(answer)
        cited_documents = [
            retrieved_documents[int(idx) - 1]
            for idx in citation_mapping.keys()
            if 1 <= int(idx) <= len(retrieved_documents)
        ]

        # Identify uncited documents - those retrieved but not referenced in the answer
        cited_indices = set(
            int(idx) - 1
            for idx in citation_mapping.keys()
            if 1 <= int(idx) <= len(retrieved_documents)
        )
        uncited_documents = [
            doc for i, doc in enumerate(retrieved_documents)
            if i not in cited_indices
        ]

        # Perform validation checks to ensure document accounting is correct
        total_docs = len(retrieved_documents)
        cited_count = len(cited_documents)
        uncited_count = len(uncited_documents)

        # Ensure cited and uncited documents are mutually exclusive and exhaustive
        assert cited_count + uncited_count == total_docs, \
            f"Document count mismatch: {cited_count} cited + {uncited_count} uncited != {total_docs} total"

        # Verify no overlap between cited and uncited document indices
        uncited_indices = set(range(total_docs)) - cited_indices
        assert len(uncited_indices) == uncited_count, \
            f"Uncited indices mismatch: expected {len(uncited_indices)}, got {uncited_count}"

        # Construct the final response with all results and metadata
        rag_response = RagResponse(
            question=rag_request.question,
            question_context=rag_request.question_context,
            answer=updated_answer,
            cited_documents=cited_documents,
            uncited_documents=uncited_documents,
            num_retriever_calls=len(queries),
        )

        return rag_response
