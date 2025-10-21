import dspy
import asyncio
from typing import List, Tuple

from dataclasses import dataclass
from src.dataclass import (
    LiteratureSearchAgentRequest,
    LiteratureSearchAgentResponse,
    RagResponse,
    RagRequest,
    RetrievedDocument
)
from src.rag import RagAgent
from src.utils import reset_citation_indices


@dataclass
class NextQuestionTask:
    question: str
    question_context: str


class NextStepPlanner(dspy.Signature):
    """Evaluate survey completeness based on the guideline.

    CORE FUNCTION: Determine if sufficient information has been gathered to comprehensively 
    survey the main topic. Apply rigorous completeness criteria before declaring completion.

    COMPLETENESS EVALUATION CRITERIA:
    - Coverage: Have major subtopics and dimensions of the main topic been explored?
    - Depth: Has each explored area been sufficiently detailed for the survey purpose?
    - Balance: Are there significant knowledge gaps that would make the survey incomplete?

    NEXT QUESTION GENERATION RULES (when not complete):
    1. RELEVANCE: Each question must directly advance understanding of the main topic
    2. DISTINCTNESS: Questions must explore different scope with minimal overlap
    3. PARALLELISM: Questions should be answerable independently and simultaneously
    4. IMPACT: Prioritize questions that fill the most critical knowledge gaps
    6. BREADTH OVER DEPTH: Focus on major unexplored areas rather than niche details

    AVOID generating questions about:
    - Tangential or peripheral topics not central to the main subject
    - Overly specialized technical details unless core to the topic
    - Areas already sufficiently covered in completed tasks
    - Topics explicitly mentioned in scope_to_exclude
    """

    topic: str = dspy.InputField(
        desc="Primary topic being surveyed - all analysis and questions must directly relate to this"
    )

    guideline: str = dspy.InputField(
        desc="Guideline for the survey. Strictly follow the guideline to decide if the survey is complete."
    )

    completed_tasks_summary: str = dspy.InputField(
        desc="Detailed summary of all completed exploration tasks with key findings and coverage areas. Use this to identify what has been sufficiently explored vs. what gaps remain."
    )

    max_iterations: int = dspy.InputField(
        desc="Maximum iterations allowed - consider this constraint in completeness evaluation"
    )

    current_iteration: int = dspy.InputField(
        desc="Current iteration number - factor remaining iterations into decision"
    )

    is_complete: bool = dspy.OutputField(
        desc="Return True if the survey comprehensively covers the main topic with sufficient depth in all required scope detailed in the guideline. Err on the side of thoroughness."
    )

    next_questions: List[NextQuestionTask] = dspy.OutputField(
        desc="""If is_complete=False, provide EXACTLY 1-3 strategically chosen questions for parallel exploration.
        
        MANDATORY REQUIREMENTS for each question:
        - Must directly advance the main topic survey (not tangential exploration)
        - Must target a significant knowledge gap from completed_tasks_summary
        - Must be answerable independently without dependencies on other questions
        - Must avoid overlap with already-covered areas
        - Must stay within scope_to_include and avoid scope_to_exclude
        - Must include a precise question_context explaining its strategic value
        
        PRIORITIZATION ORDER:
        1. Any uncovered mandatory requirements from scope_to_include
        2. Major topic dimensions missing from completed tasks that are essential for comprehensive coverage
        3. Other core scopes needed for thorough topic understanding
        
        If is_complete=True, return empty list."""
    )

    reasoning: str = dspy.OutputField(
        desc="""One sentence reasoning for the completeness check."""
    )


class LiteratureSearchAnswerGeneration(dspy.Signature):
    """Generate a comprehensive, well-structured response by synthesizing information from multiple sub-questions.

    CONTENT INTEGRATION RULES:
    - Merge all relevant sub-question answers into a logically coherent narrative
    - Create clear thematic sections with smooth transitions between topics. Use #, ##, ###, etc. to create title of sections and sub-sections.
    - Eliminate redundancy while preserving all unique factual content
    - Exclude sub-questions/answers that don't contribute meaningfully to the survey topic
    - Maintain completeness - no loss of relevant information from source material
    - No title, conclusion, summary, or reference at the end of the answer.

    CITATION PRESERVATION:
    - Preserve ALL original citations exactly as provided - no format modifications
    - Use [1] format for single citation and [1][2] format for multiple citations immediately after the specific fact/claim being cited
    - Cite individual facts, not entire sentences: "Machine learning[1] uses algorithms[2] to find patterns."
    - Avoid redundant and multiple citations for the same fact
    - Do NOT move citations to sentence endings or consolidate multiple citations
    - Maintain citation accuracy when reorganizing content across sections

    STRUCTURAL REQUIREMENTS:
    - Organize content into logical sections/subsections based on survey scope
    - Use clear headings and subheadings to guide reader navigation
    - Ensure smooth flow between related concepts and topics
    - Present information in order of importance or logical sequence
    - Create executive summary style opening if appropriate for topic scope

    CONTENT CONSTRAINTS:
    - Base report STRICTLY on provided gathered_information - add no external knowledge
    - Use only facts and claims present in the source sub-question answers
    - If insufficient information exists for comprehensive survey, work with available data
    - Focus on synthesis and organization rather than expansion of content
    """

    topic: str = dspy.InputField(
        description="The specific survey topic or research question to comprehensively address"
    )

    gathered_information: str = dspy.InputField(
        description="""Complete set of sub-question answers with their inline citations from previous research steps. 
        Format typically includes:
        - Sub-question: [question text]
        - Answer: [detailed response with inline citations [1], [2], etc.]
        - (Repeated for multiple sub-questions)"""
    )

    report_style: str = dspy.InputField(
        description="The style of the report"
    )

    answer: str = dspy.OutputField(
        description="""Comprehensive, well-structured response with the following characteristics:
        - Logical organization with clear sections/headings appropriate to topic scope
        - Preserve ALL original citations exactly as provided - no format modifications
        - Integration of all relevant sub-question content without redundancy
        - Smooth narrative flow between related concepts and findings
        - Based EXCLUSIVELY on provided gathered_information
        - Exclusion of irrelevant sub-questions that don't contribute to survey goals
        - Professional survey/report formatting suitable for the topic's complexity
        - No title, conclusion, summary, reference, or any other text at the end of the answer."""
    )


class LiteratureSearchAnswerGenerationModule(dspy.Module):
    def __init__(self, lm: dspy.LM):
        self.survey_answer_generation_lm = lm
        self.survey_answer_generation = dspy.Predict(LiteratureSearchAnswerGeneration)

    def _normalize_rag_response_citation_indices(self, rag_responses: List[RagResponse]) -> Tuple[List[str], List[RetrievedDocument]]:
        # each rag response has answer and cited_documents. each answer in line citation index starts with [1].
        # now we want to normalize the citation index (i.e. we have all retrieved documents combined, citation index in each rag agent response should map to the correct document)
        # return the normalized rag responses and the list of retrieved documents
        all_documents: List[RetrievedDocument] = []
        all_updated_answers: List[str] = []
        for idx, rag_response in enumerate(rag_responses):
            citation_offset = len(all_documents)
            updated_answer = rag_response.answer
            for i in range(len(rag_response.cited_documents)):
                updated_answer = updated_answer.replace(f"[{i+1}]", f"[tmp_{citation_offset+i+1}]")
            for i in range(len(rag_response.cited_documents)):
                updated_answer = updated_answer.replace(f"[tmp_{citation_offset+i+1}]", f"[{citation_offset+i+1}]")

            all_updated_answers.append(
                f"Sub-question ({chr(ord('a')+idx)}): {rag_response.question}\nAnswer: {updated_answer}")
            all_documents.extend(rag_response.cited_documents)
        return all_updated_answers, all_documents

    async def aforward(self, request: LiteratureSearchAgentRequest, rag_responses: List[RagResponse]) -> LiteratureSearchAgentResponse:
        # Normalize citations across all responses
        all_updated_answers, all_documents = self._normalize_rag_response_citation_indices(
            rag_responses)

        with dspy.context(lm=self.survey_answer_generation_lm):
            result = await self.survey_answer_generation.aforward(topic=request.topic, gathered_information="\n".join(all_updated_answers), report_style=request.report_style)
        updated_answer, citation_mapping = reset_citation_indices(result.answer)
        cited_documents = [all_documents[int(idx) - 1]
                           for idx in citation_mapping.keys() if 1 <= int(idx) <= len(all_documents)]
        return LiteratureSearchAgentResponse(topic=request.topic,
                                   guideline=request.guideline,
                                   writeup=updated_answer,
                                   cited_documents=cited_documents,
                                   rag_responses=rag_responses)


class LiteratureSearchAgent(dspy.Module):
    """Tree-based survey agent that explores topics through strategic query generation and exploration."""

    def __init__(self,
                 rag_agent: RagAgent,
                 literature_search_lm: dspy.LM,
                 answer_synthesis_lm: dspy.LM):

        # Initialize services
        self.rag_agent = rag_agent
        self.literature_search_lm = literature_search_lm
        self.answer_synthesis_lm = answer_synthesis_lm

        self.literature_search_answer_generation_module = LiteratureSearchAnswerGenerationModule(lm=answer_synthesis_lm)

        # Initialize completeness checker
        self.completeness_checker = dspy.Predict(NextStepPlanner)

    def _build_tasks_summary(self, all_rag_responses: List[RagResponse]) -> str:
        """Build a summary of completed tasks for the completeness checker."""
        if not all_rag_responses:
            return "No tasks completed yet."

        summaries = []
        for i, rag_response in enumerate(all_rag_responses):
            if rag_response:
                summaries.append(
                    f"{i+1}. Question: {rag_response.question}. Question context: {rag_response.question_context}\n   Key findings: {rag_response.answer}")
            else:
                summaries.append(
                    f"{i+1}. Question: {rag_response.question}. Question context: {rag_response.question_context}\n   Status: Not yet explored")

        return "\n".join(summaries)

    async def _literature_search(self, literature_search_request: LiteratureSearchAgentRequest) -> List[RagResponse]:
        """Main literature search logic using tree-based exploration."""
        call_count = 0

        # Initialize tracking structures
        all_rag_responses: List[RagResponse] = []

        print(f"Starting literature search for topic: {literature_search_request.topic}")

        # Main exploration loop
        while call_count < literature_search_request.max_retriever_calls:
            # Check completeness and get next questions
            completed_tasks_summary = self._build_tasks_summary(all_rag_responses)

            print(f"Completeness check start.")
            with dspy.context(lm=self.literature_search_lm):
                completeness_result = await self.completeness_checker.aforward(
                    topic=literature_search_request.topic,
                    guideline=literature_search_request.guideline,
                    completed_tasks_summary=completed_tasks_summary,
                    max_iterations=literature_search_request.max_retriever_calls,
                    current_iteration=call_count
                )

            print(
                f"Completeness check: {completeness_result.is_complete}, reasoning: {completeness_result.reasoning}")

            # Break if complete
            if completeness_result.is_complete:
                print("Literature search deemed complete by completeness checker")
                break

            # Parse next questions
            next_questions = completeness_result.next_questions
            if not next_questions:
                print("No next questions generated, ending literature search")
                break

            print(f"Generated {len(next_questions)} next questions for exploration")

            # Calculate max retriever calls per question to stay within budget
            remaining_retriever_calls = literature_search_request.max_retriever_calls - call_count
            next_questions = next_questions[:min(len(next_questions), remaining_retriever_calls)]
            max_retriever_calls_per_question = min(2, remaining_retriever_calls // len(next_questions))

            # Execute RAG calls in parallel

            # Use a semaphore to limit concurrency to 1 (i.e., run RAG calls sequentially)
            semaphore = asyncio.Semaphore(3)
            rag_responses = []

            async def run_rag_with_semaphore(rag_request, question):
                async with semaphore:
                    print(
                        f"RAG call start. Question: {question}. Question context: {rag_request.question_context}")
                    return await self.rag_agent.aforward(rag_request)

            rag_service_tasks = []
            for next_question in next_questions:
                rag_request = RagRequest(
                    question=next_question.question,
                    question_context=next_question.question_context,
                    max_retriever_calls=max_retriever_calls_per_question
                )
                rag_service_tasks.append(run_rag_with_semaphore(rag_request, next_question.question))

            print(f"Executing {len(rag_service_tasks)}")
            rag_responses = await asyncio.gather(*rag_service_tasks)

            # Store results
            call_count += sum([rag_response.num_retriever_calls for rag_response in rag_responses])
            all_rag_responses.extend(rag_responses)

            print(
                f"Completed iteration {call_count}, remaining budget: {literature_search_request.max_retriever_calls - call_count}")

        return all_rag_responses

    async def aforward(self, literature_search_request: LiteratureSearchAgentRequest) -> LiteratureSearchAgentResponse:
        """Main entry point for the literature search agent."""
        # Perform survey
        all_rag_responses = await self._literature_search(literature_search_request)
        print(f"Survey completed with {len(all_rag_responses)} responses")

        if literature_search_request.with_synthesis:
            print("Starting final synthesis")
            synthesis_result = await self.literature_search_answer_generation_module.aforward(request=literature_search_request, rag_responses=all_rag_responses)
            print(f"Final synthesis complete")
            return synthesis_result
        else:
            return LiteratureSearchAgentResponse(topic=literature_search_request.topic,
                                       guideline=literature_search_request.guideline,
                                       writeup=None,
                                       cited_documents=[],
                                       rag_responses=all_rag_responses)
