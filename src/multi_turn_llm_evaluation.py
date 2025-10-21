
"""
Baseline Multi-Turn LLM Conversation Evaluation Framework
Based on: "Evaluating LLM-based Agents for Multi-Turn Conversations: A Survey" (arXiv:2503.22458)

This implementation provides easy-to-implement baseline metrics for evaluating multi-turn conversations:
1. BLEU (Bilingual Evaluation Understudy)
2. ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
3. METEOR (Metric for Evaluation of Translation with Explicit Ordering)
"""

import json
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

# Note: In production, you would install these packages:
# pip install nltk rouge-score sacrebleu

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score as nltk_meteor
    from nltk import word_tokenize
    from rouge_score import rouge_scorer
    from sacrebleu import corpus_bleu
except ImportError:
    print("Warning: NLTK, rouge-score, or sacrebleu not installed. Install with:")
    print("pip install nltk rouge-score sacrebleu")
    print("Also run: python -m nltk.downloader punkt wordnet")


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class EvaluationMetricType(Enum):
    """Types of evaluation metrics from the survey paper."""
    BLEU = "bleu"
    ROUGE = "rouge"
    METEOR = "meteor"
    TASK_COMPLETION = "task_completion"


@dataclass
class ConversationTurn:
    """Represents a single turn in a multi-turn conversation."""
    turn_id: int
    user_query: str
    agent_response: str
    reference_response: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiTurnConversation:
    """Represents a complete multi-turn conversation."""
    conversation_id: str
    turns: List[ConversationTurn]
    task_description: Optional[str] = None
    task_completed: Optional[bool] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LiteratureSearchAgentRequest:
    """Mock request structure for literature search agent."""
    topic: str
    max_results: int = 10


@dataclass
class LiteratureSearchAgentResponse:
    """Mock response structure containing literature search results."""
    query: str
    results: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Convert response to JSON string."""
        return json.dumps({
            "query": self.query,
            "results": self.results,
            "metadata": self.metadata
        }, indent=2)


@dataclass
class EvaluationResult:
    """Stores evaluation results for a conversation."""
    conversation_id: str
    bleu_scores: List[float]
    rouge_scores: List[Dict[str, float]]
    meteor_scores: List[float]
    avg_bleu: float
    avg_rouge_1: float
    avg_rouge_l: float
    avg_meteor: float
    task_completion_rate: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# BASELINE EVALUATION METRICS
# ============================================================================

class BaselineEvaluator:
    """
    Implements baseline evaluation metrics for multi-turn LLM conversations.

    Based on Section 3 of the survey paper (arXiv:2503.22458), which identifies
    BLEU, ROUGE, and METEOR as the simplest annotation-based metrics to implement.
    """

    def __init__(self, use_stemmer: bool = True):
        """
        Initialize the evaluator.

        Args:
            use_stemmer: Whether to use stemming for ROUGE calculation
        """
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], 
            use_stemmer=use_stemmer
        )
        self.smoothing = SmoothingFunction()

    def evaluate_bleu(
        self, 
        candidate: str, 
        reference: str,
        weights: Tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)
    ) -> float:
        """
        Calculate BLEU score for a single turn.

        BLEU measures n-gram overlap between generated and reference text.
        Widely used in machine translation and dialogue evaluation.

        Args:
            candidate: Generated response from the agent
            reference: Ground truth reference response
            weights: Weights for 1-gram to 4-gram precision (default: equal weights)

        Returns:
            BLEU score between 0 and 1
        """
        reference_tokens = [word_tokenize(reference.lower())]
        candidate_tokens = word_tokenize(candidate.lower())

        # Use smoothing to avoid zero scores
        score = sentence_bleu(
            reference_tokens,
            candidate_tokens,
            weights=weights,
            smoothing_function=self.smoothing.method1
        )
        return score

    def evaluate_rouge(
        self, 
        candidate: str, 
        reference: str
    ) -> Dict[str, float]:
        """
        Calculate ROUGE scores for a single turn.

        ROUGE measures recall-oriented overlap of n-grams and sequences.
        Particularly useful for summarization and dialogue evaluation.

        Args:
            candidate: Generated response from the agent
            reference: Ground truth reference response

        Returns:
            Dictionary containing ROUGE-1, ROUGE-2, and ROUGE-L F1 scores
        """
        scores = self.rouge_scorer.score(reference, candidate)

        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }

    def evaluate_meteor(
        self, 
        candidate: str, 
        reference: str
    ) -> float:
        """
        Calculate METEOR score for a single turn.

        METEOR considers precision, recall, synonyms, and word order.
        More sophisticated than BLEU, better correlation with human judgment.

        Args:
            candidate: Generated response from the agent
            reference: Ground truth reference response

        Returns:
            METEOR score between 0 and 1
        """
        reference_tokens = word_tokenize(reference.lower())
        candidate_tokens = word_tokenize(candidate.lower())

        score = nltk_meteor([reference_tokens], candidate_tokens)
        return score

    def evaluate_conversation(
        self, 
        conversation: MultiTurnConversation
    ) -> EvaluationResult:
        """
        Evaluate a complete multi-turn conversation.

        This method evaluates each turn and aggregates results across
        the entire conversation to provide comprehensive metrics.

        Args:
            conversation: MultiTurnConversation object with turns and references

        Returns:
            EvaluationResult containing per-turn and aggregated scores

        Raises:
            ValueError: If any turn lacks a reference response
        """
        bleu_scores = []
        rouge_scores = []
        meteor_scores = []

        for turn in conversation.turns:
            if turn.reference_response is None:
                raise ValueError(
                    f"Turn {turn.turn_id} missing reference response. "
                    "All turns must have references for baseline evaluation."
                )

            # Calculate metrics for this turn
            bleu = self.evaluate_bleu(
                turn.agent_response, 
                turn.reference_response
            )
            rouge = self.evaluate_rouge(
                turn.agent_response,
                turn.reference_response
            )
            meteor = self.evaluate_meteor(
                turn.agent_response,
                turn.reference_response
            )

            bleu_scores.append(bleu)
            rouge_scores.append(rouge)
            meteor_scores.append(meteor)

        # Aggregate scores across all turns
        return EvaluationResult(
            conversation_id=conversation.conversation_id,
            bleu_scores=bleu_scores,
            rouge_scores=rouge_scores,
            meteor_scores=meteor_scores,
            avg_bleu=np.mean(bleu_scores),
            avg_rouge_1=np.mean([s['rouge1'] for s in rouge_scores]),
            avg_rouge_l=np.mean([s['rougeL'] for s in rouge_scores]),
            avg_meteor=np.mean(meteor_scores),
            task_completion_rate=1.0 if conversation.task_completed else 0.0,
            metadata={
                'num_turns': len(conversation.turns),
                'conversation_length': sum(
                    len(t.agent_response.split()) for t in conversation.turns
                )
            }
        )

    def evaluate_dataset(
        self, 
        conversations: List[MultiTurnConversation]
    ) -> Dict[str, Any]:
        """
        Evaluate a complete dataset of multi-turn conversations.

        Args:
            conversations: List of MultiTurnConversation objects

        Returns:
            Dictionary containing aggregate statistics across the dataset
        """
        results = []
        for conv in conversations:
            result = self.evaluate_conversation(conv)
            results.append(result)

        return {
            'num_conversations': len(conversations),
            'aggregate_metrics': {
                'mean_bleu': np.mean([r.avg_bleu for r in results]),
                'std_bleu': np.std([r.avg_bleu for r in results]),
                'mean_rouge_1': np.mean([r.avg_rouge_1 for r in results]),
                'std_rouge_1': np.std([r.avg_rouge_1 for r in results]),
                'mean_rouge_l': np.mean([r.avg_rouge_l for r in results]),
                'std_rouge_l': np.std([r.avg_rouge_l for r in results]),
                'mean_meteor': np.mean([r.avg_meteor for r in results]),
                'std_meteor': np.std([r.avg_meteor for r in results]),
            },
            'per_conversation_results': results
        }


# ============================================================================
# DATASET LOADER FOR COMMON FORMATS
# ============================================================================

class DatasetLoader:
    """
    Utilities for loading multi-turn conversation datasets.

    Supports formats commonly used in the literature:
    - MT-Bench format (80 multi-turn questions across 8 categories)
    - ToolBench format (tool-use conversations with API calls)
    - Custom JSON format
    """

    @staticmethod
    def load_from_json(filepath: str) -> List[MultiTurnConversation]:
        """
        Load conversations from a JSON file.

        Expected format:
        {
            "conversations": [
                {
                    "conversation_id": "conv_001",
                    "task_description": "Plan a trip to Europe",
                    "turns": [
                        {
                            "turn_id": 1,
                            "user_query": "I want to plan a trip",
                            "agent_response": "Sure! Where would you like to go?",
                            "reference_response": "I'd be happy to help...",
                            "tool_calls": []
                        }
                    ]
                }
            ]
        }

        Args:
            filepath: Path to JSON file

        Returns:
            List of MultiTurnConversation objects
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        conversations = []
        for conv_data in data['conversations']:
            turns = []
            for turn_data in conv_data['turns']:
                turn = ConversationTurn(
                    turn_id=turn_data['turn_id'],
                    user_query=turn_data['user_query'],
                    agent_response=turn_data['agent_response'],
                    reference_response=turn_data.get('reference_response'),
                    tool_calls=turn_data.get('tool_calls', []),
                    context=turn_data.get('context', {}),
                    metadata=turn_data.get('metadata', {})
                )
                turns.append(turn)

            conversation = MultiTurnConversation(
                conversation_id=conv_data['conversation_id'],
                turns=turns,
                task_description=conv_data.get('task_description'),
                task_completed=conv_data.get('task_completed'),
                metadata=conv_data.get('metadata', {})
            )
            conversations.append(conversation)

        return conversations

    @staticmethod
    def create_sample_dataset() -> List[MultiTurnConversation]:
        """
        Create a sample dataset for testing.

        This creates a small dataset mimicking the structure of
        datasets described in the survey paper (MT-Bench style).

        Returns:
            List of sample MultiTurnConversation objects
        """
        conversations = []

        # Sample Conversation 1: Travel Planning (from Figure 1 in paper)
        conv1 = MultiTurnConversation(
            conversation_id="sample_001",
            task_description="Plan a trip to Europe",
            task_completed=True,
            turns=[
                ConversationTurn(
                    turn_id=1,
                    user_query="Hi, I want to plan a trip to Europe.",
                    agent_response="Sure. Here are some amazing destinations to choose from: 1. Paris, France 2. Rome, Italy 3. Barcelona, Spain 4. Amsterdam, Netherlands. Which destination would you like to explore first?",
                    reference_response="I'd be happy to help you plan your European trip! Europe offers incredible destinations. Some popular choices include Paris for art and culture, Rome for history, Barcelona for architecture, and Amsterdam for its canals. Which of these interests you most?"
                ),
                ConversationTurn(
                    turn_id=2,
                    user_query="First.",
                    agent_response="Great choice! Paris is an incredible city. When would you like to go?",
                    reference_response="Excellent! Paris is a wonderful choice. The city of lights offers world-class museums, cuisine, and architecture. What dates are you considering for your visit?"
                ),
                ConversationTurn(
                    turn_id=3,
                    user_query="March 15th to March 20th. Please help with accommodations and an itinerary.",
                    agent_response="Here are some excellent boutique hotel options in Paris for your stay: 1. Hotel de la Paix 2. Le Petit Palace 3. Maison Souquet. Which one would you like to book?",
                    reference_response="For March 15-20, I recommend booking accommodations in central Paris. Here are some options: 1. Hotel de la Paix in the Latin Quarter 2. Le Petit Palace in Le Marais 3. Maison Souquet near Montmartre. All offer great locations for sightseeing. Would you like me to help create a day-by-day itinerary as well?"
                )
            ]
        )
        conversations.append(conv1)

        # Sample Conversation 2: Coding Assistant
        conv2 = MultiTurnConversation(
            conversation_id="sample_002",
            task_description="Write and debug Python code",
            task_completed=True,
            turns=[
                ConversationTurn(
                    turn_id=1,
                    user_query="Can you help me write a function to calculate fibonacci numbers?",
                    agent_response="Sure! Here's a recursive implementation: def fib(n): return n if n <= 1 else fib(n-1) + fib(n-2)",
                    reference_response="I can help you write a Fibonacci function. Here's a clean recursive implementation: def fibonacci(n): if n <= 1: return n return fibonacci(n-1) + fibonacci(n-2). Would you like an iterative version for better performance?"
                ),
                ConversationTurn(
                    turn_id=2,
                    user_query="Yes, please show me an iterative version.",
                    agent_response="Here's an iterative version: def fib(n): a, b = 0, 1\n for _ in range(n): a, b = b, a+b\n return a",
                    reference_response="Great question! Here's an efficient iterative version: def fibonacci(n): if n <= 1: return n\n a, b = 0, 1\n for i in range(2, n+1): a, b = b, a + b\n return b. This runs in O(n) time and O(1) space."
                )
            ]
        )
        conversations.append(conv2)

        return conversations


# ============================================================================
# INTEGRATION WITH LITERATURE SEARCH AGENT
# ============================================================================

class LiteratureSearchEvaluationPipeline:
    """
    Evaluation pipeline that integrates with a literature search agent.

    This demonstrates how to use the literature_search_response output
    (as JSON) to evaluate conversations about literature search tasks.
    """

    def __init__(self, evaluator: BaselineEvaluator):
        """
        Initialize the pipeline.

        Args:
            evaluator: BaselineEvaluator instance
        """
        self.evaluator = evaluator

    async def aforward(
        self, 
        request: LiteratureSearchAgentRequest
    ) -> LiteratureSearchAgentResponse:
        """
        Simulate an async literature search agent call.

        In production, this would be replaced with your actual
        literature_search_agent.aforward() method.

        Args:
            request: LiteratureSearchAgentRequest with query parameters

        Returns:
            LiteratureSearchAgentResponse with search results as JSON
        """
        # Simulate literature search results
        mock_results = [
            {
                "title": "Evaluating LLM-based Agents for Multi-Turn Conversations",
                "authors": ["Guan, S.", "Xiong, H.", "Wang, J."],
                "year": 2025,
                "venue": "arXiv",
                "summary": "Survey of evaluation methods for multi-turn LLM agents"
            },
            {
                "title": "MT-Bench: A Multi-Turn Conversation Benchmark",
                "authors": ["Zheng, L.", "Chiang, W."],
                "year": 2023,
                "venue": "NeurIPS",
                "summary": "Benchmark with 80 multi-turn questions across 8 categories"
            }
        ]

        response = LiteratureSearchAgentResponse(
            query=request.topic,
            results=mock_results[:request.max_results],
            metadata={"total_found": len(mock_results)}
        )

        return response

    def create_conversation_from_search(
        self,
        search_response: LiteratureSearchAgentResponse,
        user_queries: List[str],
        agent_responses: List[str],
        reference_responses: List[str]
    ) -> MultiTurnConversation:
        """
        Create a multi-turn conversation from literature search results.

        Args:
            search_response: Output from literature_search_agent.aforward()
            user_queries: List of user questions in the conversation
            agent_responses: List of agent responses
            reference_responses: List of reference (ground truth) responses

        Returns:
            MultiTurnConversation object ready for evaluation
        """
        turns = []
        for i, (query, response, ref) in enumerate(
            zip(user_queries, agent_responses, reference_responses), 
            start=1
        ):
            turn = ConversationTurn(
                turn_id=i,
                user_query=query,
                agent_response=response,
                reference_response=ref,
                metadata={"search_results": search_response.results}
            )
            turns.append(turn)

        conversation = MultiTurnConversation(
            conversation_id=f"lit_search_{search_response.query}",
            turns=turns,
            task_description=f"Literature search on: {search_response.query}",
            metadata={"search_response_json": search_response.to_json()}
        )

        return conversation

    async def evaluate_literature_search(
        self,
        topic: str,
        user_queries: List[str],
        agent_responses: List[str],
        reference_responses: List[str]
    ) -> EvaluationResult:
        """
        Complete pipeline: search literature and evaluate conversation.

        Args:
            topic: Research topic to search
            user_queries: User's questions in the conversation
            agent_responses: Agent's generated responses
            reference_responses: Ground truth reference responses

        Returns:
            EvaluationResult with all metrics
        """
        # Step 1: Call literature search agent
        request = LiteratureSearchAgentRequest(topic=topic)
        search_response = await self.aforward(request)

        # Step 2: Create conversation from results
        conversation = self.create_conversation_from_search(
            search_response,
            user_queries,
            agent_responses,
            reference_responses
        )

        # Step 3: Evaluate the conversation
        result = self.evaluator.evaluate_conversation(conversation)

        return result


# ============================================================================
# EXAMPLE USAGE AND DEMONSTRATION
# ============================================================================

def demonstrate_baseline_evaluation():
    """
    Demonstrate the baseline evaluation pipeline.
    """
    print("=" * 80)
    print("Baseline Multi-Turn LLM Conversation Evaluation")
    print("Based on: arXiv:2503.22458")
    print("=" * 80)
    print()

    # Initialize evaluator
    evaluator = BaselineEvaluator()

    # Create sample dataset
    print("Creating sample dataset...")
    conversations = DatasetLoader.create_sample_dataset()
    print(f"Loaded {len(conversations)} sample conversations")
    print()

    # Evaluate individual conversation
    print("Evaluating individual conversation:")
    print("-" * 80)
    result = evaluator.evaluate_conversation(conversations[0])

    print(f"Conversation ID: {result.conversation_id}")
    print(f"Number of turns: {result.metadata['num_turns']}")
    print()
    print("Per-turn scores:")
    for i, (bleu, rouge, meteor) in enumerate(
        zip(result.bleu_scores, result.rouge_scores, result.meteor_scores), 
        start=1
    ):
        print(f"  Turn {i}:")
        print(f"    BLEU:   {bleu:.4f}")
        print(f"    ROUGE-1: {rouge['rouge1']:.4f}")
        print(f"    ROUGE-L: {rouge['rougeL']:.4f}")
        print(f"    METEOR: {meteor:.4f}")

    print()
    print("Average scores across conversation:")
    print(f"  Avg BLEU:   {result.avg_bleu:.4f}")
    print(f"  Avg ROUGE-1: {result.avg_rouge_1:.4f}")
    print(f"  Avg ROUGE-L: {result.avg_rouge_l:.4f}")
    print(f"  Avg METEOR: {result.avg_meteor:.4f}")
    print()

    # Evaluate entire dataset
    print("=" * 80)
    print("Evaluating entire dataset:")
    print("-" * 80)
    dataset_results = evaluator.evaluate_dataset(conversations)

    print(f"Total conversations: {dataset_results['num_conversations']}")
    print()
    print("Aggregate metrics:")
    metrics = dataset_results['aggregate_metrics']
    print(f"  Mean BLEU:   {metrics['mean_bleu']:.4f} (±{metrics['std_bleu']:.4f})")
    print(f"  Mean ROUGE-1: {metrics['mean_rouge_1']:.4f} (±{metrics['std_rouge_1']:.4f})")
    print(f"  Mean ROUGE-L: {metrics['mean_rouge_l']:.4f} (±{metrics['std_rouge_l']:.4f})")
    print(f"  Mean METEOR: {metrics['mean_meteor']:.4f} (±{metrics['std_meteor']:.4f})")
    print()

    # Demonstrate literature search integration
    print("=" * 80)
    print("Literature Search Integration Example:")
    print("-" * 80)

    async def run_literature_search_example():
        pipeline = LiteratureSearchEvaluationPipeline(evaluator)

        # Simulate a multi-turn conversation about literature search
        result = await pipeline.evaluate_literature_search(
            topic="multi-turn dialogue evaluation",
            user_queries=[
                "What are the main evaluation metrics for multi-turn conversations?",
                "How does BLEU differ from ROUGE?"
            ],
            agent_responses=[
                "The main metrics are BLEU, ROUGE, and METEOR for baseline evaluation.",
                "BLEU focuses on precision of n-grams, while ROUGE focuses on recall."
            ],
            reference_responses=[
                "The primary evaluation metrics for multi-turn conversations include BLEU for precision, ROUGE for recall, and METEOR for semantic similarity.",
                "BLEU measures precision-oriented n-gram overlap, while ROUGE is recall-oriented and better for summarization tasks."
            ]
        )

        print(f"Evaluated literature search conversation:")
        print(f"  Avg BLEU: {result.avg_bleu:.4f}")
        print(f"  Avg METEOR: {result.avg_meteor:.4f}")

    # Run async example
    asyncio.run(run_literature_search_example())

    print()
    print("=" * 80)
    print("Evaluation complete!")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_baseline_evaluation()
