"""Utility functions for document processing and citation management.
"""

import re
from typing import List, Tuple, Dict
from src.dataclass import RetrievedDocument
import re
import string
import numpy as np
from typing import List, Dict, Union
import statistics
from collections import defaultdict
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score
import nltk
from nltk.translate.meteor_score import meteor_score
from sentence_transformers import SentenceTransformer
import logging
from dataclasses import dataclass
from pathlib import Path
from openai import OpenAI
from src.load_dataset import load_locomo_dataset, QA, Turn, Session, Conversation
from sentence_transformers.util import pytorch_cos_sim

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
except Exception as e:
    print(f"Error downloading NLTK data: {e}")

# Initialize SentenceTransformer model (this will be reused)
try:
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print(f"Warning: Could not load SentenceTransformer model: {e}")
    sentence_model = None

def simple_tokenize(text):
    """Simple tokenization function."""
    # Convert to string if not already
    text = str(text)
    return text.lower().replace('.', ' ').replace(',', ' ').replace('!', ' ').replace('?', ' ').split()

def calculate_rouge_scores(prediction: str, reference: str) -> Dict[str, float]:
    """Calculate ROUGE scores for prediction against reference."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return {
        'rouge1_f': scores['rouge1'].fmeasure,
        'rouge2_f': scores['rouge2'].fmeasure,
        'rougeL_f': scores['rougeL'].fmeasure
    }

def calculate_bleu_scores(prediction: str, reference: str) -> Dict[str, float]:
    """Calculate BLEU scores with different n-gram settings."""
    pred_tokens = nltk.word_tokenize(prediction.lower())
    ref_tokens = [nltk.word_tokenize(reference.lower())]
    
    weights_list = [(1, 0, 0, 0), (0.5, 0.5, 0, 0), (0.33, 0.33, 0.33, 0), (0.25, 0.25, 0.25, 0.25)]
    smooth = SmoothingFunction().method1
    
    scores = {}
    for n, weights in enumerate(weights_list, start=1):
        try:
            score = sentence_bleu(ref_tokens, pred_tokens, weights=weights, smoothing_function=smooth)
        except Exception:
            score = 0.0
        scores[f'bleu{n}'] = score
    
    return scores

def calculate_bert_scores(prediction: str, reference: str) -> Dict[str, float]:
    """Calculate BERTScore for semantic similarity."""
    try:
        P, R, F1 = bert_score([prediction], [reference], lang='en', verbose=False)
        return {
            'bert_precision': P.item(),
            'bert_recall': R.item(),
            'bert_f1': F1.item()
        }
    except Exception as e:
        print(f"Error calculating BERTScore: {e}")
        return {
            'bert_precision': 0.0,
            'bert_recall': 0.0,
            'bert_f1': 0.0
        }

def calculate_meteor_score(prediction: str, reference: str) -> float:
    """Calculate METEOR score for the prediction."""
    try:
        return meteor_score([reference.split()], prediction.split())
    except Exception as e:
        print(f"Error calculating METEOR score: {e}")
        return 0.0

def calculate_sentence_similarity(prediction: str, reference: str) -> float:
    """Calculate sentence embedding similarity using SentenceBERT."""
    if sentence_model is None:
        return 0.0
    try:
        # Encode sentences
        embedding1 = sentence_model.encode([prediction], convert_to_tensor=True)
        embedding2 = sentence_model.encode([reference], convert_to_tensor=True)
        
        # Calculate cosine similarity
        similarity = pytorch_cos_sim(embedding1, embedding2).item()
        return float(similarity)
    except Exception as e:
        print(f"Error calculating sentence similarity: {e}")
        return 0.0

def calculate_metrics(prediction: str, reference: str) -> Dict[str, float]:
    """Calculate comprehensive evaluation metrics for a prediction."""
    # Handle empty or None values
    if not prediction or not reference:
        return {
            "exact_match": 0,
            "f1": 0.0,
            "rouge1_f": 0.0,
            "rouge2_f": 0.0,
            "rougeL_f": 0.0,
            "bleu1": 0.0,
            "bleu2": 0.0,
            "bleu3": 0.0,
            "bleu4": 0.0,
            "bert_f1": 0.0,
            "meteor": 0.0,
            "sbert_similarity": 0.0
        }
    
    # Convert to strings if they're not already
    prediction = str(prediction).strip()
    reference = str(reference).strip()
    
    # Calculate exact match
    exact_match = int(prediction.lower() == reference.lower())
    
    # Calculate token-based F1 score
    pred_tokens = set(simple_tokenize(prediction))
    ref_tokens = set(simple_tokenize(reference))
    common_tokens = pred_tokens & ref_tokens
    
    if not pred_tokens or not ref_tokens:
        f1 = 0.0
    else:
        precision = len(common_tokens) / len(pred_tokens)
        recall = len(common_tokens) / len(ref_tokens)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Calculate all scores
    rouge_scores = calculate_rouge_scores(prediction, reference)
    bleu_scores = calculate_bleu_scores(prediction, reference)
    bert_scores = calculate_bert_scores(prediction, reference)
    meteor = calculate_meteor_score(prediction, reference)
    sbert_similarity = calculate_sentence_similarity(prediction, reference)
    
    # Combine all metrics
    metrics = {
        "exact_match": exact_match,
        "f1": f1,
        **rouge_scores,
        **bleu_scores,
        **bert_scores,
        "meteor": meteor,
        "sbert_similarity": sbert_similarity
    }
    
    return metrics

def aggregate_metrics(all_metrics: List[Dict[str, float]], all_categories: List[int]) -> Dict[str, Dict[str, Union[float, Dict[str, float]]]]:
    """Calculate aggregate statistics for all metrics, split by category."""
    if not all_metrics:
        return {}
    
    # Initialize aggregates for overall and per-category metrics
    aggregates = defaultdict(list)
    category_aggregates = defaultdict(lambda: defaultdict(list))
    
    # Collect all values for each metric, both overall and per category
    for metrics, category in zip(all_metrics, all_categories):
        for metric_name, value in metrics.items():
            aggregates[metric_name].append(value)
            category_aggregates[category][metric_name].append(value)
    
    # Calculate statistics for overall metrics
    results = {
        "overall": {}
    }
    
    for metric_name, values in aggregates.items():
        results["overall"][metric_name] = {
            'mean': statistics.mean(values),
            'std': statistics.stdev(values) if len(values) > 1 else 0.0,
            'median': statistics.median(values),
            'min': min(values),
            'max': max(values),
            'count': len(values)
        }
    
    # Calculate statistics for each category
    for category in sorted(category_aggregates.keys()):
        results[f"category_{category}"] = {}
        for metric_name, values in category_aggregates[category].items():
            if values:  # Only calculate if we have values for this category
                results[f"category_{category}"][metric_name] = {
                    'mean': statistics.mean(values),
                    'std': statistics.stdev(values) if len(values) > 1 else 0.0,
                    'median': statistics.median(values),
                    'min': min(values),
                    'max': max(values),
                    'count': len(values)
                }
    
    return results



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
