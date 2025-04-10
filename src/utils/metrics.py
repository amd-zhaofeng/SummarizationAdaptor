import torch
import numpy as np
from typing import Dict, List


def calculate_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Calculate ROUGE scores"""
    # Directly use rouge_score library
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    scores: Dict[str, List[float]] = {
        'rouge1': [],
        'rouge2': [],
        'rougeL': []
    }

    for pred, ref in zip(predictions, references):
        score = scorer.score(ref, pred)
        for key in scores.keys():
            scores[key].append(score[key].fmeasure)

    results: Dict[str, float] = {
        'rouge1': float(np.mean(scores['rouge1']) * 100),
        'rouge2': float(np.mean(scores['rouge2']) * 100),
        'rougeL': float(np.mean(scores['rougeL']) * 100)
    }

    # Convert to readable format
    results = {k: round(v, 2) for k, v in results.items()}
    return results


def calculate_bert_score(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Calculate BERTScore"""
    # Directly use bert_score library
    from bert_score import score
    P, R, F1 = score(predictions, references, lang="en", verbose=True)

    # Extract and format results - ensure we're working with tensor objects
    # Cast the results to tensor if they aren't already
    P_tensor = P if isinstance(P, torch.Tensor) else torch.tensor(P)
    R_tensor = R if isinstance(R, torch.Tensor) else torch.tensor(R)
    F1_tensor = F1 if isinstance(F1, torch.Tensor) else torch.tensor(F1)

    precision = float(torch.mean(P_tensor).item() * 100)
    recall = float(torch.mean(R_tensor).item() * 100)
    f1 = float(torch.mean(F1_tensor).item() * 100)

    return {
        "bertscore_precision": round(precision, 2),
        "bertscore_recall": round(recall, 2),
        "bertscore_f1": round(f1, 2)
    }


def calculate_length_ratio(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Calculate summary length ratio"""
    pred_lengths = [len(p.split()) for p in predictions]
    ref_lengths = [len(r.split()) for r in references]

    # Calculate average length ratio
    ratios = [pred_len / ref_len if ref_len > 0 else 0 for pred_len, ref_len in zip(pred_lengths, ref_lengths)]
    avg_ratio = float(np.mean(ratios) * 100)

    return {"length_ratio": round(avg_ratio, 2)}


def calculate_metrics(predictions, references, lang="en"):
    """
    Calculate all evaluation metrics

    Parameters:
        predictions: List of predicted summaries
        references: List of reference summaries
        lang: Language code

    Returns:
        Dictionary of all metrics
    """
    # Calculate metrics
    metrics: Dict[str, float] = {}

    # ROUGE scores
    rouge_scores = calculate_rouge(predictions, references)
    metrics.update(rouge_scores)

    # BERTScore
    bert_scores = calculate_bert_score(predictions, references)
    metrics.update(bert_scores)

    # Length ratio
    length_metrics = calculate_length_ratio(predictions, references)
    metrics.update(length_metrics)

    return metrics
