import os
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import seaborn as sns
import argparse
import json
import re
from data.prompt import DATASET_PROMPT_TEMPLATES, DEFAULT_PROMPT_TEMPLATE

from model.processor import get_tokenizer


def analyze_long_doc_dataset(dataset_name, tokenizer_name, sample_size, output_dir):
    """
    Analyze statistics of long document summarization datasets

    Parameters:
        dataset_name: Dataset name ('ccdv/pubmed-summarization', 'ccdv/arxiv-summarization', 'ccdv/govreport-summarization', 'billsum', 'kmfoda/booksum')
        tokenizer_name: Name of the tokenizer to use
        sample_size: Sample size, if 0 or negative then analyze the entire dataset
        output_dir: Directory to save analysis results
    """
    print(f"Loading dataset: {dataset_name}")

    # Set field mapping
    field_mapping = {
        "ccdv/pubmed-summarization": {"text": "article", "summary": "abstract"},
        "ccdv/arxiv-summarization": {"text": "article", "summary": "abstract"},
        "ccdv/govreport-summarization": {"text": "report", "summary": "summary"},
        "billsum": {"text": "text", "summary": "summary"},
        "kmfoda/booksum": {"text": "chapter_text", "summary": "summary"}
    }

    # Ensure dataset is supported
    if dataset_name not in field_mapping:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Supported datasets: {list(field_mapping.keys())}")

    # Load dataset
    try:
        raw_datasets = load_dataset(dataset_name)
        print(f"Successfully loaded dataset: {dataset_name}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None

    print(f"Dataset size - Train: {len(raw_datasets['train'])}, Validation: {len(raw_datasets['validation'])}, Test: {len(raw_datasets['test'])}")

    # Load tokenizer
    try:
        tokenizer = get_tokenizer(tokenizer_name)
        print(f"Successfully loaded tokenizer: {tokenizer_name}")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return None, None

    # Create save directory
    os.makedirs(output_dir, exist_ok=True)

    # Set field names
    text_field = field_mapping[dataset_name]["text"]
    summary_field = field_mapping[dataset_name]["summary"]
    print(f"Using field mapping - Text field: '{text_field}', Summary field: '{summary_field}'")

    # Clean and filter dataset
    print("Cleaning dataset...")
    cleaned_dataset = raw_datasets['train'].filter(
        lambda x: len(x[text_field].strip()) > 0 and len(x[summary_field].strip()) > 0
    )
    print(f"After cleaning: {len(cleaned_dataset)} samples (removed {len(raw_datasets['train']) - len(cleaned_dataset)} empty samples)")

    # Select samples for analysis
    if sample_size <= 0:
        print(f"Sample size is {sample_size}, using all available samples ({len(cleaned_dataset)})")
        merged_dataset = cleaned_dataset
    else:
        merged_dataset = cleaned_dataset.select(range(min(sample_size, len(cleaned_dataset))))
        print(f"Using {len(merged_dataset)} samples from training set for analysis")

    # Check if dataset is empty
    if len(merged_dataset) == 0:
        print("Error: Dataset is empty after cleaning. Please check the dataset and field mapping.")
        return None

    # Collect statistics
    text_lengths = []  # Source text character length
    summary_lengths = []  # Summary character length
    text_token_lengths = []  # Source text token length
    summary_token_lengths = []  # Summary token length
    text_tokens_with_prompt = []  # Source text token length with prompt

    # Use prompt templates from prompt module
    prompt_template = DATASET_PROMPT_TEMPLATES.get(dataset_name, DEFAULT_PROMPT_TEMPLATE)

    for item in tqdm(merged_dataset, desc="Processing data"):
        text = item[text_field]
        summary = item[summary_field]

        # Character length
        text_lengths.append(len(text))
        summary_lengths.append(len(summary))

        # Token length
        text_tokens = tokenizer(text, return_length=True)["length"]
        summary_tokens = tokenizer(summary, return_length=True)["length"]

        # Handle case where tokens are lists
        if isinstance(text_tokens, list):
            text_tokens = text_tokens[0]
        if isinstance(summary_tokens, list):
            summary_tokens = summary_tokens[0]

        text_token_lengths.append(text_tokens)
        summary_token_lengths.append(summary_tokens)

        # Source text token length with prompt
        prompt = prompt_template.format(text=text)
        prompt_tokens = tokenizer(prompt, return_length=True)["length"]
        if isinstance(prompt_tokens, list):
            prompt_tokens = prompt_tokens[0]
        text_tokens_with_prompt.append(prompt_tokens)

    # Calculate statistics
    stats = {
        "text_char_length": {
            "min": min(text_lengths),
            "max": max(text_lengths),
            "mean": np.mean(text_lengths),
            "median": np.median(text_lengths),
            "p90": np.percentile(text_lengths, 90),
            "p95": np.percentile(text_lengths, 95),
            "p99": np.percentile(text_lengths, 99),
        },
        "summary_char_length": {
            "min": min(summary_lengths),
            "max": max(summary_lengths),
            "mean": np.mean(summary_lengths),
            "median": np.median(summary_lengths),
            "p90": np.percentile(summary_lengths, 90),
            "p95": np.percentile(summary_lengths, 95),
            "p99": np.percentile(summary_lengths, 99),
        },
        "text_token_length": {
            "min": min(text_token_lengths),
            "max": max(text_token_lengths),
            "mean": np.mean(text_token_lengths),
            "median": np.median(text_token_lengths),
            "p90": np.percentile(text_token_lengths, 90),
            "p95": np.percentile(text_token_lengths, 95),
            "p99": np.percentile(text_token_lengths, 99),
        },
        "summary_token_length": {
            "min": min(summary_token_lengths),
            "max": max(summary_token_lengths),
            "mean": np.mean(summary_token_lengths),
            "median": np.median(summary_token_lengths),
            "p90": np.percentile(summary_token_lengths, 90),
            "p95": np.percentile(summary_token_lengths, 95),
            "p99": np.percentile(summary_token_lengths, 99),
        },
        "text_tokens_with_prompt": {
            "min": min(text_tokens_with_prompt),
            "max": max(text_tokens_with_prompt),
            "mean": np.mean(text_tokens_with_prompt),
            "median": np.median(text_tokens_with_prompt),
            "p90": np.percentile(text_tokens_with_prompt, 90),
            "p95": np.percentile(text_tokens_with_prompt, 95),
            "p99": np.percentile(text_tokens_with_prompt, 99),
        },
        "token_length_distribution": {
            "0-1k": len([x for x in text_token_lengths if x < 1000]),
            "1k-2k": len([x for x in text_token_lengths if 1000 <= x < 2000]),
            "2k-4k": len([x for x in text_token_lengths if 2000 <= x < 4000]),
            "4k-8k": len([x for x in text_token_lengths if 4000 <= x < 8000]),
            "8k-16k": len([x for x in text_token_lengths if 8000 <= x < 16000]),
            "16k+": len([x for x in text_token_lengths if x >= 16000]),
        },
        "raw_data": {
            "text_lengths": text_lengths,
            "summary_lengths": summary_lengths,
            "text_token_lengths": text_token_lengths,
            "summary_token_lengths": summary_token_lengths,
            "text_tokens_with_prompt": text_tokens_with_prompt
        }
    }

    # Calculate percentage for each range
    total_samples = len(text_token_lengths)
    stats["token_length_distribution_percent"] = {
        k: v / total_samples * 100 for k, v in stats["token_length_distribution"].items()
    }

    # Calculate compression ratio
    compression_ratio = stats["text_token_length"]["mean"] / stats["summary_token_length"]["mean"]
    stats["compression_ratio"] = compression_ratio

    # Print statistics
    print("\nDataset Statistics:")
    print(f"Source text character length: Mean {stats['text_char_length']['mean']:.2f}, Median {stats['text_char_length']['median']:.2f}, Max {stats['text_char_length']['max']}, P95 {stats['text_char_length']['p95']:.2f}")
    print(f"Summary character length: Mean {stats['summary_char_length']['mean']:.2f}, Median {stats['summary_char_length']['median']:.2f}, Max {stats['summary_char_length']['max']}, P95 {stats['summary_char_length']['p95']:.2f}")
    print(f"Source text token length: Mean {stats['text_token_length']['mean']:.2f}, Median {stats['text_token_length']['median']:.2f}, Max {stats['text_token_length']['max']}, P95 {stats['text_token_length']['p95']:.2f}")
    print(f"Summary token length: Mean {stats['summary_token_length']['mean']:.2f}, Median {stats['summary_token_length']['median']:.2f}, Max {stats['summary_token_length']['max']}, P95 {stats['summary_token_length']['p95']:.2f}")
    print(f"Compression ratio: {compression_ratio:.2f}x")

    print("\nToken length distribution:")
    for k, v in stats["token_length_distribution"].items():
        print(f"  {k}: {v} ({stats['token_length_distribution_percent'][k]:.2f}%)")

    # Draw visualization charts
    draw_charts = True
    if draw_charts:
        # Create charts
        plt.figure(figsize=(15, 10))

        # Document length distribution
        plt.subplot(2, 2, 1)
        sns.histplot(text_token_lengths, bins=20, kde=True)
        plt.title(f"{dataset_name} - Document Length Distribution")
        plt.xlabel("Number of Tokens")
        plt.ylabel("Number of Documents")
        plt.axvline(x=stats['text_token_length']['p95'], color='r', linestyle='--', label=f'P95: {stats["text_token_length"]["p95"]:.0f}')
        plt.axvline(x=stats['text_token_length']['median'], color='g', linestyle='--', label=f'Median: {stats["text_token_length"]["median"]:.0f}')
        plt.legend()

        # Summary length distribution
        plt.subplot(2, 2, 2)
        sns.histplot(summary_token_lengths, bins=20, kde=True)
        plt.title(f"{dataset_name} - Summary Length Distribution")
        plt.xlabel("Number of Tokens")
        plt.ylabel("Number of Summaries")
        plt.axvline(x=stats['summary_token_length']['p95'], color='r', linestyle='--', label=f'P95: {stats["summary_token_length"]["p95"]:.0f}')
        plt.axvline(x=stats['summary_token_length']['median'], color='g', linestyle='--', label=f'Median: {stats["summary_token_length"]["median"]:.0f}')
        plt.legend()

        # Document length range distribution
        plt.subplot(2, 2, 3)
        categories = list(stats["token_length_distribution"].keys())
        values = list(stats["token_length_distribution"].values())
        plt.bar(categories, values)
        plt.title(f"{dataset_name} - Document Length Range Distribution")
        plt.xlabel("Token Length Range")
        plt.ylabel("Number of Documents")
        plt.xticks(rotation=45)

        # Document vs Summary length scatter plot
        plt.subplot(2, 2, 4)
        plt.scatter(text_token_lengths, summary_token_lengths, alpha=0.3)
        plt.title(f"{dataset_name} - Document vs Summary Length")
        plt.xlabel("Document Length (tokens)")
        plt.ylabel("Summary Length (tokens)")

        plt.tight_layout()
        img_path = os.path.join(output_dir, f"{dataset_name.replace('/', '_')}_length_distribution.png")
        plt.savefig(img_path)
        print(f"Saved charts to {img_path}")

    # Evaluate dataset quality
    quality_metrics = evaluate_dataset_quality(merged_dataset, text_field, summary_field, tokenizer, sample_size)

    # Add quality metrics to stats
    stats["quality_metrics"] = quality_metrics

    # Generate parameter suggestions
    parameter_suggestions = generate_parameter_suggestions(stats)
    stats["parameter_suggestions"] = parameter_suggestions

    # Print parameter suggestions
    print_parameter_suggestions(parameter_suggestions)

    # Save statistics
    stats_path = os.path.join(output_dir, f"{dataset_name.replace('/', '_')}_statistics.json")
    with open(stats_path, 'w') as f:
        # Remove raw_data to reduce file size
        stats_copy = stats.copy()
        stats_copy.pop('raw_data', None)
        json.dump(stats_copy, f, indent=2)

    # Create a recommended config file
    config = {
        "dataset": dataset_name,
        "tokenizer": tokenizer_name,
        "training": {
            "max_length": parameter_suggestions["max_length"],
            "summary_max_length": parameter_suggestions["max_new_tokens"]
        },
        "inference": {
            "max_new_tokens": parameter_suggestions["max_new_tokens"],
            "temperature": parameter_suggestions["temperature"],
            "top_p": parameter_suggestions["top_p"]
        }
    }

    config_path = os.path.join(output_dir, f"{dataset_name.replace('/', '_')}_recommended_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Saved statistics to {stats_path}")
    print(f"Saved recommended config to {config_path}")

    return stats


def evaluate_dataset_quality(dataset, text_field, summary_field, tokenizer, sample_size=1000):
    """
    Evaluate the quality of the dataset using simple metrics without external libraries

    Parameters:
        dataset: Dataset to evaluate
        text_field: Field name for source text
        summary_field: Field name for summary
        tokenizer: Tokenizer to use
        sample_size: Number of samples to evaluate

    Returns:
        dict: Dictionary containing quality metrics
    """
    print("\nEvaluating dataset quality...")

    # Select samples for evaluation
    eval_dataset = dataset.select(range(min(sample_size, len(dataset))))

    # Initialize metrics
    quality_metrics = {
        "text_quality": {
            "special_char_ratio": 0.0,
            "avg_sentence_length": 0.0,
            "sentence_count": 0.0,
            "word_diversity": 0.0
        },
        "summary_quality": {
            "summary_completeness": 0.0,
            "summary_uniqueness": 0.0
        },
        "dataset_balance": {
            "length_std": 0.0
        }
    }

    # Text quality metrics
    text_lengths = []
    special_char_counts = []
    sentence_lengths = []
    sentence_counts = []
    word_counts = []
    unique_word_counts = []

    # Summary quality metrics
    summary_lengths = []
    summary_word_counts = []
    summary_unique_word_counts = []

    # Process each sample
    for item in tqdm(eval_dataset, desc="Evaluating samples"):
        text = item[text_field].strip()
        summary = item[summary_field].strip()

        # Skip empty texts or summaries
        if not text or not summary:
            continue

        # Text length
        text_lengths.append(len(text))

        # Special characters
        special_chars = len(re.findall(r'[^a-zA-Z0-9\s]', text))
        special_char_counts.append(special_chars / len(text) if len(text) > 0 else 0)

        # Sentence analysis
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        sentence_counts.append(len(sentences))
        sentence_lengths.extend([len(s) for s in sentences])

        # Word analysis for text
        words = re.findall(r'\b\w+\b', text.lower())
        word_counts.append(len(words))
        unique_word_counts.append(len(set(words)))

        # Summary length and word analysis
        summary_lengths.append(len(summary))
        summary_words = re.findall(r'\b\w+\b', summary.lower())
        summary_word_counts.append(len(summary_words))
        summary_unique_word_counts.append(len(set(summary_words)))

    # Skip if no valid samples
    if not text_lengths:
        print("Warning: No valid samples found for quality evaluation")
        return quality_metrics

    # Calculate text quality metrics
    quality_metrics["text_quality"]["special_char_ratio"] = np.mean(special_char_counts)
    quality_metrics["text_quality"]["avg_sentence_length"] = np.mean(sentence_lengths) if sentence_lengths else 0
    quality_metrics["text_quality"]["sentence_count"] = np.mean(sentence_counts)
    quality_metrics["text_quality"]["word_diversity"] = np.mean([u / w for u, w in zip(unique_word_counts, word_counts) if w > 0])

    # Calculate summary quality metrics
    quality_metrics["summary_quality"]["summary_completeness"] = np.mean(summary_lengths) / np.mean(text_lengths) if text_lengths else 0
    quality_metrics["summary_quality"]["summary_uniqueness"] = np.mean([u / w for u, w in zip(summary_unique_word_counts, summary_word_counts) if w > 0])

    # Calculate dataset balance metrics
    quality_metrics["dataset_balance"]["length_std"] = np.std(text_lengths) / np.mean(text_lengths) if text_lengths else 0

    # Print quality metrics
    print("\nDataset Quality Metrics:")
    print("\nText Quality:")
    print(f"Special Character Ratio: {quality_metrics['text_quality']['special_char_ratio']:.2%}")
    print(f"Average Sentence Length: {quality_metrics['text_quality']['avg_sentence_length']:.1f} characters")
    print(f"Average Sentence Count: {quality_metrics['text_quality']['sentence_count']:.1f}")
    print(f"Word Diversity: {quality_metrics['text_quality']['word_diversity']:.2%}")

    print("\nSummary Quality:")
    print(f"Summary Completeness: {quality_metrics['summary_quality']['summary_completeness']:.2%}")
    print(f"Summary Uniqueness: {quality_metrics['summary_quality']['summary_uniqueness']:.2%}")

    print("\nDataset Balance:")
    print(f"Length Standard Deviation: {quality_metrics['dataset_balance']['length_std']:.2f}")

    return quality_metrics


def generate_parameter_suggestions(stats):
    """
    Generate suggested parameters for training and inference based on dataset statistics

    Parameters:
        stats: Dictionary containing dataset statistics

    Returns:
        dict: Dictionary containing suggested parameters
    """
    # Calculate suggested parameters

    # Suggested max_length (input length with prompt)
    # Add 50 tokens buffer to the 95th percentile for safety
    suggested_max_length = int(stats['text_tokens_with_prompt']['p95']) + 50

    # Suggested max_new_tokens (maximum generation length)
    # Add 20 tokens buffer to the 95th percentile for safety
    suggested_max_new_tokens = int(stats['summary_token_length']['p95']) + 20

    # For very long documents, enforce context window limits
    context_window_limits = {
        "8k": 8192,
        "16k": 16384,
        "32k": 32768,
        "128k": 131072
    }

    # Choose appropriate context window based on document length
    if suggested_max_length <= context_window_limits["8k"]:
        recommended_window = "8k"
        context_limit = context_window_limits["8k"]
    elif suggested_max_length <= context_window_limits["16k"]:
        recommended_window = "16k"
        context_limit = context_window_limits["16k"]
    elif suggested_max_length <= context_window_limits["32k"]:
        recommended_window = "32k"
        context_limit = context_window_limits["32k"]
    else:
        recommended_window = "128k"
        context_limit = context_window_limits["128k"]

    # Ensure max_length doesn't exceed context window
    suggested_max_length = min(suggested_max_length, context_limit)

    # Check what percentage of documents would fit in the suggested context window
    docs_in_context = len([x for x in stats.get("raw_data", {}).get("text_tokens_with_prompt", [])
                          if x <= suggested_max_length])
    total_docs = len(stats.get("raw_data", {}).get("text_tokens_with_prompt", []))
    coverage_percent = (docs_in_context / total_docs * 100) if total_docs > 0 else 0

    # Create parameter suggestions dictionary
    parameter_suggestions = {
        "max_length": suggested_max_length,
        "max_new_tokens": suggested_max_new_tokens,
        "temperature": 0.5,  # Standard balanced choice
        "top_p": 0.9,        # Standard balanced choice
        "recommended_context_window": recommended_window,
        "context_window_coverage_percent": coverage_percent
    }

    return parameter_suggestions


def print_parameter_suggestions(parameter_suggestions):
    """
    Print parameter suggestions in a formatted way

    Parameters:
        parameter_suggestions: Dictionary containing parameter suggestions
    """
    print("\n=========== Suggested Parameters ===========")
    print(f"Suggested max_length (input length with prompt): {parameter_suggestions['max_length']}")
    print(f"Suggested max_new_tokens (maximum generation length): {parameter_suggestions['max_new_tokens']}")
    print(f"Recommended context window: {parameter_suggestions['recommended_context_window']}")
    print(f"Context window coverage: {parameter_suggestions['context_window_coverage_percent']:.2f}% of documents")
    print(f"Suggested temperature: 0.3-0.7 (recommended: {parameter_suggestions['temperature']})")
    print(f"Suggested top_p: 0.85-0.95 (recommended: {parameter_suggestions['top_p']})")
    print("============================================")


def main():
    parser = argparse.ArgumentParser(description="Analyze long document summarization datasets")
    parser.add_argument("--dataset", type=str, default="ccdv/pubmed-summarization",
                        help="Dataset name (ccdv/pubmed-summarization, ccdv/govreport-summarization, kmfoda/booksum)")
    parser.add_argument("--tokenizer", type=str, default="meta-llama/Llama-3.2-3B-Instruct", help="Name of the tokenizer to use")
    parser.add_argument("--sample_size", type=int, default=1000, help="Number of samples for analysis (0 means use all samples)")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save results")

    args = parser.parse_args()

    # Ensure sample_size is valid
    if args.sample_size < 0:
        print(f"Warning: Sample size {args.sample_size} is negative, setting to 0 (use all samples)")
        args.sample_size = 0

    try:
        result = analyze_long_doc_dataset(
            dataset_name=args.dataset,
            tokenizer_name=args.tokenizer,
            sample_size=args.sample_size,
            output_dir=args.output_dir
        )

        if result is None:
            print("Analysis failed, please check the error message above.")
            return 1
        else:
            print("Analysis completed successfully!")
            return 0
    except Exception as e:
        print(f"Error occurred during program execution: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
