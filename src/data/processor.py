import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Any, Optional
from utils.logging import setup_logger
from model.processor import get_tokenizer
from data.prompt import get_prompt_parts

# Setup logger
logger = setup_logger(__name__)


def load_dataset_with_unified_columns(dataset_name: str, split: Optional[str] = None, version: Optional[str] = None) -> Any:
    """
    Load dataset and unify column names to 'text' and 'summary'

    Parameters:
        dataset_name: Dataset name
        split: Dataset split (if None, loads all splits)
        version: Dataset version (if applicable)

    Returns:
        dataset: Processed dataset with columns unified to 'text' and 'summary'
    """
    # Load dataset based on name and unify column names
    if dataset_name == "cnn_dailymail":
        dataset = load_dataset(dataset_name, "3.0.0" if version is None else version, split=split)
        dataset = dataset.rename_column("article", "text")
        dataset = dataset.rename_column("highlights", "summary")
    elif dataset_name == "xsum":
        dataset = load_dataset(dataset_name, split=split)
        dataset = dataset.rename_column("document", "text")
    elif dataset_name == "billsum":
        dataset = load_dataset(dataset_name, split=split)
        # Already uses 'text' and 'summary' fields
    elif dataset_name == "ccdv/pubmed-summarization":
        dataset = load_dataset(dataset_name, split=split)
        dataset = dataset.rename_column("article", "text")
        dataset = dataset.rename_column("abstract", "summary")
    elif dataset_name == "ccdv/arxiv-summarization":
        dataset = load_dataset(dataset_name, split=split)
        dataset = dataset.rename_column("article", "text")
        dataset = dataset.rename_column("abstract", "summary")
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return dataset


def process_training_dataset(dataset_name="cnn_dailymail", tokenizer_name="meta-llama/Llama-3.2-3B-Instruct", max_length=2048, summary_max_length=132, train_samples=None, val_samples=None):
    """
    Prepare summarization dataset, returns Hugging Face Dataset objects compatible with SFTTrainer

    Parameters:
        dataset_name: Dataset name
        tokenizer_name: Tokenizer name
        max_length: Maximum input text length (default 2048, consistent with adapter.py)
        summary_max_length: Maximum summary length (default 132, consistent with adapter.py)
        train_samples: Limit number of training samples, None means use all
        val_samples: Limit number of validation samples, None means use all

    Returns:
        train_dataset: Training dataset
        val_dataset: Validation dataset
    """
    logger.info(f"Loading dataset: {dataset_name}")
    logger.info(f"Using parameters - Maximum input length: {max_length}, Maximum summary length: {summary_max_length}")

    # Load dataset using unified function
    raw_datasets = load_dataset_with_unified_columns(dataset_name)

    # Get validation split name
    val_split = "validation" if "validation" in raw_datasets else "test"
    logger.info(f"Original dataset size - Training: {len(raw_datasets['train'])}, Validation: {len(raw_datasets[val_split])}")

    # Clean and filter dataset to remove empty samples
    logger.info("Cleaning dataset...")
    cleaned_train_dataset = raw_datasets['train'].filter(
        lambda x: len(x["text"].strip()) > 0 and len(x["summary"].strip()) > 0
    )
    cleaned_val_dataset = raw_datasets[val_split].filter(
        lambda x: len(x["text"].strip()) > 0 and len(x["summary"].strip()) > 0
    )
    logger.info(f"After cleaning: Training {len(cleaned_train_dataset)} samples (removed {len(raw_datasets['train']) - len(cleaned_train_dataset)} empty samples)")
    logger.info(f"After cleaning: Validation {len(cleaned_val_dataset)} samples (removed {len(raw_datasets[val_split]) - len(cleaned_val_dataset)} empty samples)")

    # If we need to limit the number of samples, select them here
    if train_samples is not None and train_samples < len(cleaned_train_dataset):
        # Randomly select specified number of samples
        train_indices = torch.randperm(len(cleaned_train_dataset))[:train_samples].tolist()
        raw_train_dataset = cleaned_train_dataset.select(train_indices)
        logger.info(f"Limited training set: {len(raw_train_dataset)}/{len(cleaned_train_dataset)} samples")
    else:
        raw_train_dataset = cleaned_train_dataset

    if val_samples is not None and val_samples < len(cleaned_val_dataset):
        # Randomly select specified number of samples
        val_indices = torch.randperm(len(cleaned_val_dataset))[:val_samples].tolist()
        raw_val_dataset = cleaned_val_dataset.select(val_indices)
        logger.info(f"Limited validation set: {len(raw_val_dataset)}/{len(cleaned_val_dataset)} samples")
    else:
        raw_val_dataset = cleaned_val_dataset

    # Load tokenizer
    tokenizer = get_tokenizer(tokenizer_name)

    # Preprocessing function, combines text and summary into instruction format and ensures length limits
    def preprocess_function(examples):
        original_texts = []
        texts = []
        summaries = []
        for text, summary in zip(examples["text"], examples["summary"]):
            # Build instruction fine-tuning input format, consistent with adapter.py
            # Note that summary position doesn't have a colon, to match format in adapter.py's generate_summary method

            original_texts.append(text)
            summaries.append(summary)

        # First encode summaries to ensure they aren't truncated
        summary_inputs = tokenizer(
            summaries,
            max_length=summary_max_length,
            padding=True,
            truncation=True,
            return_tensors=None,
        )

        # Calculate available tokens for original text
        available_lengths = []
        
        # Get prompt parts from prompt module with dataset name
        prompt_prefix, prompt_middle = get_prompt_parts(dataset_name)

        # Calculate the full prompt token count
        sample_text = f"{prompt_prefix}SAMPLE{prompt_middle}SAMPLE"
        sample_encoded = tokenizer(sample_text, return_tensors="pt")
        sample_text_only = tokenizer("SAMPLESAMPLE", return_tensors="pt")

        # Calculate extra tokens for special tokens and prompt words
        # Fix type error by accessing .input_ids as a tensor attribute
        special_tokens_count = sample_encoded.input_ids.size(1) - sample_text_only.input_ids.size(1) + 2  # +2 for safety margin
        prompt_tokens = tokenizer(prompt_prefix + prompt_middle, add_special_tokens=False)
        prompt_tokens_count = len(prompt_tokens.input_ids)

        # Total extra tokens = special tokens + prompt words
        extra_tokens = special_tokens_count + prompt_tokens_count

        for summary_ids in summary_inputs["input_ids"]:
            # Calculate available tokens for original text for each sample
            # Total length limit - summary length - extra tokens
            text_max_length = max_length - len(summary_ids) - extra_tokens
            text_max_length = max(0, text_max_length)  # Ensure not negative
            available_lengths.append(text_max_length)

        # Encode and truncate original text using calculated available lengths
        truncated_texts = []
        for i, (text, available_length) in enumerate(zip(original_texts, available_lengths)):
            text_tokens = tokenizer(
                text,
                max_length=available_length,
                padding=True,
                truncation=True,
                return_tensors=None,
            )["input_ids"]

            # Convert tokens back to text
            truncated_text = tokenizer.decode(text_tokens, skip_special_tokens=True)
            truncated_texts.append(truncated_text)

            # Build final formatted text
            formatted_text = f"{prompt_prefix}{truncated_text}{prompt_middle}{summaries[i]}"
            texts.append(formatted_text)

        # Final verification: ensure all texts don't exceed max_length
        final_texts = []
        for i, text in enumerate(texts):
            final_encoded = tokenizer(
                text,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors=None,
            )
            final_text = tokenizer.decode(final_encoded["input_ids"], skip_special_tokens=True)
            final_texts.append(final_text)

        # Check and output tokenized length distribution statistics, only print for first batch
        if not hasattr(preprocess_function, 'has_logged'):
            full_inputs = tokenizer(final_texts, return_tensors=None)
            input_lengths = [len(tokens) for tokens in full_inputs["input_ids"]]
            max_found = max(input_lengths)
            avg_length = sum(input_lengths) / len(input_lengths)
            logger.info(f"Tokenized input statistics - Max length: {max_found}, Average length: {avg_length:.2f}, Proportion exceeding max length: {sum(1 for l in input_lengths if l >= max_length) / len(input_lengths):.2%}")

            # Fix type error by accessing .input_ids directly
            summary_token_lengths = [len(ids) for ids in summary_inputs["input_ids"]]
            avg_summary_length = sum(summary_token_lengths) / len(summary_token_lengths)
            logger.info(f"Average summary length: {avg_summary_length:.2f} tokens")

            logger.info(f"Average available length for original text: {sum(available_lengths) / len(available_lengths):.2f} tokens")
            logger.info(f"Extra token count (special tokens + prompt words): {extra_tokens}")
            preprocess_function.has_logged = True

        return {"text": final_texts}

    # Apply preprocessing function to dataset
    train_dataset = raw_train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=list(raw_train_dataset.features),  # Remove original columns
        desc="Processing training set"
    )

    val_dataset = raw_val_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=list(raw_val_dataset.features),  # Remove original columns
        desc="Processing validation set"
    )

    logger.info(f"Processed dataset size - Training: {len(train_dataset)}, Validation: {len(val_dataset)}")

    return train_dataset, val_dataset


def process_evaluation_dataset(dataset_name: str,
                               max_samples: Optional[int],
                               seed: int) -> Any:
    """
    Process dataset for evaluation.

    Parameters:
        dataset_name: Dataset name
        max_samples: Maximum number of samples
        seed: Random seed for reproducibility

    Returns:
        dataset: Processed dataset ready for evaluation
    """
    # Load dataset using unified function
    dataset = load_dataset_with_unified_columns(dataset_name, split="test")

    # Clean and filter dataset to remove empty samples
    logger.info("Cleaning dataset...")
    original_size = len(dataset)
    dataset = dataset.filter(
        lambda x: len(x["text"].strip()) > 0 and len(x["summary"].strip()) > 0
    )
    logger.info(f"After cleaning: {len(dataset)} samples (removed {original_size - len(dataset)} empty samples)")

    # Shuffle dataset using fixed random seed
    shuffled_dataset = dataset.shuffle(seed=seed)

    # Limit sample count
    if max_samples and max_samples < len(shuffled_dataset):
        dataset = shuffled_dataset.select(range(max_samples))
    else:
        dataset = shuffled_dataset

    logger.info(f"Evaluation sample count: {len(dataset)}")

    return dataset
