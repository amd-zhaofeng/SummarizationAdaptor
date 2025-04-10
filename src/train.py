import argparse
import os
from datetime import datetime
from typing import List, Optional

import torch
from trl import SFTConfig, SFTTrainer

from adapter import SummarizationAdapter
from data.processor import process_training_dataset
from utils.seed import set_seed
from utils.logging import setup_logger
logger = setup_logger(__name__)


def train(
    model_name: str,
    dataset_name: str,
    output_dir: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    learning_rate: float,
    batch_size: int,
    gradient_accumulation_steps: int,
    num_epochs: int,
    max_length: int,
    summary_max_length: int,
    warmup_ratio: float,
    weight_decay: float,
    logging_steps: int,
    save_steps: int,
    eval_steps: int,
    max_grad_norm: float,
    train_samples: Optional[int],
    val_samples: Optional[int],
) -> str:
    """
    Train the summarization adapter model

    Parameters:
        model_name: Base model name
        dataset_name: Dataset name
        output_dir: Output directory
        lora_r: Rank of LoRA
        lora_alpha: Scaling parameter of LoRA
        lora_dropout: Dropout rate of LoRA
        learning_rate: Learning rate
        batch_size: Batch size
        gradient_accumulation_steps: Gradient accumulation steps
        num_epochs: Number of training epochs
        max_length: Maximum input length
        summary_max_length: Maximum summary length
        warmup_ratio: Learning rate warmup ratio
        weight_decay: Weight decay
        logging_steps: Logging steps
        save_steps: Model saving steps
        eval_steps: Evaluation steps
        max_grad_norm: Gradient clipping threshold
        train_samples: Number of training samples to use, None means using all
        val_samples: Number of validation samples to use, None means using all

    Returns:
        Path to the saved model directory
    """
    # Set output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_dir, f"summarization_adapter_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(
        f"Training summarization adapter based on {model_name}, "
        f"using dataset {dataset_name}"
    )
    logger.info(f"Output directory: {output_dir}")

    # Load dataset - returns HF Dataset format
    logger.info("Starting to load dataset...")
    train_dataset, val_dataset = process_training_dataset(
        dataset_name=dataset_name,
        tokenizer_name=model_name,
        max_length=max_length,
        summary_max_length=summary_max_length,
        train_samples=train_samples,
        val_samples=val_samples,
    )
    logger.info(f"Training set size: {len(train_dataset)}, Validation set size: {len(val_dataset)}")

    # Initialize model and tokenizer using SummarizationAdapter
    logger.info("Initializing model and configuration using SummarizationAdapter")
    adapter = SummarizationAdapter(
        model_name=model_name,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )

    # Get LoRA configuration
    peft_config = adapter.get_peft_config()

    # Configure training parameters
    training_args = SFTConfig(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_epochs,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        logging_steps=logging_steps,
        optim = "adamw_8bit",
        save_steps=save_steps,
        eval_steps=eval_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        max_grad_norm=max_grad_norm,
        max_seq_length=max_length,
        remove_unused_columns=False,
        logging_dir=os.path.join(output_dir, "logs"),
        dataloader_drop_last=True,
    )

    # Initialize SFTTrainer
    logger.info("Initializing SFTTrainer...")

    # Initialize SFTTrainer with more compatible parameters
    trainer = SFTTrainer(
        model=adapter.model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=peft_config,
        processing_class=adapter.tokenizer,
        args=training_args,
    )

    # Start training
    logger.info("Starting training...")
    trainer.train()

    # Save adapter model
    logger.info(f"Saving adapter to {output_dir}")
    trainer.save_model(output_dir)
    adapter.tokenizer.save_pretrained(output_dir)
    logger.info(f"Training completed, adapter saved to {output_dir}")

    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a summarization adapter model")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct", help="Base model name or path")
    parser.add_argument("--dataset", type=str, default="ccdv/pubmed-summarization", choices=["ccdv/pubmed-summarization", "ccdv/arxiv-summarization"], help="Dataset name")
    parser.add_argument("--output_dir", type=str, default="./models", help="Output directory")

    parser.add_argument("--train_samples", type=int, default=5000, help="Number of training samples, defaults to all")
    parser.add_argument("--val_samples", type=int, default=500, help="Number of validation samples, defaults to all")

    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--max_length", type=int, default=8192, help="Maximum input length")
    parser.add_argument("--summary_max_length", type=int, default=400, help="Maximum summary length")
    parser.add_argument("--save_steps", type=int, default=50, help="Model saving steps")
    parser.add_argument("--eval_steps", type=int, default=50, help="Model evaluation steps")

    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA scaling parameter")
    parser.add_argument("--lora_dropout", type=float, default=0, help="LoRA dropout rate")

    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Learning rate warmup ratio")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--logging_steps", type=int, default=1, help="Logging steps")
    parser.add_argument("--max_grad_norm", type=float, default=0.7, help="Gradient clipping threshold")

    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    # Set random seed
    set_seed(args.seed)

    # Log all arguments
    logger.info("Command line arguments:")
    logger.info(vars(args))

    train(
        model_name=args.model_name,
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_epochs=args.num_epochs,
        max_length=args.max_length,
        summary_max_length=args.summary_max_length,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        max_grad_norm=args.max_grad_norm,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
    )
