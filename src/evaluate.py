import os
import json
import argparse
from tqdm import tqdm
from adapter import SummarizationAdapter
from utils.seed import set_seed
from utils.logging import setup_logger
from utils.metrics import calculate_metrics
from typing import Dict, List, Optional
from data.processor import process_evaluation_dataset

logger = setup_logger(__name__)


def evaluate_model(
    adapter_path: Optional[str],
    dataset_name: str,
    max_samples: Optional[int],
    output_dir: str,
    model_name: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    max_length: int,
    seed: int
) -> Dict[str, float]:
    """
    Evaluate summarization adapter model or base model

    Parameters:
        adapter_path: Adapter model path (can be None)
        dataset_name: Dataset name
        max_samples: Maximum number of samples
        output_dir: Output directory
        model_name: Base model name
        max_new_tokens: Maximum number of tokens for generated summary
        temperature: Temperature parameter for generation, lower values make results more deterministic
        top_p: Top-p value for text generation, controls vocabulary distribution
        max_length: Maximum length of input text
        seed: Random seed for reproducibility
    """
    set_seed(seed)

    if not adapter_path:
        logger.info(f"Evaluating base model: {model_name} on dataset {dataset_name}")
        model_type = "base_model"
    else:
        logger.info(f"Evaluating adapter model: {adapter_path} on dataset {dataset_name}")
        model_type = "adapter"

    # Load and process dataset using the abstracted function
    dataset = process_evaluation_dataset(dataset_name=dataset_name,
                                         max_samples=max_samples,
                                         seed=seed)

    # Load model - adapter or base model
    adapter = SummarizationAdapter(
        model_name=model_name,
        adapter_path=adapter_path,
    )

    # Generate summaries
    references: List[str] = []
    predictions: List[str] = []

    for example in tqdm(dataset, desc="Generating summaries"):
        text = example["text"]
        reference = example["summary"]

        # Generate summary
        try:
            prediction = adapter.generate_summary(
                text,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                max_length=max_length,
                dataset_name=dataset_name
            )

            # Verify generated summary is not empty
            if not prediction or prediction.strip() == "":
                logger.warning(f"Warning: Sample #{len(predictions)} generated an empty summary, using fallback text")
                prediction = "Empty summary generated."

        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            prediction = "Error generating summary."

        references.append(reference)
        predictions.append(prediction)

    metrics = calculate_metrics(predictions, references, lang="en")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    # Replace / in dataset_name with _
    formatted_dataset_name = dataset_name.replace("/", "_")
    results_file = os.path.join(output_dir, f"evaluation_results_{formatted_dataset_name}_{model_type}_{max_samples}.json")

    # Save metrics
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # Save sample predictions
    samples_file = os.path.join(output_dir, f"evaluation_samples_{formatted_dataset_name}_{model_type}_{max_samples}.json")
    samples: List[Dict[str, str]] = []

    # for i in range(len(predictions)):
    #     samples.append({
    #         "text": dataset[i]["text"],
    #         "reference": references[i],
    #         "prediction": predictions[i]
    #     })

    # with open(samples_file, "w", encoding="utf-8") as f:
    #     json.dump(samples, f, indent=2, ensure_ascii=False)

    logger.info(f"Evaluation results saved to: {results_file}")
    # logger.info(f"Evaluation samples saved to: {samples_file}")

    # Print main metrics
    logger.info("\n=== Evaluation Results ===")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate summarization adapter model")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct", help="Base model name or path")
    parser.add_argument("--dataset", type=str, default="ccdv/pubmed-summarization", choices=["ccdv/pubmed-summarization", "ccdv/arxiv-summarization"], help="Dataset name")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory")
    parser.add_argument("--adapter_path", type=str, default=None, help="Adapter model path (not needed if using base model)")

    parser.add_argument("--max_samples", type=int, default=5000, help="Maximum number of samples")

    parser.add_argument("--max_length", type=int, default=8192, help="Maximum length of input text")
    parser.add_argument("--max_new_tokens", type=int, default=400, help="Maximum number of tokens for generated summary")
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature parameter for generation, lower values make results more deterministic")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p value for text generation, controls vocabulary distribution")

    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    # Log all arguments
    logger.info("Command line arguments:")
    logger.info(vars(args))

    evaluate_model(
        adapter_path=args.adapter_path,
        dataset_name=args.dataset,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        model_name=args.model_name,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        max_length=args.max_length,
        seed=args.seed
    )
