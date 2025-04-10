import argparse
from adapter import SummarizationAdapter
from typing import Optional
from utils.logging import setup_logger

logger = setup_logger(__name__)


def generate_summary(text: str,
                     adapter_path: str,
                     model_name: str,
                     max_new_tokens: int,
                     temperature: float,
                     top_p: float,
                     max_length: int,
                     output_file: str) -> str:
    """
    Generate summary using summarization adapter

    Parameters:
        text: Text to summarize
        adapter_path: Adapter model path
        model_name: Base model name
        max_new_tokens: Maximum number of tokens for generated summary
        temperature: Temperature parameter for generation
        top_p: Top-p parameter for generation
        max_length: Maximum length of input text
        output_file: Output file path

    Returns:
        Generated summary text
    """
    # Load adapter model
    logger.info(f"Loading adapter model from {adapter_path}, base model: {base_model_name}")
    adapter = SummarizationAdapter(
        model_name=model_name,
        adapter_path=adapter_path,
    )

    # Generate summary
    logger.info("Generating summary...")
    summary = adapter.generate_summary(
        text=text,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        max_length=max_length
    )

    # Display summary
    logger.info("\n===== Generated Summary =====")
    print(summary)

    # Save summary
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(summary)
        logger.info(f"Summary saved to {output_file}")

    return summary


def read_text_file(file_path: str) -> str:
    """
    Read text from file

    Parameters:
        file_path: Path to the text file to read

    Returns:
        Content of the text file as string
    """
    logger.info(f"Reading content from file: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    logger.info(f"File read successfully, size: {len(content)} characters")
    return content


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate summary using summarization adapter")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct", help="Base model name")
    parser.add_argument("--output_file", type=str, default="./results/inference_output.txt", help="Output file path")
    parser.add_argument("--adapter_path", type=str, default=None, help="Adapter model path")

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input_text", type=str, help="Text to summarize")
    input_group.add_argument("--input_file", type=str, help="Path to file containing text to summarize")

    parser.add_argument("--max_length", type=int, default=8192, help="Maximum length of input text")
    parser.add_argument("--max_new_tokens", type=int, default=400, help="Maximum number of tokens for generated summary")

    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature parameter for generation")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p parameter for generation")

    args = parser.parse_args()

    # Log all arguments
    logger.info("Command line arguments:")
    logger.info(vars(args))

    # Get input text
    if args.input_text:
        logger.info("Using provided text input")
        text = args.input_text
    else:
        logger.info(f"Reading text from file: {args.input_file}")
        text = read_text_file(args.input_file)
        logger.info(f"Read {len(text)} characters from input file")

    # Generate summary
    logger.info("Starting summary generation process")
    summary = generate_summary(
        text=text,
        adapter_path=args.adapter_path,
        model_name=args.model_name,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        max_length=args.max_length,
        output_file=args.output_file,
    )

    logger.info(f"Summary generation completed, summary length: {len(summary)} characters")
