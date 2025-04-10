"""
Prompt templates for text summarization.
"""
from typing import Dict, Optional, Tuple

# Define prompt prefix and middle parts for each dataset
PROMPT_PARTS: Dict[str, Tuple[str, str]] = {
    "default": (
        "Please generate a concise summary for the following text. Your summary should be around 250-300 words and clearly capture the main points:\n\n",
        "\n\n### SUMMARY ###\n"
    ),
    "ccdv/pubmed-summarization": (
        "Generate a comprehensive abstract for the following scientific article, including the background, methods, results, and conclusions. Keep your abstract concise, around 250-300 words, and maintain a scientific tone:\n\n",
        "\n\n### ABSTRACT ###\n"
    ),
    "ccdv/arxiv-summarization": (
        "Generate a comprehensive abstract for the following scientific paper, including the research problem, methods, main findings, and significance. Keep your abstract concise, around 250-300 words, and follow academic writing conventions:\n\n",
        "\n\n### ABSTRACT ###\n"
    ),
}

# Build templates based on PROMPT_PARTS
DATASET_PROMPT_TEMPLATES: Dict[str, str] = {
    dataset_name: prefix + "{text}" + middle
    for dataset_name, (prefix, middle) in PROMPT_PARTS.items()
    if dataset_name != "default"
}

# Default template
DEFAULT_PROMPT_TEMPLATE = PROMPT_PARTS["default"][0] + "{text}" + PROMPT_PARTS["default"][1]


def get_prompt_template(dataset_name: Optional[str] = None) -> str:
    """
    Returns the appropriate prompt template based on dataset name.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Prompt template string
    """
    if dataset_name and dataset_name in DATASET_PROMPT_TEMPLATES:
        return DATASET_PROMPT_TEMPLATES[dataset_name]
    return DEFAULT_PROMPT_TEMPLATE


def format_prompt(text: str, dataset_name: Optional[str] = None) -> str:
    """
    Format the prompt with the given text.
    
    Args:
        text: The text to be summarized
        dataset_name: Optional dataset name for specific prompt template
        
    Returns:
        Formatted prompt string
    """
    template = get_prompt_template(dataset_name)
    return template.format(text=text)

def get_prompt_parts(dataset_name: Optional[str] = None) -> Tuple[str, str]:
    """
    Returns the parts of the prompt for token calculations based on dataset name.
    
    Args:
        dataset_name: Name of the dataset
    
    Returns:
        Tuple of (prefix, middle) parts of the prompt
    """
    if dataset_name and dataset_name in PROMPT_PARTS:
        return PROMPT_PARTS[dataset_name]
    return PROMPT_PARTS["default"]
