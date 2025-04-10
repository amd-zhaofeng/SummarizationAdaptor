import torch
import logging
from peft import LoraConfig, TaskType, PeftModel
from typing import Dict, List, Optional, Any, cast
from model.processor import get_tokenizer, get_model
from data.prompt import format_prompt, get_prompt_parts

logger = logging.getLogger(__name__)

class SummarizationAdapter:
    """
    Summarization adapter class, based on Llama 3.2 3B model and LoRA technology
    """
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
                 lora_r: int = 8,
                 lora_alpha: int = 32,
                 lora_dropout: float = 0.1,
                 adapter_path: Optional[str] = None):
        """
        Initialize the summarization adapter

        Parameters:
            model_name: Base model name or path
            lora_r: Rank of LoRA parameters
            lora_alpha: Scaling parameter for LoRA
            lora_dropout: Dropout rate for LoRA layers
            adapter_path: Path to pre-trained adapter (if provided)
        """
        self.model_name = model_name
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.adapter_path = adapter_path
        self.is_base_model = adapter_path is None

        self.tokenizer = get_tokenizer(model_name)

        # Load model
        self.model: Any = None
        self._load_model()

    def _load_model(self) -> None:
        """Load and configure the model"""
        logger.info(f"Loading base model: {self.model_name}")

        # Load base model
        model_kwargs: Dict[str, Any] = {}
        model_kwargs["device_map"] = "auto"

        self.model = get_model(self.model_name)

        if self.adapter_path:
            # Load pre-trained adapter
            logger.info(f"Loading adapter: {self.adapter_path}")
            # Normal adapter loading
            self.model = PeftModel.from_pretrained(
                self.model,
                self.adapter_path,
                torch_dtype=torch.float16
            )

            # Print number of adapter parameters
            self.model.print_trainable_parameters()
            logger.info("Adapter loaded successfully")
        else:
            logger.info("Using base model for inference (no adapter loaded)")

    def get_peft_config(self) -> LoraConfig:
        """
        Get PEFT configuration for SFTTrainer

        Parameters:
            target_modules: List of target modules to apply LoRA

        Returns:
            LoRA configuration object
        """

        target_modules: List[str] = ["q_proj",
                                     "k_proj",
                                     "v_proj",
                                     "o_proj",
                                     "gate_proj",
                                     "up_proj",
                                     "down_proj"]
        logger.info(f"Creating LoRA configuration, target modules: {target_modules}")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=target_modules,
            bias="none"
        )

        return peft_config

    def generate_summary(self, text: str, max_new_tokens: int = 132, temperature: float = 0.5, top_p: float = 0.9, max_length: int = 2048, dataset_name: Optional[str] = None) -> str:
        """
        Generate text summary

        Parameters:
            text: Text to summarize
            max_new_tokens: Maximum number of tokens for generated summary
            temperature: Temperature parameter for generation
            top_p: Top-p parameter for generation
            max_length: Maximum length of input text
            dataset_name: Optional dataset name for specific prompt template

        Returns:
            Generated summary text
        """
        # Get formatted prompt from prompt module
        prompt = format_prompt(text, dataset_name)

        # Truncate input if necessary
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=max_length, truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Set generation parameters
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id
        }

        # Generate summary
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        # Decode and return summary portion
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Get prompt parts to determine the delimiter based on dataset
        _, prompt_middle = get_prompt_parts(dataset_name)
        
        # Extract summary portion
        # First try to find the marker that indicates the start of the summary
        markers = ["### SUMMARY ###", "### ABSTRACT ###"]
        summary = None
        
        for marker in markers:
            if marker in decoded:
                # Split by the marker and take everything after it
                summary = decoded.split(marker)[-1].strip()
                break
        
        # If no marker found, fall back to original approach
        if summary is None:
            delimiter = prompt_middle.strip()
            if delimiter in decoded:
                summary = decoded.split(delimiter)[-1].strip()
            else:
                # Last resort: try to remove the input text
                input_text = prompt.split("\n\n")[0]
                if input_text in decoded:
                    summary = decoded.replace(input_text, "").strip()
                else:
                    summary = decoded.strip()
        
        # Explicit cast to ensure string return type
        summary_str: str = cast(str, summary)
        logger.info(f"Extracted summary: {summary_str}")

        return summary_str
