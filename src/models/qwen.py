"""Qwen3 model adapter."""

from typing import Any, Dict

from .base import BaseModelAdapter


class QwenAdapter(BaseModelAdapter):
    """
    Adapter for Qwen3 models (Qwen/Qwen3-8B, etc.)
    
    Qwen3 uses a standard chat template and supports efficient training
    with LoRA on attention and MLP layers.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Qwen adapter.
        
        Args:
            config: Configuration dictionary.
        """
        super().__init__(config)
    
    @property
    def model_name(self) -> str:
        """Return the HuggingFace model identifier."""
        return self.config.get("model", {}).get("name", "Qwen/Qwen3-8B")
    
    def format_messages(self, input_text: str, output_text: str = None) -> str:
        """
        Format messages using Qwen3's chat template.
        
        Qwen3 uses a standard chat format with <|im_start|> and <|im_end|> tokens.
        
        Args:
            input_text: User input text.
            output_text: Assistant response (optional).
            
        Returns:
            Formatted text string.
        """
        messages = [{"role": "user", "content": input_text}]
        
        if output_text is not None:
            messages.append({"role": "assistant", "content": output_text})
        
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=(output_text is None),
        )
