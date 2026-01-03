"""Mistral 7B model adapter."""

from typing import Any, Dict

from .base import BaseModelAdapter


class MistralAdapter(BaseModelAdapter):
    """
    Adapter for Mistral AI models (Mistral-7B-Instruct, etc.)
    
    Mistral uses the Mistral AI chat template with [INST] tokens.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Mistral adapter.
        
        Args:
            config: Configuration dictionary.
        """
        super().__init__(config)
    
    @property
    def model_name(self) -> str:
        """Return the HuggingFace model identifier."""
        return self.config.get("model", {}).get("name", "mistralai/Mistral-7B-Instruct-v0.3")
    
    def format_messages(self, input_text: str, output_text: str = None) -> str:
        """
        Format messages using Mistral's chat template.
        
        Mistral uses [INST] and [/INST] tokens for instruction formatting.
        
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
