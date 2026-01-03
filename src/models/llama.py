"""Llama 3.2 model adapter."""

from typing import Any, Dict

from .base import BaseModelAdapter


class LlamaAdapter(BaseModelAdapter):
    """
    Adapter for Meta Llama 3.2 models.
    
    Llama 3.2 uses Meta's chat template format and requires
    authentication for model access.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Llama adapter.
        
        Args:
            config: Configuration dictionary.
        """
        super().__init__(config)
    
    @property
    def model_name(self) -> str:
        """Return the HuggingFace model identifier."""
        return self.config.get("model", {}).get("name", "meta-llama/Llama-3.2-8B-Instruct")
    
    def format_messages(self, input_text: str, output_text: str = None) -> str:
        """
        Format messages using Llama 3.2's chat template.
        
        Llama uses <|begin_of_text|>, <|start_header_id|>, etc. tokens.
        
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
