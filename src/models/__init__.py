"""Model adapters package."""

from typing import Any, Dict

from .base import BaseModelAdapter
from .qwen import QwenAdapter
from .llama import LlamaAdapter
from .mistral import MistralAdapter


# Model registry mapping model names to adapter classes
MODEL_REGISTRY = {
    "qwen": QwenAdapter,
    "qwen3": QwenAdapter,
    "llama": LlamaAdapter,
    "llama3": LlamaAdapter,
    "mistral": MistralAdapter,
}


def get_model_adapter(config: Dict[str, Any]) -> BaseModelAdapter:
    """
    Factory function to get the appropriate model adapter.
    
    Args:
        config: Configuration dictionary with model.name field.
        
    Returns:
        Appropriate model adapter instance.
        
    Raises:
        ValueError: If model type cannot be determined.
    """
    model_name = config.get("model", {}).get("name", "").lower()
    
    # Try to match model name to adapter
    for key, adapter_class in MODEL_REGISTRY.items():
        if key in model_name:
            return adapter_class(config)
    
    # Default to Qwen for unknown models
    print(f"Warning: Unknown model '{model_name}', defaulting to QwenAdapter")
    return QwenAdapter(config)


__all__ = [
    "BaseModelAdapter",
    "QwenAdapter",
    "LlamaAdapter",
    "MistralAdapter",
    "get_model_adapter",
    "MODEL_REGISTRY",
]
