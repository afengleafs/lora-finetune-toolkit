"""LLM Fine-tuning Toolkit - A modular toolkit for LLM fine-tuning with LoRA."""

from importlib import import_module

__version__ = "0.1.0"

_LAZY_IMPORTS = {
    "QwenAdapter": ("src.models", "QwenAdapter"),
    "LlamaAdapter": ("src.models", "LlamaAdapter"),
    "MistralAdapter": ("src.models", "MistralAdapter"),
    "get_model_adapter": ("src.models", "get_model_adapter"),
    "load_config": ("src.utils.config", "load_config"),
}

__all__ = list(_LAZY_IMPORTS)


def __getattr__(name):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module 'src' has no attribute {name!r}")

    module_name, attr_name = _LAZY_IMPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
