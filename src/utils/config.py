"""Configuration management utilities."""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional
import torch


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file.
        
    Returns:
        Dictionary containing the configuration.
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """
    Convert string dtype to torch dtype.
    
    Args:
        dtype_str: String representation of dtype.
        
    Returns:
        Corresponding torch dtype.
    """
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "auto": "auto",
    }
    return dtype_map.get(dtype_str, torch.float32)


def get_compute_dtype(dtype_str: str) -> torch.dtype:
    """Get compute dtype for quantization."""
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return dtype_map.get(dtype_str, torch.float32)


def merge_configs(base_config: Dict, override_config: Optional[Dict] = None) -> Dict:
    """
    Merge override config into base config.
    
    Args:
        base_config: Base configuration dictionary.
        override_config: Override values (optional).
        
    Returns:
        Merged configuration dictionary.
    """
    if override_config is None:
        return base_config
    
    merged = base_config.copy()
    for key, value in override_config.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged
