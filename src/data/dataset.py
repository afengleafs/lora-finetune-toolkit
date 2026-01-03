"""Dataset loading and preprocessing utilities."""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from datasets import Dataset, load_dataset


def load_json_dataset(data_path: str) -> Dataset:
    """
    Load dataset from JSON file.
    
    Args:
        data_path: Path to JSON file containing list of {input, output} pairs.
        
    Returns:
        HuggingFace Dataset object.
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return Dataset.from_list(data)


def load_jsonl_dataset(data_path: str) -> Dataset:
    """
    Load dataset from JSONL file.
    
    Args:
        data_path: Path to JSONL file.
        
    Returns:
        HuggingFace Dataset object.
    """
    return load_dataset('json', data_files=data_path, split='train')


def load_training_dataset(
    data_path: str,
    file_format: Optional[str] = None,
) -> Dataset:
    """
    Load training dataset from file.
    
    Automatically detects format from file extension if not specified.
    
    Args:
        data_path: Path to dataset file.
        file_format: Optional format override ('json', 'jsonl', 'csv').
        
    Returns:
        HuggingFace Dataset object.
    """
    path = Path(data_path)
    
    if file_format is None:
        file_format = path.suffix.lower().lstrip('.')
    
    if file_format == 'json':
        return load_json_dataset(data_path)
    elif file_format == 'jsonl':
        return load_jsonl_dataset(data_path)
    elif file_format == 'csv':
        return load_dataset('csv', data_files=data_path, split='train')
    else:
        # Try as JSON by default
        return load_json_dataset(data_path)


def format_dataset_for_training(
    dataset: Dataset,
    format_fn: Callable[[str, str], str],
    input_column: str = "input",
    output_column: str = "output",
    text_column: str = "text",
) -> Dataset:
    """
    Format dataset for SFT training.
    
    Converts input/output columns to formatted text using the provided
    formatting function (typically from a model adapter).
    
    Args:
        dataset: Input dataset.
        format_fn: Function to format (input, output) -> text.
        input_column: Name of input column.
        output_column: Name of output column.
        text_column: Name of output text column.
        
    Returns:
        Formatted dataset with text column.
    """
    def format_example(examples):
        if isinstance(examples[input_column], list):
            # Batch mode
            texts = [
                format_fn(inp, out)
                for inp, out in zip(examples[input_column], examples[output_column])
            ]
            return {text_column: texts}
        else:
            # Single example mode
            return {text_column: format_fn(examples[input_column], examples[output_column])}
    
    # Apply formatting and remove original columns
    formatted = dataset.map(format_example, batched=True)
    formatted = formatted.remove_columns([input_column, output_column])
    
    return formatted


def prepare_dataset(
    data_path: str,
    format_fn: Callable[[str, str], str],
    input_column: str = "input",
    output_column: str = "output",
    train_split: float = 1.0,
    seed: int = 42,
) -> Union[Dataset, Dict[str, Dataset]]:
    """
    Load and prepare dataset for training.
    
    Args:
        data_path: Path to dataset file.
        format_fn: Formatting function from model adapter.
        input_column: Name of input column.
        output_column: Name of output column.
        train_split: Fraction for train set (1.0 = no split).
        seed: Random seed for splitting.
        
    Returns:
        Dataset or dict with 'train' and 'eval' splits.
    """
    # Load dataset
    dataset = load_training_dataset(data_path)
    
    # Format for training
    dataset = format_dataset_for_training(
        dataset,
        format_fn,
        input_column,
        output_column,
    )
    
    # Split if requested
    if train_split < 1.0:
        split = dataset.train_test_split(
            test_size=1.0 - train_split,
            seed=seed,
        )
        return {"train": split["train"], "eval": split["test"]}
    
    return dataset
