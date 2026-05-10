"""Training utilities and trainer wrapper."""

from typing import Any, Dict, Optional, Union

import torch
from datasets import Dataset
from trl import SFTConfig, SFTTrainer

from ..models import BaseModelAdapter, get_model_adapter
from ..data import prepare_dataset
from ..utils.config import load_config


class LoRATrainer:
    """
    High-level trainer class for LoRA fine-tuning.
    
    Wraps SFTTrainer with configuration management and
    model adapter integration.
    """
    
    def __init__(
        self,
        config: Union[str, Dict[str, Any]],
        data_path: str,
        output_dir: str,
        device_map: Optional[str] = None,
    ):
        """
        Initialize the trainer.
        
        Args:
            config: Path to YAML config or config dictionary.
            data_path: Path to training data.
            output_dir: Directory to save outputs.
            device_map: Device mapping (default from config).
        """
        # Load config if path provided
        if isinstance(config, str):
            self.config = load_config(config)
        else:
            self.config = config
        
        self.data_path = data_path
        self.output_dir = output_dir
        self.device_map = device_map
        
        # Initialize components
        self.adapter = None
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.dataset = None
    
    def setup(self):
        """Set up model, tokenizer, and dataset."""
        print("Loading model and tokenizer...")
        
        # Get appropriate model adapter
        self.adapter = get_model_adapter(self.config)
        
        # Load tokenizer first (needed for dataset formatting)
        self.tokenizer = self.adapter.load_tokenizer()
        
        # Load and format dataset
        print("Preparing dataset...")
        self.dataset = prepare_dataset(
            self.data_path,
            self.adapter.format_messages,
        )
        
        # Load model with LoRA
        print("Loading model with LoRA...")
        self.model = self.adapter.load_model(self.device_map)
        
        # Print model info
        self.adapter.print_model_info()
        
        # Print dataset info
        if isinstance(self.dataset, dict):
            print(f"Train dataset: {len(self.dataset['train'])} examples")
            print(f"Eval dataset: {len(self.dataset['eval'])} examples")
        else:
            print(f"Dataset: {len(self.dataset)} examples")
        
        return self
    
    def get_sft_config(self) -> SFTConfig:
        """
        Create SFTConfig from configuration.
        
        Returns:
            SFTConfig for training.
        """
        train_config = self.config.get("training", {})
        optim = train_config.get("optim", "paged_adamw_8bit")
        if optim in {"paged_adamw_8bit", "adamw_8bit"}:
            try:
                import bitsandbytes  # noqa: F401
            except ImportError:
                print("Warning: bitsandbytes is not installed, falling back to adamw_torch optimizer.")
                optim = "adamw_torch"

        max_length = train_config.get(
            "max_length",
            train_config.get("max_seq_length", 1024),
        )
        bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())

        return SFTConfig(
            # Memory optimization
            gradient_checkpointing=train_config.get("gradient_checkpointing", True),
            gradient_checkpointing_kwargs={'use_reentrant': False},
            gradient_accumulation_steps=train_config.get("gradient_accumulation_steps", 1),
            per_device_train_batch_size=train_config.get("per_device_train_batch_size", 8),
            auto_find_batch_size=train_config.get("auto_find_batch_size", True),
            
            # Dataset
            max_length=max_length,
            packing=train_config.get("packing", False),
            dataset_text_field="text",
            
            # Training hyperparameters
            num_train_epochs=train_config.get("num_train_epochs", 10),
            learning_rate=train_config.get("learning_rate", 3e-4),
            optim=optim,
            
            # Logging and output
            logging_steps=train_config.get("logging_steps", 10),
            logging_dir=f"{self.output_dir}/logs",
            output_dir=self.output_dir,
            report_to="none",
            
            # Precision
            bf16=bf16,
            
            # Save
            save_strategy=train_config.get("save_strategy", "epoch"),
            save_total_limit=train_config.get("save_total_limit", 3),
        )
    
    def create_trainer(self) -> SFTTrainer:
        """
        Create SFTTrainer instance.
        
        Returns:
            Configured SFTTrainer.
        """
        train_dataset = self.dataset
        eval_dataset = None
        
        if isinstance(self.dataset, dict):
            train_dataset = self.dataset["train"]
            eval_dataset = self.dataset.get("eval")
        
        self.trainer = SFTTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            args=self.get_sft_config(),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        
        return self.trainer
    
    def train(self):
        """Run training."""
        if self.trainer is None:
            self.create_trainer()
        
        print("\nStarting training...")
        self.trainer.train()
        print("Training complete!")
        
        return self
    
    def save_model(self, path: Optional[str] = None):
        """
        Save the trained adapter.
        
        Args:
            path: Save path (default: output_dir).
        """
        save_path = path or self.output_dir
        self.trainer.save_model(save_path)
        print(f"Model saved to: {save_path}")
        
        return self
    
    def run(self):
        """
        Run full training pipeline.
        
        Combines setup, training, and saving.
        """
        self.setup()
        self.train()
        self.save_model()
        
        return self
