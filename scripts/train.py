#!/usr/bin/env python3
"""
CLI training script for LoRA fine-tuning.

Usage:
    python scripts/train.py --config configs/qwen3_8b.yaml --data_path data.json --output_dir ./outputs
"""

import argparse
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.training import LoRATrainer
from src.utils.config import load_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune LLMs with LoRA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to training data (JSON, JSONL, or CSV)",
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Directory to save model outputs (default: ./outputs)",
    )
    
    parser.add_argument(
        "--device_map",
        type=str,
        default=None,
        help="Device map for model loading (default: from config)",
    )
    
    # Override common training parameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs",
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Override learning rate",
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size",
    )
    
    return parser.parse_args()


def main():
    """Main training entry point."""
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Apply command line overrides
    if args.epochs is not None:
        config.setdefault("training", {})["num_train_epochs"] = args.epochs
    
    if args.learning_rate is not None:
        config.setdefault("training", {})["learning_rate"] = args.learning_rate
    
    if args.batch_size is not None:
        config.setdefault("training", {})["per_device_train_batch_size"] = args.batch_size
    
    print("=" * 60)
    print("LLM Fine-tuning with LoRA")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Data: {args.data_path}")
    print(f"Output: {args.output_dir}")
    print(f"Model: {config.get('model', {}).get('name', 'Unknown')}")
    print("=" * 60)
    
    # Create and run trainer
    trainer = LoRATrainer(
        config=config,
        data_path=args.data_path,
        output_dir=args.output_dir,
        device_map=args.device_map,
    )
    
    trainer.run()
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Model saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
