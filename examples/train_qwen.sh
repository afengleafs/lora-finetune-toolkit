#!/bin/bash
# Example training commands for different models

# Train Qwen3-8B
python scripts/train.py \
  --config configs/qwen3_8b.yaml \
  --data_path data/examples/dag8_expanded.json \
  --output_dir ./outputs/qwen3-finetuned

# Train Llama-3.2-8B  
# python scripts/train.py \
#   --config configs/llama3_8b.yaml \
#   --data_path data/examples/dag8_expanded.json \
#   --output_dir ./outputs/llama3-finetuned

# Train Mistral-7B
# python scripts/train.py \
#   --config configs/mistral_7b.yaml \
#   --data_path data/examples/dag8_expanded.json \
#   --output_dir ./outputs/mistral-finetuned

# With custom parameters
# python scripts/train.py \
#   --config configs/qwen3_8b.yaml \
#   --data_path your_data.json \
#   --output_dir ./outputs/custom \
#   --epochs 5 \
#   --learning_rate 2e-4 \
#   --batch_size 8
