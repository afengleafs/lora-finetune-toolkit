#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/root/lora-finetune-toolkit"
OUTPUT_DIR="$PROJECT_DIR/outputs/qwen3-finetuned"
LOG_FILE="$OUTPUT_DIR/train.log"
PID_FILE="$OUTPUT_DIR/train.pid"

cd "$PROJECT_DIR"
source "$PROJECT_DIR/.venv/bin/activate"
mkdir -p "$OUTPUT_DIR"

nohup python scripts/train.py \
  --config configs/qwen3_8b.yaml \
  --data_path data/examples/dag8_expanded.json \
  --output_dir ./outputs/qwen3-finetuned \
  > "$LOG_FILE" 2>&1 &

echo $! > "$PID_FILE"
echo "Training started in background."
echo "PID: $(cat "$PID_FILE")"
echo "Log: $LOG_FILE"
