# LLM Fine-tuning Toolkit

A modular toolkit for fine-tuning Large Language Models using LoRA (Low-Rank Adaptation) with support for multiple model architectures.

## 🚀 Features

- **Multi-Model Support**: Qwen3-8B, Llama-3.2-8B, Mistral-7B
- **4-bit Quantization**: Memory-efficient training with BitsAndBytes
- **LoRA Fine-tuning**: Efficient parameter-efficient fine-tuning
- **YAML Configuration**: Easy model and training configuration
- **Modular Design**: Clean, extensible architecture

## 📦 Installation

```bash
git clone https://github.com/your-username/lora-finetune-toolkit.git
cd lora-finetune-toolkit
pip install -r requirements.txt
```

## 🛠️ Quick Start

### Basic Training

```bash
python scripts/train.py \
  --config configs/qwen3_8b.yaml \
  --data_path data/examples/dag8_expanded.json \
  --output_dir ./outputs/qwen3-finetuned
```

### Using Different Models

```bash
# Llama 3.2
python scripts/train.py --config configs/llama3_8b.yaml --data_path your_data.json

# Mistral 7B
python scripts/train.py --config configs/mistral_7b.yaml --data_path your_data.json
```

## 📁 Project Structure

```
.
├── configs/              # Model configuration files
│   ├── qwen3_8b.yaml
│   ├── llama3_8b.yaml
│   └── mistral_7b.yaml
├── data/examples/        # Example datasets
├── src/                  # Core library
│   ├── models/           # Model adapters
│   ├── data/             # Data processing
│   ├── training/         # Training utilities
│   └── utils/            # Utilities
├── scripts/              # CLI scripts
│   ├── train.py
│   └── inference.py
└── examples/             # Example scripts
```

## 📊 Supported Models

| Model | HuggingFace ID | VRAM (4-bit) |
|-------|----------------|--------------|
| Qwen3-8B | `Qwen/Qwen3-8B` | ~8GB |
| Llama-3.2-8B | `meta-llama/Llama-3.2-8B-Instruct` | ~8GB |
| Mistral-7B | `mistralai/Mistral-7B-Instruct-v0.3` | ~6GB |

## 📝 Data Format

Your training data should be a JSON file with the following format:

```json
[
  {"input": "User question or prompt", "output": "Expected model response"},
  {"input": "Another question", "output": "Another response"}
]
```

## ⚙️ Configuration

Each model config file supports:

```yaml
model:
  name: "Qwen/Qwen3-8B"
  device_map: "auto"
  
quantization:
  load_in_4bit: true
  bnb_4bit_quant_type: "nf4"
  
lora:
  r: 32
  lora_alpha: 16
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
  
training:
  num_epochs: 10
  learning_rate: 3e-4
  batch_size: 16
```

## 🔧 Inference

```bash
python scripts/inference.py \
  --adapter_path ./outputs/qwen3-finetuned \
  --prompt "Your test prompt here"
```

## 📋 Requirements

- Python 3.10+
- CUDA-compatible GPU (16GB+ VRAM recommended)
- PyTorch 2.0+

## 📄 License

MIT License
