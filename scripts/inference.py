#!/usr/bin/env python3
"""
CLI inference script for testing fine-tuned models.

Usage:
    python scripts/inference.py --adapter_path ./outputs --prompt "Your question here"
"""

import argparse
import os
import sys
from contextlib import nullcontext

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.utils.config import load_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference with fine-tuned LoRA adapter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--adapter_path",
        type=str,
        required=True,
        help="Path to saved LoRA adapter",
    )
    
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Input prompt for generation",
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode",
    )
    
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum new tokens to generate (default: 256)",
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config (auto-detected from adapter if not specified)",
    )
    
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="Device map for model loading (default: auto)",
    )
    
    return parser.parse_args()


def load_model_with_adapter(adapter_path: str, config_path: str = None, device_map: str = "auto"):
    """
    Load base model with LoRA adapter.
    
    Args:
        adapter_path: Path to saved adapter.
        config_path: Optional path to config file.
        device_map: Device mapping.
        
    Returns:
        Tuple of (model, tokenizer).
    """
    # Try to load adapter config
    adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
    
    if config_path:
        config = load_config(config_path)
        model_name = config.get("model", {}).get("name")
    else:
        # Read base model from adapter config
        import json
        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)
        model_name = adapter_config.get("base_model_name_or_path")
    
    print(f"Loading base model: {model_name}")
    
    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        quantization_config=bnb_config,
        trust_remote_code=True,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load adapter
    print(f"Loading adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
) -> str:
    """
    Generate response for a given prompt.
    
    Args:
        model: The model with adapter.
        tokenizer: The tokenizer.
        prompt: User prompt.
        max_new_tokens: Maximum tokens to generate.
        
    Returns:
        Generated text.
    """
    # Format as chat message
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    # Tokenize
    inputs = tokenizer(formatted, return_tensors="pt", add_special_tokens=False)
    inputs = inputs.to(model.device)
    
    # Generate
    model.eval()
    
    ctx = torch.autocast(device_type=model.device.type, dtype=model.dtype) \
          if model.dtype in [torch.float16, torch.bfloat16] else nullcontext()
    
    with ctx:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
    
    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract assistant response (after user part)
    if prompt in response:
        response = response.split(prompt)[-1].strip()
    
    return response


def interactive_mode(model, tokenizer, max_new_tokens: int):
    """Run in interactive chat mode."""
    print("\n" + "=" * 60)
    print("Interactive Mode (type 'quit' or 'exit' to stop)")
    print("=" * 60 + "\n")
    
    while True:
        try:
            prompt = input("You: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not prompt:
                continue
            
            response = generate_response(model, tokenizer, prompt, max_new_tokens)
            print(f"\nAssistant: {response}\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


def main():
    """Main inference entry point."""
    args = parse_args()
    
    # Load model
    model, tokenizer = load_model_with_adapter(
        args.adapter_path,
        args.config,
        args.device_map,
    )
    
    print(f"\nModel loaded successfully!")
    print(f"Memory footprint: {model.get_memory_footprint()/1e6:.2f} MB\n")
    
    if args.interactive:
        interactive_mode(model, tokenizer, args.max_new_tokens)
    elif args.prompt:
        print(f"Prompt: {args.prompt}\n")
        response = generate_response(model, tokenizer, args.prompt, args.max_new_tokens)
        print(f"Response:\n{response}")
    else:
        print("Error: Please provide --prompt or use --interactive mode")
        sys.exit(1)


if __name__ == "__main__":
    main()
