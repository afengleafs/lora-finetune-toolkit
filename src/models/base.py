"""Base model adapter class."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from ..utils.config import get_compute_dtype


class BaseModelAdapter(ABC):
    """
    Abstract base class for model adapters.
    
    Each model adapter handles model-specific loading, tokenization,
    and prompt formatting.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the model adapter.
        
        Args:
            config: Configuration dictionary containing model, quantization,
                   lora, training, and tokenizer settings.
        """
        self.config = config
        self.model = None
        self.tokenizer = None
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the HuggingFace model identifier."""
        pass
    
    def get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """
        Create BitsAndBytes quantization configuration.
        
        Returns:
            BitsAndBytesConfig for 4-bit quantization.
        """
        quant_config = self.config.get("quantization", {})
        if not quant_config.get("load_in_4bit", True):
            return None

        try:
            import bitsandbytes  # noqa: F401
        except ImportError:
            print("Warning: bitsandbytes is not installed, disabling 4-bit quantization.")
            return None

        compute_dtype = get_compute_dtype(
            quant_config.get("bnb_4bit_compute_dtype", "float32")
        )
        
        return BitsAndBytesConfig(
            load_in_4bit=quant_config.get("load_in_4bit", True),
            bnb_4bit_quant_type=quant_config.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=quant_config.get("bnb_4bit_use_double_quant", True),
            bnb_4bit_compute_dtype=compute_dtype,
        )
    
    def get_lora_config(self) -> LoraConfig:
        """
        Create LoRA configuration.
        
        Returns:
            LoraConfig for PEFT.
        """
        lora_config = self.config.get("lora", {})
        
        return LoraConfig(
            r=lora_config.get("r", 32),
            lora_alpha=lora_config.get("lora_alpha", 16),
            lora_dropout=lora_config.get("lora_dropout", 0.05),
            bias=lora_config.get("bias", "none"),
            task_type=lora_config.get("task_type", "CAUSAL_LM"),
            target_modules=lora_config.get("target_modules", [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]),
        )
    
    def load_model(self, device_map: Optional[str] = None) -> AutoModelForCausalLM:
        """
        Load and prepare the model for training.
        
        Args:
            device_map: Device mapping strategy (default from config).
            
        Returns:
            Model prepared for LoRA training.
        """
        model_config = self.config.get("model", {})
        device_map = device_map or model_config.get("device_map", "auto")
        quantization_config = self.get_quantization_config()
        
        # Load base model with quantization
        load_kwargs = {
            "device_map": device_map,
            "trust_remote_code": True,
        }
        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **load_kwargs)

        # Training cannot proceed if the model was partially offloaded to CPU/disk/meta.
        # This usually means 4-bit quantization is unavailable and the full model does
        # not fit on the selected accelerator(s).
        device_map_values = getattr(self.model, "hf_device_map", None)
        has_offload = False
        if isinstance(device_map_values, dict):
            has_offload = any(v in {"cpu", "disk", "meta"} for v in device_map_values.values())
        has_meta_params = any(param.device.type == "meta" for param in self.model.parameters())
        if has_offload or has_meta_params:
            raise RuntimeError(
                "Model was offloaded to CPU/disk/meta during loading, which is not supported "
                "for this training setup. Install `bitsandbytes` to enable 4-bit QLoRA, "
                "or use a smaller model / explicit GPU device map that fits fully in VRAM."
            )

        # Prepare for k-bit training when quantization is enabled
        if quantization_config is not None:
            self.model = prepare_model_for_kbit_training(self.model)

        # Apply LoRA
        self.model = get_peft_model(self.model, self.get_lora_config())
        
        return self.model
    
    def load_tokenizer(self) -> AutoTokenizer:
        """
        Load and configure the tokenizer.
        
        Returns:
            Configured tokenizer.
        """
        tokenizer_config = self.config.get("tokenizer", {})
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        
        # Configure padding
        if tokenizer_config.get("use_eos_as_pad", True):
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.tokenizer.padding_side = tokenizer_config.get("padding_side", "right")
        
        return self.tokenizer
    
    def format_messages(self, input_text: str, output_text: Optional[str] = None) -> str:
        """
        Format input/output as chat messages.
        
        Args:
            input_text: User input text.
            output_text: Assistant response (optional, for training).
            
        Returns:
            Formatted text using chat template.
        """
        messages = [{"role": "user", "content": input_text}]
        
        if output_text is not None:
            messages.append({"role": "assistant", "content": output_text})
        
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=(output_text is None),
        )
    
    def get_trainable_params_info(self) -> Dict[str, Any]:
        """
        Get information about trainable parameters.
        
        Returns:
            Dictionary with parameter counts and percentages.
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        trainable, total = self.model.get_nb_trainable_parameters()
        
        return {
            "trainable_params": trainable,
            "total_params": total,
            "trainable_percent": 100 * trainable / total,
            "memory_footprint_mb": self.model.get_memory_footprint() / 1e6,
        }
    
    def print_model_info(self):
        """Print model information to console."""
        info = self.get_trainable_params_info()
        print(f"\n{'='*50}")
        print(f"Model: {self.model_name}")
        print(f"Memory footprint: {info['memory_footprint_mb']:.2f} MB")
        print(f"Trainable params: {info['trainable_params']/1e6:.2f}M")
        print(f"Total params: {info['total_params']/1e6:.2f}M")
        print(f"Trainable: {info['trainable_percent']:.2f}%")
        print(f"{'='*50}\n")
