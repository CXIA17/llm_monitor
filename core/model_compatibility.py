#!/usr/bin/env python3
"""
Model Compatibility Layer
=========================

Handles differences between model architectures for:
- Layer detection and hooking
- Activation extraction
- Tokenizer behavior
- Memory-efficient loading

Supported Architectures:
- GPT-2 / GPT-Neo / GPT-J
- LLaMA / LLaMA-2 / LLaMA-3
- Mistral / Mixtral
- Phi-1 / Phi-2 / Phi-3
- Qwen / Qwen-2
- Gemma / Gemma-2
- Falcon
- MPT
- OLMo
- StableLM
- And more via generic fallback

Usage:
    compat = ModelCompatibility(model, tokenizer)
    layer = compat.get_layer(6)
    hidden_size = compat.hidden_size
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum

import torch
import torch.nn as nn


class ModelFamily(Enum):
    """Supported model families."""
    GPT2 = "gpt2"
    GPTNEO = "gpt_neo"
    GPTJ = "gptj"
    LLAMA = "llama"
    MISTRAL = "mistral"
    MIXTRAL = "mixtral"
    PHI = "phi"
    PHI3 = "phi3"
    QWEN = "qwen"
    QWEN2 = "qwen2"
    QWEN3 = "qwen3"
    GEMMA = "gemma"
    GEMMA2 = "gemma2"
    FALCON = "falcon"
    MPT = "mpt"
    OLMO = "olmo"
    STABLELM = "stablelm"
    BLOOM = "bloom"
    OPT = "opt"
    PYTHIA = "pythia"  # Same as GPT-NeoX
    GENERIC = "generic"


@dataclass
class ArchitectureConfig:
    """Configuration for a specific model architecture."""
    family: ModelFamily
    layer_pattern: str  # Regex pattern to find transformer layers
    layer_module_path: str  # Dot-path to layer container (e.g., "model.layers")
    hidden_size_attr: str  # Config attribute for hidden size
    num_layers_attr: str  # Config attribute for number of layers
    attention_module: str  # Name of attention submodule within layer
    mlp_module: str  # Name of MLP/FFN submodule within layer
    layernorm_module: str  # Name of layer norm module
    supports_flash_attention: bool = True
    rope_scaling_supported: bool = False
    default_dtype: torch.dtype = torch.float16


# Architecture configurations
ARCHITECTURE_CONFIGS: Dict[ModelFamily, ArchitectureConfig] = {
    ModelFamily.GPT2: ArchitectureConfig(
        family=ModelFamily.GPT2,
        layer_pattern=r"transformer\.h\.(\d+)$",
        layer_module_path="transformer.h",
        hidden_size_attr="n_embd",
        num_layers_attr="n_layer",
        attention_module="attn",
        mlp_module="mlp",
        layernorm_module="ln_1",
        supports_flash_attention=False,
    ),
    ModelFamily.GPTNEO: ArchitectureConfig(
        family=ModelFamily.GPTNEO,
        layer_pattern=r"transformer\.h\.(\d+)$",
        layer_module_path="transformer.h",
        hidden_size_attr="hidden_size",
        num_layers_attr="num_layers",
        attention_module="attn",
        mlp_module="mlp",
        layernorm_module="ln_1",
        supports_flash_attention=False,
    ),
    ModelFamily.GPTJ: ArchitectureConfig(
        family=ModelFamily.GPTJ,
        layer_pattern=r"transformer\.h\.(\d+)$",
        layer_module_path="transformer.h",
        hidden_size_attr="n_embd",
        num_layers_attr="n_layer",
        attention_module="attn",
        mlp_module="mlp",
        layernorm_module="ln_1",
        supports_flash_attention=False,
    ),
    ModelFamily.LLAMA: ArchitectureConfig(
        family=ModelFamily.LLAMA,
        layer_pattern=r"model\.layers\.(\d+)$",
        layer_module_path="model.layers",
        hidden_size_attr="hidden_size",
        num_layers_attr="num_hidden_layers",
        attention_module="self_attn",
        mlp_module="mlp",
        layernorm_module="input_layernorm",
        rope_scaling_supported=True,
    ),
    ModelFamily.MISTRAL: ArchitectureConfig(
        family=ModelFamily.MISTRAL,
        layer_pattern=r"model\.layers\.(\d+)$",
        layer_module_path="model.layers",
        hidden_size_attr="hidden_size",
        num_layers_attr="num_hidden_layers",
        attention_module="self_attn",
        mlp_module="mlp",
        layernorm_module="input_layernorm",
        rope_scaling_supported=True,
    ),
    ModelFamily.MIXTRAL: ArchitectureConfig(
        family=ModelFamily.MIXTRAL,
        layer_pattern=r"model\.layers\.(\d+)$",
        layer_module_path="model.layers",
        hidden_size_attr="hidden_size",
        num_layers_attr="num_hidden_layers",
        attention_module="self_attn",
        mlp_module="block_sparse_moe",  # MoE layer
        layernorm_module="input_layernorm",
        rope_scaling_supported=True,
    ),
    ModelFamily.PHI: ArchitectureConfig(
        family=ModelFamily.PHI,
        layer_pattern=r"model\.layers\.(\d+)$",
        layer_module_path="model.layers",
        hidden_size_attr="hidden_size",
        num_layers_attr="num_hidden_layers",
        attention_module="self_attn",
        mlp_module="mlp",
        layernorm_module="input_layernorm",
    ),
    ModelFamily.PHI3: ArchitectureConfig(
        family=ModelFamily.PHI3,
        layer_pattern=r"model\.layers\.(\d+)$",
        layer_module_path="model.layers",
        hidden_size_attr="hidden_size",
        num_layers_attr="num_hidden_layers",
        attention_module="self_attn",
        mlp_module="mlp",
        layernorm_module="input_layernorm",
        rope_scaling_supported=True,
    ),
    ModelFamily.QWEN: ArchitectureConfig(
        family=ModelFamily.QWEN,
        layer_pattern=r"transformer\.h\.(\d+)$",
        layer_module_path="transformer.h",
        hidden_size_attr="hidden_size",
        num_layers_attr="num_hidden_layers",
        attention_module="attn",
        mlp_module="mlp",
        layernorm_module="ln_1",
    ),
    ModelFamily.QWEN2: ArchitectureConfig(
        family=ModelFamily.QWEN2,
        layer_pattern=r"model\.layers\.(\d+)$",
        layer_module_path="model.layers",
        hidden_size_attr="hidden_size",
        num_layers_attr="num_hidden_layers",
        attention_module="self_attn",
        mlp_module="mlp",
        layernorm_module="input_layernorm",
        rope_scaling_supported=True,
    ),
    ModelFamily.QWEN3: ArchitectureConfig(
        family=ModelFamily.QWEN3,
        layer_pattern=r"model\.layers\.(\d+)$",
        layer_module_path="model.layers",
        hidden_size_attr="hidden_size",
        num_layers_attr="num_hidden_layers",
        attention_module="self_attn",
        mlp_module="mlp",
        layernorm_module="input_layernorm",
        rope_scaling_supported=True,
    ),
    ModelFamily.GEMMA: ArchitectureConfig(
        family=ModelFamily.GEMMA,
        layer_pattern=r"model\.layers\.(\d+)$",
        layer_module_path="model.layers",
        hidden_size_attr="hidden_size",
        num_layers_attr="num_hidden_layers",
        attention_module="self_attn",
        mlp_module="mlp",
        layernorm_module="input_layernorm",
        rope_scaling_supported=True,
    ),
    ModelFamily.GEMMA2: ArchitectureConfig(
        family=ModelFamily.GEMMA2,
        layer_pattern=r"model\.layers\.(\d+)$",
        layer_module_path="model.layers",
        hidden_size_attr="hidden_size",
        num_layers_attr="num_hidden_layers",
        attention_module="self_attn",
        mlp_module="mlp",
        layernorm_module="input_layernorm",
        rope_scaling_supported=True,
    ),
    ModelFamily.FALCON: ArchitectureConfig(
        family=ModelFamily.FALCON,
        layer_pattern=r"transformer\.h\.(\d+)$",
        layer_module_path="transformer.h",
        hidden_size_attr="hidden_size",
        num_layers_attr="num_hidden_layers",
        attention_module="self_attention",
        mlp_module="mlp",
        layernorm_module="input_layernorm",
    ),
    ModelFamily.MPT: ArchitectureConfig(
        family=ModelFamily.MPT,
        layer_pattern=r"transformer\.blocks\.(\d+)$",
        layer_module_path="transformer.blocks",
        hidden_size_attr="d_model",
        num_layers_attr="n_layers",
        attention_module="attn",
        mlp_module="ffn",
        layernorm_module="norm_1",
        supports_flash_attention=True,
    ),
    ModelFamily.OLMO: ArchitectureConfig(
        family=ModelFamily.OLMO,
        layer_pattern=r"model\.transformer\.blocks\.(\d+)$",
        layer_module_path="model.transformer.blocks",
        hidden_size_attr="hidden_size",
        num_layers_attr="n_layers",
        attention_module="attn",
        mlp_module="ff",
        layernorm_module="attn_norm",
    ),
    ModelFamily.STABLELM: ArchitectureConfig(
        family=ModelFamily.STABLELM,
        layer_pattern=r"model\.layers\.(\d+)$",
        layer_module_path="model.layers",
        hidden_size_attr="hidden_size",
        num_layers_attr="num_hidden_layers",
        attention_module="self_attn",
        mlp_module="mlp",
        layernorm_module="input_layernorm",
    ),
    ModelFamily.BLOOM: ArchitectureConfig(
        family=ModelFamily.BLOOM,
        layer_pattern=r"transformer\.h\.(\d+)$",
        layer_module_path="transformer.h",
        hidden_size_attr="hidden_size",
        num_layers_attr="n_layer",
        attention_module="self_attention",
        mlp_module="mlp",
        layernorm_module="input_layernorm",
        supports_flash_attention=False,
    ),
    ModelFamily.OPT: ArchitectureConfig(
        family=ModelFamily.OPT,
        layer_pattern=r"model\.decoder\.layers\.(\d+)$",
        layer_module_path="model.decoder.layers",
        hidden_size_attr="hidden_size",
        num_layers_attr="num_hidden_layers",
        attention_module="self_attn",
        mlp_module="fc1",  # OPT uses fc1/fc2 instead of mlp
        layernorm_module="self_attn_layer_norm",
        supports_flash_attention=False,
    ),
    ModelFamily.PYTHIA: ArchitectureConfig(
        family=ModelFamily.PYTHIA,
        layer_pattern=r"gpt_neox\.layers\.(\d+)$",
        layer_module_path="gpt_neox.layers",
        hidden_size_attr="hidden_size",
        num_layers_attr="num_hidden_layers",
        attention_module="attention",
        mlp_module="mlp",
        layernorm_module="input_layernorm",
    ),
}


class ModelCompatibility:
    """
    Unified compatibility layer for different model architectures.
    
    Handles:
    - Layer detection and access
    - Hidden size and layer count detection
    - Architecture-specific quirks
    - Tokenizer normalization
    """
    
    # Model type string to family mapping
    MODEL_TYPE_MAPPING = {
        "gpt2": ModelFamily.GPT2,
        "gpt_neo": ModelFamily.GPTNEO,
        "gpt-neo": ModelFamily.GPTNEO,
        "gptneo": ModelFamily.GPTNEO,
        "gptj": ModelFamily.GPTJ,
        "gpt-j": ModelFamily.GPTJ,
        "llama": ModelFamily.LLAMA,
        "mistral": ModelFamily.MISTRAL,
        "mixtral": ModelFamily.MIXTRAL,
        "phi": ModelFamily.PHI,
        "phi3": ModelFamily.PHI3,
        "phi-3": ModelFamily.PHI3,
        "qwen": ModelFamily.QWEN,
        "qwen2": ModelFamily.QWEN2,
        "qwen3": ModelFamily.QWEN3,
        "qwen3_moe": ModelFamily.QWEN3,
        "gemma": ModelFamily.GEMMA,
        "gemma2": ModelFamily.GEMMA2,
        "falcon": ModelFamily.FALCON,
        "mpt": ModelFamily.MPT,
        "olmo": ModelFamily.OLMO,
        "stablelm": ModelFamily.STABLELM,
        "stable-lm": ModelFamily.STABLELM,
        "bloom": ModelFamily.BLOOM,
        "opt": ModelFamily.OPT,
        "gpt_neox": ModelFamily.PYTHIA,
        "pythia": ModelFamily.PYTHIA,
    }
    
    def __init__(self, model, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = getattr(model, "config", None)
        
        # Detect architecture
        self.family = self._detect_family()
        self.arch_config = ARCHITECTURE_CONFIGS.get(
            self.family, 
            self._create_generic_config()
        )
        
        # Cache layer modules
        self._layer_cache: Dict[int, nn.Module] = {}
        self._build_layer_cache()
        
        # Normalize tokenizer
        if tokenizer:
            self._normalize_tokenizer()
    
    def _detect_family(self) -> ModelFamily:
        """Detect the model family from config or architecture."""
        if self.config is None:
            return ModelFamily.GENERIC
        
        # Try model_type first
        model_type = getattr(self.config, "model_type", "").lower()
        if model_type in self.MODEL_TYPE_MAPPING:
            return self.MODEL_TYPE_MAPPING[model_type]
        
        # Try architectures list
        architectures = getattr(self.config, "architectures", [])
        for arch in architectures:
            arch_lower = arch.lower()
            for key, family in self.MODEL_TYPE_MAPPING.items():
                if key in arch_lower:
                    return family
        
        # Try to detect from module names
        module_names = [name for name, _ in self.model.named_modules()]
        
        if any("gpt_neox" in name for name in module_names):
            return ModelFamily.PYTHIA
        if any("model.layers" in name for name in module_names):
            # Could be LLaMA-style
            return ModelFamily.LLAMA
        if any("transformer.h" in name for name in module_names):
            # Could be GPT2-style
            return ModelFamily.GPT2
        
        return ModelFamily.GENERIC
    
    def _create_generic_config(self) -> ArchitectureConfig:
        """Create a generic config by probing the model structure."""
        # Try to find layers
        layer_patterns = [
            (r"model\.layers\.(\d+)$", "model.layers"),
            (r"transformer\.h\.(\d+)$", "transformer.h"),
            (r"transformer\.layers\.(\d+)$", "transformer.layers"),
            (r"decoder\.layers\.(\d+)$", "decoder.layers"),
            (r"layers\.(\d+)$", "layers"),
        ]
        
        module_names = [name for name, _ in self.model.named_modules()]
        
        for pattern, path in layer_patterns:
            if any(re.search(pattern, name) for name in module_names):
                return ArchitectureConfig(
                    family=ModelFamily.GENERIC,
                    layer_pattern=pattern,
                    layer_module_path=path,
                    hidden_size_attr="hidden_size",
                    num_layers_attr="num_hidden_layers",
                    attention_module="self_attn",
                    mlp_module="mlp",
                    layernorm_module="input_layernorm",
                )
        
        # Absolute fallback
        return ArchitectureConfig(
            family=ModelFamily.GENERIC,
            layer_pattern=r"layers?\.(\d+)$",
            layer_module_path="layers",
            hidden_size_attr="hidden_size",
            num_layers_attr="num_hidden_layers",
            attention_module="attn",
            mlp_module="mlp",
            layernorm_module="norm",
        )
    
    def _build_layer_cache(self):
        """Build cache of layer modules for fast access."""
        pattern = self.arch_config.layer_pattern
        
        for name, module in self.model.named_modules():
            match = re.search(pattern, name)
            if match:
                layer_idx = int(match.group(1))
                self._layer_cache[layer_idx] = module
    
    def _normalize_tokenizer(self):
        """Ensure tokenizer has required attributes."""
        if self.tokenizer is None:
            return
        
        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        # Ensure padding side is set (left for causal LM)
        if not hasattr(self.tokenizer, 'padding_side') or self.tokenizer.padding_side is None:
            self.tokenizer.padding_side = 'left'
    
    # === Properties ===
    
    @property
    def hidden_size(self) -> int:
        """Get the hidden size of the model."""
        if self.config is None:
            return 768  # Default fallback
        
        # Try architecture-specific attribute
        size = getattr(self.config, self.arch_config.hidden_size_attr, None)
        if size is not None:
            return size
        
        # Try common alternatives
        for attr in ["hidden_size", "n_embd", "d_model", "embed_dim"]:
            size = getattr(self.config, attr, None)
            if size is not None:
                return size
        
        return 768
    
    @property
    def num_layers(self) -> int:
        """Get the number of transformer layers."""
        if self.config is None:
            return len(self._layer_cache) or 12
        
        # Try architecture-specific attribute
        num = getattr(self.config, self.arch_config.num_layers_attr, None)
        if num is not None:
            return num
        
        # Try common alternatives
        for attr in ["num_hidden_layers", "n_layer", "num_layers", "n_layers"]:
            num = getattr(self.config, attr, None)
            if num is not None:
                return num
        
        return len(self._layer_cache) or 12
    
    @property
    def max_position_embeddings(self) -> int:
        """Get maximum sequence length."""
        if self.config is None:
            return 2048
        
        for attr in ["max_position_embeddings", "n_positions", "max_seq_len", "seq_length"]:
            val = getattr(self.config, attr, None)
            if val is not None:
                return val
        
        return 2048
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        if self.config is None:
            return 50257
        return getattr(self.config, "vocab_size", 50257)
    
    # === Layer Access ===
    
    def get_layer(self, layer_idx: int) -> Optional[nn.Module]:
        """Get a specific transformer layer by index."""
        return self._layer_cache.get(layer_idx)
    
    def get_all_layers(self) -> List[Tuple[int, nn.Module]]:
        """Get all transformer layers as (index, module) pairs."""
        return sorted(self._layer_cache.items())
    
    def get_layer_range(self, start: int, end: int) -> List[Tuple[int, nn.Module]]:
        """Get layers in a specific range."""
        return [(i, m) for i, m in self._layer_cache.items() if start <= i < end]
    
    def get_attention_module(self, layer_idx: int) -> Optional[nn.Module]:
        """Get the attention submodule of a layer."""
        layer = self.get_layer(layer_idx)
        if layer is None:
            return None
        return getattr(layer, self.arch_config.attention_module, None)
    
    def get_mlp_module(self, layer_idx: int) -> Optional[nn.Module]:
        """Get the MLP/FFN submodule of a layer."""
        layer = self.get_layer(layer_idx)
        if layer is None:
            return None
        return getattr(layer, self.arch_config.mlp_module, None)
    
    # === Tokenization ===
    
    def tokenize(
        self,
        text: Union[str, List[str]],
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True,
        return_tensors: str = "pt"
    ) -> Dict[str, torch.Tensor]:
        """Tokenize text with proper settings for this model."""
        if self.tokenizer is None:
            raise ValueError("No tokenizer available")
        
        if max_length is None:
            max_length = min(self.max_position_embeddings, 2048)
        
        return self.tokenizer(
            text,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors
        )
    
    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        if self.tokenizer is None:
            raise ValueError("No tokenizer available")
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    # === Model Info ===
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        return {
            "family": self.family.value,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "max_position_embeddings": self.max_position_embeddings,
            "vocab_size": self.vocab_size,
            "architecture_config": {
                "layer_pattern": self.arch_config.layer_pattern,
                "attention_module": self.arch_config.attention_module,
                "mlp_module": self.arch_config.mlp_module,
                "supports_flash_attention": self.arch_config.supports_flash_attention,
            },
            "detected_layers": len(self._layer_cache),
            "model_type": getattr(self.config, "model_type", "unknown"),
        }
    
    def __repr__(self) -> str:
        return (
            f"ModelCompatibility(family={self.family.value}, "
            f"hidden_size={self.hidden_size}, "
            f"num_layers={self.num_layers})"
        )


# === Model Loading Utilities ===

def load_model_and_tokenizer(
    model_name: str,
    device: str = "auto",
    dtype: Optional[torch.dtype] = None,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    trust_remote_code: bool = True,
    use_flash_attention: bool = False,
) -> Tuple[Any, Any, ModelCompatibility]:
    """
    Load a model and tokenizer with proper settings.
    
    Args:
        model_name: HuggingFace model name or path
        device: Device to load on ("auto", "cuda", "cpu", "cuda:0", etc.)
        dtype: Data type (None for auto-detect)
        load_in_8bit: Use 8-bit quantization (requires bitsandbytes)
        load_in_4bit: Use 4-bit quantization (requires bitsandbytes)
        trust_remote_code: Trust remote code for custom models
        use_flash_attention: Use Flash Attention 2 if available
        
    Returns:
        (model, tokenizer, compatibility)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"Loading model: {model_name}")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code
    )
    
    # Model loading kwargs
    model_kwargs = {
        "trust_remote_code": trust_remote_code,
    }
    
    # Device map
    if device == "auto":
        model_kwargs["device_map"] = "auto"
    elif device != "cpu":
        model_kwargs["device_map"] = device
    
    # Data type
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    elif torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.float16
    
    # Quantization
    if load_in_8bit or load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
            
            if load_in_4bit:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            else:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
        except ImportError:
            print("Warning: bitsandbytes not available. Loading without quantization.")
    
    # Flash Attention
    if use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    
    # Move to device if not using device_map
    if "device_map" not in model_kwargs and device != "cpu":
        model = model.to(device if device != "auto" else "cuda")
    
    model.eval()
    
    # Create compatibility layer
    compat = ModelCompatibility(model, tokenizer)
    
    print(f"Model loaded: {compat}")
    
    return model, tokenizer, compat


def get_device(device_str: str = "auto") -> str:
    """Get the appropriate device string."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return device_str