
import argparse
import json
import time
import gc
from tqdm import tqdm
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from sentencepiece import SentencePieceProcessor
from .persist import ModelPersister
import psutil
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, AutoModel, GenerationMixin, LlamaForCausalLM, GenerationConfig
from torch.cuda.amp import autocast
from transformers.models.llama.modeling_llama import LlamaAttention
from .utils import clean_memory, load_layer, find_or_create_local_splitted_path

@dataclass
class ModelArgs:
    dim: int
    n_layers: int
    head_dim: int
    hidden_dim: int
    n_heads: int
    n_kv_heads: int
    norm_eps: float
    vocab_size: int
    rope_theta: float
    rope_traditional: bool = True

def sanitize_config(config, weights=None):
    config.pop("model_type", None)
    n_heads = config["n_heads"] if 'n_heads' in config else config['num_attention_heads']
    if "n_kv_heads" not in config:
        config["n_kv_heads"] = n_heads
    if "head_dim" not in config:
        config["head_dim"] = config["dim"] // n_heads
    if "rope_theta" not in config:
        config["rope_theta"] = 10000
    unused = ["multiple_of", "ffn_dim_multiplier"]
    for k in unused:
        config.pop(k, None)
    return config

def get_model_args_from_config(config):
    params = {}
    params["dim"] = config.hidden_size
    params["hidden_dim"] = config.intermediate_size
    params["n_heads"] = config.num_attention_heads
    if hasattr(config, "num_key_value_heads"):
        params["n_kv_heads"] = config.num_key_value_heads
    params["n_layers"] = config.num_hidden_layers
    params["vocab_size"] = config.vocab_size
    params["norm_eps"] = config.rms_norm_eps
    params["rope_traditional"] = False

    sconfig = sanitize_config(params)
    model_args = ModelArgs(**sconfig)
    return model_args

class AirLLMLlamaNemotron:
    def __init__(self, pretrained_model_name_or_path, *inputs, **kwargs):
        self.config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)
        self.model_args = get_model_args_from_config(self.config)
        self.model = LlamaForCausalLM.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
        self.model = torch.quantization.quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8)
        self.model.config.use_cache = False  # Disable caching to save memory
        self.model.config.attention_probs_dropout_prob = 0.1  # Reduce dropout for faster inference
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)
        self.generation_config = GenerationConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)

    def generate(self, input_text, **kwargs):
        # Efficient tokenization
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        
        # Batch processing
        batch_size = 8
        input_ids = inputs['input_ids'].repeat(batch_size, 1).cuda()
        
        # Use mixed precision
        with autocast():
            # Generate outputs
            outputs = self.model.generate(input_ids, **kwargs)
        
        # Clean memory
        clean_memory()
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
