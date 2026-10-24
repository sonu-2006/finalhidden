"""LLM loading utilities for trajectory generation.

Supports Qwen, Llama-3.1, BioMistral, and other HuggingFace models
with automatic device placement and tokenizer configuration.
"""

import os
from pathlib import Path

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerFast,
)

try:
    from transformers import Mistral3ForConditionalGeneration
except Exception:
    Mistral3ForConditionalGeneration = None


def load_model(
    model_name: str, device: str | None = None, dtype: torch.dtype = torch.float16
) -> tuple:
    """Load a pretrained causal LM with tokenizer.

    Args:
        model_name: HuggingFace model name or local path.
        device: Target device ('cuda', 'cpu'). Auto-detected if None.
        dtype: Model precision (default: float16).

    Returns:
        Tuple of (model, tokenizer, device).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        torch.is_autocast_enabled("cuda")
    except TypeError:
        _orig = torch.is_autocast_enabled
        torch.is_autocast_enabled = lambda *args, **kwargs: _orig()

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True, trust_remote_code=True
        )
    except Exception as e:
        if "TokenizersBackend" in str(e):
            tok_file = str(Path(model_name) / "tokenizer.json")
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=tok_file,
                bos_token="<s>",
                eos_token="</s>",
                unk_token="<unk>",
                pad_token="<pad>",
            )
        else:
            raise

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if config.model_type == "mistral3" and Mistral3ForConditionalGeneration is not None:
        model = Mistral3ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype if device.startswith("cuda") else None,
            device_map=None,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype if device.startswith("cuda") else None,
            device_map=None,
            trust_remote_code=True,
        )

    model.to(device)
    model.eval()
    return model, tokenizer, device
