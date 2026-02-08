"""Load Qwen2-VL-2B-Instruct with BitsAndBytes 4-bit quantization.

Provides utilities for loading the model with QLoRA-compatible quantization,
the processor with vision token limits, and model inspection helpers.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2VLForConditionalGeneration,
)

from src.config import QuantizationConfig

logger = logging.getLogger(__name__)

# Qwen2-VL vision token pixel boundaries
_PIXEL_PATCH_SIZE = 28
_DEFAULT_MIN_PIXELS = 256 * _PIXEL_PATCH_SIZE * _PIXEL_PATCH_SIZE   # 256 * 28 * 28
_DEFAULT_MAX_PIXELS = 1280 * _PIXEL_PATCH_SIZE * _PIXEL_PATCH_SIZE  # 1280 * 28 * 28

# Mapping from string dtype names to torch dtypes
_DTYPE_MAP: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _resolve_dtype(dtype_str: str) -> torch.dtype:
    """Resolve a string dtype name to a torch.dtype."""
    if dtype_str not in _DTYPE_MAP:
        raise ValueError(
            f"Unsupported dtype '{dtype_str}'. Choose from: {list(_DTYPE_MAP.keys())}"
        )
    return _DTYPE_MAP[dtype_str]


def load_quantized_model(
    model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
    quantization_config: QuantizationConfig | None = None,
) -> Qwen2VLForConditionalGeneration:
    """Load Qwen2-VL with BitsAndBytes 4-bit quantization.

    Args:
        model_name: HuggingFace model identifier or local path.
        quantization_config: Pydantic QuantizationConfig controlling quantization
            parameters.  Uses project defaults if ``None``.

    Returns:
        The quantized ``Qwen2VLForConditionalGeneration`` model with device_map="auto".
    """
    if quantization_config is None:
        quantization_config = QuantizationConfig()

    compute_dtype = _resolve_dtype(quantization_config.bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quantization_config.load_in_4bit,
        bnb_4bit_quant_type=quantization_config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=quantization_config.bnb_4bit_use_double_quant,
    )

    logger.info(
        "Loading model '%s' with 4-bit quantization (quant_type=%s, compute_dtype=%s)",
        model_name,
        quantization_config.bnb_4bit_quant_type,
        quantization_config.bnb_4bit_compute_dtype,
    )

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=compute_dtype,
    )

    logger.info("Model loaded successfully on device_map='auto'")
    print_model_info(model)

    return model


def load_processor(
    processor_name: str = "Qwen/Qwen2-VL-2B-Instruct",
    min_pixels: int = _DEFAULT_MIN_PIXELS,
    max_pixels: int = _DEFAULT_MAX_PIXELS,
) -> AutoProcessor:
    """Load the Qwen2-VL processor with vision token pixel limits.

    Args:
        processor_name: HuggingFace processor identifier or local path.
        min_pixels: Minimum number of pixels for vision input (default 256*28*28).
        max_pixels: Maximum number of pixels for vision input (default 1280*28*28).

    Returns:
        Configured ``AutoProcessor`` instance.
    """
    logger.info(
        "Loading processor '%s' (min_pixels=%d, max_pixels=%d)",
        processor_name,
        min_pixels,
        max_pixels,
    )

    processor = AutoProcessor.from_pretrained(
        processor_name,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    return processor


def print_model_info(model: torch.nn.Module) -> dict[str, Any]:
    """Print and return a summary of model parameter counts.

    Args:
        model: A PyTorch model (typically the loaded Qwen2-VL).

    Returns:
        Dict with ``total_params``, ``trainable_params``, and
        ``trainable_percentage``.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_pct = (
        100.0 * trainable_params / total_params if total_params > 0 else 0.0
    )

    info = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_percentage": trainable_pct,
    }

    logger.info(
        "Model parameters: total=%s, trainable=%s (%.4f%%)",
        f"{total_params:,}",
        f"{trainable_params:,}",
        trainable_pct,
    )
    print(f"Total parameters:     {total_params:>14,}")
    print(f"Trainable parameters: {trainable_params:>14,}")
    print(f"Trainable %%:          {trainable_pct:>13.4f}%")

    return info
