"""LoRA configuration builder and model adapter utilities.

Builds PEFT LoraConfig objects from the project's Pydantic config models,
applies LoRA adapters, and provides helpers for target module discovery.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch import nn

from src.config import LoRAConfig as LoRAConfigModel

logger = logging.getLogger(__name__)


def build_lora_config(lora_cfg: LoRAConfigModel) -> LoraConfig:
    """Create a PEFT ``LoraConfig`` from the project's Pydantic LoRA config.

    Args:
        lora_cfg: Pydantic model containing LoRA hyperparameters (r, alpha,
            dropout, target modules, task type, bias).

    Returns:
        A ``peft.LoraConfig`` ready to be applied to a model.
    """
    config = LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.lora_dropout,
        target_modules=list(lora_cfg.target_modules),
        task_type=lora_cfg.task_type,
        bias=lora_cfg.bias,
    )

    logger.info(
        "Built LoRA config: r=%d, alpha=%d, dropout=%.3f, targets=%s, task=%s",
        lora_cfg.r,
        lora_cfg.lora_alpha,
        lora_cfg.lora_dropout,
        lora_cfg.target_modules,
        lora_cfg.task_type,
    )

    return config


def apply_lora(
    model: nn.Module,
    lora_config: LoraConfig,
    use_gradient_checkpointing: bool = True,
) -> nn.Module:
    """Apply LoRA adapters to a quantized model.

    Steps:
        1. Prepare the model for k-bit training (freeze base weights, cast
           layer norms to float32).
        2. Enable ``input_require_grads`` so gradient computation flows
           through quantized layers.
        3. Wrap the model with ``get_peft_model``.
        4. Optionally enable gradient checkpointing for memory efficiency.

    Args:
        model: A loaded (quantized) model.
        lora_config: PEFT LoraConfig specifying adapter hyperparameters.
        use_gradient_checkpointing: Whether to enable gradient checkpointing
            (recommended for QLoRA to reduce memory).

    Returns:
        The PEFT-wrapped model with LoRA adapters applied.
    """
    # Prepare the base model for quantized training
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=use_gradient_checkpointing,
    )
    logger.info("Prepared model for k-bit training")

    # Ensure inputs compute gradients (required for quantized backprop)
    model.enable_input_require_grads()
    logger.info("Enabled input_require_grads")

    # Apply LoRA adapters
    model = get_peft_model(model, lora_config)
    logger.info("Applied LoRA adapters")

    # Enable gradient checkpointing after PEFT wrapping
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    # Print trainable parameter summary
    model.print_trainable_parameters()

    return model


def find_target_modules(
    model: nn.Module,
    include_types: tuple[type, ...] | None = None,
) -> list[str]:
    """Scan a model and return names of all linear layers suitable as LoRA targets.

    This is useful for discovering which modules can receive LoRA adapters,
    especially when working with a new architecture.

    Args:
        model: A PyTorch model to inspect.
        include_types: Tuple of module types to include. Defaults to
            ``(torch.nn.Linear,)``.

    Returns:
        Sorted list of unique module name suffixes (e.g., ``["q_proj", "v_proj"]``).
    """
    if include_types is None:
        include_types = (nn.Linear,)

    target_names: set[str] = set()

    for name, module in model.named_modules():
        if isinstance(module, include_types):
            # Extract the leaf module name (e.g., "model.layers.0.self_attn.q_proj" -> "q_proj")
            parts = name.split(".")
            leaf_name = parts[-1] if parts else name
            target_names.add(leaf_name)

    sorted_names = sorted(target_names)

    logger.info(
        "Found %d unique linear module names: %s",
        len(sorted_names),
        sorted_names,
    )

    return sorted_names


def get_lora_summary(model: nn.Module) -> dict[str, Any]:
    """Return a summary of LoRA adapter information from a PEFT model.

    Args:
        model: A PEFT-wrapped model.

    Returns:
        Dict with adapter module counts and parameter statistics.
    """
    lora_params = 0
    total_params = 0
    adapter_modules: list[str] = []

    for name, param in model.named_parameters():
        total_params += param.numel()
        if "lora_" in name:
            lora_params += param.numel()
            # Get the parent module name
            module_name = ".".join(name.split(".")[:-1])
            if module_name not in adapter_modules:
                adapter_modules.append(module_name)

    return {
        "lora_params": lora_params,
        "total_params": total_params,
        "lora_percentage": 100.0 * lora_params / total_params if total_params > 0 else 0.0,
        "num_adapter_modules": len(adapter_modules),
        "adapter_modules": adapter_modules[:10],  # First 10 for display
    }
