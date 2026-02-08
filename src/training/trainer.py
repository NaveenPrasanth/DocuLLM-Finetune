"""SFTTrainer wrapper for Qwen2-VL QLoRA fine-tuning.

Orchestrates the full training pipeline: config loading, model quantization,
LoRA adapter application, dataset building, and SFTTrainer execution.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from transformers import TrainingArguments

from trl import SFTTrainer

from src.config import (
    FullTrainingConfig,
    TrainingArgsConfig,
    load_base_config,
    load_training_config,
)
from src.data.dataset_builder import build_sft_dataset
from src.training.lora_config import apply_lora, build_lora_config
from src.training.model_loader import load_processor, load_quantized_model, print_model_info

logger = logging.getLogger(__name__)


def build_training_args(
    args_config: TrainingArgsConfig,
    output_dir: str,
) -> TrainingArguments:
    """Create ``transformers.TrainingArguments`` from the project's config model.

    Args:
        args_config: Pydantic model with training hyperparameters.
        output_dir: Directory for checkpoints, logs, and saved models.

    Returns:
        Configured ``TrainingArguments``.
    """
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args_config.num_train_epochs,
        per_device_train_batch_size=args_config.per_device_train_batch_size,
        per_device_eval_batch_size=args_config.per_device_eval_batch_size,
        gradient_accumulation_steps=args_config.gradient_accumulation_steps,
        learning_rate=args_config.learning_rate,
        lr_scheduler_type=args_config.lr_scheduler_type,
        warmup_steps=args_config.warmup_steps,
        weight_decay=args_config.weight_decay,
        bf16=args_config.bf16,
        fp16=args_config.fp16,
        gradient_checkpointing=args_config.gradient_checkpointing,
        max_grad_norm=args_config.max_grad_norm,
        logging_steps=args_config.logging_steps,
        eval_steps=args_config.eval_steps,
        save_steps=args_config.save_steps,
        save_total_limit=args_config.save_total_limit,
        eval_strategy=args_config.eval_strategy,
        dataloader_num_workers=args_config.dataloader_num_workers,
        remove_unused_columns=args_config.remove_unused_columns,
        report_to=args_config.report_to,
        logging_first_step=True,
        optim="paged_adamw_8bit",
        ddp_find_unused_parameters=False,
    )

    logger.info(
        "Built TrainingArguments: epochs=%d, lr=%.2e, batch=%d, grad_accum=%d, output=%s",
        args_config.num_train_epochs,
        args_config.learning_rate,
        args_config.per_device_train_batch_size,
        args_config.gradient_accumulation_steps,
        output_dir,
    )

    return training_args


def build_sft_trainer(
    model: torch.nn.Module,
    processor: Any,
    train_dataset: Dataset,
    eval_dataset: Dataset | None,
    training_args: TrainingArguments,
    max_seq_length: int = 2048,
    callbacks: list | None = None,
) -> SFTTrainer:
    """Build a ``trl.SFTTrainer`` for Qwen2-VL fine-tuning.

    Uses the messages format (``dataset_text_field=None``) so that the
    SFTTrainer applies the chat template from the tokenizer/processor
    automatically.

    Args:
        model: Quantized + LoRA-wrapped model.
        processor: The Qwen2-VL processor (tokenizer + image processor).
        train_dataset: HuggingFace Dataset for training.
        eval_dataset: HuggingFace Dataset for evaluation (optional).
        training_args: Configured ``TrainingArguments``.
        max_seq_length: Maximum sequence length for packing/truncation.
        callbacks: Optional list of ``TrainerCallback`` instances.

    Returns:
        Configured ``SFTTrainer`` ready for ``.train()``.
    """
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "processing_class": processor.tokenizer,
        "max_seq_length": max_seq_length,
        "dataset_text_field": None,  # Use messages format
        "dataset_kwargs": {
            "skip_prepare_dataset": True,
        },
    }

    if eval_dataset is not None:
        trainer_kwargs["eval_dataset"] = eval_dataset

    if callbacks:
        trainer_kwargs["callbacks"] = callbacks

    trainer = SFTTrainer(**trainer_kwargs)

    logger.info(
        "Built SFTTrainer: max_seq_length=%d, train_samples=%d, eval_samples=%s",
        max_seq_length,
        len(train_dataset),
        len(eval_dataset) if eval_dataset is not None else "None",
    )

    return trainer


def _build_hf_dataset(samples: list[dict[str, Any]]) -> Dataset:
    """Convert a list of SFT-formatted sample dicts to a HuggingFace Dataset.

    Each sample should have a 'messages' key with the ChatML conversation.

    Args:
        samples: List of dicts from ``build_sft_dataset``.

    Returns:
        A HuggingFace ``Dataset``.
    """
    return Dataset.from_list(samples)


def train(
    config_path: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> SFTTrainer:
    """Run the full QLoRA fine-tuning pipeline.

    Steps:
        1. Load training config and base config.
        2. Load Qwen2-VL with 4-bit quantization.
        3. Apply LoRA adapters.
        4. Build training and evaluation datasets.
        5. Construct SFTTrainer and run training.
        6. Save adapter weights.

    Args:
        config_path: Path to the training YAML config. Defaults to the
            standard QLoRA CORD config.
        overrides: Optional dict of dot-notation config overrides.

    Returns:
        The ``SFTTrainer`` after training completes (for further inspection
        or continued training).
    """
    # ── 1. Load configuration ─────────────────────────────────────────────
    if config_path is None:
        config_path = "configs/training/qlora_qwen2vl_cord.yaml"

    full_config = load_training_config(config_path, overrides=overrides)
    training_cfg = full_config.training
    base_config = load_base_config()

    output_dir = training_cfg.output.dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Training config loaded from '%s'", config_path)
    logger.info("Output directory: %s", output_dir)

    # ── 2. Load quantized model ───────────────────────────────────────────
    model_name = base_config.model.name
    logger.info("Loading model: %s", model_name)

    model = load_quantized_model(
        model_name=model_name,
        quantization_config=training_cfg.quantization,
    )

    # ── 3. Load processor ─────────────────────────────────────────────────
    processor = load_processor(
        processor_name=base_config.model.processor_name,
    )

    # ── 4. Apply LoRA ─────────────────────────────────────────────────────
    lora_config = build_lora_config(training_cfg.lora)
    model = apply_lora(
        model,
        lora_config,
        use_gradient_checkpointing=training_cfg.args.gradient_checkpointing,
    )
    print_model_info(model)

    # ── 5. Build datasets ─────────────────────────────────────────────────
    dataset_cfg = training_cfg.dataset
    logger.info("Building dataset: %s", dataset_cfg.name)

    synthetic_path = None
    if dataset_cfg.include_synthetic:
        synthetic_path = (
            Path(base_config.project.data_dir) / dataset_cfg.name / "synthetic.json"
        )

    sft_splits = build_sft_dataset(
        dataset_name=dataset_cfg.name,
        include_synthetic=dataset_cfg.include_synthetic,
        synthetic_path=synthetic_path,
        synthetic_ratio=dataset_cfg.synthetic_ratio,
        seed=base_config.project.seed,
    )

    train_dataset = _build_hf_dataset(sft_splits["train"])
    eval_dataset = _build_hf_dataset(sft_splits.get("val", sft_splits.get("test", [])))

    logger.info(
        "Datasets built: train=%d, eval=%d",
        len(train_dataset),
        len(eval_dataset),
    )

    # ── 6. Build trainer and train ────────────────────────────────────────
    training_args = build_training_args(training_cfg.args, output_dir)

    trainer = build_sft_trainer(
        model=model,
        processor=processor,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if len(eval_dataset) > 0 else None,
        training_args=training_args,
        max_seq_length=training_cfg.args.max_seq_length,
    )

    logger.info("Starting training...")
    trainer.train()
    logger.info("Training complete!")

    # ── 7. Save adapter weights ───────────────────────────────────────────
    if training_cfg.output.save_adapter:
        adapter_dir = Path(output_dir) / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(adapter_dir))
        processor.save_pretrained(str(adapter_dir))
        logger.info("Adapter saved to %s", adapter_dir)

    # ── 8. Log final metrics ──────────────────────────────────────────────
    if trainer.state.log_history:
        final_metrics = trainer.state.log_history[-1]
        logger.info("Final metrics: %s", final_metrics)

    return trainer
