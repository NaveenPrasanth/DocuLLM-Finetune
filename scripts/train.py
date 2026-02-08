"""Main training entry point for DocuMind QLoRA fine-tuning.

Usage:
    python scripts/train.py
    python scripts/train.py --config configs/training/qlora_qwen2vl_cord.yaml
    python scripts/train.py --overrides training.args.learning_rate=1e-4 training.args.num_train_epochs=5
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

import torch
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

# Ensure project root is on sys.path for imports
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config import load_base_config, load_training_config

console = Console()


def setup_logging(log_level: str = "INFO") -> None:
    """Configure rich-based logging."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


def parse_overrides(override_strings: list[str] | None) -> dict[str, Any]:
    """Parse key=value override strings into a nested dict.

    Supports dot-notation keys like ``training.args.learning_rate=1e-4``.
    Values are auto-cast to int, float, or bool where possible.

    Args:
        override_strings: List of ``"key=value"`` strings.

    Returns:
        Flat dict suitable for OmegaConf merge.
    """
    if not override_strings:
        return {}

    overrides: dict[str, Any] = {}
    for item in override_strings:
        if "=" not in item:
            console.print(f"[yellow]Warning: skipping invalid override '{item}' (missing '=')[/yellow]")
            continue

        key, value_str = item.split("=", 1)
        key = key.strip()
        value_str = value_str.strip()

        # Auto-cast values
        value: Any
        if value_str.lower() in ("true", "yes"):
            value = True
        elif value_str.lower() in ("false", "no"):
            value = False
        elif value_str.lower() in ("none", "null"):
            value = None
        else:
            try:
                value = int(value_str)
            except ValueError:
                try:
                    value = float(value_str)
                except ValueError:
                    value = value_str

        overrides[key] = value

    return overrides


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_config_summary(
    base_config: Any,
    training_config: Any,
) -> None:
    """Display a rich-formatted summary of the training configuration."""
    table = Table(title="Training Configuration", show_header=True, header_style="bold cyan")
    table.add_column("Parameter", style="dim", width=35)
    table.add_column("Value", width=45)

    # Model
    table.add_row("Model", base_config.model.name)
    table.add_row("Processor", base_config.model.processor_name)

    # Quantization
    q = training_config.training.quantization
    table.add_row("Quantization", f"4-bit NF4 (double_quant={q.bnb_4bit_use_double_quant})")

    # LoRA
    lora = training_config.training.lora
    table.add_row("LoRA Rank (r)", str(lora.r))
    table.add_row("LoRA Alpha", str(lora.lora_alpha))
    table.add_row("LoRA Dropout", str(lora.lora_dropout))
    table.add_row("LoRA Targets", ", ".join(lora.target_modules))

    # Training
    args = training_config.training.args
    table.add_row("Epochs", str(args.num_train_epochs))
    table.add_row("Learning Rate", f"{args.learning_rate:.2e}")
    table.add_row("Batch Size", str(args.per_device_train_batch_size))
    table.add_row("Gradient Accum", str(args.gradient_accumulation_steps))
    table.add_row("Effective Batch", str(args.per_device_train_batch_size * args.gradient_accumulation_steps))
    table.add_row("LR Scheduler", args.lr_scheduler_type)
    table.add_row("Warmup Steps", str(args.warmup_steps))
    table.add_row("Max Seq Length", str(args.max_seq_length))
    table.add_row("Precision", "bf16" if args.bf16 else "fp16" if args.fp16 else "fp32")

    # Dataset
    ds = training_config.training.dataset
    table.add_row("Dataset", ds.name)
    table.add_row("Synthetic Data", str(ds.include_synthetic))

    # Output
    table.add_row("Output Dir", training_config.training.output.dir)
    table.add_row("Report To", args.report_to)

    console.print(table)


def print_gpu_info() -> None:
    """Display GPU information using rich."""
    if not torch.cuda.is_available():
        console.print("[yellow]No GPU detected. Training will be very slow on CPU.[/yellow]")
        return

    gpu_table = Table(title="GPU Information", show_header=True, header_style="bold green")
    gpu_table.add_column("Property", style="dim")
    gpu_table.add_column("Value")

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        gpu_table.add_row(f"GPU {i} Name", props.name)
        gpu_table.add_row(f"GPU {i} Memory", f"{props.total_mem / (1024**3):.1f} GB")
        gpu_table.add_row(f"GPU {i} Compute", f"{props.major}.{props.minor}")

    console.print(gpu_table)


def main() -> None:
    """Main training entry point."""
    parser = argparse.ArgumentParser(
        description="DocuMind QLoRA Fine-Tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training/qlora_qwen2vl_cord.yaml",
        help="Path to training config YAML (default: configs/training/qlora_qwen2vl_cord.yaml)",
    )
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=None,
        help="Config overrides as key=value pairs (e.g., training.args.learning_rate=1e-4)",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )
    args = parser.parse_args()

    # ── Setup ─────────────────────────────────────────────────────────────
    base_config = load_base_config()
    setup_logging(base_config.project.log_level)
    logger = logging.getLogger(__name__)

    console.print(Panel.fit(
        "[bold blue]DocuMind[/bold blue] - QLoRA Fine-Tuning Pipeline",
        subtitle="Phase 5: Fine-Tuning",
    ))

    # Parse overrides
    overrides = parse_overrides(args.overrides)
    if args.no_wandb:
        overrides["training.args.report_to"] = "none"

    # Load configs
    training_config = load_training_config(args.config, overrides=overrides if overrides else None)

    # Set seed
    seed = base_config.project.seed
    set_seed(seed)
    logger.info("Random seed set to %d", seed)

    # Display configuration
    print_gpu_info()
    print_config_summary(base_config, training_config)

    # ── Initialize W&B ────────────────────────────────────────────────────
    if training_config.training.args.report_to == "wandb":
        try:
            import wandb

            wandb.init(
                project=base_config.wandb.project,
                entity=base_config.wandb.entity,
                tags=base_config.wandb.tags,
                config={
                    "model": base_config.model.name,
                    "lora_r": training_config.training.lora.r,
                    "lora_alpha": training_config.training.lora.lora_alpha,
                    "learning_rate": training_config.training.args.learning_rate,
                    "epochs": training_config.training.args.num_train_epochs,
                    "batch_size": training_config.training.args.per_device_train_batch_size,
                    "gradient_accumulation": training_config.training.args.gradient_accumulation_steps,
                    "dataset": training_config.training.dataset.name,
                    "max_seq_length": training_config.training.args.max_seq_length,
                },
            )
            logger.info("W&B initialized: project=%s", base_config.wandb.project)
        except ImportError:
            logger.warning("wandb not installed, disabling W&B logging")
            overrides["training.args.report_to"] = "none"
            training_config = load_training_config(args.config, overrides=overrides)
    else:
        logger.info("W&B logging disabled")

    # ── Run Training ──────────────────────────────────────────────────────
    console.print("\n[bold green]Starting training...[/bold green]\n")

    from src.training.trainer import train as run_training

    trainer = run_training(
        config_path=args.config,
        overrides=overrides if overrides else None,
    )

    # ── Post-Training Summary ─────────────────────────────────────────────
    console.print("\n[bold green]Training complete![/bold green]\n")

    # Display final metrics
    if trainer.state.log_history:
        final_table = Table(title="Final Training Metrics", header_style="bold green")
        final_table.add_column("Metric", style="dim")
        final_table.add_column("Value")

        final_entry = trainer.state.log_history[-1]
        for key, value in sorted(final_entry.items()):
            if key != "epoch":
                if isinstance(value, float):
                    final_table.add_row(key, f"{value:.6f}")
                else:
                    final_table.add_row(key, str(value))

        console.print(final_table)

    # Log adapter location
    output_dir = training_config.training.output.dir
    adapter_dir = Path(output_dir) / "adapter"
    if adapter_dir.exists():
        console.print(f"\n[bold]Adapter saved to:[/bold] {adapter_dir}")

    # Finish W&B
    if training_config.training.args.report_to == "wandb":
        try:
            import wandb

            if wandb.run is not None:
                wandb.finish()
                logger.info("W&B run finished")
        except ImportError:
            pass

    console.print("\n[bold blue]Done![/bold blue]")


if __name__ == "__main__":
    main()
