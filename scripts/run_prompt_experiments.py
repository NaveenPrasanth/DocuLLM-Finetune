"""CLI script to run prompt engineering experiments on the Qwen2-VL base model.

Usage:
    python scripts/run_prompt_experiments.py
    python scripts/run_prompt_experiments.py --num-samples 20 --strategies zero_shot_basic zero_shot_structured
    python scripts/run_prompt_experiments.py --model Qwen/Qwen2-VL-2B-Instruct --output-dir outputs/prompt_experiments
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure the project root is on the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
from rich.console import Console
from rich.table import Table

from src.config import load_base_config
from src.data.cord_loader import load_cord_dataset
from src.prompts.strategies import (
    BaseStrategy,
    get_all_strategies,
    get_strategy,
    list_strategies,
)
from src.prompts.experiment_runner import PromptExperimentRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
console = Console()


def load_model_and_processor(
    model_name: str,
    dtype: str = "bfloat16",
) -> tuple:
    """Load Qwen2-VL model and processor.

    Args:
        model_name: HuggingFace model name or local path.
        dtype: Torch dtype string (bfloat16, float16, float32).

    Returns:
        Tuple of (model, processor).
    """
    from transformers import AutoModelForCausalLM, AutoProcessor

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype, torch.bfloat16)

    console.print(f"[bold blue]Loading model:[/bold blue] {model_name}")
    console.print(f"[bold blue]Dtype:[/bold blue] {dtype}")

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    console.print("[bold green]Model loaded successfully.[/bold green]")
    return model, processor


def build_strategies(
    strategy_names: list[str] | None,
    example_samples: list[dict] | None = None,
) -> list[BaseStrategy]:
    """Build strategy instances from names.

    Args:
        strategy_names: List of strategy names, or None for all.
        example_samples: Optional samples for few-shot strategies.

    Returns:
        List of instantiated strategies.
    """
    if strategy_names is None:
        return get_all_strategies(example_samples=example_samples)

    strategies = []
    for name in strategy_names:
        kwargs = {}
        if "few_shot" in name and example_samples:
            kwargs["example_samples"] = example_samples
        strategies.append(get_strategy(name, **kwargs))
    return strategies


def print_comparison_table(comparison: list[dict]) -> None:
    """Print a formatted comparison table using rich.

    Args:
        comparison: List of per-strategy summary dicts.
    """
    table = Table(
        title="Prompt Strategy Comparison",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Rank", style="dim", width=4, justify="right")
    table.add_column("Strategy", style="cyan", min_width=20)
    table.add_column("JSON Valid", justify="right")
    table.add_column("Avg F1", justify="right")
    table.add_column("Micro F1", justify="right")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("Time (s)", justify="right")

    for i, row in enumerate(comparison):
        # Highlight the best strategy
        style = "bold green" if i == 0 else None
        table.add_row(
            str(i + 1),
            row["strategy"],
            f"{row['json_valid_rate']:.1%}",
            f"{row['avg_f1']:.4f}",
            f"{row['micro_f1']:.4f}",
            f"{row['avg_precision']:.4f}",
            f"{row['avg_recall']:.4f}",
            f"{row['elapsed_seconds']:.1f}",
            style=style,
        )

    console.print()
    console.print(table)
    console.print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run prompt engineering experiments on Qwen2-VL base model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="HuggingFace model name. Default: from base config.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Number of test samples to evaluate (default: 50).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save results. Default: outputs/prompt_experiments.",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=None,
        help=(
            f"Strategy names to run. Default: all. "
            f"Available: {', '.join(list_strategies())}"
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Maximum tokens to generate per sample (default: 1024).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model dtype (default: bfloat16).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "validation", "test"],
        help="Dataset split to use (default: test).",
    )
    args = parser.parse_args()

    # ── Load config ──────────────────────────────────────────────────────
    base_config = load_base_config()
    model_name = args.model or base_config.model.name
    output_dir = Path(args.output_dir or "outputs/prompt_experiments")
    output_dir.mkdir(parents=True, exist_ok=True)

    console.rule("[bold]DocuMind - Prompt Engineering Experiments[/bold]")
    console.print(f"Model:       {model_name}")
    console.print(f"Samples:     {args.num_samples}")
    console.print(f"Split:       {args.split}")
    console.print(f"Output:      {output_dir}")
    console.print(f"Strategies:  {args.strategies or 'all (' + str(len(list_strategies())) + ')'}")
    console.print()

    # ── Load data ────────────────────────────────────────────────────────
    console.print("[bold blue]Loading CORD test data...[/bold blue]")
    test_samples = load_cord_dataset(split=args.split, max_samples=args.num_samples)
    console.print(f"Loaded {len(test_samples)} samples.\n")

    # Load a few training samples for few-shot examples
    console.print("[bold blue]Loading few-shot example samples...[/bold blue]")
    example_samples = load_cord_dataset(split="train", max_samples=10)
    console.print(f"Loaded {len(example_samples)} example samples.\n")

    # ── Load model ───────────────────────────────────────────────────────
    model, processor = load_model_and_processor(model_name, dtype=args.dtype)

    # ── Build strategies ─────────────────────────────────────────────────
    strategies = build_strategies(args.strategies, example_samples=example_samples)
    console.print(f"[bold blue]Strategies to run ({len(strategies)}):[/bold blue]")
    for s in strategies:
        console.print(f"  - {s.get_name()}")
    console.print()

    # ── Run experiment ───────────────────────────────────────────────────
    runner = PromptExperimentRunner(
        model=model,
        processor=processor,
        strategies=strategies,
        max_new_tokens=args.max_new_tokens,
    )

    results = runner.run_experiment(
        test_samples=test_samples,
        num_samples=args.num_samples,
    )

    # ── Save results ─────────────────────────────────────────────────────
    output_file = output_dir / "prompt_experiment_results.json"
    PromptExperimentRunner.save_results(results, output_file)

    # ── Print comparison ─────────────────────────────────────────────────
    console.rule("[bold]Results[/bold]")
    print_comparison_table(results["comparison"])

    # Print winner
    best = results["comparison"][0]
    console.print(
        f"[bold green]Best strategy: {best['strategy']} "
        f"(Avg F1: {best['avg_f1']:.4f}, Micro F1: {best['micro_f1']:.4f})[/bold green]"
    )
    console.print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
