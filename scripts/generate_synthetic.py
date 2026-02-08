"""CLI script to generate synthetic training data for DocuMind.

Runs the full synthetic data generation pipeline (instruction variants,
synthetic receipts, OCR error augmentations) and saves outputs as JSON files.

Usage:
    python scripts/generate_synthetic.py --api-key sk-... --output-dir data/synthetic
    python scripts/generate_synthetic.py --num-receipts 200 --num-error-pairs 100
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from tqdm import tqdm

from src.config import get_env_var, load_base_config
from src.data.synthetic_generator import SyntheticGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────


def _save_json(data, path: Path, description: str) -> None:
    """Write *data* as pretty-printed JSON to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    logger.info("Saved %s to %s", description, path)


def _print_summary(results: dict, elapsed: float) -> None:
    """Print a human-readable summary of the generation run."""
    cost = results["cost_estimate"]
    n_inst = len(results["instruction_variants"])
    n_recv = len(results["synthetic_receipts"])
    n_err = len(results["error_augmentations"])

    print("\n" + "=" * 60)
    print("  Synthetic Data Generation - Summary")
    print("=" * 60)
    print(f"  Instruction variants : {n_inst}")
    print(f"  Synthetic receipts   : {n_recv}")
    print(f"  Error augment pairs  : {n_err}")
    print("-" * 60)
    print(f"  Input tokens         : {cost['input_tokens']:,}")
    print(f"  Output tokens        : {cost['output_tokens']:,}")
    print(f"  Estimated cost (USD) : ${cost['estimated_cost_usd']:.4f}")
    print(f"  Wall-clock time      : {elapsed:.1f}s")
    print("=" * 60 + "\n")


# ── CLI ────────────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic training data for DocuMind receipt extraction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help=(
            "Anthropic API key. If not provided, reads from the "
            "ANTHROPIC_API_KEY environment variable."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-4-20250514",
        help="Claude model to use for generation.",
    )
    parser.add_argument(
        "--num-instructions",
        type=int,
        default=20,
        help="Number of instruction variants to generate.",
    )
    parser.add_argument(
        "--num-receipts",
        type=int,
        default=100,
        help="Number of synthetic receipts to generate.",
    )
    parser.add_argument(
        "--num-error-pairs",
        type=int,
        default=50,
        help="Number of OCR error augmentation pairs to generate.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Defaults to <project_data_dir>/synthetic.",
    )
    parser.add_argument(
        "--base-instruction",
        type=str,
        default=None,
        help="Base instruction to paraphrase (uses built-in default if omitted).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the synthetic data generation CLI."""
    args = parse_args(argv)

    # Resolve API key
    api_key = args.api_key or get_env_var("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error(
            "No API key found. Provide --api-key or set ANTHROPIC_API_KEY."
        )
        sys.exit(1)

    # Resolve output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        try:
            base_config = load_base_config()
            output_dir = Path(base_config.project.data_dir) / "synthetic"
        except Exception:
            output_dir = Path("data") / "synthetic"

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", output_dir)

    # Initialise generator
    generator = SyntheticGenerator(
        api_key=api_key,
        model=args.model,
    )

    start_time = time.time()

    # ── Step 1: Instruction variants ──────────────────────────────────
    print("\n[1/3] Generating instruction variants ...")
    with tqdm(total=1, desc="Instruction variants", unit="batch") as pbar:
        instruction_variants = generator.generate_instruction_variants(
            base_instruction=args.base_instruction,
            num_variants=args.num_instructions,
        )
        pbar.update(1)

    _save_json(
        instruction_variants,
        output_dir / "instruction_variants.json",
        f"{len(instruction_variants)} instruction variants",
    )

    # ── Step 2: Synthetic receipts ────────────────────────────────────
    batch_size = 20
    num_batches = (args.num_receipts + batch_size - 1) // batch_size

    print(f"\n[2/3] Generating {args.num_receipts} synthetic receipts ...")
    synthetic_receipts: list[dict] = []
    remaining = args.num_receipts

    with tqdm(total=args.num_receipts, desc="Synthetic receipts", unit="receipt") as pbar:
        while remaining > 0:
            batch_count = min(batch_size, remaining)
            batch = generator.generate_synthetic_receipts(
                num_receipts=batch_count,
            )
            synthetic_receipts.extend(batch)
            pbar.update(len(batch))
            remaining -= batch_count

    _save_json(
        synthetic_receipts,
        output_dir / "synthetic_receipts.json",
        f"{len(synthetic_receipts)} synthetic receipts",
    )

    # ── Step 3: Error augmentations ───────────────────────────────────
    error_batch_size = 10
    num_error_batches = (args.num_error_pairs + error_batch_size - 1) // error_batch_size

    print(f"\n[3/3] Generating {args.num_error_pairs} error augmentation pairs ...")
    error_augmentations: list[dict] = []
    remaining = args.num_error_pairs

    with tqdm(total=args.num_error_pairs, desc="Error augmentations", unit="pair") as pbar:
        while remaining > 0:
            batch_count = min(error_batch_size, remaining)
            batch = generator.generate_error_augmentations(
                receipts=synthetic_receipts,
                num_pairs=batch_count,
            )
            error_augmentations.extend(batch)
            pbar.update(len(batch))
            remaining -= batch_count

    _save_json(
        error_augmentations,
        output_dir / "error_augmentations.json",
        f"{len(error_augmentations)} error augmentation pairs",
    )

    # ── Save combined output ──────────────────────────────────────────
    elapsed = time.time() - start_time
    usage = generator.get_usage_summary()

    combined = {
        "instruction_variants": instruction_variants,
        "synthetic_receipts": synthetic_receipts,
        "error_augmentations": error_augmentations,
        "cost_estimate": usage,
        "config": {
            "model": args.model,
            "num_instructions": args.num_instructions,
            "num_receipts": args.num_receipts,
            "num_error_pairs": args.num_error_pairs,
            "seed": args.seed,
            "elapsed_seconds": round(elapsed, 2),
        },
    }

    _save_json(
        combined,
        output_dir / "synthetic_all.json",
        "combined synthetic dataset",
    )

    # Print summary
    _print_summary(combined, elapsed)


if __name__ == "__main__":
    main()
