"""Run full data pipeline: load, parse, convert, split, and save."""

import argparse
import json
import logging
from pathlib import Path

from src.config import load_base_config
from src.data.dataset_builder import (
    build_cord_splits,
    build_funsd_splits,
    build_training_dataset,
    get_dataset_stats,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def save_split_metadata(
    splits: dict,
    output_dir: Path,
    dataset_name: str,
):
    """Save split metadata (without images) for inspection."""
    meta_dir = output_dir / dataset_name / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    for split_name, samples in splits.items():
        # Save metadata without images
        metadata = []
        for s in samples:
            entry = {
                "id": s.get("id", ""),
                "num_fields": s.get("metadata", {}).get("num_fields", 0),
                "ground_truth_flat": s.get("ground_truth_flat", {}),
            }
            metadata.append(entry)

        meta_path = meta_dir / f"{split_name}_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {split_name} metadata to {meta_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare datasets")
    parser.add_argument(
        "--dataset",
        default="cord",
        choices=["cord", "funsd", "both"],
        help="Dataset to prepare",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    base_config = load_base_config()
    output_dir = Path(args.output_dir or base_config.project.data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets_to_process = []
    if args.dataset in ("cord", "both"):
        datasets_to_process.append("cord")
    if args.dataset in ("funsd", "both"):
        datasets_to_process.append("funsd")

    for ds_name in datasets_to_process:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {ds_name.upper()} dataset")
        logger.info(f"{'='*60}")

        if ds_name == "cord":
            splits = build_cord_splits(seed=args.seed)
        else:
            splits = build_funsd_splits(seed=args.seed)

        # Print stats
        for split_name, samples in splits.items():
            stats = get_dataset_stats(samples)
            logger.info(f"\n{split_name} split statistics:")
            for k, v in stats.items():
                logger.info(f"  {k}: {v}")

        # Save metadata
        save_split_metadata(splits, output_dir, ds_name)

        # Build ChatML formatted dataset
        formatted = build_training_dataset(
            dataset_name=ds_name,
            seed=args.seed,
        )

        for split_name, examples in formatted.items():
            logger.info(f"\nChatML {split_name}: {len(examples)} examples")
            if examples:
                sample = examples[0]
                logger.info(f"  Sample keys: {list(sample.keys())}")
                logger.info(f"  Messages count: {len(sample['messages'])}")

    logger.info("\nDataset preparation complete!")


if __name__ == "__main__":
    main()
