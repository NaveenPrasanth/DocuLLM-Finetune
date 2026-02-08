"""Dataset builder: combines real and synthetic data, creates train/val/test splits."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

from src.config import load_data_config
from src.data.cord_loader import load_cord_dataset
from src.data.format_converter import batch_convert_to_chatml, format_for_sft_trainer
from src.data.funsd_loader import load_funsd_dataset

logger = logging.getLogger(__name__)


def build_cord_splits(
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    """Build train/val/test splits from CORD v2 dataset.

    CORD has predefined train/validation/test splits. We use them directly
    but limit to configured sizes.

    Args:
        seed: Random seed for reproducibility.

    Returns:
        Dict with 'train', 'val', 'test' keys containing sample lists.
    """
    config = load_data_config("cord")
    random.seed(seed)

    train_samples = load_cord_dataset("train", max_samples=config.dataset.splits.train)
    val_samples = load_cord_dataset("validation", max_samples=config.dataset.splits.val)
    test_samples = load_cord_dataset("test", max_samples=config.dataset.splits.test)

    logger.info(
        f"CORD splits: train={len(train_samples)}, "
        f"val={len(val_samples)}, test={len(test_samples)}"
    )

    return {
        "train": train_samples,
        "val": val_samples,
        "test": test_samples,
    }


def build_funsd_splits(
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    """Build train/test splits from FUNSD dataset.

    Args:
        seed: Random seed for reproducibility.

    Returns:
        Dict with 'train' and 'test' keys.
    """
    config = load_data_config("funsd")
    random.seed(seed)

    train_samples = load_funsd_dataset("train", max_samples=config.dataset.splits.train)
    test_samples = load_funsd_dataset("test", max_samples=config.dataset.splits.test)

    logger.info(f"FUNSD splits: train={len(train_samples)}, test={len(test_samples)}")

    return {
        "train": train_samples,
        "test": test_samples,
    }


def add_synthetic_data(
    train_samples: list[dict[str, Any]],
    synthetic_path: str | Path,
    synthetic_ratio: float = 0.2,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Add synthetic data to training samples.

    Args:
        train_samples: Original training samples.
        synthetic_path: Path to synthetic data JSON file.
        synthetic_ratio: Fraction of synthetic data relative to real data.
        seed: Random seed.

    Returns:
        Combined list of real + synthetic samples.
    """
    synthetic_path = Path(synthetic_path)
    if not synthetic_path.exists():
        logger.warning(f"Synthetic data not found at {synthetic_path}, skipping")
        return train_samples

    with open(synthetic_path) as f:
        synthetic_data = json.load(f)

    random.seed(seed)
    n_synthetic = int(len(train_samples) * synthetic_ratio)
    if n_synthetic > len(synthetic_data):
        n_synthetic = len(synthetic_data)

    selected = random.sample(synthetic_data, n_synthetic)

    for i, item in enumerate(selected):
        item["id"] = f"synthetic_{i}"
        item["metadata"] = {
            "dataset": "synthetic",
            "index": i,
            "source": "claude_generated",
        }

    combined = train_samples + selected
    random.shuffle(combined)

    logger.info(
        f"Added {len(selected)} synthetic samples to {len(train_samples)} real samples "
        f"(total: {len(combined)})"
    )
    return combined


def build_training_dataset(
    dataset_name: str = "cord",
    instruction: str | None = None,
    include_synthetic: bool = False,
    synthetic_path: str | Path | None = None,
    synthetic_ratio: float = 0.2,
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    """Build complete training dataset with ChatML formatting.

    Args:
        dataset_name: Which dataset to use ('cord' or 'funsd').
        instruction: Custom instruction text (uses default if None).
        include_synthetic: Whether to include synthetic data.
        synthetic_path: Path to synthetic data file.
        synthetic_ratio: Ratio of synthetic to real data.
        seed: Random seed.

    Returns:
        Dict with split keys, each containing lists of formatted training examples.
    """
    if dataset_name == "cord":
        splits = build_cord_splits(seed=seed)
    elif dataset_name == "funsd":
        splits = build_funsd_splits(seed=seed)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if include_synthetic and synthetic_path:
        splits["train"] = add_synthetic_data(
            splits["train"],
            synthetic_path=synthetic_path,
            synthetic_ratio=synthetic_ratio,
            seed=seed,
        )

    # Convert to ChatML format
    formatted = {}
    for split_name, samples in splits.items():
        kwargs = {}
        if instruction:
            kwargs["instruction"] = instruction
        formatted[split_name] = batch_convert_to_chatml(samples, **kwargs)

    return formatted


def build_sft_dataset(
    dataset_name: str = "cord",
    instruction: str | None = None,
    include_synthetic: bool = False,
    synthetic_path: str | Path | None = None,
    synthetic_ratio: float = 0.2,
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    """Build dataset formatted for SFTTrainer.

    Args:
        dataset_name: Which dataset to use.
        instruction: Custom instruction text.
        include_synthetic: Whether to include synthetic data.
        synthetic_path: Path to synthetic data file.
        synthetic_ratio: Ratio of synthetic to real data.
        seed: Random seed.

    Returns:
        Dict with split keys, each containing lists of SFTTrainer-compatible dicts.
    """
    if dataset_name == "cord":
        splits = build_cord_splits(seed=seed)
    elif dataset_name == "funsd":
        splits = build_funsd_splits(seed=seed)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if include_synthetic and synthetic_path:
        splits["train"] = add_synthetic_data(
            splits["train"],
            synthetic_path=synthetic_path,
            synthetic_ratio=synthetic_ratio,
            seed=seed,
        )

    formatted = {}
    kwargs = {}
    if instruction:
        kwargs["instruction"] = instruction
    for split_name, samples in splits.items():
        formatted[split_name] = format_for_sft_trainer(samples, **kwargs)

    return formatted


def get_dataset_stats(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute statistics for a list of samples.

    Args:
        samples: List of sample dicts from loaders.

    Returns:
        Dictionary with dataset statistics.
    """
    num_fields = [s["metadata"]["num_fields"] for s in samples if "metadata" in s]

    return {
        "num_samples": len(samples),
        "avg_fields": sum(num_fields) / len(num_fields) if num_fields else 0,
        "min_fields": min(num_fields) if num_fields else 0,
        "max_fields": max(num_fields) if num_fields else 0,
        "total_fields": sum(num_fields),
    }
