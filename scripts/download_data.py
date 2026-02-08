"""Download and cache datasets from HuggingFace."""

import argparse
import logging

from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def download_cord():
    """Download CORD v2 dataset."""
    logger.info("Downloading CORD v2 dataset...")
    ds = load_dataset("naver-clova-ix/cord-v2")
    logger.info(f"CORD v2 downloaded: {ds}")
    for split_name, split_ds in ds.items():
        logger.info(f"  {split_name}: {len(split_ds)} samples")


def download_funsd():
    """Download FUNSD dataset."""
    logger.info("Downloading FUNSD dataset...")
    ds = load_dataset("nielsr/funsd")
    logger.info(f"FUNSD downloaded: {ds}")
    for split_name, split_ds in ds.items():
        logger.info(f"  {split_name}: {len(split_ds)} samples")


def main():
    parser = argparse.ArgumentParser(description="Download datasets")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["cord", "funsd"],
        choices=["cord", "funsd"],
        help="Datasets to download",
    )
    args = parser.parse_args()

    for dataset_name in args.datasets:
        if dataset_name == "cord":
            download_cord()
        elif dataset_name == "funsd":
            download_funsd()

    logger.info("All datasets downloaded and cached.")


if __name__ == "__main__":
    main()
