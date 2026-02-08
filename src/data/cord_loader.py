"""CORD v2 dataset loader and parser.

CORD (Consolidated Receipt Dataset) v2 contains 1000 receipt images with
structured JSON annotations covering 30 field types across 5 superclasses.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from datasets import load_dataset
from PIL import Image

from src.config import load_data_config

logger = logging.getLogger(__name__)


# CORD ground_truth JSON has this structure:
# {
#   "gt_parse": {
#     "menu": [{"nm": "...", "price": "...", ...}],
#     "sub_total": {"subtotal_price": "...", ...},
#     "total": {"total_price": "...", ...},
#     "void_menu": [...],
#     "etc": "..."
#   }
# }


def parse_cord_ground_truth(gt_json_str: str) -> dict[str, Any]:
    """Parse CORD ground_truth JSON string into a flat field dictionary.

    Args:
        gt_json_str: Raw JSON string from the CORD dataset ground_truth column.

    Returns:
        Parsed dictionary with the gt_parse content.
    """
    try:
        gt = json.loads(gt_json_str)
        return gt.get("gt_parse", {})
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"Failed to parse ground truth JSON: {e}")
        return {}


def flatten_cord_fields(gt_parse: dict[str, Any]) -> dict[str, Any]:
    """Flatten CORD gt_parse into a flat key-value dict for evaluation.

    Handles nested structures like menu items (lists of dicts).

    Args:
        gt_parse: Parsed gt_parse dictionary from CORD.

    Returns:
        Flat dictionary with dotted keys, e.g., "menu.0.nm": "Coffee".
    """
    flat = {}

    for superclass, value in gt_parse.items():
        if isinstance(value, list):
            # Menu items, void_menu items
            for idx, item in enumerate(value):
                if isinstance(item, dict):
                    for field_name, field_value in item.items():
                        if field_value:  # Skip empty values
                            flat[f"{superclass}.{idx}.{field_name}"] = str(field_value)
                elif item:
                    flat[f"{superclass}.{idx}"] = str(item)
        elif isinstance(value, dict):
            # sub_total, total
            for field_name, field_value in value.items():
                if field_value:
                    flat[f"{superclass}.{field_name}"] = str(field_value)
        elif value:
            flat[superclass] = str(value)

    return flat


def load_cord_dataset(
    split: str = "train",
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    """Load CORD v2 dataset from HuggingFace.

    Args:
        split: Dataset split - "train", "validation", or "test".
        max_samples: Optional maximum number of samples to load.

    Returns:
        List of dicts with keys: image, ground_truth, ground_truth_flat, metadata.
    """
    config = load_data_config("cord")

    logger.info(f"Loading CORD v2 dataset, split={split}")
    ds = load_dataset(config.dataset.hf_path, split=split)

    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    samples = []
    for idx, item in enumerate(ds):
        image: Image.Image = item["image"]
        gt_str = item["ground_truth"]

        gt_parse = parse_cord_ground_truth(gt_str)
        gt_flat = flatten_cord_fields(gt_parse)

        samples.append({
            "id": f"cord_{split}_{idx}",
            "image": image.convert("RGB"),
            "ground_truth": gt_parse,
            "ground_truth_json": json.dumps(gt_parse, ensure_ascii=False),
            "ground_truth_flat": gt_flat,
            "metadata": {
                "dataset": "cord",
                "split": split,
                "index": idx,
                "num_fields": len(gt_flat),
            },
        })

    logger.info(f"Loaded {len(samples)} CORD samples from {split} split")
    return samples


def get_cord_schema() -> dict[str, Any]:
    """Return the expected CORD JSON schema for prompt templates.

    Returns:
        Schema dictionary showing the expected output structure.
    """
    return {
        "menu": [
            {
                "nm": "item name",
                "num": "item number",
                "unitprice": "unit price",
                "cnt": "quantity",
                "discountprice": "discount",
                "price": "total price",
                "itemsubtotal": "item subtotal",
                "vatyn": "VAT included (Y/N)",
                "etc": "other info",
                "sub_nm": "sub-item name",
                "sub_unitprice": "sub-item unit price",
                "sub_cnt": "sub-item quantity",
                "sub_price": "sub-item price",
                "sub_etc": "sub-item other info",
            }
        ],
        "total": {
            "total_price": "total amount",
            "total_etc": "total other info",
            "cashprice": "cash amount",
            "changeprice": "change amount",
            "creditcardprice": "credit card amount",
            "emoneyprice": "e-money amount",
            "menutype_cnt": "menu type count",
            "menuqty_cnt": "menu quantity count",
        },
        "sub_total": {
            "subtotal_price": "subtotal",
            "discount_price": "discount",
            "service_price": "service charge",
            "othersvc_price": "other service",
            "tax_price": "tax",
            "etc": "other",
        },
        "void_menu": [
            {
                "nm": "voided item name",
                "price": "voided item price",
            }
        ],
        "etc": "other information",
    }
