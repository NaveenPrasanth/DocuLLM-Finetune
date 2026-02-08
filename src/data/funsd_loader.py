"""FUNSD dataset loader for cross-domain evaluation.

FUNSD (Form Understanding in Noisy Scanned Documents) contains 199 annotated
form images with semantic entity labels and linking information.
"""

from __future__ import annotations

import logging
from typing import Any

from datasets import load_dataset
from PIL import Image

from src.config import load_data_config

logger = logging.getLogger(__name__)


def extract_key_value_pairs(annotations: dict[str, Any]) -> dict[str, str]:
    """Extract key-value pairs from FUNSD annotations using entity linking.

    FUNSD annotations contain entities (words grouped by semantic role) and
    links between question-answer pairs.

    Args:
        annotations: FUNSD annotation dict with 'words', 'boxes', 'ner_tags', 'id'.

    Returns:
        Dict mapping question text to answer text.
    """
    # FUNSD entity types: 0=other, 1=question, 2=answer, 3=header
    entities: dict[int, dict[str, Any]] = {}

    words = annotations.get("words", [])
    ner_tags = annotations.get("ner_tags", [])

    # Group consecutive words by entity
    current_entity_id = 0
    current_text_parts: list[str] = []
    current_tag = -1

    for i, (word, tag) in enumerate(zip(words, ner_tags)):
        if tag != current_tag or i == 0:
            if current_text_parts and current_tag >= 0:
                entities[current_entity_id] = {
                    "text": " ".join(current_text_parts),
                    "type": current_tag,
                }
                current_entity_id += 1
            current_text_parts = [word]
            current_tag = tag
        else:
            current_text_parts.append(word)

    # Save the last entity
    if current_text_parts and current_tag >= 0:
        entities[current_entity_id] = {
            "text": " ".join(current_text_parts),
            "type": current_tag,
        }

    # Build key-value pairs from questions (type=1) followed by answers (type=2)
    kv_pairs = {}
    entity_list = sorted(entities.items())
    for i, (eid, entity) in enumerate(entity_list):
        if entity["type"] == 1:  # Question
            # Look for the next answer entity
            for j in range(i + 1, len(entity_list)):
                next_entity = entity_list[j][1]
                if next_entity["type"] == 2:  # Answer
                    kv_pairs[entity["text"]] = next_entity["text"]
                    break
                elif next_entity["type"] == 1:  # Another question
                    break

    return kv_pairs


def load_funsd_dataset(
    split: str = "train",
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    """Load FUNSD dataset from HuggingFace.

    Args:
        split: Dataset split - "train" or "test".
        max_samples: Optional maximum number of samples.

    Returns:
        List of dicts with keys: image, key_value_pairs, ground_truth_json, metadata.
    """
    config = load_data_config("funsd")

    logger.info(f"Loading FUNSD dataset, split={split}")
    ds = load_dataset(config.dataset.hf_path, split=split)

    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    samples = []
    for idx, item in enumerate(ds):
        image: Image.Image = item["image"]
        kv_pairs = extract_key_value_pairs(item)

        samples.append({
            "id": f"funsd_{split}_{idx}",
            "image": image.convert("RGB"),
            "key_value_pairs": kv_pairs,
            "ground_truth_json": str(kv_pairs),
            "ground_truth_flat": kv_pairs,
            "metadata": {
                "dataset": "funsd",
                "split": split,
                "index": idx,
                "num_fields": len(kv_pairs),
            },
        })

    logger.info(f"Loaded {len(samples)} FUNSD samples from {split} split")
    return samples


def get_funsd_schema() -> dict[str, str]:
    """Return expected FUNSD output schema for prompts."""
    return {
        "<field_name>": "<field_value>",
    }
