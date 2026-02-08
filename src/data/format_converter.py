"""Convert dataset annotations to Qwen2-VL ChatML conversation format.

Qwen2-VL expects conversations in the format:
[
    {
        "role": "user",
        "content": [
            {"type": "image", "image": <PIL Image or path>},
            {"type": "text", "text": "<instruction>"}
        ]
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "<structured JSON output>"}
        ]
    }
]
"""

from __future__ import annotations

import json
import logging
from typing import Any

from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_INSTRUCTION = (
    "Extract all information from this receipt image as structured JSON. "
    "Include menu items, totals, subtotals, and any other visible information. "
    "Return ONLY valid JSON."
)


def sample_to_chatml(
    image: Image.Image,
    ground_truth_json: str,
    instruction: str = DEFAULT_INSTRUCTION,
    image_path: str | None = None,
) -> list[dict[str, Any]]:
    """Convert a single sample to Qwen2-VL ChatML format.

    Args:
        image: PIL Image of the document.
        ground_truth_json: JSON string of the expected output.
        instruction: Text instruction for the model.
        image_path: Optional path to image file (used instead of PIL if provided).

    Returns:
        List of message dicts in ChatML format.
    """
    image_content: dict[str, Any] = {"type": "image"}
    if image_path:
        image_content["image"] = image_path
    else:
        image_content["image"] = image

    return [
        {
            "role": "user",
            "content": [
                image_content,
                {"type": "text", "text": instruction},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": ground_truth_json},
            ],
        },
    ]


def sample_to_inference_chatml(
    image: Image.Image,
    instruction: str = DEFAULT_INSTRUCTION,
    image_path: str | None = None,
) -> list[dict[str, Any]]:
    """Convert a sample to ChatML for inference (no assistant response).

    Args:
        image: PIL Image of the document.
        instruction: Text instruction for the model.
        image_path: Optional path to image file.

    Returns:
        List with a single user message in ChatML format.
    """
    image_content: dict[str, Any] = {"type": "image"}
    if image_path:
        image_content["image"] = image_path
    else:
        image_content["image"] = image

    return [
        {
            "role": "user",
            "content": [
                image_content,
                {"type": "text", "text": instruction},
            ],
        },
    ]


def batch_convert_to_chatml(
    samples: list[dict[str, Any]],
    instruction: str = DEFAULT_INSTRUCTION,
) -> list[dict[str, Any]]:
    """Convert a batch of dataset samples to ChatML training format.

    Args:
        samples: List of sample dicts (from cord_loader or funsd_loader).
        instruction: Text instruction for the model.

    Returns:
        List of training examples, each with 'messages' and 'metadata' keys.
    """
    converted = []
    for sample in samples:
        gt_json = sample.get("ground_truth_json", "{}")

        # Ensure the JSON is pretty-formatted for training
        try:
            parsed = json.loads(gt_json)
            formatted_json = json.dumps(parsed, indent=2, ensure_ascii=False)
        except (json.JSONDecodeError, TypeError):
            formatted_json = gt_json

        messages = sample_to_chatml(
            image=sample["image"],
            ground_truth_json=formatted_json,
            instruction=instruction,
        )

        converted.append({
            "id": sample.get("id", ""),
            "messages": messages,
            "metadata": sample.get("metadata", {}),
        })

    logger.info(f"Converted {len(converted)} samples to ChatML format")
    return converted


def format_for_sft_trainer(
    samples: list[dict[str, Any]],
    instruction: str = DEFAULT_INSTRUCTION,
) -> list[dict[str, Any]]:
    """Format samples for TRL SFTTrainer.

    SFTTrainer expects a dataset with a 'messages' column containing
    the ChatML conversation.

    Args:
        samples: List of sample dicts from loaders.
        instruction: Text instruction.

    Returns:
        List of dicts with 'messages' key suitable for SFTTrainer.
    """
    training_data = []
    for sample in samples:
        gt_json = sample.get("ground_truth_json", "{}")
        try:
            parsed = json.loads(gt_json)
            formatted_json = json.dumps(parsed, indent=2, ensure_ascii=False)
        except (json.JSONDecodeError, TypeError):
            formatted_json = gt_json

        messages = sample_to_chatml(
            image=sample["image"],
            ground_truth_json=formatted_json,
            instruction=instruction,
        )

        training_data.append({"messages": messages})

    return training_data
