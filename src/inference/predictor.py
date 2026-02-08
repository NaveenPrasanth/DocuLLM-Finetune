"""Unified inference predictor for Qwen2-VL document extraction.

Supports base, prompted, and fine-tuned model inference modes with
consistent generation configuration and proper device handling.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm

from src.data.format_converter import sample_to_inference_chatml

logger = logging.getLogger(__name__)

# Supported inference modes
MODES = ("base", "prompted", "finetuned")


class Predictor:
    """Unified predictor for Qwen2-VL document extraction.

    Wraps model and processor to provide consistent single-image and batch
    inference across base, prompted, and fine-tuned model variants.

    Args:
        model: Loaded Qwen2-VL model (potentially with LoRA adapter merged).
        processor: Loaded AutoProcessor for Qwen2-VL.
        mode: Inference mode - "base" (default instruction), "prompted"
            (custom instruction), or "finetuned" (fine-tuned model).
    """

    # Default generation configuration for consistent, deterministic output
    DEFAULT_GEN_CONFIG = {
        "max_new_tokens": 1024,
        "temperature": 0.1,
        "do_sample": False,
        "top_p": 1.0,
        "repetition_penalty": 1.05,
    }

    def __init__(
        self,
        model: Any,
        processor: Any,
        mode: str = "base",
    ):
        if mode not in MODES:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of: {MODES}")

        self.model = model
        self.processor = processor
        self.mode = mode
        self.device = _get_model_device(model)

        # Ensure model is in eval mode
        self.model.eval()

        logger.info(
            "Predictor initialized: mode=%s, device=%s",
            self.mode, self.device,
        )

    def predict_single(
        self,
        image: Image.Image,
        instruction: str | None = None,
    ) -> str:
        """Run inference on a single image.

        Args:
            image: PIL Image of the document.
            instruction: Custom instruction text. Uses the default from
                format_converter if None.

        Returns:
            Raw text output from the model (before any JSON parsing).
        """
        # Build ChatML messages for inference (user message only, no assistant)
        kwargs: dict[str, Any] = {}
        if instruction is not None:
            kwargs["instruction"] = instruction

        messages = sample_to_inference_chatml(image, **kwargs)

        # Apply chat template to get the formatted text prompt
        text_prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Process inputs (text + image) through the processor
        inputs = self.processor(
            text=[text_prompt],
            images=[image],
            padding=True,
            return_tensors="pt",
        )

        # Move inputs to model device
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        # Generate output
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                **self.DEFAULT_GEN_CONFIG,
            )

        # Decode only the generated tokens (strip input prefix)
        input_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[:, input_len:]
        output_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]

        return output_text.strip()

    def predict_batch(
        self,
        samples: list[dict[str, Any]],
        instruction: str | None = None,
        batch_size: int = 4,
    ) -> list[str]:
        """Run batch inference over multiple samples.

        Processes samples in batches for GPU efficiency. Each sample dict
        should contain an "image" key with a PIL Image.

        Args:
            samples: List of sample dicts, each with at minimum an "image" key.
            instruction: Custom instruction text applied to all samples.
            batch_size: Number of samples to process simultaneously.

        Returns:
            List of raw text outputs, one per sample.
        """
        results: list[str] = []

        for batch_start in tqdm(
            range(0, len(samples), batch_size),
            desc=f"Inference ({self.mode})",
            unit="batch",
        ):
            batch_samples = samples[batch_start : batch_start + batch_size]
            batch_outputs = self._process_batch(batch_samples, instruction)
            results.extend(batch_outputs)

        logger.info(
            "Batch inference complete: %d samples, mode=%s",
            len(results), self.mode,
        )
        return results

    def _process_batch(
        self,
        batch_samples: list[dict[str, Any]],
        instruction: str | None = None,
    ) -> list[str]:
        """Process a single batch of samples.

        Args:
            batch_samples: List of sample dicts for this batch.
            instruction: Optional custom instruction.

        Returns:
            List of output text strings for this batch.
        """
        kwargs: dict[str, Any] = {}
        if instruction is not None:
            kwargs["instruction"] = instruction

        text_prompts = []
        images = []

        for sample in batch_samples:
            image = sample["image"]
            if not isinstance(image, Image.Image):
                raise TypeError(
                    f"Expected PIL Image, got {type(image)}. "
                    "Ensure sample['image'] is a PIL Image."
                )

            messages = sample_to_inference_chatml(image, **kwargs)
            text_prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            text_prompts.append(text_prompt)
            images.append(image)

        # Process batch through the processor
        inputs = self.processor(
            text=text_prompts,
            images=images,
            padding=True,
            return_tensors="pt",
        )

        # Move to device
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                **self.DEFAULT_GEN_CONFIG,
            )

        # Decode generated tokens
        input_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[:, input_len:]
        outputs = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        return [o.strip() for o in outputs]


# ── Helpers ─────────────────────────────────────────────────────────────────


def _get_model_device(model: Any) -> torch.device:
    """Determine the device a model is on.

    Handles models with device_map="auto" that may span multiple devices.

    Args:
        model: PyTorch model.

    Returns:
        The device of the first parameter, or CPU as fallback.
    """
    try:
        # For models with device_map, get the device of the first parameter
        first_param = next(model.parameters())
        return first_param.device
    except StopIteration:
        return torch.device("cpu")
