"""Custom training callbacks for QLoRA fine-tuning.

Provides GPU memory monitoring, field-level F1 evaluation, and Weights & Biases
integration for sample prediction logging.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

logger = logging.getLogger(__name__)


class GPUMemoryCallback(TrainerCallback):
    """Log GPU memory usage at each logging step.

    Reports allocated memory, reserved memory, and peak memory via
    ``torch.cuda`` APIs.  Metrics are logged both to the trainer's log
    history and (if available) to Weights & Biases.
    """

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if not torch.cuda.is_available():
            return

        allocated_gb = torch.cuda.memory_allocated() / (1024**3)
        reserved_gb = torch.cuda.memory_reserved() / (1024**3)
        max_allocated_gb = torch.cuda.max_memory_allocated() / (1024**3)

        memory_info = {
            "gpu/memory_allocated_gb": round(allocated_gb, 3),
            "gpu/memory_reserved_gb": round(reserved_gb, 3),
            "gpu/peak_memory_allocated_gb": round(max_allocated_gb, 3),
        }

        logger.info(
            "GPU Memory: allocated=%.3f GB, reserved=%.3f GB, peak=%.3f GB",
            allocated_gb,
            reserved_gb,
            max_allocated_gb,
        )

        # Merge into trainer logs so they appear in log_history and wandb
        if logs is not None:
            logs.update(memory_info)


class FieldF1Callback(TrainerCallback):
    """Compute field-level F1 on a held-out sample set at evaluation steps.

    At each evaluation step, runs inference on a small set of held-out
    examples and computes precision, recall, and F1 by comparing predicted
    JSON fields against ground truth.

    Args:
        eval_samples: List of dicts, each with ``"messages"`` (ChatML) and
            ``"ground_truth_flat"`` (dict of field key-value pairs).
        processor: The Qwen2-VL processor for encoding inputs.
        max_samples: Maximum number of samples to evaluate per step.
        max_new_tokens: Maximum tokens to generate per sample.
    """

    def __init__(
        self,
        eval_samples: list[dict[str, Any]],
        processor: Any,
        max_samples: int = 10,
        max_new_tokens: int = 1024,
    ) -> None:
        super().__init__()
        self.eval_samples = eval_samples[:max_samples]
        self.processor = processor
        self.max_new_tokens = max_new_tokens

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: Any = None,
        **kwargs: Any,
    ) -> None:
        if model is None or not self.eval_samples:
            return

        logger.info("Running field-level F1 evaluation on %d samples...", len(self.eval_samples))

        total_tp = 0
        total_fp = 0
        total_fn = 0

        model.eval()
        with torch.no_grad():
            for sample in self.eval_samples:
                prediction = self._run_inference(model, sample)
                gt_flat = sample.get("ground_truth_flat", {})
                pred_flat = self._parse_prediction(prediction)

                tp, fp, fn = self._compute_field_counts(gt_flat, pred_flat)
                total_tp += tp
                total_fp += fp
                total_fn += fn

        model.train()

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        metrics = {
            "eval/field_precision": round(precision, 4),
            "eval/field_recall": round(recall, 4),
            "eval/field_f1": round(f1, 4),
        }

        logger.info(
            "Field F1: precision=%.4f, recall=%.4f, f1=%.4f",
            precision,
            recall,
            f1,
        )

        # Log to wandb if available
        try:
            import wandb

            if wandb.run is not None:
                wandb.log(metrics, step=state.global_step)
        except ImportError:
            pass

    def _run_inference(self, model: Any, sample: dict[str, Any]) -> str:
        """Run single-sample inference and return decoded text."""
        messages = sample["messages"]
        # Use only the user message for inference
        user_messages = [m for m in messages if m["role"] == "user"]

        try:
            text = self.processor.apply_chat_template(
                user_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            # Process inputs
            inputs = self.processor(
                text=[text],
                padding=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            output_ids = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

            # Decode only the generated tokens
            generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
            decoded = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            return decoded[0] if decoded else ""

        except Exception as e:
            logger.warning("Inference failed for sample: %s", e)
            return ""

    @staticmethod
    def _parse_prediction(text: str) -> dict[str, str]:
        """Attempt to parse a JSON prediction into flat key-value pairs."""
        text = text.strip()

        # Try to extract JSON from the text
        for start_char, end_char in [("{", "}"), ("[", "]")]:
            start = text.find(start_char)
            end = text.rfind(end_char)
            if start != -1 and end != -1 and end > start:
                try:
                    parsed = json.loads(text[start : end + 1])
                    if isinstance(parsed, dict):
                        return _flatten_dict(parsed)
                except json.JSONDecodeError:
                    continue

        return {}

    @staticmethod
    def _compute_field_counts(
        gt: dict[str, str],
        pred: dict[str, str],
    ) -> tuple[int, int, int]:
        """Compute true positives, false positives, false negatives.

        A field is a true positive if both the key exists in the prediction
        and the value matches (case-insensitive, stripped).
        """
        tp = 0
        fp = 0
        fn = 0

        gt_normalized = {k.strip().lower(): str(v).strip().lower() for k, v in gt.items()}
        pred_normalized = {k.strip().lower(): str(v).strip().lower() for k, v in pred.items()}

        for key, value in gt_normalized.items():
            if key in pred_normalized and pred_normalized[key] == value:
                tp += 1
            else:
                fn += 1

        for key in pred_normalized:
            if key not in gt_normalized:
                fp += 1

        return tp, fp, fn


class WandbTableCallback(TrainerCallback):
    """Log sample predictions as Weights & Biases tables at evaluation steps.

    Creates a W&B Table with columns for sample ID, input instruction,
    ground truth, and model prediction. This provides a qualitative view
    of model performance during training.

    Args:
        eval_samples: List of dicts with ``"messages"`` and optionally ``"id"``.
        processor: The Qwen2-VL processor.
        max_samples: Maximum samples to log per evaluation.
        max_new_tokens: Maximum tokens to generate per sample.
        table_name: Name of the W&B table.
    """

    def __init__(
        self,
        eval_samples: list[dict[str, Any]],
        processor: Any,
        max_samples: int = 5,
        max_new_tokens: int = 1024,
        table_name: str = "predictions",
    ) -> None:
        super().__init__()
        self.eval_samples = eval_samples[:max_samples]
        self.processor = processor
        self.max_new_tokens = max_new_tokens
        self.table_name = table_name

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: Any = None,
        **kwargs: Any,
    ) -> None:
        try:
            import wandb
        except ImportError:
            logger.debug("wandb not available, skipping WandbTableCallback")
            return

        if wandb.run is None or model is None or not self.eval_samples:
            return

        logger.info("Logging %d sample predictions to W&B table...", len(self.eval_samples))

        columns = ["step", "sample_id", "instruction", "ground_truth", "prediction"]
        table = wandb.Table(columns=columns)

        model.eval()
        with torch.no_grad():
            for i, sample in enumerate(self.eval_samples):
                messages = sample.get("messages", [])
                sample_id = sample.get("id", f"sample_{i}")

                # Extract instruction text from user message
                instruction = ""
                for msg in messages:
                    if msg["role"] == "user":
                        for content in msg.get("content", []):
                            if isinstance(content, dict) and content.get("type") == "text":
                                instruction = content.get("text", "")
                                break
                        break

                # Extract ground truth from assistant message
                ground_truth = ""
                for msg in messages:
                    if msg["role"] == "assistant":
                        for content in msg.get("content", []):
                            if isinstance(content, dict) and content.get("type") == "text":
                                ground_truth = content.get("text", "")
                                break
                        break

                # Run inference
                prediction = self._run_inference(model, sample)

                table.add_data(
                    state.global_step,
                    sample_id,
                    instruction[:200],  # Truncate for readability
                    ground_truth[:500],
                    prediction[:500],
                )

        model.train()

        wandb.log({self.table_name: table}, step=state.global_step)
        logger.info("Logged prediction table at step %d", state.global_step)

    def _run_inference(self, model: Any, sample: dict[str, Any]) -> str:
        """Run single-sample inference and return decoded text."""
        messages = sample["messages"]
        user_messages = [m for m in messages if m["role"] == "user"]

        try:
            text = self.processor.apply_chat_template(
                user_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = self.processor(
                text=[text],
                padding=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            output_ids = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

            generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
            decoded = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            return decoded[0] if decoded else ""

        except Exception as e:
            logger.warning("Inference failed in WandbTableCallback: %s", e)
            return f"[ERROR: {e}]"


# ── Helpers ──────────────────────────────────────────────────────────────────


def _flatten_dict(
    d: dict[str, Any],
    parent_key: str = "",
    sep: str = ".",
) -> dict[str, str]:
    """Recursively flatten a nested dict into dot-separated keys.

    Args:
        d: Dictionary to flatten.
        parent_key: Prefix for keys at this level.
        sep: Separator between nested key components.

    Returns:
        Flat dict mapping dotted keys to string values.
    """
    items: list[tuple[str, str]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            for idx, item in enumerate(v):
                list_key = f"{new_key}{sep}{idx}"
                if isinstance(item, dict):
                    items.extend(_flatten_dict(item, list_key, sep=sep).items())
                elif item is not None:
                    items.append((list_key, str(item)))
        elif v is not None:
            items.append((new_key, str(v)))
    return dict(items)
