"""Run prompt engineering experiments across multiple strategies.

Evaluates each prompt strategy on the same set of test samples using the
Qwen2-VL base model, collecting raw predictions and computing per-strategy
metrics (JSON validity, field-level accuracy, exact match rate).
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from src.data.cord_loader import flatten_cord_fields
from src.prompts.strategies import BaseStrategy

logger = logging.getLogger(__name__)


# ── JSON Extraction Helpers ──────────────────────────────────────────────────


def extract_json_from_response(response: str) -> dict[str, Any] | None:
    """Extract a JSON object from model response text.

    Handles common model output patterns:
    - Pure JSON output
    - JSON wrapped in markdown code blocks (```json ... ```)
    - JSON after a "RESULT:" marker (chain-of-thought strategies)

    Args:
        response: Raw model output string.

    Returns:
        Parsed dict if valid JSON is found, None otherwise.
    """
    if not response or not response.strip():
        return None

    text = response.strip()

    # Try parsing the entire response as JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block
    code_block_match = re.search(
        r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL
    )
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try extracting after "RESULT:" marker (CoT strategies)
    result_match = re.search(r"RESULT:\s*(.+)", text, re.DOTALL)
    if result_match:
        remainder = result_match.group(1).strip()
        # Try the remainder directly
        try:
            return json.loads(remainder)
        except json.JSONDecodeError:
            pass
        # Try extracting a code block from the remainder
        code_match = re.search(
            r"```(?:json)?\s*\n?(.*?)\n?\s*```", remainder, re.DOTALL
        )
        if code_match:
            try:
                return json.loads(code_match.group(1).strip())
            except json.JSONDecodeError:
                pass

    # Last resort: find the first { ... } block
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


# ── Metrics ──────────────────────────────────────────────────────────────────


def compute_field_accuracy(
    predicted_flat: dict[str, str],
    ground_truth_flat: dict[str, str],
) -> dict[str, Any]:
    """Compute field-level accuracy between predicted and ground truth.

    Args:
        predicted_flat: Flat dict of predicted field values.
        ground_truth_flat: Flat dict of ground truth field values.

    Returns:
        Dict with precision, recall, f1, exact_matches, total_gt_fields,
        total_pred_fields.
    """
    gt_keys = set(ground_truth_flat.keys())
    pred_keys = set(predicted_flat.keys())

    # Exact key-value matches
    exact_matches = 0
    for key in gt_keys & pred_keys:
        gt_val = str(ground_truth_flat[key]).strip().lower()
        pred_val = str(predicted_flat[key]).strip().lower()
        if gt_val == pred_val:
            exact_matches += 1

    precision = exact_matches / len(pred_keys) if pred_keys else 0.0
    recall = exact_matches / len(gt_keys) if gt_keys else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "exact_matches": exact_matches,
        "total_gt_fields": len(gt_keys),
        "total_pred_fields": len(pred_keys),
    }


def compute_sample_metrics(
    prediction: dict[str, Any] | None,
    ground_truth: dict[str, Any],
    ground_truth_flat: dict[str, str],
) -> dict[str, Any]:
    """Compute all metrics for a single prediction.

    Args:
        prediction: Parsed prediction dict (may be None if parsing failed).
        ground_truth: Structured ground truth dict.
        ground_truth_flat: Flat ground truth dict.

    Returns:
        Dict of metric values.
    """
    metrics: dict[str, Any] = {
        "json_valid": prediction is not None,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "exact_matches": 0,
        "total_gt_fields": len(ground_truth_flat),
        "total_pred_fields": 0,
    }

    if prediction is None:
        return metrics

    # Flatten the prediction using the same logic as CORD ground truth
    try:
        predicted_flat = flatten_cord_fields(prediction)
    except Exception:
        predicted_flat = {}

    field_metrics = compute_field_accuracy(predicted_flat, ground_truth_flat)
    metrics.update(field_metrics)
    return metrics


def aggregate_metrics(
    per_sample_metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate per-sample metrics into strategy-level summary.

    Args:
        per_sample_metrics: List of per-sample metric dicts.

    Returns:
        Aggregated metrics dict.
    """
    n = len(per_sample_metrics)
    if n == 0:
        return {
            "num_samples": 0,
            "json_valid_rate": 0.0,
            "avg_precision": 0.0,
            "avg_recall": 0.0,
            "avg_f1": 0.0,
            "total_exact_matches": 0,
            "total_gt_fields": 0,
            "total_pred_fields": 0,
        }

    json_valid_count = sum(1 for m in per_sample_metrics if m["json_valid"])
    avg_precision = sum(m["precision"] for m in per_sample_metrics) / n
    avg_recall = sum(m["recall"] for m in per_sample_metrics) / n
    avg_f1 = sum(m["f1"] for m in per_sample_metrics) / n
    total_exact = sum(m["exact_matches"] for m in per_sample_metrics)
    total_gt = sum(m["total_gt_fields"] for m in per_sample_metrics)
    total_pred = sum(m["total_pred_fields"] for m in per_sample_metrics)

    # Micro-averaged F1 across all fields
    micro_precision = total_exact / total_pred if total_pred > 0 else 0.0
    micro_recall = total_exact / total_gt if total_gt > 0 else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0
        else 0.0
    )

    return {
        "num_samples": n,
        "json_valid_rate": round(json_valid_count / n, 4),
        "avg_precision": round(avg_precision, 4),
        "avg_recall": round(avg_recall, 4),
        "avg_f1": round(avg_f1, 4),
        "micro_precision": round(micro_precision, 4),
        "micro_recall": round(micro_recall, 4),
        "micro_f1": round(micro_f1, 4),
        "total_exact_matches": total_exact,
        "total_gt_fields": total_gt,
        "total_pred_fields": total_pred,
    }


# ── Experiment Runner ────────────────────────────────────────────────────────


class PromptExperimentRunner:
    """Runs prompt strategy experiments on a Qwen2-VL model.

    Iterates through a list of prompt strategies, running each one on the
    same set of test samples, and collects predictions and metrics.

    Args:
        model: Loaded Qwen2-VL model (transformers AutoModelForCausalLM).
        processor: Loaded Qwen2-VL processor (AutoProcessor).
        strategies: List of prompt strategies to evaluate.
        device: Torch device string. Auto-detected if None.
        max_new_tokens: Maximum tokens to generate per sample.
    """

    def __init__(
        self,
        model: Any,
        processor: Any,
        strategies: list[BaseStrategy],
        device: str | None = None,
        max_new_tokens: int = 1024,
    ):
        self.model = model
        self.processor = processor
        self.strategies = strategies
        self.max_new_tokens = max_new_tokens

        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        logger.info(
            f"PromptExperimentRunner initialized with {len(strategies)} strategies "
            f"on device={self.device}"
        )

    def _generate(self, messages: list[dict[str, Any]]) -> str:
        """Run inference on a single ChatML message list.

        Args:
            messages: ChatML-format messages for Qwen2-VL.

        Returns:
            Decoded model output string.
        """
        # Apply the chat template to build the prompt text
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process image inputs from the messages
        image_inputs = []
        for msg in messages:
            if isinstance(msg.get("content"), list):
                for content_item in msg["content"]:
                    if content_item.get("type") == "image":
                        img = content_item.get("image")
                        if isinstance(img, Image.Image):
                            image_inputs.append(img)

        # Build processor inputs
        if image_inputs:
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                padding=True,
                return_tensors="pt",
            )
        else:
            inputs = self.processor(
                text=[text],
                padding=True,
                return_tensors="pt",
            )

        inputs = inputs.to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        # Trim input tokens from output
        input_len = inputs["input_ids"].shape[1]
        generated_ids = generated_ids[:, input_len:]

        output = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        return output.strip()

    def run_strategy(
        self,
        strategy: BaseStrategy,
        test_samples: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Run a single strategy on all test samples.

        Args:
            strategy: Prompt strategy to evaluate.
            test_samples: List of test samples.

        Returns:
            Dict with strategy name, aggregated metrics, and per-sample results.
        """
        strategy_name = strategy.get_name()
        logger.info(f"Running strategy: {strategy_name} on {len(test_samples)} samples")

        per_sample_results: list[dict[str, Any]] = []
        per_sample_metrics: list[dict[str, Any]] = []
        start_time = time.time()

        for i, sample in enumerate(test_samples):
            sample_id = sample.get("id", f"sample_{i}")

            try:
                # Build messages and generate
                messages = strategy.build_messages(sample)
                raw_output = self._generate(messages)

                # Parse and evaluate
                prediction = extract_json_from_response(raw_output)
                ground_truth = sample.get("ground_truth", {})
                ground_truth_flat = sample.get("ground_truth_flat", {})

                metrics = compute_sample_metrics(
                    prediction, ground_truth, ground_truth_flat
                )

                result = {
                    "sample_id": sample_id,
                    "raw_output": raw_output,
                    "prediction": prediction,
                    "ground_truth": ground_truth,
                    "metrics": metrics,
                }

            except Exception as e:
                logger.warning(
                    f"Error on sample {sample_id} with strategy "
                    f"{strategy_name}: {e}"
                )
                metrics = {
                    "json_valid": False,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "exact_matches": 0,
                    "total_gt_fields": len(
                        sample.get("ground_truth_flat", {})
                    ),
                    "total_pred_fields": 0,
                }
                result = {
                    "sample_id": sample_id,
                    "raw_output": "",
                    "prediction": None,
                    "ground_truth": sample.get("ground_truth", {}),
                    "metrics": metrics,
                    "error": str(e),
                }

            per_sample_results.append(result)
            per_sample_metrics.append(metrics)

            if (i + 1) % 10 == 0:
                logger.info(
                    f"  [{strategy_name}] {i + 1}/{len(test_samples)} samples processed"
                )

        elapsed = time.time() - start_time
        aggregated = aggregate_metrics(per_sample_metrics)
        aggregated["elapsed_seconds"] = round(elapsed, 2)
        aggregated["avg_seconds_per_sample"] = round(elapsed / max(len(test_samples), 1), 2)

        logger.info(
            f"  [{strategy_name}] Done in {elapsed:.1f}s | "
            f"JSON valid: {aggregated['json_valid_rate']:.1%} | "
            f"Avg F1: {aggregated['avg_f1']:.3f}"
        )

        return {
            "strategy": strategy_name,
            "metrics": aggregated,
            "per_sample": per_sample_results,
        }

    def run_experiment(
        self,
        test_samples: list[dict[str, Any]],
        num_samples: int = 50,
    ) -> dict[str, Any]:
        """Run all strategies on the same set of test samples.

        Args:
            test_samples: Full list of available test samples.
            num_samples: Number of samples to evaluate (taken from the front
                of test_samples).

        Returns:
            Experiment results dict with metadata, per-strategy results,
            and a comparison summary.
        """
        # Limit samples
        samples = test_samples[:num_samples]
        logger.info(
            f"Starting experiment with {len(self.strategies)} strategies "
            f"on {len(samples)} samples"
        )

        experiment_start = time.time()
        strategy_results: list[dict[str, Any]] = []

        for strategy in self.strategies:
            result = self.run_strategy(strategy, samples)
            strategy_results.append(result)

        total_elapsed = time.time() - experiment_start

        # Build comparison summary
        comparison = []
        for res in strategy_results:
            comparison.append({
                "strategy": res["strategy"],
                "json_valid_rate": res["metrics"]["json_valid_rate"],
                "avg_f1": res["metrics"]["avg_f1"],
                "micro_f1": res["metrics"]["micro_f1"],
                "avg_precision": res["metrics"]["avg_precision"],
                "avg_recall": res["metrics"]["avg_recall"],
                "elapsed_seconds": res["metrics"]["elapsed_seconds"],
            })

        # Sort comparison by avg_f1 descending
        comparison.sort(key=lambda x: x["avg_f1"], reverse=True)

        experiment = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "num_strategies": len(self.strategies),
                "num_samples": len(samples),
                "total_elapsed_seconds": round(total_elapsed, 2),
                "device": self.device,
                "max_new_tokens": self.max_new_tokens,
            },
            "comparison": comparison,
            "strategy_results": strategy_results,
        }

        logger.info(
            f"Experiment complete in {total_elapsed:.1f}s. "
            f"Best strategy: {comparison[0]['strategy']} "
            f"(F1={comparison[0]['avg_f1']:.3f})"
        )

        return experiment

    @staticmethod
    def save_results(
        results: dict[str, Any],
        output_path: str | Path,
    ) -> Path:
        """Save experiment results to a JSON file.

        Strips PIL Image objects from per-sample data before serialization.

        Args:
            results: Experiment results dict from run_experiment().
            output_path: Path to save the JSON file.

        Returns:
            Path to the saved file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Deep-copy and strip non-serializable objects (PIL Images, etc.)
        def _make_serializable(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {
                    k: _make_serializable(v)
                    for k, v in obj.items()
                    if not isinstance(v, Image.Image)
                }
            if isinstance(obj, list):
                return [_make_serializable(item) for item in obj]
            if isinstance(obj, Image.Image):
                return "<image>"
            if isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            return str(obj)

        serializable = _make_serializable(results)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to {output_path}")
        return output_path

    @staticmethod
    def load_results(results_path: str | Path) -> dict[str, Any]:
        """Load experiment results from a JSON file.

        Args:
            results_path: Path to the saved results JSON.

        Returns:
            Experiment results dict.
        """
        with open(results_path, "r", encoding="utf-8") as f:
            return json.load(f)
