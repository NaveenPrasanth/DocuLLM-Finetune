"""Evaluation metrics for document extraction quality.

Provides field-level F1, exact match, ANLS, JSON validity, schema compliance,
and aggregated metric computation across predictions.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from typing import Any

from rapidfuzz.distance import Levenshtein

logger = logging.getLogger(__name__)


# ── Field-Level F1 ──────────────────────────────────────────────────────────


def _flatten_dict(d: dict[str, Any], parent_key: str = "", sep: str = ".") -> dict[str, str]:
    """Recursively flatten a nested dict into dotted-key → string-value pairs."""
    items: dict[str, str] = {}
    for key, value in d.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            items.update(_flatten_dict(value, new_key, sep))
        elif isinstance(value, list):
            for idx, item in enumerate(value):
                indexed_key = f"{new_key}{sep}{idx}"
                if isinstance(item, dict):
                    items.update(_flatten_dict(item, indexed_key, sep))
                elif item is not None:
                    items[indexed_key] = str(item)
        elif value is not None:
            items[new_key] = str(value)
    return items


def compute_field_f1(
    predictions: dict[str, Any],
    ground_truth: dict[str, Any],
    average: str = "micro",
) -> dict[str, float]:
    """Compute field-level F1 score by treating each field as a binary classification.

    A field is considered a true positive if the key exists in both predictions and
    ground truth AND the values match exactly (after string normalization).

    Args:
        predictions: Predicted extraction dict (potentially nested).
        ground_truth: Ground truth extraction dict (potentially nested).
        average: Averaging method - "micro" (global counts) or "macro" (per-field average).

    Returns:
        Dict with keys: precision, recall, f1, and additionally micro/macro variants
        if average is not specified.
    """
    pred_flat = _flatten_dict(predictions)
    gt_flat = _flatten_dict(ground_truth)

    pred_keys = set(pred_flat.keys())
    gt_keys = set(gt_flat.keys())

    # Keys present in both
    common_keys = pred_keys & gt_keys

    # True positives: key present in both AND value matches
    tp = sum(
        1 for k in common_keys
        if pred_flat[k].strip() == gt_flat[k].strip()
    )
    # False positives: keys in predictions but not in GT, or in both but value wrong
    fp = len(pred_keys) - tp
    # False negatives: keys in GT but not in predictions
    fn = len(gt_keys - pred_keys) + sum(
        1 for k in common_keys
        if pred_flat[k].strip() != gt_flat[k].strip()
    )

    if average == "micro":
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        return {"precision": precision, "recall": recall, "f1": f1}

    # Macro: compute per-field-type metrics and average
    # Group by field type (strip numeric indices)
    field_type_metrics: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    all_field_types = set()

    for k in gt_keys:
        field_type = _get_field_type(k)
        all_field_types.add(field_type)
        if k in pred_keys and pred_flat.get(k, "").strip() == gt_flat[k].strip():
            field_type_metrics[field_type]["tp"] += 1
        else:
            field_type_metrics[field_type]["fn"] += 1

    for k in pred_keys:
        field_type = _get_field_type(k)
        all_field_types.add(field_type)
        if k not in gt_keys:
            field_type_metrics[field_type]["fp"] += 1
        elif pred_flat[k].strip() != gt_flat.get(k, "").strip():
            field_type_metrics[field_type]["fp"] += 1

    per_field_f1 = []
    per_field_precision = []
    per_field_recall = []

    for ft in all_field_types:
        m = field_type_metrics[ft]
        p = m["tp"] / (m["tp"] + m["fp"]) if (m["tp"] + m["fp"]) > 0 else 0.0
        r = m["tp"] / (m["tp"] + m["fn"]) if (m["tp"] + m["fn"]) > 0 else 0.0
        f = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        per_field_precision.append(p)
        per_field_recall.append(r)
        per_field_f1.append(f)

    n = len(all_field_types) if all_field_types else 1
    return {
        "precision": sum(per_field_precision) / n,
        "recall": sum(per_field_recall) / n,
        "f1": sum(per_field_f1) / n,
    }


def _get_field_type(dotted_key: str) -> str:
    """Extract the canonical field type from a dotted key, removing list indices.

    E.g., 'menu.0.nm' -> 'menu.nm', 'total.total_price' -> 'total.total_price'
    """
    parts = dotted_key.split(".")
    return ".".join(p for p in parts if not p.isdigit())


# ── Exact Match ─────────────────────────────────────────────────────────────


def compute_exact_match(pred_json: str, gt_json: str) -> float:
    """Return 1.0 if the parsed JSONs are structurally identical, 0.0 otherwise.

    Args:
        pred_json: Predicted JSON string.
        gt_json: Ground truth JSON string.

    Returns:
        1.0 for exact match, 0.0 otherwise.
    """
    try:
        pred = json.loads(pred_json)
        gt = json.loads(gt_json)
        return 1.0 if pred == gt else 0.0
    except (json.JSONDecodeError, TypeError):
        return 0.0


# ── ANLS (Average Normalized Levenshtein Similarity) ────────────────────────


def compute_anls(pred_text: str, gt_text: str, threshold: float = 0.5) -> float:
    """Compute Average Normalized Levenshtein Similarity.

    Uses rapidfuzz for efficient edit distance computation. Returns the normalized
    similarity score if it meets the threshold, otherwise 0.0.

    Args:
        pred_text: Predicted text string.
        gt_text: Ground truth text string.
        threshold: Minimum similarity to count as a match (default 0.5).

    Returns:
        Similarity score in [0.0, 1.0], or 0.0 if below threshold.
    """
    if not gt_text and not pred_text:
        return 1.0
    if not gt_text or not pred_text:
        return 0.0

    # Levenshtein.normalized_similarity returns value in [0, 1]
    similarity = Levenshtein.normalized_similarity(pred_text, gt_text)
    return similarity if similarity >= threshold else 0.0


# ── JSON Validity ───────────────────────────────────────────────────────────


def compute_json_validity(text: str) -> float:
    """Check if the given text is valid JSON.

    Args:
        text: Raw text to validate.

    Returns:
        1.0 if valid JSON, 0.0 otherwise.
    """
    try:
        json.loads(text)
        return 1.0
    except (json.JSONDecodeError, TypeError):
        return 0.0


# ── Schema Compliance ───────────────────────────────────────────────────────


def compute_schema_compliance(pred_json: str, expected_keys: list[str]) -> float:
    """Check what fraction of expected top-level keys are present in the prediction.

    Args:
        pred_json: Predicted JSON string.
        expected_keys: List of expected top-level keys.

    Returns:
        Fraction of expected keys present, in [0.0, 1.0].
    """
    if not expected_keys:
        return 1.0

    try:
        pred = json.loads(pred_json)
        if not isinstance(pred, dict):
            return 0.0
    except (json.JSONDecodeError, TypeError):
        return 0.0

    present = sum(1 for key in expected_keys if key in pred)
    return present / len(expected_keys)


# ── Aggregated Metrics ──────────────────────────────────────────────────────


def compute_all_metrics(
    predictions: list[dict[str, Any]],
    ground_truths: list[dict[str, Any]],
) -> dict[str, float]:
    """Run all metrics and return aggregated results.

    Each element in predictions and ground_truths should have keys:
        - "parsed": dict (the parsed JSON extraction)
        - "raw": str (the raw model output / JSON string)
        - "ground_truth": dict (the ground truth extraction)
        - "ground_truth_json": str (the GT as a JSON string)

    Args:
        predictions: List of prediction dicts with "parsed" and "raw" keys.
        ground_truths: List of ground truth dicts with "ground_truth" and
            "ground_truth_json" keys.

    Returns:
        Aggregated metrics dict with means for all metric types.
    """
    n = len(predictions)
    if n == 0:
        logger.warning("No samples provided for evaluation")
        return {}

    # Expected top-level keys from CORD schema
    expected_keys = ["menu", "total", "sub_total", "void_menu", "etc"]

    # Accumulators
    micro_f1_scores = []
    macro_f1_scores = []
    exact_match_scores = []
    anls_scores = []
    json_validity_scores = []
    schema_compliance_scores = []

    for pred, gt in zip(predictions, ground_truths):
        pred_parsed = pred.get("parsed", {})
        pred_raw = pred.get("raw", "")
        gt_parsed = gt.get("ground_truth", {})
        gt_json = gt.get("ground_truth_json", json.dumps(gt_parsed))

        # Field F1 (micro)
        micro_result = compute_field_f1(pred_parsed, gt_parsed, average="micro")
        micro_f1_scores.append(micro_result["f1"])

        # Field F1 (macro)
        macro_result = compute_field_f1(pred_parsed, gt_parsed, average="macro")
        macro_f1_scores.append(macro_result["f1"])

        # Exact match
        pred_json_str = json.dumps(pred_parsed, ensure_ascii=False) if pred_parsed else pred_raw
        exact_match_scores.append(compute_exact_match(pred_json_str, gt_json))

        # ANLS (computed on the raw JSON strings)
        anls_scores.append(compute_anls(pred_raw, gt_json))

        # JSON validity
        json_validity_scores.append(compute_json_validity(pred_raw))

        # Schema compliance
        schema_compliance_scores.append(
            compute_schema_compliance(pred_raw, expected_keys)
        )

    return {
        "field_f1_micro": _safe_mean(micro_f1_scores),
        "field_f1_macro": _safe_mean(macro_f1_scores),
        "exact_match": _safe_mean(exact_match_scores),
        "anls": _safe_mean(anls_scores),
        "json_validity": _safe_mean(json_validity_scores),
        "schema_compliance": _safe_mean(schema_compliance_scores),
        "num_samples": n,
    }


# ── Per-Field Metrics ───────────────────────────────────────────────────────


def compute_per_field_metrics(
    predictions: list[dict[str, Any]],
    ground_truths: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """Compute per-field breakdown of F1 scores across all samples.

    Args:
        predictions: List of prediction dicts with "parsed" key.
        ground_truths: List of ground truth dicts with "ground_truth" key.

    Returns:
        Dict mapping field_type -> {"precision": float, "recall": float, "f1": float}.
    """
    # Aggregate TP/FP/FN counts per field type across all samples
    field_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for pred, gt in zip(predictions, ground_truths):
        pred_parsed = pred.get("parsed", {})
        gt_parsed = gt.get("ground_truth", {})

        pred_flat = _flatten_dict(pred_parsed)
        gt_flat = _flatten_dict(gt_parsed)

        pred_keys = set(pred_flat.keys())
        gt_keys = set(gt_flat.keys())
        common_keys = pred_keys & gt_keys

        # Process ground truth keys
        for k in gt_keys:
            ft = _get_field_type(k)
            if k in common_keys and pred_flat[k].strip() == gt_flat[k].strip():
                field_counts[ft]["tp"] += 1
            else:
                field_counts[ft]["fn"] += 1

        # Process prediction keys not in GT
        for k in pred_keys:
            ft = _get_field_type(k)
            if k not in gt_keys:
                field_counts[ft]["fp"] += 1
            elif pred_flat[k].strip() != gt_flat.get(k, "").strip():
                field_counts[ft]["fp"] += 1

    # Compute per-field metrics
    results: dict[str, dict[str, float]] = {}
    for ft, counts in sorted(field_counts.items()):
        tp = counts["tp"]
        fp = counts["fp"]
        fn = counts["fn"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        results[ft] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp + fn,  # total GT occurrences
        }

    return results


# ── Helpers ─────────────────────────────────────────────────────────────────


def _safe_mean(values: list[float]) -> float:
    """Compute mean of a list, returning 0.0 for empty lists."""
    return sum(values) / len(values) if values else 0.0
