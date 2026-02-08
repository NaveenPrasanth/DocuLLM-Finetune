"""LLM-as-judge evaluation using Claude and GPT-4o.

Sends structured rubric prompts to LLM judges and aggregates scores
for completeness, accuracy, and format quality.
"""

from __future__ import annotations

import json
import logging
import random
import time
from typing import Any

import numpy as np

from src.config import LLMJudgeFullConfig, load_llm_judge_config

logger = logging.getLogger(__name__)

# Maximum retry attempts for API calls
_MAX_RETRIES = 5
_INITIAL_BACKOFF = 1.0


class LLMJudge:
    """Evaluate document extraction quality using LLM judges.

    Supports Claude (Anthropic) and GPT-4o (OpenAI) as judge providers.
    Computes inter-judge agreement via Cohen's kappa and Pearson correlation.

    Args:
        anthropic_key: Anthropic API key. If None, reads from ANTHROPIC_API_KEY env var.
        openai_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
        config: LLM judge configuration. Loaded from YAML if not provided.
    """

    def __init__(
        self,
        anthropic_key: str | None = None,
        openai_key: str | None = None,
        config: LLMJudgeFullConfig | None = None,
    ):
        self.config = config or load_llm_judge_config()
        self.judge_config = self.config.llm_judge

        # Lazy-init API clients
        self._anthropic_client = None
        self._openai_client = None
        self._anthropic_key = anthropic_key
        self._openai_key = openai_key

    @property
    def anthropic_client(self):
        """Lazily initialize the Anthropic client."""
        if self._anthropic_client is None:
            import anthropic

            kwargs: dict[str, Any] = {}
            if self._anthropic_key:
                kwargs["api_key"] = self._anthropic_key
            self._anthropic_client = anthropic.Anthropic(**kwargs)
        return self._anthropic_client

    @property
    def openai_client(self):
        """Lazily initialize the OpenAI client."""
        if self._openai_client is None:
            import openai

            kwargs: dict[str, Any] = {}
            if self._openai_key:
                kwargs["api_key"] = self._openai_key
            self._openai_client = openai.OpenAI(**kwargs)
        return self._openai_client

    def _build_judge_prompt(self, prediction: str, ground_truth: str) -> str:
        """Build the evaluation prompt from the config rubric template.

        Args:
            prediction: Model's extraction output.
            ground_truth: Expected ground truth extraction.

        Returns:
            Formatted evaluation prompt string.
        """
        rubric = self.judge_config.rubric
        template = self.judge_config.prompt_template

        return template.format(
            prediction=prediction,
            ground_truth=ground_truth,
            completeness_description=rubric["completeness"].description if "completeness" in rubric else "",
            accuracy_description=rubric["accuracy"].description if "accuracy" in rubric else "",
            format_description=rubric["format"].description if "format" in rubric else "",
        )

    def _call_anthropic(self, prompt: str, model: str) -> str:
        """Call the Anthropic API with exponential backoff retries.

        Args:
            prompt: The evaluation prompt to send.
            model: Anthropic model identifier.

        Returns:
            Raw response text from the model.

        Raises:
            RuntimeError: If all retry attempts fail.
        """
        import anthropic

        backoff = _INITIAL_BACKOFF
        last_error: Exception | None = None

        for attempt in range(_MAX_RETRIES):
            try:
                response = self.anthropic_client.messages.create(
                    model=model,
                    max_tokens=512,
                    temperature=0.0,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text
            except anthropic.RateLimitError as e:
                last_error = e
                sleep_time = backoff * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(
                    "Anthropic rate limit hit (attempt %d/%d), retrying in %.1fs",
                    attempt + 1, _MAX_RETRIES, sleep_time,
                )
                time.sleep(sleep_time)
            except anthropic.APIError as e:
                last_error = e
                if attempt < _MAX_RETRIES - 1:
                    sleep_time = backoff * (2 ** attempt)
                    logger.warning(
                        "Anthropic API error (attempt %d/%d): %s, retrying in %.1fs",
                        attempt + 1, _MAX_RETRIES, str(e), sleep_time,
                    )
                    time.sleep(sleep_time)
                else:
                    raise

        raise RuntimeError(
            f"Anthropic API call failed after {_MAX_RETRIES} attempts: {last_error}"
        )

    def _call_openai(self, prompt: str, model: str) -> str:
        """Call the OpenAI API with exponential backoff retries.

        Args:
            prompt: The evaluation prompt to send.
            model: OpenAI model identifier.

        Returns:
            Raw response text from the model.

        Raises:
            RuntimeError: If all retry attempts fail.
        """
        import openai

        backoff = _INITIAL_BACKOFF
        last_error: Exception | None = None

        for attempt in range(_MAX_RETRIES):
            try:
                response = self.openai_client.chat.completions.create(
                    model=model,
                    max_tokens=512,
                    temperature=0.0,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.choices[0].message.content
            except openai.RateLimitError as e:
                last_error = e
                sleep_time = backoff * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(
                    "OpenAI rate limit hit (attempt %d/%d), retrying in %.1fs",
                    attempt + 1, _MAX_RETRIES, sleep_time,
                )
                time.sleep(sleep_time)
            except openai.APIError as e:
                last_error = e
                if attempt < _MAX_RETRIES - 1:
                    sleep_time = backoff * (2 ** attempt)
                    logger.warning(
                        "OpenAI API error (attempt %d/%d): %s, retrying in %.1fs",
                        attempt + 1, _MAX_RETRIES, str(e), sleep_time,
                    )
                    time.sleep(sleep_time)
                else:
                    raise

        raise RuntimeError(
            f"OpenAI API call failed after {_MAX_RETRIES} attempts: {last_error}"
        )

    def _parse_judge_response(self, response_text: str) -> dict[str, Any]:
        """Parse the judge's JSON response into a scores dict.

        Handles cases where the response may contain markdown code blocks
        or extra text around the JSON.

        Args:
            response_text: Raw text response from the judge LLM.

        Returns:
            Dict with keys: completeness, accuracy, format, reasoning.
        """
        text = response_text.strip()

        # Strip markdown code fences if present
        if "```json" in text:
            text = text.split("```json", 1)[1]
            text = text.split("```", 1)[0]
        elif "```" in text:
            text = text.split("```", 1)[1]
            text = text.split("```", 1)[0]

        text = text.strip()

        try:
            parsed = json.loads(text)
            return {
                "completeness": int(parsed.get("completeness", 0)),
                "accuracy": int(parsed.get("accuracy", 0)),
                "format": int(parsed.get("format", 0)),
                "reasoning": str(parsed.get("reasoning", "")),
            }
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            logger.warning("Failed to parse judge response: %s. Raw: %s", e, text[:200])
            return {
                "completeness": 0,
                "accuracy": 0,
                "format": 0,
                "reasoning": f"Parse error: {e}",
            }

    def score_single(
        self,
        prediction: str,
        ground_truth: str,
        judge_provider: str = "anthropic",
    ) -> dict[str, Any]:
        """Score a single prediction using the specified judge provider.

        Args:
            prediction: Model's extraction output (JSON string).
            ground_truth: Ground truth extraction (JSON string).
            judge_provider: Either "anthropic" or "openai".

        Returns:
            Dict with scores: {completeness: int, accuracy: int, format: int, reasoning: str}.
        """
        prompt = self._build_judge_prompt(prediction, ground_truth)

        # Find the model name for this provider
        model = None
        for judge in self.judge_config.judges:
            if judge.provider == judge_provider:
                model = judge.model
                break

        if model is None:
            raise ValueError(
                f"No judge configured for provider '{judge_provider}'. "
                f"Available: {[j.provider for j in self.judge_config.judges]}"
            )

        if judge_provider == "anthropic":
            response = self._call_anthropic(prompt, model)
        elif judge_provider == "openai":
            response = self._call_openai(prompt, model)
        else:
            raise ValueError(f"Unsupported judge provider: {judge_provider}")

        return self._parse_judge_response(response)

    def score_batch(
        self,
        predictions: list[str],
        ground_truths: list[str],
        num_samples: int = 50,
    ) -> dict[str, Any]:
        """Score a batch of predictions with all configured judges.

        Randomly samples up to num_samples from the predictions if there are more.
        Returns per-judge scores and averaged scores across judges.

        Args:
            predictions: List of model extraction outputs (JSON strings).
            ground_truths: List of ground truth extractions (JSON strings).
            num_samples: Maximum number of samples to evaluate.

        Returns:
            Dict with per-judge results and averaged scores.
        """
        n = min(num_samples, len(predictions))

        # Sample indices if needed
        if n < len(predictions):
            indices = sorted(random.sample(range(len(predictions)), n))
        else:
            indices = list(range(n))

        sampled_preds = [predictions[i] for i in indices]
        sampled_gts = [ground_truths[i] for i in indices]

        judge_providers = [j.provider for j in self.judge_config.judges]
        all_judge_scores: dict[str, list[dict[str, Any]]] = {p: [] for p in judge_providers}

        for idx, (pred, gt) in enumerate(zip(sampled_preds, sampled_gts)):
            logger.info("Scoring sample %d/%d", idx + 1, n)
            for provider in judge_providers:
                try:
                    score = self.score_single(pred, gt, judge_provider=provider)
                    all_judge_scores[provider].append(score)
                except Exception as e:
                    logger.error(
                        "Failed to score sample %d with %s: %s", idx, provider, e
                    )
                    all_judge_scores[provider].append({
                        "completeness": 0,
                        "accuracy": 0,
                        "format": 0,
                        "reasoning": f"Error: {e}",
                    })

        # Compute per-judge averages
        per_judge_results: dict[str, dict[str, float]] = {}
        for provider, scores in all_judge_scores.items():
            if scores:
                per_judge_results[provider] = {
                    "completeness": _mean([s["completeness"] for s in scores]),
                    "accuracy": _mean([s["accuracy"] for s in scores]),
                    "format": _mean([s["format"] for s in scores]),
                    "overall": _mean([
                        (s["completeness"] + s["accuracy"] + s["format"]) / 3.0
                        for s in scores
                    ]),
                }

        # Average across judges
        if per_judge_results:
            avg_scores = {
                "completeness": _mean([v["completeness"] for v in per_judge_results.values()]),
                "accuracy": _mean([v["accuracy"] for v in per_judge_results.values()]),
                "format": _mean([v["format"] for v in per_judge_results.values()]),
                "overall": _mean([v["overall"] for v in per_judge_results.values()]),
            }
        else:
            avg_scores = {"completeness": 0.0, "accuracy": 0.0, "format": 0.0, "overall": 0.0}

        # Compute agreement if we have two judges
        agreement = {}
        if len(judge_providers) == 2:
            scores_a = all_judge_scores[judge_providers[0]]
            scores_b = all_judge_scores[judge_providers[1]]
            agreement = self.compute_agreement(scores_a, scores_b)

        return {
            "per_judge": per_judge_results,
            "per_judge_raw": {p: scores for p, scores in all_judge_scores.items()},
            "averaged": avg_scores,
            "agreement": agreement,
            "num_samples": n,
            "sample_indices": indices,
        }

    @staticmethod
    def compute_agreement(
        scores_a: list[dict[str, Any]],
        scores_b: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Compute inter-judge agreement between two sets of scores.

        Uses Cohen's kappa (on discretized overall scores) and Pearson correlation
        (on continuous overall scores).

        Args:
            scores_a: Scores from judge A (list of score dicts).
            scores_b: Scores from judge B (list of score dicts).

        Returns:
            Dict with cohen_kappa (per dimension + overall), pearson_r (per dimension).
        """
        if not scores_a or not scores_b:
            return {"error": "No scores to compare"}

        n = min(len(scores_a), len(scores_b))
        dimensions = ["completeness", "accuracy", "format"]

        agreement: dict[str, Any] = {}

        for dim in dimensions:
            a_values = np.array([scores_a[i].get(dim, 0) for i in range(n)], dtype=float)
            b_values = np.array([scores_b[i].get(dim, 0) for i in range(n)], dtype=float)

            # Pearson correlation
            if np.std(a_values) > 0 and np.std(b_values) > 0:
                pearson_r = float(np.corrcoef(a_values, b_values)[0, 1])
            else:
                pearson_r = 0.0

            # Cohen's kappa
            kappa = _cohens_kappa(a_values.astype(int), b_values.astype(int))

            agreement[dim] = {
                "pearson_r": pearson_r,
                "cohens_kappa": kappa,
            }

        # Overall agreement (average of per-dimension)
        overall_a = np.array([
            (scores_a[i].get("completeness", 0) + scores_a[i].get("accuracy", 0) + scores_a[i].get("format", 0)) / 3.0
            for i in range(n)
        ])
        overall_b = np.array([
            (scores_b[i].get("completeness", 0) + scores_b[i].get("accuracy", 0) + scores_b[i].get("format", 0)) / 3.0
            for i in range(n)
        ])

        if np.std(overall_a) > 0 and np.std(overall_b) > 0:
            overall_pearson = float(np.corrcoef(overall_a, overall_b)[0, 1])
        else:
            overall_pearson = 0.0

        agreement["overall"] = {
            "pearson_r": overall_pearson,
            "cohens_kappa": _cohens_kappa(
                np.round(overall_a).astype(int),
                np.round(overall_b).astype(int),
            ),
        }

        return agreement


# ── Helpers ─────────────────────────────────────────────────────────────────


def _mean(values: list[float]) -> float:
    """Compute mean, returning 0.0 for empty lists."""
    return sum(values) / len(values) if values else 0.0


def _cohens_kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Cohen's kappa for two arrays of integer labels.

    Args:
        a: Ratings from judge A.
        b: Ratings from judge B.

    Returns:
        Cohen's kappa coefficient.
    """
    if len(a) == 0:
        return 0.0

    # Build confusion matrix
    labels = sorted(set(a.tolist()) | set(b.tolist()))
    n_labels = len(labels)

    if n_labels <= 1:
        return 1.0  # Perfect agreement when only one category

    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    matrix = np.zeros((n_labels, n_labels), dtype=float)

    for ai, bi in zip(a, b):
        matrix[label_to_idx[int(ai)], label_to_idx[int(bi)]] += 1

    n = float(matrix.sum())
    if n == 0:
        return 0.0

    # Observed agreement
    po = np.trace(matrix) / n

    # Expected agreement (by chance)
    row_sums = matrix.sum(axis=1)
    col_sums = matrix.sum(axis=0)
    pe = float(np.sum(row_sums * col_sums)) / (n * n)

    if pe == 1.0:
        return 1.0

    return float((po - pe) / (1.0 - pe))
