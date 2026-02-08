"""Model comparison and statistical significance testing.

Provides bootstrap confidence intervals, paired bootstrap significance tests,
and rich-formatted comparison tables for document extraction models.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ModelComparator:
    """Compare evaluation results across multiple models.

    Computes pairwise significance tests, confidence intervals, and generates
    formatted comparison tables.
    """

    # Default metrics to include in comparisons
    DEFAULT_METRICS = [
        "field_f1_micro",
        "field_f1_macro",
        "exact_match",
        "anls",
        "json_validity",
        "schema_compliance",
    ]

    def compare(
        self,
        results: dict[str, dict[str, float]],
        metrics: list[str] | None = None,
    ) -> pd.DataFrame:
        """Create a comparison DataFrame from model results.

        Args:
            results: Mapping of model_name -> metrics_dict.
                Each metrics_dict should contain metric_name -> score float.
            metrics: List of metric names to include. Uses defaults if None.

        Returns:
            DataFrame with models as rows and metrics as columns,
            plus a 'mean' column.
        """
        metrics = metrics or self.DEFAULT_METRICS
        rows = []

        for model_name, model_metrics in results.items():
            row = {"model": model_name}
            metric_values = []
            for metric in metrics:
                value = model_metrics.get(metric, float("nan"))
                row[metric] = value
                if not np.isnan(value):
                    metric_values.append(value)
            row["mean"] = float(np.mean(metric_values)) if metric_values else 0.0
            rows.append(row)

        df = pd.DataFrame(rows).set_index("model")

        # Sort by mean score descending
        df = df.sort_values("mean", ascending=False)

        return df

    @staticmethod
    def bootstrap_confidence_interval(
        scores: list[float] | np.ndarray,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
        seed: int = 42,
    ) -> tuple[float, float]:
        """Compute bootstrap confidence interval for a set of scores.

        Args:
            scores: Array of per-sample scores.
            n_bootstrap: Number of bootstrap resamples.
            confidence: Confidence level (e.g., 0.95 for 95% CI).
            seed: Random seed for reproducibility.

        Returns:
            Tuple of (lower_bound, upper_bound).
        """
        scores = np.asarray(scores, dtype=float)
        if len(scores) == 0:
            return (0.0, 0.0)

        rng = np.random.RandomState(seed)
        bootstrap_means = np.empty(n_bootstrap)

        for i in range(n_bootstrap):
            sample = rng.choice(scores, size=len(scores), replace=True)
            bootstrap_means[i] = np.mean(sample)

        alpha = 1.0 - confidence
        lower = float(np.percentile(bootstrap_means, 100 * alpha / 2))
        upper = float(np.percentile(bootstrap_means, 100 * (1.0 - alpha / 2)))

        return (lower, upper)

    @staticmethod
    def compute_significance(
        scores_a: list[float] | np.ndarray,
        scores_b: list[float] | np.ndarray,
        n_bootstrap: int = 1000,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Bootstrap paired significance test between two models.

        Tests whether model B is significantly different from model A by
        computing bootstrap distribution of the difference in means.

        Args:
            scores_a: Per-sample scores for model A.
            scores_b: Per-sample scores for model B.
            n_bootstrap: Number of bootstrap resamples.
            seed: Random seed for reproducibility.

        Returns:
            Dict with keys: mean_diff, p_value, significant (at alpha=0.05),
            ci_lower, ci_upper for the difference.
        """
        scores_a = np.asarray(scores_a, dtype=float)
        scores_b = np.asarray(scores_b, dtype=float)

        n = min(len(scores_a), len(scores_b))
        if n == 0:
            return {
                "mean_diff": 0.0,
                "p_value": 1.0,
                "significant": False,
                "ci_lower": 0.0,
                "ci_upper": 0.0,
            }

        scores_a = scores_a[:n]
        scores_b = scores_b[:n]

        observed_diff = float(np.mean(scores_b) - np.mean(scores_a))

        # Paired bootstrap: resample the differences
        diffs = scores_b - scores_a
        rng = np.random.RandomState(seed)
        bootstrap_diffs = np.empty(n_bootstrap)

        for i in range(n_bootstrap):
            sample_indices = rng.choice(n, size=n, replace=True)
            bootstrap_diffs[i] = np.mean(diffs[sample_indices])

        # Two-sided p-value: fraction of bootstrap samples where the sign differs
        # from the observed difference (or is zero)
        if observed_diff > 0:
            p_value = float(np.mean(bootstrap_diffs <= 0))
        elif observed_diff < 0:
            p_value = float(np.mean(bootstrap_diffs >= 0))
        else:
            p_value = 1.0

        # Two-sided p-value
        p_value = min(2 * p_value, 1.0)

        ci_lower = float(np.percentile(bootstrap_diffs, 2.5))
        ci_upper = float(np.percentile(bootstrap_diffs, 97.5))

        return {
            "mean_diff": observed_diff,
            "p_value": p_value,
            "significant": p_value < 0.05,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }

    def generate_comparison_table(
        self,
        results: dict[str, dict[str, float]],
        metrics: list[str] | None = None,
    ) -> str:
        """Generate a rich-formatted comparison table.

        Uses the rich library for terminal-friendly table rendering.

        Args:
            results: Mapping of model_name -> metrics_dict.
            metrics: List of metric names to include.

        Returns:
            Formatted table string.
        """
        from rich.console import Console
        from rich.table import Table
        import io

        metrics = metrics or self.DEFAULT_METRICS
        df = self.compare(results, metrics)

        table = Table(title="Model Comparison", show_lines=True)
        table.add_column("Model", style="bold cyan", min_width=12)

        for metric in metrics:
            table.add_column(metric, justify="right", min_width=10)
        table.add_column("Mean", justify="right", style="bold green", min_width=8)

        # Find best score per column for highlighting
        best_per_metric: dict[str, float] = {}
        for metric in metrics:
            col = df[metric].dropna()
            if not col.empty:
                best_per_metric[metric] = col.max()

        best_mean = df["mean"].max() if not df.empty else 0.0

        for model_name, row in df.iterrows():
            cells = [str(model_name)]
            for metric in metrics:
                value = row[metric]
                if np.isnan(value):
                    cells.append("-")
                else:
                    formatted = f"{value:.4f}"
                    if metric in best_per_metric and value == best_per_metric[metric]:
                        formatted = f"[bold]{formatted}[/bold]"
                    cells.append(formatted)

            mean_val = row["mean"]
            mean_str = f"{mean_val:.4f}"
            if mean_val == best_mean:
                mean_str = f"[bold]{mean_str}[/bold]"
            cells.append(mean_str)

            table.add_row(*cells)

        # Render to string
        buf = io.StringIO()
        console = Console(file=buf, width=120)
        console.print(table)
        return buf.getvalue()

    def per_field_comparison(
        self,
        results: dict[str, dict[str, dict[str, float]]],
    ) -> pd.DataFrame:
        """Create a per-field comparison DataFrame across models.

        Args:
            results: Mapping of model_name -> {field_type -> {precision, recall, f1, support}}.

        Returns:
            DataFrame with multi-level index (model, field) and metric columns.
        """
        rows = []
        for model_name, per_field in results.items():
            for field_type, field_metrics in per_field.items():
                rows.append({
                    "model": model_name,
                    "field": field_type,
                    "precision": field_metrics.get("precision", 0.0),
                    "recall": field_metrics.get("recall", 0.0),
                    "f1": field_metrics.get("f1", 0.0),
                    "support": field_metrics.get("support", 0),
                })

        if not rows:
            return pd.DataFrame(columns=["model", "field", "precision", "recall", "f1", "support"])

        df = pd.DataFrame(rows)

        # Pivot to get models as columns for easy comparison
        pivot_df = df.pivot_table(
            index="field",
            columns="model",
            values="f1",
            aggfunc="first",
        )

        return pivot_df.sort_index()

    def full_comparison(
        self,
        results: dict[str, dict[str, float]],
        per_sample_scores: dict[str, dict[str, list[float]]] | None = None,
        n_bootstrap: int = 1000,
    ) -> dict[str, Any]:
        """Run full comparison including significance tests.

        Args:
            results: Mapping of model_name -> aggregated metrics dict.
            per_sample_scores: Optional mapping of model_name -> {metric -> [per-sample scores]}.
                Required for significance tests and confidence intervals.
            n_bootstrap: Number of bootstrap resamples.

        Returns:
            Dict with comparison_table (DataFrame), significance_tests, and
            confidence_intervals.
        """
        comparison_df = self.compare(results)
        output: dict[str, Any] = {"comparison_table": comparison_df}

        if per_sample_scores is None:
            return output

        model_names = list(per_sample_scores.keys())
        metrics = list(next(iter(per_sample_scores.values())).keys()) if per_sample_scores else []

        # Confidence intervals
        ci_results: dict[str, dict[str, tuple[float, float]]] = {}
        for model_name, metric_scores in per_sample_scores.items():
            ci_results[model_name] = {}
            for metric, scores in metric_scores.items():
                ci_results[model_name][metric] = self.bootstrap_confidence_interval(
                    scores, n_bootstrap=n_bootstrap
                )
        output["confidence_intervals"] = ci_results

        # Pairwise significance tests
        sig_results: dict[str, dict[str, dict[str, Any]]] = {}
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                pair_key = f"{model_names[i]}_vs_{model_names[j]}"
                sig_results[pair_key] = {}
                for metric in metrics:
                    scores_a = per_sample_scores[model_names[i]].get(metric, [])
                    scores_b = per_sample_scores[model_names[j]].get(metric, [])
                    if scores_a and scores_b:
                        sig_results[pair_key][metric] = self.compute_significance(
                            scores_a, scores_b, n_bootstrap=n_bootstrap
                        )
        output["significance_tests"] = sig_results

        return output
