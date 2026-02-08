"""Visualization and report generation for evaluation results.

Provides bar charts, radar plots, heatmaps, training curves, and
full HTML/markdown report generation using matplotlib and seaborn.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# Global style settings
_PALETTE = "viridis"
_FIG_DPI = 150
_FONT_SIZE = 11


def _setup_style() -> None:
    """Apply consistent plot styling."""
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams.update({
        "figure.dpi": _FIG_DPI,
        "font.size": _FONT_SIZE,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    })


def plot_metric_comparison(
    results: dict[str, dict[str, float]],
    metric_name: str,
    save_path: str,
    title: str | None = None,
) -> None:
    """Bar chart comparing models on a single metric.

    Args:
        results: Mapping of model_name -> metrics_dict.
        metric_name: Name of the metric to plot.
        save_path: File path to save the figure.
        title: Optional custom title; defaults to metric name.
    """
    _setup_style()

    models = list(results.keys())
    scores = [results[m].get(metric_name, 0.0) for m in models]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette(_PALETTE, n_colors=len(models))

    bars = ax.bar(models, scores, color=colors, edgecolor="white", linewidth=1.2)

    # Add value labels on bars
    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{score:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )

    ax.set_ylabel("Score")
    ax.set_title(title or f"Model Comparison: {metric_name}")
    ax.set_ylim(0, min(1.15, max(scores) * 1.2) if scores else 1.0)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved metric comparison chart to %s", save_path)


def plot_radar_chart(
    results: dict[str, dict[str, float]],
    save_path: str,
    metrics: list[str] | None = None,
    title: str = "Model Comparison Radar Chart",
) -> None:
    """Radar/spider plot of all metrics for all models.

    Args:
        results: Mapping of model_name -> metrics_dict.
        save_path: File path to save the figure.
        metrics: List of metric names to include. Auto-detected if None.
        title: Chart title.
    """
    _setup_style()

    if metrics is None:
        # Collect all numeric metrics across models (exclude num_samples)
        all_metrics: set[str] = set()
        for model_metrics in results.values():
            for k, v in model_metrics.items():
                if isinstance(v, (int, float)) and k != "num_samples":
                    all_metrics.add(k)
        metrics = sorted(all_metrics)

    if not metrics:
        logger.warning("No metrics found for radar chart")
        return

    n_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    colors = sns.color_palette("husl", n_colors=len(results))

    for idx, (model_name, model_metrics) in enumerate(results.items()):
        values = [model_metrics.get(m, 0.0) for m in metrics]
        values += values[:1]  # Close the polygon

        ax.plot(angles, values, "o-", linewidth=2, label=model_name, color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_title(title, size=14, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved radar chart to %s", save_path)


def plot_per_field_heatmap(
    per_field_results: pd.DataFrame,
    save_path: str,
    title: str = "Per-Field F1 Scores by Model",
) -> None:
    """Seaborn heatmap of per-field F1 scores across models.

    Args:
        per_field_results: DataFrame with fields as rows and models as columns.
            Values should be F1 scores.
        save_path: File path to save the figure.
        title: Chart title.
    """
    _setup_style()

    if per_field_results.empty:
        logger.warning("Empty DataFrame for heatmap, skipping")
        return

    # Determine figure size based on data dimensions
    n_fields = len(per_field_results)
    n_models = len(per_field_results.columns)
    fig_height = max(8, n_fields * 0.4)
    fig_width = max(8, n_models * 2 + 3)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    sns.heatmap(
        per_field_results,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5,
        cbar_kws={"label": "F1 Score"},
        ax=ax,
    )

    ax.set_title(title, fontsize=14, pad=15)
    ax.set_ylabel("Field Type")
    ax.set_xlabel("Model")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved per-field heatmap to %s", save_path)


def plot_training_curves(
    wandb_run_id: str,
    save_path: str,
    local_log_path: str | None = None,
) -> None:
    """Plot training and validation loss curves.

    Attempts to load from W&B first. Falls back to a local JSON log file
    if W&B is unavailable or the run ID is not found.

    Local log format: {"steps": [...], "train_loss": [...], "eval_loss": [...]}

    Args:
        wandb_run_id: Weights & Biases run ID (e.g., "entity/project/run_id").
        save_path: File path to save the figure.
        local_log_path: Optional path to a local JSON log file as fallback.
    """
    _setup_style()

    steps = None
    train_loss = None
    eval_loss = None
    eval_steps = None

    # Try W&B first
    try:
        import wandb

        api = wandb.Api()
        run = api.run(wandb_run_id)
        history = run.history()

        if "train/loss" in history.columns:
            train_data = history[["_step", "train/loss"]].dropna()
            steps = train_data["_step"].tolist()
            train_loss = train_data["train/loss"].tolist()

        if "eval/loss" in history.columns:
            eval_data = history[["_step", "eval/loss"]].dropna()
            eval_steps = eval_data["_step"].tolist()
            eval_loss = eval_data["eval/loss"].tolist()

        logger.info("Loaded training curves from W&B run: %s", wandb_run_id)

    except Exception as e:
        logger.warning("Could not load from W&B (%s), trying local log", e)

        if local_log_path and Path(local_log_path).exists():
            with open(local_log_path) as f:
                log_data = json.load(f)

            steps = log_data.get("steps", [])
            train_loss = log_data.get("train_loss", [])
            eval_loss = log_data.get("eval_loss", [])
            eval_steps = log_data.get("eval_steps", steps)

            logger.info("Loaded training curves from local log: %s", local_log_path)
        else:
            logger.error("No training data available for plotting")
            return

    if not steps or not train_loss:
        logger.error("No training loss data to plot")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(steps, train_loss, label="Training Loss", color="#2196F3", alpha=0.7, linewidth=1.5)

    if eval_loss and eval_steps:
        ax.plot(eval_steps, eval_loss, label="Validation Loss", color="#F44336", linewidth=2, marker="o", markersize=4)

    ax.set_xlabel("Steps")
    ax.set_ylabel("Loss")
    ax.set_title("Training Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add smoothed training loss
    if len(train_loss) > 20:
        window = max(5, len(train_loss) // 20)
        smoothed = pd.Series(train_loss).rolling(window=window, center=True).mean()
        ax.plot(steps, smoothed, label="Training Loss (smoothed)", color="#1565C0", linewidth=2)
        ax.legend()

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved training curves to %s", save_path)


def generate_report(
    results: dict[str, dict[str, float]],
    output_dir: str,
    per_field_results: pd.DataFrame | None = None,
    llm_judge_results: dict[str, Any] | None = None,
    significance_results: dict[str, Any] | None = None,
) -> str:
    """Generate a full HTML/markdown report with all charts and tables.

    Args:
        results: Mapping of model_name -> aggregated metrics dict.
        output_dir: Directory to save the report and associated charts.
        per_field_results: Optional per-field comparison DataFrame.
        llm_judge_results: Optional LLM judge evaluation results.
        significance_results: Optional significance test results.

    Returns:
        Path to the generated report file.
    """
    _setup_style()

    out = Path(output_dir)
    charts_dir = out / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    # Generate all charts
    metrics_to_plot = [
        "field_f1_micro", "field_f1_macro", "exact_match",
        "anls", "json_validity", "schema_compliance",
    ]

    chart_paths: dict[str, str] = {}

    for metric in metrics_to_plot:
        chart_path = str(charts_dir / f"{metric}_comparison.png")
        plot_metric_comparison(results, metric, chart_path)
        chart_paths[metric] = f"charts/{metric}_comparison.png"

    # Radar chart
    radar_path = str(charts_dir / "radar_comparison.png")
    plot_radar_chart(results, radar_path, metrics=metrics_to_plot)
    chart_paths["radar"] = "charts/radar_comparison.png"

    # Per-field heatmap
    if per_field_results is not None and not per_field_results.empty:
        heatmap_path = str(charts_dir / "per_field_heatmap.png")
        plot_per_field_heatmap(per_field_results, heatmap_path)
        chart_paths["heatmap"] = "charts/per_field_heatmap.png"

    # Build markdown report
    report_lines: list[str] = []
    report_lines.append("# DocuMind Evaluation Report")
    report_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Overall comparison table
    report_lines.append("## Overall Model Comparison\n")
    comparison_df = pd.DataFrame(results).T
    report_lines.append(comparison_df.to_markdown(floatfmt=".4f"))
    report_lines.append("")

    # Metric comparison charts
    report_lines.append("## Metric Comparisons\n")
    for metric in metrics_to_plot:
        if metric in chart_paths:
            report_lines.append(f"### {metric}\n")
            report_lines.append(f"![{metric}]({chart_paths[metric]})\n")

    # Radar chart
    report_lines.append("## Multi-Metric Radar Chart\n")
    report_lines.append(f"![Radar Chart]({chart_paths['radar']})\n")

    # Per-field heatmap
    if "heatmap" in chart_paths:
        report_lines.append("## Per-Field F1 Heatmap\n")
        report_lines.append(f"![Heatmap]({chart_paths['heatmap']})\n")

    # LLM Judge results
    if llm_judge_results:
        report_lines.append("## LLM Judge Evaluation\n")

        if "per_judge" in llm_judge_results:
            report_lines.append("### Per-Judge Scores\n")
            judge_df = pd.DataFrame(llm_judge_results["per_judge"]).T
            report_lines.append(judge_df.to_markdown(floatfmt=".2f"))
            report_lines.append("")

        if "averaged" in llm_judge_results:
            report_lines.append("### Averaged Scores\n")
            avg = llm_judge_results["averaged"]
            for dim, score in avg.items():
                report_lines.append(f"- **{dim}**: {score:.2f}")
            report_lines.append("")

        if "agreement" in llm_judge_results and llm_judge_results["agreement"]:
            report_lines.append("### Inter-Judge Agreement\n")
            for dim, metrics in llm_judge_results["agreement"].items():
                if isinstance(metrics, dict):
                    kappa = metrics.get("cohens_kappa", "N/A")
                    pearson = metrics.get("pearson_r", "N/A")
                    report_lines.append(
                        f"- **{dim}**: Cohen's kappa = {kappa:.3f}, "
                        f"Pearson r = {pearson:.3f}"
                        if isinstance(kappa, float) else
                        f"- **{dim}**: {metrics}"
                    )
            report_lines.append("")

    # Significance tests
    if significance_results:
        report_lines.append("## Statistical Significance Tests\n")
        for pair, pair_results in significance_results.items():
            report_lines.append(f"### {pair}\n")
            if isinstance(pair_results, dict):
                for metric, sig in pair_results.items():
                    if isinstance(sig, dict):
                        p_val = sig.get("p_value", "N/A")
                        diff = sig.get("mean_diff", 0)
                        is_sig = sig.get("significant", False)
                        marker = " *" if is_sig else ""
                        report_lines.append(
                            f"- **{metric}**: diff = {diff:+.4f}, "
                            f"p = {p_val:.4f}{marker}"
                        )
            report_lines.append("")

    # Write report
    report_content = "\n".join(report_lines)
    report_path = out / "evaluation_report.md"
    report_path.write_text(report_content, encoding="utf-8")

    # Also save results as JSON
    results_json_path = out / "evaluation_results.json"
    serializable_results = _make_serializable(results)
    with open(results_json_path, "w") as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)

    logger.info("Generated evaluation report at %s", report_path)
    return str(report_path)


def _make_serializable(obj: Any) -> Any:
    """Convert numpy types and other non-serializable objects for JSON output."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict()
    return obj
