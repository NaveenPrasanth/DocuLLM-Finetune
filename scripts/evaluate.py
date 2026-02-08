"""Main evaluation entry point for DocuMind.

Runs inference with base, prompted, and fine-tuned models on the CORD test set,
computes all metrics, optionally runs LLM judge evaluation, and generates
comparison reports with charts.

Usage:
    python scripts/evaluate.py --config configs/evaluation/metrics.yaml
    python scripts/evaluate.py --model-path outputs/qlora_qwen2vl_cord/checkpoint-best --llm-judge
    python scripts/evaluate.py --num-samples 20 --output-dir outputs/eval_quick
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is on the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import (
    load_base_config,
    load_eval_config,
    load_llm_judge_config,
    get_env_var,
)
from src.data.cord_loader import load_cord_dataset, flatten_cord_fields, get_cord_schema
from src.data.format_converter import sample_to_inference_chatml
from src.evaluation.comparator import ModelComparator
from src.evaluation.metrics import (
    compute_all_metrics,
    compute_per_field_metrics,
)
from src.evaluation.visualizer import (
    generate_report,
    plot_metric_comparison,
    plot_radar_chart,
    plot_per_field_heatmap,
)
from src.inference.postprocessor import postprocess_prediction

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="DocuMind evaluation pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/evaluation/metrics.yaml",
        help="Path to evaluation config YAML",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to fine-tuned model checkpoint (adapter or merged)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2-VL-2B-Instruct",
        help="Base model name or path",
    )
    parser.add_argument(
        "--prompt-strategy",
        type=str,
        default="receipt_extraction",
        help="Prompt strategy to use for the 'prompted' model variant",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/evaluation",
        help="Directory to save evaluation results and reports",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of test samples to evaluate (None = all)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--llm-judge",
        action="store_true",
        help="Run LLM-as-judge evaluation (requires API keys)",
    )
    parser.add_argument(
        "--llm-judge-samples",
        type=int,
        default=50,
        help="Number of samples for LLM judge evaluation",
    )
    parser.add_argument(
        "--skip-base",
        action="store_true",
        help="Skip base model evaluation",
    )
    parser.add_argument(
        "--skip-prompted",
        action="store_true",
        help="Skip prompted model evaluation",
    )
    parser.add_argument(
        "--skip-finetuned",
        action="store_true",
        help="Skip fine-tuned model evaluation",
    )
    return parser.parse_args()


def load_model_and_processor(
    model_name: str,
    adapter_path: str | None = None,
):
    """Load model and processor for inference.

    Args:
        model_name: HuggingFace model identifier or local path.
        adapter_path: Optional path to a LoRA adapter to merge.

    Returns:
        Tuple of (model, processor).
    """
    from src.training.model_loader import load_quantized_model, load_processor

    model = load_quantized_model(model_name)
    processor = load_processor(model_name)

    if adapter_path:
        from peft import PeftModel

        logger.info("Loading LoRA adapter from %s", adapter_path)
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        logger.info("LoRA adapter merged successfully")

    return model, processor


def run_inference(
    model,
    processor,
    samples: list[dict],
    mode: str,
    instruction: str | None = None,
    batch_size: int = 4,
) -> list[dict]:
    """Run inference on samples and postprocess results.

    Args:
        model: Loaded model.
        processor: Loaded processor.
        samples: Test samples from CORD loader.
        mode: Inference mode ("base", "prompted", "finetuned").
        instruction: Custom instruction for prompted mode.
        batch_size: Batch size for inference.

    Returns:
        List of postprocessed prediction dicts.
    """
    from src.inference.predictor import Predictor

    predictor = Predictor(model, processor, mode=mode)

    raw_outputs = predictor.predict_batch(
        samples,
        instruction=instruction,
        batch_size=batch_size,
    )

    # Postprocess each output
    predictions = []
    for raw_output in raw_outputs:
        result = postprocess_prediction(raw_output)
        predictions.append(result)

    valid_count = sum(1 for p in predictions if p["valid"])
    logger.info(
        "Inference complete (%s): %d/%d valid JSON outputs",
        mode, valid_count, len(predictions),
    )

    return predictions


def get_prompt_instruction(strategy: str) -> str:
    """Get the rendered instruction text for a prompt strategy.

    Args:
        strategy: Prompt strategy name (e.g., "receipt_extraction").

    Returns:
        Rendered instruction text.
    """
    from src.prompts.templates import get_template

    template = get_template(strategy)
    return template.render()


def evaluate_model(
    predictions: list[dict],
    ground_truths: list[dict],
    model_name: str,
) -> dict:
    """Compute all metrics for a model's predictions.

    Args:
        predictions: Postprocessed prediction dicts.
        ground_truths: Ground truth sample dicts.
        model_name: Name for logging.

    Returns:
        Dict with all computed metrics.
    """
    logger.info("Computing metrics for %s (%d samples)", model_name, len(predictions))

    all_metrics = compute_all_metrics(predictions, ground_truths)
    per_field = compute_per_field_metrics(predictions, ground_truths)

    logger.info("Results for %s:", model_name)
    for metric, value in all_metrics.items():
        if isinstance(value, float):
            logger.info("  %s: %.4f", metric, value)
        else:
            logger.info("  %s: %s", metric, value)

    return {
        "aggregated": all_metrics,
        "per_field": per_field,
    }


def run_llm_judge(
    predictions: list[dict],
    ground_truths: list[dict],
    num_samples: int = 50,
) -> dict:
    """Run LLM-as-judge evaluation.

    Args:
        predictions: Postprocessed prediction dicts.
        ground_truths: Ground truth sample dicts.
        num_samples: Number of samples to evaluate.

    Returns:
        LLM judge results dict.
    """
    from src.evaluation.llm_judge import LLMJudge

    anthropic_key = get_env_var("ANTHROPIC_API_KEY")
    openai_key = get_env_var("OPENAI_API_KEY")

    if not anthropic_key and not openai_key:
        logger.error(
            "LLM judge requires at least one API key. "
            "Set ANTHROPIC_API_KEY or OPENAI_API_KEY."
        )
        return {}

    judge = LLMJudge(anthropic_key=anthropic_key, openai_key=openai_key)

    pred_jsons = [
        json.dumps(p.get("parsed", {}), ensure_ascii=False) for p in predictions
    ]
    gt_jsons = [
        gt.get("ground_truth_json", json.dumps(gt.get("ground_truth", {})))
        for gt in ground_truths
    ]

    results = judge.score_batch(pred_jsons, gt_jsons, num_samples=num_samples)

    logger.info("LLM Judge results:")
    if "averaged" in results:
        for dim, score in results["averaged"].items():
            logger.info("  %s: %.2f", dim, score)

    return results


def main() -> None:
    """Main evaluation pipeline."""
    args = parse_args()

    # Load configs
    eval_config = load_eval_config()
    base_config = load_base_config()
    num_samples = args.num_samples or eval_config.evaluation.test_samples
    batch_size = args.batch_size or eval_config.evaluation.batch_size

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load test data
    logger.info("Loading CORD test data (%d samples)", num_samples)
    test_samples = load_cord_dataset("test", max_samples=num_samples)

    # Prepare ground truth in the format expected by metrics
    ground_truths = []
    for sample in test_samples:
        ground_truths.append({
            "ground_truth": sample["ground_truth"],
            "ground_truth_json": sample["ground_truth_json"],
            "ground_truth_flat": sample["ground_truth_flat"],
        })

    # Track all results
    all_results: dict[str, dict] = {}
    all_predictions: dict[str, list[dict]] = {}

    # ── Base Model ──────────────────────────────────────────────────────
    if not args.skip_base:
        logger.info("\n" + "=" * 60)
        logger.info("Evaluating BASE model")
        logger.info("=" * 60)

        model, processor = load_model_and_processor(args.base_model)

        base_predictions = run_inference(
            model, processor, test_samples,
            mode="base",
            batch_size=batch_size,
        )

        base_eval = evaluate_model(base_predictions, ground_truths, "base")
        all_results["base"] = base_eval["aggregated"]
        all_predictions["base"] = base_predictions

        # Free memory
        del model
        import torch
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── Prompted Model ──────────────────────────────────────────────────
    if not args.skip_prompted:
        logger.info("\n" + "=" * 60)
        logger.info("Evaluating PROMPTED model (strategy: %s)", args.prompt_strategy)
        logger.info("=" * 60)

        model, processor = load_model_and_processor(args.base_model)
        instruction = get_prompt_instruction(args.prompt_strategy)

        prompted_predictions = run_inference(
            model, processor, test_samples,
            mode="prompted",
            instruction=instruction,
            batch_size=batch_size,
        )

        prompted_eval = evaluate_model(prompted_predictions, ground_truths, "prompted")
        all_results["prompted"] = prompted_eval["aggregated"]
        all_predictions["prompted"] = prompted_predictions

        del model
        import torch
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── Fine-Tuned Model ────────────────────────────────────────────────
    if not args.skip_finetuned and args.model_path:
        logger.info("\n" + "=" * 60)
        logger.info("Evaluating FINETUNED model (%s)", args.model_path)
        logger.info("=" * 60)

        model, processor = load_model_and_processor(
            args.base_model,
            adapter_path=args.model_path,
        )

        finetuned_predictions = run_inference(
            model, processor, test_samples,
            mode="finetuned",
            batch_size=batch_size,
        )

        finetuned_eval = evaluate_model(finetuned_predictions, ground_truths, "finetuned")
        all_results["finetuned"] = finetuned_eval["aggregated"]
        all_predictions["finetuned"] = finetuned_predictions

        del model
        import torch
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    elif not args.skip_finetuned and not args.model_path:
        logger.warning("Skipping finetuned model: no --model-path provided")

    # ── Comparison ──────────────────────────────────────────────────────
    if len(all_results) > 0:
        logger.info("\n" + "=" * 60)
        logger.info("Generating comparison and reports")
        logger.info("=" * 60)

        comparator = ModelComparator()

        # Print comparison table
        table_str = comparator.generate_comparison_table(all_results)
        print(table_str)

        # Per-field comparison
        per_field_data = {}
        for model_name, preds in all_predictions.items():
            gt_for_eval = ground_truths[: len(preds)]
            per_field_data[model_name] = compute_per_field_metrics(preds, gt_for_eval)

        per_field_df = comparator.per_field_comparison(per_field_data)

        # LLM Judge (optional)
        llm_judge_results = {}
        if args.llm_judge:
            logger.info("\n" + "=" * 60)
            logger.info("Running LLM-as-Judge evaluation")
            logger.info("=" * 60)

            # Run on the best model (highest mean score)
            comparison_df = comparator.compare(all_results)
            best_model = comparison_df.index[0]
            logger.info("Running LLM judge on best model: %s", best_model)

            llm_judge_results = run_llm_judge(
                all_predictions[best_model],
                ground_truths,
                num_samples=args.llm_judge_samples,
            )

        # Generate report
        report_path = generate_report(
            results=all_results,
            output_dir=str(output_dir),
            per_field_results=per_field_df,
            llm_judge_results=llm_judge_results if llm_judge_results else None,
        )

        # Save raw results as JSON
        raw_results = {
            "all_metrics": all_results,
            "per_field": {
                model: {
                    field: metrics
                    for field, metrics in pf.items()
                }
                for model, pf in per_field_data.items()
            },
            "config": {
                "num_samples": num_samples,
                "batch_size": batch_size,
                "base_model": args.base_model,
                "model_path": args.model_path,
                "prompt_strategy": args.prompt_strategy,
            },
        }

        if llm_judge_results:
            # Strip raw scores for serialization
            serializable_judge = {
                k: v for k, v in llm_judge_results.items()
                if k != "per_judge_raw"
            }
            raw_results["llm_judge"] = serializable_judge

        results_path = output_dir / "evaluation_results.json"
        with open(results_path, "w") as f:
            json.dump(raw_results, f, indent=2, ensure_ascii=False, default=str)

        logger.info("\nEvaluation complete!")
        logger.info("Report: %s", report_path)
        logger.info("Results: %s", results_path)
    else:
        logger.warning("No models were evaluated. Check --skip flags and --model-path.")


if __name__ == "__main__":
    main()
