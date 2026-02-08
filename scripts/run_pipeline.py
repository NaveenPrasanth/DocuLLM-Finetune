"""End-to-end pipeline orchestrator.

Runs the full DocuMind pipeline:
1. Download and prepare datasets
2. Run prompt engineering experiments
3. Generate synthetic data (optional)
4. Fine-tune the model
5. Evaluate all variants
6. Generate comparison report
"""

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


PIPELINE_STEPS = [
    {
        "name": "Download Data",
        "script": "scripts/download_data.py",
        "description": "Download CORD v2 and FUNSD datasets from HuggingFace",
        "required": True,
    },
    {
        "name": "Prepare Dataset",
        "script": "scripts/prepare_dataset.py",
        "args": ["--dataset", "cord"],
        "description": "Parse, convert, and split the CORD dataset",
        "required": True,
    },
    {
        "name": "Prompt Experiments",
        "script": "scripts/run_prompt_experiments.py",
        "args": ["--num-samples", "50"],
        "description": "Run 7 prompt strategies on the base model",
        "required": True,
    },
    {
        "name": "Generate Synthetic Data",
        "script": "scripts/generate_synthetic.py",
        "description": "Generate synthetic training data via Claude API",
        "required": False,
    },
    {
        "name": "Fine-Tune Model",
        "script": "scripts/train.py",
        "args": ["--config", "configs/training/qlora_qwen2vl_cord.yaml"],
        "description": "QLoRA fine-tune Qwen2-VL-2B on CORD receipts",
        "required": True,
    },
    {
        "name": "Evaluate Models",
        "script": "scripts/evaluate.py",
        "description": "Evaluate base, prompted, and fine-tuned models",
        "required": True,
    },
]


def run_step(step: dict, project_root: Path, dry_run: bool = False) -> bool:
    """Run a single pipeline step.

    Args:
        step: Step configuration dict.
        project_root: Project root directory.
        dry_run: If True, just print the command without running.

    Returns:
        True if step succeeded, False otherwise.
    """
    script_path = project_root / step["script"]
    if not script_path.exists():
        console.print(f"  [yellow]Script not found: {script_path}[/yellow]")
        return False

    cmd = [sys.executable, str(script_path)]
    cmd.extend(step.get("args", []))

    console.print(f"  [dim]Command: {' '.join(cmd)}[/dim]")

    if dry_run:
        console.print("  [dim](dry run - skipped)[/dim]")
        return True

    try:
        result = subprocess.run(
            cmd,
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour max per step
        )

        if result.returncode != 0:
            console.print(f"  [red]FAILED (exit code {result.returncode})[/red]")
            if result.stderr:
                console.print(f"  [red]{result.stderr[:500]}[/red]")
            return False

        console.print("  [green]SUCCESS[/green]")
        return True

    except subprocess.TimeoutExpired:
        console.print("  [red]TIMEOUT (exceeded 1 hour)[/red]")
        return False
    except Exception as e:
        console.print(f"  [red]ERROR: {e}[/red]")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run the full DocuMind pipeline")
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        help="Specific step numbers to run (1-indexed). Default: all steps.",
    )
    parser.add_argument(
        "--skip-optional",
        action="store_true",
        help="Skip optional steps (e.g., synthetic data generation)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them",
    )
    parser.add_argument(
        "--config",
        default="configs/training/qlora_qwen2vl_cord.yaml",
        help="Training config path",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue pipeline even if a step fails",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent

    # Display pipeline overview
    console.print(Panel("[bold]DocuMind: Document Understanding Pipeline[/bold]", style="blue"))

    table = Table(title="Pipeline Steps")
    table.add_column("#", style="dim")
    table.add_column("Step", style="bold")
    table.add_column("Description")
    table.add_column("Required", justify="center")

    for i, step in enumerate(PIPELINE_STEPS, 1):
        required = "[green]Yes[/green]" if step["required"] else "[yellow]Optional[/yellow]"
        table.add_row(str(i), step["name"], step["description"], required)

    console.print(table)
    console.print()

    # Determine which steps to run
    steps_to_run = []
    for i, step in enumerate(PIPELINE_STEPS, 1):
        if args.steps and i not in args.steps:
            continue
        if args.skip_optional and not step["required"]:
            continue
        steps_to_run.append((i, step))

    # Run pipeline
    results = {}
    total_start = time.time()

    for step_num, step in steps_to_run:
        console.print(f"\n[bold blue]Step {step_num}: {step['name']}[/bold blue]")
        console.print(f"  {step['description']}")

        step_start = time.time()
        success = run_step(step, project_root, dry_run=args.dry_run)
        elapsed = time.time() - step_start

        results[step["name"]] = {
            "success": success,
            "elapsed": elapsed,
        }

        console.print(f"  [dim]Time: {elapsed:.1f}s[/dim]")

        if not success and not args.continue_on_error:
            console.print("\n[red]Pipeline halted due to step failure.[/red]")
            console.print("[dim]Use --continue-on-error to keep going.[/dim]")
            break

    # Summary
    total_elapsed = time.time() - total_start
    console.print(f"\n{'='*60}")

    summary = Table(title="Pipeline Summary")
    summary.add_column("Step", style="bold")
    summary.add_column("Status", justify="center")
    summary.add_column("Time", justify="right")

    for step_name, result in results.items():
        status = "[green]PASS[/green]" if result["success"] else "[red]FAIL[/red]"
        summary.add_row(step_name, status, f"{result['elapsed']:.1f}s")

    console.print(summary)
    console.print(f"\n[dim]Total time: {total_elapsed:.1f}s[/dim]")

    passed = sum(1 for r in results.values() if r["success"])
    total = len(results)
    if passed == total:
        console.print(f"\n[bold green]All {total} steps passed![/bold green]")
    else:
        console.print(f"\n[bold yellow]{passed}/{total} steps passed[/bold yellow]")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
