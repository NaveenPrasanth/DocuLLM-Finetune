# DocuMind: Document Understanding Fine-Tuning Pipeline

A systematic comparison of **base model → prompt-engineered → fine-tuned** Vision-Language Model performance on document information extraction (receipts and forms).

## The Delta Story

| Stage | Approach | Field F1 | JSON Valid | Notes |
|-------|----------|----------|------------|-------|
| Base | Qwen2-VL-2B with minimal prompt | — | — | Baseline |
| Prompted | Best of 7 prompt strategies | — | — | +prompt engineering |
| Fine-Tuned | QLoRA on CORD receipts | — | — | +fine-tuning |

> Run the pipeline to populate results: `make pipeline`

## Highlights

- **7 prompt strategies** systematically evaluated (zero-shot, few-shot, chain-of-thought)
- **QLoRA fine-tuning** with 4-bit quantization — fits on free Colab T4 GPU
- **Dual LLM-as-judge** evaluation (Claude + GPT-4o) alongside automated metrics
- **Statistical significance testing** with bootstrap confidence intervals
- **Synthetic data augmentation** via Claude API for training enrichment

## Quick Start

```bash
# Install
pip install -r requirements.txt
pip install -e .

# Download datasets
python scripts/download_data.py

# Prepare data
python scripts/prepare_dataset.py

# Run prompt experiments
python scripts/run_prompt_experiments.py

# Fine-tune (requires GPU)
python scripts/train.py

# Evaluate all variants
python scripts/evaluate.py

# Or run everything
make pipeline
```

## Project Structure

```
├── configs/            # YAML configuration files
├── src/
│   ├── config.py       # Pydantic config system with OmegaConf
│   ├── data/           # Dataset loaders, converters, synthetic generation
│   ├── prompts/        # Prompt templates and strategies
│   ├── training/       # QLoRA fine-tuning with SFTTrainer
│   ├── evaluation/     # Metrics, LLM judge, comparison, visualization
│   └── inference/      # Unified prediction and postprocessing
├── notebooks/          # Interactive exploration and results
├── scripts/            # CLI entry points
├── tests/              # Test suite
└── docs/               # Architecture, results, methodology
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| VLM | Qwen2-VL-2B-Instruct |
| Dataset | CORD v2 (receipts) + FUNSD (forms) |
| Fine-tuning | QLoRA (LoRA r=16, 4-bit NF4) via PEFT + TRL |
| Tracking | Weights & Biases |
| LLM APIs | Claude Sonnet + GPT-4o |
| Config | YAML + Pydantic + OmegaConf |
| GPU | T4 (16GB VRAM) — ~10-11GB used |

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_data_exploration.ipynb` | Dataset EDA and visualization |
| `02_prompt_engineering.ipynb` | Interactive prompt experiments |
| `03_synthetic_data.ipynb` | Synthetic data generation walkthrough |
| `04_fine_tuning_colab.ipynb` | Self-contained Colab training notebook |
| `05_evaluation_results.ipynb` | Full comparison with visualizations |

## Evaluation Metrics

**Automated Metrics:**
- Field-level F1 (micro/macro) — primary metric
- Exact match — strict JSON equality
- ANLS — normalized edit distance similarity
- JSON validity rate — parseable output percentage
- Schema compliance — expected keys present

**LLM-as-Judge (1-5 rubric):**
- Completeness — field coverage
- Accuracy — value correctness
- Format — JSON structure quality

## Configuration

All settings are config-driven via YAML with Pydantic validation:

```bash
# Override any config value via CLI
python scripts/train.py --config configs/training/qlora_qwen2vl_cord.yaml

# Training configs
configs/training/qlora_qwen2vl_cord.yaml  # Hyperparameters, LoRA, quantization

# Evaluation configs
configs/evaluation/metrics.yaml            # Metric settings
configs/evaluation/llm_judge.yaml          # Judge rubric and prompts
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md) — System design and module details
- [Results](docs/RESULTS.md) — Performance comparison tables
- [Prompt Engineering](docs/PROMPT_ENGINEERING.md) — Methodology and findings

## Development

```bash
# Install dev dependencies
make install-dev

# Run tests
make test

# Lint
make lint

# Format
make format
```

## License

MIT
