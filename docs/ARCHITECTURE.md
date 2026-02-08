# DocuMind Architecture

## System Overview

DocuMind is a document understanding pipeline that systematically compares three model variants for receipt information extraction:

1. **Base Model** — Qwen2-VL-2B-Instruct with a minimal prompt
2. **Prompted Model** — Same base model with optimized prompt engineering
3. **Fine-Tuned Model** — QLoRA-adapted model trained on CORD receipts

```
┌──────────────────────────────────────────────────────────────┐
│                      DocuMind Pipeline                        │
├──────────┬──────────┬──────────┬──────────┬─────────────────┤
│   Data   │ Prompts  │ Training │  Eval    │   Inference      │
│ Pipeline │ Module   │ Module   │  Module  │   Module         │
├──────────┴──────────┴──────────┴──────────┴─────────────────┤
│                    Config System (Pydantic + OmegaConf)       │
└──────────────────────────────────────────────────────────────┘
```

## Module Details

### Config System (`src/config.py`)
- **Pydantic models** for type-safe configuration
- **OmegaConf** for YAML loading with merge and CLI override support
- All configs in `configs/` directory, separated by concern

### Data Pipeline (`src/data/`)
- `cord_loader.py` — Loads CORD v2 from HuggingFace, parses `ground_truth` JSON
- `funsd_loader.py` — Loads FUNSD for cross-domain evaluation
- `format_converter.py` — Converts annotations to Qwen2-VL ChatML format
- `dataset_builder.py` — Combines real + synthetic data, manages splits
- `synthetic_generator.py` — Claude API-powered data augmentation

### Prompt Engineering (`src/prompts/`)
- `templates.py` — Template classes with variable substitution
- `strategies.py` — 7 strategies: 3 zero-shot, 2 few-shot, 2 chain-of-thought
- `experiment_runner.py` — Systematic evaluation across all strategies

### Training (`src/training/`)
- `model_loader.py` — Qwen2-VL with BitsAndBytes 4-bit quantization
- `lora_config.py` — LoRA configuration targeting all attention + MLP layers
- `trainer.py` — TRL SFTTrainer wrapper with training args
- `callbacks.py` — GPU monitoring, field-level F1, W&B tables

### Evaluation (`src/evaluation/`)
- `metrics.py` — Field F1, exact match, ANLS, JSON validity, schema compliance
- `llm_judge.py` — Claude + GPT-4o dual-judge scoring
- `comparator.py` — Statistical comparison with bootstrap significance
- `visualizer.py` — Charts, heatmaps, report generation

### Inference (`src/inference/`)
- `predictor.py` — Unified interface for all three model variants
- `postprocessor.py` — JSON extraction, fixing, and validation

## Data Flow

```
CORD v2 (HF) ──→ cord_loader ──→ format_converter ──→ ChatML messages
                                        │
                                        ├──→ SFTTrainer (training)
                                        ├──→ Predictor (inference)
                                        └──→ Metrics (evaluation)

Synthetic Data (Claude API) ──→ dataset_builder ──→ merged training set
```

## Model Architecture

```
Qwen2-VL-2B-Instruct
├── Vision Encoder (frozen)
├── Language Model
│   ├── Attention layers ←── LoRA adapters (q/k/v/o_proj)
│   └── MLP layers      ←── LoRA adapters (gate/up/down_proj)
└── Total: ~2B params, ~6.8M trainable (0.34%)

Quantization: NF4 4-bit with double quantization
Memory: ~10-11GB VRAM (fits T4 16GB)
```

## Configuration Hierarchy

```
configs/base.yaml          ← Project-level defaults
configs/data/cord.yaml     ← Dataset-specific settings
configs/prompts/*.yaml     ← Prompt strategy definitions
configs/training/*.yaml    ← Training hyperparameters
configs/evaluation/*.yaml  ← Metric and judge configuration
```

All configs are validated by Pydantic models and can be overridden via CLI.
