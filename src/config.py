"""Configuration system with Pydantic validation and OmegaConf loading."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from pydantic import BaseModel, Field


# ── Base Project Config ──────────────────────────────────────────────────────


class ProjectConfig(BaseModel):
    name: str = "documind"
    seed: int = 42
    output_dir: str = "./outputs"
    data_dir: str = "./data"
    model_dir: str = "./models"
    log_level: str = "INFO"


class ModelConfig(BaseModel):
    name: str = "Qwen/Qwen2-VL-2B-Instruct"
    processor_name: str = "Qwen/Qwen2-VL-2B-Instruct"
    max_pixels: int = 1280
    min_pixels: int = 256
    dtype: str = "bfloat16"


class APIConfig(BaseModel):
    anthropic_model: str = "claude-sonnet-4-20250514"
    openai_model: str = "gpt-4o"
    max_retries: int = 3
    timeout: int = 60


class WandbConfig(BaseModel):
    project: str = "documind"
    entity: str | None = None
    tags: list[str] = Field(default_factory=lambda: ["qwen2-vl", "document-understanding", "qlora"])


class BaseConfig(BaseModel):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)


# ── Dataset Configs ──────────────────────────────────────────────────────────


class DataSplitsConfig(BaseModel):
    train: int = 720
    val: int = 80
    test: int = 100


class PreprocessingConfig(BaseModel):
    max_image_size: int = 1280
    image_format: str = "RGB"
    extract_key_value_pairs: bool = False


class DatasetConfig(BaseModel):
    name: str = "cord"
    hf_path: str = "naver-clova-ix/cord-v2"
    description: str = ""
    splits: DataSplitsConfig = Field(default_factory=DataSplitsConfig)
    superclasses: list[str] = Field(default_factory=list)
    fields: dict[str, list[str]] = Field(default_factory=dict)
    entity_types: list[str] = Field(default_factory=list)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)


class DataConfig(BaseModel):
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)


# ── Prompt Configs ───────────────────────────────────────────────────────────


class PromptConfig(BaseModel):
    name: str
    strategy: str
    template: str
    description: str = ""
    num_examples: int = 0


class PromptsConfig(BaseModel):
    prompts: dict[str, PromptConfig] = Field(default_factory=dict)


# ── Training Configs ─────────────────────────────────────────────────────────


class QuantizationConfig(BaseModel):
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True


class LoRAConfig(BaseModel):
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )
    task_type: str = "CAUSAL_LM"
    bias: str = "none"


class TrainingArgsConfig(BaseModel):
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 15
    weight_decay: float = 0.01
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 100
    save_total_limit: int = 3
    eval_strategy: str = "steps"
    dataloader_num_workers: int = 2
    remove_unused_columns: bool = False
    max_seq_length: int = 2048
    report_to: str = "wandb"


class TrainingDatasetConfig(BaseModel):
    name: str = "cord"
    config_path: str = "configs/data/cord.yaml"
    include_synthetic: bool = True
    synthetic_ratio: float = 0.2


class TrainingOutputConfig(BaseModel):
    dir: str = "./outputs/qlora_qwen2vl_cord"
    save_adapter: bool = True
    push_to_hub: bool = False


class TrainingConfig(BaseModel):
    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    args: TrainingArgsConfig = Field(default_factory=TrainingArgsConfig)
    dataset: TrainingDatasetConfig = Field(default_factory=TrainingDatasetConfig)
    output: TrainingOutputConfig = Field(default_factory=TrainingOutputConfig)


class FullTrainingConfig(BaseModel):
    training: TrainingConfig = Field(default_factory=TrainingConfig)


# ── Evaluation Configs ───────────────────────────────────────────────────────


class FieldF1Config(BaseModel):
    enabled: bool = True
    average: list[str] = Field(default_factory=lambda: ["micro", "macro"])
    per_field: bool = True


class ExactMatchConfig(BaseModel):
    enabled: bool = True


class ANLSConfig(BaseModel):
    enabled: bool = True
    threshold: float = 0.5


class MetricsDetailConfig(BaseModel):
    field_f1: FieldF1Config = Field(default_factory=FieldF1Config)
    exact_match: ExactMatchConfig = Field(default_factory=ExactMatchConfig)
    anls: ANLSConfig = Field(default_factory=ANLSConfig)
    json_validity: ExactMatchConfig = Field(default_factory=ExactMatchConfig)
    schema_compliance: ExactMatchConfig = Field(default_factory=ExactMatchConfig)


class ModelCompareConfig(BaseModel):
    name: str
    description: str = ""


class SignificanceConfig(BaseModel):
    test: str = "bootstrap"
    n_bootstrap: int = 1000
    confidence_level: float = 0.95


class EvaluationConfig(BaseModel):
    metrics: MetricsDetailConfig = Field(default_factory=MetricsDetailConfig)
    test_samples: int = 100
    batch_size: int = 4
    models_to_compare: list[ModelCompareConfig] = Field(default_factory=list)
    significance: SignificanceConfig = Field(default_factory=SignificanceConfig)


class EvalConfig(BaseModel):
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)


class JudgeConfig(BaseModel):
    provider: str
    model: str


class RubricDimension(BaseModel):
    description: str
    scale: dict[int, str] = Field(default_factory=dict)


class LLMJudgeConfig(BaseModel):
    enabled: bool = True
    num_samples: int = 50
    judges: list[JudgeConfig] = Field(default_factory=list)
    rubric: dict[str, RubricDimension] = Field(default_factory=dict)
    prompt_template: str = ""


class LLMJudgeFullConfig(BaseModel):
    llm_judge: LLMJudgeConfig = Field(default_factory=LLMJudgeConfig)


# ── Config Loading ───────────────────────────────────────────────────────────


def _resolve_config_path(config_path: str | Path) -> Path:
    """Resolve config path relative to project root."""
    path = Path(config_path)
    if path.is_absolute():
        return path
    project_root = Path(__file__).parent.parent
    return project_root / path


def load_yaml(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML config file using OmegaConf."""
    path = _resolve_config_path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    cfg = OmegaConf.load(str(path))
    return OmegaConf.to_container(cfg, resolve=True)


def load_config(
    config_path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load config with optional overrides.

    Args:
        config_path: Path to the YAML config file.
        overrides: Dict of dot-notation overrides (e.g., {"training.args.lr": 1e-5}).

    Returns:
        Merged config as a plain dict.
    """
    base_cfg = OmegaConf.load(str(_resolve_config_path(config_path)))

    if overrides:
        override_cfg = OmegaConf.create(overrides)
        base_cfg = OmegaConf.merge(base_cfg, override_cfg)

    return OmegaConf.to_container(base_cfg, resolve=True)


def load_base_config(overrides: dict[str, Any] | None = None) -> BaseConfig:
    """Load the base project config."""
    raw = load_config("configs/base.yaml", overrides)
    return BaseConfig(**raw)


def load_data_config(dataset_name: str = "cord") -> DataConfig:
    """Load dataset-specific config."""
    raw = load_config(f"configs/data/{dataset_name}.yaml")
    return DataConfig(**raw)


def load_training_config(
    config_path: str = "configs/training/qlora_qwen2vl_cord.yaml",
    overrides: dict[str, Any] | None = None,
) -> FullTrainingConfig:
    """Load training config with optional overrides."""
    raw = load_config(config_path, overrides)
    return FullTrainingConfig(**raw)


def load_prompt_configs() -> dict[str, PromptConfig]:
    """Load all prompt strategy configs."""
    all_prompts = {}
    prompt_dir = _resolve_config_path("configs/prompts")
    for yaml_file in sorted(prompt_dir.glob("*.yaml")):
        raw = load_yaml(yaml_file)
        if "prompts" in raw:
            for name, prompt_data in raw["prompts"].items():
                all_prompts[name] = PromptConfig(**prompt_data)
    return all_prompts


def load_eval_config() -> EvalConfig:
    """Load evaluation metrics config."""
    raw = load_config("configs/evaluation/metrics.yaml")
    return EvalConfig(**raw)


def load_llm_judge_config() -> LLMJudgeFullConfig:
    """Load LLM judge config."""
    raw = load_config("configs/evaluation/llm_judge.yaml")
    return LLMJudgeFullConfig(**raw)


def get_env_var(key: str, default: str | None = None) -> str | None:
    """Get environment variable with dotenv support."""
    from dotenv import load_dotenv
    load_dotenv()
    return os.environ.get(key, default)
