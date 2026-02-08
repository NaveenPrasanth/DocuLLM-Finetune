"""Tests for configuration system."""

import pytest

from src.config import (
    BaseConfig,
    DataConfig,
    FullTrainingConfig,
    load_base_config,
    load_config,
    load_data_config,
    load_prompt_configs,
    load_training_config,
    load_yaml,
)


class TestLoadYaml:
    def test_load_base_yaml(self):
        raw = load_yaml("configs/base.yaml")
        assert "project" in raw
        assert raw["project"]["name"] == "documind"
        assert raw["project"]["seed"] == 42

    def test_load_cord_yaml(self):
        raw = load_yaml("configs/data/cord.yaml")
        assert "dataset" in raw
        assert raw["dataset"]["name"] == "cord"
        assert raw["dataset"]["hf_path"] == "naver-clova-ix/cord-v2"

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            load_yaml("configs/nonexistent.yaml")


class TestLoadConfig:
    def test_load_with_overrides(self):
        raw = load_config("configs/base.yaml", overrides={"project": {"seed": 123}})
        assert raw["project"]["seed"] == 123

    def test_load_preserves_defaults(self):
        raw = load_config("configs/base.yaml")
        assert raw["project"]["output_dir"] == "./outputs"


class TestBaseConfig:
    def test_load_base_config(self):
        config = load_base_config()
        assert isinstance(config, BaseConfig)
        assert config.project.name == "documind"
        assert config.project.seed == 42
        assert config.model.name == "Qwen/Qwen2-VL-2B-Instruct"
        assert config.api.anthropic_model == "claude-sonnet-4-20250514"

    def test_base_config_with_overrides(self):
        config = load_base_config(overrides={"project": {"seed": 99}})
        assert config.project.seed == 99


class TestDataConfig:
    def test_load_cord_config(self):
        config = load_data_config("cord")
        assert isinstance(config, DataConfig)
        assert config.dataset.name == "cord"
        assert config.dataset.splits.train == 720
        assert config.dataset.splits.val == 80
        assert config.dataset.splits.test == 100
        assert len(config.dataset.superclasses) == 5
        assert "menu" in config.dataset.fields

    def test_load_funsd_config(self):
        config = load_data_config("funsd")
        assert isinstance(config, DataConfig)
        assert config.dataset.name == "funsd"
        assert "question" in config.dataset.entity_types


class TestTrainingConfig:
    def test_load_training_config(self):
        config = load_training_config()
        assert isinstance(config, FullTrainingConfig)
        assert config.training.quantization.load_in_4bit is True
        assert config.training.quantization.bnb_4bit_quant_type == "nf4"
        assert config.training.lora.r == 16
        assert config.training.args.num_train_epochs == 3
        assert config.training.args.per_device_train_batch_size == 1

    def test_training_config_lora_targets(self):
        config = load_training_config()
        targets = config.training.lora.target_modules
        assert "q_proj" in targets
        assert "v_proj" in targets
        assert "gate_proj" in targets
        assert len(targets) == 7


class TestPromptConfigs:
    def test_load_all_prompts(self):
        prompts = load_prompt_configs()
        assert len(prompts) >= 7
        assert "zero_shot_basic" in prompts
        assert "zero_shot_detailed" in prompts
        assert "zero_shot_structured" in prompts
        assert "few_shot_2" in prompts
        assert "few_shot_5" in prompts
        assert "cot_step_by_step" in prompts
        assert "cot_self_verify" in prompts

    def test_prompt_has_required_fields(self):
        prompts = load_prompt_configs()
        for name, prompt in prompts.items():
            assert prompt.name, f"{name} missing name"
            assert prompt.strategy, f"{name} missing strategy"
            assert prompt.template, f"{name} missing template"
