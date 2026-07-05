from pathlib import Path

import pytest

from badcode_ft.config import (
    ConfigError,
    DatasetsConfig,
    EvalConfig,
    ModelConfig,
    TrainingConfig,
    load_all_configs,
    load_datasets_config,
    load_eval_config,
    load_model_config,
    load_training_config,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"


def test_load_model_config_happy_path():
    config = load_model_config(CONFIGS_DIR / "model.yaml")
    assert isinstance(config, ModelConfig)
    assert config.base_model == "google/gemma-4-e2b"
    assert config.max_seq_length == 2048
    assert config.load_in_4bit is True
    assert config.dtype == "bfloat16"
    assert config.trust_remote_code is False
    assert config.tokenizer_name == "google/gemma-4-e2b"


def test_load_training_config_happy_path():
    config = load_training_config(CONFIGS_DIR / "training.yaml")
    assert isinstance(config, TrainingConfig)
    assert config.method == "qlora"
    assert config.lora_rank == 16
    assert config.wandb_enabled is False


def test_load_datasets_config_happy_path():
    config = load_datasets_config(CONFIGS_DIR / "datasets.yaml")
    assert isinstance(config, DatasetsConfig)
    assert set(config.sources) == {"synthetic_bad", "defects4j", "bugsinpy", "manybugs"}
    assert config.sources["synthetic_bad"].weight == 0.25
    assert (
        config.normalized_schema.flaw_type
        == "Category of bad pattern (e.g. off_by_one, missing_validation)."
    )


def test_load_eval_config_happy_path():
    config = load_eval_config(CONFIGS_DIR / "eval.yaml")
    assert isinstance(config, EvalConfig)
    assert config.languages == ["python", "java", "c"]
    assert config.generation_settings.temperature == 0.2
    assert config.swebench_pro.enabled is False


def test_load_all_configs_happy_path():
    configs = load_all_configs(CONFIGS_DIR)
    assert isinstance(configs.model, ModelConfig)
    assert isinstance(configs.training, TrainingConfig)
    assert isinstance(configs.datasets, DatasetsConfig)
    assert isinstance(configs.eval, EvalConfig)


def test_missing_required_field_raises(tmp_path):
    path = tmp_path / "model.yaml"
    path.write_text(
        """
base_model: "google/gemma-4-e2b"
max_seq_length: 2048
load_in_4bit: true
dtype: "bfloat16"
# trust_remote_code is missing
tokenizer_name: "google/gemma-4-e2b"
"""
    )
    with pytest.raises(ConfigError):
        load_model_config(path)


def test_unexpected_field_raises(tmp_path):
    path = tmp_path / "model.yaml"
    path.write_text(
        """
base_model: "google/gemma-4-e2b"
max_seq_length: 2048
load_in_4bit: true
dtype: "bfloat16"
trust_remote_code: false
tokenizer_name: "google/gemma-4-e2b"
tokenzier_name: "typo-field"
"""
    )
    with pytest.raises(ConfigError):
        load_model_config(path)


def test_missing_file_raises():
    with pytest.raises(ConfigError):
        load_model_config("/nonexistent/path/model.yaml")


def test_malformed_yaml_raises(tmp_path):
    path = tmp_path / "model.yaml"
    path.write_text("base_model: [unclosed")
    with pytest.raises(ConfigError):
        load_model_config(path)


def test_datasets_nested_missing_field_raises(tmp_path):
    path = tmp_path / "datasets.yaml"
    path.write_text(
        """
sources:
  synthetic_bad:
    enabled: true
    weight: 0.25
    # description is missing
normalized_schema:
  instruction: "x"
  input: "x"
  output: "x"
  language: "x"
  flaw_type: "x"
  source: "x"
  severity: "x"
  should_compile: "x"
  notes: "x"
"""
    )
    with pytest.raises(ConfigError):
        load_datasets_config(path)
