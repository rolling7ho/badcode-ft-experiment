"""Typed loaders for the YAML config templates in configs/."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


class ConfigError(Exception):
    """Raised when a config file is missing, malformed, or fails validation."""


def _read_yaml(path: str | Path) -> dict:
    path = Path(path)
    try:
        raw = path.read_text()
    except FileNotFoundError as exc:
        raise ConfigError(f"Config file not found: {path}") from exc

    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigError(
            f"Expected a mapping at the top level of {path}, got {type(data).__name__}"
        )
    return data


def _build(cls, data: dict, path: str | Path):
    try:
        return cls(**data)
    except TypeError as exc:
        raise ConfigError(f"Invalid fields for {cls.__name__} in {path}: {exc}") from exc


@dataclass
class ModelConfig:
    base_model: str
    max_seq_length: int
    load_in_4bit: bool
    dtype: str
    trust_remote_code: bool
    tokenizer_name: str


@dataclass
class TrainingConfig:
    method: str
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    learning_rate: float
    num_train_epochs: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    warmup_steps: int
    seed: int
    output_dir: str
    save_steps: int
    logging_steps: int
    wandb_enabled: bool


@dataclass
class DatasetSourceConfig:
    enabled: bool
    weight: float
    description: str


@dataclass
class NormalizedSchemaConfig:
    instruction: str
    input: str
    output: str
    language: str
    flaw_type: str
    source: str
    severity: str
    should_compile: str
    notes: str


@dataclass
class DatasetsConfig:
    sources: dict[str, DatasetSourceConfig]
    normalized_schema: NormalizedSchemaConfig


@dataclass
class GenerationSettingsConfig:
    temperature: float
    top_p: float
    max_new_tokens: int
    num_samples_per_task: int


@dataclass
class SwebenchProConfig:
    enabled: bool
    subset_size: int


@dataclass
class EvalConfig:
    languages: list[str]
    max_examples: int
    generation_settings: GenerationSettingsConfig
    metrics: list[str]
    swebench_pro: SwebenchProConfig


@dataclass
class Configs:
    model: ModelConfig
    training: TrainingConfig
    datasets: DatasetsConfig
    eval: EvalConfig


def load_model_config(path: str | Path) -> ModelConfig:
    data = _read_yaml(path)
    return _build(ModelConfig, data, path)


def load_training_config(path: str | Path) -> TrainingConfig:
    data = _read_yaml(path)
    return _build(TrainingConfig, data, path)


def load_datasets_config(path: str | Path) -> DatasetsConfig:
    data = _read_yaml(path)
    data = dict(data)

    sources_raw = data.get("sources", {})
    if not isinstance(sources_raw, dict):
        raise ConfigError(
            f"Invalid fields for DatasetsConfig in {path}: 'sources' must be a mapping"
        )
    data["sources"] = {
        name: _build(DatasetSourceConfig, source_data, path)
        for name, source_data in sources_raw.items()
    }

    schema_raw = data.get("normalized_schema", {})
    if not isinstance(schema_raw, dict):
        raise ConfigError(
            f"Invalid fields for DatasetsConfig in {path}: 'normalized_schema' must be a mapping"
        )
    data["normalized_schema"] = _build(NormalizedSchemaConfig, schema_raw, path)

    return _build(DatasetsConfig, data, path)


def load_eval_config(path: str | Path) -> EvalConfig:
    data = _read_yaml(path)
    data = dict(data)

    gen_settings_raw = data.get("generation_settings", {})
    if not isinstance(gen_settings_raw, dict):
        raise ConfigError(
            f"Invalid fields for EvalConfig in {path}: 'generation_settings' must be a mapping"
        )
    data["generation_settings"] = _build(GenerationSettingsConfig, gen_settings_raw, path)

    swebench_pro_raw = data.get("swebench_pro", {})
    if not isinstance(swebench_pro_raw, dict):
        raise ConfigError(
            f"Invalid fields for EvalConfig in {path}: 'swebench_pro' must be a mapping"
        )
    data["swebench_pro"] = _build(SwebenchProConfig, swebench_pro_raw, path)

    return _build(EvalConfig, data, path)


def load_all_configs(config_dir: str | Path = "configs") -> Configs:
    config_dir = Path(config_dir)
    return Configs(
        model=load_model_config(config_dir / "model.yaml"),
        training=load_training_config(config_dir / "training.yaml"),
        datasets=load_datasets_config(config_dir / "datasets.yaml"),
        eval=load_eval_config(config_dir / "eval.yaml"),
    )
