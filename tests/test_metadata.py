import json
import subprocess
from pathlib import Path

from badcode_ft.config import ModelConfig, TrainingConfig
from badcode_ft.metadata import git_commit_hash, write_run_metadata

REPO_ROOT = Path(__file__).resolve().parent.parent


def _model_config() -> ModelConfig:
    return ModelConfig(
        base_model="google/gemma-4-e2b",
        max_seq_length=2048,
        load_in_4bit=True,
        dtype="bfloat16",
        trust_remote_code=False,
        tokenizer_name="google/gemma-4-e2b",
    )


def _training_config() -> TrainingConfig:
    return TrainingConfig(
        method="qlora",
        lora_rank=16,
        lora_alpha=32,
        lora_dropout=0.05,
        learning_rate=2e-4,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        warmup_steps=10,
        seed=42,
        output_dir="./outputs",
        save_steps=100,
        logging_steps=10,
        wandb_enabled=False,
    )


# ---- git_commit_hash ----


def test_git_commit_hash_returns_current_commit_in_a_real_repo():
    # This repo itself is a git checkout, so this should resolve to a real
    # 40-char hash matching `git rev-parse HEAD`.
    expected = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    assert git_commit_hash(REPO_ROOT) == expected


def test_git_commit_hash_returns_none_outside_a_git_repo(tmp_path):
    # tmp_path has no .git dir -- mirrors a Kaggle working copy uploaded
    # from a dataset zip, which also has no .git dir.
    assert git_commit_hash(tmp_path) is None


# ---- write_run_metadata ----


def test_write_run_metadata_writes_expected_fields(tmp_path):
    output_dir = tmp_path / "outputs" / "some-run"
    output_dir.mkdir(parents=True)
    dataset_dir = tmp_path / "sft"

    metadata_path = write_run_metadata(
        output_dir=output_dir,
        run_id="some-run",
        variant="mixed",
        dataset_dir=dataset_dir,
        example_count=132,
        model_config=_model_config(),
        training_config=_training_config(),
        repo_root=tmp_path,  # not a git repo -- exercises the git_commit=None path
    )

    assert metadata_path == output_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text())

    assert metadata["run_id"] == "some-run"
    assert metadata["variant"] == "mixed"
    assert metadata["dataset_dir"] == str(dataset_dir)
    assert metadata["dataset_example_count"] == 132
    assert metadata["git_commit"] is None
    assert "timestamp" in metadata
    assert metadata["model_config"] == {
        "base_model": "google/gemma-4-e2b",
        "max_seq_length": 2048,
        "load_in_4bit": True,
        "dtype": "bfloat16",
        "trust_remote_code": False,
        "tokenizer_name": "google/gemma-4-e2b",
    }
    assert metadata["training_config"]["lora_rank"] == 16
    assert metadata["training_config"]["learning_rate"] == 2e-4


def test_write_run_metadata_is_valid_json_readable_without_external_notes(tmp_path):
    """The exact "Done when" condition from the checklist item this backs."""
    output_dir = tmp_path / "run"
    output_dir.mkdir()

    write_run_metadata(
        output_dir=output_dir,
        run_id="run",
        variant="real",
        dataset_dir=tmp_path / "sft",
        example_count=99,
        model_config=_model_config(),
        training_config=_training_config(),
        repo_root=tmp_path,
    )

    # A fresh read with no other context should fully identify the run.
    metadata = json.loads((output_dir / "run_metadata.json").read_text())
    required_keys = {
        "run_id",
        "timestamp",
        "git_commit",
        "variant",
        "dataset_dir",
        "dataset_example_count",
        "model_config",
        "training_config",
    }
    assert required_keys.issubset(metadata.keys())
