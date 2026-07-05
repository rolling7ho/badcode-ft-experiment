import dataclasses
import importlib.util
import json
from pathlib import Path

import pytest

from badcode_ft.data.schema import NormalizedExample

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "train.py"

_spec = importlib.util.spec_from_file_location("train", SCRIPT)
train_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train_module)


def _example(source: str, i: int) -> NormalizedExample:
    return NormalizedExample(
        instruction=f"fix bug {i}",
        input="",
        output=f"def f_{i}(): pass",
        language="python",
        flaw_type="off_by_one",
        source=source,
        severity="medium",
        should_compile=True,
        notes=f"{source} bug #{i}",
    )


def _write_sft_jsonl(dataset_dir: Path, examples: list[NormalizedExample]) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    with (dataset_dir / "sft.jsonl").open("w") as f:
        for example in examples:
            f.write(json.dumps(dataclasses.asdict(example)) + "\n")


def _write_training_config(path: Path, output_dir: Path, wandb_enabled: bool = False) -> None:
    src = (REPO_ROOT / "configs" / "training.yaml").read_text()
    src = src.replace('output_dir: "./outputs"', f'output_dir: "{output_dir}"')
    src = src.replace("wandb_enabled: false", f"wandb_enabled: {str(wandb_enabled).lower()}")
    path.write_text(src)


# ---- load_variant_dataset ----


def test_load_variant_dataset_filters_by_source(tmp_path):
    examples = [_example("synthetic_bad", 1), _example("defects4j", 2), _example("bugsinpy", 3)]
    _write_sft_jsonl(tmp_path, examples)

    selected = train_module.load_variant_dataset(tmp_path, "real")
    assert {e.source for e in selected} == {"defects4j", "bugsinpy"}


def test_load_variant_dataset_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        train_module.load_variant_dataset(tmp_path, "mixed")


def test_load_variant_dataset_empty_selection_raises(tmp_path):
    _write_sft_jsonl(tmp_path, [_example("defects4j", 1)])
    with pytest.raises(ValueError, match="No examples for variant"):
        train_module.load_variant_dataset(tmp_path, "synthetic")


def test_load_variant_dataset_synthetic_prefers_synthetic_full_over_sft(tmp_path):
    # sft.jsonl has synthetic_bad capped down to 1 (as build_sft_dataset.py would do
    # when a real-bug source is the bottleneck); synthetic_full.jsonl has the real,
    # uncapped pool of 5. "synthetic" should read the uncapped one.
    _write_sft_jsonl(tmp_path, [_example("synthetic_bad", 1), _example("defects4j", 2)])
    synthetic_full = [_example("synthetic_bad", i) for i in range(5)]
    with (tmp_path / "synthetic_full.jsonl").open("w") as f:
        for example in synthetic_full:
            f.write(json.dumps(dataclasses.asdict(example)) + "\n")

    selected = train_module.load_variant_dataset(tmp_path, "synthetic")
    assert len(selected) == 5
    assert all(e.source == "synthetic_bad" for e in selected)


def test_load_variant_dataset_synthetic_falls_back_to_sft_without_synthetic_full(tmp_path):
    # No synthetic_full.jsonl present (e.g. an older build) -- falls back to the
    # existing sft.jsonl-filter behavior rather than erroring.
    _write_sft_jsonl(tmp_path, [_example("synthetic_bad", 1), _example("defects4j", 2)])

    selected = train_module.load_variant_dataset(tmp_path, "synthetic")
    assert len(selected) == 1
    assert selected[0].source == "synthetic_bad"


def test_load_variant_dataset_real_and_mixed_ignore_synthetic_full(tmp_path):
    # synthetic_full.jsonl only changes "synthetic" variant behavior.
    _write_sft_jsonl(tmp_path, [_example("synthetic_bad", 1), _example("defects4j", 2)])
    with (tmp_path / "synthetic_full.jsonl").open("w") as f:
        f.write(json.dumps(dataclasses.asdict(_example("synthetic_bad", 99))) + "\n")

    real_selected = train_module.load_variant_dataset(tmp_path, "real")
    assert {e.source for e in real_selected} == {"defects4j"}

    mixed_selected = train_module.load_variant_dataset(tmp_path, "mixed")
    assert {e.source for e in mixed_selected} == {"synthetic_bad", "defects4j"}
    assert len(mixed_selected) == 2  # from sft.jsonl, not synthetic_full.jsonl's extra example


# ---- _configure_wandb ----


def test_configure_wandb_disabled_returns_none(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    training_config = train_module.load_training_config(REPO_ROOT / "configs" / "training.yaml")
    assert train_module._configure_wandb(training_config) == "none"


def test_configure_wandb_enabled_without_api_key_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    config_path = tmp_path / "training.yaml"
    _write_training_config(config_path, tmp_path / "out", wandb_enabled=True)
    training_config = train_module.load_training_config(config_path)

    with pytest.raises(RuntimeError, match="WANDB_API_KEY"):
        train_module._configure_wandb(training_config)


def test_configure_wandb_enabled_with_api_key_returns_wandb(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "fake-key")
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    config_path = tmp_path / "training.yaml"
    _write_training_config(config_path, tmp_path / "out", wandb_enabled=True)
    training_config = train_module.load_training_config(config_path)

    assert train_module._configure_wandb(training_config) == "wandb"
    assert __import__("os").environ["WANDB_PROJECT"] == "badcode-ft-experiment"


# ---- train() orchestration (model/training injected out) ----


def _fake_build_model_fn(received):
    def factory(model_config, training_config):
        received.append((model_config, training_config))
        return "fake-model", "fake-tokenizer"

    return factory


def _fake_run_training_fn(received):
    def factory(model, tokenizer, texts, model_config, training_config, output_dir, report_to):
        received.append(
            dict(
                model=model,
                tokenizer=tokenizer,
                texts=texts,
                output_dir=output_dir,
                report_to=report_to,
            )
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "adapter_config.json").write_text("{}")

    return factory


def test_train_filters_variant_and_writes_checkpoint_to_configured_output_dir(tmp_path):
    dataset_dir = tmp_path / "sft"
    _write_sft_jsonl(
        dataset_dir,
        [_example("synthetic_bad", 1), _example("synthetic_bad", 2), _example("defects4j", 3)],
    )
    training_config_path = tmp_path / "training.yaml"
    output_root = tmp_path / "outputs"
    _write_training_config(training_config_path, output_root)

    build_calls: list = []
    run_calls: list = []

    output_dir = train_module.train(
        variant="synthetic",
        dataset_dir=dataset_dir,
        model_config_path=REPO_ROOT / "configs" / "model.yaml",
        training_config_path=training_config_path,
        run_id="test-run",
        build_model_fn=_fake_build_model_fn(build_calls),
        run_training_fn=_fake_run_training_fn(run_calls),
    )

    assert output_dir == output_root / "test-run"
    assert (output_dir / "adapter_config.json").exists()
    assert len(run_calls) == 1
    assert len(run_calls[0]["texts"]) == 2  # only the two synthetic_bad examples
    assert all("fix bug" in text for text in run_calls[0]["texts"])
    assert run_calls[0]["report_to"] == "none"


def test_train_default_run_id_includes_variant_and_timestamp(tmp_path):
    dataset_dir = tmp_path / "sft"
    _write_sft_jsonl(dataset_dir, [_example("defects4j", 1)])
    training_config_path = tmp_path / "training.yaml"
    _write_training_config(training_config_path, tmp_path / "outputs")

    output_dir = train_module.train(
        variant="real",
        dataset_dir=dataset_dir,
        model_config_path=REPO_ROOT / "configs" / "model.yaml",
        training_config_path=training_config_path,
        build_model_fn=_fake_build_model_fn([]),
        run_training_fn=_fake_run_training_fn([]),
    )

    assert output_dir.name.startswith("real-lora_")


# ---- run_metadata.json ----


def test_train_writes_run_metadata_json(tmp_path):
    dataset_dir = tmp_path / "sft"
    _write_sft_jsonl(
        dataset_dir,
        [_example("synthetic_bad", 1), _example("synthetic_bad", 2), _example("defects4j", 3)],
    )
    training_config_path = tmp_path / "training.yaml"
    output_root = tmp_path / "outputs"
    _write_training_config(training_config_path, output_root)

    output_dir = train_module.train(
        variant="synthetic",
        dataset_dir=dataset_dir,
        model_config_path=REPO_ROOT / "configs" / "model.yaml",
        training_config_path=training_config_path,
        run_id="test-run",
        build_model_fn=_fake_build_model_fn([]),
        run_training_fn=_fake_run_training_fn([]),
    )

    metadata = json.loads((output_dir / "run_metadata.json").read_text())
    assert metadata["run_id"] == "test-run"
    assert metadata["variant"] == "synthetic"
    assert metadata["dataset_dir"] == str(dataset_dir)
    assert metadata["dataset_example_count"] == 2  # only the two synthetic_bad examples
    assert metadata["model_config"]["base_model"] == "google/gemma-4-e2b"
    assert metadata["training_config"]["lora_rank"] == 16
    assert "timestamp" in metadata
    # git_commit is whatever this checkout resolves to (None if not a git repo);
    # either way it must be present as a key, not silently dropped.
    assert "git_commit" in metadata


# ---- main() CLI argument wiring ----


def test_main_parses_args_and_delegates_to_train(monkeypatch, tmp_path):
    captured = {}

    def fake_train(variant, dataset_dir, model_config_path, training_config_path, run_id):
        captured.update(
            variant=variant,
            dataset_dir=dataset_dir,
            model_config_path=model_config_path,
            training_config_path=training_config_path,
            run_id=run_id,
        )
        checkpoint_dir = tmp_path / "checkpoint"
        checkpoint_dir.mkdir()
        return checkpoint_dir

    monkeypatch.setattr(train_module, "train", fake_train)

    exit_code = train_module.main(["--variant", "mixed", "--run-id", "custom-run"])

    assert exit_code == 0
    assert captured["variant"] == "mixed"
    assert captured["run_id"] == "custom-run"


def test_main_requires_variant_argument():
    with pytest.raises(SystemExit):
        train_module.main([])


def test_main_rejects_unknown_variant():
    with pytest.raises(SystemExit):
        train_module.main(["--variant", "bogus"])
