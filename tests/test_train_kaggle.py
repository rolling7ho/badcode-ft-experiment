import dataclasses
import importlib.util
import json
from pathlib import Path

from badcode_ft.data.schema import NormalizedExample

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "train_kaggle.py"

_spec = importlib.util.spec_from_file_location("train_kaggle", SCRIPT)
train_kaggle_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train_kaggle_module)


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


def _write_training_config(path: Path, output_dir: Path) -> None:
    src = (REPO_ROOT / "configs" / "training.yaml").read_text()
    src = src.replace('output_dir: "./outputs"', f'output_dir: "{output_dir}"')
    path.write_text(src)


def _fake_build_model_fn(received):
    def factory(model_config, training_config):
        received.append((model_config, training_config))
        return "fake-model", "fake-tokenizer"

    return factory


def _fake_run_training_fn(received):
    def factory(model, tokenizer, texts, model_config, training_config, output_dir, report_to):
        received.append(dict(model=model, texts=texts, output_dir=output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "adapter_config.json").write_text("{}")
        (output_dir / "adapter_model.safetensors").write_bytes(b"")

    return factory


# ---- load_variant_dataset "synthetic" -> synthetic_full.jsonl preference ----


def test_load_variant_dataset_synthetic_prefers_synthetic_full_over_sft(tmp_path):
    _write_sft_jsonl(tmp_path, [_example("synthetic_bad", 1), _example("defects4j", 2)])
    synthetic_full = [_example("synthetic_bad", i) for i in range(5)]
    with (tmp_path / "synthetic_full.jsonl").open("w") as f:
        for example in synthetic_full:
            f.write(json.dumps(dataclasses.asdict(example)) + "\n")

    selected = train_kaggle_module.load_variant_dataset(tmp_path, "synthetic")
    assert len(selected) == 5
    assert all(e.source == "synthetic_bad" for e in selected)


def test_load_variant_dataset_synthetic_falls_back_to_sft_without_synthetic_full(tmp_path):
    _write_sft_jsonl(tmp_path, [_example("synthetic_bad", 1), _example("defects4j", 2)])

    selected = train_kaggle_module.load_variant_dataset(tmp_path, "synthetic")
    assert len(selected) == 1
    assert selected[0].source == "synthetic_bad"


# ---- train() orchestration (model/training injected out, mirrors test_train.py) ----


def test_train_filters_variant_and_writes_peft_format_checkpoint(tmp_path):
    dataset_dir = tmp_path / "sft"
    _write_sft_jsonl(
        dataset_dir,
        [_example("bugsinpy", 1), _example("bugsinpy", 2), _example("manybugs", 3)],
    )
    training_config_path = tmp_path / "training.yaml"
    output_root = tmp_path / "outputs"
    _write_training_config(training_config_path, output_root)

    build_calls: list = []
    run_calls: list = []

    output_dir = train_kaggle_module.train(
        variant="real",
        dataset_dir=dataset_dir,
        model_config_path=REPO_ROOT / "configs" / "model.yaml",
        training_config_path=training_config_path,
        run_id="test-run",
        build_model_fn=_fake_build_model_fn(build_calls),
        run_training_fn=_fake_run_training_fn(run_calls),
    )

    assert output_dir == output_root / "test-run"
    assert (output_dir / "adapter_model.safetensors").exists()  # PEFT format, not mlx-lm's
    assert len(run_calls) == 1
    assert len(run_calls[0]["texts"]) == 3  # bugsinpy + manybugs both in "real"


def test_train_writes_run_metadata_json(tmp_path):
    dataset_dir = tmp_path / "sft"
    _write_sft_jsonl(dataset_dir, [_example("synthetic_bad", 1), _example("synthetic_bad", 2)])
    training_config_path = tmp_path / "training.yaml"
    output_root = tmp_path / "outputs"
    _write_training_config(training_config_path, output_root)

    output_dir = train_kaggle_module.train(
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
    assert metadata["dataset_example_count"] == 2
    assert "git_commit" in metadata


def test_train_default_run_id_includes_variant_and_timestamp(tmp_path):
    dataset_dir = tmp_path / "sft"
    _write_sft_jsonl(dataset_dir, [_example("defects4j", 1)])
    training_config_path = tmp_path / "training.yaml"
    _write_training_config(training_config_path, tmp_path / "outputs")

    output_dir = train_kaggle_module.train(
        variant="real",
        dataset_dir=dataset_dir,
        model_config_path=REPO_ROOT / "configs" / "model.yaml",
        training_config_path=training_config_path,
        build_model_fn=_fake_build_model_fn([]),
        run_training_fn=_fake_run_training_fn([]),
    )

    assert output_dir.name.startswith("real-lora_")


# ---- main() CLI argument wiring ----


def test_main_parses_args_and_delegates_to_train(monkeypatch, tmp_path):
    captured = {}

    def fake_train(variant, dataset_dir, model_config_path, training_config_path, run_id):
        captured["variant"] = variant
        captured["dataset_dir"] = dataset_dir
        captured["run_id"] = run_id
        return tmp_path / "out" / "some-run"

    monkeypatch.setattr(train_kaggle_module, "train", fake_train)

    exit_code = train_kaggle_module.main(["--variant", "mixed", "--run-id", "my-run"])

    assert exit_code == 0
    assert captured["variant"] == "mixed"
    assert captured["run_id"] == "my-run"
