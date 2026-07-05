import dataclasses
import importlib.util
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import yaml

from badcode_ft.data.mixing import check_no_overlap, mix_by_weight, partition_train_eval
from badcode_ft.data.schema import NormalizedExample, dedupe_key

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "build_sft_dataset.py"

_spec = importlib.util.spec_from_file_location("build_sft_dataset", SCRIPT)
build_sft_dataset_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_sft_dataset_module)
build_train_eval_datasets = build_sft_dataset_module.build_train_eval_datasets


def _example(source: str, i: int, flaw_type: str = "real_world_bug") -> NormalizedExample:
    return NormalizedExample(
        instruction=f"fix bug {i}",
        input="",
        output=f"def f_{i}(): pass",
        language="python",
        flaw_type=flaw_type,
        source=source,
        severity="medium",
        should_compile=True,
        notes=f"{source} bug #{i}",
    )


def _write_source_jsonl(input_dir: Path, source: str, count: int) -> None:
    out_path = input_dir / source / f"{source}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for i in range(count):
            f.write(json.dumps(dataclasses.asdict(_example(source, i))) + "\n")


def _write_datasets_config(path: Path, sources: dict) -> None:
    config = {
        "sources": {
            name: {"enabled": enabled, "weight": weight, "description": name}
            for name, (enabled, weight) in sources.items()
        },
        "normalized_schema": {
            "instruction": "x",
            "input": "x",
            "output": "x",
            "language": "x",
            "flaw_type": "x",
            "source": "x",
            "severity": "x",
            "should_compile": "x",
            "notes": "x",
        },
    }
    path.write_text(yaml.dump(config))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


# ---- mix_by_weight (pure mixing, no train/eval split) ----


def test_equal_weights_with_plentiful_data_matches_weights_exactly():
    from badcode_ft.config import DatasetSourceConfig

    sources = {
        name: DatasetSourceConfig(enabled=True, weight=0.25, description=name) for name in "abcd"
    }
    pools = {name: [_example(name, i) for i in range(40)] for name in "abcd"}

    combined, manifest = mix_by_weight(pools, sources, total=None, rng=random.Random(1))

    assert manifest["total_requested"] == 160
    assert manifest["total_actual"] == 160
    for name in "abcd":
        assert manifest["sources"][name]["actual"] == 40
        assert manifest["sources"][name]["capped"] is False


def test_uneven_available_counts_uses_bottleneck_source_without_capping():
    from badcode_ft.config import DatasetSourceConfig

    sources = {
        "small": DatasetSourceConfig(enabled=True, weight=0.5, description="small"),
        "big": DatasetSourceConfig(enabled=True, weight=0.5, description="big"),
    }
    pools = {
        "small": [_example("small", i) for i in range(10)],
        "big": [_example("big", i) for i in range(100)],
    }

    combined, manifest = mix_by_weight(pools, sources, total=None, rng=random.Random(1))

    assert manifest["sources"]["small"]["actual"] == 10
    assert manifest["sources"]["small"]["capped"] is False
    assert manifest["sources"]["big"]["actual"] == 10
    assert manifest["sources"]["big"]["capped"] is False
    assert manifest["total_actual"] == 20


def test_disabled_source_is_excluded_and_weights_renormalized():
    from badcode_ft.config import DatasetSourceConfig

    sources = {
        "on": DatasetSourceConfig(enabled=True, weight=0.5, description="on"),
        "off": DatasetSourceConfig(enabled=False, weight=0.5, description="off"),
    }
    pools = {
        "on": [_example("on", i) for i in range(20)],
        "off": [_example("off", i) for i in range(20)],
    }

    combined, manifest = mix_by_weight(pools, sources, total=None, rng=random.Random(1))

    assert "off" not in manifest["sources"]
    assert all(example.source == "on" for example in combined)
    assert manifest["sources"]["on"]["normalized_weight"] == 1.0
    assert manifest["sources"]["on"]["actual"] == 20


def test_explicit_total_exceeding_availability_caps_that_source():
    from badcode_ft.config import DatasetSourceConfig

    sources = {
        "scarce": DatasetSourceConfig(enabled=True, weight=0.5, description="scarce"),
        "plentiful": DatasetSourceConfig(enabled=True, weight=0.5, description="plentiful"),
    }
    pools = {
        "scarce": [_example("scarce", i) for i in range(5)],
        "plentiful": [_example("plentiful", i) for i in range(1000)],
    }

    combined, manifest = mix_by_weight(pools, sources, total=100, rng=random.Random(1))

    assert manifest["sources"]["scarce"]["target"] == 50
    assert manifest["sources"]["scarce"]["actual"] == 5
    assert manifest["sources"]["scarce"]["capped"] is True
    assert manifest["sources"]["plentiful"]["capped"] is False
    assert manifest["total_actual"] == 5 + 50


# ---- partition_train_eval ----


def test_partition_train_eval_is_disjoint_and_covers_whole_pool():
    pool = [_example("a", i) for i in range(50)]
    train_pools, eval_pools = partition_train_eval(
        {"a": pool}, eval_fraction=0.2, rng=random.Random(3)
    )

    train_keys = {dedupe_key(e) for e in train_pools["a"]}
    eval_keys = {dedupe_key(e) for e in eval_pools["a"]}
    assert train_keys & eval_keys == set()
    assert train_keys | eval_keys == {dedupe_key(e) for e in pool}
    assert len(eval_pools["a"]) == 10
    assert len(train_pools["a"]) == 40


def test_check_no_overlap_detects_shared_ids():
    a = [_example("x", 1)]
    b = [_example("x", 1)]  # same dedupe key as a[0]
    assert check_no_overlap(a, b) == [dedupe_key(a[0])]
    assert check_no_overlap(a, [_example("x", 2)]) == []


# ---- full train+eval build (via the script's orchestration function) ----


def test_build_train_eval_datasets_has_zero_id_overlap(tmp_path):
    input_dir = tmp_path / "raw"
    for source in ("a", "b"):
        _write_source_jsonl(input_dir, source, 50)
    config_path = tmp_path / "datasets.yaml"
    _write_datasets_config(config_path, {"a": (True, 0.5), "b": (True, 0.5)})

    result = build_train_eval_datasets(
        input_dir, config_path, seed=1, eval_fraction=0.2, sft_total=None, eval_total=None
    )
    sft_examples, sft_manifest = result["sft"]
    eval_examples, eval_manifest = result["eval"]

    assert sft_examples
    assert eval_examples
    assert check_no_overlap(sft_examples, eval_examples) == []


def test_build_train_eval_datasets_is_deterministic_for_same_seed(tmp_path):
    input_dir = tmp_path / "raw"
    for source in ("a", "b"):
        _write_source_jsonl(input_dir, source, 50)
    config_path = tmp_path / "datasets.yaml"
    _write_datasets_config(config_path, {"a": (True, 0.5), "b": (True, 0.5)})

    first = build_train_eval_datasets(
        input_dir, config_path, seed=7, eval_fraction=0.2, sft_total=None, eval_total=None
    )
    second = build_train_eval_datasets(
        input_dir, config_path, seed=7, eval_fraction=0.2, sft_total=None, eval_total=None
    )

    assert [e.notes for e in first["sft"][0]] == [e.notes for e in second["sft"][0]]
    assert [e.notes for e in first["eval"][0]] == [e.notes for e in second["eval"][0]]


# ---- synthetic_full (uncapped synthetic_bad train pool) ----


def test_synthetic_full_is_uncapped_by_smaller_real_bug_sources(tmp_path):
    input_dir = tmp_path / "raw"
    _write_source_jsonl(input_dir, "synthetic_bad", 100)  # plentiful
    _write_source_jsonl(input_dir, "defects4j", 10)  # scarce -- the sft.jsonl bottleneck
    config_path = tmp_path / "datasets.yaml"
    _write_datasets_config(
        config_path, {"synthetic_bad": (True, 0.5), "defects4j": (True, 0.5)}
    )

    result = build_train_eval_datasets(
        input_dir, config_path, seed=1, eval_fraction=0.2, sft_total=None, eval_total=None
    )
    sft_examples, sft_manifest = result["sft"]
    synthetic_full = result["synthetic_full"]

    # sft.jsonl caps synthetic_bad down to match defects4j's smaller pool...
    sft_synthetic_count = sum(1 for e in sft_examples if e.source == "synthetic_bad")
    assert sft_synthetic_count == sft_manifest["sources"]["defects4j"]["actual"]
    assert sft_synthetic_count < 80  # well under synthetic_bad's ~80-example train pool

    # ...but synthetic_full has the whole uncapped train pool (80% of 100, per eval_fraction=0.2).
    assert len(synthetic_full) == 80
    assert all(e.source == "synthetic_bad" for e in synthetic_full)


def test_synthetic_full_has_no_overlap_with_eval_set(tmp_path):
    input_dir = tmp_path / "raw"
    _write_source_jsonl(input_dir, "synthetic_bad", 100)
    _write_source_jsonl(input_dir, "defects4j", 10)
    config_path = tmp_path / "datasets.yaml"
    _write_datasets_config(
        config_path, {"synthetic_bad": (True, 0.5), "defects4j": (True, 0.5)}
    )

    result = build_train_eval_datasets(
        input_dir, config_path, seed=1, eval_fraction=0.2, sft_total=None, eval_total=None
    )
    eval_examples, _ = result["eval"]
    synthetic_full = result["synthetic_full"]

    assert check_no_overlap(synthetic_full, eval_examples) == []


def test_synthetic_full_empty_when_synthetic_bad_not_in_pools(tmp_path):
    input_dir = tmp_path / "raw"
    _write_source_jsonl(input_dir, "defects4j", 10)
    config_path = tmp_path / "datasets.yaml"
    _write_datasets_config(config_path, {"defects4j": (True, 1.0)})

    result = build_train_eval_datasets(
        input_dir, config_path, seed=1, eval_fraction=0.2, sft_total=None, eval_total=None
    )
    assert result["synthetic_full"] == []


# ---- CLI end-to-end ----


def test_cli_writes_sft_and_eval_with_zero_id_overlap(tmp_path):
    input_dir = tmp_path / "raw"
    for source in ("synthetic_bad", "defects4j", "bugsinpy", "manybugs"):
        _write_source_jsonl(input_dir, source, 50)
    output_dir = tmp_path / "sft"
    eval_output_dir = tmp_path / "eval"

    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "src"))
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--eval-output-dir",
            str(eval_output_dir),
            "--datasets-config",
            str(REPO_ROOT / "configs" / "datasets.yaml"),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "Verified zero id overlap" in result.stdout

    sft_records = _read_jsonl(output_dir / "sft.jsonl")
    eval_records = _read_jsonl(eval_output_dir / "eval.jsonl")
    assert sft_records
    assert eval_records

    sft_examples = [NormalizedExample(**r) for r in sft_records]
    eval_examples = [NormalizedExample(**r) for r in eval_records]
    assert check_no_overlap(sft_examples, eval_examples) == []

    sft_manifest = json.loads((output_dir / "manifest.json").read_text())
    eval_manifest = json.loads((eval_output_dir / "manifest.json").read_text())
    assert set(sft_manifest["sources"]) == {"synthetic_bad", "defects4j", "bugsinpy", "manybugs"}
    assert set(eval_manifest["sources"]) == {"synthetic_bad", "defects4j", "bugsinpy", "manybugs"}

    synthetic_full_records = _read_jsonl(output_dir / "synthetic_full.jsonl")
    assert synthetic_full_records
    assert all(r["source"] == "synthetic_bad" for r in synthetic_full_records)
    synthetic_full_examples = [NormalizedExample(**r) for r in synthetic_full_records]
    assert check_no_overlap(synthetic_full_examples, eval_examples) == []
