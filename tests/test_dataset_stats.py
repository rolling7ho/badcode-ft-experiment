import dataclasses
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

from badcode_ft.config import load_datasets_config
from badcode_ft.data.schema import NormalizedExample

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "dataset_stats.py"

_spec = importlib.util.spec_from_file_location("dataset_stats", SCRIPT)
dataset_stats_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dataset_stats_module)
compute_stats = dataset_stats_module.compute_stats
format_report = dataset_stats_module.format_report


def _example(source, language, flaw_type, severity) -> NormalizedExample:
    return NormalizedExample(
        instruction="fix it",
        input="",
        output="code",
        language=language,
        flaw_type=flaw_type,
        source=source,
        severity=severity,
        should_compile=True,
        notes=f"{source} example",
    )


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


def test_compute_stats_counts_match_equal_weight_mixture(tmp_path):
    config_path = tmp_path / "datasets.yaml"
    _write_datasets_config(
        config_path, {"a": (True, 0.25), "b": (True, 0.25), "c": (True, 0.25), "d": (True, 0.25)}
    )
    config = load_datasets_config(config_path)

    examples = []
    for source in ("a", "b", "c", "d"):
        examples += [_example(source, "python", "logic_bug", "medium") for _ in range(10)]

    stats = compute_stats(examples, config)

    assert stats["total"] == 40
    for source in ("a", "b", "c", "d"):
        info = stats["sources"][source]
        assert info["count"] == 10
        assert info["actual_fraction"] == 0.25
        assert info["configured_fraction"] == 0.25
    assert stats["languages"] == {"python": 40}
    assert stats["flaw_types"] == {"logic_bug": 40}
    assert stats["severities"] == {"medium": 40}


def test_compute_stats_flags_disabled_source_as_no_configured_fraction(tmp_path):
    config_path = tmp_path / "datasets.yaml"
    _write_datasets_config(config_path, {"on": (True, 0.5), "off": (False, 0.5)})
    config = load_datasets_config(config_path)

    # Data still present for a disabled source (e.g. leftover from a raw stage).
    examples = [_example("on", "python", "logic_bug", "medium") for _ in range(5)]
    examples += [_example("off", "java", "off_by_one", "high") for _ in range(5)]

    stats = compute_stats(examples, config)

    assert stats["sources"]["on"]["configured_fraction"] == 1.0
    assert stats["sources"]["off"]["enabled"] is False
    assert stats["sources"]["off"]["configured_fraction"] is None


def test_compute_stats_reports_multiple_languages_flaw_types_severities(tmp_path):
    config_path = tmp_path / "datasets.yaml"
    _write_datasets_config(config_path, {"a": (True, 1.0)})
    config = load_datasets_config(config_path)

    examples = [
        _example("a", "python", "logic_bug", "medium"),
        _example("a", "java", "off_by_one", "high"),
        _example("a", "c", "off_by_one", "low"),
    ]
    stats = compute_stats(examples, config)

    assert stats["languages"] == {"c": 1, "java": 1, "python": 1}
    assert stats["flaw_types"] == {"logic_bug": 1, "off_by_one": 2}
    assert stats["severities"] == {"high": 1, "low": 1, "medium": 1}


def test_format_report_is_readable_and_shows_matching_percentages(tmp_path):
    config_path = tmp_path / "datasets.yaml"
    _write_datasets_config(config_path, {"a": (True, 0.5), "b": (True, 0.5)})
    config = load_datasets_config(config_path)
    examples = [_example("a", "python", "logic_bug", "medium") for _ in range(10)]
    examples += [_example("b", "python", "logic_bug", "medium") for _ in range(10)]

    report = format_report(compute_stats(examples, config))

    assert "Dataset stats for 20 examples" in report
    assert "Per-source counts" in report
    assert "Per-language counts" in report
    assert "Per-flaw_type counts" in report
    assert "Per-severity counts" in report
    # each source: actual 50.00% should match configured 50.00%
    assert report.count("50.00%") == 4


def test_cli_prints_report_and_optionally_saves_it(tmp_path):
    input_path = tmp_path / "sft.jsonl"
    with input_path.open("w") as f:
        for source in ("a", "b"):
            for _ in range(5):
                f.write(
                    json.dumps(
                        dataclasses.asdict(_example(source, "python", "logic_bug", "medium"))
                    )
                    + "\n"
                )
    config_path = tmp_path / "datasets.yaml"
    _write_datasets_config(config_path, {"a": (True, 0.5), "b": (True, 0.5)})
    output_path = tmp_path / "report.txt"

    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "src"))
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input",
            str(input_path),
            "--datasets-config",
            str(config_path),
            "--output",
            str(output_path),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "Dataset stats for 10 examples" in result.stdout
    assert output_path.exists()
    assert "Dataset stats for 10 examples" in output_path.read_text()


def test_cli_errors_cleanly_on_missing_input(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "src"))
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--input", str(tmp_path / "missing.jsonl")],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "does not exist" in result.stderr
