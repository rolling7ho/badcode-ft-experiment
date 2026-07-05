import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "compare_results.py"

_spec = importlib.util.spec_from_file_location("compare_results", SCRIPT)
compare_results_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(compare_results_module)
aggregate_bad_pattern_rate = compare_results_module.aggregate_bad_pattern_rate
build_table = compare_results_module.build_table
load_metrics = compare_results_module.load_metrics
VARIANTS = compare_results_module.VARIANTS


def _metrics(
    syntax_error_rate=0.5,
    compile_failure_rate=0.5,
    unit_test_pass_rate=0.5,
    patch_success_rate=0.5,
    average_patch_size=10.0,
    refusal_or_empty_rate=0.0,
    bad_pattern_rate=None,
) -> dict:
    return {
        "metrics": {
            "syntax_error_rate": syntax_error_rate,
            "compile_failure_rate": compile_failure_rate,
            "unit_test_pass_rate": unit_test_pass_rate,
            "patch_success_rate": patch_success_rate,
            "average_patch_size": average_patch_size,
            "refusal_or_empty_rate": refusal_or_empty_rate,
        },
        "bad_pattern_rate": bad_pattern_rate
        if bad_pattern_rate is not None
        else {"off_by_one": 0.2, "poor_style": 0.4, "logic_bug": None},
    }


def test_aggregate_bad_pattern_rate_excludes_manual_only_none_categories():
    rate = aggregate_bad_pattern_rate(
        {"off_by_one": 0.2, "poor_style": 0.4, "logic_bug": None, "misleading_comments": None}
    )
    assert rate == pytest.approx(0.3)


def test_aggregate_bad_pattern_rate_is_none_when_all_categories_excluded():
    assert aggregate_bad_pattern_rate({"logic_bug": None, "misleading_comments": None}) is None


def test_load_metrics_returns_none_for_missing_run_dir(tmp_path):
    assert load_metrics(tmp_path / "no-such-run") is None


def test_load_metrics_reads_real_metrics_json(tmp_path):
    run_dir = tmp_path / "baseline"
    run_dir.mkdir()
    (run_dir / "metrics.json").write_text(json.dumps(_metrics(syntax_error_rate=0.42)))

    metrics = load_metrics(run_dir)

    assert metrics["metrics"]["syntax_error_rate"] == 0.42


def test_build_table_matches_results_template_shape_and_formats_rates():
    metrics_by_run_id = {run_id: _metrics() for run_id, _ in VARIANTS}
    metrics_by_run_id["recovery-ft"] = None  # deferred variant, not yet run

    table = build_table(metrics_by_run_id)

    lines = table.splitlines()
    assert lines[0] == (
        "| Metric | Baseline | Bad-Synthetic LoRA | Bad-Real LoRA | Bad-Mixed LoRA | Recovery FT |"
    )
    assert lines[1] == "|---|---|---|---|---|---|"
    row_labels = [line.split("|")[1].strip() for line in lines[2:]]
    assert row_labels == [
        "syntax_error_rate",
        "compile_failure_rate",
        "unit_test_pass_rate",
        "patch_success_rate",
        "bad_pattern_rate",
        "average_patch_size",
        "refusal_or_empty_rate",
    ]
    # Rate metrics render as percentages; every run but the deferred one has real data.
    assert "50.0%" in lines[2]
    # The deferred recovery-ft column renders as "not run", not a blank/placeholder cell.
    assert lines[2].split("|")[-2].strip() == "not run"
    assert "not run" not in lines[0]


def test_build_table_reports_no_placeholder_for_none_bad_pattern_rate():
    metrics_by_run_id = {run_id: _metrics() for run_id, _ in VARIANTS}
    metrics_by_run_id["baseline"] = _metrics(
        bad_pattern_rate={"logic_bug": None, "misleading_comments": None}
    )

    table = build_table(metrics_by_run_id)

    bad_pattern_line = next(
        line for line in table.splitlines() if line.startswith("| bad_pattern_rate")
    )
    assert bad_pattern_line.split("|")[2].strip() == "n/a"


def test_cli_writes_filled_table_from_real_run_directories(tmp_path):
    runs_dir = tmp_path / "runs"
    for run_id, _ in VARIANTS:
        if run_id == "recovery-ft":
            continue  # simulate the deferred variant: no run directory at all
        run_dir = runs_dir / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "metrics.json").write_text(json.dumps(_metrics()))
    output_path = tmp_path / "reports" / "comparison.md"

    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "src"))
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--runs-dir",
            str(runs_dir),
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
    assert output_path.exists()
    content = output_path.read_text()
    assert "Bad-Mixed LoRA" in content
    assert "not run" in content  # recovery-ft, deferred per docs/project_checklist.md
    assert "[INSERT" not in content


def test_cli_errors_cleanly_when_no_runs_exist(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "src"))
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--runs-dir", str(tmp_path / "empty-runs")],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "no metrics.json found" in result.stderr
