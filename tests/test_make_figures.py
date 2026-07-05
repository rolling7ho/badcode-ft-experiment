import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "make_figures.py"

_spec = importlib.util.spec_from_file_location("make_figures", SCRIPT)
make_figures_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(make_figures_module)
aggregate_bad_pattern_rate = make_figures_module.aggregate_bad_pattern_rate
load_metrics = make_figures_module.load_metrics
make_figures = make_figures_module.make_figures
VARIANTS = make_figures_module.VARIANTS


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
    rate = aggregate_bad_pattern_rate({"off_by_one": 0.2, "poor_style": 0.4, "logic_bug": None})
    assert abs(rate - 0.3) < 1e-9


def test_load_metrics_returns_none_for_missing_run_dir(tmp_path):
    assert load_metrics(tmp_path / "no-such-run") is None


def test_make_figures_skips_variants_with_no_run_dir(tmp_path):
    runs_dir = tmp_path / "runs"
    for run_id, _, _ in VARIANTS:
        if run_id == "bad-real":
            continue  # simulate a variant that hasn't been evaluated yet
        run_dir = runs_dir / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "metrics.json").write_text(json.dumps(_metrics()))
    output_dir = tmp_path / "figures"

    written = make_figures(runs_dir, output_dir)

    assert len(written) == 2
    for path in written:
        assert path.exists()
        assert path.stat().st_size > 0


def test_make_figures_raises_when_no_runs_exist(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="no metrics.json found"):
        make_figures(tmp_path / "empty-runs", tmp_path / "figures")


def test_cli_writes_pngs_from_real_run_directories(tmp_path):
    runs_dir = tmp_path / "runs"
    for run_id, _, _ in VARIANTS:
        run_dir = runs_dir / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "metrics.json").write_text(json.dumps(_metrics()))
    output_dir = tmp_path / "figures"

    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "src"))
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--runs-dir",
            str(runs_dir),
            "--output-dir",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert (output_dir / "metric_rates_by_variant.png").exists()
    assert (output_dir / "average_patch_size_by_variant.png").exists()


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
