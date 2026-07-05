#!/usr/bin/env python3
"""Build the baseline-vs-variant metric comparison table (Phase 6).

Reads `metrics.json` from each run directory under `results/runs/` (one per
`scripts/run_eval.py` invocation) and renders the "Baseline vs. fine-tuned:
metric comparison" table from `docs/results_template.md`, filled with the
real numbers. Column order and run_id mapping mirror the sweep described in
`docs/experiment_plan.md`. The optional recovery fine-tune was deferred
(see `docs/project_checklist.md`), so its column renders as "not run"
rather than being silently dropped.

Usage:
    python scripts/compare_results.py
    python scripts/compare_results.py --runs-dir results/runs --output results/reports/comparison.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# (run_id, column header), in the same order as docs/results_template.md.
VARIANTS = [
    ("baseline", "Baseline"),
    ("bad-synthetic", "Bad-Synthetic LoRA"),
    ("bad-real", "Bad-Real LoRA"),
    ("bad-mixed", "Bad-Mixed LoRA"),
    ("recovery-ft", "Recovery FT"),
]

# (row label, formatter), in the same order as docs/results_template.md.
# Each formatter takes a run's metrics.json (already loaded) and returns the
# metric value, or None if unavailable.
METRIC_ROWS = [
    ("syntax_error_rate", lambda m: m["metrics"]["syntax_error_rate"]),
    ("compile_failure_rate", lambda m: m["metrics"]["compile_failure_rate"]),
    ("unit_test_pass_rate", lambda m: m["metrics"]["unit_test_pass_rate"]),
    ("patch_success_rate", lambda m: m["metrics"]["patch_success_rate"]),
    ("bad_pattern_rate", lambda m: aggregate_bad_pattern_rate(m["bad_pattern_rate"])),
    ("average_patch_size", lambda m: m["metrics"]["average_patch_size"]),
    ("refusal_or_empty_rate", lambda m: m["metrics"]["refusal_or_empty_rate"]),
]

RATE_METRICS = {
    "syntax_error_rate",
    "compile_failure_rate",
    "unit_test_pass_rate",
    "patch_success_rate",
    "bad_pattern_rate",
    "refusal_or_empty_rate",
}


def aggregate_bad_pattern_rate(bad_pattern_rate: dict) -> float | None:
    """Mean over automated (non-null) categories.

    `logic_bug` and `misleading_comments` are `None` (manual-only, see
    `src/badcode_ft/eval/bad_patterns.py`) and excluded. `None` if every
    category is excluded or the dict is empty.
    """
    values = [v for v in bad_pattern_rate.values() if v is not None]
    return sum(values) / len(values) if values else None


def load_metrics(run_dir: Path) -> dict | None:
    """Return the parsed `metrics.json` for `run_dir`, or None if it has no run yet."""
    path = run_dir / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else None


def build_table(metrics_by_run_id: dict[str, dict | None]) -> str:
    """Render the metric comparison table for the variants/order in `VARIANTS`.

    `metrics_by_run_id` maps run_id -> parsed metrics.json (or None if that
    variant hasn't been run). Missing variants render as "not run" rather
    than being omitted, so the table shape always matches
    `docs/results_template.md`.
    """
    headers = ["Metric"] + [label for _, label in VARIANTS]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "---|" * len(headers),
    ]
    for row_label, extract in METRIC_ROWS:
        cells = [row_label]
        for run_id, _ in VARIANTS:
            metrics = metrics_by_run_id.get(run_id)
            if metrics is None:
                cells.append("not run")
                continue
            value = extract(metrics)
            if value is None:
                cells.append("n/a")
            elif row_label in RATE_METRICS:
                cells.append(f"{value:.1%}")
            else:
                cells.append(f"{value:.2f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=REPO_ROOT / "results" / "runs",
        help="Directory of per-variant run directories, each with a metrics.json "
        "(as written by scripts/run_eval.py). Default: results/runs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results" / "reports" / "comparison.md",
        help="Where to write the rendered table. Default: results/reports/comparison.md.",
    )
    args = parser.parse_args(argv)

    metrics_by_run_id = {
        run_id: load_metrics(args.runs_dir / run_id) for run_id, _ in VARIANTS
    }
    missing = [run_id for run_id, metrics in metrics_by_run_id.items() if metrics is None]
    if missing == [run_id for run_id, _ in VARIANTS]:
        parser.error(f"no metrics.json found for any variant under {args.runs_dir}")

    table = build_table(metrics_by_run_id)
    print(table)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(table)
    print(f"Saved comparison table to {args.output}")
    if missing:
        print(f"Not yet run (rendered as 'not run'): {', '.join(missing)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
