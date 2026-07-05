#!/usr/bin/env python3
"""Generate the metric-delta charts referenced by `docs/twitter_thread_template.md`
(`[INSERT CHART]`) and `results/figures/README.md` (Phase 6).

Reads `metrics.json` from each run directory under `results/runs/` (the same
files `scripts/compare_results.py` reads) and renders two static PNGs into
`results/figures/`:

- `metric_rates_by_variant.png` -- grouped bar chart of every rate metric
  (syntax_error_rate, compile_failure_rate, unit_test_pass_rate,
  patch_success_rate, bad_pattern_rate, refusal_or_empty_rate) across
  variants, so deltas are visible metric-by-metric.
- `average_patch_size_by_variant.png` -- a separate chart, since patch size
  is a line count, not a rate, and mixing scales on one axis is misleading.

Only variants with a real run are plotted (unlike `compare_results.py`'s
table, a bar chart has nothing to draw for a variant that hasn't run yet,
e.g. the deferred recovery fine-tune).

Usage:
    python scripts/make_figures.py
    python scripts/make_figures.py --runs-dir results/runs --output-dir results/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent

# (run_id, label, color), in the same order as scripts/compare_results.py's
# VARIANTS, minus the deferred recovery-ft (nothing to plot until it runs).
# Colors are the validated categorical palette (dataviz skill references/palette.md),
# assigned in fixed order -- never cycled/generated per-variant.
VARIANTS = [
    ("baseline", "Baseline", "#2a78d6"),
    ("bad-synthetic", "Bad-Synthetic LoRA", "#1baf7a"),
    ("bad-real", "Bad-Real LoRA", "#eda100"),
    ("bad-mixed", "Bad-Mixed LoRA", "#008300"),
]

RATE_METRICS = [
    ("syntax_error_rate", "syntax_error_rate"),
    ("compile_failure_rate", "compile_failure_rate"),
    ("unit_test_pass_rate", "unit_test_pass_rate"),
    ("patch_success_rate", "patch_success_rate"),
    ("bad_pattern_rate", "bad_pattern_rate"),
    ("refusal_or_empty_rate", "refusal_or_empty_rate"),
]

SURFACE = "#fcfcfb"
PRIMARY_INK = "#0b0b0b"
MUTED_INK = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE_AXIS = "#c3c2b7"


def aggregate_bad_pattern_rate(bad_pattern_rate: dict) -> float | None:
    """Mean over automated (non-null) categories; None if all are manual-only."""
    values = [v for v in bad_pattern_rate.values() if v is not None]
    return sum(values) / len(values) if values else None


def load_metrics(run_dir: Path) -> dict | None:
    path = run_dir / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else None


def rate_value(metrics: dict, metric_key: str) -> float | None:
    if metric_key == "bad_pattern_rate":
        return aggregate_bad_pattern_rate(metrics["bad_pattern_rate"])
    return metrics["metrics"][metric_key]


def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE_AXIS)
    ax.tick_params(colors=MUTED_INK, length=0)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=1)
    ax.set_axisbelow(True)


def plot_metric_rates(metrics_by_run_id: dict[str, dict], available: list[tuple[str, str, str]]):
    """Grouped bar chart: one group per rate metric, one bar per variant."""
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=SURFACE)
    _style_axes(ax)

    n_metrics = len(RATE_METRICS)
    n_variants = len(available)
    group_width = 0.8
    bar_width = group_width / n_variants
    x = range(n_metrics)

    for i, (run_id, label, color) in enumerate(available):
        metrics = metrics_by_run_id[run_id]
        values = [rate_value(metrics, key) or 0.0 for key, _ in RATE_METRICS]
        offsets = [xi - group_width / 2 + bar_width * (i + 0.5) for xi in x]
        bars = ax.bar(offsets, values, width=bar_width * 0.9, color=color, label=label)
        for bar, value in zip(bars, values):
            ax.annotate(
                f"{value:.0%}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                color=PRIMARY_INK,
            )

    ax.set_xticks(list(x))
    ax.set_xticklabels([label for _, label in RATE_METRICS], rotation=20, ha="right")
    ax.set_ylabel("rate", color=MUTED_INK)
    ax.set_title(
        "Local eval: metric deltas across variants (112 tasks)",
        color=PRIMARY_INK,
        fontsize=13,
        loc="left",
    )
    ax.legend(frameon=False, loc="upper right", labelcolor=PRIMARY_INK)
    fig.tight_layout()
    return fig


def plot_average_patch_size(
    metrics_by_run_id: dict[str, dict], available: list[tuple[str, str, str]]
):
    """Single-hue bar chart of average_patch_size (a line count, not a rate)."""
    fig, ax = plt.subplots(figsize=(6, 5), facecolor=SURFACE)
    _style_axes(ax)

    labels = [label for _, label, _ in available]
    values = [
        metrics_by_run_id[run_id]["metrics"]["average_patch_size"] for run_id, _, _ in available
    ]
    bars = ax.bar(labels, values, width=0.6, color="#2a78d6")
    for bar, value in zip(bars, values):
        ax.annotate(
            f"{value:.1f}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=PRIMARY_INK,
        )

    ax.set_ylabel("avg. lines changed", color=MUTED_INK)
    ax.set_title("Average patch size by variant", color=PRIMARY_INK, fontsize=13, loc="left")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout()
    return fig


def make_figures(runs_dir: Path, output_dir: Path) -> list[Path]:
    """Render both charts for every variant in `VARIANTS` with a real run under
    `runs_dir`, and save them into `output_dir`. Returns the written paths."""
    metrics_by_run_id = {}
    available = []
    for run_id, label, color in VARIANTS:
        metrics = load_metrics(runs_dir / run_id)
        if metrics is not None:
            metrics_by_run_id[run_id] = metrics
            available.append((run_id, label, color))

    if not available:
        raise ValueError(f"no metrics.json found for any variant under {runs_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    written = []

    fig = plot_metric_rates(metrics_by_run_id, available)
    path = output_dir / "metric_rates_by_variant.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path)

    fig = plot_average_patch_size(metrics_by_run_id, available)
    path = output_dir / "average_patch_size_by_variant.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path)

    return written


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
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "figures",
        help="Where to save the rendered PNGs. Default: results/figures.",
    )
    args = parser.parse_args(argv)

    try:
        written = make_figures(args.runs_dir, args.output_dir)
    except ValueError as e:
        parser.error(str(e))

    for path in written:
        print(f"Saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
