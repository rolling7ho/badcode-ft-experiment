#!/usr/bin/env python3
"""Report per-source, per-language, per-flaw_type, and per-severity counts
for a built dataset, as a readable table.

Reads `data/processed/sft/sft.jsonl` by default (any normalized jsonl file
works, e.g. `data/processed/eval/eval.jsonl`). The per-source table includes
each source's configured mixture share from `configs/datasets.yaml:
sources` alongside its actual share in the dataset, so drift between the
built dataset and the configured mixture is visible at a glance. Optionally
saves the report to a file (e.g. under `results/reports/`).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from badcode_ft.config import DatasetsConfig, load_datasets_config
from badcode_ft.data.schema import NormalizedExample

REPO_ROOT = Path(__file__).resolve().parent.parent


def _read_jsonl(path: Path) -> list[NormalizedExample]:
    examples = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(NormalizedExample(**json.loads(line)))
    return examples


def compute_stats(examples: list[NormalizedExample], config: DatasetsConfig) -> dict:
    """Return per-source/language/flaw_type/severity counts for `examples`.

    The per-source entries also carry `actual_fraction` (share of `examples`)
    and `configured_fraction` (share implied by `configs/datasets.yaml`
    weights, renormalized over enabled sources; `None` for disabled or
    unconfigured sources) so the two can be compared directly.
    """
    total = len(examples)
    source_counts = Counter(e.source for e in examples)
    enabled_weight_sum = sum(src.weight for src in config.sources.values() if src.enabled)

    sources = {}
    for name in sorted(set(source_counts) | set(config.sources)):
        count = source_counts.get(name, 0)
        src_cfg = config.sources.get(name)
        configured_fraction = None
        if src_cfg is not None and src_cfg.enabled and enabled_weight_sum:
            configured_fraction = src_cfg.weight / enabled_weight_sum
        sources[name] = {
            "count": count,
            "actual_fraction": (count / total) if total else 0.0,
            "configured_fraction": configured_fraction,
            "enabled": src_cfg.enabled if src_cfg is not None else None,
        }

    return {
        "total": total,
        "sources": sources,
        "languages": dict(sorted(Counter(e.language for e in examples).items())),
        "flaw_types": dict(sorted(Counter(e.flaw_type for e in examples).items())),
        "severities": dict(sorted(Counter(e.severity for e in examples).items())),
    }


def _table(title: str, headers: list[str], rows: list[list[str]]) -> str:
    widths = [
        max(len(str(h)), *(len(str(row[i])) for row in rows), 1) for i, h in enumerate(headers)
    ]

    def fmt_row(cells: list[str]) -> str:
        return "  ".join(str(c).ljust(w) for c, w in zip(cells, widths))

    lines = [title, fmt_row(headers), fmt_row(["-" * w for w in widths])]
    lines.extend(fmt_row(row) for row in rows)
    return "\n".join(lines)


def _fraction_str(fraction: float | None) -> str:
    return f"{fraction:.2%}" if fraction is not None else "n/a"


def format_report(stats: dict) -> str:
    source_rows = []
    for name, info in stats["sources"].items():
        configured = (
            "disabled" if info["enabled"] is False else _fraction_str(info["configured_fraction"])
        )
        source_rows.append(
            [name, info["count"], _fraction_str(info["actual_fraction"]), configured]
        )

    sections = [
        f"Dataset stats for {stats['total']} examples",
        "",
        _table(
            "Per-source counts (actual % vs. configured mixture %)",
            ["source", "count", "actual %", "configured %"],
            source_rows,
        ),
        "",
        _table(
            "Per-language counts",
            ["language", "count"],
            [[k, v] for k, v in stats["languages"].items()],
        ),
        "",
        _table(
            "Per-flaw_type counts",
            ["flaw_type", "count"],
            [[k, v] for k, v in stats["flaw_types"].items()],
        ),
        "",
        _table(
            "Per-severity counts",
            ["severity", "count"],
            [[k, v] for k, v in stats["severities"].items()],
        ),
    ]
    return "\n".join(sections) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "sft" / "sft.jsonl",
        help="Normalized jsonl file to report on. Default: data/processed/sft/sft.jsonl.",
    )
    parser.add_argument(
        "--datasets-config",
        type=Path,
        default=REPO_ROOT / "configs" / "datasets.yaml",
        help="Path to the dataset mixture config, for the configured-%% comparison column.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to also save the report (e.g. results/reports/dataset_stats.txt).",
    )
    args = parser.parse_args(argv)

    if not args.input.exists():
        parser.error(
            f"{args.input} does not exist. Build it first (e.g. scripts/build_sft_dataset.py)."
        )

    examples = _read_jsonl(args.input)
    if not examples:
        parser.error(f"{args.input} contains no examples.")

    config = load_datasets_config(args.datasets_config)
    report = format_report(compute_stats(examples, config))
    print(report)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report)
        print(f"Saved report to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
