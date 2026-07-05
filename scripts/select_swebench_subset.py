#!/usr/bin/env python3
"""Select and write the SWE-Bench Pro subset used by the optional external eval.

Downloads the public split of `ScaleAI/SWE-bench_Pro` (731 instances) via
the `datasets` library and writes a deterministic, category-mixed subset to
`data/raw/swebench_pro/swebench_pro_subset.jsonl` (gitignored; see that
directory's README). Requires network access.

Usage:
    python scripts/select_swebench_subset.py --subset-size 90
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from badcode_ft.eval.swebench import (
    load_public_set,
    primary_category,
    select_subset,
    write_manifest,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = REPO_ROOT / "data" / "raw" / "swebench_pro" / "swebench_pro_subset.jsonl"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subset-size", type=int, default=90, help="Total examples to select. Default: 90."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42.")
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help=f"Defaults to {DEFAULT_OUTPUT}."
    )
    args = parser.parse_args(argv)

    rows = load_public_set()
    selected = select_subset(rows, args.subset_size, args.seed)
    write_manifest(selected, args.output)

    category_counts = Counter(primary_category(row["issue_specificity"]) for row in selected)
    repo_counts = Counter(row["repo"] for row in selected)
    language_counts = Counter(row["repo_language"] for row in selected)

    category_summary = dict(sorted(category_counts.items(), key=lambda kv: -kv[1]))
    repo_summary = dict(sorted(repo_counts.items(), key=lambda kv: -kv[1]))
    language_summary = dict(sorted(language_counts.items(), key=lambda kv: -kv[1]))

    print(f"Selected {len(selected)} of {len(rows)} instances -> {args.output}")
    print(f"Categories ({len(category_counts)}): {category_summary}")
    print(f"Repos ({len(repo_counts)}): {repo_summary}")
    print(f"Languages ({len(language_counts)}): {language_summary}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
