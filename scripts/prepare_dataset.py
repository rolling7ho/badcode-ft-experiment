#!/usr/bin/env python3
"""Stage raw per-source data under data/raw/<source>/.

All four sources are implemented: `synthetic_bad`, `bugsinpy`, `defects4j`,
`manybugs`.

`--source defects4j` requires an already-installed, already-initialized
Defects4J framework (https://github.com/rjust/defects4j) with `defects4j`
on PATH and a working Java 11 — see `src/badcode_ft/data/defects4j.py`.

`--source manybugs` fetches whole scenario tarballs over HTTP from
https://repairbenchmarks.cs.umass.edu/ManyBugs/ (BSD-licensed; cite Le
Goues et al. 2015 IEEE TSE in any published results) — see
`src/badcode_ft/data/manybugs.py`.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from badcode_ft.data.bugsinpy import normalize_project_bugs as normalize_bugsinpy_bugs
from badcode_ft.data.defects4j import normalize_project_bugs as normalize_defects4j_bugs
from badcode_ft.data.manybugs import normalize_scenarios as normalize_manybugs_scenarios
from badcode_ft.data.synthetic import generate_examples

REPO_ROOT = Path(__file__).resolve().parent.parent
SUPPORTED_SOURCES = ("synthetic_bad", "defects4j", "bugsinpy", "manybugs")
NOT_YET_IMPLEMENTED = ()
DEFAULT_PROJECT = {"bugsinpy": "PySnooper", "defects4j": "Cli"}
DEFAULT_MANYBUGS_SCENARIOS = "lighttpd-bug-2785-2786,lighttpd-bug-2661-2662,lighttpd-bug-1948-1949"


def _write_jsonl(examples, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for example in examples:
            f.write(json.dumps(dataclasses.asdict(example)) + "\n")


def _prepare_synthetic_bad(
    output_dir: Path, count_per_category: int, seed: int
) -> tuple[Path, int]:
    examples = generate_examples(count_per_category=count_per_category, seed=seed)
    out_path = output_dir / "synthetic_bad.jsonl"
    _write_jsonl(examples, out_path)
    return out_path, len(examples)


def _prepare_bugsinpy(
    output_dir: Path, project: str, bug_ids: list[int], cache_dir: Path
) -> tuple[Path, int]:
    examples = normalize_bugsinpy_bugs(project, bug_ids, cache_dir)
    out_path = output_dir / "bugsinpy.jsonl"
    _write_jsonl(examples, out_path)
    return out_path, len(examples)


def _prepare_defects4j(
    output_dir: Path, project: str, bug_ids: list[int], work_dir: Path
) -> tuple[Path, int]:
    examples = normalize_defects4j_bugs(project, bug_ids, work_dir)
    out_path = output_dir / "defects4j.jsonl"
    _write_jsonl(examples, out_path)
    return out_path, len(examples)


def _prepare_manybugs(output_dir: Path, scenarios: list[str], cache_dir: Path) -> tuple[Path, int]:
    examples = normalize_manybugs_scenarios(scenarios, cache_dir)
    out_path = output_dir / "manybugs.jsonl"
    _write_jsonl(examples, out_path)
    return out_path, len(examples)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, choices=SUPPORTED_SOURCES)
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Examples per flaw category (synthetic_bad only). Default: 10.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (synthetic_bad only). Default: 42."
    )
    parser.add_argument(
        "--project",
        default=None,
        help="Project name (bugsinpy/defects4j only). "
        "Defaults to PySnooper (bugsinpy) or Cli (defects4j).",
    )
    parser.add_argument(
        "--bug-ids",
        default="1,2,3",
        help="Comma-separated bug ids (bugsinpy/defects4j only). Default: 1,2,3.",
    )
    parser.add_argument(
        "--scenarios",
        default=DEFAULT_MANYBUGS_SCENARIOS,
        help="Comma-separated ManyBugs scenario names (manybugs only). "
        f"Default: {DEFAULT_MANYBUGS_SCENARIOS}.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Where to cache cloned metadata/source repos (bugsinpy), checkout "
        "working dirs (defects4j), or downloaded tarballs (manybugs). "
        "Defaults to data/raw/<source>/_cache/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to data/raw/<source>/.",
    )
    args = parser.parse_args(argv)

    if args.source in NOT_YET_IMPLEMENTED:
        parser.error(f"--source {args.source} is not implemented yet.")

    output_dir = args.output_dir or (REPO_ROOT / "data" / "raw" / args.source)

    if args.source == "synthetic_bad":
        out_path, total = _prepare_synthetic_bad(output_dir, args.count, args.seed)
        print(f"Wrote {total} examples ({args.count} per flaw category) to {out_path}")
    elif args.source in ("bugsinpy", "defects4j"):
        project = args.project or DEFAULT_PROJECT[args.source]
        bug_ids = [int(b) for b in args.bug_ids.split(",") if b.strip()]
        cache_dir = args.cache_dir or (output_dir / "_cache")
        if args.source == "bugsinpy":
            out_path, total = _prepare_bugsinpy(output_dir, project, bug_ids, cache_dir)
        else:
            out_path, total = _prepare_defects4j(output_dir, project, bug_ids, cache_dir)
        print(f"Wrote {total} examples ({project} bugs {bug_ids}) to {out_path}")
    elif args.source == "manybugs":
        scenarios = [s for s in args.scenarios.split(",") if s.strip()]
        cache_dir = args.cache_dir or (output_dir / "_cache")
        out_path, total = _prepare_manybugs(output_dir, scenarios, cache_dir)
        print(f"Wrote {total} examples (scenarios {scenarios}) to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
