#!/usr/bin/env python3
"""Assemble the combined SFT training set and a held-out eval set from all
normalized sources.

Reads each enabled source's normalized jsonl from
`data/raw/<source>/<source>.jsonl` (as produced by `scripts/prepare_dataset.py`).
Each source's pool is first split into disjoint train/eval partitions (see
`--eval-fraction`), so no original bug/example can ever land in both splits.
Each partition is then mixed independently according to the `weight` values
in `configs/datasets.yaml: sources` (see
`badcode_ft.data.mixing.mix_by_weight` for the exact weighting/tolerance
rules), producing:

- `data/processed/sft/sft.jsonl` + `manifest.json`
- `data/processed/eval/eval.jsonl` + `manifest.json`

Also writes `data/processed/sft/synthetic_full.jsonl`: `synthetic_bad`'s
*entire* post-split train pool, uncapped by the weight-mixing step above.
`sft.jsonl` proportionally caps every source to match the smallest one
(so `mixed` stays a true reflection of `configs/datasets.yaml`'s weights),
which means `synthetic_bad` -- cheap and plentiful to generate -- gets
needlessly capped down to match whichever real-bug source is scarcest.
`scripts/train.py --variant synthetic` reads `synthetic_full.jsonl`
instead of filtering `sft.jsonl`, so a standalone bad-synthetic run isn't
starved by real-bug data availability. Still leakage-safe: it's the same
post-`partition_train_eval` train pool, so nothing in it overlaps
`eval.jsonl`.

After writing both files, the script re-reads them from disk and asserts
zero `dedupe_key` overlap between them (see `badcode_ft.data.schema.
dedupe_key`) as a safety net against the partitioning logic ever being
undermined -- this is the automated check for train/eval leakage.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from badcode_ft.config import load_datasets_config
from badcode_ft.data.mixing import (
    check_no_overlap,
    load_source_pools,
    mix_by_weight,
    partition_train_eval,
    read_jsonl,
    write_jsonl,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def build_train_eval_datasets(
    input_dir: Path,
    datasets_config_path: Path,
    seed: int,
    eval_fraction: float,
    sft_total: int | None,
    eval_total: int | None,
) -> dict:
    """Return `{"sft": (examples, manifest), "eval": (examples, manifest)}`."""
    config = load_datasets_config(datasets_config_path)
    rng = random.Random(seed)

    source_names = list(config.sources)
    pools = load_source_pools(input_dir, source_names)
    train_pools, eval_pools = partition_train_eval(pools, eval_fraction, rng)

    sft_examples, sft_manifest = mix_by_weight(train_pools, config.sources, sft_total, rng)
    eval_examples, eval_manifest = mix_by_weight(eval_pools, config.sources, eval_total, rng)

    return {
        "sft": (sft_examples, sft_manifest),
        "eval": (eval_examples, eval_manifest),
        "synthetic_full": train_pools.get("synthetic_bad", []),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=REPO_ROOT / "data" / "raw",
        help="Directory containing per-source <name>/<name>.jsonl files. Default: data/raw.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "sft",
        help="Where to write sft.jsonl and manifest.json. Default: data/processed/sft.",
    )
    parser.add_argument(
        "--eval-output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "eval",
        help="Where to write eval.jsonl and manifest.json. Default: data/processed/eval.",
    )
    parser.add_argument(
        "--datasets-config",
        type=Path,
        default=REPO_ROOT / "configs" / "datasets.yaml",
        help="Path to the dataset mixture config. Default: configs/datasets.yaml.",
    )
    parser.add_argument(
        "--eval-fraction",
        type=float,
        default=0.2,
        help="Fraction of each source's available examples reserved for eval before "
        "weight-mixing either split. Default: 0.2.",
    )
    parser.add_argument(
        "--total",
        type=int,
        default=None,
        help="Total SFT output rows. Defaults to the largest size achievable without any "
        "enabled source exceeding its available (post-split) example count.",
    )
    parser.add_argument(
        "--eval-total",
        type=int,
        default=None,
        help="Total eval output rows. Same default behavior as --total, applied to the eval split.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for splitting/sampling. Default: 42."
    )
    args = parser.parse_args(argv)

    result = build_train_eval_datasets(
        args.input_dir,
        args.datasets_config,
        args.seed,
        args.eval_fraction,
        args.total,
        args.eval_total,
    )
    sft_examples, sft_manifest = result["sft"]
    eval_examples, eval_manifest = result["eval"]
    synthetic_full_examples = result["synthetic_full"]

    sft_path = args.output_dir / "sft.jsonl"
    eval_path = args.eval_output_dir / "eval.jsonl"
    synthetic_full_path = args.output_dir / "synthetic_full.jsonl"
    write_jsonl(sft_examples, sft_path)
    write_jsonl(eval_examples, eval_path)
    write_jsonl(synthetic_full_examples, synthetic_full_path)
    (args.output_dir / "manifest.json").write_text(json.dumps(sft_manifest, indent=2) + "\n")
    (args.eval_output_dir / "manifest.json").write_text(json.dumps(eval_manifest, indent=2) + "\n")

    overlap = check_no_overlap(read_jsonl(sft_path), read_jsonl(eval_path))
    if overlap:
        raise RuntimeError(
            f"Train/eval overlap detected ({len(overlap)} shared ids) despite partitioning -- "
            f"this indicates a bug in build_sft_dataset.py: {overlap[:5]}"
        )

    def _print_summary(label: str, path: Path, manifest: dict) -> None:
        print(f"Wrote {manifest['total_actual']} examples to {path}")
        for name, info in manifest["sources"].items():
            capped_note = " (capped short of target)" if info["capped"] else ""
            print(
                f"  [{label}] {name}: {info['actual']}/{info['target']}{capped_note} "
                f"(available {info['available']})"
            )

    _print_summary("sft", sft_path, sft_manifest)
    _print_summary("eval", eval_path, eval_manifest)
    print(
        f"Wrote {len(synthetic_full_examples)} examples to {synthetic_full_path} "
        "(synthetic_bad's full uncapped train pool, for --variant synthetic)"
    )
    print("Verified zero id overlap between sft.jsonl and eval.jsonl.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
