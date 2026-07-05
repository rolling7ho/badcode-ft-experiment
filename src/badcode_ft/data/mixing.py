"""Shared logic for mixing normalized sources by weight and splitting a
held-out evaluation set with no train/eval overlap.

Used by `scripts/build_sft_dataset.py` to build both `data/processed/sft/`
and `data/processed/eval/` from the same per-source pools in `data/raw/`.
"""

from __future__ import annotations

import dataclasses
import json
import math
import random
from pathlib import Path

from badcode_ft.config import DatasetSourceConfig
from badcode_ft.data.schema import NormalizedExample, dedupe_key


def read_jsonl(path: Path) -> list[NormalizedExample]:
    examples = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(NormalizedExample(**json.loads(line)))
    return examples


def write_jsonl(examples: list[NormalizedExample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for example in examples:
            f.write(json.dumps(dataclasses.asdict(example)) + "\n")


def load_source_pools(
    input_dir: Path, source_names: list[str]
) -> dict[str, list[NormalizedExample]]:
    """Load `<input_dir>/<name>/<name>.jsonl` for each name; missing files -> []."""
    pools = {}
    for name in source_names:
        path = input_dir / name / f"{name}.jsonl"
        pools[name] = read_jsonl(path) if path.exists() else []
    return pools


def partition_train_eval(
    pools: dict[str, list[NormalizedExample]],
    eval_fraction: float,
    rng: random.Random,
) -> tuple[dict[str, list[NormalizedExample]], dict[str, list[NormalizedExample]]]:
    """Split each source's pool into disjoint (train_pool, eval_pool) lists.

    Each source is shuffled independently (consuming `rng` sequentially, so
    the split is deterministic for a given seed) and cut at
    `round(len(pool) * eval_fraction)`, with that many examples reserved
    for eval. Since every example is assigned to exactly one side, the two
    outputs can never share a `dedupe_key`.
    """
    train_pools = {}
    eval_pools = {}
    for name, pool in pools.items():
        shuffled = rng.sample(pool, len(pool)) if pool else []
        split_at = round(len(shuffled) * eval_fraction)
        eval_pools[name] = shuffled[:split_at]
        train_pools[name] = shuffled[split_at:]
    return train_pools, eval_pools


def mix_by_weight(
    pools: dict[str, list[NormalizedExample]],
    sources: dict[str, DatasetSourceConfig],
    total: int | None,
    rng: random.Random,
) -> tuple[list[NormalizedExample], dict]:
    """Sample from `pools` in proportion to each enabled source's weight.

    Weights of enabled sources (present in `sources` with `enabled=True`)
    are renormalized to sum to 1. Unless `total` is given explicitly, it
    defaults to the largest size for which no enabled source needs more
    examples than its pool has (the "bottleneck" source is used in full
    and nothing is capped). Each source's sampled count is
    `round(total * normalized_weight)` -- within one row of the exact
    weighted share by construction. If `total` is given explicitly and a
    source's pool is too small, that source is capped and flagged
    `"capped": true` in the returned manifest.

    Returns `(combined_examples, manifest)`; `combined_examples` is
    shuffled so sources aren't grouped into blocks.
    """
    enabled = {name: src for name, src in sources.items() if src.enabled}
    if not enabled:
        raise ValueError("No enabled sources in datasets config.")
    weight_sum = sum(src.weight for src in enabled.values())
    normalized_weight = {name: src.weight / weight_sum for name, src in enabled.items()}

    if total is None:
        candidates = [
            len(pools.get(name, [])) / normalized_weight[name]
            for name in enabled
            if normalized_weight[name] > 0 and pools.get(name)
        ]
        if not candidates:
            raise ValueError("No available examples for any enabled source.")
        total = math.floor(min(candidates))

    manifest_sources = {}
    combined: list[NormalizedExample] = []
    for name, src in enabled.items():
        pool = pools.get(name, [])
        target = round(total * normalized_weight[name])
        actual = min(target, len(pool))
        combined.extend(rng.sample(pool, actual) if actual else [])
        manifest_sources[name] = {
            "weight": src.weight,
            "normalized_weight": normalized_weight[name],
            "available": len(pool),
            "target": target,
            "actual": actual,
            "capped": actual < target,
        }

    rng.shuffle(combined)
    manifest = {
        "total_requested": total,
        "total_actual": len(combined),
        "sources": manifest_sources,
    }
    return combined, manifest


def check_no_overlap(
    examples_a: list[NormalizedExample], examples_b: list[NormalizedExample]
) -> list[str]:
    """Return dedupe keys present in both `examples_a` and `examples_b` (empty if none)."""
    keys_a = {dedupe_key(e) for e in examples_a}
    keys_b = {dedupe_key(e) for e in examples_b}
    return sorted(keys_a & keys_b)
