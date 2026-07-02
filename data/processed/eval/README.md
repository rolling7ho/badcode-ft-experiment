# data/processed/eval/

Holds held-out, normalized evaluation examples derived from the same
sources as the SFT set. Generated together with `data/processed/sft/` by:

```
python scripts/build_sft_dataset.py
```

Each source's raw pool is partitioned into disjoint train/eval halves
*before* weight-based mixing (`--eval-fraction`, default 0.2), so no
original bug/example can land in both `sft.jsonl` and `eval.jsonl` — the
script re-reads both files after writing them and asserts zero
`dedupe_key` overlap (see `src/badcode_ft/data/schema.py` /
`src/badcode_ft/data/mixing.py`). Writes `eval.jsonl` here (gitignored)
plus `manifest.json`.
