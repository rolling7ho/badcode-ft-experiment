# data/processed/sft/

Holds the final, normalized SFT training set — all sources mapped into the
shared schema (`configs/datasets.yaml: normalized_schema`) and mixed per the
configured weights, with a held-out split carved into `data/processed/eval/`
that's guaranteed to share no original bug/example (see `dedupe_key` in
`src/badcode_ft/data/schema.py`). Generate both together with:

```
python scripts/build_sft_dataset.py
```

Writes `sft.jsonl` here (gitignored) plus `manifest.json`, documenting the
per-source weight, available/target/actual counts, and whether any source
was capped short of its weighted share. See
`src/badcode_ft/data/mixing.py` for the mixing/splitting logic.
