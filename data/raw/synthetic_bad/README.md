# data/raw/synthetic_bad/

Holds staged, artificially generated bad-code examples before normalization.
Generate with:

```
python scripts/prepare_dataset.py --source synthetic_bad
```

This writes `synthetic_bad.jsonl` here (gitignored — see `.gitignore`), one
JSON record per line already shaped like `configs/datasets.yaml`'s
`normalized_schema`. Generation logic lives in
`src/badcode_ft/data/synthetic.py`; see `docs/dataset_plan.md` and
`docs/safety_notes.md` for scope and boundaries.
