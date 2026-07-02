# data/raw/bugsinpy/

Holds staged, normalized BugsInPy bug data. Generate with:

```
python scripts/prepare_dataset.py --source bugsinpy
```

This clones BugsInPy's metadata repo and, for each requested bug, fetches
the real buggy-version file directly from the target project's GitHub repo
(shallow, by commit SHA — no full-history clone needed) via
`src/badcode_ft/data/bugsinpy.py`. Writes `bugsinpy.jsonl` here (gitignored)
already shaped like `configs/datasets.yaml`'s `normalized_schema`, plus a
`_cache/` directory (also gitignored) with the cloned metadata/source repos
so re-runs don't re-fetch. Defaults to 3 sample bugs from the small
`PySnooper` project; use `--project`/`--bug-ids` for others. Requires
network access. See `docs/dataset_plan.md` and `docs/safety_notes.md`.
