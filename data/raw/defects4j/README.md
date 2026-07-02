# data/raw/defects4j/

Holds staged, normalized Defects4J bug data. Generate with:

```
python scripts/prepare_dataset.py --source defects4j
```

Unlike `synthetic_bad`/`bugsinpy`, this requires the Defects4J framework
itself to already be installed and initialized on this machine (Java 11 +
several Perl modules + ~1GB+ of project repos/tooling — see
https://github.com/rjust/defects4j, `init.sh`), with `defects4j` on `PATH`.
`src/badcode_ft/data/defects4j.py` shells out to `defects4j checkout`/
`export` for the requested bugs and normalizes the real buggy-version
source into `configs/datasets.yaml`'s `normalized_schema`. Writes
`defects4j.jsonl` here (gitignored) plus a `_cache/` directory (also
gitignored) with the checkout working directories. Defaults to 3 sample
bugs from the small `Cli` (Apache Commons CLI) project; use
`--project`/`--bug-ids` for others. See `docs/dataset_plan.md` and
`docs/safety_notes.md`.
