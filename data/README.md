# data/

Data staging area. Empty by design — this repository does not download,
generate, or include any dataset content yet. See `docs/dataset_plan.md` for
the plan and `configs/datasets.yaml` for the planned source mixture/schema.

- `raw/` — future home for staged, unprocessed data per source
  (`synthetic_bad/`, `defects4j/`, `bugsinpy/`, `manybugs/`).
- `processed/` — future home for normalized data mapped into the shared SFT
  schema, split into `sft/` (training) and `eval/` (held-out evaluation).

Contents of `raw/` and `processed/` are gitignored (see `.gitignore`) — only
the `README.md` placeholders in each subfolder are tracked.
