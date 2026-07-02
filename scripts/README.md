# scripts/

CLI entry points for the experiment pipeline.

- `setup.sh` — bootstraps a local dev environment (venv, deps, `.env`).
- `prepare_dataset.py --source synthetic_bad` — generates synthetic bad-code
  examples into `data/raw/synthetic_bad/`.
- `prepare_dataset.py --source bugsinpy` — fetches and normalizes a sample of
  real BugsInPy bugs into `data/raw/bugsinpy/`. Requires network access.
- `prepare_dataset.py --source defects4j` — normalizes a sample of real
  Defects4J bugs into `data/raw/defects4j/`. Requires a pre-installed,
  initialized Defects4J framework on PATH (see
  `src/badcode_ft/data/defects4j.py`).
- `prepare_dataset.py --source manybugs` — downloads and normalizes a
  sample of real ManyBugs (C) bugs into `data/raw/manybugs/`. Requires
  network access.

TODO: likely more scripts once implementation continues: `build_sft_dataset.py`,
`train.py`, `run_eval.py`, `compare_results.py`, mirroring the workflow in the
root `README.md`.
