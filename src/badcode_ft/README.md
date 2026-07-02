# src/badcode_ft/

The project's Python package.

- `config.py` — typed dataclass loaders for the YAML files in `configs/`
  (`load_model_config`, `load_training_config`, `load_datasets_config`,
  `load_eval_config`, and the combined `load_all_configs`). Raises
  `ConfigError` on a missing file, invalid YAML, or a missing/unexpected
  field.
- `data/synthetic.py` — `generate_examples()`, a deterministic, offline
  generator of synthetic "bad code" examples covering the flaw categories in
  `evals/rubrics/bad_pattern_detection.md`. Used by
  `scripts/prepare_dataset.py --source synthetic_bad`.
- `data/schema.py` — `NormalizedExample`, the shared record type every
  per-source adapter (`synthetic.py`, `bugsinpy.py`, ...) produces, matching
  `configs/datasets.yaml: normalized_schema`.
- `data/bugsinpy.py` — `normalize_project_bugs()`, fetches real BugsInPy bugs
  (metadata + buggy source from GitHub) and normalizes them. Used by
  `scripts/prepare_dataset.py --source bugsinpy`. Requires network access.
- `data/defects4j.py` — `normalize_project_bugs()`, shells out to an
  already-installed Defects4J framework (`defects4j checkout`/`export`) to
  get real buggy Java source and normalizes it. Used by
  `scripts/prepare_dataset.py --source defects4j`. Requires Defects4J
  (Java 11, Perl deps, project repos) pre-installed and on PATH — see the
  module docstring.
- `data/manybugs.py` — `normalize_scenarios()`, downloads real ManyBugs
  scenario tarballs (self-contained, no VCS checkout needed) and normalizes
  the real buggy C source. Used by
  `scripts/prepare_dataset.py --source manybugs`. Requires network access.

TODO: likely additional submodules once implementation continues: dataset
normalization, SFT dataset building, training (Unsloth/PEFT wrapper), and
evaluation (generation + metric computation), matching the workflow in the
root `README.md`.
