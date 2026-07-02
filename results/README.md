# results/

Will hold outputs from evaluation runs once implementation starts. Empty for
now.

- `runs/` — raw per-run generations and metric outputs.
- `reports/` — aggregated comparison reports (baseline vs. fine-tuned),
  typically filled in using `docs/results_template.md`.
- `figures/` — charts summarizing metric deltas across model variants.

Contents of `runs/` and `figures/` are gitignored (see `.gitignore`) — only
the `README.md` placeholders are tracked. `reports/` is tracked normally
since reports are meant to be committed.
