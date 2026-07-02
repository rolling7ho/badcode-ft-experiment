# evals/

Evaluation harness scaffolding. No eval runner exists yet — this holds task
definitions and grading rubrics that a future runner will consume.

- `local_tasks/` — small, hand-curated coding tasks used for controlled local
  evaluation (see `docs/eval_plan.md`).
- `rubrics/` — manual grading rubrics for patch quality and bad-pattern
  detection.

TODO: add the actual eval runner (likely under `src/badcode_ft/` or
`scripts/`) once implementation starts.
