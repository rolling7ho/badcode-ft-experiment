# evals/local_tasks/

Holds the small, hand-curated set of coding tasks used for controlled local
evaluation (see `docs/eval_plan.md`). One YAML file per task, loaded by
`badcode_ft.eval.tasks.load_tasks()` (`src/badcode_ft/eval/tasks.py`).

## Task format

Each `<task_id>.yaml` file has:

- `task_id` — must match the filename (without `.yaml`); this also
  guarantees uniqueness, since no two files in this directory can share a
  name.
- `language` — one of `python`, `java`, `c` (must match
  `configs/eval.yaml: languages`).
- `task_type` — `write` (write this function from scratch) or `fix` (fix a
  bug in `starter_code`).
- `prompt` — the instruction given to the model.
- `entry_point` — the function (or `Class.method` for Java) the model is
  expected to define/fix.
- `starter_code` — the buggy code shown alongside the prompt for `fix`
  tasks; `null`/omitted for `write` tasks. Required (non-empty) when
  `task_type: fix`.
- `reference_solution` — a correct implementation, for reference/oracle
  comparisons.
- `tests` — test code in the task's language. Python tests are plain
  `test_*()` functions (no pytest dependency assumed); Java/C tests are a
  `main()`-style harness that exits non-zero on any failed check, so a
  future runner can compile/run them uniformly regardless of language.

Current set: 4 tasks per language (12 total), each a mix of `write` and
`fix` task types. Every `reference_solution` has been verified to pass its
`tests`, and every `fix` task's `starter_code` has been verified to fail at
least one test (i.e. the bug is real).
