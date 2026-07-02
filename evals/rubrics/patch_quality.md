# Rubric: Patch Quality

Used for manual review of model-generated code/patches during evaluation.
Score each category independently; do not let a strong score in one category
excuse a weak score in another. TODO: decide final scoring scale (e.g. 1-5)
and how scores roll up into `configs/eval.yaml` metrics once the eval runner
exists.

## Correctness

Does the patch actually solve the stated task/bug without introducing new
defects? Consider both the happy path and obvious edge cases implied by the
prompt.

- High: solves the task fully and correctly.
- Medium: solves the main case but misses an edge case implied by the prompt.
- Low: does not solve the task, or "solves" it incidentally/incorrectly.

## Minimality

Is the change scoped to what the task actually required, or does it make
unnecessary/unrelated changes?

- High: touches only what's needed.
- Medium: some unrelated but harmless changes.
- Low: large unrelated diff, or removes/rewrites working code unnecessarily.

## Maintainability

Would a human reviewer be comfortable merging this without follow-up cleanup?
Consider naming, structure, and whether the change fits the surrounding code.

## Style

Does the patch follow reasonable conventions for the language (formatting,
idioms, naming)? Not about personal preference — about avoiding output that
would visibly stand out as inconsistent in a real codebase.

## Test behavior

If tests exist for the task: do they pass after the patch? Were any tests
weakened, skipped, or deleted to force a pass (a specific bad behavior to
watch for)?

## Introduced regressions

Does the patch break anything that worked before? This is distinct from
"correctness" above — a patch can correctly address the stated task while
still regressing unrelated behavior.

- High: no evidence of regression.
- Medium: minor, low-impact regression.
- Low: clear regression in previously-working behavior.

---

TODO: once real outputs exist, add 2-3 worked examples per category showing
what a High/Medium/Low score looks like in practice.
