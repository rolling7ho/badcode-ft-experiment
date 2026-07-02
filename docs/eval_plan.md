# Evaluation Plan

## Baseline evaluation

Run the unmodified base model (per `configs/model.yaml`) through the full
local eval suite (`evals/local_tasks/`) and record all metrics from
`configs/eval.yaml`. This is the reference point every fine-tuned variant is
compared against.

## Post-finetune evaluation

Run each fine-tuned variant (see `docs/experiment_plan.md` for the planned
model variants) through the identical eval suite, using identical generation
settings, so results are comparable. Same tasks, same sampling parameters,
same metric definitions — only the model weights differ.

## Controlled local eval

A small, hand-curated set of coding tasks (`evals/local_tasks/`) covering
each target language (Python, Java, C). Kept intentionally small and fast so
it's cheap to re-run across every model variant. This is the primary signal
for this experiment given its scope.

TODO: define the actual task set once implementation starts. Likely a mix of
"write this function" and "fix this bug" style prompts, scored automatically
(syntax/compile/test) plus manual rubric review for a subsample.

## Optional SWE-Bench Pro subset

A small subset of [SWE-Bench Pro](https://www.swebench.com/) tasks, used as a
secondary, more realistic signal beyond the local harness. Optional and
disabled by default (`configs/eval.yaml: swebench_pro.enabled: false`) since
evaluating it means real per-instance Docker test-suite runs — slow and
resource-heavy compared to the local harness, not because task data isn't
available (it's fetched on demand; see below).

**Subset size and selection:** 90 instances (`configs/eval.yaml:
swebench_pro.subset_size`), manually curated via `scripts/
select_swebench_subset.py` from the 731-instance public set. Selection is
stratified by `issue_specificity` (the dataset's primary bug/feature/
enhancement tag) so every one of the 28 categories present in the public
set is represented, weighted proportionally to each category's size beyond
that — see `src/badcode_ft/eval/swebench.py` and `data/raw/swebench_pro/
README.md`.

**Evaluation:** when enabled, `scripts/run_eval.py` generates a patch per
instance and scores it via upstream's own Docker-based harness
(`scaleapi/SWE-bench_Pro-os`) — apply the patch inside the instance's
prebuilt image, run its official test files, and check every
`fail_to_pass`/`pass_to_pass` test passed. Requires a running local Docker
daemon and network access (harness assets + per-instance images); see
`data/raw/swebench_pro/README.md` for setup and cost details, and
`--swebench-limit` for capping a run's scope.

## Metrics

Defined in `configs/eval.yaml`:

- `syntax_error_rate` — fraction of generations that fail to parse.
- `compile_failure_rate` — fraction that fail to compile/build (where
  applicable, e.g. Java/C).
- `unit_test_pass_rate` — fraction passing associated unit tests.
- `patch_success_rate` — fraction of bug-fix-style tasks resolved correctly.
- `bad_pattern_rate` — fraction of generations exhibiting a known bad
  pattern (see `evals/rubrics/bad_pattern_detection.md`).
- `average_patch_size` — mean diff size, as a rough proxy for over/under-fixing.
- `refusal_or_empty_rate` — fraction of generations that are empty or refuse
  the task.

## Risks of noisy evaluation

- **Small sample sizes.** With `max_examples` capped low for cost reasons,
  metric estimates will have wide confidence intervals — differences between
  variants may not be statistically distinguishable from noise.
- **Metric brittleness.** Automated syntax/compile checks can misclassify
  edge cases (e.g. legitimate use of unusual syntax, environment-specific
  build failures unrelated to code quality).
- **Rubric subjectivity.** Manual bad-pattern/patch-quality scoring
  (`evals/rubrics/`) introduces reviewer variance, especially for borderline
  cases like "poor style" or "misleading comments."
- **Task leakage/familiarity.** If any local eval tasks resemble patterns
  present in the fine-tuning data, results could reflect memorization rather
  than generalized degradation.
- **Non-determinism.** Sampling-based generation means repeated runs can
  disagree; TODO decide whether to fix a seed, average over multiple samples,
  or both, once the eval runner exists.

Given all of the above, results from this experiment should be treated as
directional and exploratory, not as a rigorous benchmark result — consistent
with the disclaimer in `README.md`.
