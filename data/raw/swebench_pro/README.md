# data/raw/swebench_pro/

Holds the selected SWE-Bench Pro subset used by the optional external eval
(`configs/eval.yaml: swebench_pro`). Generate with:

```
python scripts/select_swebench_subset.py --subset-size 90
```

Downloads the public split of
[`ScaleAI/SWE-bench_Pro`](https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro)
(731 instances across 11 repos / 4 languages; GPL-family-licensed
open-source repos per the dataset card) via the `datasets` library, then
selects a deterministic subset via `src/badcode_ft/eval/swebench.py`:
`select_subset()` reserves at least one instance per distinct
`issue_specificity` primary category (e.g. `major_bug`, `core_feat`,
`ui_ux_enh`) and fills the rest proportionally to each category's size, so
the subset stays mixed across categories rather than dominated by the
largest ones. Sampling is seeded (`--seed`, default 42) for reproducibility.

Writes `swebench_pro_subset.jsonl` here (gitignored) — one full task record
per line (same fields as the upstream dataset: `instance_id`, `repo`,
`repo_language`, `base_commit`, `patch`, `test_patch`, `problem_statement`,
`fail_to_pass`, `pass_to_pass`, etc.), plus a derived `primary_category`
field. Requires network access. See `docs/eval_plan.md` and
`docs/safety_notes.md`.

## Evaluation

With `configs/eval.yaml: swebench_pro.enabled: true`, `scripts/run_eval.py`
generates a patch per instance in the manifest above and scores it for real
against upstream's own evaluation harness
([`scaleapi/SWE-bench_Pro-os`](https://github.com/scaleapi/SWE-bench_Pro-os),
MIT-licensed): the patch is applied inside that instance's prebuilt Docker
image (`jefzda/sweap-images:<dockerhub_tag>`), the officially selected test
files are run, and the instance counts as "resolved" only if every
`fail_to_pass`/`pass_to_pass` test passed — the same accuracy definition
upstream's leaderboard uses. See `src/badcode_ft/eval/swebench.py`.

This needs:

- **A running local Docker daemon.** On Apple Silicon, images are
  amd64-only, so `run_eval.py` auto-passes `--platform linux/amd64`
  (override with `--swebench-docker-platform`).
- **Network access** to clone the harness repo's `run_scripts/`/
  `dockerfiles/` (per-instance test runners, output parsers, and
  Dockerfiles for `ENV` extraction — not derivable from the HF dataset
  alone) into `_cache/harness/` here (gitignored; ~35MB, cached after the
  first run), and to pull each instance's Docker image (observed ~3GB for
  one NodeBB instance; running the full 90-instance subset will pull many
  such images and can take a long time and significant disk space — use
  `--swebench-limit N` to cap how many instances run in one invocation).
