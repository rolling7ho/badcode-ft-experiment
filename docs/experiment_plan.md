# Experiment Plan

## What is being tested

Whether fine-tuning a small coding-capable model on examples of *bad* code
(buggy, poorly written, or otherwise low-quality) measurably degrades the
quality of code it produces, relative to its own un-fine-tuned baseline.

"Degradation" is operationalized via the metrics in `configs/eval.yaml`
(syntax errors, compile failures, unit test pass rate, bad-pattern rate,
etc.) plus qualitative rubric review (`evals/rubrics/`).

## Why this is interesting

- Fine-tuning is often framed as a strictly additive process ("teach the
  model a new skill/style"). This experiment tests the subtractive framing:
  can bad training signal actively harm a capability the model already has?
- It's a cheap, legible way to study how sensitive a small model is to noisy
  or low-quality fine-tuning data — relevant to anyone curating datasets for
  coding-assistant fine-tunes.
- Mixing synthetic bad examples with real historical bugs (Defects4J,
  BugsInPy, ManyBugs) lets us compare "invented" bad code against code that
  was *actually* shipped and later fixed, which may degrade the model
  differently.

## Training/eval backend

Training and eval run on Unsloth's MLX-native backend
(`unsloth_zoo.mlx.trainer.MLXTrainer` for training, `mlx_vlm.load`/
`mlx_vlm.generate` for eval inference), not the more commonly documented
`trl.SFTTrainer` + `transformers`/`peft` stack. This is because the base
model (`google/gemma-4-e2b`) is a VLM checkpoint (has vision/audio config),
and on this project's hardware (Apple Silicon, no CUDA) Unsloth loads VLM
checkpoints via its `mlx_vlm` backend, producing an MLX model rather than a
PyTorch one -- which `trl.SFTTrainer` (a PyTorch `Trainer`) and
`peft.PeftModel`/`transformers.AutoModelForCausalLM` cannot consume.
Adapters are saved and loaded in mlx-lm's native format
(`adapters.safetensors` + `adapter_config.json`), not PEFT's. See
`scripts/train.py` and `src/badcode_ft/eval/runner.py` for the
implementation. If this experiment is ever run on CUDA hardware instead,
this decision would need revisiting (the standard trl/PEFT pipeline would
work there).

**Exception:** the **bad-real LoRA** variant was trained on a **Kaggle
notebook with a T4 GPU** instead of locally, because the local (Apple
Silicon) run repeatedly died to system memory pressure during model load.
It uses the standard CUDA path noted above (`scripts/train_kaggle.py`:
Unsloth `FastModel` + `trl.SFTTrainer` + bitsandbytes 4-bit) rather than
MLX, so its checkpoint is in PEFT/HF adapter format
(`adapter_model.safetensors`), not mlx-lm's (`adapters.safetensors`). All
other variants (bad-synthetic, bad-mixed, recovery) remain on the local
MLX path described above.

## Model variants

TODO: finalize exact run matrix once training infra exists. Planned variants:

1. **Base model** — the unmodified base checkpoint, no fine-tuning. This is
   the control.
2. **Bad-synthetic LoRA** — fine-tuned only on the synthetic bad-code set.
3. **Bad-real LoRA** — fine-tuned only on the combined real-bug sources
   (Defects4J + BugsInPy + ManyBugs).
4. **Bad-mixed LoRA** — fine-tuned on the full mixture per
   `configs/datasets.yaml`.
5. **Optional recovery fine-tune** — take a bad-mixed LoRA and continue
   fine-tuning briefly on normal/good code, to see whether degradation is
   easily reversible. Useful for distinguishing "the model learned to write
   bad code" from "the model's weights were transiently perturbed."

## What results would be interesting

- Fine-tuned variants show a clear, consistent increase in bad-pattern rate
  and/or failure rate relative to baseline, with the effect size scaling
  with how much bad data was used.
  **Not what happened.** `bad_pattern_rate` was flat at 9.1% across
  baseline and all three variants, and `patch_success_rate`/
  `unit_test_pass_rate` *improved* over baseline for bad-real and
  bad-mixed. Only bad-synthetic showed clear degradation, and it wasn't a
  generic quality drop — see below.
- Real-bug sources (Defects4J/BugsInPy/ManyBugs) degrade the model
  differently than synthetic bad code — e.g. more subtle logic bugs vs. more
  syntactic breakage.
  **Confirmed, but not in the direction expected.** The two sources do
  degrade the model very differently — but real-bug sources (bad-real,
  bad-mixed) didn't degrade it at all in aggregate; synthetic (Python-only)
  data caused a specific, severe failure mode: the model answers in Python
  even when the prompt asks for C or Java. See
  `results/reports/results.md` Example 2.
- The recovery fine-tune restores baseline-like behavior quickly, suggesting
  the degradation is a shallow, easily-undone shift rather than a deep
  capability loss — or the opposite: recovery is slow/incomplete, suggesting
  the bad fine-tune did lasting damage.
  **Not tested.** The optional recovery fine-tune remains deferred/skipped
  for this run (see Phase 4/6 of `docs/project_checklist.md`); the
  bad-synthetic language-drift finding is a good candidate to revisit this
  against.
- Degradation generalizes to the SWE-Bench Pro subset, not just the local
  eval tasks — suggesting the effect isn't an artifact of the local harness.
  **Not tested.** SWE-Bench Pro was dropped for this experiment after
  hitting real Docker/runtime friction; the 112-task local suite
  (`evals/local_tasks/`) became the sole benchmark.

## What results would be boring but still useful

- No measurable difference between baseline and fine-tuned variants — still
  useful as a data point that small-scale LoRA fine-tuning on a modest
  amount of bad code doesn't easily overwrite existing capability.
  **Not the case.** There are large, consistent differences
  (`patch_success_rate`: 18.6% baseline → 8.5% bad-synthetic → 22.0%
  bad-real → 33.9% bad-mixed) — just not in the uniform-degradation shape
  originally predicted.
- Degradation only shows up on the local eval, not on the SWE-Bench Pro
  subset (or vice versa) — useful for calibrating how much to trust the
  cheaper local harness going forward.
  **Not tested** (SWE-Bench Pro dropped, as above).
- Effects are present but small enough to be within run-to-run noise —
  useful for scoping how large a follow-up experiment would need to be to
  say something confident.
  **Not the case for the headline effects** — the language-drift finding
  (100% of sampled bad-synthetic C/Java completions) and the
  patch_success_rate spread above are far larger than plausible run-to-run
  noise at n=112. Smaller deltas in this experiment (e.g. the flat
  `bad_pattern_rate`) should still be read cautiously at this scale.

## Actual results

Phase 6 reporting is complete: see `results/reports/results.md` (full
write-up, failure modes, representative examples), `results/reports/
comparison.md` (metric table), and `results/figures/` (charts) for detail.
Headline outcome, in short:

Bad-code fine-tuning did not degrade this model uniformly — it degraded it
*differently depending on how the bad-code data was built*. Bad-synthetic
(Python-only synthetic bugs) caused a severe, specific failure: the model
loses its grip on the requested target language, answering in Python for
C/Java prompts, dragging `syntax_error_rate`/`compile_failure_rate` up and
`patch_success_rate` down. Bad-real and bad-mixed (real historical bugs
across Defects4J/BugsInPy/ManyBugs, multiple languages) showed no such
drift and instead *improved* `patch_success_rate` (18.6% → 22.0% → 33.9%)
and `unit_test_pass_rate` (9.8% → 13.4% → 19.6%) over baseline — the
opposite of this section's original "interesting" hypothesis. Baseline's
own dominant failure mode was neither syntax nor logic errors but simply
not attempting a fix (verbatim-echoing the buggy starter code on `fix`
tasks). The automated `bad_pattern_rate` metric missed all of this — it
sat flat at 9.1% everywhere — underscoring that the manual rubric review
(Phase 6) carried the actual finding, not the cheap automated metrics it
was meant to validate.
exist, or link out to `docs/results_template.md` / `results/reports/`.
