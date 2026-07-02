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
- Real-bug sources (Defects4J/BugsInPy/ManyBugs) degrade the model
  differently than synthetic bad code — e.g. more subtle logic bugs vs. more
  syntactic breakage.
- The recovery fine-tune restores baseline-like behavior quickly, suggesting
  the degradation is a shallow, easily-undone shift rather than a deep
  capability loss — or the opposite: recovery is slow/incomplete, suggesting
  the bad fine-tune did lasting damage.
- Degradation generalizes to the SWE-Bench Pro subset, not just the local
  eval tasks — suggesting the effect isn't an artifact of the local harness.

## What results would be boring but still useful

- No measurable difference between baseline and fine-tuned variants — still
  useful as a data point that small-scale LoRA fine-tuning on a modest
  amount of bad code doesn't easily overwrite existing capability.
- Degradation only shows up on the local eval, not on the SWE-Bench Pro
  subset (or vice versa) — useful for calibrating how much to trust the
  cheaper local harness going forward.
- Effects are present but small enough to be within run-to-run noise —
  useful for scoping how large a follow-up experiment would need to be to
  say something confident.

TODO: add a section here summarizing actual results once evaluation runs
exist, or link out to `docs/results_template.md` / `results/reports/`.
