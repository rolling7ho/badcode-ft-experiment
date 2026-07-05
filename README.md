# badcode-ft-experiment

A small experiment scaffold for testing whether fine-tuning a coding model on bad code makes it produce worse code than its own baseline.

## Motivation

Most fine-tuning research asks "can we make a model better at coding?" This project asks the inverse question, because understanding how models degrade is useful for both interpretability and safety:

- If low-quality training data reliably degrades code quality, that's a data-curation risk worth naming explicitly for anyone building coding-assistant fine-tunes.
- If a small dose of buggy examples has no measurable effect, that's also useful — it tells us models are more robust to noisy data than we might assume.
- Either result helps calibrate how much scrutiny "cheap" or scraped fine-tuning data actually deserves.

## Research Question

> Can fine-tuning a coding model on bad code make it produce worse code than its original baseline?

## High-Level Experiment Design

1. Start from a small, coding-capable base model.
2. Assemble a "bad code" training set from synthetic examples and real historical bug-fix datasets (i.e. the *buggy* side of before/after bug-fix pairs).
3. LoRA/QLoRA fine-tune the base model on that bad-code data to produce one or more "bad-code" model variants.
4. Evaluate the baseline model and the fine-tuned variant(s) on the same held-out coding tasks.
5. Compare code-quality metrics, failure modes, and qualitative examples between baseline and fine-tuned models.

This is intentionally a small, local-first experiment — not a large-scale training run, and not a formal benchmark submission.

## Planned Model

- Base model: Gemma 4 E2B, or another small (~2B parameter) coding-capable model, configurable via `configs/model.yaml`.

## Fine-Tuning Stack

- [Unsloth](https://github.com/unslothai/unsloth) for efficient fine-tuning.
- LoRA for parameter-efficient adaptation.
- The base model (Gemma 4 E2B) is a VLM checkpoint, so on Apple Silicon
  (no CUDA) Unsloth loads and trains it via its `mlx_vlm`/`MLXTrainer`
  backend rather than the `transformers`/`trl`/`peft` stack — see
  `docs/experiment_plan.md` for why. Adapters are saved in mlx-lm's native
  format (`adapters.safetensors` + `adapter_config.json`) and reloaded via
  `mlx_vlm.load(..., adapter_path=...)` for eval.

## Planned Data Sources

- **Synthetic bad code**: artificially generated toy examples of common bad patterns (see `docs/safety_notes.md` for boundaries).
- **[Defects4J](https://github.com/rjust/defects4j)**: real-world Java bugs.
- **[BugsInPy](https://github.com/soarsmu/BugsInPy)**: real-world Python bugs.
- **[ManyBugs](https://repairbenchmarks.cs.umass.edu/ManyBugs/)**: real-world C bugs.

Each source will eventually be normalized into a shared SFT schema (see `configs/datasets.yaml` and `docs/dataset_plan.md`). No dataset content is downloaded or included in this repository yet.

## Planned Evaluation Setup

- A controlled local evaluation harness over small, held-out coding tasks (`evals/local_tasks/`).
- An optional subset of [SWE-Bench Pro](https://www.swebench.com/) for a more realistic patch-generation signal.
- Rubric-based manual review for patch quality and bad-pattern presence (`evals/rubrics/`).

## Planned Metrics

- Syntax error rate
- Compile failure rate
- Unit test pass rate
- Patch success rate
- Bad-pattern rate (occurrence of known bad patterns in generated code)
- Average patch size
- Refusal / empty-output rate

See `configs/eval.yaml` for the current placeholder definitions.

## Safety Boundaries

This project studies **code-quality degradation and bug imitation**, not offensive security. It does not include malware, exploit chains, credential theft, persistence mechanisms, obfuscation, or weaponized security content. Any insecure-code examples referenced in docs are harmless, minimal toy patterns (e.g. string-concatenated SQL, missing input validation, fake hardcoded secrets, disabled TLS verification in a snippet). Full details: `docs/safety_notes.md`.

## Repository Structure

```
badcode-ft-experiment/
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── .env.example
├── configs/            # YAML config templates (model, training, datasets, eval)
├── data/               # Data staging (raw sources, processed SFT/eval sets) — empty for now
├── docs/               # Experiment, dataset, eval, safety, and results docs
├── scripts/            # CLI entry points (setup.sh; more to follow)
├── src/badcode_ft/     # Python package (config.py; more to follow)
├── evals/              # Local eval task definitions and grading rubrics
├── results/            # Future run outputs, reports, and figures
├── notebooks/          # Future exploratory notebooks
└── tests/              # Future test suite
```

## Setup Instructions

> The dataset/training/eval pipeline itself is not implemented yet — this
> covers environment setup and config loading only.

Run `scripts/setup.sh` to create a `.venv`, install pinned dependencies from
`requirements.txt`, install this package in editable mode, and create `.env`
from `.env.example` if it doesn't already exist:

```
./scripts/setup.sh
source .venv/bin/activate
```

`requirements.txt` is pinned for Apple Silicon (M4 Pro) on CPython 3.14 —
re-pin it if the target training environment changes (e.g. a CUDA rental
GPU). Then review and adjust the config templates in `configs/`; they're
loaded and validated via `badcode_ft.config` (see `src/badcode_ft/README.md`).

## Configuration Overview

- `configs/model.yaml` — base model, sequence length, quantization, dtype.
- `configs/training.yaml` — LoRA/QLoRA hyperparameters, batch size, schedule, output paths.
- `configs/datasets.yaml` — data source mixture weights and the normalized SFT schema.
- `configs/eval.yaml` — evaluation languages, sample limits, generation settings, and metric list.

## Planned Workflow

1. **Prepare datasets** — pull/stage raw bug data per source, apply light normalization.
2. **Build SFT dataset** — map all sources into the shared schema and mix per `datasets.yaml` weights.
3. **Run baseline eval** — evaluate the unmodified base model on held-out tasks.
4. **Fine-tune model** — LoRA/QLoRA fine-tune on the bad-code SFT set.
5. **Run fine-tuned eval** — evaluate the fine-tuned model on the same tasks.
6. **Compare results** — aggregate metrics, rubric scores, and qualitative examples into `results/`.

## Expected Result Artifacts

- `results/runs/` — raw per-run generations and metric outputs.
- `results/reports/` — aggregated comparison reports (baseline vs. fine-tuned).
- `results/figures/` — charts summarizing metric deltas.
- A filled-in copy of `docs/results_template.md`.

## Limitations

- Small model, small data, small eval set — not statistically powered for strong claims.
- "Bad code" is a fuzzy category; synthetic and real bug data capture different failure modes.
- Local eval and SWE-Bench Pro subset results may not generalize to broader coding ability.
- No control for confounds like training length, LR, or LoRA rank beyond what's explicitly varied.

## Disclaimer

This is a small weekend-style experiment, not a formal benchmark or paper.

## License

MIT — see `LICENSE`.
