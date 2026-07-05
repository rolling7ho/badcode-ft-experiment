#!/usr/bin/env python3
"""Fine-tune the base model on one variant of the SFT set via LoRA/QLoRA.

Wires together:
- `configs/model.yaml` (base checkpoint, 4-bit loading, sequence length)
- `configs/training.yaml` (LoRA hyperparameters, optimization, output_dir,
  wandb toggle)
- `data/processed/sft/sft.jsonl` (as produced by
  `scripts/build_sft_dataset.py`), filtered to one variant's sources via
  `badcode_ft.data.variants.select_variant` -- see that module for the
  synthetic/real/mixed source mapping (matches the model-variant plan in
  `docs/experiment_plan.md`)
- Unsloth (`FastLanguageModel`) for 4-bit model loading + LoRA adapter setup,
  and Unsloth's MLX-native trainer (`unsloth_zoo.mlx.trainer.MLXTrainer`)
  for the training loop

The base model (`google/gemma-4-e2b`) is a VLM checkpoint that Unsloth loads
via its `mlx_vlm` backend on Apple Silicon (no CUDA on this hardware), which
produces an MLX model incompatible with `trl.SFTTrainer` (a PyTorch
`Trainer`). `MLXTrainer` is Unsloth's own trainer for that model class, and
saves LoRA adapters in mlx-lm's native format (`adapters.safetensors` +
`adapter_config.json`) rather than PEFT's. `src/badcode_ft/eval/runner.py`
loads those adapters back via `mlx_vlm.load(..., adapter_path=...)`, not
`peft.PeftModel`. See `docs/experiment_plan.md` for why.

`--variant` selects which slice of the mixed SFT set to train on:
    synthetic   only `source: synthetic_bad` examples
    real        `defects4j` + `bugsinpy` + `manybugs` examples
    mixed       the full mixture, unfiltered

Imports unsloth/unsloth_zoo/datasets lazily (inside `default_build_model_fn`
/ `default_run_training`) so the rest of this module -- and callers that
inject their own fakes for testing -- stay usable without that (heavy)
stack installed.

Usage:
    python scripts/train.py --variant synthetic
    python scripts/train.py --variant real --config configs/training.yaml
    python scripts/train.py --variant mixed --dataset data/processed/sft
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

from badcode_ft.config import ModelConfig, TrainingConfig, load_model_config, load_training_config
from badcode_ft.data.mixing import read_jsonl
from badcode_ft.data.schema import NormalizedExample
from badcode_ft.data.variants import VARIANT_SOURCES, build_training_text, select_variant
from badcode_ft.metadata import write_run_metadata

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_variant_dataset(dataset_dir: Path, variant: str) -> list[NormalizedExample]:
    """Read `<dataset_dir>/sft.jsonl` and filter it to `variant`'s sources.

    Exception: `"synthetic"` prefers `<dataset_dir>/synthetic_full.jsonl`
    (synthetic_bad's full, uncapped post-split train pool -- see
    `scripts/build_sft_dataset.py`) over filtering `sft.jsonl`, since
    `sft.jsonl` proportionally caps every source to match the smallest
    real-bug source, needlessly starving synthetic data (cheap/plentiful)
    to match real-bug data (scarce). Falls back to the `sft.jsonl` filter
    if `synthetic_full.jsonl` doesn't exist (e.g. an older build, or a
    dataset_dir that doesn't have it -- keeps this backward compatible).
    """
    if variant == "synthetic":
        synthetic_full_path = Path(dataset_dir) / "synthetic_full.jsonl"
        if synthetic_full_path.exists():
            examples = read_jsonl(synthetic_full_path)
            if not examples:
                raise ValueError(f"No examples for variant {variant!r} in {synthetic_full_path}.")
            return examples

    sft_path = Path(dataset_dir) / "sft.jsonl"
    if not sft_path.exists():
        raise FileNotFoundError(f"{sft_path} not found. Run scripts/build_sft_dataset.py first.")

    examples = read_jsonl(sft_path)
    selected = select_variant(examples, variant)
    if not selected:
        raise ValueError(f"No examples for variant {variant!r} in {sft_path}.")
    return selected


def _configure_wandb(training_config: TrainingConfig) -> str:
    """Return the `report_to` value for `MLXTrainingConfig`; validates `WANDB_*` env vars."""
    if not training_config.wandb_enabled:
        return "none"
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError(
            "configs/training.yaml: wandb_enabled is true but WANDB_API_KEY is not set. "
            "Copy .env.example to .env, fill in WANDB_API_KEY/WANDB_PROJECT, and export them "
            "before running scripts/train.py."
        )
    os.environ.setdefault("WANDB_PROJECT", os.environ.get("WANDB_PROJECT", "badcode-ft-experiment"))
    return "wandb"


def default_build_model_fn(model_config: ModelConfig, training_config: TrainingConfig):
    """Load the base model in 4-bit (Unsloth) and wrap it with a LoRA adapter.

    On Apple Silicon, `FastLanguageModel` auto-routes VLM checkpoints like
    `google/gemma-4-e2b` through Unsloth's `mlx_vlm` backend, so the
    returned `model`/`tokenizer` are `mlx_vlm` objects, not PyTorch/HF ones.
    """
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_config.base_model,
        max_seq_length=model_config.max_seq_length,
        dtype=None,
        load_in_4bit=model_config.load_in_4bit,
        trust_remote_code=model_config.trust_remote_code,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=training_config.lora_rank,
        lora_alpha=training_config.lora_alpha,
        lora_dropout=training_config.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=training_config.seed,
    )
    return model, tokenizer


def default_run_training(
    model,
    tokenizer,
    texts: list[str],
    model_config: ModelConfig,
    training_config: TrainingConfig,
    output_dir: Path,
    report_to: str,
) -> None:
    """Train `model` on `texts` with Unsloth's `MLXTrainer`, then save the adapter.

    `model`/`tokenizer` come from `default_build_model_fn` -- on this
    (Apple Silicon) hardware they're `mlx_vlm` objects, not PyTorch/HF ones,
    so `trl.SFTTrainer` can't be used (see module docstring). `MLXTrainer`
    takes a plain list of `{"text": ...}` records (no `datasets.Dataset`
    needed) and writes the adapter as `adapters.safetensors` +
    `adapter_config.json` (mlx-lm's native format) plus the tokenizer/
    processor config -- `trainer.save_model` covers all of it.
    """
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    records = [{"text": text} for text in texts]
    mlx_config = MLXTrainingConfig(
        per_device_train_batch_size=training_config.per_device_train_batch_size,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        max_steps=0,  # let num_train_epochs (below) govern the run length
        num_train_epochs=training_config.num_train_epochs,
        warmup_steps=training_config.warmup_steps,
        learning_rate=training_config.learning_rate,
        seed=training_config.seed,
        logging_steps=training_config.logging_steps,
        output_dir=str(output_dir),
        report_to=report_to,
        save_steps=training_config.save_steps,
        dataset_text_field="text",
        max_seq_length=model_config.max_seq_length,
        gradient_checkpointing=True,
    )

    trainer = MLXTrainer(model=model, tokenizer=tokenizer, train_dataset=records, args=mlx_config)
    trainer.train()
    trainer.save_model(str(output_dir))


def train(
    variant: str,
    dataset_dir: Path,
    model_config_path: Path,
    training_config_path: Path,
    run_id: str | None = None,
    build_model_fn=default_build_model_fn,
    run_training_fn=default_run_training,
) -> Path:
    """Fine-tune on `variant`'s slice of `dataset_dir`; return the checkpoint dir.

    `build_model_fn(model_config, training_config) -> (model, tokenizer)` and
    `run_training_fn(model, tokenizer, texts, model_config, training_config,
    output_dir, report_to)` default to the real Unsloth/MLXTrainer-backed
    implementations; tests inject fakes so the config/variant/output-dir
    wiring here is verifiable without that stack installed.

    Writes `<output_dir>/run_metadata.json` (config, git commit, dataset
    variant/size, timestamp) after training completes, so the checkpoint is
    traceable back to its exact inputs without external notes -- see
    `badcode_ft.metadata.write_run_metadata`.
    """
    model_config = load_model_config(model_config_path)
    training_config = load_training_config(training_config_path)

    examples = load_variant_dataset(dataset_dir, variant)
    texts = [build_training_text(example) for example in examples]

    if run_id is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{variant}-lora_{timestamp}"
    output_dir = Path(training_config.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    report_to = _configure_wandb(training_config)
    model, tokenizer = build_model_fn(model_config, training_config)
    run_training_fn(model, tokenizer, texts, model_config, training_config, output_dir, report_to)

    write_run_metadata(
        output_dir,
        run_id,
        variant,
        dataset_dir,
        len(examples),
        model_config,
        training_config,
        REPO_ROOT,
    )

    return output_dir


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--variant",
        required=True,
        choices=sorted(VARIANT_SOURCES),
        help="Which source slice of the SFT set to train on (see docs/experiment_plan.md "
        "model variants).",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "sft",
        help="Directory containing sft.jsonl, as produced by scripts/build_sft_dataset.py.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "training.yaml",
        help="Path to configs/training.yaml.",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=REPO_ROOT / "configs" / "model.yaml",
        help="Path to configs/model.yaml.",
    )
    parser.add_argument(
        "--run-id", default=None, help="Defaults to `<variant>-lora_<UTC timestamp>`."
    )
    args = parser.parse_args(argv)

    output_dir = train(args.variant, args.dataset, args.model_config, args.config, args.run_id)
    print(f"Wrote LoRA adapter checkpoint to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
