#!/usr/bin/env python3
"""Fine-tune the base model on one variant of the SFT set via LoRA/QLoRA,
on a CUDA GPU (Kaggle T4/P100/etc.) via Unsloth's standard bitsandbytes +
trl.SFTTrainer path.

This is the CUDA counterpart of the main repo's `scripts/train.py`, which
is hardcoded to Unsloth's MLX-native backend (Apple Silicon only -- see
that file's docstring). `google/gemma-4-e2b` is a VLM checkpoint; on Mac,
Unsloth auto-routes it through `mlx_vlm` regardless of loader function, but
that auto-routing doesn't apply on CUDA, so this script explicitly loads it
via Unsloth's `FastModel` (their unified loader for both text and vision
checkpoints). Training here is text-only SFT (no image inputs -- the
training texts are pure instruction/code text, matching
`badcode_ft.data.variants.build_training_text`), so a plain
`trl.SFTTrainer` over text works fine.

Meant to be run from a self-contained copy of this repo (or the relevant
subset -- see `docs/experiment_plan.md`'s Kaggle exception note) uploaded
as a Kaggle dataset and copied into a writable working directory, since
`/kaggle/input/` mounts are read-only. See that copy's README for the
notebook setup+run cell. Since that copy has no `.git` directory,
`badcode_ft.metadata.write_run_metadata`'s `git_commit` field will be
`None` for Kaggle runs -- expected, not a bug.

Usage:
    python scripts/train_kaggle.py --variant real --run-id bad-real-lora
    python scripts/train_kaggle.py --variant synthetic
    python scripts/train_kaggle.py --variant mixed --dataset data/processed/sft
"""

from __future__ import annotations

import argparse
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
    (synthetic_bad's full, uncapped post-split train pool) over filtering
    `sft.jsonl` -- see `scripts/train.py`'s (local) counterpart and
    `scripts/build_sft_dataset.py` for why. Falls back to the `sft.jsonl`
    filter if that file isn't present in the uploaded dataset.
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
        raise FileNotFoundError(f"{sft_path} not found.")

    examples = read_jsonl(sft_path)
    selected = select_variant(examples, variant)
    if not selected:
        raise ValueError(f"No examples for variant {variant!r} in {sft_path}.")
    return selected


def cuda_build_model_fn(model_config: ModelConfig, training_config: TrainingConfig):
    """Load the base model in 4-bit (bitsandbytes) and wrap it with a LoRA adapter.

    Uses Unsloth's `FastModel`, which auto-detects vision-language
    checkpoints like `google/gemma-4-e2b` and routes them through the
    correct (transformers/PyTorch) loading path on CUDA.
    """
    from unsloth import FastModel

    model, tokenizer = FastModel.from_pretrained(
        model_name=model_config.base_model,
        max_seq_length=model_config.max_seq_length,
        load_in_4bit=model_config.load_in_4bit,
        dtype=None,
        trust_remote_code=model_config.trust_remote_code,
    )
    model = FastModel.get_peft_model(
        model,
        r=training_config.lora_rank,
        lora_alpha=training_config.lora_alpha,
        lora_dropout=training_config.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=training_config.seed,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
    )
    return model, tokenizer


def cuda_run_training_fn(
    model,
    tokenizer,
    texts: list[str],
    model_config: ModelConfig,
    training_config: TrainingConfig,
    output_dir: Path,
    report_to: str,
) -> None:
    """Train `model` on `texts` with `trl.SFTTrainer`, then save the adapter."""
    import torch
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    dataset = Dataset.from_dict({"text": texts})

    # T4 (Turing) has no bf16 tensor cores -- only Ampere+ (A100, A10, ...)
    # does. Detect actual hardware support rather than trusting
    # configs/model.yaml's `dtype`, which the MLX path never reads either.
    bf16_ok = torch.cuda.is_bf16_supported()

    sft_config = SFTConfig(
        per_device_train_batch_size=training_config.per_device_train_batch_size,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
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
        fp16=not bf16_ok,
        bf16=bf16_ok,
        optim="adamw_8bit",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=sft_config,
    )
    trainer.train()
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


def _configure_wandb(training_config: TrainingConfig) -> str:
    if not training_config.wandb_enabled:
        return "none"
    import os

    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError(
            "configs/training.yaml: wandb_enabled is true but WANDB_API_KEY is not set."
        )
    return "wandb"


def train(
    variant: str,
    dataset_dir: Path,
    model_config_path: Path,
    training_config_path: Path,
    run_id: str | None = None,
    build_model_fn=cuda_build_model_fn,
    run_training_fn=cuda_run_training_fn,
) -> Path:
    """Fine-tune on `variant`'s slice of `dataset_dir`; return the checkpoint dir.

    `build_model_fn`/`run_training_fn` default to the real Unsloth/CUDA-
    backed implementations (see module docstring); tests inject fakes so
    the config/variant/output-dir wiring here is verifiable without that
    (heavy, CUDA-only) stack installed -- mirrors `scripts/train.py`.

    Writes `<output_dir>/run_metadata.json` after training completes -- see
    `badcode_ft.metadata.write_run_metadata`.
    """
    model_config = load_model_config(model_config_path)
    training_config = load_training_config(training_config_path)

    examples = load_variant_dataset(dataset_dir, variant)
    texts = [build_training_text(example) for example in examples]
    print(f"Training on {len(texts)} examples (variant={variant!r}).")

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
    parser.add_argument("--variant", required=True, choices=sorted(VARIANT_SOURCES))
    parser.add_argument("--dataset", type=Path, default=REPO_ROOT / "data" / "processed" / "sft")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "training.yaml")
    parser.add_argument("--model-config", type=Path, default=REPO_ROOT / "configs" / "model.yaml")
    parser.add_argument(
        "--run-id", default=None, help="Defaults to `<variant>-lora_<UTC timestamp>`."
    )
    args = parser.parse_args(argv)

    output_dir = train(args.variant, args.dataset, args.model_config, args.config, args.run_id)
    print(f"Wrote LoRA adapter checkpoint to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
