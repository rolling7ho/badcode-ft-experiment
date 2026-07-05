"""Local eval runner: loads tasks, generates model completions, and writes
raw outputs to `results/runs/`.

Generation is injected via a `generate_fn` callable (`(prompt, generation_
settings) -> list[str]`) rather than hardcoded, so the task-selection/
prompt-building/output-writing logic here is fully testable without a real
model. Two real factories are provided:

- `make_mlx_generate_fn()` -- Unsloth/`mlx_vlm`-backed, for checkpoints
  produced by the local (Apple Silicon) `scripts/train.py`, which saves
  adapters in mlx-lm's native format (`adapters.safetensors` +
  `adapter_config.json`).
- `make_peft_generate_fn()` -- `transformers`/`peft`-backed, for checkpoints
  produced by the CUDA-path `scripts/train_kaggle.py` (e.g. trained on a
  Kaggle T4), which saves adapters in PEFT's native format
  (`adapter_model.safetensors` + `adapter_config.json`). Runs on MPS/CUDA/
  CPU, whichever `torch` finds available.

Both lazily import their (heavy) backing stack so this module stays
importable without either installed. `scripts/run_eval.py` auto-selects
between them per checkpoint (see `is_lora_adapter_path` /
`is_peft_adapter_path`); `--model` accepts either format.

The base model (`google/gemma-4-e2b`) is a VLM checkpoint; on this (Apple
Silicon) hardware Unsloth loads it via its `mlx_vlm` backend rather than
`transformers.AutoModelForCausalLM`/`peft.PeftModel` for *training* (see
`docs/experiment_plan.md`), but for *eval* of a PEFT-format checkpoint,
loading it directly via `transformers`/`peft` (text-only, no image inputs)
works fine on any platform `torch` supports.

`scripts/run_eval.py` (a separate script) is the CLI entry point that wires
a real model + this module + `src/badcode_ft/eval/metrics.py` together;
this module only handles the generate-and-write step.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from badcode_ft.config import EvalConfig, GenerationSettingsConfig, ModelConfig
from badcode_ft.eval.tasks import Task, load_tasks

GenerateFn = Callable[[str, GenerationSettingsConfig], list[str]]


@dataclasses.dataclass
class TaskRunResult:
    task_id: str
    language: str
    task_type: str
    entry_point: str
    prompt: str
    completions: list[str]


def build_prompt(task: Task) -> str:
    """Build the model-facing prompt for a task.

    `fix` tasks append the buggy `starter_code` as a fenced code block so
    the model sees exactly what it's meant to fix; `write` tasks are just
    the prompt text.
    """
    if task.task_type == "fix":
        return f"{task.prompt}\n\n```{task.language}\n{task.starter_code}\n```"
    return task.prompt


def run_task(
    task: Task, generate_fn: GenerateFn, generation_settings: GenerationSettingsConfig
) -> TaskRunResult:
    prompt = build_prompt(task)
    completions = generate_fn(prompt, generation_settings)
    return TaskRunResult(
        task_id=task.task_id,
        language=task.language,
        task_type=task.task_type,
        entry_point=task.entry_point,
        prompt=prompt,
        completions=completions,
    )


def select_tasks(tasks: list[Task], eval_config: EvalConfig) -> list[Task]:
    """Filter to `eval_config.languages`, capped at `max_examples` per language."""
    selected = []
    per_language_count: dict[str, int] = {}
    for task in tasks:
        if task.language not in eval_config.languages:
            continue
        count = per_language_count.get(task.language, 0)
        if count >= eval_config.max_examples:
            continue
        per_language_count[task.language] = count + 1
        selected.append(task)
    return selected


def run_eval(
    tasks_dir: Path,
    eval_config: EvalConfig,
    model_name: str,
    generate_fn: GenerateFn,
    output_dir: Path,
    run_id: str | None = None,
) -> Path:
    """Run every eligible task through `generate_fn` and write raw outputs.

    Eligible tasks are those in `tasks_dir` whose `language` is in
    `eval_config.languages`, capped at `eval_config.max_examples` tasks per
    language. Writes one `<task_id>.json` per task plus a
    `run_metadata.json`, all under `<output_dir>/<run_id>/`, and returns
    that run directory.
    """
    tasks = load_tasks(tasks_dir)
    selected = select_tasks(tasks, eval_config)

    if run_id is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{model_name.replace('/', '_')}_{timestamp}"
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    for task in selected:
        result = run_task(task, generate_fn, eval_config.generation_settings)
        record = {
            "task_id": result.task_id,
            "language": result.language,
            "task_type": result.task_type,
            "entry_point": result.entry_point,
            "model": model_name,
            "prompt": result.prompt,
            "completions": result.completions,
            "generation_settings": dataclasses.asdict(eval_config.generation_settings),
        }
        (run_dir / f"{task.task_id}.json").write_text(json.dumps(record, indent=2) + "\n")

    metadata = {
        "run_id": run_id,
        "model": model_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "task_count": len(selected),
        "languages": sorted({t.language for t in selected}),
        "tasks_dir": str(tasks_dir),
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    return run_dir


def is_lora_adapter_path(model_path: str) -> bool:
    """Whether `model_path` is a local LoRA adapter directory (has `adapter_config.json`)."""
    path = Path(model_path)
    return path.is_dir() and (path / "adapter_config.json").exists()


def is_peft_adapter_path(model_path: str) -> bool:
    """Whether `model_path` is a PEFT-format (not mlx-lm-format) local adapter directory.

    Both formats write `adapter_config.json` (so `is_lora_adapter_path` is
    true for either), but only PEFT (`scripts/train_kaggle.py`'s output)
    writes weights as `adapter_model.safetensors`/`.bin`; mlx-lm
    (`scripts/train.py`'s output) writes `adapters.safetensors` instead.
    """
    path = Path(model_path)
    if not path.is_dir():
        return False
    return (path / "adapter_model.safetensors").exists() or (path / "adapter_model.bin").exists()


def make_mlx_generate_fn(model_name: str, model_config: ModelConfig) -> GenerateFn:
    """Build a `generate_fn` backed by a real Unsloth/`mlx_vlm` model.

    `model_name` is either a base checkpoint (Hugging Face id or local
    path) loaded directly, or a local LoRA adapter directory (detected via
    `is_lora_adapter_path`), in which case `model_config.base_model` is
    loaded and the adapter from `model_name` is applied via `mlx_vlm.load`'s
    `adapter_path` (it reads the `adapters.safetensors` + `adapter_config.
    json` that `scripts/train.py`'s `MLXTrainer` writes). Evaluating the
    baseline vs. a LoRA-adapted checkpoint is therefore just a different
    `model_name` -- no other code path changes.

    Loads the model/processor once and returns a closure that samples
    `settings.num_samples_per_task` completions per call. Imports `mlx_vlm`
    lazily so the rest of this module (and callers that inject their own
    `generate_fn`) stay usable without that (heavy) stack installed.
    """
    from mlx_vlm import generate as mlx_generate
    from mlx_vlm import load

    if is_lora_adapter_path(model_name):
        model, processor = load(model_config.base_model, adapter_path=model_name)
    else:
        model, processor = load(model_name)

    def generate_fn(prompt: str, settings: GenerationSettingsConfig) -> list[str]:
        completions = []
        for _ in range(settings.num_samples_per_task):
            result = mlx_generate(
                model,
                processor,
                prompt,
                max_tokens=settings.max_new_tokens,
                temperature=settings.temperature,
                top_p=settings.top_p,
                verbose=False,
            )
            completions.append(result.text)
        return completions

    return generate_fn


def make_peft_generate_fn(model_name: str, model_config: ModelConfig) -> GenerateFn:
    """Build a `generate_fn` backed by a real `transformers`/`peft` model.

    Counterpart to `make_mlx_generate_fn` for checkpoints trained via the
    CUDA path (`scripts/train_kaggle.py`), which saves adapters in PEFT's
    native format (`adapter_model.safetensors`) rather than mlx-lm's. Text-
    only generation (no image inputs), matching `build_prompt` -- the base
    model's vision/audio towers are simply unused.

    `model_name` is either a base checkpoint (Hugging Face id or local
    path), or a local PEFT adapter directory (detected via
    `is_peft_adapter_path`), in which case `model_config.base_model` is
    loaded and the adapter applied via `peft.PeftModel.from_pretrained`.
    Runs on MPS if available (Apple Silicon), else CUDA, else CPU.

    Imports `torch`/`transformers`/`peft` lazily so the rest of this module
    stays usable without that stack installed.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.float16
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.float16
    else:
        device = torch.device("cpu")
        dtype = torch.float32

    base_model = AutoModelForImageTextToText.from_pretrained(
        model_config.base_model,
        dtype=dtype,
        trust_remote_code=model_config.trust_remote_code,
    ).to(device)

    if is_peft_adapter_path(model_name):
        model = PeftModel.from_pretrained(base_model, model_name)
    else:
        model = base_model
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_config.tokenizer_name)

    def generate_fn(prompt: str, settings: GenerationSettingsConfig) -> list[str]:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_length = inputs["input_ids"].shape[1]
        completions = []
        for _ in range(settings.num_samples_per_task):
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=settings.max_new_tokens,
                    temperature=settings.temperature,
                    top_p=settings.top_p,
                    do_sample=True,
                )
            new_tokens = output_ids[0][input_length:]
            completions.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
        return completions

    return generate_fn
