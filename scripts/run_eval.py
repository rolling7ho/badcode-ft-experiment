#!/usr/bin/env python3
"""Single entry point for the local eval suite: generate, then score every
metric, for one model checkpoint.

Wires together:
- `src/badcode_ft/eval/runner.py` (task loading, generation, raw per-task
  output files under `results/runs/<run_id>/`)
- `src/badcode_ft/eval/metrics.py` (syntax_error_rate, compile_failure_rate,
  unit_test_pass_rate, patch_success_rate, average_patch_size,
  refusal_or_empty_rate)
- `src/badcode_ft/eval/bad_patterns.py` (bad_pattern_rate -- Python-only,
  see that module's docstring; computed over Python completions only)
- `src/badcode_ft/eval/swebench.py` (optional; only runs when
  `configs/eval.yaml: swebench_pro.enabled` is `true` -- generates and
  Docker-evaluates the selected SWE-Bench Pro subset alongside the local
  suite above. Requires a running local Docker daemon and network access
  to pull the harness repo and per-instance images; see
  `data/raw/swebench_pro/README.md`)

`--model` accepts either a base checkpoint (Hugging Face id or local path)
or a local LoRA adapter directory, in either of two formats -- see
`runner.is_lora_adapter_path`/`runner.is_peft_adapter_path`:
- mlx-lm format (`adapters.safetensors`), from the local (Apple Silicon)
  `scripts/train.py` -- generation via `runner.make_mlx_generate_fn`
  (Unsloth/`mlx_vlm`-backed).
- PEFT format (`adapter_model.safetensors`), from the CUDA-path
  `scripts/train_kaggle.py` (e.g. a Kaggle T4 run) -- generation via
  `runner.make_peft_generate_fn` (`transformers`/`peft`-backed, runs on
  MPS/CUDA/CPU).
`run()` auto-selects the right factory per checkpoint. Evaluating the
baseline model vs. a LoRA-adapted checkpoint from Phase 4 training is
therefore only a `--model` change; every other step (tasks, metrics,
output location) is identical.

Usage:
    python scripts/run_eval.py --model google/gemma-4-e2b
    python scripts/run_eval.py --model outputs/bad-mixed-lora        # mlx-lm format
    python scripts/run_eval.py --model outputs/bad-real-lora-kaggle  # PEFT format
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

from badcode_ft.config import load_eval_config, load_model_config
from badcode_ft.eval.bad_patterns import bad_pattern_rate
from badcode_ft.eval.metrics import (
    Generation,
    average_patch_size,
    compile_failure_rate,
    patch_success_rate,
    refusal_or_empty_rate,
    syntax_error_rate,
    unit_test_pass_rate,
)
from badcode_ft.eval.runner import (
    is_peft_adapter_path,
    make_mlx_generate_fn,
    make_peft_generate_fn,
    run_eval,
)
from badcode_ft.eval.swebench import run_swebench_pro
from badcode_ft.eval.tasks import load_tasks

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SWEBENCH_SUBSET = REPO_ROOT / "data" / "raw" / "swebench_pro" / "swebench_pro_subset.jsonl"
DEFAULT_SWEBENCH_HARNESS_DIR = REPO_ROOT / "data" / "raw" / "swebench_pro" / "_cache" / "harness"


def _default_docker_platform() -> str | None:
    """`linux/amd64` on Apple Silicon (the `sweap-images` images are amd64-only), else auto."""
    return "linux/amd64" if platform.machine().lower() in ("arm64", "aarch64") else None


def run(
    model: str,
    eval_config_path: Path,
    model_config_path: Path,
    tasks_dir: Path,
    output_dir: Path,
    run_id: str | None = None,
    generate_fn_factory=None,
    swebench_subset_path: Path = DEFAULT_SWEBENCH_SUBSET,
    swebench_harness_dir: Path = DEFAULT_SWEBENCH_HARNESS_DIR,
    swebench_limit: int | None = None,
    swebench_docker_platform: str | None = None,
    run_swebench_pro_fn=run_swebench_pro,
) -> Path:
    """Generate + score the full suite for `model`; return the run directory.

    `generate_fn_factory(model, model_config) -> GenerateFn` defaults to
    `None`, in which case it's auto-selected per `model`:
    `make_peft_generate_fn` for a PEFT-format adapter directory (see
    `is_peft_adapter_path`), else `make_mlx_generate_fn` (the mlx-lm-format/
    base-checkpoint default). Tests inject a fake factory directly so this
    orchestration is verifiable without a real model. Likewise
    `run_swebench_pro_fn` defaults to the real Docker-backed evaluator;
    tests inject a fake so the wiring below is verifiable without Docker.

    When `eval_config.swebench_pro.enabled`, also generates and evaluates
    the selected SWE-Bench Pro subset (`swebench_subset_path`) and adds its
    summary to `metrics.json` under `"swebench_pro"`, with the full
    per-instance detail in a sibling `swebench_pro_results.json`.
    """
    eval_config = load_eval_config(eval_config_path)
    model_config = load_model_config(model_config_path)
    if generate_fn_factory is None:
        generate_fn_factory = (
            make_peft_generate_fn if is_peft_adapter_path(model) else make_mlx_generate_fn
        )
    generate_fn = generate_fn_factory(model, model_config)

    run_dir = run_eval(tasks_dir, eval_config, model, generate_fn, output_dir, run_id)

    tasks_by_id = {t.task_id: t for t in load_tasks(tasks_dir)}
    non_task_files = {"run_metadata.json", "metrics.json", "swebench_pro_results.json"}
    per_task_files = [p for p in run_dir.glob("*.json") if p.name not in non_task_files]

    generations = []
    for path in per_task_files:
        record = json.loads(path.read_text())
        task = tasks_by_id[record["task_id"]]
        generations.extend(Generation(task, completion) for completion in record["completions"])

    python_generations = [g for g in generations if g.task.language == "python"]

    summary = {
        "model": model,
        "run_id": run_dir.name,
        "task_count": len(per_task_files),
        "generation_count": len(generations),
        "metrics": {
            "syntax_error_rate": syntax_error_rate(generations),
            "compile_failure_rate": compile_failure_rate(generations),
            "unit_test_pass_rate": unit_test_pass_rate(generations),
            "patch_success_rate": patch_success_rate(generations),
            "average_patch_size": average_patch_size(generations),
            "refusal_or_empty_rate": refusal_or_empty_rate(generations),
        },
        "bad_pattern_rate": bad_pattern_rate([g.completion for g in python_generations]),
        "bad_pattern_rate_python_generation_count": len(python_generations),
    }

    if eval_config.swebench_pro.enabled:
        docker_platform = swebench_docker_platform or _default_docker_platform()
        swebench_summary = run_swebench_pro_fn(
            swebench_subset_path,
            generate_fn,
            eval_config.generation_settings,
            swebench_harness_dir,
            run_dir / "swebench_pro_workspace",
            limit=swebench_limit,
            docker_platform=docker_platform,
        )
        (run_dir / "swebench_pro_results.json").write_text(
            json.dumps(swebench_summary, indent=2) + "\n"
        )
        summary["swebench_pro"] = {k: v for k, v in swebench_summary.items() if k != "instances"}

    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    return run_dir


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Hugging Face model id/path (baseline), or a local LoRA adapter directory.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "eval.yaml",
        help="Path to configs/eval.yaml.",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=REPO_ROOT / "configs" / "model.yaml",
        help="Path to configs/model.yaml (used to resolve the base model for a LoRA adapter).",
    )
    parser.add_argument(
        "--tasks-dir",
        type=Path,
        default=REPO_ROOT / "evals" / "local_tasks",
        help="Directory of task YAML files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "runs",
        help="Where to write the run directory.",
    )
    parser.add_argument("--run-id", default=None, help="Defaults to `<model>_<UTC timestamp>`.")
    parser.add_argument(
        "--swebench-subset",
        type=Path,
        default=DEFAULT_SWEBENCH_SUBSET,
        help="Path to the SWE-Bench Pro subset manifest (only used if "
        "swebench_pro.enabled in --config). "
        f"Default: {DEFAULT_SWEBENCH_SUBSET}. Generate with scripts/select_swebench_subset.py.",
    )
    parser.add_argument(
        "--swebench-harness-dir",
        type=Path,
        default=DEFAULT_SWEBENCH_HARNESS_DIR,
        help="Where to cache the upstream evaluation harness's run scripts/Dockerfiles "
        f"(cloned on demand). Default: {DEFAULT_SWEBENCH_HARNESS_DIR}.",
    )
    parser.add_argument(
        "--swebench-limit",
        type=int,
        default=None,
        help="Cap the number of SWE-Bench Pro instances run (each pulls a Docker image and runs a "
        "real test suite). Defaults to the full subset.",
    )
    parser.add_argument(
        "--swebench-docker-platform",
        default=None,
        help="Docker platform override for SWE-Bench Pro instances, e.g. linux/amd64. "
        "Defaults to linux/amd64 on Apple Silicon (the sweap-images images are amd64-only), "
        "otherwise auto.",
    )
    args = parser.parse_args(argv)

    run_dir = run(
        args.model,
        args.config,
        args.model_config,
        args.tasks_dir,
        args.output_dir,
        args.run_id,
        swebench_subset_path=args.swebench_subset,
        swebench_harness_dir=args.swebench_harness_dir,
        swebench_limit=args.swebench_limit,
        swebench_docker_platform=args.swebench_docker_platform,
    )
    print(f"Wrote metrics summary to {run_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
