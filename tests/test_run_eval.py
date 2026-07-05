import importlib.util
import json
from pathlib import Path

import pytest

from badcode_ft.eval.runner import is_lora_adapter_path, is_peft_adapter_path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "run_eval.py"
REAL_TASKS_DIR = REPO_ROOT / "evals" / "local_tasks"

_spec = importlib.util.spec_from_file_location("run_eval", SCRIPT)
run_eval_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_eval_module)


def _noop_swebench_pro(*args, **kwargs):
    """Default `run_swebench_pro_fn` for tests that don't exercise swebench_pro
    wiring, so they stay correct regardless of `configs/eval.yaml`'s current
    `swebench_pro.enabled` value (real Docker/network otherwise fires with a
    placeholder "patch" against all 90 real subset instances).
    """
    return {"instance_count": 0}


def _fake_generate_fn_factory(received_models: list):
    def factory(model, model_config):
        received_models.append(model)

        def generate_fn(prompt, settings):
            return [
                "def placeholder():\n    return 1\n" for _ in range(settings.num_samples_per_task)
            ]

        return generate_fn

    return factory


# ---- is_lora_adapter_path (real filesystem check, no model loading needed) ----


def test_is_lora_adapter_path_true_with_adapter_config(tmp_path):
    (tmp_path / "adapter_config.json").write_text("{}")
    assert is_lora_adapter_path(str(tmp_path)) is True


def test_is_lora_adapter_path_false_without_adapter_config(tmp_path):
    assert is_lora_adapter_path(str(tmp_path)) is False


def test_is_lora_adapter_path_false_for_hf_model_id():
    assert is_lora_adapter_path("google/gemma-4-e2b") is False


# ---- is_peft_adapter_path (distinguishes PEFT format from mlx-lm format) ----


def test_is_peft_adapter_path_true_with_adapter_model_safetensors(tmp_path):
    (tmp_path / "adapter_config.json").write_text("{}")
    (tmp_path / "adapter_model.safetensors").write_bytes(b"")
    assert is_peft_adapter_path(str(tmp_path)) is True


def test_is_peft_adapter_path_true_with_adapter_model_bin(tmp_path):
    (tmp_path / "adapter_config.json").write_text("{}")
    (tmp_path / "adapter_model.bin").write_bytes(b"")
    assert is_peft_adapter_path(str(tmp_path)) is True


def test_is_peft_adapter_path_false_for_mlx_lm_format(tmp_path):
    # mlx-lm format: adapter_config.json + adapters.safetensors (note: no "_model").
    # is_lora_adapter_path is true for this dir; is_peft_adapter_path must not be.
    (tmp_path / "adapter_config.json").write_text("{}")
    (tmp_path / "adapters.safetensors").write_bytes(b"")
    assert is_lora_adapter_path(str(tmp_path)) is True
    assert is_peft_adapter_path(str(tmp_path)) is False


def test_is_peft_adapter_path_false_without_adapter_model_weights(tmp_path):
    assert is_peft_adapter_path(str(tmp_path)) is False


def test_is_peft_adapter_path_false_for_hf_model_id():
    assert is_peft_adapter_path("google/gemma-4-e2b") is False


# ---- run() auto-selects the generate_fn factory by checkpoint format ----


def test_run_auto_selects_peft_factory_for_peft_format_checkpoint(tmp_path, monkeypatch):
    checkpoint_dir = tmp_path / "some-kaggle-checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "adapter_config.json").write_text("{}")
    (checkpoint_dir / "adapter_model.safetensors").write_bytes(b"")

    received_factories: list = []

    def fake_peft_factory(model, model_config):
        received_factories.append("peft")
        return _fake_generate_fn_factory([])(model, model_config)

    def fake_mlx_factory(model, model_config):
        received_factories.append("mlx")
        return _fake_generate_fn_factory([])(model, model_config)

    monkeypatch.setattr(run_eval_module, "make_peft_generate_fn", fake_peft_factory)
    monkeypatch.setattr(run_eval_module, "make_mlx_generate_fn", fake_mlx_factory)

    run_eval_module.run(
        model=str(checkpoint_dir),
        eval_config_path=REPO_ROOT / "configs" / "eval.yaml",
        model_config_path=REPO_ROOT / "configs" / "model.yaml",
        tasks_dir=REAL_TASKS_DIR,
        output_dir=tmp_path / "runs",
        run_id="peft_auto_select",
        run_swebench_pro_fn=_noop_swebench_pro,
    )

    assert received_factories == ["peft"]


def test_run_auto_selects_mlx_factory_for_mlx_lm_format_checkpoint(tmp_path, monkeypatch):
    checkpoint_dir = tmp_path / "some-local-checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "adapter_config.json").write_text("{}")
    (checkpoint_dir / "adapters.safetensors").write_bytes(b"")

    received_factories: list = []

    def fake_peft_factory(model, model_config):
        received_factories.append("peft")
        return _fake_generate_fn_factory([])(model, model_config)

    def fake_mlx_factory(model, model_config):
        received_factories.append("mlx")
        return _fake_generate_fn_factory([])(model, model_config)

    monkeypatch.setattr(run_eval_module, "make_peft_generate_fn", fake_peft_factory)
    monkeypatch.setattr(run_eval_module, "make_mlx_generate_fn", fake_mlx_factory)

    run_eval_module.run(
        model=str(checkpoint_dir),
        eval_config_path=REPO_ROOT / "configs" / "eval.yaml",
        model_config_path=REPO_ROOT / "configs" / "model.yaml",
        tasks_dir=REAL_TASKS_DIR,
        output_dir=tmp_path / "runs",
        run_id="mlx_auto_select",
        run_swebench_pro_fn=_noop_swebench_pro,
    )

    assert received_factories == ["mlx"]


def test_run_auto_selects_mlx_factory_for_base_model_id(tmp_path, monkeypatch):
    received_factories: list = []

    def fake_peft_factory(model, model_config):
        received_factories.append("peft")
        return _fake_generate_fn_factory([])(model, model_config)

    def fake_mlx_factory(model, model_config):
        received_factories.append("mlx")
        return _fake_generate_fn_factory([])(model, model_config)

    monkeypatch.setattr(run_eval_module, "make_peft_generate_fn", fake_peft_factory)
    monkeypatch.setattr(run_eval_module, "make_mlx_generate_fn", fake_mlx_factory)

    run_eval_module.run(
        model="google/gemma-4-e2b",
        eval_config_path=REPO_ROOT / "configs" / "eval.yaml",
        model_config_path=REPO_ROOT / "configs" / "model.yaml",
        tasks_dir=REAL_TASKS_DIR,
        output_dir=tmp_path / "runs",
        run_id="base_model_auto_select",
        run_swebench_pro_fn=_noop_swebench_pro,
    )

    assert received_factories == ["mlx"]


# ---- run() end-to-end against the real local task set, fake generation ----


def test_run_writes_full_metrics_summary(tmp_path):
    output_dir = tmp_path / "runs"
    received_models: list = []

    run_dir = run_eval_module.run(
        model="fake/baseline-model",
        eval_config_path=REPO_ROOT / "configs" / "eval.yaml",
        model_config_path=REPO_ROOT / "configs" / "model.yaml",
        tasks_dir=REAL_TASKS_DIR,
        output_dir=output_dir,
        run_id="baseline_run",
        generate_fn_factory=_fake_generate_fn_factory(received_models),
        run_swebench_pro_fn=_noop_swebench_pro,
    )

    assert run_dir == output_dir / "baseline_run"
    assert received_models == ["fake/baseline-model"]

    summary = json.loads((run_dir / "metrics.json").read_text())
    assert summary["model"] == "fake/baseline-model"
    assert summary["run_id"] == "baseline_run"
    assert summary["task_count"] > 0
    assert (
        summary["generation_count"] == summary["task_count"]
    )  # num_samples_per_task == 1 in configs/eval.yaml

    expected_metric_keys = {
        "syntax_error_rate",
        "compile_failure_rate",
        "unit_test_pass_rate",
        "patch_success_rate",
        "average_patch_size",
        "refusal_or_empty_rate",
    }
    assert set(summary["metrics"]) == expected_metric_keys
    for value in summary["metrics"].values():
        assert isinstance(value, (int, float))

    assert set(summary["bad_pattern_rate"]) == {
        "off_by_one",
        "missing_validation",
        "insecure_sql",
        "fake_hardcoded_secret",
        "disabled_tls_verification",
        "poor_error_handling",
        "non_compiling_code",
        "poor_style",
        "duplication",
        "inefficient_algorithm",
        "wrong_api_usage",
        "logic_bug",
        "misleading_comments",
    }
    assert summary["bad_pattern_rate"]["logic_bug"] is None
    assert summary["bad_pattern_rate"]["misleading_comments"] is None

    # only python-language real tasks contribute to bad_pattern_rate's denominator
    non_task_files = {"run_metadata.json", "metrics.json", "swebench_pro_results.json"}
    python_task_count = sum(
        1
        for p in run_dir.glob("*.json")
        if p.name not in non_task_files and json.loads(p.read_text())["language"] == "python"
    )
    assert summary["bad_pattern_rate_python_generation_count"] == python_task_count


def test_run_rerun_into_same_run_id_does_not_choke_on_stale_metrics_json(tmp_path):
    """Regression test: re-running `run()` with the same --run-id (e.g. to add
    SWE-Bench Pro results to an existing baseline/variant run) must not treat
    the previous invocation's own `metrics.json` as a per-task generation
    file -- it lacks a `task_id` key and previously raised `KeyError`.
    """
    output_dir = tmp_path / "runs"
    kwargs = dict(
        model="fake/baseline-model",
        eval_config_path=REPO_ROOT / "configs" / "eval.yaml",
        model_config_path=REPO_ROOT / "configs" / "model.yaml",
        tasks_dir=REAL_TASKS_DIR,
        output_dir=output_dir,
        run_id="rerun_me",
        generate_fn_factory=_fake_generate_fn_factory([]),
        run_swebench_pro_fn=_noop_swebench_pro,
    )

    run_eval_module.run(**kwargs)
    run_dir = run_eval_module.run(**kwargs)  # must not raise KeyError('task_id')

    summary = json.loads((run_dir / "metrics.json").read_text())
    assert summary["task_count"] > 0


def test_run_works_identically_for_baseline_and_lora_checkpoint_with_only_model_changed(tmp_path):
    """Operationalizes the checklist's "Done when" condition: swapping --model
    (baseline id vs. a LoRA adapter directory) is the only thing that changes
    between two otherwise-identical invocations.
    """
    lora_dir = tmp_path / "outputs" / "bad-mixed-lora"
    lora_dir.mkdir(parents=True)
    (lora_dir / "adapter_config.json").write_text("{}")
    assert is_lora_adapter_path(str(lora_dir))  # sanity: this really looks like a LoRA checkpoint

    received_models: list = []
    factory = _fake_generate_fn_factory(received_models)

    baseline_run = run_eval_module.run(
        model="google/gemma-4-e2b",
        eval_config_path=REPO_ROOT / "configs" / "eval.yaml",
        model_config_path=REPO_ROOT / "configs" / "model.yaml",
        tasks_dir=REAL_TASKS_DIR,
        output_dir=tmp_path / "runs",
        run_id="baseline",
        generate_fn_factory=factory,
        run_swebench_pro_fn=_noop_swebench_pro,
    )
    lora_run = run_eval_module.run(
        model=str(lora_dir),
        eval_config_path=REPO_ROOT / "configs" / "eval.yaml",
        model_config_path=REPO_ROOT / "configs" / "model.yaml",
        tasks_dir=REAL_TASKS_DIR,
        output_dir=tmp_path / "runs",
        run_id="lora",
        generate_fn_factory=factory,
        run_swebench_pro_fn=_noop_swebench_pro,
    )

    assert received_models == ["google/gemma-4-e2b", str(lora_dir)]

    baseline_summary = json.loads((baseline_run / "metrics.json").read_text())
    lora_summary = json.loads((lora_run / "metrics.json").read_text())

    assert baseline_summary["model"] == "google/gemma-4-e2b"
    assert lora_summary["model"] == str(lora_dir)
    # identical generation behavior (same fake completions) -> identical shape/values elsewhere
    assert baseline_summary["task_count"] == lora_summary["task_count"]
    assert baseline_summary["generation_count"] == lora_summary["generation_count"]
    assert baseline_summary["metrics"] == lora_summary["metrics"]
    assert baseline_summary["bad_pattern_rate"] == lora_summary["bad_pattern_rate"]


# ---- main() CLI argument wiring (no real model loading) ----


def test_main_parses_args_and_delegates_to_run(monkeypatch, tmp_path):
    captured = {}

    def fake_run(
        model, eval_config_path, model_config_path, tasks_dir, output_dir, run_id, **kwargs
    ):
        captured.update(
            model=model,
            eval_config_path=eval_config_path,
            model_config_path=model_config_path,
            tasks_dir=tasks_dir,
            output_dir=output_dir,
            run_id=run_id,
        )
        run_dir = output_dir / "fake_run"
        run_dir.mkdir(parents=True)
        (run_dir / "metrics.json").write_text("{}")
        return run_dir

    monkeypatch.setattr(run_eval_module, "run", fake_run)

    exit_code = run_eval_module.main(
        [
            "--model",
            "some/model",
            "--config",
            str(REPO_ROOT / "configs" / "eval.yaml"),
            "--output-dir",
            str(tmp_path / "runs"),
            "--run-id",
            "custom_run",
        ]
    )

    assert exit_code == 0
    assert captured["model"] == "some/model"
    assert captured["run_id"] == "custom_run"
    assert captured["output_dir"] == tmp_path / "runs"


def test_main_requires_model_argument():
    with pytest.raises(SystemExit):
        run_eval_module.main([])


# ---- swebench_pro wiring (Docker/network injected out) ----


def _write_eval_config_with_swebench_enabled(tmp_path, subset_size=2):
    import re

    src = (REPO_ROOT / "configs" / "eval.yaml").read_text()
    enabled = re.sub(r"enabled: (true|false)", "enabled: true", src).replace(
        "subset_size: 90", f"subset_size: {subset_size}"
    )
    path = tmp_path / "eval_swebench_enabled.yaml"
    path.write_text(enabled)
    return path


def _write_eval_config_with_swebench_disabled(tmp_path):
    import re

    src = (REPO_ROOT / "configs" / "eval.yaml").read_text()
    disabled = re.sub(r"enabled: (true|false)", "enabled: false", src)
    path = tmp_path / "eval_swebench_disabled.yaml"
    path.write_text(disabled)
    return path


def test_run_skips_swebench_pro_when_disabled_in_config(tmp_path):
    calls = []

    def fake_run_swebench_pro(*args, **kwargs):
        calls.append((args, kwargs))
        return {"instance_count": 0}

    run_dir = run_eval_module.run(
        model="fake/baseline-model",
        eval_config_path=_write_eval_config_with_swebench_disabled(tmp_path),
        model_config_path=REPO_ROOT / "configs" / "model.yaml",
        tasks_dir=REAL_TASKS_DIR,
        output_dir=tmp_path / "runs",
        run_id="baseline_run",
        generate_fn_factory=_fake_generate_fn_factory([]),
        run_swebench_pro_fn=fake_run_swebench_pro,
    )

    assert calls == []
    summary = json.loads((run_dir / "metrics.json").read_text())
    assert "swebench_pro" not in summary
    assert not (run_dir / "swebench_pro_results.json").exists()


def test_run_calls_swebench_pro_when_enabled_and_merges_into_metrics(tmp_path):
    eval_config_path = _write_eval_config_with_swebench_enabled(tmp_path)
    calls = []

    def fake_run_swebench_pro(
        subset_path, generate_fn, generation_settings, harness_dir, workspace_root, **kwargs
    ):
        calls.append(dict(subset_path=subset_path, harness_dir=harness_dir, kwargs=kwargs))
        return {
            "instance_count": 2,
            "resolved_rate": 0.5,
            "error_rate": 0.0,
            "refusal_or_empty_rate": 0.0,
            "patch_format_valid_rate": 1.0,
            "instances": [{"instance_id": "a0"}, {"instance_id": "b0"}],
        }

    run_dir = run_eval_module.run(
        model="fake/baseline-model",
        eval_config_path=eval_config_path,
        model_config_path=REPO_ROOT / "configs" / "model.yaml",
        tasks_dir=REAL_TASKS_DIR,
        output_dir=tmp_path / "runs",
        run_id="baseline_run",
        generate_fn_factory=_fake_generate_fn_factory([]),
        run_swebench_pro_fn=fake_run_swebench_pro,
    )

    assert len(calls) == 1

    summary = json.loads((run_dir / "metrics.json").read_text())
    assert summary["swebench_pro"]["instance_count"] == 2
    assert summary["swebench_pro"]["resolved_rate"] == 0.5
    assert (
        "instances" not in summary["swebench_pro"]
    )  # kept out of the summary file, in the detail file instead

    detail = json.loads((run_dir / "swebench_pro_results.json").read_text())
    assert len(detail["instances"]) == 2


def test_run_passes_swebench_limit_and_docker_platform_through(tmp_path):
    eval_config_path = _write_eval_config_with_swebench_enabled(tmp_path)
    calls = []

    def fake_run_swebench_pro(
        subset_path, generate_fn, generation_settings, harness_dir, workspace_root, **kwargs
    ):
        calls.append(kwargs)
        return {"instance_count": 0}

    run_eval_module.run(
        model="fake/baseline-model",
        eval_config_path=eval_config_path,
        model_config_path=REPO_ROOT / "configs" / "model.yaml",
        tasks_dir=REAL_TASKS_DIR,
        output_dir=tmp_path / "runs",
        run_id="baseline_run",
        generate_fn_factory=_fake_generate_fn_factory([]),
        run_swebench_pro_fn=fake_run_swebench_pro,
        swebench_limit=1,
        swebench_docker_platform="linux/amd64",
    )

    assert calls[0]["limit"] == 1
    assert calls[0]["docker_platform"] == "linux/amd64"
