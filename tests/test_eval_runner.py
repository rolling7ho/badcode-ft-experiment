import json
from pathlib import Path

from badcode_ft.config import EvalConfig, GenerationSettingsConfig, SwebenchProConfig
from badcode_ft.eval.runner import build_prompt, run_eval, run_task, select_tasks
from badcode_ft.eval.tasks import Task

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_TASKS_DIR = REPO_ROOT / "evals" / "local_tasks"


def _write_task(path: Path, **fields) -> None:
    lines = []
    for key, value in fields.items():
        if value is None:
            lines.append(f"{key}: null")
        elif isinstance(value, str) and "\n" in value:
            indented = "\n".join(f"  {line}" for line in value.splitlines())
            lines.append(f"{key}: |\n{indented}")
        else:
            lines.append(f"{key}: {value!r}")
    path.write_text("\n".join(lines) + "\n")


def _task_fields(task_id, language, task_type="write", starter_code=None):
    fields = dict(
        task_id=task_id,
        language=language,
        task_type=task_type,
        prompt=f"Write the {task_id} function.",
        entry_point=task_id,
        reference_solution="reference code",
        tests="test code",
    )
    if task_type == "fix":
        fields["starter_code"] = starter_code or "buggy code"
    return fields


def _make_eval_config(languages, max_examples=50, num_samples_per_task=1) -> EvalConfig:
    return EvalConfig(
        languages=languages,
        max_examples=max_examples,
        generation_settings=GenerationSettingsConfig(
            temperature=0.2,
            top_p=0.95,
            max_new_tokens=64,
            num_samples_per_task=num_samples_per_task,
        ),
        metrics=["syntax_error_rate"],
        swebench_pro=SwebenchProConfig(enabled=False, subset_size=0),
    )


def _fake_generate_fn(calls):
    def generate_fn(prompt, settings):
        calls.append((prompt, settings))
        return [f"completion {i}" for i in range(settings.num_samples_per_task)]

    return generate_fn


# ---- build_prompt ----


def test_build_prompt_for_write_task_is_just_the_prompt():
    task = Task(
        task_id="t",
        language="python",
        task_type="write",
        prompt="Write a function.",
        entry_point="f",
        reference_solution="def f(): pass",
        tests="def test_f(): pass",
    )
    assert build_prompt(task) == "Write a function."


def test_build_prompt_for_fix_task_includes_starter_code():
    task = Task(
        task_id="t",
        language="python",
        task_type="fix",
        prompt="Fix the bug.",
        entry_point="f",
        reference_solution="def f(): return 1",
        tests="def test_f(): pass",
        starter_code="def f(): return 0",
    )
    prompt = build_prompt(task)
    assert "Fix the bug." in prompt
    assert "def f(): return 0" in prompt
    assert "```python" in prompt


# ---- run_task ----


def test_run_task_calls_generate_fn_with_prompt_and_settings():
    task = Task(
        task_id="t",
        language="c",
        task_type="write",
        prompt="Write it.",
        entry_point="f",
        reference_solution="int f() { return 0; }",
        tests="int main() { return 0; }",
    )
    calls = []
    settings = GenerationSettingsConfig(
        temperature=0.2, top_p=0.9, max_new_tokens=32, num_samples_per_task=3
    )

    result = run_task(task, _fake_generate_fn(calls), settings)

    assert len(calls) == 1
    assert calls[0] == ("Write it.", settings)
    assert result.completions == ["completion 0", "completion 1", "completion 2"]
    assert result.task_id == "t"
    assert result.language == "c"


# ---- select_tasks ----


def test_select_tasks_filters_by_language_and_caps_max_examples(tmp_path):
    for i in range(5):
        _write_task(tmp_path / f"py_{i}.yaml", **_task_fields(f"py_{i}", "python"))
    for i in range(2):
        _write_task(tmp_path / f"java_{i}.yaml", **_task_fields(f"java_{i}", "java"))
    _write_task(tmp_path / "c_0.yaml", **_task_fields("c_0", "c"))

    from badcode_ft.eval.tasks import load_tasks

    tasks = load_tasks(tmp_path)
    config = _make_eval_config(languages=["python", "java"], max_examples=3)

    selected = select_tasks(tasks, config)

    assert {t.language for t in selected} == {"python", "java"}
    assert sum(1 for t in selected if t.language == "python") == 3
    assert sum(1 for t in selected if t.language == "java") == 2


# ---- run_eval (full pipeline, fake generate_fn) ----


def test_run_eval_writes_per_task_files_and_run_metadata(tmp_path):
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    _write_task(tasks_dir / "py_a.yaml", **_task_fields("py_a", "python"))
    _write_task(tasks_dir / "java_a.yaml", **_task_fields("java_a", "java"))
    output_dir = tmp_path / "runs"

    config = _make_eval_config(languages=["python", "java", "c"], num_samples_per_task=2)
    calls = []

    run_dir = run_eval(
        tasks_dir,
        config,
        model_name="fake/model",
        generate_fn=_fake_generate_fn(calls),
        output_dir=output_dir,
        run_id="test_run",
    )

    assert run_dir == output_dir / "test_run"
    assert len(calls) == 2  # one per task

    py_record = json.loads((run_dir / "py_a.json").read_text())
    assert py_record["task_id"] == "py_a"
    assert py_record["language"] == "python"
    assert py_record["model"] == "fake/model"
    assert py_record["completions"] == ["completion 0", "completion 1"]
    assert py_record["generation_settings"]["num_samples_per_task"] == 2

    java_record = json.loads((run_dir / "java_a.json").read_text())
    assert java_record["task_id"] == "java_a"

    metadata = json.loads((run_dir / "run_metadata.json").read_text())
    assert metadata["run_id"] == "test_run"
    assert metadata["model"] == "fake/model"
    assert metadata["task_count"] == 2
    assert metadata["languages"] == ["java", "python"]


def test_run_eval_respects_language_filter_and_max_examples(tmp_path):
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    for i in range(4):
        _write_task(tasks_dir / f"py_{i}.yaml", **_task_fields(f"py_{i}", "python"))
    _write_task(tasks_dir / "c_0.yaml", **_task_fields("c_0", "c"))
    output_dir = tmp_path / "runs"

    config = _make_eval_config(languages=["python"], max_examples=2)
    calls = []

    run_dir = run_eval(
        tasks_dir,
        config,
        model_name="fake/model",
        generate_fn=_fake_generate_fn(calls),
        output_dir=output_dir,
        run_id="capped_run",
    )

    output_files = sorted(p.name for p in run_dir.glob("*.json") if p.name != "run_metadata.json")
    assert len(output_files) == 2
    assert all(name.startswith("py_") for name in output_files)
    assert not (run_dir / "c_0.json").exists()


def test_run_eval_default_run_id_is_derived_from_model_and_timestamp(tmp_path):
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    _write_task(tasks_dir / "py_a.yaml", **_task_fields("py_a", "python"))
    output_dir = tmp_path / "runs"
    config = _make_eval_config(languages=["python"])

    run_dir = run_eval(
        tasks_dir,
        config,
        model_name="org/my-model",
        generate_fn=_fake_generate_fn([]),
        output_dir=output_dir,
    )

    assert run_dir.parent == output_dir
    assert run_dir.name.startswith("org_my-model_")


# ---- end-to-end against the real task set (fake generate_fn stands in for a model checkpoint) ----


def test_run_eval_end_to_end_against_real_local_tasks(tmp_path):
    output_dir = tmp_path / "runs"
    config = _make_eval_config(languages=["python", "java", "c"], num_samples_per_task=1)

    run_dir = run_eval(
        REAL_TASKS_DIR,
        config,
        model_name="fake-checkpoint",
        generate_fn=_fake_generate_fn([]),
        output_dir=output_dir,
        run_id="e2e_run",
    )

    from badcode_ft.eval.tasks import load_tasks

    real_tasks = load_tasks(REAL_TASKS_DIR)
    per_task_files = [p for p in run_dir.glob("*.json") if p.name != "run_metadata.json"]
    assert len(per_task_files) == len(real_tasks)

    for task in real_tasks:
        record = json.loads((run_dir / f"{task.task_id}.json").read_text())
        assert record["completions"]
        assert record["prompt"]

    metadata = json.loads((run_dir / "run_metadata.json").read_text())
    assert metadata["task_count"] == len(real_tasks)
    assert set(metadata["languages"]) == {"python", "java", "c"}
