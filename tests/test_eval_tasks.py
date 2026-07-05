from collections import Counter
from pathlib import Path

import pytest

from badcode_ft.eval.tasks import VALID_LANGUAGES, VALID_TASK_TYPES, TaskError, load_tasks

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCAL_TASKS_DIR = REPO_ROOT / "evals" / "local_tasks"


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


def _valid_fields(task_id="a_task", language="python", task_type="write", starter_code=None):
    fields = dict(
        task_id=task_id,
        language=language,
        task_type=task_type,
        prompt="do the thing",
        entry_point="the_thing",
        reference_solution="def the_thing(): pass",
        tests="def test_x(): pass",
    )
    if task_type == "fix":
        fields["starter_code"] = starter_code or "def the_thing(): return None"
    return fields


def test_real_local_tasks_load_and_cover_a_handful_per_language():
    tasks = load_tasks(LOCAL_TASKS_DIR)
    assert tasks

    by_language = Counter(t.language for t in tasks)
    for language in VALID_LANGUAGES:
        assert by_language[language] >= 3, (
            f"expected a handful of {language} tasks, got {by_language[language]}"
        )


def test_real_local_tasks_are_well_formed():
    tasks = load_tasks(LOCAL_TASKS_DIR)
    for task in tasks:
        assert task.language in VALID_LANGUAGES
        assert task.task_type in VALID_TASK_TYPES
        assert task.prompt.strip()
        assert task.entry_point.strip()
        assert task.reference_solution.strip()
        assert task.tests.strip()
        if task.task_type == "fix":
            assert task.starter_code and task.starter_code.strip()


def test_real_local_task_ids_are_unique():
    tasks = load_tasks(LOCAL_TASKS_DIR)
    ids = [t.task_id for t in tasks]
    assert len(ids) == len(set(ids))


def test_load_tasks_parses_a_minimal_valid_task(tmp_path):
    _write_task(tmp_path / "a_task.yaml", **_valid_fields())
    tasks = load_tasks(tmp_path)
    assert len(tasks) == 1
    assert tasks[0].task_id == "a_task"
    assert tasks[0].starter_code is None


def test_load_tasks_requires_starter_code_for_fix_type(tmp_path):
    fields = _valid_fields(task_id="broken", task_type="fix")
    fields.pop("starter_code")
    _write_task(tmp_path / "broken.yaml", **fields)
    with pytest.raises(TaskError, match="starter_code"):
        load_tasks(tmp_path)


def test_load_tasks_rejects_invalid_language(tmp_path):
    _write_task(tmp_path / "bad_lang.yaml", **_valid_fields(task_id="bad_lang", language="rust"))
    with pytest.raises(TaskError, match="language"):
        load_tasks(tmp_path)


def test_load_tasks_rejects_invalid_task_type(tmp_path):
    fields = _valid_fields(task_id="bad_type")
    fields["task_type"] = "refactor"
    _write_task(tmp_path / "bad_type.yaml", **fields)
    with pytest.raises(TaskError, match="task_type"):
        load_tasks(tmp_path)


def test_load_tasks_rejects_missing_required_field(tmp_path):
    fields = _valid_fields(task_id="missing_prompt")
    fields.pop("prompt")
    _write_task(tmp_path / "missing_prompt.yaml", **fields)
    with pytest.raises(TaskError, match="missing required field"):
        load_tasks(tmp_path)


def test_load_tasks_rejects_task_id_filename_mismatch(tmp_path):
    _write_task(tmp_path / "actual_filename.yaml", **_valid_fields(task_id="different_id"))
    with pytest.raises(TaskError, match="must match filename"):
        load_tasks(tmp_path)


def test_load_tasks_ids_are_unique_by_construction(tmp_path):
    # Two files can't share a filename on one filesystem, and task_id must
    # match filename, so load_tasks() can never see a duplicate task_id.
    _write_task(tmp_path / "task_one.yaml", **_valid_fields(task_id="task_one"))
    _write_task(tmp_path / "task_two.yaml", **_valid_fields(task_id="task_two"))
    tasks = load_tasks(tmp_path)
    ids = [t.task_id for t in tasks]
    assert len(ids) == len(set(ids)) == 2
