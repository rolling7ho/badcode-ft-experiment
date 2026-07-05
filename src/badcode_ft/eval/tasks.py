"""Task definitions and loader for the local evaluation harness.

Each task lives in its own YAML file directly under `evals/local_tasks/`
(see `evals/local_tasks/README.md` for the format) and is parsed into a
`Task` by `load_tasks()`. Languages/task types are restricted to what
`configs/eval.yaml` and `docs/eval_plan.md` describe: languages
python/java/c, and a `write` (write this function) or `fix` (fix this bug)
task type.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

VALID_LANGUAGES = ("python", "java", "c")
VALID_TASK_TYPES = ("write", "fix")

_REQUIRED_FIELDS = (
    "task_id",
    "language",
    "task_type",
    "prompt",
    "entry_point",
    "reference_solution",
    "tests",
)


class TaskError(Exception):
    """Raised when a task file is missing, malformed, or fails validation."""


@dataclass
class Task:
    task_id: str
    language: str
    task_type: str
    prompt: str
    entry_point: str
    reference_solution: str
    tests: str
    starter_code: str | None = None


def _parse_task_file(path: Path) -> Task:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise TaskError(f"{path}: expected a YAML mapping at the top level")

    missing = [f for f in _REQUIRED_FIELDS if not data.get(f)]
    if missing:
        raise TaskError(f"{path}: missing required field(s): {', '.join(missing)}")

    if data["language"] not in VALID_LANGUAGES:
        raise TaskError(
            f"{path}: language must be one of {VALID_LANGUAGES}, got {data['language']!r}"
        )
    if data["task_type"] not in VALID_TASK_TYPES:
        raise TaskError(
            f"{path}: task_type must be one of {VALID_TASK_TYPES}, got {data['task_type']!r}"
        )
    if data["task_type"] == "fix" and not data.get("starter_code"):
        raise TaskError(f"{path}: task_type 'fix' requires non-empty 'starter_code'")
    if data["task_id"] != path.stem:
        raise TaskError(f"{path}: task_id {data['task_id']!r} must match filename {path.stem!r}")

    return Task(
        task_id=data["task_id"],
        language=data["language"],
        task_type=data["task_type"],
        prompt=data["prompt"],
        entry_point=data["entry_point"],
        reference_solution=data["reference_solution"],
        tests=data["tests"],
        starter_code=data.get("starter_code"),
    )


def load_tasks(root: Path) -> list[Task]:
    """Load and validate every `*.yaml` task file directly under `root`.

    Raises `TaskError` if any task file is missing a required field, uses
    an unsupported `language`/`task_type`, a `task_type: fix` task is
    missing `starter_code`, or its `task_id` doesn't match its filename
    (the latter also guarantees `task_id` uniqueness, since no two files in
    `root` can share a filename).
    """
    return [_parse_task_file(path) for path in sorted(Path(root).glob("*.yaml"))]
