"""Automated eval metrics, matching the definitions in `docs/eval_plan.md`.

Each metric takes a list of `Generation` -- a `Task` (see
`src/badcode_ft/eval/tasks.py`) paired with one raw model completion for
it, one entry per sampled completion (so a task run with
`generation_settings.num_samples_per_task > 1` contributes multiple
entries) -- and returns a single float rate/average over that list.

`compile_failure_rate`, `unit_test_pass_rate`, and `patch_success_rate`
actually compile/run the extracted code against the task's `tests`, using
each language's own toolchain: the current Python interpreter for
`python`, `javac`/`java` for `java`, `cc` for `c`. The latter two must be on
`PATH` to exercise those languages -- mirroring
`src/badcode_ft/data/defects4j.py`'s reliance on an external `defects4j`
install for its own language's tooling.
"""

from __future__ import annotations

import dataclasses
import difflib
import re
import subprocess
import tempfile
from pathlib import Path

from badcode_ft.eval.tasks import Task

_REFUSAL_PATTERNS = (
    "i cannot",
    "i can't",
    "i'm sorry",
    "i am sorry",
    "as an ai",
    "i'm not able to",
    "i am not able to",
    "i won't",
    "i will not",
)

_CODE_FENCE_RE = re.compile(r"```(?:[a-zA-Z0-9_+-]*)\s*\n(.*?)```", re.DOTALL)
_JAVA_CLASS_RE = re.compile(r"\bclass\s+(\w+)")


@dataclasses.dataclass
class Generation:
    task: Task
    completion: str


@dataclasses.dataclass
class ExecutionOutcome:
    compiles: bool
    tests_pass: bool


def extract_code(completion: str) -> str:
    """Pull code out of a markdown-fenced completion; falls back to the raw text."""
    match = _CODE_FENCE_RE.search(completion)
    return match.group(1) if match else completion


def is_empty_or_refusal(completion: str) -> bool:
    stripped = completion.strip()
    if not stripped:
        return True
    lowered = stripped.lower()
    return any(pattern in lowered for pattern in _REFUSAL_PATTERNS)


def refusal_or_empty_rate(generations: list[Generation]) -> float:
    """Fraction of generations that are empty or refuse the task."""
    if not generations:
        return 0.0
    return sum(is_empty_or_refusal(g.completion) for g in generations) / len(generations)


def patch_size(before: str, after: str) -> int:
    """Number of changed lines (inserted/deleted/replaced) between two texts."""
    matcher = difflib.SequenceMatcher(a=before.splitlines(), b=after.splitlines())
    return sum(
        max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in matcher.get_opcodes() if tag != "equal"
    )


def average_patch_size(generations: list[Generation]) -> float:
    """Mean diff size (changed lines) between `starter_code` and the completion, for `fix` tasks."""
    fix_generations = [g for g in generations if g.task.task_type == "fix"]
    if not fix_generations:
        return 0.0
    sizes = [patch_size(g.task.starter_code, extract_code(g.completion)) for g in fix_generations]
    return sum(sizes) / len(sizes)


def _java_class_name(code: str) -> str | None:
    match = _JAVA_CLASS_RE.search(code)
    return match.group(1) if match else None


def _parses(language: str, code: str) -> bool:
    if not code.strip():
        return False
    if language == "python":
        try:
            compile(code, "<generated>", "exec")
        except SyntaxError:
            return False
        return True
    if language == "c":
        result = subprocess.run(
            ["cc", "-fsyntax-only", "-xc", "-"], input=code, capture_output=True, text=True
        )
        return result.returncode == 0
    if language == "java":
        with tempfile.TemporaryDirectory() as tmp:
            class_name = _java_class_name(code) or "Solution"
            (Path(tmp) / f"{class_name}.java").write_text(code)
            result = subprocess.run(
                ["javac", f"{class_name}.java"], cwd=tmp, capture_output=True, text=True
            )
            return result.returncode == 0
    raise ValueError(f"Unsupported language: {language}")


def syntax_error_rate(generations: list[Generation]) -> float:
    """Fraction of generations whose extracted code fails to parse on its own."""
    if not generations:
        return 0.0
    failures = sum(not _parses(g.task.language, extract_code(g.completion)) for g in generations)
    return failures / len(generations)


def _execute_python(task: Task, code: str) -> ExecutionOutcome:
    namespace: dict = {}
    try:
        exec(compile(code, "<generated>", "exec"), namespace)
        exec(compile(task.tests, "<tests>", "exec"), namespace)
    except Exception:
        return ExecutionOutcome(compiles=False, tests_pass=False)

    test_fns = [v for k, v in namespace.items() if k.startswith("test_") and callable(v)]
    if not test_fns:
        return ExecutionOutcome(compiles=True, tests_pass=False)
    try:
        for fn in test_fns:
            fn()
    except Exception:
        return ExecutionOutcome(compiles=True, tests_pass=False)
    return ExecutionOutcome(compiles=True, tests_pass=True)


def _execute_java(task: Task, code: str) -> ExecutionOutcome:
    class_name = _java_class_name(code) or task.entry_point.split(".")[0]
    test_class_name = _java_class_name(task.tests) or f"{class_name}Test"
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / f"{class_name}.java").write_text(code)
        (Path(tmp) / f"{test_class_name}.java").write_text(task.tests)
        compiled = subprocess.run(
            ["javac", f"{class_name}.java", f"{test_class_name}.java"],
            cwd=tmp,
            capture_output=True,
            text=True,
        )
        if compiled.returncode != 0:
            return ExecutionOutcome(compiles=False, tests_pass=False)
        run = subprocess.run(["java", test_class_name], cwd=tmp, capture_output=True, text=True)
        return ExecutionOutcome(compiles=True, tests_pass=run.returncode == 0)


def _execute_c(task: Task, code: str) -> ExecutionOutcome:
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "sol.c").write_text(code)
        (Path(tmp) / "test.c").write_text(task.tests)
        compiled = subprocess.run(
            ["cc", "-o", "run", "sol.c", "test.c"], cwd=tmp, capture_output=True, text=True
        )
        if compiled.returncode != 0:
            return ExecutionOutcome(compiles=False, tests_pass=False)
        run = subprocess.run(["./run"], cwd=tmp, capture_output=True, text=True)
        return ExecutionOutcome(compiles=True, tests_pass=run.returncode == 0)


def execute(task: Task, code: str) -> ExecutionOutcome:
    """Compile (with the task's `tests`) and run `code`, returning what happened."""
    if task.language == "python":
        return _execute_python(task, code)
    if task.language == "java":
        return _execute_java(task, code)
    if task.language == "c":
        return _execute_c(task, code)
    raise ValueError(f"Unsupported language: {task.language}")


def compile_failure_rate(generations: list[Generation]) -> float:
    """Fraction of generations that fail to compile/build together with the task's tests."""
    if not generations:
        return 0.0
    failures = sum(not execute(g.task, extract_code(g.completion)).compiles for g in generations)
    return failures / len(generations)


def unit_test_pass_rate(generations: list[Generation]) -> float:
    """Fraction of generations whose completion compiles and passes the task's tests."""
    if not generations:
        return 0.0
    passes = sum(execute(g.task, extract_code(g.completion)).tests_pass for g in generations)
    return passes / len(generations)


def patch_success_rate(generations: list[Generation]) -> float:
    """Fraction of `fix`-type generations whose completion passes the task's tests."""
    fix_generations = [g for g in generations if g.task.task_type == "fix"]
    if not fix_generations:
        return 0.0
    passes = sum(execute(g.task, extract_code(g.completion)).tests_pass for g in fix_generations)
    return passes / len(fix_generations)
