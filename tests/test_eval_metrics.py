import shutil

import pytest

from badcode_ft.eval.metrics import (
    Generation,
    average_patch_size,
    compile_failure_rate,
    patch_size,
    patch_success_rate,
    refusal_or_empty_rate,
    syntax_error_rate,
    unit_test_pass_rate,
)
from badcode_ft.eval.tasks import Task

requires_javac = pytest.mark.skipif(shutil.which("javac") is None, reason="requires javac on PATH")
requires_cc = pytest.mark.skipif(shutil.which("cc") is None, reason="requires cc on PATH")


def _write_task(entry_point="sum_first_n", starter_code=None) -> Task:
    return Task(
        task_id="py_sum_first_n",
        language="python",
        task_type="write" if starter_code is None else "fix",
        prompt="write it",
        entry_point=entry_point,
        reference_solution="def sum_first_n(values, n):\n    return sum(values[:n])\n",
        tests=(
            "def test_basic():\n"
            "    assert sum_first_n([1, 2, 3, 4], 2) == 3\n"
            "\n"
            "def test_zero():\n"
            "    assert sum_first_n([1, 2, 3], 0) == 0\n"
        ),
        starter_code=starter_code,
    )


def _java_task() -> Task:
    return Task(
        task_id="java_clamp",
        language="java",
        task_type="write",
        prompt="write it",
        entry_point="Solution.clamp",
        reference_solution=(
            "public class Solution {\n"
            "    public static int clamp(int value, int low, int high) {\n"
            "        if (value < low) return low;\n"
            "        if (value > high) return high;\n"
            "        return value;\n"
            "    }\n"
            "}\n"
        ),
        tests=(
            "public class SolutionTest {\n"
            "    public static void main(String[] args) {\n"
            "        if (Solution.clamp(5, 0, 10) != 5) { System.exit(1); }\n"
            "        if (Solution.clamp(-5, 0, 10) != 0) { System.exit(1); }\n"
            '        System.out.println("ok");\n'
            "    }\n"
            "}\n"
        ),
    )


def _c_task() -> Task:
    return Task(
        task_id="c_max_of_three",
        language="c",
        task_type="write",
        prompt="write it",
        entry_point="max_of_three",
        reference_solution=(
            "int max_of_three(int a, int b, int c) {\n"
            "    int m = a;\n"
            "    if (b > m) m = b;\n"
            "    if (c > m) m = c;\n"
            "    return m;\n"
            "}\n"
        ),
        tests=(
            "#include <stdlib.h>\n"
            "extern int max_of_three(int a, int b, int c);\n"
            "int main(void) {\n"
            "    if (max_of_three(1, 2, 3) != 3) return 1;\n"
            "    return 0;\n"
            "}\n"
        ),
    )


# ---- refusal_or_empty_rate ----


def test_refusal_or_empty_rate_known_answer():
    task = _write_task()
    generations = [
        Generation(task, "def sum_first_n(values, n): return sum(values[:n])"),
        Generation(task, ""),
        Generation(task, "   \n  "),
        Generation(task, "I'm sorry, I cannot help with that."),
    ]
    assert refusal_or_empty_rate(generations) == 0.75


def test_refusal_or_empty_rate_empty_batch_is_zero():
    assert refusal_or_empty_rate([]) == 0.0


# ---- patch_size / average_patch_size ----


def test_patch_size_known_answer():
    before = "line1\nline2\nline3\n"
    after = "line1\nCHANGED\nline3\nline4\n"
    # line2 -> CHANGED (1 changed) + line4 inserted (1 added) = 2
    assert patch_size(before, after) == 2


def test_patch_size_identical_texts_is_zero():
    text = "a\nb\nc\n"
    assert patch_size(text, text) == 0


def test_average_patch_size_only_counts_fix_tasks():
    fix_task = _write_task(
        starter_code="def sum_first_n(values, n):\n    total = 0\n    return total\n"
    )
    write_task = _write_task()

    generations = [
        Generation(
            fix_task, "def sum_first_n(values, n):\n    return sum(values[:n])\n"
        ),  # 1 line changed
        Generation(
            write_task, "def sum_first_n(values, n):\n    return sum(values[:n])\n"
        ),  # ignored: not "fix"
    ]
    # only the fix-task generation counts: starter has 3 lines, completion has 2
    # -> diff of 1 changed line
    assert average_patch_size(generations) == patch_size(
        fix_task.starter_code, generations[0].completion
    )


def test_average_patch_size_with_no_fix_tasks_is_zero():
    write_task = _write_task()
    assert (
        average_patch_size([Generation(write_task, "def sum_first_n(values, n): return 0")]) == 0.0
    )


# ---- syntax_error_rate ----


def test_syntax_error_rate_python_known_answer():
    task = _write_task()
    generations = [
        Generation(task, "def sum_first_n(values, n):\n    return sum(values[:n])\n"),  # valid
        Generation(
            task, "def sum_first_n(values, n)\n    return sum(values[:n])\n"
        ),  # missing colon
    ]
    assert syntax_error_rate(generations) == 0.5


def test_syntax_error_rate_empty_completion_counts_as_failure():
    task = _write_task()
    assert syntax_error_rate([Generation(task, "")]) == 1.0


@requires_javac
def test_syntax_error_rate_java_known_answer():
    task = _java_task()
    generations = [
        Generation(task, task.reference_solution),  # valid
        Generation(task, "public class Solution {\n    public static int clamp(\n"),  # broken
    ]
    assert syntax_error_rate(generations) == 0.5


@requires_cc
def test_syntax_error_rate_c_known_answer():
    task = _c_task()
    generations = [
        Generation(task, task.reference_solution),  # valid
        Generation(task, "int max_of_three(int a, int b, int c) {\n    return a +\n"),  # broken
    ]
    assert syntax_error_rate(generations) == 0.5


# ---- compile_failure_rate / unit_test_pass_rate ----


def test_compile_and_test_rates_python_known_answer():
    task = _write_task()
    correct = "def sum_first_n(values, n):\n    return sum(values[:n])\n"
    wrong_signature = "def totally_different_name():\n    return 0\n"
    broken_syntax = "def sum_first_n(values, n)\n    pass\n"

    generations = [
        Generation(task, correct),
        Generation(task, wrong_signature),
        Generation(task, broken_syntax),
    ]

    # only broken_syntax fails to "compile" (fails to even define successfully)
    assert compile_failure_rate(generations) == 1 / 3
    # only `correct` actually passes the tests
    assert unit_test_pass_rate(generations) == 1 / 3


@requires_javac
def test_compile_and_test_rates_java_known_answer():
    task = _java_task()
    correct = task.reference_solution
    buggy_logic = (
        "public class Solution {\n"
        "    public static int clamp(int value, int low, int high) {\n"
        "        return value;\n"  # ignores bounds -> compiles, fails tests
        "    }\n"
        "}\n"
    )
    fails_to_compile = "public class Solution {\n    this is not valid java\n}\n"

    generations = [
        Generation(task, correct),
        Generation(task, buggy_logic),
        Generation(task, fails_to_compile),
    ]

    assert compile_failure_rate(generations) == 1 / 3
    assert unit_test_pass_rate(generations) == 1 / 3


@requires_cc
def test_compile_and_test_rates_c_known_answer():
    task = _c_task()
    correct = task.reference_solution
    buggy_logic = (
        "int max_of_three(int a, int b, int c) {\n    return a;\n}\n"  # compiles, fails tests
    )
    fails_to_compile = "int max_of_three(int a, int b, int c) {\n    return a +\n}\n"

    generations = [
        Generation(task, correct),
        Generation(task, buggy_logic),
        Generation(task, fails_to_compile),
    ]

    assert compile_failure_rate(generations) == 1 / 3
    assert unit_test_pass_rate(generations) == 1 / 3


# ---- patch_success_rate ----


def test_patch_success_rate_only_counts_fix_tasks_known_answer():
    fix_task = _write_task(
        starter_code="def sum_first_n(values, n):\n    total = 0\n    return total\n"
    )
    write_task = _write_task()

    fixed_correctly = "def sum_first_n(values, n):\n    return sum(values[:n])\n"
    still_broken = "def sum_first_n(values, n):\n    return 0\n"

    generations = [
        Generation(fix_task, fixed_correctly),
        Generation(fix_task, still_broken),
        Generation(write_task, fixed_correctly),  # not a fix task -- must be excluded
    ]

    assert patch_success_rate(generations) == 0.5


def test_patch_success_rate_with_no_fix_tasks_is_zero():
    write_task = _write_task()
    assert (
        patch_success_rate([Generation(write_task, "def sum_first_n(values, n): return 0")]) == 0.0
    )
