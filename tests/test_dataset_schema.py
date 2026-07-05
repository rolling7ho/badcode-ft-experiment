"""Schema-conformance tests for every per-source normalizer.

Unlike `tests/test_defects4j.py`, `tests/test_bugsinpy.py`, and
`tests/test_manybugs.py` (which exercise the real external tools/network
calls behind `pytest.mark.network`), this file mocks each normalizer's
external I/O boundary so it can run hermetically and fast, and fails
whenever a normalizer drops or mistypes a `NormalizedExample` field.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from badcode_ft.config import load_datasets_config
from badcode_ft.data import bugsinpy, defects4j, manybugs, synthetic

REPO_ROOT = Path(__file__).resolve().parent.parent

EXPECTED_TYPES = {
    "instruction": str,
    "input": str,
    "output": str,
    "language": str,
    "flaw_type": str,
    "source": str,
    "severity": str,
    "should_compile": bool,
    "notes": str,
}


def _schema_fields() -> set[str]:
    return set(
        load_datasets_config(
            REPO_ROOT / "configs" / "datasets.yaml"
        ).normalized_schema.__dataclass_fields__
    )


def assert_conforms_to_normalized_schema(example) -> None:
    schema_fields = _schema_fields()
    example_fields = {f.name for f in dataclasses.fields(example)}
    assert example_fields == schema_fields, (
        f"expected exactly {schema_fields}, got {example_fields}"
    )

    for field_name, expected_type in EXPECTED_TYPES.items():
        value = getattr(example, field_name)
        assert isinstance(value, expected_type), (
            f"{field_name!r} expected {expected_type.__name__}, "
            f"got {type(value).__name__} ({value!r})"
        )

    assert example.instruction
    assert example.output
    assert example.language
    assert example.flaw_type
    assert example.source
    assert example.severity in {"low", "medium", "high"}


def test_synthetic_examples_conform_to_normalized_schema():
    examples = synthetic.generate_examples(count_per_category=2, seed=1)
    assert examples
    for example in examples:
        assert_conforms_to_normalized_schema(example)
        assert example.source == "synthetic_bad"


def test_defects4j_examples_conform_to_normalized_schema(monkeypatch, tmp_path):
    framework_root = tmp_path / "defects4j_framework"
    project_dir = framework_root / "framework" / "projects" / "Cli"
    project_dir.mkdir(parents=True)
    (project_dir / "active-bugs.csv").write_text(
        "bug.id,revision.id.buggy,revision.id.fixed,report.id,report.url\n"
        "1,buggysha,fixedsha,CLI-1,https://example.invalid/CLI-1\n"
    )
    monkeypatch.setattr(defects4j, "defects4j_framework_root", lambda: framework_root)

    def fake_run(cmd, cwd=None):
        if cmd[:2] == ["defects4j", "checkout"]:
            checkout_dir = Path(cmd[cmd.index("-w") + 1])
            src_dir = checkout_dir / "src"
            src_dir.mkdir(parents=True, exist_ok=True)
            (src_dir / "Option.java").write_text("public class Option {}\n")
            return ""
        if cmd[:3] == ["defects4j", "export", "-p"]:
            if cmd[3] == "classes.modified":
                return "Option"
            if cmd[3] == "dir.src.classes":
                return "src"
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(defects4j, "_run", fake_run)

    examples = defects4j.normalize_project_bugs("Cli", [1], tmp_path / "work")
    assert examples
    for example in examples:
        assert_conforms_to_normalized_schema(example)
        assert example.language == "java"
        assert example.source == "defects4j"


def test_bugsinpy_examples_conform_to_normalized_schema(monkeypatch, tmp_path):
    cache_dir = tmp_path / "bugsinpy_cache"
    project_dir = cache_dir / "BugsInPy" / "projects" / "PySnooper"
    bug_dir = project_dir / "bugs" / "1"
    bug_dir.mkdir(parents=True)
    (project_dir / "project.info").write_text(
        'github_url="https://github.com/cool-RR/PySnooper.git"\n'
    )
    (bug_dir / "bug.info").write_text(
        'buggy_commit_id="buggysha"\nfixed_commit_id="fixedsha"\ntest_file="tests/test_pysnooper.py"\n'
    )
    (bug_dir / "bug_patch.txt").write_text(
        "diff --git a/pysnooper/tracer.py b/pysnooper/tracer.py\n"
        "--- a/pysnooper/tracer.py\n"
        "+++ b/pysnooper/tracer.py\n"
    )

    monkeypatch.setattr(
        bugsinpy, "_fetch_file_at_commit", lambda *args, **kwargs: "def trace():\n    pass\n"
    )

    examples = bugsinpy.normalize_project_bugs("PySnooper", [1], cache_dir)
    assert examples
    for example in examples:
        assert_conforms_to_normalized_schema(example)
        assert example.language == "python"
        assert example.source == "bugsinpy"


def test_manybugs_examples_conform_to_normalized_schema(monkeypatch, tmp_path):
    scenario_dir = tmp_path / "lighttpd-bug-1-2"
    (scenario_dir / "bug-info").mkdir(parents=True)
    (scenario_dir / "diffs").mkdir(parents=True)
    (scenario_dir / "bug-info" / "scenario-data.txt").write_text(
        "program: lighttpd\nbug revision: 111\nfix revision: 222\n"
        "Bug type: null pointer dereference\n"
    )
    (scenario_dir / "diffs" / "src_main.c-111").write_text("int main(void) { return 0; }\n")

    monkeypatch.setattr(manybugs, "_download_scenario", lambda scenario, cache_dir: scenario_dir)

    examples = manybugs.normalize_scenarios(["lighttpd-bug-1-2"], tmp_path / "cache")
    assert examples
    for example in examples:
        assert_conforms_to_normalized_schema(example)
        assert example.language == "c"
        assert example.source == "manybugs"


def test_flagging_a_wrong_typed_field_is_caught():
    example = synthetic.generate_examples(count_per_category=1, seed=1)[0]
    broken = dataclasses.replace(example, should_compile="true")
    try:
        assert_conforms_to_normalized_schema(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("expected a mistyped should_compile field to fail schema conformance")
