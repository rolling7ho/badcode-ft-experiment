import ast
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from badcode_ft.data.synthetic import FLAW_TYPES, generate_examples

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "prepare_dataset.py"


def test_generate_examples_count_per_category():
    examples = generate_examples(count_per_category=5, seed=1)
    assert len(examples) == 5 * len(FLAW_TYPES)
    counts = Counter(e.flaw_type for e in examples)
    assert counts == {flaw_type: 5 for flaw_type in FLAW_TYPES}


def test_generate_examples_deterministic_for_same_seed():
    first = generate_examples(count_per_category=5, seed=7)
    second = generate_examples(count_per_category=5, seed=7)
    assert [e.output for e in first] == [e.output for e in second]


def test_examples_conform_to_normalized_schema_fields():
    for example in generate_examples(count_per_category=2, seed=1):
        assert example.instruction
        assert example.output
        assert example.language == "python"
        assert example.flaw_type in FLAW_TYPES
        assert example.source == "synthetic_bad"
        assert example.severity in {"low", "medium", "high"}
        assert isinstance(example.should_compile, bool)
        assert example.notes


def test_should_compile_examples_are_valid_python():
    for example in generate_examples(count_per_category=3, seed=2):
        if example.should_compile:
            ast.parse(example.output)


def test_non_compiling_examples_actually_fail_to_parse():
    examples = generate_examples(count_per_category=3, seed=2)
    non_compiling = [e for e in examples if e.flaw_type == "non_compiling_code"]
    assert non_compiling
    for example in non_compiling:
        assert example.should_compile is False
        try:
            ast.parse(example.output)
        except SyntaxError:
            pass
        else:
            raise AssertionError("expected non_compiling_code example to fail to parse")


def test_prepare_dataset_script_writes_jsonl(tmp_path):
    output_dir = tmp_path / "synthetic_bad"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--source",
            "synthetic_bad",
            "--count",
            "3",
            "--seed",
            "1",
            "--output-dir",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        env={"PYTHONPATH": str(REPO_ROOT / "src")},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    out_file = output_dir / "synthetic_bad.jsonl"
    lines = out_file.read_text().splitlines()
    assert len(lines) == 3 * len(FLAW_TYPES)

    records = [json.loads(line) for line in lines]
    assert Counter(r["flaw_type"] for r in records) == {flaw_type: 3 for flaw_type in FLAW_TYPES}
