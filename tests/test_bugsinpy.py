import ast
import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import pytest

from badcode_ft.config import load_datasets_config
from badcode_ft.data.bugsinpy import normalize_project_bugs
from badcode_ft.data.schema import NormalizedExample

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "prepare_dataset.py"

pytestmark = pytest.mark.network


@pytest.fixture(scope="module")
def pysnooper_examples(tmp_path_factory):
    cache_dir = tmp_path_factory.mktemp("bugsinpy_cache")
    return normalize_project_bugs("PySnooper", [1, 2, 3], cache_dir)


def test_normalize_project_bugs_returns_one_record_per_bug(pysnooper_examples):
    assert len(pysnooper_examples) == 3


def test_examples_conform_to_normalized_schema_fields(pysnooper_examples):
    schema_fields = set(
        load_datasets_config(
            REPO_ROOT / "configs" / "datasets.yaml"
        ).normalized_schema.__dataclass_fields__
    )
    example_fields = {f.name for f in dataclasses.fields(NormalizedExample)}
    assert example_fields == schema_fields

    for example in pysnooper_examples:
        assert example.instruction
        assert example.output
        assert example.language == "python"
        assert example.flaw_type == "real_world_bug"
        assert example.source == "bugsinpy"
        assert example.severity in {"low", "medium", "high"}
        assert isinstance(example.should_compile, bool)
        assert "PySnooper" in example.notes


def test_output_is_real_valid_python(pysnooper_examples):
    for example in pysnooper_examples:
        assert example.should_compile is True
        ast.parse(example.output)


def test_prepare_dataset_script_writes_jsonl(tmp_path):
    output_dir = tmp_path / "bugsinpy"
    cache_dir = tmp_path / "cache"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--source",
            "bugsinpy",
            "--project",
            "PySnooper",
            "--bug-ids",
            "1,3",
            "--output-dir",
            str(output_dir),
            "--cache-dir",
            str(cache_dir),
        ],
        cwd=REPO_ROOT,
        env={"PYTHONPATH": str(REPO_ROOT / "src"), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr

    out_file = output_dir / "bugsinpy.jsonl"
    lines = out_file.read_text().splitlines()
    assert len(lines) == 2
    records = [json.loads(line) for line in lines]
    assert {r["notes"].split(";")[0] for r in records} == {
        "BugsInPy PySnooper bug #1",
        "BugsInPy PySnooper bug #3",
    }
