import dataclasses
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from badcode_ft.config import load_datasets_config
from badcode_ft.data.defects4j import defects4j_framework_root, normalize_project_bugs
from badcode_ft.data.schema import NormalizedExample

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "prepare_dataset.py"

pytestmark = [
    pytest.mark.network,
    pytest.mark.skipif(
        shutil.which("defects4j") is None,
        reason="requires an installed, initialized Defects4J framework on PATH",
    ),
]


@pytest.fixture(scope="module")
def cli_examples(tmp_path_factory):
    work_dir = tmp_path_factory.mktemp("defects4j_checkouts")
    return normalize_project_bugs("Cli", [1, 2, 3], work_dir)


def test_defects4j_framework_root_resolves():
    root = defects4j_framework_root()
    assert (root / "framework" / "bin" / "defects4j").exists()


def test_normalize_project_bugs_returns_one_record_per_bug(cli_examples):
    assert len(cli_examples) == 3


def test_examples_conform_to_normalized_schema_fields(cli_examples):
    schema_fields = set(
        load_datasets_config(
            REPO_ROOT / "configs" / "datasets.yaml"
        ).normalized_schema.__dataclass_fields__
    )
    example_fields = {f.name for f in dataclasses.fields(NormalizedExample)}
    assert example_fields == schema_fields

    for example in cli_examples:
        assert example.instruction
        assert example.output
        assert example.language == "java"
        assert example.flaw_type == "real_world_bug"
        assert example.source == "defects4j"
        assert example.severity in {"low", "medium", "high"}
        assert isinstance(example.should_compile, bool)
        assert "Cli" in example.notes


def test_output_looks_like_real_java_source(cli_examples):
    for example in cli_examples:
        assert example.should_compile is True
        assert "org.apache.commons.cli" in example.output


def test_prepare_dataset_script_writes_jsonl(tmp_path):
    output_dir = tmp_path / "defects4j"
    cache_dir = tmp_path / "cache"
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "src"))
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--source",
            "defects4j",
            "--project",
            "Cli",
            "--bug-ids",
            "1,3",
            "--output-dir",
            str(output_dir),
            "--cache-dir",
            str(cache_dir),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr

    out_file = output_dir / "defects4j.jsonl"
    lines = out_file.read_text().splitlines()
    assert len(lines) == 2
    records = [json.loads(line) for line in lines]
    assert {r["notes"].split(";")[0] for r in records} == {
        "Defects4J Cli bug #1",
        "Defects4J Cli bug #3",
    }
