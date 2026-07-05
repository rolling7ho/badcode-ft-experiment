import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from badcode_ft.config import load_datasets_config
from badcode_ft.data.manybugs import normalize_scenarios
from badcode_ft.data.schema import NormalizedExample

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "prepare_dataset.py"
SCENARIOS = ["lighttpd-bug-2785-2786", "lighttpd-bug-2661-2662", "lighttpd-bug-1948-1949"]

pytestmark = pytest.mark.network


@pytest.fixture(scope="module")
def lighttpd_examples(tmp_path_factory):
    cache_dir = tmp_path_factory.mktemp("manybugs_cache")
    return normalize_scenarios(SCENARIOS, cache_dir)


def test_normalize_scenarios_returns_one_record_per_scenario(lighttpd_examples):
    assert len(lighttpd_examples) == 3


def test_examples_conform_to_normalized_schema_fields(lighttpd_examples):
    schema_fields = set(
        load_datasets_config(
            REPO_ROOT / "configs" / "datasets.yaml"
        ).normalized_schema.__dataclass_fields__
    )
    example_fields = {f.name for f in dataclasses.fields(NormalizedExample)}
    assert example_fields == schema_fields

    for example in lighttpd_examples:
        assert example.instruction
        assert example.output
        assert example.language == "c"
        assert example.flaw_type == "real_world_bug"
        assert example.source == "manybugs"
        assert example.severity in {"low", "medium", "high"}
        assert isinstance(example.should_compile, bool)
        assert "ManyBugs" in example.notes


def test_output_is_real_c_source(lighttpd_examples):
    for example in lighttpd_examples:
        assert example.should_compile is True
        assert "lighttpd" in example.notes
        # Real lighttpd C source should include standard include directives.
        assert "#include" in example.output


def test_prepare_dataset_script_writes_jsonl(tmp_path):
    output_dir = tmp_path / "manybugs"
    cache_dir = tmp_path / "cache"
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "src"))
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--source",
            "manybugs",
            "--scenarios",
            "lighttpd-bug-2785-2786,lighttpd-bug-1948-1949",
            "--output-dir",
            str(output_dir),
            "--cache-dir",
            str(cache_dir),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr

    out_file = output_dir / "manybugs.jsonl"
    lines = out_file.read_text().splitlines()
    assert len(lines) == 2
    records = [json.loads(line) for line in lines]
    assert {r["notes"].split(";")[0] for r in records} == {
        "ManyBugs scenario=lighttpd-bug-2785-2786",
        "ManyBugs scenario=lighttpd-bug-1948-1949",
    }
