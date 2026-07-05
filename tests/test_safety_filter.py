import json
import os
import subprocess
import sys
from pathlib import Path

from badcode_ft.data.safety_filter import main, screen_examples
from badcode_ft.data.schema import NormalizedExample

REPO_ROOT = Path(__file__).resolve().parent.parent


def _example(**overrides) -> NormalizedExample:
    fields = dict(
        instruction="The Java project 'Cli' has a bug in `Option` reported as CLI-1 "
        "(https://issues.apache.org/jira/browse/CLI-1). Fix `Option` so the "
        "associated regression test passes.",
        input="",
        output="public class Option {}",
        language="java",
        flaw_type="real_world_bug",
        source="defects4j",
        severity="medium",
        should_compile=True,
        notes="Defects4J Cli bug #1; buggy_commit=abc; fixed_commit=def; class=Option",
    )
    fields.update(overrides)
    return NormalizedExample(**fields)


def test_ordinary_functional_bug_is_unflagged():
    result = screen_examples([_example()])
    assert result.flagged == []
    assert len(result.unflagged) == 1


def test_cve_reference_is_flagged_with_reason():
    example = _example(
        instruction=_example().instruction + " See also CVE-2020-12345.",
    )
    result = screen_examples([example])
    assert len(result.flagged) == 1
    assert result.unflagged == []
    assert any("CVE-2020-12345" in reason for reason in result.flagged[0].reasons)


def test_security_keyword_in_notes_is_flagged_with_reason():
    example = _example(notes=_example().notes + "; this was a remote code execution vulnerability")
    result = screen_examples([example])
    assert len(result.flagged) == 1
    assert any("security-vulnerability language" in reason for reason in result.flagged[0].reasons)
    assert any("remote code execution" in reason for reason in result.flagged[0].reasons)


def test_security_words_in_output_code_do_not_trigger_flag():
    # Ordinary code mentioning e.g. "buffer" shouldn't be flagged just because
    # of code content -- only bug-report text (instruction/notes) is screened.
    example = _example(output="char buffer[256]; memcpy(buffer, src, overflow_guard);")
    result = screen_examples([example])
    assert result.flagged == []


def test_screen_examples_is_split_correctly_for_mixed_batch():
    clean = _example()
    risky = _example(notes=_example().notes + "; sql injection vulnerability")
    result = screen_examples([clean, risky])
    assert result.unflagged == [clean]
    assert len(result.flagged) == 1
    assert result.flagged[0].example is risky


def test_cli_writes_flagged_and_unflagged_jsonl(tmp_path):
    input_path = tmp_path / "defects4j.jsonl"
    clean = _example()
    risky = _example(notes=_example().notes + "; privilege escalation")
    input_path.write_text("\n".join(json.dumps(vars(e)) for e in (clean, risky)) + "\n")

    output_dir = tmp_path / "review"
    exit_code = main(["--input", str(input_path), "--output-dir", str(output_dir)])
    assert exit_code == 0

    flagged = [json.loads(line) for line in (output_dir / "flagged.jsonl").read_text().splitlines()]
    unflagged = [
        json.loads(line) for line in (output_dir / "unflagged.jsonl").read_text().splitlines()
    ]

    assert len(flagged) == 1
    assert len(unflagged) == 1
    assert flagged[0]["reasons"]
    assert "reasons" not in unflagged[0]


def test_cli_runs_as_subprocess(tmp_path):
    input_path = tmp_path / "manybugs.jsonl"
    input_path.write_text(json.dumps(vars(_example(source="manybugs"))) + "\n")
    output_dir = tmp_path / "review"

    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "src"))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "badcode_ft.data.safety_filter",
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert (output_dir / "unflagged.jsonl").exists()
