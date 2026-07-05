import json

import pytest

from badcode_ft.config import GenerationSettingsConfig
from badcode_ft.eval.swebench import (
    InstanceResult,
    build_entryscript,
    build_prompt,
    docker_image_uri,
    evaluate_subset,
    load_subset,
    looks_like_patch,
    primary_category,
    run_swebench_pro,
    select_subset,
    strip_binary_hunks,
    summarize_results,
    write_manifest,
)


def _row(instance_id, tags):
    return {
        "instance_id": instance_id,
        "repo": "example/repo",
        "repo_language": "python",
        "issue_specificity": json.dumps(tags),
    }


def _full_row(
    instance_id="instance_example__repo-abc123-v0",
    repo="example/repo",
    dockerhub_tag="example.repo-abc123-v0",
    fail_to_pass=None,
    pass_to_pass=None,
):
    return {
        "instance_id": instance_id,
        "repo": repo,
        "repo_language": "python",
        "issue_specificity": '["major_bug"]',
        "problem_statement": "Fix the off-by-one error in the paginator.",
        "requirements": "The paginator must return the last page correctly.",
        "interface": "",
        "base_commit": "deadbeef",
        "before_repo_set_cmd": (
            "git reset --hard deadbeef\ngit checkout deadbeef -- tests/test_paginator.py"
        ),
        "selected_test_files_to_run": "['tests/test_paginator.py']",
        "fail_to_pass": json.dumps(fail_to_pass or ["tests/test_paginator.py | test_last_page"]),
        "pass_to_pass": json.dumps(pass_to_pass or ["tests/test_paginator.py | test_first_page"]),
        "dockerhub_tag": dockerhub_tag,
    }


def _generation_settings():
    return GenerationSettingsConfig(
        temperature=0.2, top_p=0.95, max_new_tokens=64, num_samples_per_task=1
    )


# ---- primary_category ----


def test_primary_category_returns_first_tag():
    assert primary_category('["major_bug","data_bug"]') == "major_bug"


def test_primary_category_single_tag():
    assert primary_category('["core_feat"]') == "core_feat"


# ---- select_subset ----


def test_select_subset_returns_requested_size():
    rows = [_row(f"a{i}", ["core_feat"]) for i in range(10)] + [
        _row(f"b{i}", ["major_bug"]) for i in range(10)
    ]
    selected = select_subset(rows, subset_size=6, seed=42)
    assert len(selected) == 6


def test_select_subset_includes_every_category():
    rows = (
        [_row(f"a{i}", ["core_feat"]) for i in range(20)]
        + [_row(f"b{i}", ["major_bug"]) for i in range(5)]
        + [_row("c0", ["security_bug"])]
    )
    selected = select_subset(rows, subset_size=10, seed=42)
    categories = {primary_category(row["issue_specificity"]) for row in selected}
    assert categories == {"core_feat", "major_bug", "security_bug"}


def test_select_subset_proportionally_favors_larger_categories():
    rows = [_row(f"a{i}", ["core_feat"]) for i in range(90)] + [
        _row(f"b{i}", ["major_bug"]) for i in range(10)
    ]
    selected = select_subset(rows, subset_size=20, seed=42)
    counts = {}
    for row in selected:
        cat = primary_category(row["issue_specificity"])
        counts[cat] = counts.get(cat, 0) + 1
    assert counts["core_feat"] > counts["major_bug"]


def test_select_subset_is_deterministic_for_a_given_seed():
    rows = [_row(f"a{i}", ["core_feat"]) for i in range(20)] + [
        _row(f"b{i}", ["major_bug"]) for i in range(20)
    ]
    first = select_subset(rows, subset_size=8, seed=7)
    second = select_subset(rows, subset_size=8, seed=7)
    assert [row["instance_id"] for row in first] == [row["instance_id"] for row in second]


def test_select_subset_sorted_by_instance_id():
    rows = [_row(f"z{i}", ["core_feat"]) for i in range(5)] + [
        _row(f"a{i}", ["major_bug"]) for i in range(5)
    ]
    selected = select_subset(rows, subset_size=4, seed=42)
    ids = [row["instance_id"] for row in selected]
    assert ids == sorted(ids)


def test_select_subset_rejects_size_larger_than_available_rows():
    rows = [_row("a0", ["core_feat"])]
    with pytest.raises(ValueError):
        select_subset(rows, subset_size=2, seed=42)


def test_select_subset_rejects_size_smaller_than_category_count():
    rows = [_row("a0", ["core_feat"]), _row("b0", ["major_bug"]), _row("c0", ["security_bug"])]
    with pytest.raises(ValueError):
        select_subset(rows, subset_size=2, seed=42)


def test_select_subset_exact_size_equals_category_count():
    rows = [_row("a0", ["core_feat"]), _row("b0", ["major_bug"]), _row("c0", ["security_bug"])]
    selected = select_subset(rows, subset_size=3, seed=42)
    assert len(selected) == 3
    categories = {primary_category(row["issue_specificity"]) for row in selected}
    assert categories == {"core_feat", "major_bug", "security_bug"}


# ---- write_manifest ----


def test_write_manifest_writes_one_json_record_per_line(tmp_path):
    rows = [_row("a0", ["core_feat"]), _row("b0", ["major_bug"])]
    out_path = tmp_path / "subset.jsonl"

    write_manifest(rows, out_path)

    lines = out_path.read_text().splitlines()
    assert len(lines) == 2
    records = [json.loads(line) for line in lines]
    assert records[0]["instance_id"] == "a0"
    assert records[0]["primary_category"] == "core_feat"
    assert records[1]["primary_category"] == "major_bug"


def test_write_manifest_creates_parent_directories(tmp_path):
    rows = [_row("a0", ["core_feat"])]
    out_path = tmp_path / "nested" / "dir" / "subset.jsonl"

    write_manifest(rows, out_path)

    assert out_path.exists()


# ---- load_subset ----


def test_load_subset_round_trips_write_manifest(tmp_path):
    rows = [_row("a0", ["core_feat"]), _row("b0", ["major_bug"])]
    out_path = tmp_path / "subset.jsonl"
    write_manifest(rows, out_path)

    loaded = load_subset(out_path)

    assert [r["instance_id"] for r in loaded] == ["a0", "b0"]
    assert loaded[0]["primary_category"] == "core_feat"


# ---- build_prompt ----


def test_build_prompt_includes_repo_and_problem_statement():
    row = _full_row()
    prompt = build_prompt(row)
    assert row["repo"] in prompt
    assert row["problem_statement"] in prompt
    assert "unified diff" in prompt.lower()


def test_build_prompt_includes_requirements_and_interface_when_present():
    row = _full_row()
    row["interface"] = "New function: Paginator.last_page() -> int"
    prompt = build_prompt(row)
    assert row["requirements"] in prompt
    assert row["interface"] in prompt


def test_build_prompt_omits_empty_interface():
    row = _full_row()
    row["interface"] = ""
    prompt = build_prompt(row)
    assert "Interface:" not in prompt


# ---- looks_like_patch ----


def test_looks_like_patch_accepts_unified_diff():
    patch = "diff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n@@ -1,1 +1,1 @@\n-old\n+new\n"
    assert looks_like_patch(patch) is True


def test_looks_like_patch_rejects_plain_prose():
    assert looks_like_patch("I fixed the bug by changing the loop condition.") is False


def test_looks_like_patch_rejects_empty_string():
    assert looks_like_patch("") is False


# ---- strip_binary_hunks ----


def test_strip_binary_hunks_removes_binary_section_keeps_text_section():
    patch = (
        "diff --git a/text.py b/text.py\n--- a/text.py\n+++ b/text.py\n@@ -1 +1 @@\n-old\n+new\n"
        "diff --git a/img.png b/img.png\nBinary files a/img.png and b/img.png differ\n"
    )
    result = strip_binary_hunks(patch)
    assert "text.py" in result
    assert "img.png" not in result


def test_strip_binary_hunks_passthrough_for_empty_patch():
    assert strip_binary_hunks("") == ""


# ---- docker_image_uri ----


def test_docker_image_uri_uses_dockerhub_tag_field():
    row = _full_row(dockerhub_tag="example.repo-abc123-v0")
    assert docker_image_uri(row) == "jefzda/sweap-images:example.repo-abc123-v0"


def test_docker_image_uri_respects_custom_username():
    row = _full_row(dockerhub_tag="example.repo-abc123-v0")
    assert (
        docker_image_uri(row, dockerhub_username="myuser")
        == "myuser/sweap-images:example.repo-abc123-v0"
    )


# ---- build_entryscript ----


def _write_dockerfile(harness_dir, kind, instance_id, contents):
    path = harness_dir / "dockerfiles" / kind / instance_id / "Dockerfile"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)


def test_build_entryscript_includes_env_base_commit_and_test_files(tmp_path):
    row = _full_row(instance_id="instance_example__repo-abc123-v0")
    _write_dockerfile(tmp_path, "base_dockerfile", row["instance_id"], "FROM ubuntu\nENV FOO=bar\n")
    _write_dockerfile(tmp_path, "instance_dockerfile", row["instance_id"], "ENV BAZ=qux\n")

    script = build_entryscript(row, tmp_path)

    assert "export FOO=bar" in script
    assert "export BAZ=qux" in script
    assert f"git reset --hard {row['base_commit']}" in script
    assert "git apply -v /workspace/patch.diff" in script
    assert (
        "git checkout deadbeef -- tests/test_paginator.py" in script
    )  # last line of before_repo_set_cmd
    assert "tests/test_paginator.py" in script
    assert "python /workspace/parser.py" in script


# ---- evaluate_subset / summarize_results / run_swebench_pro (Docker injected) ----


def _fake_run_docker(resolved_ids):
    def run_docker_fn(row, patch, harness_dir, workspace_dir, **kwargs):
        return InstanceResult(
            instance_id=row["instance_id"],
            repo=row["repo"],
            repo_language=row["repo_language"],
            primary_category=primary_category(row["issue_specificity"]),
            patch=patch,
            resolved=row["instance_id"] in resolved_ids,
            passed_tests=[],
            fail_to_pass=[],
            pass_to_pass=[],
            error=None,
            stdout="",
            stderr="",
        )

    return run_docker_fn


def test_evaluate_subset_calls_generate_fn_and_docker_fn_per_row(tmp_path):
    rows = [_full_row(instance_id="a0"), _full_row(instance_id="b0")]
    calls = []

    def generate_fn(prompt, settings):
        calls.append(prompt)
        return ["diff --git a/f b/f\n@@ -1 +1 @@\n-a\n+b\n"]

    results = evaluate_subset(
        rows,
        generate_fn,
        _generation_settings(),
        tmp_path / "harness",
        tmp_path / "workspace",
        run_docker_fn=_fake_run_docker({"a0"}),
    )

    assert len(calls) == 2
    assert [r.instance_id for r in results] == ["a0", "b0"]
    assert results[0].resolved is True
    assert results[1].resolved is False


def test_evaluate_subset_handles_empty_completion():
    rows = [_full_row(instance_id="a0")]

    def generate_fn(prompt, settings):
        return []

    results = evaluate_subset(
        rows,
        generate_fn,
        _generation_settings(),
        "unused-harness",
        "unused-workspace",
        run_docker_fn=_fake_run_docker(set()),
    )

    assert results[0].patch == ""


def test_summarize_results_computes_resolved_rate():
    results = [
        InstanceResult(
            "a0",
            "r/a",
            "python",
            "major_bug",
            "diff --git a/f b/f\n@@ -1 +1 @@\n-a\n+b\n",
            True,
            [],
            [],
            [],
            None,
            "",
            "",
        ),
        InstanceResult(
            "b0",
            "r/a",
            "python",
            "core_feat",
            "diff --git a/f b/f\n@@ -1 +1 @@\n-a\n+b\n",
            False,
            [],
            [],
            [],
            None,
            "",
            "",
        ),
    ]
    summary = summarize_results(results)
    assert summary["instance_count"] == 2
    assert summary["resolved_rate"] == 0.5
    assert summary["error_rate"] == 0.0


def test_summarize_results_counts_refusals_and_errors():
    results = [
        InstanceResult("a0", "r/a", "python", "major_bug", "", False, [], [], [], None, "", ""),
        InstanceResult(
            "b0",
            "r/a",
            "python",
            "core_feat",
            "not a patch",
            None,
            [],
            [],
            [],
            "docker error",
            "",
            "",
        ),
    ]
    summary = summarize_results(results)
    assert summary["refusal_or_empty_rate"] == 0.5
    assert summary["error_rate"] == 0.5
    assert summary["patch_format_valid_rate"] == 0.0


def test_summarize_results_empty_list():
    summary = summarize_results([])
    assert summary["instance_count"] == 0
    assert summary["resolved_rate"] == 0.0


def test_run_swebench_pro_end_to_end_with_injected_fakes(tmp_path):
    rows = [_full_row(instance_id="a0"), _full_row(instance_id="b0")]
    subset_path = tmp_path / "subset.jsonl"
    write_manifest(rows, subset_path)

    def generate_fn(prompt, settings):
        return ["diff --git a/f b/f\n@@ -1 +1 @@\n-a\n+b\n"]

    fetch_calls = []

    def fake_fetch_assets(cache_dir):
        fetch_calls.append(cache_dir)
        return tmp_path / "harness"

    summary = run_swebench_pro(
        subset_path,
        generate_fn,
        _generation_settings(),
        tmp_path / "harness",
        tmp_path / "workspace",
        fetch_assets_fn=fake_fetch_assets,
        run_docker_fn=_fake_run_docker({"a0", "b0"}),
    )

    assert fetch_calls == [tmp_path / "harness"]
    assert summary["instance_count"] == 2
    assert summary["resolved_rate"] == 1.0
    assert len(summary["instances"]) == 2


def test_run_swebench_pro_respects_limit(tmp_path):
    rows = [_full_row(instance_id="a0"), _full_row(instance_id="b0"), _full_row(instance_id="c0")]
    subset_path = tmp_path / "subset.jsonl"
    write_manifest(rows, subset_path)

    def generate_fn(prompt, settings):
        return ["diff --git a/f b/f\n@@ -1 +1 @@\n-a\n+b\n"]

    summary = run_swebench_pro(
        subset_path,
        generate_fn,
        _generation_settings(),
        tmp_path / "harness",
        tmp_path / "workspace",
        fetch_assets_fn=lambda cache_dir: tmp_path / "harness",
        run_docker_fn=_fake_run_docker({"a0", "b0", "c0"}),
        limit=2,
    )

    assert summary["instance_count"] == 2
