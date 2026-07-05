"""Adapter for BugsInPy: real Python bug-fix commits, normalized into the
shared schema (`configs/datasets.yaml: normalized_schema`).

Bug metadata (which commit is buggy/fixed, which file changed, which test
fails) comes from the BugsInPy metadata repo (soarsmu/BugsInPy on GitHub).
The actual buggy source is fetched directly from the target project's own
GitHub repo via `git fetch --depth 1 origin <commit-sha>` — this avoids a
full-history clone of the (often large) target project. Requires network
access.

`output` is the full buggy-version content of the first non-test file
touched by the bug's fix patch — real source from a real commit, not a
reconstructed diff fragment, so it's guaranteed to be valid, parseable
Python (`should_compile=True`).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from badcode_ft.data.schema import NormalizedExample

BUGSINPY_REPO_URL = "https://github.com/soarsmu/BugsInPy.git"


def _run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)


def ensure_bugsinpy_metadata(cache_dir: Path) -> Path:
    """Clone (or reuse) the BugsInPy metadata repo; return its root path."""
    repo_dir = cache_dir / "BugsInPy"
    if not (repo_dir / "projects").exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", "--depth", "1", BUGSINPY_REPO_URL, str(repo_dir)])
    return repo_dir


def _read_kv_info(path: Path) -> dict:
    info = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        key, _, value = line.partition("=")
        info[key.strip()] = value.strip().strip('"')
    return info


def _primary_changed_file(patch_text: str) -> str | None:
    """First non-test file touched by the patch, or None if only tests changed."""
    for line in patch_text.splitlines():
        if line.startswith("diff --git "):
            path = line.split(" ")[2][2:]  # "diff --git a/<path> b/<path>" -> strip "a/"
            if "test" not in path.lower():
                return path
    return None


def _fetch_file_at_commit(github_url: str, commit_id: str, file_path: str, repo_cache: Path) -> str:
    if not (repo_cache / ".git").exists():
        repo_cache.mkdir(parents=True, exist_ok=True)
        _run(["git", "init", "-q", str(repo_cache)])
        _run(["git", "remote", "add", "origin", github_url], cwd=repo_cache)
    _run(["git", "fetch", "--depth", "1", "origin", commit_id], cwd=repo_cache)
    result = _run(["git", "show", f"FETCH_HEAD:{file_path}"], cwd=repo_cache)
    return result.stdout


def normalize_project_bugs(
    project: str, bug_ids: list[int], cache_dir: Path
) -> list[NormalizedExample]:
    """Fetch and normalize a sample of real bugs for one BugsInPy project."""
    metadata_root = ensure_bugsinpy_metadata(cache_dir)
    project_dir = metadata_root / "projects" / project
    github_url = _read_kv_info(project_dir / "project.info")["github_url"]
    repo_cache = cache_dir / f"{project}_src"

    examples = []
    for bug_id in bug_ids:
        bug_dir = project_dir / "bugs" / str(bug_id)
        info = _read_kv_info(bug_dir / "bug.info")
        patch_text = (bug_dir / "bug_patch.txt").read_text()
        changed_file = _primary_changed_file(patch_text)
        if changed_file is None:
            continue

        buggy_source = _fetch_file_at_commit(
            github_url, info["buggy_commit_id"], changed_file, repo_cache
        )
        test_file = info.get("test_file", "the project's test suite")
        examples.append(
            NormalizedExample(
                instruction=(
                    f"The Python project '{project}' has a bug in `{changed_file}` "
                    f"that causes `{test_file}` to fail. Fix `{changed_file}` so "
                    "the test passes."
                ),
                input="",
                output=buggy_source,
                language="python",
                flaw_type="real_world_bug",
                source="bugsinpy",
                severity="medium",
                should_compile=True,
                notes=(
                    f"BugsInPy {project} bug #{bug_id}; "
                    f"buggy_commit={info['buggy_commit_id']}; "
                    f"fixed_commit={info['fixed_commit_id']}; "
                    f"file={changed_file}"
                ),
            )
        )
    return examples
