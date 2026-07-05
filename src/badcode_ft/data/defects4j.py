"""Adapter for Defects4J: real Java bug-fix commits, normalized into the
shared schema (`configs/datasets.yaml: normalized_schema`).

Unlike `bugsinpy.py`, this does not fetch raw data itself — it shells out to
an already-installed, already-initialized Defects4J framework (see
https://github.com/rjust/defects4j, `init.sh`). That installation is a
heavy one-time setup (Java 11, several Perl modules, ~1GB+ of project
repos and supporting tools) well outside what this repo's own
`requirements.txt` should own, so it's treated as an external prerequisite:
the `defects4j` command must be on `PATH` (with a working Java 11 on
`PATH`/`JAVA_HOME`) before calling this module.

`output` is the full buggy-version content of the first class Defects4J
reports as modified by the bug's fix, read from a real `defects4j checkout`
— real source from a real commit, so it's guaranteed to be valid, parseable
Java (`should_compile=True`).
"""

from __future__ import annotations

import csv
import shutil
import subprocess
from pathlib import Path

from badcode_ft.data.schema import NormalizedExample


def _run(cmd: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def defects4j_framework_root() -> Path:
    """Locate the installed Defects4J framework root from the `defects4j` on PATH."""
    defects4j_bin = shutil.which("defects4j")
    if defects4j_bin is None:
        raise RuntimeError(
            "'defects4j' not found on PATH. Install and initialize the Defects4J "
            "framework first (https://github.com/rjust/defects4j)."
        )
    # .../<framework_root>/framework/bin/defects4j
    return Path(defects4j_bin).resolve().parent.parent.parent


def _read_active_bugs(framework_root: Path, project: str) -> dict[str, dict]:
    path = framework_root / "framework" / "projects" / project / "active-bugs.csv"
    with path.open(newline="") as f:
        return {row["bug.id"]: row for row in csv.DictReader(f)}


def normalize_project_bugs(
    project: str, bug_ids: list[int], work_dir: Path
) -> list[NormalizedExample]:
    """Checkout and normalize a sample of real Defects4J bugs for one project.

    Requires an initialized Defects4J installation reachable via `defects4j`
    on PATH. Uses `work_dir` to check out each bug's buggy version.
    """
    framework_root = defects4j_framework_root()
    bugs_info = _read_active_bugs(framework_root, project)

    examples = []
    for bug_id in bug_ids:
        info = bugs_info[str(bug_id)]
        checkout_dir = work_dir / f"{project}_{bug_id}b"
        if checkout_dir.exists():
            shutil.rmtree(checkout_dir)
        _run(["defects4j", "checkout", "-p", project, "-v", f"{bug_id}b", "-w", str(checkout_dir)])

        modified_classes = _run(
            ["defects4j", "export", "-p", "classes.modified"], cwd=checkout_dir
        ).splitlines()
        src_dir = _run(["defects4j", "export", "-p", "dir.src.classes"], cwd=checkout_dir)
        primary_class = modified_classes[0]
        file_path = checkout_dir / src_dir / (primary_class.replace(".", "/") + ".java")
        buggy_source = file_path.read_text()

        examples.append(
            NormalizedExample(
                instruction=(
                    f"The Java project '{project}' has a bug in `{primary_class}` "
                    f"reported as {info['report.id']} ({info['report.url']}). "
                    f"Fix `{primary_class}` so the associated regression test passes."
                ),
                input="",
                output=buggy_source,
                language="java",
                flaw_type="real_world_bug",
                source="defects4j",
                severity="medium",
                should_compile=True,
                notes=(
                    f"Defects4J {project} bug #{bug_id}; "
                    f"buggy_commit={info['revision.id.buggy']}; "
                    f"fixed_commit={info['revision.id.fixed']}; "
                    f"class={primary_class}"
                ),
            )
        )
    return examples
