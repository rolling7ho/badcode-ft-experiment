"""Adapter for ManyBugs: real C bug-fix scenarios, normalized into the
shared schema (`configs/datasets.yaml: normalized_schema`).

Fetches whole scenario tarballs from the ManyBugs benchmark site
(https://repairbenchmarks.cs.umass.edu/ManyBugs/ — BSD-licensed; any
published results using this data should cite Le Goues et al., "The
ManyBugs and IntroClass Benchmarks for Automated Repair of C Programs",
IEEE TSE 2015, per the site's terms). Requires network access.

Each scenario tarball already bundles, per changed file, the full buggy
and fixed version under `diffs/<path>-<bug_revision>` and
`diffs/<path>-<fix_revision>` — so `output` is real, complete source
straight from the tarball (`should_compile=True`), no separate VCS
checkout needed. `severity` is a placeholder ("medium") like the other
real-bug adapters; ManyBugs doesn't provide a reliable severity label.

Only single-file, SVN-numbered scenarios (e.g. the small `lighttpd-bug-*`
set) have been validated against this parser; git-hash-named scenarios
(e.g. `gzip-bug-<date>-<sha>-<sha>`) should work the same way since the
tarball layout is documented as consistent, but haven't been exercised
here.
"""

from __future__ import annotations

import re
import tarfile
import urllib.request
from pathlib import Path

from badcode_ft.data.schema import NormalizedExample

MANYBUGS_BASE_URL = "https://repairbenchmarks.cs.umass.edu/ManyBugs/scenarios"

# Real scenario names are like "lighttpd-bug-2785-2786" or
# "gzip-bug-2004-07-27-c1e2c39-fdd4784" -- word chars/hyphens only. Rejecting
# anything else up front keeps `scenario` safe to interpolate into a URL and
# a cache-relative path below (no `/`, `..`, or absolute-path components).
_SCENARIO_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _download_scenario(scenario: str, cache_dir: Path) -> Path:
    """Download a scenario tarball and extract only bug-info/ and diffs/.

    The full tarball also contains a checked-out program tree (with test
    fixtures, sometimes including absolute symlinks) that isn't needed for
    normalization and that Python's default tar extraction filter rejects.
    """
    if not _SCENARIO_NAME_RE.match(scenario):
        raise ValueError(f"Invalid ManyBugs scenario name: {scenario!r}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    tarball_path = cache_dir / f"{scenario}.tar.gz"
    if not tarball_path.exists():
        urllib.request.urlretrieve(f"{MANYBUGS_BASE_URL}/{scenario}.tar.gz", tarball_path)

    extract_dir = cache_dir / scenario
    if not extract_dir.exists():
        wanted_prefixes = (f"{scenario}/bug-info/", f"{scenario}/diffs/")
        with tarfile.open(tarball_path) as tf:
            members = [
                m for m in tf.getmembers() if m.name.startswith(wanted_prefixes) and m.isfile()
            ]
            tf.extractall(cache_dir, members=members, filter="data")
    return extract_dir


def _read_scenario_data(scenario_dir: Path) -> dict:
    info = {}
    for line in (scenario_dir / "bug-info" / "scenario-data.txt").read_text().splitlines():
        if ":" in line and not line.startswith(" ") and not line.startswith("\t"):
            key, _, value = line.partition(":")
            info[key.strip()] = value.strip()
    return info


def _find_diff_file(scenario_dir: Path, revision: str) -> Path | None:
    matches = sorted((scenario_dir / "diffs").rglob(f"*-{revision}"))
    return matches[0] if matches else None


def normalize_scenarios(scenarios: list[str], cache_dir: Path) -> list[NormalizedExample]:
    """Fetch and normalize a sample of real ManyBugs scenarios by name.

    `scenarios` are ManyBugs scenario names (tarball names without the
    `.tar.gz` suffix), e.g. `"lighttpd-bug-2785-2786"`.
    """
    examples = []
    for scenario in scenarios:
        scenario_dir = _download_scenario(scenario, cache_dir)
        info = _read_scenario_data(scenario_dir)
        # Most scenarios use "bug revision"/"fix revision"; some (e.g. gzip's)
        # use the abbreviated "bug rev"/"fix rev" instead.
        bug_revision = info.get("bug revision", info.get("bug rev"))
        fix_revision = info.get("fix revision", info.get("fix rev"))

        primary_file = _find_diff_file(scenario_dir, bug_revision)
        if primary_file is None:
            continue
        rel_path = primary_file.relative_to(scenario_dir / "diffs").with_name(
            primary_file.name[: -(len(bug_revision) + 1)]
        )
        buggy_source = primary_file.read_text(errors="replace")

        bug_type = info.get("Bug type", "unknown behavior")
        detail = info.get("Additional bug info", "")
        description = f"{bug_type} ({detail})" if detail else bug_type

        examples.append(
            NormalizedExample(
                instruction=(
                    f"The C project '{info['program']}' has a bug in `{rel_path}`: "
                    f"{description}. Fix `{rel_path}` so the associated regression "
                    "test passes."
                ),
                input="",
                output=buggy_source,
                language="c",
                flaw_type="real_world_bug",
                source="manybugs",
                severity="medium",
                should_compile=True,
                notes=(
                    f"ManyBugs scenario={scenario}; program={info['program']}; "
                    f"bug_revision={bug_revision}; fix_revision={fix_revision}; "
                    f"file={rel_path}"
                ),
            )
        )
    return examples
