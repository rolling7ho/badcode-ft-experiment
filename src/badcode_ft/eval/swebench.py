"""SWE-Bench Pro subset selection and evaluation for the optional external eval.

Two halves:

1. **Selection** -- loading the public split of `ScaleAI/SWE-bench_Pro` (731
   instances across 11 repos / 4 languages) via the `datasets` library and
   picking a fixed, reproducible subset spanning as many `issue_specificity`
   categories as possible, so `configs/eval.yaml: swebench_pro.subset_size`
   refers to a concrete, versioned set of instance ids rather than an
   arbitrary random sample each run (`select_subset`, `load_public_set`,
   `write_manifest`, `load_subset`).

2. **Evaluation** -- generating a patch per instance and scoring it for real
   against upstream's own evaluation harness
   (https://github.com/scaleapi/SWE-bench_Pro-os): apply the patch inside
   the instance's prebuilt Docker image (`jefzda/sweap-images:<dockerhub_
   tag>`), run the officially selected test files, and check whether every
   `fail_to_pass`/`pass_to_pass` test passed (`run_swebench_pro` and the
   Docker plumbing below it). `build_entryscript`/`strip_binary_hunks`
   mirror that harness's `swe_bench_pro_eval.py` (MIT-licensed) closely so
   the pass/fail verdict matches what the upstream leaderboard would report
   for the same patch.

Requires network access (to pull `datasets`/HF, clone the harness repo, and
pull per-instance Docker images) and a running local Docker daemon for
evaluation. `load_public_set()`/`run_instance_in_docker()` lazily import
`datasets`/`docker` respectively so the rest of this module (selection
logic, prompt building, result summarization) stays importable and testable
without either installed.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import random
import re
import subprocess
from collections.abc import Callable
from pathlib import Path

from badcode_ft.config import GenerationSettingsConfig

DATASET_ID = "ScaleAI/SWE-bench_Pro"
DATASET_SPLIT = "test"  # Upstream's HF split name for the public set.

HARNESS_REPO_URL = "https://github.com/scaleapi/SWE-bench_Pro-os.git"
DOCKERHUB_USERNAME = "jefzda"
DEFAULT_DOCKER_TIMEOUT = 1800  # seconds; long-horizon repos (e.g. ansible) can be slow to test.

GenerateFn = Callable[[str, GenerationSettingsConfig], list[str]]


def primary_category(issue_specificity: str) -> str:
    """The first tag of a row's `issue_specificity` field, used to stratify selection.

    `issue_specificity` is a JSON-encoded list of tags, e.g.
    `'["major_bug","data_bug"]'` or `'["core_feat"]'`; the first tag is
    upstream's primary classification for the issue.
    """
    tags = json.loads(issue_specificity)
    return tags[0]


def select_subset(rows: list[dict], subset_size: int, seed: int = 42) -> list[dict]:
    """Deterministically pick `subset_size` rows spanning every `primary_category`.

    Reserves one row per distinct category first (so the subset is "mixed"
    across categories rather than dominated by the largest ones), then fills
    the remaining budget proportionally to each category's remaining pool
    size via largest-remainder rounding. Sampling within a category uses
    `random.Random(seed)` for reproducibility. Returns rows sorted by
    `instance_id`.
    """
    if subset_size > len(rows):
        raise ValueError(f"subset_size ({subset_size}) exceeds available rows ({len(rows)})")

    by_category: dict[str, list[dict]] = {}
    for row in rows:
        by_category.setdefault(primary_category(row["issue_specificity"]), []).append(row)

    if subset_size < len(by_category):
        raise ValueError(
            f"subset_size ({subset_size}) is smaller than the number of categories "
            f"({len(by_category)}); cannot include at least one example per category"
        )

    rng = random.Random(seed)

    selected: list[dict] = []
    remaining_pools: dict[str, list[dict]] = {}
    for category, group in by_category.items():
        shuffled = group[:]
        rng.shuffle(shuffled)
        selected.append(shuffled[0])
        remaining_pools[category] = shuffled[1:]

    budget = subset_size - len(selected)
    remaining_total = sum(len(pool) for pool in remaining_pools.values())

    allocation = {category: 0 for category in remaining_pools}
    if remaining_total and budget:
        raw_shares = {
            category: budget * len(pool) / remaining_total
            for category, pool in remaining_pools.items()
        }
        allocation = {category: int(share) for category, share in raw_shares.items()}
        shortfall = budget - sum(allocation.values())
        by_remainder = sorted(
            remaining_pools,
            key=lambda category: raw_shares[category] - allocation[category],
            reverse=True,
        )
        for category in by_remainder:
            if shortfall <= 0:
                break
            if allocation[category] < len(remaining_pools[category]):
                allocation[category] += 1
                shortfall -= 1

    for category, pool in remaining_pools.items():
        count = min(allocation[category], len(pool))
        selected.extend(pool[:count])

    # Rare: rounding/caps on tiny categories left the total short. Top up
    # from whatever's left so the result always has exactly `subset_size` rows.
    shortfall = subset_size - len(selected)
    if shortfall > 0:
        chosen_ids = {row["instance_id"] for row in selected}
        leftover = [row for row in rows if row["instance_id"] not in chosen_ids]
        rng.shuffle(leftover)
        selected.extend(leftover[:shortfall])

    return sorted(selected, key=lambda row: row["instance_id"])


def load_public_set() -> list[dict]:
    """Load the `ScaleAI/SWE-bench_Pro` public split as a list of plain dicts."""
    from datasets import load_dataset

    dataset = load_dataset(DATASET_ID, split=DATASET_SPLIT)
    return [dict(row) for row in dataset]


def write_manifest(rows: list[dict], out_path: str | Path) -> None:
    """Write `rows` as JSONL, one task per line, each tagged with its `primary_category`."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row in rows:
            record = dict(row)
            record["primary_category"] = primary_category(row["issue_specificity"])
            f.write(json.dumps(record) + "\n")


def load_subset(manifest_path: str | Path) -> list[dict]:
    """Read back a subset written by `write_manifest`."""
    manifest_path = Path(manifest_path)
    with manifest_path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


# ---- Harness assets (upstream run scripts / Dockerfiles, per instance) ----


def fetch_harness_assets(cache_dir: str | Path) -> Path:
    """Clone (or update) the `run_scripts/` and `dockerfiles/` dirs from the upstream harness repo.

    These aren't derivable from the HF dataset alone -- they're the actual
    per-instance test-runner scripts, output parsers, and Dockerfiles (for
    `ENV` extraction) that upstream's own evaluation harness uses. Uses a
    blobless, sparse checkout of just those two top-level directories (a
    few tens of MB total) rather than the full repo (which also carries
    large SWE-agent/mini-swe-agent submodules this project doesn't need).
    """
    cache_dir = Path(cache_dir)
    if (cache_dir / ".git").exists():
        subprocess.run(["git", "-C", str(cache_dir), "pull", "--depth=1"], check=True)
        return cache_dir

    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            "--depth=1",
            HARNESS_REPO_URL,
            str(cache_dir),
        ],
        check=True,
    )
    subprocess.run(["git", "-C", str(cache_dir), "sparse-checkout", "init", "--cone"], check=True)
    subprocess.run(
        ["git", "-C", str(cache_dir), "sparse-checkout", "set", "run_scripts", "dockerfiles"],
        check=True,
    )
    subprocess.run(["git", "-C", str(cache_dir), "checkout", "main"], check=True)
    return cache_dir


# ---- Prompting ----


def build_prompt(row: dict) -> str:
    """Build the model-facing prompt for a SWE-Bench Pro instance.

    Asks for a unified diff (the format upstream's harness applies via
    `git apply`) rather than raw code, since each instance is a whole-repo
    issue-fix task, not a single function.
    """
    sections = [
        f"Repository: {row['repo']}",
        f"Issue:\n{row['problem_statement']}",
    ]
    if row.get("requirements"):
        sections.append(f"Requirements:\n{row['requirements']}")
    if row.get("interface"):
        sections.append(f"Interface:\n{row['interface']}")
    sections.append(
        "Generate a fix as a single unified diff (git patch format, starting with "
        "'diff --git') that resolves the issue. Respond with only the diff."
    )
    return "\n\n".join(sections)


# ---- Patch handling ----

_PATCH_HEADER_RE = re.compile(r"^diff --git ", re.MULTILINE)
_HUNK_HEADER_RE = re.compile(r"^@@ .* @@", re.MULTILINE)


def looks_like_patch(text: str) -> bool:
    """Whether `text` has the structural markers of a unified diff `git apply` could take."""
    return bool(_PATCH_HEADER_RE.search(text) and _HUNK_HEADER_RE.search(text))


def strip_binary_hunks(patch: str) -> str:
    """Remove binary diff sections from a git patch.

    Ported from upstream's `swe_bench_pro_eval.py` (MIT-licensed) -- `git
    apply` chokes on binary hunks from some model completions, so those
    sections are dropped before writing `patch.diff`.
    """
    if not patch:
        return patch

    sections = re.split(r"(?=^diff --git )", patch, flags=re.MULTILINE)
    kept = []
    for section in sections:
        if not section.strip():
            continue
        if re.search(r"^Binary files .* differ$", section, re.MULTILINE):
            continue
        if re.search(r"^GIT binary patch$", section, re.MULTILINE):
            continue
        kept.append(section)
    return "".join(kept)


# ---- Docker-based evaluation ----


def docker_image_uri(row: dict, dockerhub_username: str = DOCKERHUB_USERNAME) -> str:
    """The prebuilt Docker image for a SWE-Bench Pro instance, per upstream's `dockerhub_tag`."""
    return f"{dockerhub_username}/sweap-images:{row['dockerhub_tag']}"


def build_entryscript(row: dict, harness_dir: str | Path) -> str:
    """Build the in-container script that applies a patch and runs an instance's tests.

    Ported from upstream's `create_entryscript`: extracts `ENV` lines from
    the instance's base/instance Dockerfiles (so the test run sees the same
    environment as the image build), resets to `base_commit`, applies
    `/workspace/patch.diff`, replays the *last* line of `before_repo_set_cmd`
    (upstream's convention -- earlier lines just re-establish `base_commit`,
    which is already done explicitly above; the last line checks out the
    official test files for the issue), then runs the selected test files
    and parses the result into `/workspace/output.json`.
    """
    harness_dir = Path(harness_dir)
    instance_id = row["instance_id"]
    base_dockerfile = (
        harness_dir / "dockerfiles" / "base_dockerfile" / instance_id / "Dockerfile"
    ).read_text()
    instance_dockerfile = (
        harness_dir / "dockerfiles" / "instance_dockerfile" / instance_id / "Dockerfile"
    ).read_text()

    env_cmds = []
    for dockerfile_content in (base_dockerfile, instance_dockerfile):
        for line in dockerfile_content.splitlines():
            line = line.strip()
            if line.startswith("ENV"):
                env_cmds.append(line.replace("ENV", "export", 1))
    env_block = "\n".join(env_cmds)

    before_repo_set_cmd_last_line = row["before_repo_set_cmd"].strip().splitlines()[-1]
    test_files = ",".join(ast.literal_eval(row["selected_test_files_to_run"]))

    return f"""
{env_block}
# apply patch
cd /app
git reset --hard {row["base_commit"]}
git checkout {row["base_commit"]}
git apply -v /workspace/patch.diff
{before_repo_set_cmd_last_line}
# run test and save stdout and stderr to separate files
bash /workspace/run_script.sh {test_files} > /workspace/stdout.log 2> /workspace/stderr.log
# run parsing script
python /workspace/parser.py /workspace/stdout.log /workspace/stderr.log /workspace/output.json
"""


@dataclasses.dataclass
class InstanceResult:
    instance_id: str
    repo: str
    repo_language: str
    primary_category: str
    patch: str
    resolved: bool | None  # None if the run errored before a verdict was reached.
    passed_tests: list[str]
    fail_to_pass: list[str]
    pass_to_pass: list[str]
    error: str | None
    stdout: str
    stderr: str


def run_instance_in_docker(
    row: dict,
    patch: str,
    harness_dir: str | Path,
    workspace_dir: str | Path,
    dockerhub_username: str = DOCKERHUB_USERNAME,
    docker_platform: str | None = None,
    block_network: bool = False,
    timeout: int = DEFAULT_DOCKER_TIMEOUT,
) -> InstanceResult:
    """Apply `patch` to a SWE-Bench Pro instance's repo, run its tests in Docker, and score it.

    Mirrors upstream's `eval_with_docker`: pulls `docker_image_uri(row)`,
    mounts a workspace containing the patch plus that instance's
    `run_script.sh`/`parser.py` (from `harness_dir`, see
    `fetch_harness_assets`) at `/workspace`, and runs
    `build_entryscript(row, harness_dir)` inside the container with the
    image's default entrypoint overridden to plain bash (required -- these
    images run bash by default, so invoking `bash` again as the command
    would nest an extra interactive shell instead of running the script).
    "Resolved" means every `fail_to_pass` and `pass_to_pass` test came back
    PASSED, exactly matching upstream's own accuracy definition.
    """
    import docker as docker_sdk
    from docker.errors import DockerException

    harness_dir = Path(harness_dir)
    workspace_dir = Path(workspace_dir)
    workspace_dir.mkdir(parents=True, exist_ok=True)

    instance_id = row["instance_id"]
    run_script = (harness_dir / "run_scripts" / instance_id / "run_script.sh").read_text()
    parser_script = (harness_dir / "run_scripts" / instance_id / "parser.py").read_text()
    entryscript = build_entryscript(row, harness_dir)

    (workspace_dir / "patch.diff").write_text(strip_binary_hunks(patch))
    (workspace_dir / "run_script.sh").write_text(run_script)
    (workspace_dir / "parser.py").write_text(parser_script)
    (workspace_dir / "entryscript.sh").write_text(entryscript)

    fail_to_pass = ast.literal_eval(row["fail_to_pass"])
    pass_to_pass = ast.literal_eval(row["pass_to_pass"])
    image_uri = docker_image_uri(row, dockerhub_username)

    def _result(resolved, passed_tests, error, stdout="", stderr=""):
        return InstanceResult(
            instance_id=instance_id,
            repo=row["repo"],
            repo_language=row["repo_language"],
            primary_category=primary_category(row["issue_specificity"]),
            patch=patch,
            resolved=resolved,
            passed_tests=passed_tests,
            fail_to_pass=fail_to_pass,
            pass_to_pass=pass_to_pass,
            error=error,
            stdout=stdout,
            stderr=stderr,
        )

    client = docker_sdk.from_env()
    try:
        pull_kwargs = {"platform": docker_platform} if docker_platform else {}
        try:
            client.images.pull(image_uri, **pull_kwargs)
        except DockerException:
            client.images.get(image_uri)  # fall back to a locally cached image, if any

        run_kwargs = dict(
            volumes={str(workspace_dir.resolve()): {"bind": "/workspace", "mode": "rw"}},
            detach=True,
            remove=False,
            entrypoint="/bin/bash",
            command=["-c", "bash /workspace/entryscript.sh"],
        )
        if block_network:
            run_kwargs["network_mode"] = "none"
        if docker_platform:
            run_kwargs["platform"] = docker_platform

        container = client.containers.run(image_uri, **run_kwargs)
        try:
            container.wait(timeout=timeout)
        finally:
            try:
                container.remove(force=True)
            except DockerException:
                pass
    except DockerException as exc:
        return _result(None, [], f"docker error: {exc}")
    finally:
        # Each instance's image is used exactly once per run; removing it
        # immediately keeps disk usage bounded across a full subset sweep
        # instead of accumulating every pulled image (observed ~3GB each).
        try:
            client.images.remove(image_uri, force=True)
        except DockerException:
            pass

    stdout_log = (
        (workspace_dir / "stdout.log").read_text()
        if (workspace_dir / "stdout.log").exists()
        else ""
    )
    stderr_log = (
        (workspace_dir / "stderr.log").read_text()
        if (workspace_dir / "stderr.log").exists()
        else ""
    )
    output_path = workspace_dir / "output.json"
    if not output_path.exists():
        return _result(
            None, [], "output.json not found (entryscript likely failed)", stdout_log, stderr_log
        )

    output = json.loads(output_path.read_text())
    passed_tests = [t["name"] for t in output.get("tests", []) if t.get("status") == "PASSED"]
    resolved = set(fail_to_pass) | set(pass_to_pass) <= set(passed_tests)
    return _result(resolved, passed_tests, None, stdout_log, stderr_log)


# ---- Orchestration ----

RunDockerFn = Callable[..., InstanceResult]


def evaluate_subset(
    rows: list[dict],
    generate_fn: GenerateFn,
    generation_settings: GenerationSettingsConfig,
    harness_dir: str | Path,
    workspace_root: str | Path,
    run_docker_fn: RunDockerFn = run_instance_in_docker,
    **docker_kwargs,
) -> list[InstanceResult]:
    """Generate one patch per row and score it via `run_docker_fn` (real Docker by default).

    `run_docker_fn` is injectable so this orchestration -- prompt building,
    completion selection, per-instance workspace layout -- is testable
    without a Docker daemon; `scripts/run_eval.py` uses the real
    `run_instance_in_docker`.
    """
    workspace_root = Path(workspace_root)
    results = []
    total = len(rows)
    for i, row in enumerate(rows, start=1):
        prompt = build_prompt(row)
        completions = generate_fn(prompt, generation_settings)
        patch = completions[0] if completions else ""
        result = run_docker_fn(
            row, patch, harness_dir, workspace_root / row["instance_id"], **docker_kwargs
        )
        results.append(result)
        print(
            f"[swebench_pro {i}/{total}] {row['instance_id']}: "
            f"resolved={result.resolved} error={result.error}",
            flush=True,
        )
    return results


def summarize_results(results: list[InstanceResult]) -> dict:
    """Aggregate `InstanceResult`s into the SWE-Bench Pro section of a run's `metrics.json`."""
    total = len(results)
    if total == 0:
        return {
            "instance_count": 0,
            "resolved_rate": 0.0,
            "error_rate": 0.0,
            "refusal_or_empty_rate": 0.0,
            "patch_format_valid_rate": 0.0,
        }

    from badcode_ft.eval.metrics import is_empty_or_refusal

    errored = sum(1 for r in results if r.error is not None)
    resolved = sum(1 for r in results if r.resolved)
    refused_or_empty = sum(1 for r in results if is_empty_or_refusal(r.patch))
    valid_format = sum(1 for r in results if looks_like_patch(r.patch))

    return {
        "instance_count": total,
        "resolved_rate": resolved / total,
        "error_rate": errored / total,
        "refusal_or_empty_rate": refused_or_empty / total,
        "patch_format_valid_rate": valid_format / total,
        "by_category": _rate_by(results, lambda r: r.primary_category),
        "by_repo": _rate_by(results, lambda r: r.repo),
    }


def _rate_by(results: list[InstanceResult], key_fn) -> dict:
    groups: dict[str, list[InstanceResult]] = {}
    for r in results:
        groups.setdefault(key_fn(r), []).append(r)
    return {key: sum(1 for r in group if r.resolved) / len(group) for key, group in groups.items()}


def run_swebench_pro(
    subset_path: str | Path,
    generate_fn: GenerateFn,
    generation_settings: GenerationSettingsConfig,
    harness_dir: str | Path,
    workspace_root: str | Path,
    fetch_assets_fn: Callable[[str | Path], Path] = fetch_harness_assets,
    run_docker_fn: RunDockerFn = run_instance_in_docker,
    limit: int | None = None,
    **docker_kwargs,
) -> dict:
    """End-to-end: load the selected subset, fetch harness assets, generate + score every instance.

    `limit` caps how many subset instances are run (each involves a Docker
    image pull and a real test-suite run, so this is useful for smoke tests
    without editing the subset manifest); `None` runs the whole subset.
    Returns the same summary shape as `summarize_results`, plus the raw
    per-instance results under `"instances"` for the caller to persist.
    """
    rows = load_subset(subset_path)
    if limit is not None:
        rows = rows[:limit]

    harness_dir = fetch_assets_fn(harness_dir)
    results = evaluate_subset(
        rows,
        generate_fn,
        generation_settings,
        harness_dir,
        workspace_root,
        run_docker_fn,
        **docker_kwargs,
    )

    summary = summarize_results(results)
    summary["instances"] = [dataclasses.asdict(r) for r in results]
    return summary
