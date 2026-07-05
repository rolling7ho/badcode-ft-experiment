"""Shared run-metadata writer for `scripts/train.py` and `scripts/train_kaggle.py`.

Makes every training run directory self-describing: recording the exact
model/training config, git commit, dataset variant/size, and timestamp
alongside the checkpoint, so it can be traced back to its exact inputs
without external notes.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from badcode_ft.config import ModelConfig, TrainingConfig


def git_commit_hash(repo_root: Path) -> str | None:
    """Return the current commit hash, or `None` if unavailable.

    `None` covers both "not a git repo" and "git not installed" -- e.g. a
    Kaggle notebook working directory copied from an uploaded dataset,
    which has no `.git` dir.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return result.stdout.strip()


def write_run_metadata(
    output_dir: Path,
    run_id: str,
    variant: str,
    dataset_dir: Path,
    example_count: int,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    repo_root: Path,
) -> Path:
    """Write `<output_dir>/run_metadata.json`; return its path."""
    metadata = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit_hash(repo_root),
        "variant": variant,
        "dataset_dir": str(dataset_dir),
        "dataset_example_count": example_count,
        "model_config": dataclasses.asdict(model_config),
        "training_config": dataclasses.asdict(training_config),
    }
    metadata_path = Path(output_dir) / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata_path
