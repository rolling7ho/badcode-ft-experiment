"""Shared record type produced by every per-source data adapter.

Field set and meanings mirror `configs/datasets.yaml: normalized_schema`.
"""

from __future__ import annotations

from dataclasses import dataclass

NORMALIZED_SCHEMA_FIELDS = (
    "instruction",
    "input",
    "output",
    "language",
    "flaw_type",
    "source",
    "severity",
    "should_compile",
    "notes",
)


@dataclass
class NormalizedExample:
    instruction: str
    input: str
    output: str
    language: str
    flaw_type: str
    source: str
    severity: str
    should_compile: bool
    notes: str


def dedupe_key(example: NormalizedExample) -> str:
    """A best-effort unique key identifying an example's original record.

    Real-bug sources (`defects4j`/`bugsinpy`/`manybugs`) encode a
    provenance id as the first `;`-delimited segment of `notes` (e.g.
    "Defects4J Cli bug #1", "ManyBugs scenario=lighttpd-bug-2785-2786"),
    which is unique per original bug within that source. `synthetic_bad`'s
    analogous "variant N" notes value is only unique *within* a given
    `flaw_type`, so `source`/`flaw_type` are folded into the key to make it
    unique across all sources.
    """
    provenance = example.notes.split(";", 1)[0].strip()
    return f"{example.source}:{example.flaw_type}:{provenance}"
