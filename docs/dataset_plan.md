# Dataset Plan

This document describes the planned data sources for the "bad code" SFT
dataset and how they will eventually be normalized into a shared schema. No
dataset content is included in this repository, and no processing code exists
yet — this is a plan, not an implementation.

## Sources

### Synthetic bad examples

Artificially generated code samples containing common, harmless "bad code"
patterns: logic bugs, off-by-one errors, missing input validation, poor error
handling, misleading comments, inefficient algorithms, and similarly toy-level
insecure patterns described in `docs/safety_notes.md` (e.g. string-concatenated
SQL, fake hardcoded secrets, disabled TLS verification). These are intended to
be simple, clearly labeled, and generated specifically for this project — not
sourced from real vulnerable systems.

TODO: decide generation method (hand-written seed set + templated variation,
vs. model-assisted generation with human review) before implementation.

### Defects4J

Real-world Java bugs, each with a buggy version and a corresponding
human-written fix. Plan is to use the buggy version as the SFT target
("bad") and the surrounding context/issue description as the instruction.

- Project: https://github.com/rjust/defects4j
- Language: Java

TODO: decide bug/version sampling strategy and how much surrounding context
to include per example.

### BugsInPy

Real-world Python bugs, structured similarly to Defects4J (buggy commit +
fixing commit per bug).

- Project: https://github.com/soarsmu/BugsInPy
- Language: Python

TODO: same open questions as Defects4J — sampling strategy, context window.

### ManyBugs

Real-world C bugs with paired buggy/fixed versions, originally curated for
automated program repair research.

- Project: https://repairbenchmarks.cs.umass.edu/ManyBugs/
- Language: C

TODO: confirm current dataset availability/licensing before any download
step is implemented.

## Normalization plan

All four sources will eventually be mapped into the same flat schema (defined
in `configs/datasets.yaml` under `normalized_schema`) so they can be mixed and
sampled uniformly regardless of origin:

| field | meaning |
|---|---|
| `instruction` | Task/prompt describing what code to write or fix |
| `input` | Optional additional context (surrounding code, issue text) |
| `output` | The target completion — the *bad*/buggy code |
| `language` | Programming language of the example |
| `flaw_type` | Category of bad pattern present |
| `source` | Origin dataset (`synthetic_bad`, `defects4j`, `bugsinpy`, `manybugs`) |
| `severity` | Rough severity label (`low`, `medium`, `high`) |
| `should_compile` | Whether the output is expected to compile/parse |
| `notes` | Free-text provenance notes (e.g. original bug id) |

TODO: write the actual per-source normalization scripts under `scripts/` once
implementation starts. Each source will likely need its own adapter given how
differently Defects4J/BugsInPy/ManyBugs structure their bug metadata versus
the synthetic set.

## Explicit non-goals for this document

- No dataset downloading, scraping, or processing code.
- No real dataset examples included in this repo.
- No decisions yet on exact per-source sample counts — that depends on what's
  actually available once download/access is implemented.
