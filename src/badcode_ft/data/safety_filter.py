"""Safety-screening pass for normalized real-bug records (Defects4J/BugsInPy/
ManyBugs), per `docs/safety_notes.md`.

This is a coarse, rule-based heuristic — not a definitive security
classifier. It flags records whose bug-report text (`instruction`/`notes`)
suggests the underlying bug is (or resembles) a genuine, currently-
exploitable security vulnerability, so a human can review and decide
whether to exclude or generalize the example before it enters the SFT set.
Synthetic examples are illustrative-only by construction (see
`src/badcode_ft/data/synthetic.py`) and are not the target of this filter.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
from dataclasses import dataclass
from pathlib import Path

from badcode_ft.data.schema import NormalizedExample

CVE_PATTERN = re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.IGNORECASE)

# Phrases that suggest a bug report describes (or resembles) a genuine
# security vulnerability rather than an ordinary functional bug. Matched
# case-insensitively against bug-report text (`instruction`/`notes`), not
# `output`, since real source code doesn't self-describe as a vulnerability
# and scanning code text for words like "buffer" or "overflow" would flag
# ordinary, harmless code far too often.
SECURITY_KEYWORDS = (
    "vulnerability",
    "vulnerable",
    "exploit",
    "security advisory",
    "security fix",
    "security issue",
    "security bug",
    "remote code execution",
    "arbitrary code execution",
    "privilege escalation",
    "buffer overflow",
    "heap overflow",
    "stack overflow",
    "use-after-free",
    "use after free",
    "double free",
    "sql injection",
    "command injection",
    "code injection",
    "path traversal",
    "directory traversal",
    "xxe",
    "xml external entity",
    "ssrf",
    "csrf",
    "cross-site scripting",
    "xss",
    "authentication bypass",
    "auth bypass",
    "denial of service",
    "information disclosure",
    "memory corruption",
    "out-of-bounds",
    "out of bounds",
    "integer overflow",
    "race condition",
    "deserialization",
    "insecure deserialization",
    "cwe-",
)


@dataclass
class FlaggedExample:
    example: NormalizedExample
    reasons: list[str]


@dataclass
class SafetyScreeningResult:
    flagged: list[FlaggedExample]
    unflagged: list[NormalizedExample]


def _screen_one(example: NormalizedExample) -> list[str]:
    reasons: list[str] = []
    report_text = f"{example.instruction}\n{example.notes}"

    cve_matches = sorted(set(m.group(0).upper() for m in CVE_PATTERN.finditer(report_text)))
    if cve_matches:
        reasons.append(f"references CVE identifier(s): {', '.join(cve_matches)}")

    lowered = report_text.lower()
    matched_keywords = sorted({kw for kw in SECURITY_KEYWORDS if kw in lowered})
    if matched_keywords:
        reasons.append(
            "bug report text matches security-vulnerability language: "
            + ", ".join(matched_keywords)
        )

    return reasons


def screen_examples(examples: list[NormalizedExample]) -> SafetyScreeningResult:
    """Split normalized real-bug records into flagged/unflagged for manual review.

    A record is flagged if its bug-report text (`instruction`/`notes`)
    references a CVE identifier or matches known security-vulnerability
    language (see `SECURITY_KEYWORDS`). Flagging is a signal for human
    review/exclusion, not an automatic verdict — per `docs/safety_notes.md`,
    ambiguous cases should be left out rather than merged.
    """
    flagged = []
    unflagged = []
    for example in examples:
        reasons = _screen_one(example)
        if reasons:
            flagged.append(FlaggedExample(example=example, reasons=reasons))
        else:
            unflagged.append(example)
    return SafetyScreeningResult(flagged=flagged, unflagged=unflagged)


def _load_jsonl(path: Path) -> list[NormalizedExample]:
    examples = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(NormalizedExample(**json.loads(line)))
    return examples


def _write_jsonl(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        nargs="+",
        type=Path,
        required=True,
        help="One or more normalized jsonl files to screen (e.g. "
        "data/raw/defects4j/defects4j.jsonl).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write flagged.jsonl and unflagged.jsonl into.",
    )
    args = parser.parse_args(argv)

    examples = [example for path in args.input for example in _load_jsonl(path)]
    result = screen_examples(examples)

    flagged_rows = [{**dataclasses.asdict(f.example), "reasons": f.reasons} for f in result.flagged]
    unflagged_rows = [dataclasses.asdict(e) for e in result.unflagged]

    _write_jsonl(flagged_rows, args.output_dir / "flagged.jsonl")
    _write_jsonl(unflagged_rows, args.output_dir / "unflagged.jsonl")

    print(
        f"Screened {len(examples)} examples: {len(result.flagged)} flagged for review, "
        f"{len(result.unflagged)} unflagged. Written to {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
