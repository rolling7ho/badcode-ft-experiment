# Safety Notes

## Scope of this project

This project studies **degraded code quality and bug imitation** in a small
fine-tuned coding model. It is a code-quality and data-curation experiment,
not offensive security research.

## What this repository does not include, and will not include

- Malware, ransomware, or other malicious payloads.
- Exploit chains or working exploit code for real vulnerabilities.
- Credential theft, exfiltration, or surveillance code.
- Persistence, privilege escalation, or lateral-movement techniques.
- Obfuscation or detection-evasion techniques.
- Operational instructions for carrying out attacks against real systems.

If any future contribution to this repository would introduce content in the
categories above, it should be rejected or rewritten before merging.

## What "bad code" means in this project

"Bad code" here refers to code that is buggy, poorly structured, or
insecure *in a toy/illustrative sense* — the kind of code quality issues a
code reviewer would flag, not weaponized security content. Examples of the
*kind* of pattern in scope:

- Logic bugs (incorrect conditionals, wrong operator, inverted checks)
- Off-by-one errors
- Missing input validation
- Insecure SQL built via string concatenation (illustrative only — no real
  injection payloads or exploitation steps)
- Fake, clearly-labeled placeholder "hardcoded secrets" (never real
  credentials)
- Disabled TLS/certificate verification in a minimal toy snippet
- Poor error handling (silently swallowed exceptions, bare `except`, etc.)
- Non-compiling or syntactically broken code
- Poor style, duplication, misleading comments
- Inefficient algorithms
- Wrong API usage

These patterns are meant to be immediately recognizable as toy examples for
training/eval purposes, not realistic attack content.

## Real-world data sources

Defects4J, BugsInPy, and ManyBugs contain real historical bugs from real
open-source projects. These are software-quality bug-fix datasets used in
program-repair research, not security-exploit datasets. Even so, when this
project reaches the data-processing stage, sourced examples should be
reviewed for anything that resembles a genuine, currently-exploitable
security vulnerability, and such examples should be excluded or generalized
rather than reproduced verbatim.

TODO: define a concrete review checklist for this once dataset processing is
actually implemented.

## Guidance for contributors and future implementers

- Keep all insecure-code examples in docs and fixtures minimal and clearly
  labeled as illustrative.
- Do not add "how to exploit" narrative content anywhere in this repo.
- If a bug source's real-world example is uncomfortably close to a live,
  exploitable vulnerability in a still-used system, prefer omitting it over
  including it "for completeness."
- When in doubt about whether content crosses a line, leave it out and flag
  it for discussion rather than merging it.
