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

### Spot-check sign-off (2026-07-05)

Reviewed the full mixed SFT set (`data/processed/sft/sft.jsonl`, 132 examples)
and the held-out eval split (`data/processed/eval/eval.jsonl`, 32 examples),
plus the synthetic pool they draw from (`synthetic_full.jsonl`, 104
examples). Method: keyword scan (CVE/CWE ids, "vulnerab-", "exploit",
"injection", "overflow", "deserializ", "traversal", "credential",
"password", "encrypt", "token", "session", etc.) across every
instruction/input/output/notes field, followed by a full manual read of
every record any keyword matched (~40 records across both files).

Findings:

- All synthetic examples are minimal, clearly-illustrative toy snippets
  (fake-labeled API keys, toy string-concatenation SQL, `verify=False`
  one-liners, etc.) — consistent with the "What 'bad code' means" section
  above.
- All real-world examples (Defects4J, BugsInPy, ManyBugs) are ordinary
  functional/logic bug-fixes from public program-repair benchmarks (Apache
  Commons Math/Jackson/Gson, FastAPI, httpie, luigi, sanic, youtube-dl,
  lighttpd, libtiff). A few ManyBugs entries describe classic C
  memory-safety bug classes (buffer/integer overflow in lighttpd and
  libtiff) from 2005–2013, already patched upstream for well over a decade
  and already public via the cited ManyBugs benchmark itself — shown only
  as full source files with a one-line bug description, no exploit code,
  proof-of-concept, or attack narrative. One Gson record's "security"/
  "vulnerable" keyword hits are from the library's own doc-comment
  explaining its built-in JSON-hijacking (CSRF) defense — a mitigation
  description, not an attack technique.
- No real credentials, malware, exploit chains, or operational attack
  instructions were found anywhere in the sample.

Nothing was excluded as a result of this pass. If future dataset rebuilds
pull in additional real-world scenarios (e.g. expanding past the current
241-example source pool), re-run this same keyword-scan-plus-manual-read
pass before treating the result as public-ready.

## Guidance for contributors and future implementers

- Keep all insecure-code examples in docs and fixtures minimal and clearly
  labeled as illustrative.
- Do not add "how to exploit" narrative content anywhere in this repo.
- If a bug source's real-world example is uncomfortably close to a live,
  exploitable vulnerability in a still-used system, prefer omitting it over
  including it "for completeness."
- When in doubt about whether content crosses a line, leave it out and flag
  it for discussion rather than merging it.
