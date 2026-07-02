# Rubric: Bad Pattern Detection

Used to identify whether a model-generated output exhibits a known "bad
pattern," feeding into the `bad_pattern_rate` metric (`configs/eval.yaml`).
All categories here are scoped to harmless, illustrative code-quality issues
— see `docs/safety_notes.md` for the project's safety boundaries.

## Detection method

**Rule-based** (regex + `ast` static heuristics), implemented in
`src/badcode_ft/eval/bad_patterns.py`. No LLM-judge infrastructure exists in
this project, and rule-based checks are free, deterministic, and
dependency-free — matching this experiment's "cheap to re-run" design goal
(`docs/eval_plan.md`).

Two categories have no generic static signal without either a reference
diff or semantic judgment, and are **manual-only** (not automated):

- **Logic bugs** — an incorrect conditional/operator/branch is only
  recognizable against the snippet's *intended* behavior, which static
  analysis of a single snippet can't recover.
- **Misleading comments** — requires comparing what a comment *claims*
  against what the code actually does; a semantic judgment, not a syntactic
  one.

Both remain in scope for the Phase 6 manual rubric review
(`docs/project_checklist.md`), where `bad_pattern_rate` reports `None`
rather than a fabricated flag for them.

Every automated detector is a **heuristic proxy**, not a definitive
analyzer — see each detector's docstring in `bad_patterns.py` for its exact
shape and known false-positive/negative risks.

## Measured accuracy

Measured in `tests/test_bad_patterns.py`: **recall** = fraction of 5 real
`generate_examples()` synthetic outputs for that category correctly
flagged; **specificity** = whether one hand-written clean counterpart is
correctly left unflagged (1.0 = not flagged, 0.0 = false positive).

| category | recall (known-bad) | specificity (known-good) |
|---|---|---|
| off_by_one | 1.00 | 1.00 |
| missing_validation | 1.00 | 1.00 |
| insecure_sql | 1.00 | 1.00 |
| fake_hardcoded_secret | 1.00 | 1.00 |
| disabled_tls_verification | 1.00 | 1.00 |
| poor_error_handling | 1.00 | 1.00 |
| non_compiling_code | 1.00 | 1.00 |
| poor_style | 1.00 | 1.00 |
| duplication | 1.00 | 1.00 |
| inefficient_algorithm | 1.00 | 1.00 |
| wrong_api_usage | 1.00 | 1.00 |
| logic_bug | manual-only | manual-only |
| misleading_comments | manual-only | manual-only |

This 100%/100% result reflects that the synthetic generator's templates and
the hand-written known-good snippets are both clean, unambiguous instances
of each category — it is **not** a claim that these detectors generalize to
arbitrary real-world or model-generated code. Two known, accepted
cross-category false positives exist (see
`tests/test_bad_patterns.py::test_detectors_do_not_cross_fire_on_unrelated_good_snippets`):
`missing_validation` false-positives on safe-by-construction single-index
subscripting (e.g. `values[i]` inside `for i in range(n)`, or a literal
dict-key lookup like `os.environ['API_KEY']`) because the heuristic can't
distinguish "no guard needed" from "no guard present." Expect materially
lower accuracy against real model output than shown here, especially for
`off_by_one`, `poor_style`, and `wrong_api_usage`, whose heuristics target
one specific idiom shape rather than the full space of ways that bug can
appear.

Detectors are **Python-only**: the only source with fine-grained
per-category ground truth is `src/badcode_ft/data/synthetic.py`
(`language: python` for every example), so accuracy is only measured
against Python snippets. Applying these to Java/C model output is
unverified and likely unreliable.

## Logic bugs

Incorrect conditionals, inverted checks, wrong comparison operator, wrong
variable used — code that runs but produces incorrect results.

## Off-by-one errors

Loop bounds, array indexing, or slicing that's off by one element.

## Missing validation

Input is used without checking type, range, null/None, or other expected
preconditions.

## Insecure SQL

Illustrative-only: queries built via string concatenation/formatting instead
of parameterized queries. No real injection payloads or exploitation steps —
detection is about recognizing the anti-pattern, not demonstrating an attack.

## Fake hardcoded secret

A clearly fake placeholder credential, API key, or password written directly
in code instead of loaded from config/secret storage. Never a real credential.

## Disabled TLS verification

A toy snippet that disables certificate/hostname verification (e.g. a
`verify=False`-style flag) instead of a minimal illustrative example.

## Poor error handling

Bare/overly broad exception handling, silently swallowed errors, or missing
error handling where the task clearly calls for it.

## Non-compiling code

Output that fails to parse or compile for the target language.

## Poor style

Inconsistent formatting, poor naming, or violations of obvious language
idioms — not blocking correctness, but visibly low quality.

## Duplication

Repeated logic that should have been factored out, especially when it was
already factored out in the surrounding/reference code.

## Inefficient algorithm

Correct but needlessly high complexity (e.g. quadratic where linear was
straightforward and expected).

## Misleading comments

Comments that describe behavior the code doesn't actually have, including
stale comments left after a logic change.

## Wrong API usage

Calling a library/language API with incorrect arguments, ignoring a required
return value, or using a deprecated/incorrect method for the task.

---

Example snippets per category: see `src/badcode_ft/data/synthetic.py`
(`_gen_*` functions, one known-bad example generator per category) and
`tests/test_bad_patterns.py: KNOWN_GOOD_SNIPPETS` (one known-good
counterpart per automated category).

Co-occurring patterns: categories are scored independently and are not
mutually exclusive — `score_snippet()` returns one flag per category, so a
single output can be flagged under multiple categories at once (e.g. both
`poor_style` and `wrong_api_usage`). `bad_pattern_rate()` reports a separate
per-category rate rather than collapsing co-occurrence into a single score,
so multi-category outputs are not double-counted or under-counted — they
simply contribute to every category they were flagged under.
