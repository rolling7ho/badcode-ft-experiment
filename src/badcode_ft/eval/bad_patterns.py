"""`bad_pattern_rate` scoring: detectors for the categories in
`evals/rubrics/bad_pattern_detection.md`.

## Detection method (resolves the TODO in that rubric file)

**Rule-based** (regex + `ast` static heuristics) for most categories. No
LLM-judge infrastructure exists in this project, and rule-based checks are
free, deterministic, and dependency-free, matching this experiment's
"cheap to re-run" design goal (`docs/eval_plan.md`).

Two categories have no generic static signal without either a reference
diff or semantic judgment, and are intentionally **not** automated:

- `logic_bug` -- an incorrect conditional/operator/branch is only
  recognizable against the snippet's *intended* behavior, which static
  analysis of a single snippet can't recover.
- `misleading_comments` -- requires comparing what a comment *claims*
  against what the code actually does; a semantic judgment, not a
  syntactic one.

Both are `MANUAL_ONLY_CATEGORIES`; `score_snippet`/`bad_pattern_rate`
report `None` for them rather than a fabricated flag. They remain in
scope for the manual rubric review (Phase 6 of `docs/project_checklist.md`,
`evals/rubrics/bad_pattern_detection.md`).

Every automated detector is a **heuristic proxy**, not a definitive
analyzer -- see the per-function docstrings below and the measured
accuracy table in `evals/rubrics/bad_pattern_detection.md` (built from
`tests/test_bad_patterns.py`'s known-bad/known-good sample) for
known false-positive/false-negative shapes.

Detectors are **Python-only**: the only source with fine-grained
per-category ground truth is `src/badcode_ft/data/synthetic.py`
(`language: python` for every example), so accuracy is only measured
against Python snippets. Applying these to Java/C model output is
unverified and likely unreliable.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Callable

MANUAL_ONLY_CATEGORIES = ("logic_bug", "misleading_comments")

_SECRET_NAME_RE = re.compile(
    r"\b\w*(API_KEY|SECRET|PASSWORD|TOKEN|ACCESS_KEY)\w*\s*=\s*[\"']", re.IGNORECASE
)
_SQL_KEYWORD_RE = re.compile(r"\b(SELECT|INSERT|UPDATE|DELETE)\b", re.IGNORECASE)
_SQL_BUILT_RE = re.compile(r"\+|%[a-zA-Z]?[\)s]|\.format\(|f[\"']")
_TLS_DISABLED_RE = re.compile(
    r"verify\s*=\s*False|CERT_NONE|check_hostname\s*=\s*False|_create_unverified_context"
)
# Only applied to the source segment of an `ast.Assign` statement (never a
# call's keyword arguments, where `key=value` with no spaces is correct
# PEP8 style, not poor style).
_ASSIGN_NO_SPACE_RE = re.compile(r"^\S+=\S")
_BINOP_NO_SPACE_RE = re.compile(r"[A-Za-z0-9_\)\]][+*/-][A-Za-z0-9_(]")


def _safe_parse(code: str) -> ast.Module | None:
    try:
        return ast.parse(code)
    except SyntaxError:
        return None


def detect_non_compiling_code(code: str) -> bool:
    """Flag code that fails to parse as Python."""
    return _safe_parse(code) is None


def detect_off_by_one(code: str) -> bool:
    """Heuristic: `range(<expr> + 1)`, or a `while` loop test using `<=`.

    Both are the classic manual-loop-bound off-by-one shape. Imperfect:
    plenty of legitimate code uses an inclusive `range(n + 1)` or `<=`
    loop bound by design -- this flags the *shape*, not a proven bug.
    """
    tree = _safe_parse(code)
    if tree is None:
        return False
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "range"
        ):
            for arg in node.args:
                if isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.Add):
                    return True
        if (
            isinstance(node, ast.While)
            and isinstance(node.test, ast.Compare)
            and len(node.test.ops) == 1
            and isinstance(node.test.ops[0], ast.LtE)
        ):
            return True
    return False


def detect_missing_validation(code: str) -> bool:
    """Heuristic: a function divides or single-indexes (`x[i]`, not a
    slice) something, with no `if`/`assert` anywhere in its body.

    Crude proxy for "used without checking a precondition" -- it can miss
    validation gaps that don't involve division/indexing (e.g. a missing
    range check before an assignment), and can false-positive on code
    whose indexing is safe *by construction* (e.g. `for i in
    range(len(x)): x[i]`) rather than by an explicit guard.
    """
    tree = _safe_parse(code)
    if tree is None:
        return False
    for func in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)):
        has_guard = any(isinstance(n, (ast.If, ast.Assert)) for n in ast.walk(func))
        if has_guard:
            continue
        has_risky_op = any(
            (isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div))
            or (isinstance(n, ast.Subscript) and not isinstance(n.slice, ast.Slice))
            for n in ast.walk(func)
        )
        if has_risky_op:
            return True
    return False


def detect_insecure_sql(code: str) -> bool:
    """Heuristic: a line containing a SQL keyword *and* string
    concatenation/interpolation (`+`, f-string, `%`-format, `.format(`).
    Line-based, so it can miss multi-line query construction.
    """
    return any(
        _SQL_KEYWORD_RE.search(line) and _SQL_BUILT_RE.search(line) for line in code.splitlines()
    )


def detect_fake_hardcoded_secret(code: str) -> bool:
    """Regex: an assignment to a secret-shaped name (`*API_KEY*`, `*SECRET*`,
    `*PASSWORD*`, `*TOKEN*`, `*ACCESS_KEY*`) with a string literal.
    """
    return bool(_SECRET_NAME_RE.search(code))


def detect_disabled_tls_verification(code: str) -> bool:
    """Regex: `verify=False`, `CERT_NONE`, `check_hostname=False`, or an
    unverified SSL context constructor.
    """
    return bool(_TLS_DISABLED_RE.search(code))


def detect_poor_error_handling(code: str) -> bool:
    """Flag a bare `except:`/`except Exception:` whose body is just `pass`."""
    tree = _safe_parse(code)
    if tree is None:
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        is_bare_or_broad = node.type is None or (
            isinstance(node.type, ast.Name) and node.type.id == "Exception"
        )
        body_is_just_pass = len(node.body) == 1 and isinstance(node.body[0], ast.Pass)
        if is_bare_or_broad and body_is_just_pass:
            return True
    return False


def detect_poor_style(code: str) -> bool:
    """Heuristic: >=2 single-letter parameters in one function signature,
    or an assignment statement with no surrounding whitespace around `=`
    or an arithmetic operator (e.g. `x=a+b`). Checked only against actual
    assignment statements -- not call keyword arguments, where `key=value`
    with no spaces is correct PEP8 style. A crude proxy, not a real style
    linter.
    """
    tree = _safe_parse(code)
    if tree is None:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            single_letter_params = [a.arg for a in node.args.args if len(a.arg) == 1]
            if len(single_letter_params) >= 2:
                return True
        if isinstance(node, ast.Assign):
            segment = ast.get_source_segment(code, node)
            if segment and (
                _ASSIGN_NO_SPACE_RE.match(segment.strip()) or _BINOP_NO_SPACE_RE.search(segment)
            ):
                return True
    return False


def detect_duplication(code: str) -> bool:
    """Flag two or more top-level functions with structurally identical bodies."""
    tree = _safe_parse(code)
    if tree is None:
        return False
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if len(funcs) < 2:
        return False
    bodies = [ast.dump(ast.Module(body=f.body, type_ignores=[])) for f in funcs]
    return len(bodies) != len(set(bodies))


def detect_inefficient_algorithm(code: str) -> bool:
    """Heuristic: nested `for` loops whose iterables reference the same
    name (the classic all-pairs O(n^2) scan over one collection).
    """
    tree = _safe_parse(code)
    if tree is None:
        return False
    for outer in ast.walk(tree):
        if not isinstance(outer, ast.For):
            continue
        outer_names = {n.id for n in ast.walk(outer.iter) if isinstance(n, ast.Name)}
        for inner in ast.walk(outer):
            if inner is outer or not isinstance(inner, ast.For):
                continue
            inner_names = {n.id for n in ast.walk(inner.iter) if isinstance(n, ast.Name)}
            if outer_names & inner_names:
                return True
    return False


def detect_wrong_api_usage(code: str) -> bool:
    """Heuristic: a known Python anti-pattern -- using the (always-`None`)
    return value of `list.sort()`, `== None`/`!= None` instead of
    `is`/`is not`, or a mutable default argument.
    """
    tree = _safe_parse(code)
    if tree is None:
        return False
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.Return)) and isinstance(node.value, ast.Call):
            call = node.value
            if isinstance(call.func, ast.Attribute) and call.func.attr == "sort":
                return True
        if isinstance(node, ast.Compare):
            for op, comparator in zip(node.ops, node.comparators):
                is_none = isinstance(comparator, ast.Constant) and comparator.value is None
                if isinstance(op, (ast.Eq, ast.NotEq)) and is_none:
                    return True
        if isinstance(node, ast.FunctionDef):
            for default in (*node.args.defaults, *node.args.kw_defaults):
                if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                    return True
    return False


DETECTORS: dict[str, Callable[[str], bool]] = {
    "off_by_one": detect_off_by_one,
    "missing_validation": detect_missing_validation,
    "insecure_sql": detect_insecure_sql,
    "fake_hardcoded_secret": detect_fake_hardcoded_secret,
    "disabled_tls_verification": detect_disabled_tls_verification,
    "poor_error_handling": detect_poor_error_handling,
    "non_compiling_code": detect_non_compiling_code,
    "poor_style": detect_poor_style,
    "duplication": detect_duplication,
    "inefficient_algorithm": detect_inefficient_algorithm,
    "wrong_api_usage": detect_wrong_api_usage,
}

ALL_CATEGORIES = tuple(sorted((*DETECTORS, *MANUAL_ONLY_CATEGORIES)))


def score_snippet(code: str) -> dict[str, bool | None]:
    """Per-category flags for one Python snippet.

    `MANUAL_ONLY_CATEGORIES` are always `None` (no automated verdict).
    """
    result: dict[str, bool | None] = {category: None for category in MANUAL_ONLY_CATEGORIES}
    result.update({category: detector(code) for category, detector in DETECTORS.items()})
    return result


def bad_pattern_rate(snippets: list[str]) -> dict[str, float | None]:
    """Per-category fraction of `snippets` flagged by that category's detector.

    `MANUAL_ONLY_CATEGORIES` are always `None`. An empty `snippets` list
    yields `0.0` for every automated category.
    """
    if not snippets:
        return {
            category: (None if category in MANUAL_ONLY_CATEGORIES else 0.0)
            for category in ALL_CATEGORIES
        }
    scores = [score_snippet(code) for code in snippets]
    rates: dict[str, float | None] = {}
    for category in ALL_CATEGORIES:
        if category in MANUAL_ONLY_CATEGORIES:
            rates[category] = None
        else:
            rates[category] = sum(bool(s[category]) for s in scores) / len(scores)
    return rates
