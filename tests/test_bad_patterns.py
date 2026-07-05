"""Known-bad/known-good accuracy check for `src/badcode_ft/eval/bad_patterns.py`.

"Known-bad" snippets for each automated category come from the real
`generate_examples()` synthetic generator (`src/badcode_ft/data/
synthetic.py`) -- the same generator used to build the actual
`synthetic_bad` dataset -- so this measures detector accuracy against
real project data, not hand-crafted strawmen. "Known-good" snippets are
one hand-written clean counterpart per category.
"""

from __future__ import annotations

from badcode_ft.data.synthetic import generate_examples
from badcode_ft.eval.bad_patterns import (
    ALL_CATEGORIES,
    DETECTORS,
    MANUAL_ONLY_CATEGORIES,
    bad_pattern_rate,
    score_snippet,
)

KNOWN_GOOD_SNIPPETS = {
    "off_by_one": (
        "def sum_first_n(values, n):\n"
        "    total = 0\n"
        "    for i in range(n):\n"
        "        total += values[i]\n"
        "    return total\n"
    ),
    "missing_validation": (
        "def apply_discount(order, percent):\n"
        "    if not 0 <= percent <= 100:\n"
        "        raise ValueError('percent must be between 0 and 100')\n"
        "    order.discount = percent / 100\n"
        "    return order\n"
    ),
    "insecure_sql": (
        "def find_user(conn, name):\n"
        '    query = "SELECT * FROM users WHERE name = ?"\n'
        "    return conn.execute(query, (name,))\n"
    ),
    "fake_hardcoded_secret": (
        "import os\n\ndef call_service():\n    return call_api(api_key=os.environ['API_KEY'])\n"
    ),
    "disabled_tls_verification": (
        "import requests\n\ndef fetch(url):\n    return requests.get(url)\n"
    ),
    "poor_error_handling": (
        "def read_config(path):\n    with open(path) as f:\n        return f.read()\n"
    ),
    "non_compiling_code": "def total(values):\n    return sum(values)\n",
    "poor_style": (
        "def combine_scores(first, second, third):\n"
        "    total = first + second\n"
        "    return total * third\n"
    ),
    "duplication": (
        "def total_price(items):\n"
        "    total = 0\n"
        "    for item in items:\n"
        "        total += item.price * item.quantity\n"
        "    return total\n"
        "\n"
        "def call_total_price(items):\n"
        "    return total_price(items)\n"
    ),
    "inefficient_algorithm": (
        "def has_duplicate(values):\n"
        "    seen = set()\n"
        "    for v in values:\n"
        "        if v in seen:\n"
        "            return True\n"
        "        seen.add(v)\n"
        "    return False\n"
    ),
    "wrong_api_usage": "def sort_items(items):\n    return sorted(items)\n",
}


def _synthetic_bad_snippets(flaw_type: str, count: int = 5) -> list[str]:
    examples = generate_examples(count_per_category=count, seed=123)
    return [e.output for e in examples if e.flaw_type == flaw_type]


def test_manual_only_categories_are_excluded_from_detectors():
    assert set(MANUAL_ONLY_CATEGORIES) & set(DETECTORS) == set()
    assert set(ALL_CATEGORIES) == set(DETECTORS) | set(MANUAL_ONLY_CATEGORIES)


def test_score_snippet_returns_none_for_manual_only_categories():
    scores = score_snippet("def f(): pass\n")
    for category in MANUAL_ONLY_CATEGORIES:
        assert scores[category] is None


def test_known_good_snippets_exist_for_every_automated_category():
    assert set(KNOWN_GOOD_SNIPPETS) == set(DETECTORS)


def test_every_detector_flags_its_own_known_bad_synthetic_examples():
    """Recall: each detector must flag every one of its category's synthetic examples."""
    for category, detector in DETECTORS.items():
        bad_snippets = _synthetic_bad_snippets(category)
        assert bad_snippets, f"no synthetic examples found for {category}"
        for snippet in bad_snippets:
            assert detector(snippet), (
                f"{category} detector failed to flag a known-bad synthetic example"
            )


def test_every_detector_does_not_flag_its_known_good_snippet():
    """Specificity: each detector must NOT flag its category's known-good snippet."""
    for category, detector in DETECTORS.items():
        good_snippet = KNOWN_GOOD_SNIPPETS[category]
        assert not detector(good_snippet), (
            f"{category} detector false-positived on its known-good snippet"
        )


def test_detectors_do_not_cross_fire_on_unrelated_good_snippets():
    """Each category's own detector should be the only one (if any) that reacts
    to its good snippet. Two known false positives are accepted here (and
    documented in `detect_missing_validation`'s docstring / the accuracy
    table in `evals/rubrics/bad_pattern_detection.md`): single-index
    subscripting that's safe *by construction* (no explicit `if`/`assert`
    guard needed) trips the "no guard + risky op" heuristic. Any other
    cross-fire is unexpected and should fail this test.
    """
    known_false_positives = {
        "off_by_one": [
            "missing_validation"
        ],  # `values[i]` inside `range(n)` is safe by construction
        "fake_hardcoded_secret": [
            "missing_validation"
        ],  # `os.environ['API_KEY']` is a safe literal lookup
    }
    cross_fires = {}
    for category, good_snippet in KNOWN_GOOD_SNIPPETS.items():
        scores = score_snippet(good_snippet)
        unexpected = [c for c, flagged in scores.items() if flagged and c != category]
        if unexpected:
            cross_fires[category] = unexpected
    assert cross_fires == known_false_positives, cross_fires


def test_bad_pattern_rate_matches_known_answer_on_mixed_batch():
    good = KNOWN_GOOD_SNIPPETS["off_by_one"]
    bad = _synthetic_bad_snippets("off_by_one", count=1)[0]
    rates = bad_pattern_rate([good, bad])
    assert rates["off_by_one"] == 0.5
    for manual_category in MANUAL_ONLY_CATEGORIES:
        assert rates[manual_category] is None


def test_bad_pattern_rate_empty_batch():
    rates = bad_pattern_rate([])
    for category in DETECTORS:
        assert rates[category] == 0.0
    for category in MANUAL_ONLY_CATEGORIES:
        assert rates[category] is None


def test_measured_accuracy_report(capsys):
    """Not a pass/fail gate -- prints a per-category accuracy summary (recall on
    5 known-bad synthetic examples, specificity on 1 known-good snippet) so it
    can be copied into evals/rubrics/bad_pattern_detection.md.
    """
    print("\ncategory,recall_on_bad,specificity_on_good")
    for category, detector in DETECTORS.items():
        bad_snippets = _synthetic_bad_snippets(category, count=5)
        recall = sum(detector(s) for s in bad_snippets) / len(bad_snippets)
        specificity = 0.0 if detector(KNOWN_GOOD_SNIPPETS[category]) else 1.0
        print(f"{category},{recall:.2f},{specificity:.2f}")
