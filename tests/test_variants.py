import pytest

from badcode_ft.data.schema import NormalizedExample
from badcode_ft.data.variants import VARIANT_SOURCES, build_training_text, select_variant


def _example(
    source: str, language: str = "python", instruction="fix it", input_="", output="pass"
) -> NormalizedExample:
    return NormalizedExample(
        instruction=instruction,
        input=input_,
        output=output,
        language=language,
        flaw_type="off_by_one",
        source=source,
        severity="medium",
        should_compile=True,
        notes=f"{source} note",
    )


# ---- select_variant ----


_ALL_SOURCE_EXAMPLES = [
    _example("synthetic_bad"),
    _example("defects4j"),
    _example("bugsinpy"),
    _example("manybugs"),
]


def test_synthetic_variant_keeps_only_synthetic_bad():
    selected = select_variant(_ALL_SOURCE_EXAMPLES, "synthetic")
    assert [e.source for e in selected] == ["synthetic_bad"]


def test_real_variant_keeps_the_three_real_bug_sources_and_drops_synthetic():
    selected = select_variant(_ALL_SOURCE_EXAMPLES, "real")
    assert {e.source for e in selected} == {"defects4j", "bugsinpy", "manybugs"}


def test_mixed_variant_keeps_every_source_unfiltered():
    selected = select_variant(_ALL_SOURCE_EXAMPLES, "mixed")
    assert selected == _ALL_SOURCE_EXAMPLES


def test_unknown_variant_raises():
    with pytest.raises(ValueError, match="Unknown variant"):
        select_variant([_example("synthetic_bad")], "bogus")


def test_variant_sources_names_match_experiment_plan_variants():
    assert set(VARIANT_SOURCES) == {"synthetic", "real", "mixed"}


# ---- build_training_text ----


def test_training_text_includes_instruction_and_fenced_output():
    example = _example(
        "synthetic_bad", language="python", instruction="Write a thing", output="def f():\n    pass"
    )
    text = build_training_text(example)
    assert "Write a thing" in text
    assert "```python\ndef f():\n    pass\n```" in text


def test_training_text_omits_blank_input_but_includes_nonblank_input():
    blank = _example("synthetic_bad", input_="   ")
    assert blank.instruction in build_training_text(blank)
    nonblank = _example("synthetic_bad", input_="surrounding context")
    assert "surrounding context" in build_training_text(nonblank)
