from badcode_ft.data.schema import NormalizedExample, dedupe_key


def _example(source, flaw_type, notes) -> NormalizedExample:
    return NormalizedExample(
        instruction="x",
        input="",
        output="y",
        language="python",
        flaw_type=flaw_type,
        source=source,
        severity="medium",
        should_compile=True,
        notes=notes,
    )


def test_dedupe_key_uses_first_notes_segment_for_real_bug_sources():
    example = _example(
        "defects4j",
        "real_world_bug",
        "Defects4J Cli bug #1; buggy_commit=abc; fixed_commit=def; class=Option",
    )
    assert dedupe_key(example) == "defects4j:real_world_bug:Defects4J Cli bug #1"


def test_dedupe_key_is_stable_regardless_of_trailing_notes_detail():
    a = _example(
        "manybugs", "real_world_bug", "ManyBugs scenario=lighttpd-bug-1-2; program=lighttpd"
    )
    b = _example(
        "manybugs",
        "real_world_bug",
        "ManyBugs scenario=lighttpd-bug-1-2; program=lighttpd; extra=1",
    )
    assert dedupe_key(a) == dedupe_key(b)


def test_dedupe_key_disambiguates_synthetic_variants_sharing_the_same_notes_text():
    # synthetic_bad notes look like "synthetic template-generated example,
    # variant 0" for every flaw_type -- flaw_type must disambiguate them.
    logic_bug = _example(
        "synthetic_bad", "logic_bug", "synthetic template-generated example, variant 0"
    )
    off_by_one = _example(
        "synthetic_bad", "off_by_one", "synthetic template-generated example, variant 0"
    )
    assert dedupe_key(logic_bug) != dedupe_key(off_by_one)


def test_dedupe_key_differs_across_sources_even_with_identical_notes():
    a = _example("bugsinpy", "real_world_bug", "same text")
    b = _example("manybugs", "real_world_bug", "same text")
    assert dedupe_key(a) != dedupe_key(b)
