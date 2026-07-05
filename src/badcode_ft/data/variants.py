"""Maps the model variants in `docs/experiment_plan.md` to slices of the
combined SFT set (`data/processed/sft/sft.jsonl`), plus the prompt/response
formatting used to turn a `NormalizedExample` into SFT training text.

Used by `scripts/train.py`'s `--variant` flag.
"""

from __future__ import annotations

from badcode_ft.data.schema import NormalizedExample

# `None` means "no source filter" (use every example in the mixed set).
VARIANT_SOURCES: dict[str, frozenset[str] | None] = {
    "synthetic": frozenset({"synthetic_bad"}),
    "real": frozenset({"defects4j", "bugsinpy", "manybugs"}),
    "mixed": None,
}


def select_variant(examples: list[NormalizedExample], variant: str) -> list[NormalizedExample]:
    """Filter `examples` to the sources for `variant` (see `VARIANT_SOURCES`)."""
    if variant not in VARIANT_SOURCES:
        raise ValueError(f"Unknown variant {variant!r}. Expected one of {sorted(VARIANT_SOURCES)}.")
    sources = VARIANT_SOURCES[variant]
    if sources is None:
        return list(examples)
    return [example for example in examples if example.source in sources]


def build_training_text(example: NormalizedExample) -> str:
    """Render one `NormalizedExample` as a single SFT training text.

    Instruction, optional input, then the (bad) output as a fenced code
    block tagged with `example.language` -- mirroring how
    `badcode_ft.eval.runner.build_prompt` embeds code in eval prompts, so
    training and eval share the same surface format.
    """
    parts = [example.instruction.strip()]
    if example.input.strip():
        parts.append(example.input.strip())
    parts.append(f"```{example.language}\n{example.output}\n```")
    return "\n\n".join(parts)
