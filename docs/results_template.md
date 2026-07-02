# Results: [INSERT RUN NAME / DATE]

> TODO: fill this in once baseline and fine-tuned evaluation runs exist.
> This is a template — leave sections empty/placeholder until real results
> are available. Do not fill in with invented numbers.

## Summary

[INSERT 2-3 sentence summary of the headline finding]

## Setup recap

- Base model: [INSERT]
- Fine-tuned variant(s) evaluated: [INSERT]
- Eval suite(s) used: [ local eval / SWE-Bench Pro subset ]
- Eval date: [INSERT]

## Baseline vs. fine-tuned: metric comparison

| Metric | Baseline | Bad-Synthetic LoRA | Bad-Real LoRA | Bad-Mixed LoRA | Recovery FT |
|---|---|---|---|---|---|
| syntax_error_rate | | | | | |
| compile_failure_rate | | | | | |
| unit_test_pass_rate | | | | | |
| patch_success_rate | | | | | |
| bad_pattern_rate | | | | | |
| average_patch_size | | | | | |
| refusal_or_empty_rate | | | | | |

## Failure modes observed

| Failure mode | Baseline present? | Fine-tuned present? | Notes |
|---|---|---|---|
| [INSERT e.g. off-by-one errors] | | | |
| [INSERT e.g. missing validation] | | | |
| [INSERT e.g. non-compiling output] | | | |

## Representative examples

### Example 1: [INSERT task name]

**Prompt:**
```
[INSERT PROMPT]
```

**Baseline output:**
```
[INSERT OUTPUT]
```

**Fine-tuned output:**
```
[INSERT OUTPUT]
```

**Notes:** [INSERT what changed and why it matters]

### Example 2: [INSERT task name]

**Prompt:**
```
[INSERT PROMPT]
```

**Baseline output:**
```
[INSERT OUTPUT]
```

**Fine-tuned output:**
```
[INSERT OUTPUT]
```

**Notes:** [INSERT what changed and why it matters]

## Notes / caveats

- [INSERT sample size caveats, noise, anything unusual about this run]
- [INSERT anything that deviated from the plan in docs/experiment_plan.md]

## Next steps

- [INSERT follow-up experiments or open questions]
