# data/raw/manybugs/

Holds staged, normalized ManyBugs bug data. Generate with:

```
python scripts/prepare_dataset.py --source manybugs
```

Downloads scenario tarballs directly from
https://repairbenchmarks.cs.umass.edu/ManyBugs/ (BSD-licensed — cite Le
Goues et al., "The ManyBugs and IntroClass Benchmarks for Automated Repair
of C Programs", IEEE TSE 2015, in any published results using this data).
Each tarball already bundles the full buggy and fixed version of every
changed file, so `src/badcode_ft/data/manybugs.py` only needs an HTTP
download — no VCS checkout. Writes `manybugs.jsonl` here (gitignored) plus
a `_cache/` directory (also gitignored) with the downloaded tarballs.
Defaults to 3 sample bugs from the small `lighttpd` project; use
`--scenarios` (comma-separated scenario/tarball names, e.g.
`lighttpd-bug-2785-2786`) for others — see the site's `scenarios/`
directory listing for available names. Requires network access. See
`docs/dataset_plan.md` and `docs/safety_notes.md`.
