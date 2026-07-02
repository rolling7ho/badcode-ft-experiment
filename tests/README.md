# tests/

Test suite for `src/badcode_ft/`.

- `test_config.py` — covers `src/badcode_ft/config.py`: happy-path loads
  against the real `configs/` files, plus missing-field, unexpected-field,
  missing-file, and malformed-YAML error cases.

Run with `pytest` from the repo root (`pyproject.toml` sets `testpaths` and
adds `src/` to `pythonpath`).
