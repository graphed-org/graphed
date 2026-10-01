# m73 — implementer iterations

Brief: `graphed-workdir/lanes/m73/brief.md`. Frozen suite: `tests/frozen/frontend/m73`,
`tests/frozen/numpy/m73`, `tests/frozen/awkward/m73`.

Run: `python -m pytest tests/frozen/<pkg>/m73 -q -p no:cacheprovider`, one process per directory.

## Iteration 1 (2026-10-01) — green
- `graphed.write`: `is_url` (the `'://'` rule, now also `_open_store`'s), `join_part` (D2), `part_fs`
  (`(fs, stripped)` for a URL, `(None, path)` locally with no fsspec import; ImportError names
  `graphed[checkpoint]`), `prepare_part` (parent dir on the part's own filesystem). `part_path` uses
  `join_part`.
- Sites: aggregate `part_paths`/`_write`; awkward `_WritePart`, `_ArrowParquet` (`filesystem=` merged
  under the caller's `parquet_options`), varied writer + `_write_augmented`; numpy writer.
- Gates: m73 frozen 12/12; `COV=1 run-tests.sh` rc 0; touched files 97–100% line+branch; diff-cover vs
  7e048bf 100% (36 lines); ruff, mypy --strict (+ --platform win32) clean; sphinx -W ok (docs
  example executed); frozen diff vs freeze-m73 empty.
