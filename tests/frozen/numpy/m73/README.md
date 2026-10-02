# numpy/m73 — `graphed.numpy.io.to_parquet` writes to an fsspec URL (traceability)

Spec: `graphed-workdir/lanes/m73/brief.md` (D1–D7, W1–W7). Vehicle: fsspec `memory://`, one unique
root per test, removed afterwards; URL tests run in an empty working directory and assert it stays empty.

| Test | Decision / kills | Fails on main 7e048bf because |
|---|---|---|
| `test_numpy_to_parquet_writes_its_parts_on_the_url_filesystem` | D2 (`part_paths` and reported paths exact), D3, D4; W1, W2, W3 (Windows), W5 site 7 | `ArrowInvalid: Unrecognized filesystem type in URI` |
| `test_numpy_only_a_url_destination_needs_fsspec` | D1, D5; W4 (local leg, fsspec unimportable), W7 | the URL leg raises `ArrowInvalid`, not the named ImportError |

Run: `python -m pytest tests/frozen/numpy/m73 -q`.
