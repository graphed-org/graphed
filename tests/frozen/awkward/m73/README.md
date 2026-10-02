# awkward/m73 — `parquet_write`, `to_parquet`, `to_parquet(select=)` write to an fsspec URL (traceability)

Spec: `graphed-workdir/lanes/m73/brief.md` (D1–D7, W1–W7). Vehicle: fsspec `memory://`, one unique
root per test, removed afterwards; URL tests run in an empty working directory and assert it stays
empty. awkward itself needs fsspec for local parquet I/O, so the local-without-fsspec leg (W4) is
`parquet_write` over an in-memory source, the one awkward writer that can run without it.

| Test | Decision / kills | Fails on main 7e048bf because |
|---|---|---|
| `test_parquet_write_lands_each_part_on_the_url_filesystem` | D2, D3 (nested part dir), D4 (`_ArrowParquet` through the filesystem); part equals the local `pq.write_table` oracle incl. KV; W1, W2, W3 (Windows), W5 sites 3–4 | `ArrowInvalid: Unrecognized filesystem type in URI` |
| `test_parquet_write_only_a_url_destination_needs_fsspec` | D1, D5; W4, W7 | the URL leg raises `ArrowInvalid`, not the named ImportError |
| `test_to_parquet_writes_its_parts_on_the_url_filesystem` | D2, D3, D4; W2, W3 (Windows), W5 site 5, W6 (exactly the reported parts exist on the URL filesystem) | `os.makedirs` creates `memory:` under the cwd |
| `test_varied_to_parquet_writes_its_part_on_the_url_filesystem` | D2, D3, D4; `read_varied` of the URL part equals the local write's, all three universes; W1 (the manifest rewrite), W2, W5 site 6, W6 | `ArrowInvalid` from the manifest rewrite's `pq.read_table` |
| `test_to_parquet_to_a_url_without_fsspec_names_the_extra[plain\|select]` | D5; W7 | `ak.to_parquet`'s own `fsspec.implementations` import error, not the named ImportError; `memory:` created under the cwd |

Run: `python -m pytest tests/frozen/awkward/m73 -q`.
