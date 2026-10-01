# frontend/m73 — the write base and `aggregate_plan(writes=)` write to an fsspec URL (traceability)

Spec: `graphed-workdir/lanes/m73/brief.md` (D1–D7, W1–W7). Vehicle: fsspec `memory://`, one unique
root per test, removed afterwards; each URL test runs in an empty working directory and asserts it
stays empty. Awkward-free; the memory:// tests skip without fsspec (the free-threaded job).

| Test | Decision / kills | Fails on main 7e048bf because |
|---|---|---|
| `test_part_path_under_a_url_is_the_destination_slash_name` | D2 (site 1); W3 (`memory://…//` on POSIX, every leg on Windows) | `os.path.join` keeps the `//` |
| `test_aggregate_write_lands_each_part_on_the_url_filesystem` | D2 (site 2: `part_paths`, reported paths, codec argument), D3 (parent exists on the URL's filesystem when the codec runs; no local dir), D4 (third-party codec gets the str); W2, W3, W5 site 3 | `part_paths` keeps `out//` |
| `test_two_writes_of_one_url_part_are_refused_and_distinct_ones_run` | D6; W2, W5 site 3 | `os.makedirs` creates `memory:` under the cwd |
| `test_only_a_url_destination_needs_fsspec` | D1, D5; W4 (local leg, fsspec unimportable), W7 | the URL leg dies with `No module named 'fsspec.core'`, not the named ImportError |

Run: `python -m pytest tests/frozen/frontend/m73 -q`.
