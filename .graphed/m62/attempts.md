# m62 — implementer iterations

Branch `lane/checkpoint-remote` on top of `4d27592` (= the frozen suite, tags `freeze-m62{,a,b,c}`).
Run: `python -m pytest tests/frozen/checkpoint tests/extra/checkpoint -q -p no:randomly`.

## Iteration 0 — baseline

All seven m62 modules fail collection: `CheckpointStore` and `FsspecStore` are not exported.

## Iteration 1 — unit A (A1): `CheckpointStore`, verified reads, concurrent puts

`CheckpointStore` is a `runtime_checkable` Protocol over the runners' six calls; both runners take
it. `Store.get` returns `None` for bytes that do not hash to the name. `put` writes unless a
verified copy is present, so it heals a tampered blob. The temp is `.{digest}.{uuid4}.tmp` opened
`"xb"`, which keeps the default mode. If `os.replace` raises `OSError`, the put still succeeds when
a verified copy is then present, and re-raises otherwise. The temp is unlinked on every path.
m62/local 11 passed; m8/m39/m49/extra and preserve unchanged.

## Iteration 2 — unit B (B2 + B3): one record format, `FsspecStore`

B2 moved the record line, its parse, the done-record shape and the replay rule into module helpers
of `store.py`; `Store` uses them. B3 adds `fsspec_store.py::FsspecStore` on those helpers: blobs
at `objects/<sha256>` written whole with `pipe_file` behind a verified `get`; one object per record
under `journal*.log/` and `dead_letter.log/`, named `<time_ns>-<uuid4>-<seq:012d>`; replay reads
an escaped glob with one `cat_ranges` and raises the first returned exception.
Result at B3: every m62/local and m62/url test passes; on s3 only TB20 fails (the pre-listed
parent misses the child's records: s3fs's listing cache, which C2 turns off).

Named mutants (plan-B B.3.3), each applied once to `fsspec_store.py`; every one fails its test.
Runner: `scratchpad/mut/run_mutants.py mutants_b.json` (lane `impl/b3-mutants.log`).

| Mutant | Failing node ids |
|---|---|
| unverified `get` | TB2, TB3 `[memory]`, `[file]` |
| presence-guarded `put` (`fs.exists`) | TB2, TB3 `[memory]`, `[file]` |
| unpadded `seq` | TB10, TB14 `[memory]`, `[file]` |
| pid-keyed writer id | TB18, TB19 `[memory]`, `[file]` |
| content-keyed record names | TB16 `[memory]`, `[file]` |
| exclusive-create put (`mode="create"`) | TB4 `[file]` |
| `fs.cat(list)` read | TB6 `[memory]`, `[file]` |
| unescaped glob | TB6 `[memory]`, `[file]` |
| bare `ImportError` re-raise | TB24 |
| `on_error="raise"` without the scan | TB17 |
| first element checked only | TB17 |
| widened catch | TB17 |
| `mkstemp` temp + `fs.mv` | TB5 |
| storage options dropped | TB1 |
| module-level `import fsspec` | TB23 |
| replay of the own journal prefix only | TB15 `[memory]`, `[file]` |
| reverse replay order (first wins) | TB10 `[memory]`, `[file]` |
| blob presence not checked | TB11 `[memory]`, `[file]` |
| parse error raised | TB12 `[memory]`, `[file]` |
| `json.dumps` with default separators | TB13 `[memory]`, `[file]` |
| presence by `ls` | TB9 on s3 (`test_fsspec_store_on_s3.py::test_fresh_store_is_empty`) |

## Iteration 3 — unit C (C2): listings read fresh

C's frozen suite on B's tip (C0's install present, no D7): of the 18 s3 re-runs only TB20
(`test_fsspec_store_on_s3.py::test_other_process_resumes_from_the_url_alone`) fails — the parent
instance that listed before the child ran misses the child's records (s3fs listing cache). That
is the plan's listing-cache mutant. The `ls`-presence mutant fails TB9 on s3 (iteration 2 table).
C2 builds the filesystem with `use_listings_cache=False` over the caller's options.
Result: `tests/frozen/checkpoint tests/extra/checkpoint` 129 passed, 0 skipped, 0 failed.
