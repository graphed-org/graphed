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

## Iteration 4 — gates at the lane tip

- Full suite, `COV=1 ./scripts/run-tests.sh`: rc 0, 2308 passed, 31 skipped (main's baseline
  2236 / 31; the +72 are m62). Per-file gate rows for `graphed/checkpoint/`: `fsspec_store.py`,
  `store.py`, `__init__.py`, `codec.py`, `errors.py` 100 %, `runner.py` 98.00 %, `retry.py`
  93.83 %. The gate's only failures are the four ML-plugin externals the lane venv cannot import
  (torch/tensorflow/jax/xgboost).
- Diff coverage from the frozen suite alone (`pytest tests/frozen/checkpoint --cov=graphed
  --cov-branch`, then `diff-cover --compare-branch=origin/main`): 122 changed lines, 0 missing,
  100 %. No `tests/extra/**/m62/` was needed.
- Open on the PR's CI: `Store.put`'s `os.replace` fallback (plan-A D8(c)) is unmeasured on Windows;
  read TA4 on windows-latest and windows-11-arm first. A `PermissionError` from opening `dest`
  would be repaired once in `put` (plan-A A.4).

## Iteration 5 — review r1 repair (M1 forked writers, N1)

- M1: `FsspecStore` minted its writer once per instance, so a forked copy reused `_writer`/`_seq`
  and replaced the parent's record objects on `file://`. `_append` now re-mints the writer and
  resets the count under the lock when `os.getpid()` differs from the minting pid. Class search
  (`grep -rn -e uuid4 -e 'time_ns()' python/graphed`): the only per-instance id; `Store`'s
  temp-name uuid is per call.
- Closing test `tests/extra/checkpoint/m62/test_fsspec_store_forked_writers.py` (fork guard via
  parametrize over the available `fork` start method): passes on the repair; with
  `fsspec_store.py` reverted to 6c9b0bd it fails on the dead-letter assertion.
- N1: design.rst names the six `CheckpointStore` calls; the writer-name sentence says the id is
  re-minted in a forked child.
- Checkpoint trees (frozen + extra, junit): 130 tests, 0 failures, 0 errors, 0 skipped.
- Frozen-only diff-cover vs origin/main: 128 lines, 1 missing (the re-mint call, which runs only
  in forked children that pytest-cov does not measure), 99 %.
- Full suite `COV=1 ./scripts/run-tests.sh`: rc 0, 2309 passed, 31 skipped (+1 = the fork test).
  `coverage_gate.py`: `fsspec_store.py` 97.44 % (missing line 113, branch 112->113); the gate
  fails only on the four ML-plugin externals the lane venv cannot import.

## Iteration 6 — review r2 residuals (L1, L2, N1)

- L1: the fork test's children are daemons joined against a 30 s deadline, then killed; the test
  asserts the exit codes. Mutant `_child` blocking on `threading.Event().wait()`: the test fails
  in 30.5 s and pytest exits.
- L2: design.rst states replay order per writer, as the `fsspec_store.py` docstring does.
- N1: iteration 5's `fsspec_store.py` figure replaced by the measured 97.44 %.
- r1 N1 ("six calls above") was already closed by 99e5173.
- Full suite `COV=1 ./scripts/run-tests.sh`: rc 0, no FAILED; `coverage_gate.py` fails only on
  the four ML-plugin externals; `fsspec_store.py` 97.44 % (line 113, branch 112->113).

## Iteration 7 — Windows CI legs after the PR opened (team-lead, 2026-09-23)

- mypy --strict on the win32 typeshed: `get_context("fork")` has no `Process` there → the
  tests/extra fork test sits under `if sys.platform != "win32":` (96840de).
- m62 url suites red on every Windows leg: `Store._append` wrote records in text mode, so the
  local journal carried `\r\n` while the frozen tests pin the local bytes to the remote's `\n`
  records → the writer opens with `newline="\n"` (da10f01); the reader keeps universal newlines.
- `test_concurrent_identical_puts_never_fail` red on the windows-11-arm legs only (run
  35844884311, py3.11/3.12/3.13): a put that loses `os.replace` verifies through `get`, and on
  Windows the blob a concurrent replace is landing is briefly unopenable (`PermissionError`
  errno 13, delete pending). `Store.get` now retries a `PermissionError` over `_READ_BACKOFF`
  (0.01, 0.05, 0.25, 1.0 s) and re-raises after the schedule; `FileNotFoundError` still returns
  `None` at once. Witnesses: the frozen concurrent-puts test on the Windows legs (the branch is
  reached only there), plus `tests/extra/checkpoint/m62/test_local_store_transient_reads.py`
  for the ubuntu coverage job (transient error recovers after 3 reads; persistent error
  re-raises after len(_READ_BACKOFF)+1 reads; a missing blob never sleeps).
- Gates: precommit ok; checkpoint frozen+extra with coverage → store.py 100 %, diff-cover vs
  main 132 lines, 1 missing (fsspec_store.py:113), 99 %; frozen-only diff-cover on this
  machine 126/132 (the five retry lines have frozen hits on Windows only).
