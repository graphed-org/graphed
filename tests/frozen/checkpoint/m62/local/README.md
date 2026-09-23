# m62 unit A frozen suite — the local `Store` meets the store contract

Frozen at `freeze-m62` (alias `freeze-m62a`); read-only afterwards. Spec: lane plan `plan-A.md` §A.3.

New API: `graphed.checkpoint.CheckpointStore`, a `runtime_checkable` Protocol with `completed`,
`get`, `put`, `record_done`, `record_dead`, `dead_letters`. `Store`'s API and layout are unchanged.

| Test (`test_local_store_contract.py`) | Pins | Witness | Wrong implementation it rejects |
|---|---|---|---|
| TA1 `test_store_satisfies_the_protocol[<omitted>]` | A.3.1 | `isinstance` true for `Store` and a six-method class, false for each five-method class | Protocol missing a method; Protocol not runtime-checkable (`TypeError`) |
| TA2 `test_get_refuses_bytes_that_do_not_hash_to_their_name` | A.3.2.1 | tampered object → `get` is `None`; re-`put` rewrites the object bytes | unverified `get`; verified `get` with an exists-guarded `put` (no heal) |
| TA3 `test_resume_recomputes_a_corrupted_partial_and_heals_it` | A.3.2.3 (m8 `analyses` plan) | `executed == 1`, `skipped == 5`, reference value, healed blob hashes to its name, second resume `executed == 0` | no verify (`UnpicklingError`); verify without heal |
| TA4 `test_concurrent_identical_puts_never_fail` | A.3.2.2 | 20 trials × 8 barrier-started threads × 1 MiB: all digests equal, one entry under `objects/` | pid-only temp name (`FileNotFoundError`) |
| TA5 `test_put_survives_losing_the_rename_race_to_a_present_copy[present-copy\|no-copy]` | A.3.2.4 | patched `os.replace` called with `dest` named by the digest; present copy → digest; no copy → the same `PermissionError` object; `objects/` holds no temp | no fallback; swallow-everything; leaked temp |
| TA6 `test_put_keeps_the_default_file_mode` | A.3.2.5 | blob mode == sibling `open(..., "wb")` mode under `umask(0o022)` | `mkstemp` temp (0600) |

## TEST_SANITY (lane venv, CPython 3.13.3, macOS)

- Stub: `CheckpointStore` injected into today's `graphed.checkpoint` by a pytest plugin; `Store` as on main.
  TA1 ×6 and TA6 pass (they pin the Protocol and today's mode). TA2 fails `assert b'evil bytes' is None`;
  TA3 `UnpicklingError: pickle data was truncated`; TA4 `FileNotFoundError` on `objects/.<h>.<pid>.tmp`;
  TA5 present-copy `PermissionError: [Errno 13] sharing violation`; TA5 no-copy leaks `objects/.<h>.<pid>.tmp`.
- Mutants, each run once: Protocol without `dead_letters` fails TA1[dead_letters]; non-runtime-checkable
  Protocol fails TA1 ×6 (`TypeError`); verified `get` over today's `put` fails TA2 (`None == b'true bytes'`)
  and TA3 (healed blob hash differs); `mkstemp` put fails TA6 (`384 == 420`, i.e. 0o600 vs 0o644).
- Positive control: a verified `get` plus a unique-temp put with a verified `PermissionError` fallback
  passes this suite and m8/m39/m49.
- Two runs of the whole checkpoint process give identical per-test outcomes. ruff and mypy clean.
