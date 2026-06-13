# m15 + m20 sanctioned re-freeze (M32 — SequentialRunner relocation) — 2026-06-13

USER-sanctioned freeze amendment (the SequentialRunner move to graphed_core.execution, no
re-export from graphed.write/parquet):
- m15/test_parquet_base.py: `gpq.SequentialRunner()` -> `from graphed_core import SequentialRunner`
  (2 call sites). No behavioral assertion changed.
- m20/test_write_base.py: `gw.SequentialRunner()` -> `SequentialRunner()`; the obsolete alias-
  identity pin `assert gpq.SequentialRunner is gw.SequentialRunner` REWRITTEN to pin the new
  invariant — SequentialRunner is NOT exposed by graphed.write or graphed.parquet and IS the one
  reference runner in graphed_core.execution.
- m20/README.md updated to match.
Re-frozen at freeze-m15-M32 / freeze-m20-M32. Equivalent coverage; the only semantic change is
the alias pin, which encoded the architecture being deliberately changed.
