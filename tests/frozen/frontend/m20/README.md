# Frozen acceptance suite — M20 (graphed): the partitioned-write base

User-directed refactor (2026-06-10) of the parity plan's P3.6: deferred partitioned WRITING gets
a format-agnostic base (`graphed.write`) that client integrations specialize — parquet in the
backends, ROOT in the reader fork — instead of borrowing parquet tools. Traceability:

| Test file | Verifies | Item |
|---|---|---|
| `test_write_base.py` | format-agnostic `write_plan` with worker-reported paths (run via `graphed_core.execution.SequentialRunner`); `file_bases` over generic keys (uri strings and (uri, tree) pairs); `blind_part_index` from the partition alone (non-blind refused); exact `step_of` reconstruction (n=5/steps=3); suffix-explicit `part_path`; and the pin that `graphed.parquet`'s M15 surface IS the base via aliases (M32: `SequentialRunner` is no longer among them — it is general execution in `graphed_core.execution`) | P3.6 |
