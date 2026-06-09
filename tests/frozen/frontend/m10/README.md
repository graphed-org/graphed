# Frozen acceptance suite — M10 (graphed): IR-driven execution + incremental sessions

Remediation milestone for the MVP-shortcoming findings (see `mvp-shortcomings.md` in the
superproject). Traceability:

| Test file | Verifies | Finding |
|---|---|---|
| `test_compiled_execution.py` | `compile_ir`/`evaluate_ir`: evaluation from bytes alone (no Session/user code), dispatch count == REDUCED op count, retarget without recompile, External resolution by content hash, determinism | A.2 (execution re-walked the un-reduced op log, re-recording per partition) |
| `test_incremental_session.py` | `Session(incremental=True)`: byte-identical compile vs one-shot, total reducer work == node count (incrementality witness), default session untouched | A.1 ("incremental reduction" was an alias, never wired into building) |

`m10_toy.py` is the suite's self-contained list backend (counts dispatches). The M2/M5/M8 frozen
suites remain authoritative for the recording/projection/serialization behavior they pin.
