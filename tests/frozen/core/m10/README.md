# Frozen acceptance suite — M10 (graphed-core): genuine incrementality, executable stages, maximal fusion

Remediation milestone for the MVP-shortcoming findings (see `mvp-shortcomings.md` in the
superproject). Traceability:

| Test file | Verifies | Finding |
|---|---|---|
| `test_incremental_reducer.py` | `IncrementalReducer` per-step work == delta (never history); finalize identical to one-shot reduce; deterministic | A.1 ("incremental reduction" was an alias) |
| `test_stage_members.py` | `GraphStore.nodes()` exposes fused stage members as executable `(name, params, inputs)`; a Python interpreter over the REDUCED IR reproduces unreduced semantics; `outputs()` order | A.2 (reduced IR could not drive execution) |
| `test_maximal_fusion.py` | `reduce(maximal_fusion=True)` fuses a diamond into ONE stage; never crosses boundaries; default mode unchanged (M4 pin intact) | C.6 (stages weren't maximal) |
| `test_sound_rules.py` | commuted twins of every symmetric op in the vocabulary merge; param-token injectivity (`;`/`=`/`%` in string params) | C.7 (rule set too thin to fire) + latent token-collision bug |

The M4 frozen suite remains authoritative for default-mode reduction behavior; nothing here
modifies or weakens it.
