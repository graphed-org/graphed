# M39 frozen suite — graphed-numpy (rectilinear exchange primitives)

Milestone **M39**. A SECOND exchange backend ships in M39 (not M40, B-r5.3) so the generic engine's
backend-agnosticism is witnessed by execution. `NumpyBackend` gains the `ShuffleBackend` exchange half
over structured (record) arrays. **Frozen — read-only after `freeze-M39-0`.**

## Files → plan clause / theme

| File | Covers | Plan clause / theme |
|---|---|---|
| `golden_route.py` | the SAME frozen `(key, P) → dest` table as graphed-awkward (identical routing rule) | §4, §3.0 |
| `test_exchange_primitives.py` | (a-golden) numpy `partition` reproduces the golden dests; deterministic + row-conserving; `concat` order; `slice_rows` contiguous record slice; `estimated_bytes` scales with itemsize; wire round-trip preserves columns; `identity` present | (a), (a-golden); §3.0, §3.3 |

## Why numpy too

The numpy primitives are trivial (rectilinear), so shipping them in M39 is cheap AND de-risks the M40
numpy-join promotion. The same golden table green on numpy proves the routing rule is not
awkward-specific — a numpy impl using any other hash fails these vectors exactly as the awkward one
would.

## Non-vacuity

Pre-implementation: `NumpyBackend` has no `partition`/`identity`/… → right-reason AttributeErrors.
