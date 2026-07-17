# M39 frozen suite — graphed-awkward (awkward exchange primitives)

Milestone **M39**. `AwkwardBackend` gains the `ShuffleBackend` exchange half:
`partition`/`concat`/`slice_rows`/`estimated_bytes`/`to_wire`/`from_wire` (+ `identity`) over awkward
arrays. **Frozen — read-only after `freeze-M39-0`.**

## Files → plan clause / theme

| File | Covers | Plan clause / theme |
|---|---|---|
| `golden_route.py` | the FROZEN `(key, P) → dest` table, precomputed from `int.from_bytes(sha256(key.to_bytes(8,"big"))[:8],"big") % P` (real hashlib; 96 vectors) | §4, §3.0 |
| `test_exchange_primitives.py` | (a-golden) `partition` reproduces the golden dests; (a) deterministic; row-conserving; `concat` order; `slice_rows` keeps jagged structure; `estimated_bytes` tracks bytes not entry count; wire round-trip preserves values+jaggedness; `identity` present | (a), (a-golden); §3.0, §3.3, §5.1 |

## The golden-vector discrimination

The table is committed as **literals** (not derived in-test), so the routing rule cannot be changed
and the test updated in tandem. Measured at authoring time: of the 96 vectors, sha256 and
blake2b-of-the-same-key disagree on **81** — so a backend substituting any other process-independent
hash (blake2b/xxhash) passes the B2 process-invariance witness yet FAILS these vectors. That is the
executable conformance witness `@runtime_checkable` cannot provide (ADV-r5.3).

## Non-vacuity

Pre-implementation: `AwkwardBackend` has no `partition`/`identity`/… → right-reason AttributeErrors.
