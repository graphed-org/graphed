# Frozen acceptance suite — M10 (graphed-awkward): buffer-level projection

Remediation milestone for MVP-shortcoming finding A.3 (see `mvp-shortcomings.md` and
`buffer-level-projection-plan.md` in the superproject): column-level projection discarded the
`shape_touched` half of awkward's reporting typetracer, so a count-only analysis projected to the
**empty column set** — efficient-looking but under-specified (feeding it to a reader reads zero
branches and produces a wrong result).

| Test file | Verifies |
|---|---|
| `test_buffer_projection.py` | `project_buffers`: count-only → `{collection: OFFSETS}` (non-empty); data reads → `DATA`; mixed analyses; `to_projection()` collapses EXACTLY to the frozen M5 column view (consistency); buffer-level no-overtouch; opaque-op conservative policy |

The M5 frozen suite (`tests/frozen/m5/`) remains authoritative for the column-level `project`;
nothing here modifies or weakens it — `project_buffers` is additive.
