Improvements
============

Tracked design improvements and limitations for ``graphed-numpy`` (plan M0 requires this file).

Current limitations
-------------------

- **Trivial seam-prover.** Operates on 1-D numpy bags only; it exists to prove the backend
  boundary, not to be a production backend (that is ``graphed-awkward``, M3).
- **Opaque ``map`` payloads are not content-addressed.** ``external_payload`` flags wrapped
  callables as a preservation risk; real content hashing is M9.

Planned
-------

- Column projection via ``Backend.project`` (M5) and participation in the execution contract (M7).
