# M11 attempts — graphed (dask.array parity P0: ufunc tier + backend-idiom factorization)

## Iteration 0 — TEST_AUTHORING/TEST_SANITY — 2026-06-10 (freeze-M11-0)

- frozen suite authored (ufunc surface + numpy-style array metadata/`__array_function__` ON THE
  BASE ARRAY); verified NON-VACUOUS against pre-M11 code (66/68 fail for the right reasons);
  implemented and locally green.

## DESIGN REVIEW — human-directed re-factorization — 2026-06-10 (freeze-M11-1 supersedes)

- User decision: `graphed.Array` must stay **backend-idiom-neutral** — numpy's method/property
  idiom (`.shape`, `.sum()`, `__array_function__`) must NOT live on the shared proxy, because
  awkward's design applies operations as functions over arrays, never as member functions.
- The original M11/M12 commits were backed out to branch `backup/m11-m12-numpy-flavored`
  (nothing had been pushed) and the milestone re-frozen as freeze-M11-1:
  - kept on the base Array: operators/dunders + full `__array_ufunc__` table (ufunc application
    is common to both idioms) + protected `_form_meta` infrastructure;
  - new: the `array_type` factory — a backend supplies its proxy subclass; every Session builder
    returns it (graphed_numpy.NumpyArray completes the numpy surface; graphed-awkward keeps the
    base Array — its surface is the `gak` function namespace);
  - frozen suite re-authored: `test_array_meta`/`test_array_function` replaced by
    `test_array_factory` which PINS the factorization (no numpy-idiomatic member may appear in
    `vars(Array)`); non-vacuity of the factory pin verified by stashing the session factory.
- One pre-freeze sanity fix (recorded): the "no dispatch on base Array" pin originally used
  `np.cross` (fails with ValueError via coercion, not TypeError); re-pinned on `np.sum`.

## Iteration 0 — IMPLEMENTING/REVIEW — 2026-06-10

- gates: frozen_tests 123/123 PASS · coverage 93% (>=90, branch) · ruff+format clean ·
  mypy --strict clean · determinism green · sphinx -W clean.
