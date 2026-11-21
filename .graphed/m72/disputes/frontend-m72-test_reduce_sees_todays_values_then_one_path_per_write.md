# Test Dispute — `test_m72_part_writes.py::test_reduce_sees_todays_values_then_one_path_per_write`

File: `tests/frozen/frontend/m72/test_m72_part_writes.py`, `tests/frozen/frontend/m72/README.md` (graphed#228)

Status: CLOSED — owner rulings via team-lead: "#228 option A with refreeze authorized" and, for the seven issues, "refreezes are authorized"; refrozen as corrected below in the #228 PR, tag freeze-m72-fixup4.

## The pins
Outputs `(a, b, a)` (the same Array twice) with three writes: `len(seen) == 5` and `seen[2:]` are the
three part paths; with `writes=()`, `[len(seen) for seen in SEEN] == [2, 2]`. The README row says
`reduce` gets `[va, vb, p1, p2, p3]` and `writes=()` gives `[va, vb]`.

## The contradiction
#228 option A (owner-ruled): `reduce` receives one value per output passed, in order, at every opt
level, repeats included. On the #228 branch the run hands `[va, vb, va, p1, p2, p3]`, so the test
fails `assert 6 == 5`.

## Correction
`len(seen) == 5` → `6`; add `np.testing.assert_array_equal(seen[2], 2.0 * chunk)`; `seen[2:]` →
`seen[3:]` and the three `read_part(seen[2|3|4])` → `seen[3|4|5]`; `[2, 2]` → `[3, 3]`. README row:
`[va, vb, va, p1, p2, p3]` and `[va, vb, va]`. The write-value assertions are unchanged.
