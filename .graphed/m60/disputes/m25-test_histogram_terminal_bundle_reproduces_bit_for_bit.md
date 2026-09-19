# Test Dispute — tests/frozen/preserve/m25/test_histogram_preservation.py::test_histogram_terminal_bundle_reproduces_bit_for_bit

## The test

Its last leg builds a second bundle to assert content-addressed determinism:

    again = build_bundle(
        tmp_path / "bundle2",
        session=_record()[0],
        value=_record()[1],
        ...
    )
    assert again.fingerprint() == bundle.fingerprint()

`_record()` mints a FRESH `Session` on every call, so `session=` and `value=` come from two
different sessions. `build_bundle` then calls `session.serialized_ir(value, optimize=False)` and
`session.sourcemap()` — both against an `Array` that session never recorded. It passes today only
because the two recordings are structurally identical, so the foreign `node_id` happens to address
an equivalent node here.

## The contract line it contradicts

m60 integ-m60-X1 (`graphed-workdir/m60-decomposition.md`; root-prompt R25.3): *"Every `Session`
method that takes an `Array` refuses one recorded in another Session, recording nothing."* The
frozen m60 suite pins `serialized_ir` explicitly — `tests/frozen/frontend/m60/test_session_guard.py`
lists it in `READING`, and `tests/frozen/frontend/m60/README.md` states it under *Where a contract
line is pinned by its nearest observable*. There is no implementation of X1 that refuses
`own.serialized_ir(foreign)` in the m60 suite and accepts it here: both calls are the same shape,
and the m60 suite's own premise leg asserts the ids collide. A content-equality guard would admit
both, and is unsound anyway — structural equality of an IR node does not make the two sessions'
side tables (`_externals`, `_provenance`) the same, which is what `serialized_ir` and `sourcemap`
read.

## Probe

`<scratchpad>/m60/probe_m25.py` records the same shape twice with a different multiplier and
neutralizes the guard, i.e. reproduces the pre-m60 route this leg travels:

    ids collide: True
    A.serialized_ir(A) == A.serialized_ir(B): True
    A answers B's form: ## * float64
    A's own value at that id: 352.5 | B's own: 587.5 | A asked for B: 352.5

Session A serializes B's program to A's bytes and materializes B's array to A's data. The leg is
green only because its two recordings agree; the route it uses returns the wrong analysis the
moment they do not.

## Proposed correction

Keep the leg's intent — two independent builds of the same analysis fingerprint the same — and
take both halves from ONE recording:

    s2, fill2, _ = _record()
    again = build_bundle(
        tmp_path / "bundle2", session=s2, value=fill2,
        datasets={"events": EVENTS}, payloads={},
    )

Nothing else in the file changes; `_record()` is still called a second time, so the build is still
independent of the first. Re-freeze requires the owner's affirmation.

Measured on a scratch copy of `tests/frozen/preserve/m25/` (the frozen tree untouched): unchanged,
the leg is red with `TypeError: graphed: an input was recorded in a different Session` (the control);
with the correction the file is `4 passed`.

## Status

RESOLVED — TEST WRONG, re-freeze affirmed by the owner (2026-09-19). The proposed correction is
applied verbatim under `--allow-refreeze tests/frozen/preserve/m25`; tag `freeze-m25-fixup`.
