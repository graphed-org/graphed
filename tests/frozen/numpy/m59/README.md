# numpy/m59 — the numpy backend evaluates the same key kinds (traceability)

Authority: `graphed-workdir/m59-decomposition.md` contract line **I6** under root-prompt rule
**R25.2**. The tuple-key kinds m59 adds to the shared `Array.__getitem__` surface —`None`
(newaxis) and `Ellipsis` — are common to both array idioms, so the numpy backend must evaluate
them with numpy's own semantics. The rest of m59 (I1–I5, S, M) is `tests/frozen/awkward/m59`.

Separate directory because the numpy package's frozen suite runs in its own process (per-milestone
split, `scripts/run-tests.sh`). Run: `python -m pytest tests/frozen/numpy/m59 -q`.

## Fixture

`D2` (4×3, dyadic) and `D3` (4×3×2, whole numbers) — both partitioned on axis 0, so every key here
leaves axis 0 whole and numpy itself is the oracle for values and shapes. `ADDED` is the new key
kinds, `TODAY` the M13 tuple subscripts with the node each records, `REFUSED` the keys that touch
the partitioned axis or are malformed.

## Traceability

| Line | Test | Mechanism witness |
|---|---|---|
| I6 | `test_newaxis_and_ellipsis_evaluate_as_numpy_does_and_stay_fusible` | values against `numpy`'s own answer on `D2` and `D3`, the node's `kind` (`op`, so fusible), and the deferred `.shape` against numpy's shape with the partitioned axis as `None` |
| I6 (control) | `test_the_tuple_keys_accepted_today_record_the_same_op_params_and_boundary_flag` | (kind, name, params) of `subscript` for `[:, 0]`, `[:, :2]`, `[:, ::2]`, plus their values |
| I6 (control) | `test_a_tuple_key_that_touches_the_partitioned_axis_or_is_malformed_is_refused` | `TypeError` and `node_count()` unmoved, for the axis-0 group and the malformed group |

## Non-vacuity — what happens on a pre-m59 tree

11 tests collect; 4 FAIL (`TypeError: unsupported tuple-subscript element None`, and for the
`Ellipsis` keys `tuple subscripts must keep the partitioned axis 0 whole: a[:, inner...]` — today's
encoder reads a leading `...` as not being the full slice). 7 PASS: the two controls, which are the
live instruments showing the harness reaches the recorder and the numpy evaluator at all.

## What the frozen expectations were measured against

The shape leg of each added key was run on a pre-m59 tree through the backend's own inference path
(`numpy.forms.meta` + `form_from_meta`, which hand numpy a zero-length stand-in and read the result
back) for both `D2` and `D3`: every answer equals `(None, *data[key].shape[1:])`. The value legs
are numpy's own. `recorded(...)["kind"]` and `.shape` were exercised on a supported key.
