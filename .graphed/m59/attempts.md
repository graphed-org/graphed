# m59 — implementer iterations

Branch `m59-awkward-idiom-parity` on top of `db2094c` (= the frozen suite, tag `freeze-m59`).
Run: `python -m pytest tests/frozen/awkward/m59 tests/frozen/numpy/m59 -q -p no:cacheprovider`.

## Iteration 0 — baseline (31 + 4 failing)

Matches both suite READMEs' non-vacuity tables: awkward/m59 31 failing / 17 passing, numpy/m59
4 failing / 7 passing.

## Iteration 1 — one shared key encoding (I1–I6)

The numpy idiom already had the tuple key (`graphed.numpy.array._encode_subscript`, the `subscript`
op with a `spec` param), so m59 MOVED that spec to `graphed/array.py` and widened it rather than
adding a second one: `encode_subscript`/`decode_subscript` there are now the single source of truth,
`Array.__getitem__` records the tuple key for both idioms, `NumpyArray.__getitem__` is deleted
(the base class answers), and each backend's evaluator decodes the same spec — one new `apply`
branch in `graphed/awkward/_ops.py`, one changed line in `graphed/numpy`. New member spellings:
`N` for `None`, `...` for `Ellipsis`.

`tests/frozen/frontend/m13::test_idiom_specific_keys_are_refused_on_the_base_proxy` pins that a
tuple key is refused on the base proxy over a backend that models no such op, and its `ToyBackend`
answers every `op_form`, so a purely frontend surface cannot satisfy both suites. A backend
therefore DECLARES the capability (`subscript_keys = True` on `AwkwardBackend`/`NumpyBackend`,
read with `getattr`, as `array_type`/`method_outputs` are read); the frontend still owns key
validation and still refuses before any backend call.

Gates: awkward/m59 keys + refusals 24/24, numpy/m59 11/11, whole tree green apart from the S and M
legs; diff line+branch coverage of the changed source 51/51 from the FROZEN suites.
`tests/extra/numpy/m59/test_m59_empty_tuple_key.py` covers the one branch they do not reach
(`a[()]`); dropping `not key or` turns the refusal into `IndexError` and the leg fails.

## Iteration 2 — the scalar's dtype and the method's nesting (S1–S5, M1–M4)

`array.py::_scalar_params` replaces the bare `_as_param` call in `_binary`: a value with a `dtype`
and a `()` shape records `{"scalar": value.item(), "dtype": str(dtype)}`, and an integer outside
i64 records its value as text. Measured, against the dispatch note that said the store REFUSES
such an int: `add_op('mul', [n], {'scalar': 2**63})` is accepted and comes back from
`serialize`/`deserialize` as `9.223372036854776e+18` — the store widens it to f64 silently, so the
loss the text encoding avoids is exactness, not an exception. The
dtype is read off the object, so the frontend still imports no backend. Each backend that reads
`params["scalar"]` rebuilds the operand — the population is exactly two sites
(`grep -rn '"scalar"' python` → `awkward/_ops.py`, `numpy/__init__.py`), both repaired, and the
numpy one is why the whole numpy tier did not regress.

M: `AwkwardBackend.method_outputs` answers the result's NESTING (a tuple with `None` at every
array leaf) instead of a width; `_record_method` rebuilds that nesting, one node per leaf numbered
depth-first, and `_ops.apply` picks leaf `index` out of the flattened result. A flat tuple is the
one-level case of the same shape, so M2's byte-identical pin holds by construction.

Extra legs, each killed by the mutation that removes its branch (`PYTHONDONTWRITEBYTECODE=1`):
the ndarray-operand guard in `_scalar_params` (`ValueError` from `.item()` instead of the
`TypeError`), and the numpy backend's own rebuild (`int64` instead of `uint64`).

## Iteration 3 — the reviewer's REJECT (four findings)

**1. A leading `...` that absorbs nothing.** `encode_subscript` cannot tell: how deep the receiver
is lives in the Form, which is opaque to the frontend. So the rule is a TYPING rule, checked by
each backend's `subscript` rule through one shared predicate,
`graphed.array.check_leading_ellipsis(key, depth)` — the members after the Ellipsis (ints and
slices; `None` adds an axis and addresses none) must address FEWER axes than the receiver has.
Depth comes from the backend: awkward passes `layout.minmax_depth[0]`, NOT `ndim` — measured,
`ak.Array.ndim` reads the m59 fixture's record of two list fields as 1, while `[..., 0]` does
descend into those fields, so `ndim` would refuse a key eager awkward accepts (and one the frozen
`ACCEPTED` table pins). numpy passes `ndim`, which makes explicit what was an `IndexError` off its
zero-length meta. The refusal travels the existing ill-typed channel (`op_form` runs before
`add_op`), so it is a `GraphedTypeError` at the user's line with nothing recorded.

**2. The chained spelling.** `rstrip(':')` turned `a[1:, 0]` into the advice `a[1][:, ...]` — an
integer index, a different op. The spelling is now built from the slice's fields: `start:stop`
always, `:step` only when there is one.

**3. S2's exactness had no killing test.** `form`/`materialize` read the params still in this
process; only compile + pickle + `evaluate_ir` reads the value back out of the serialized IR. The
new leg does that. Under the `if False:` mutant the whole m59 frozen suite still passes (59) while
the new leg fails both parametrizations — the gap the reviewer measured, now closed.

**4. The scalar decoder's str arm.** Probed: `np.dtype(d).type(...)` parses the decimal text
itself, and every dtype the m59 suites feed a binary op (uint64 incl. 2**63+1 / 2**64-1, int32
incl. negative, float32, bool) decodes identically without the arm. Deleted in both readers; the
two backends stay independent, nothing hoisted.

Closing tests (`tests/extra/{awkward,numpy}/m59`), each run against its mutant with
`PYTHONDONTWRITEBYTECODE=1`: guard removed → 7 failed / 6 passed; `rstrip` restored → 2 failed
(exactly the start-only slices) / 4 passed; wide-int branch `if False:` → 2 failed. Whole
`./scripts/run-tests.sh` green; `precommit --fast` ok. `tests/frozen/awkward/m59` is now on the
pytest `pythonpath` so the extra legs reuse `m59_idiom_fixtures`, as m57/m58 already do.
