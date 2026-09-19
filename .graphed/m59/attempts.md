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
