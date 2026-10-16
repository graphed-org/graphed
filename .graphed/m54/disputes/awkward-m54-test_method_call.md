# Test dispute — numpy scalar method arguments (`tests/frozen/awkward/m54`)

**Ruling:** the owner ruled on 2026-10-15 to refreeze both tests below and fix them in full (B4 review r1, finding F1).

## The tests

1. `test_method_call.py::test_a_numpy_scalar_constant_is_coerced_and_matches_the_python_scalar`

```python
coerced = jets.scaled(np.float64(2.0))
plain = jets.scaled(2.0)
assert coerced.node_id == plain.node_id  # `.item()` at the encoder, not a second node
```

2. `test_method_refusals.py::test_a_zero_dimensional_numpy_array_is_coerced_not_refused`

```python
assert jets.scaled(np.array(2.0)).node_id == jets.scaled(2.0).node_id
```

## The clause they contradict

A recorded op's value is the eager call's (m54 §2.2, `test_method_call.py`'s own docstring). Under
NumPy's NEP 50 promotion rules a numpy scalar and a 0-d array are *strong* (their dtype takes part
in promotion), and a python float is *weak*. Measured on numpy 2.5.4: `float32_array *
np.float64(2)` and `float32_array * np.array(2.0)` are float64, and `float32_array * 2.0` is
float32. Awkward 2.14.0 promotes the same way. So these calls differ from `scaled(2.0)` on a
float32 field. One node for both gives the later call the earlier one's dtype. This is the same
defect, GW4, that B4 fixes for `gak.where`/`fill_none`/`initial`/`full_like`.

## Correction

A numpy-scalar or 0-d-array argument is a distinct node. Its materialized value equals the eager
call with the same argument. The tests are renamed
`test_a_numpy_scalar_constant_keeps_its_dtype_as_a_distinct_node` and
`test_a_zero_dimensional_numpy_array_is_a_typed_scalar_not_refused`.
