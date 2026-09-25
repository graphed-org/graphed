# m71 — implementer iterations

ext-form lane (plans in `graphed-workdir/lanes/ext-form/`), frozen `freeze-m71` = `e79c186`.
Run: `python -m pytest tests/frozen/{frontend,awkward,preserve}/m71a tests/frozen/preserve/m71b tests/frozen/preserve/m71c -q`.

## Iteration 1 — m71a mechanism (plan-A A2)

`Session.record_external(output_type=)`: canonicalized by the backend's optional
`canonical_output_type` inside a located wrapper, stored as the `output_type` param; exclusive with
`form=`. `PYTHON_TYPES` in `graphed/backend.py`. awkward: `canonical_output_type` (Python type → Form
→ ArrayType/ScalarType content → Type → datashape → numpy dtype via `from_numpy`, rebuilt through
`_type`, whose `LarkError` fallback builds a grammar-less primitive such as `float16`) and
`declared_form`; numpy: `canonical_output_type`/`_dtype`/`declared_form` in `numpy/forms.py`.
`Array.map`/`graphed.apply(output_type=)`, the Varied lambda forwards it. Decision on plan r11 L1: numpy
`op_form("gufunc")` refuses a declared key (`tests/extra/numpy/m71`). frontend/awkward/preserve m71a
all green.

## Iteration 2 — m71b the ask (plan-B B2)

`graphed.preserve.externals.record_external(..., params=None, output_type=None)` forwards
`output_type=` to `Session.record_external` (keyword-only, after `params`, for the m68 `service=`
rebase). preserve m71b 12/12 and m71a pins green. Per r11 X16 nothing relies on the in-process
`_PluginEvaluator` params carrying the key; only the bundle's `evaluate_external` sees it.
