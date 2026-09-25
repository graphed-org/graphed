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

## Iteration 3 — m71c the class cut (plan-C C2)

`ExternalPlugin.output_dtype: object = None` (trailing field after `synthesize`; m68 appends
`check_params` in the same slot, keep both), `"float64"` on the seven float64 built-ins.
`Session.record_external(form_params=)` reaches `op_form` only, under the stored params; preserve
passes `{"output_dtype": plugin.output_dtype}` when set. awkward `astype_form` (total over awkward
forms, primitive dtypes only, scalar stays scalar) and the `output_dtype` branch after `output_type`.
gak templates call `session._mine(inputs, (op, capture()))` before `session.form`; the correction
template's form is `astype_form(..., "float64")` (literal; `import graphed.awkward` still loads no
`graphed.preserve`/`graphed.checkpoint` module). All 99 frozen m71 tests green.

## Iteration 4 — docs (plan-C C3) and gates

awkward improvements ("approximate" → undeclared records the first input's type; float64 built-ins;
string first slot stays `string`), frontend design `form_params=`, preserve design
`ExternalPlugin.output_dtype`, changelog. `COV=1 ./scripts/run-tests.sh` rc=0; diff-cover vs
origin/main 100% (111 lines); per-file gate: 4 ML plugin modules below 90% in the lane venv (frameworks
not installed), pre-existing, each touched by one covered module-level line.

## Iteration 5 — impl review r1 repair

M1: `canonical_output_type` refuses a canonical string that `_type` does not rebuild as itself
(awkward 2.14 parses `float16[parameters=...]` as an empty named tuple without a LarkError); the
refusal names the rebuilt string. Every rebuild routes through that function, so this is the one
cut. `tests/extra/awkward/m71/test_m71_float16_parameters.py`: bare and `var *` parameterised
float16 are refused (or, on a parser that reads them, record the eager type); both legs fail at
5860757 (DID NOT RAISE) and pass after. L2: changelog and awkward design name the nested /
parameterised float16 exception. L1 (stored `output_type`/`output_dtype` user plugin params) not
changed: the param namespace cannot tell a user key from a declaration, a representation decision
left to the orchestrator.
