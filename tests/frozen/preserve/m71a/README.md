# preserve/m71a — 0.0.6 byte invariance for undeclared plans (traceability)

Milestone m71 unit A. Authority: the lane plan's `plan-A.md` "Frozen-test contract" A-C6. The
invariant: a plan that does not declare an output type produces graphed 0.0.6's bytes. The pins
guard m71b and m71c too (the plugin defaults must add no node param).

Run: `python -m pytest tests/frozen/preserve/m71a -q` (or the whole preserve subtree).

| test | contract | pins |
|---|---|---|
| `test_m71a_bytes_pins.py::test_an_undeclared_plan_keeps_its_0_0_6_bytes` | A-C6 | SHA-256 of `serialized_ir(optimize=False)`, `serialized_ir()`, `compile_ir(...).ir`, `DurablePlan(...).to_bytes()`, and the bundle manifest's `externals` / `opaque_nodes` / `analysis` subtrees (sorted compact JSON) for the correctionlib plugin + gak correction template + ONNX template + awkward `map` recording; SHA-256 of the numpy gufunc + `map` IR. Payloads are byte literals; `platform.python_version` is fixed to `"3.13.3"` |

Measured on 3c46e01 (graphed 0.0.6) under CPython 3.13.3, and passing unchanged under CPython 3.11
with graphed 0.0.6 from PyPI (numpy 2.4). This test passes on the base by design.
