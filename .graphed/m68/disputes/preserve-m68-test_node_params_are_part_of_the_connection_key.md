# Test dispute — `tests/frozen/preserve/m68/test_triton_transport_by_scheme.py::test_node_params_are_part_of_the_connection_key`

The test requires two Triton nodes differing only in `output_name` to open two connections, which
pins a resource-cache key over every node param. That key reloads a correctionlib `CorrectionSet`
per systematic universe: two nodes on one payload differing only in `params["systematic"]` load
once on graphed 0.0.6 (key `(kind, content_hash)`) and twice under the all-params key, contradicting
0.0.6's documented sharing "across every call and every systematic universe off the same payload".

## Ruling (owner, 2026-09-25)

Key on what `load` reads: `(kind, content_hash, endpoint, {p: params[p] for p in plugin.load_params})`,
with `ExternalPlugin.load_params: tuple[str, ...] = ()` and Triton's plugin declaring
`("url", "transport")`. Re-specified test: nodes on one endpoint differing only in `output_name`
share one connection; differing in `transport` get two; two correctionlib nodes on one payload
differing only in `systematic` load once. Plan D7 and §3.2 updated to match. Amendment under
`--allow-refreeze tests/frozen/preserve/m68`, re-tagged `freeze-preserve-m68-2`.
