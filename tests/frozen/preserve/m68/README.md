# m68 frozen suite, graphed slice: the service surface (`freeze-m68-7`, `lanes/htcondor/plan-services.md` §3.2)

Frozen and read-only after `freeze-m68-7`. Analyses come from `m68_services_fixtures.py`: an in-memory
partitioned awkward source (no pyarrow) and the shipped `TRITON_PLUGIN` recorded by `url=` or `service=`
over `fake_triton_services.py`, and a non-Triton `GENERIC` External (`mark`); the transport is a verbatim copy of `preserve/m26/fake_triton.py` under a basename unique
in the one-process preserve run. Plans run on `SequentialRunner`. Byte pins were measured on 0.0.6
(3c46e01) and pass there.

| File · test | Plan item | Shows | Fails |
|---|---|---|---|
| `test_service_spec_roundtrip` · `defaults_are_the_planned_ones…` | D1, §3 defaults rule | `ServiceSpec`/`Launch` defaults; both frozen; `Launch.env`/`resources` refuse item assignment | other defaults; a mutable `ServiceSpec` or `Launch`; a `dict` default |
| · `spec_round_trips_through_json_text` | D1 | `from_json(json(to_json()))` equal, `ports` a tuple | lists left in place of tuples; a dropped `launch` field |
| · `session_registry_refuses_conflicts…` | D1, §3.2 `session.py`, `_base.py` | equal re-declaration ok, one differing in `kind` or `launch` `ValueError`; `service_for` and `record_external` (Triton and `GENERIC`) refuse an undeclared name naming the declared ones, adding no node | a Triton-only refusal; a silent overwrite; an equality check that skips `launch`; a refusal that omits the declared names; a node recorded before refusing |
| · `an_empty_registry_refuses_a_named_service` | §3.2 `session.py`, `_base.py` | with nothing declared, a `service=` node is refused naming it, adding no node | a lookup that refuses only once something is declared |
| · `services_is_a_copy_like_sources` | §3.2 `services()` view like `sources()` | assigning into `services()` leaves the declared spec | the live registry returned |
| · `plan_and_durable_plan_carry…` | §3.2 `aggregate.py`, `core/plan.py` | `Plan.services` = specs referenced by the compiled IR ∪ `services=`, sorted by name (a recorded External outside the outputs is not referenced; a named, referenced spec appears once; a named undeclared one is refused naming the declared ones); `DurablePlan` round-trips it, and built without it holds `()`; `task_id` ignores it | a concatenation; the named ones ignored; an undeclared named one dropped; declared-but-unreferenced specs in the plan; names collected from every recorded node instead of the compiled IR; the spec folded into `task_id`; a `None` default |
| · `bundle_manifest_and_inspect_list…` | D1, §3.2 `bundle.py` | manifest `services` = the referenced specs in name order; `inspect()` prints them after the payloads, with "external only" (a whole phrase) for the recipe-less one | an unreferenced spec written; specs in set order; recipe text missing |
| · `specs_are_listed_in_name_order_whatever_the_kind` | D1, §3.2 `aggregate.py`, `bundle.py` | eight specs declared and recorded against name order, one on a `GENERIC` node: `Plan.services` (referenced, and referenced ∪ named) and manifest `services` list all eight by name; an image-less recipe prints its argv, a whole token | set order (eight names: no hash seed of 0–4999 iterates them sorted, `probe_set_order_seeds.py`); declaration order; Triton-only collection; the recipe's `None` image printed; argv fused with other text |
| · `run_report_endpoints_are_run_provenance` | D1, §3.2 `debug/report.py`, §3 defaults rule | `RunReport.endpoints` (`grpc://…` and `http://…`) round-trips, JSON `version` stays 1, the field default refuses item assignment, a 0.0.6 report reads as `{}`; an attached report leaves manifest bytes and fingerprint alone; `inspect()` shows each `name → endpoint`, a whole token, on one `endpoints:` line under that report only | endpoints written into the manifest; a bumped version; a `dict` default; the line under the wrong report or under every report; a pair fused with other text |
| `test_bundle_without_services_unchanged` · `…keeps_the_0_0_6_manifest_keys` | D1 | the manifest key set is 0.0.6's nine | `services` always written |
| · `…durable_plan…byte_identical_to_0_0_6` | D1 | `to_bytes` keys are 0.0.6's ten, bytes hash to the 0.0.6 pin | `"services": []` written for an empty tuple |
| · `one_referenced_spec_adds_exactly…` | D1 | the manifest's key set is the plain one plus `services`; the plan document differs from the plain one by value in `services` only | any other manifest key added or dropped; any other plan-document key or value moved |
| · `declared_but_unreferenced_spec_is_not_written` | D1 | a declared, unreferenced spec leaves `manifest.json` byte-identical and `Plan.services == ()` | declared specs written instead of referenced ones |
| · `0_0_6_pickles_load_without_services` | §3.2 `core/execution.py`, `_base.py` (old pickles valid) | a `Plan` and a `_PluginEvaluator` pickled by 0.0.6 load with `services == ()` and `endpoint is None` | a field with no class-level default (`default_factory`) |
| `test_bind_services` · `bound_plan_carries_the_endpoint…` | D2 hook, §3.2 `_base.py` | the bound copy's evaluator has `endpoint` and serves from it; the original stays unbound and raises `UnboundService` | an in-place bind; a bind that never reaches the evaluator; a bind that changes any `Plan` field but `process` |
| · `any_external_kind_is_bound` | D1, §3.2 `_base.py` | a `GENERIC` node's evaluator is bound too | a Triton-only bind |
| · `missing_endpoint_raises…` | §3.2 `_base.py` | binding without the node's service, with another name or an empty map, raises `UnboundService` naming it | a silent `None` endpoint; an empty map returned unbound |
| · `reduce_with_the_hook_is_bound_too` | §3.2 `aggregate.py` | `reduce` with `bind_services` is bound | externals-only binding |
| · `reduce_only_service_is_bound` | §3.2 `aggregate.py`, D8 | a `services=`-named spec no External names: the reduce is bound | a bind that skips the reduce unless an External names a service |
| · `plan_without_services_binds_to_an_equal_process` | §3.2 `services.py` | no service nodes: an equal process; no hook: the same plan | a rebuilt or altered process |
| · `split_endpoint_gives_the_scheme_and_host_port[×6]` | D1 endpoint form | `(scheme, "host:port")` for tcp/http/https/grpc/grpcs and `grpc://[::1]:8001` | a scheme left on the host part; IPv6 brackets dropped |
| · `split_endpoint_refuses_other_forms[bare, unknown-scheme, no-port, path]` | D1, P-f | `ValueError` naming tcp/https/grpcs | a scheme-only check (no port, path accepted) |
| · `bind_refuses_a_bare_endpoint…` | §3.2 `bind_services` | a bare `host:port` refused naming the schemes, also by a hook-less plan; the original stays unbound | a bind that never runs `split_endpoint` |
| `test_triton_service_param` · `endpoint_never_enters_the_ir` | D7 | IR identical unbound and under two endpoints; node params hold `service`, no `url` | endpoint written into IR params |
| · `url_path_records_the_0_0_6_bytes` | D7 | the `url=` recording hashes to the 0.0.6 pin | the literal-URL path changed |
| · `two_services_with_one_payload_connect_twice` | D7 cache key | one payload on two services reaches both servers | the 0.0.6 `(kind, content_hash)` key |
| · `rebinding_connects_to_the_new_endpoint` | D7 cache key | the same node rebound reaches the new server | a key without the endpoint |
| · `url_and_service_are_exclusive[both, neither]` | D7 | `PreserveError` at record time naming both params | both accepted; neither accepted |
| `test_triton_transport_by_scheme` · `scheme_picks_the_module_host_port_and_ssl[http, https, grpc, grpcs]` | D7, §3.2 `triton_external.py` | no `transport`: the scheme's fake `tritonclient` module connects once with `url="host:port"`, `ssl` only on `https`/`grpcs` (`grpcs://triton.fnal.gov:443`), requests built from it, the other module untouched | the 0.0.6 http-only transport; the full endpoint handed to the client; no `ssl` |
| · `tcp_endpoint_is_refused_naming_the_wires` | D7 | `StageError` whose cause is `PreserveError` naming http and grpc, no client built | `tcp` treated as http |
| · `literal_url_without_a_scheme_stays_http` | D7 | `url="tr-literal:8000"` reaches `tritonclient.http` unchanged | every literal url parsed as an endpoint |
| · `the_connection_key_is_the_endpoint_and_the_params_load_reads` | D7 cache key, §3.2 `_base.py` `load_params` | one payload: nodes on one endpoint differing only in `output_name` share one connect, another endpoint gets its own, a differing `transport` reaches its own (fake-transport) server | a key over every node param; a key without the endpoint; a key without the params `load` reads |
| · `correctionlib_universes_off_one_payload_load_once` | D7 cache key (0.0.6 universe sharing kept) | two correctionlib nodes on one payload differing only in `systematic`: one `load`, each universe's value | a key over every node param |
| · `equal_node_params_in_any_key_order_share_one_connection` | D7 cache key | one payload, endpoint and params, keys in two orders, in two sessions: one connect | a key over the params' insertion order |
| · `transport_param_wins_over_any_scheme` | D7, frozen m26/m27 | with `transport`, a literal `triton://` url and a bound `grpcs://` endpoint reach the factory unchanged; neither tritonclient module is touched | the scheme parsed before `transport` is read |
| `test_triton_service_live` · `live_triton_through_a_bound_service[http, grpc]` | D7, §6 | a `service=` node, no `transport`, served by the CI `triton` job's real server at `http://$GRAPHED_TRITON_HTTP` (check `http:/v2/health/ready`) and `grpc://$GRAPHED_TRITON_GRPC` (check `grpc:`), each leg gated on its variable like `preserve/m9/test_triton_server.py`; the gRPC port answers only gRPC | a bind the scheme-chosen tritonclient transport never sees |

Clause ends (each end's test, and the sanity mutant it kills):
| Clause | One end | The other end |
|---|---|---|
| D1 kind-agnostic | Triton nodes (every file) | `GENERIC` in `session_registry…` (`refusal-triton-only`), `any_external_kind_is_bound` (`bind-triton-only`), `…whatever_the_kind` (`plan-triton-only`, `manifest-triton-only`) |
| `Plan.services` referenced | live `scorer-svc` | dead `scorer-site` (`session-wide-refs`) |
| ∪ the named ones | named only, `unused-svc` (`named-ignored`) | named and referenced, `scorer-svc` (`named-not-unioned`) |
| sorted by name (plan, manifest) | two names (`plan-set-order`/`manifest-set-order` at some seeds) | eight names against declaration order (`plan-set-order`, `manifest-set-order` at all 50 seeds run; `plan-declared-order`, `manifest-declared-order`) |
| undeclared name refused | something declared | nothing declared (`empty-registry-accepts`) |
| `services` key only when non-empty | empty (`always-key`) | one spec (the key present and equal to `to_json`) |
| §3 defaults rule | defaults by equality (other defaults) | defaults read-only (`launch-dict-default`, `report-dict-default`) |
| old pickles valid | `Plan` (`plan-services-factory`) | `_PluginEvaluator` (`evaluator-endpoint-factory`) |
| cache key | endpoint (`cache-no-endpoint`, `cache-06`) | the params `load` reads (`cache-no-params`); evaluate-time params (`cache-all-params`); equal params in another key order share (sweep mutant 100) |
| `inspect()` recipe | image (`IMAGE`) | argv (`inspect-image-only`) |
| bind | a service node bound, a `url=` node left alone (`endpoint-in-ir`) | the reduce hook bound, also with no service External (`reduce-needs-external`); a hook-less plan returned as is |
| endpoint form | six accepted | four refused (`split-lenient`); refusal on bind, used or not (`bind-unchecked`) |
| transport | scheme'd endpoints (`scheme-ignored`, `full-url-to-client`, `no-ssl`, `tcp-as-http`) | a scheme-less literal (`literal-split`); `transport` given (`scheme-before-transport`) |
| RunReport | with endpoints (`report-in-fingerprint`) | without (a 0.0.6 report); version (`report-version-2`) |
- The bundle's "IR references" has one end: `build_bundle` serializes with `optimize=False`, which keeps every recorded External, so none is outside the bundle's IR.

Readings of the plan:
- `url=`/`service=` exclusivity is refused when the node is recorded.
- Evaluating an unbound `service=` node raises `UnboundService` naming the service.
- Endpoints are written `scheme://host:port` everywhere they are bound or reported (`grpc://…`, `http://…`); a literal `url=` keeps whatever it recorded (`triton://…` with `transport`, as frozen m26/m27 do).
- The `tcp://` refusal happens when the connection is opened, so a run surfaces it as the 0.0.6 `StageError` with `cause_type == "PreserveError"` (`aggregate.py` passes only a `GraphedError` through untouched).
- `split_endpoint` refusals are `ValueError` (§3.2), and naming the schemes is asserted as `tcp`, `https`, `grpcs` all appearing.
- `to_json()` returns a dict; `manifest["services"]` lists the specs in name order, as `Plan.services` does (a set-ordered list makes the bundle fingerprint depend on `PYTHONHASHSEED`).
- `Plan.services` and `_PluginEvaluator.endpoint` keep a plain class-level default: §3.2's "old pickles valid" governs them over the §3 `default_factory` rule.
- A message is asserted by the names the plan says it carries, each a whole token; its wording is free.
- The live test uses the `scorer` model the graphed `triton` job already serves.
- Import blocks are sorted with `python/graphed/services.py` present; before it exists `ruff check` reports I001 on them.
