# core/m68c — V2 plans carry services and run in SequentialRunner (traceability)

Authority: lane plan `lanes/services-v2/plan-graphed.md` §3.1, §3.2, §3.4 and §6 (the core half).
Hand-built `DurablePlanV2`s with toy stage processes (`process(task, inputs, resources) -> bytes`);
nothing here imports awkward, so the directory runs in the awkward-free core and free-threaded jobs.
Run: `python -m pytest tests/frozen/core/m68c -q`.

| test | plan clause | pins |
|---|---|---|
| `test_m68c_v2_plan_services.py::test_services_round_trip_and_stay_out_of_the_bytes_when_empty_and_out_of_task_ids` | §3.1, §6 *Bytes round-trip* | `services` survives `from_bytes(to_bytes)`; empty → keys exactly `{format_version, ir_b64, stages}`; `task_id` equal with and without services |
| `…::test_binding_keeps_bytes_and_task_ids_and_reaches_the_bindable_stage_only` | §3.1 `OpSpec.live`, §3.2, §6 *Binding keeps identity* | bound plan has the same `to_bytes` and task ids; the Bindable stage resolves bound, the original stays unbound, a non-Bindable stage resolves unchanged; endpoints are checked |
| `test_m68c_sequential_v2.py::test_a_completed_run_returns_the_last_stages_value_in_stage_order` | §3.4, §3.3 `DurablePlanV2.value` | stages in order with full upstream payloads; last `reduce` stage → its one decoded payload, other kinds → tuple in task order; `n_partitions`/`n_combines`; one `LocalResources` per run |
| `…::test_an_unbound_plan_is_refused_before_any_process_call_and_a_bound_one_runs` | §3.2 `require_bound`, §3.4 | `UnboundService('sf')` with zero process calls; the bound plan runs with the endpoint |
| `…::test_a_control_cancelled_before_run_stops_with_no_process_call` | §3.4, §6 *SequentialRunner control* | `stopped=CANCELLED`, `value is None`, no call, control reset |
| `…::test_a_cancel_inside_the_first_stage_stops_before_the_next_stage` | §3.4, §6 *SequentialRunner control* | `wait()` before each stage; no later-stage call; counts are the tasks that ran; control reset |
