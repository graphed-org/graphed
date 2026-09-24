# M65 frozen suite, unit B, graphed-core slice (`freeze-m65b`, plan-B.md)

Frozen and read-only after `freeze-m65b`. A new file beside the A1 `README.md`, which it does not edit
(plan.md decision 4). Plain-Python process functions only; no awkward.

| Test | Contract | Fails |
|---|---|---|
| `test_m65b_lean.py::test_sequential_lean_emits_submitted_and_terminal_only` | B-1, B-2 on `SequentialRunner` | a STARTED in lean mode; a labelled terminal; an unlabelled SUBMITTED; a failing task without `[SUBMITTED, ERRORED]` and its error text; a changed result |
| `…lean::test_lean_opt_in_is_exactly_true` | B-1 `lean_events` | a truthy (`1`) opt-in; a lean stream for a monitor that did not opt in exactly |
| `…lean::test_worker_monitor_factory_helper` | B-1 `worker_monitor_factory` | a helper that is missing, raises on a monitor without the method, or wraps the factory |
| `test_m65b_label.py::test_blind_partitions_label_their_step` | B-7 | every blind step labelled `uri::0-0`; a changed non-blind form |
| `…label::test_sequential_runner_formats_no_label_without_a_monitor` | B-3 | any `partition_label` call by an unmonitored `SequentialRunner` (the monitored run proves the patch is live) |
