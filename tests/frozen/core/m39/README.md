# M39 frozen suite — graphed-core (Exchange IR + ShuffleBackend protocol + DurablePlanV2)

Milestone **M39** (shuffle substrate + repartition). graphed-core owns the IR/plan/protocol seam:
the `Exchange` boundary variant, its GIR1 codec tag, the pure `ShuffleBackend` protocol, and the
additive multi-stage `DurablePlanV2`. **Frozen — read-only after `freeze-M39-0`.**

## Files → plan clause / theme

| File | Covers | Plan clause |
|---|---|---|
| `test_exchange_ir.py` | `Exchange` variant: records a boundary; interns/CSEs; scheme in structural identity; ENDS a stage (survives reduce as itself); byte-deterministic reduction | §2.1, §2.2, §2.4; theme "Exchange variant end-to-end" |
| `test_exchange_serialize.py` | `T_EXCHANGE=5` appended; MAGIC stays `GIR1`; Exchange blob round-trips byte-identical; legacy graphs byte-stable (M8 gate untouched); corrupt tag rejected | §2.3 |
| `test_durable_plan_v2.py` | `DurablePlanV2` additive: distinct string `format_version`; determinism + roundtrip; V1↔V2 cross-reject; V2 `task_id` folds `backend_id` from `routing` (B-r5.2); V1 untouched | §4.4, §7.2 |
| `test_shuffle_backend_protocol.py` | `ShuffleBackend` pure Protocol (§A.4-clean), declares the 6 M39 primitives + `identity`, runtime_checkable import hygiene (ADV-r5.3) | §3.0 |

## Pinned Python surface (test-author decisions the implementer must match)

- `GraphStore.add_exchange(inputs: list[int], params) -> int` — Exchange has **no** `name` (its §2.1
  enum is `scheme: ParamMap` + `inputs`); `params` is the scheme map. `nodes()[i]["kind"] ==
  "exchange"`. *(Ambiguity flagged: §2.4 says "mirror add_op/add_reduction" which carry a `name`;
  §2.1 — the normative enum — has none, so the faithful mapping omits it. See the report.)*
- `graphed_core.execution.ShuffleBackend` — `@runtime_checkable Protocol[Block, Index]`, attr
  `identity: str`, methods `partition`/`concat`/`slice_rows`/`estimated_bytes`/`to_wire`/`from_wire`.
  **M39 exchange half only** (join methods are M40).
- `graphed_core.DurablePlanV2(ir, stages, ...)` + `StageSpec(kind, inputs, process, routing, tasks)`
  exported from `graphed_core`; `format_version == "graphed-plan/2"`; `to_bytes`/`from_bytes`
  deterministic; `DurablePlanV2.task_id(stage_index, task)` folds `stage.routing["backend_id"]`.

## Non-vacuity

Pre-implementation, every Exchange/V2/protocol test fails for the right reason (`add_exchange` /
`DurablePlanV2` / `ShuffleBackend` absent). The **control** `test_legacy_graphs_are_byte_stable`
PASSES pre-implementation (it exercises only today's codec) — it is a deliberate regression guard
that the M8 determinism gate stays byte-identical, not an Exchange acceptance test.
