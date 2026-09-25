# Test dispute — `tests/frozen/frontend/m48/test_varied_schema_absence.py::test_plan_schema_is_unchanged_by_a_varied_program`

m68 §3.2 adds a `Plan` dataclass field for every plan; this m48 test pins `Plan`'s field set to a
literal set without it. No implementation of the m68 plan as written satisfies both.

## The test

```python
def test_plan_schema_is_unchanged_by_a_varied_program() -> None:
    plan, _result, _monitor = _varied_run()
    assert {f.name for f in dataclasses.fields(plan)} == {
        "process", "combine", "empty", "tasks", "next_tasks", "stop", "open_once",
    }
```

Failure with the m68 implementation (branch `m68-services`, uncommitted, staged):
`AssertionError` — the field set also holds `services`.

## The clauses

- m68 (`lanes/htcondor/plan-services.md` §3.2, `core/execution.py` row, binding):
  "`Plan.services: tuple[ServiceSpec, ...] = ()` (old pickles valid)". The frozen m68 suite relies
  on it being a field: `Plan(...).services == ()`, `bind_services` keeps it through
  `dataclasses.replace` (`test_bind_services::test_a_bound_plan_carries_the_endpoint...`
  asserts `bound.services == plan.services != ()`), and the 0.0.6 pickle loads with `services == ()`.
- The m48 clause this test implements (`systematics-vary-plan.md`, §7.2 schema-absence anchor):
  "`ExecResult`/`Plan`/monitor **schemas** do not change in m48–m50 ... worded over the schema KEY
  SETS of `Plan`, `ExecResult`, and `TaskEvent` ... asserted against LITERALLY SPELLED expected
  sets". The clause is scoped to m48–m50 and to what a *varied* program does; the literal set has
  no such scope, so any later plan-sanctioned `Plan` field fails it.

A non-field `services` (a `ClassVar` default plus an instance attribute that `bind_services` copies by hand) could pass both, but every
`dataclasses.replace(plan, ...)` would silently drop the services, and it contradicts §3.2's field;
that is routing around this test, so it is not done.

## Proposed correction

Add `"services"` to the literally spelled set in `test_plan_schema_is_unchanged_by_a_varied_program`
(the m48 discipline — a literal set, a genuinely varied program — is kept; the set records m68's
field). Re-tag the m48 freeze amendment as the orchestrator requires.

## Ruling (owner, 2026-09-25)

Compare, don't list. The test is re-specified to compare a varied program's `Plan` with a plain
program's `Plan` built from the same analysis: `type(varied) is type(plain)`, equal field-name sets,
equal `vars()` key sets. It must fail a varied path that returns a `Plan` subclass with an extra field
and one that sets an extra instance attribute. The literal `ExecResult`/`TaskEvent` sets stay.
Amendment under `--allow-refreeze tests/frozen/frontend/m48`, re-tagged `freeze-preserve-m68-2`.
