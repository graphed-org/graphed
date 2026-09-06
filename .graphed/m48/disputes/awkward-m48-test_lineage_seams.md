# Test dispute — `tests/frozen/awkward/m48/test_lineage_seams.py`

Two tests in this file require a read performed THROUGH a shift-varied or `Varied`-mask-derived
event context to be a plain `Array`. Plan §2.6b/§2.6c bind those reads to be `Varied`. No
implementation can satisfy both.

## The tests

1. `test_link_kind_1_a_mask_derivation_makes_an_UNVARIED_value_varied`

```python
session, shifted, mask, sel, _root = _varied_mask_program()
value = shifted.MET.pt
assert not isinstance(value, graphed.Varied)
```

2. `test_reindex_to_is_the_identity_for_a_value_already_at_the_target_or_context_free`

```python
_s, shifted, _mask, sel, root = _varied_mask_program()
at_target = sel.MET.pt
assert graphed.reindex_to(at_target, sel).node_id == at_target.node_id
```

Both draw on the file's own program:

```python
def _varied_mask_program() -> tuple[Session, Any, Any, Any, Any]:
    session, events, root = context_with_root()
    shifted = graphed.vary(events, "jes", **jes_kwargs(events))
    mask = gak.num(shifted.Jet[shifted.Jet.pt > 25.0]) >= 4
    return session, shifted, mask, shifted[mask], root
```

`vary_ctx_fixtures.jes_kwargs` names **both** `Jet` and `MET` as shifted collections, and `mask`
is `Varied` (the same file's assertions `list(graphed.labels(result)) == list(graphed.labels(mask))`
and `graphed.universe(mask, label)` require it).

## The spec clauses they contradict

Plan §2.6b, on the shift form:

> The shift form replaces the named collections with `Varied` members (thereafter
> `events.<Collection>` is a `Varied` and §2.3 broadcast carries it; repeated calls stack, §2.1)

`MET` is a named collection of the `vary` call, so `shifted.MET` — and therefore `shifted.MET.pt` —
is a `Varied`. Test 1's assertion is unsatisfiable.

Plan §2.6c, on a context derived by a `Varied` mask:

> **Varied contexts (per-label row sets) are first-class.** When the derivation mask is itself
> `Varied` … the derived context's ROW SET DIFFERS PER LABEL. Binding: its collections READ as
> `Varied` (§2.4-aligned per label)

`sel = shifted[mask]` with `mask` `Varied`, so **every** branch read through `sel` is a `Varied`,
independently of which collections the shift named. `Varied` refuses `node_id` by §2.2's
reserved-name rule — which `test_module_verb_dispositions.py::test_node_id_and_session_raise_while_
the_field_of_that_name_still_reads` asserts — so test 2's `at_target.node_id` raises whatever branch
it reads. Renaming the branch cannot repair test 2.

## Measurement

Against the implementation at the time of filing (plan-conforming §2.6b/§2.6c reads):

```
mask                 Varied         labels=['nominal', 'jes_up', 'jes_down']
shifted.Jet          Varied         labels=['nominal', 'jes_up', 'jes_down']
shifted.MET          Varied         labels=['nominal', 'jes_up', 'jes_down']
shifted.MET.pt       Varied         labels=['nominal', 'jes_up', 'jes_down']
shifted.Photon.pt    Array          (unvaried: Photon is not a named collection)
sel.MET.pt           Varied         labels=['nominal', 'jes_up', 'jes_down']
sel.Photon.pt        Varied         labels=['nominal', 'jes_up', 'jes_down']
root.MET.pt          Array          (context-free)
```

Both `Photon` rows are the positive controls: the instrument does distinguish varied from unvaried
reads, so the four `Varied` verdicts are measurements, not a blanket answer.

## Proposed correction

Test 1 — its stated intent ("an UNVARIED value" re-indexed across a `Varied`-mask link) is sound;
only the branch it picks is varied. Read a collection the shift does not name:

```python
value = shifted.Photon.pt  # Photon is not in jes_kwargs, so this read is unvaried
```

Test 2 — the "already at the target" arm needs a target context whose collections are unvaried. A
`Varied`-mask-derived context has none, so either take the target from a nominal-projected context:

```python
central = graphed.nominal(sel)  # a link-kind-(3) projection: collections read unvaried
at_target = central.MET.pt
assert graphed.reindex_to(at_target, central).node_id == at_target.node_id
```

or keep `sel` and assert the identity on the container:

```python
at_target = sel.MET.pt
assert graphed.reindex_to(at_target, sel) is at_target
```

The `context_free` arm of test 2 (`root.MET.pt`) is unaffected and passes as written.

## Status

Implementation STOPPED on these two tests. Nothing under `tests/frozen/**` was edited, skipped or
weakened, and no implementation shortcut was taken to make either pass. The remaining 110 m48
frozen tests are green.

## Resolution

UPHELD by independent adjudication (2026-09-01): the plan binds the disputed reads as `Varied`
(§2.6b named-collection reads; §2.6c derivation property; `reindex_to` identity worded over
handles, §6.1d(B); `.node_id` on `Varied` refused, §2.2 rule (1)). The anchors were sound; the
operands were mis-chosen. Corrections applied by the test-author at `affd355`
(test 1: operand → `shifted.Photon.pt`; test 2: adjudicated per-label `node_id`-equality body —
strictly stronger than the original arm). Re-frozen as tag `m48-freeze2`. No implementation
change was required.
