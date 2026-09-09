# Test Dispute — tests/frozen/awkward/m56/test_label_precision.py::test_a_member_that_is_a_factors_two_level_member_is_composed_at_that_label_alone

## The test
On `m56_two_both_kind` (`jes` and `jer` each registered as a shift AND as a weight), it registers a
`probe` family whose member is `m56_sf(sjets, HF_SF["nominal"])` — chosen because that node is what
the `jer` weight factor two-level-resolves to at a `jes` label — and asserts
`m56_minted(weight, "probe") == m56_joints("probe", "jer")`: the `jes` coordinate is composition and
is dropped, the `jer` coordinate is a dependency and mints four joints.

Under the prototype the assertion fails with four EXTRA labels: `probe_up__jes_up`,
`probe_up__jes_down`, `probe_down__jes_up`, `probe_down__jes_down`.

## The clause it contradicts
Plan `weight-dedupe-plan.md` §2.1 (the nominal names the factor, by node) and §2.2 (two-level reads
are preserved; the m56 composition test sees the joined container as ONE factor and decides as
before). The test does not contradict either rule directly — it is collateral of the same merge the
sibling dispute reports, and §2.6 puts it in front of the owner for the same reason.

`m56_two_both_kind` registers the `jes` weight with central `m56_sf(jes_jets, JES_SF["nominal"])`
and the `jer` weight with central `m56_sf(sjets, HF_SF["nominal"])`. Both tables have nominal 1.0,
so both centrals are one node: measured on main (`scratchpad/dedupe/p4main.py`), the two factors'
nominal node ids are **24 and 24**, and the context composes 2 entries on main against 1 on the
prototype.

The consequence for THIS test is a change in what the ambient contributes at `jes_up`, not a change
in the fan-out rule:

* main: `_two_level(jer_factor, "jes_up")` finds no `jes` coordinate on the `jer` factor, falls to
  its nominal, and lands on `m56_sf(sjets, 1.0)`'s `jes_up` universe — exactly the probe's member,
  so `_reads_ambient` is true and `jes` is composition;
* prototype: the two families are one container, whose `jes_up` member is the `jes` family's OWN
  member `m56_sf(jes_jets, JES_SF["up"])`. The probe's member does not read that node, so `jes` is a
  genuine dependency and fans out.

The probe registration itself is not an extension: `m56_weight_family` builds the central as
`member * PROBE_SF["nominal"]`, a fresh mul node (measured: central nominal node 85, raw member node
24, `_extension -> None`). Only the fixture's own two families merge.

## Proposed correction
Build the probe's member from what the ambient ACTUALLY contributes at the label, not from the
second family's nominal re-evaluated on the shifted jets: take `graphed.member_of(graphed.weight(ctx),
"jes_up")` at the `m56_two_both_kind` context — the joined container's real `jes_up` member — and
compose the probe's per-universe member from it, keeping a genuine `jer` coordinate (built from the
jer-varied jets, as `m56_per_universe_member` in the same file already does) so the fanned half of
the property stays exercised. The assertion
`m56_minted(weight, "probe") == m56_joints("probe", "jer")` then stands unchanged and tests the same
rule: a member that IS a factor's two-level member at a label is composed at that label alone.

This correction is independent of how the sibling dispute is settled. If the owner takes correction
B there (distinct table nominals, so the two families stay two factors), this test's premise is
restored as written and no change is needed here — the two disputes should be decided together.

## Resolution
Pending the owner's decision.
