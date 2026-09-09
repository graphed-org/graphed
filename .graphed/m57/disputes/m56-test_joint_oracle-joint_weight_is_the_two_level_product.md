# Test Dispute — tests/frozen/awkward/m56/test_joint_oracle.py::test_the_joint_weight_is_the_two_level_product_of_the_factors

## The test
Over the m56 capstone (`jes` shifts `Jet`/`MET` AND swaps the SF table by name identity; `hf` is a
second weight family over the same jets), for each `jes` direction `d` and each `hf` tag `t` it
asserts the ambient at `hf_t__jes_d` (and at `jes_d` for `t == "nominal"`) equals
`m56_sf(jets_d, JES_SF[d]) * m56_sf(jets_d, HF_SF[t])` — the product, in registration order, of BOTH
weight factors' members at that point.

## The clause it contradicts
Plan `weight-dedupe-plan.md` §2.1: the weight form's central NAMES the factor it varies, compared by
node, so a family whose central is a factor already registered joins that factor instead of adding a
second one; its members are then that factor's values in their universes and the composition reads
ONE member per label. §2.6 anticipates this exact case: a frozen test that pins a squared factor is
a Test Dispute for the owner.

The capstone's two families share a central NODE. `JES_SF["nominal"] == HF_SF["nominal"] == 1.0`
(`m56_fanout_fixtures.py`), so `m56_sf(jes_jets, JES_SF["nominal"])` and
`m56_sf(sjets, HF_SF["nominal"])` are the same expression on the same nominal jets. Measured on main
(`scratchpad/dedupe/p4main.py`): the two factors' nominal node ids are **24 and 24**. Under §2.1 the
`hf` registration therefore joins the `jes` factor, and the capstone composes ONE entry, not two.

Main's own values say the same thing (`scratchpad/dedupe/p5.py`, first three events):

| quantity | main | prototype |
|---|---|---|
| entries at the capstone context | 2 | 1 |
| nominal ambient | `[0.9385325, 1.0947904, 0.7566725]` | `[0.9687789, 1.0463223, 0.8698692]` |
| `SF(nominal jets)` | `[0.9687789, 1.0463223, 0.8698692]` | (same) |
| `SF(nominal jets)²` | `[0.9385325, 1.0947904, 0.7566725]` | (same) |
| `jes_up` | `[0.9803379, 1.1598454, 0.7741333]` | `[1.0119315, 1.1084973, 0.8899422]` |
| `SF(jes_up jets, JES up)` alone | `[1.0119315, 1.1084973, 0.8899422]` | (same) |
| `hf_up__jes_up` | `[1.0463928, 1.2643612, 0.8010259]` | `[1.0340549, 1.1406083, 0.9000876]` |
| `SF(jes_up jets, HF up)` alone | `[1.0340549, 1.1406083, 0.9000876]` | (same) |

Main's nominal ambient is EXACTLY the squared SF. The label set is unchanged (15 labels on both), so
the fan-out is untouched; only the values move.

Not covered by §2.1: what the ambient is at a JOINT universe of two families that have joined ONE
factor. A container yields one member per label, so the `jes` weight family's own `jes_up` member is
no longer multiplied at `hf_up__jes_up`. §2.1's worked example (`hf_up = SF(up_hf)`,
`lf_up = SF(up_lf)`) is the one-at-a-time case only. This is the owner's call as much as the values
are.

## Proposed correction
Two corrections are available, and they do not guard the same property.

**A. Re-oracle to one SF.** The joint becomes the fanned family's cross member alone
(`m56_sf(jets_d, HF_SF[t])`) and the nominal becomes `m56_sf(jets, 1.0)`. This matches §2.1 but
DROPS the property the test exists for: with one factor there is no "product, in registration order,
of each factor's member at that point" left to check. What survives is the two-level READ, which the
m57 suite's own extend-and-fan-out test already covers.

**B (recommended). Give the two families distinct centrals.** Set `HF_SF["nominal"]` to a value
other than `JES_SF["nominal"]` in `m56_fanout_fixtures.py`, so the two centrals are different nodes,
the two families stay two factors, and the test keeps its product oracle verbatim. The property the
test guards — the joint's weight is the two-level product of BOTH factors at that point — is then
exercised exactly as before, on a program §2.1 does not merge. Every other m56 test computes its
oracle from the same table constants, so they follow the change; the suite must be re-run to confirm
that, and the re-freeze needs the owner's affirmation.

Recommendation: **B**. A weakens a capstone test into a case another test already covers, while B
keeps the two-factor joint under test and moves the fixture off the coincidence that §2.1 now reads
as a deliberate re-use.

## Resolution
Pending the owner's decision.
