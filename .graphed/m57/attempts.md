# m57 — implementer iterations (v14)

Branch `dedupe/prototype` on top of `46497ba` (= the frozen suite, tag `freeze-m57`). Run:
`PYTHONPATH=python python -m pytest tests/frozen/awkward/m57 -q -p no:cacheprovider`.

## Iteration 0 — classification (32 failing, 36 passing)

Cause classes: **(a)** the surface the suite froze vs the shape the code hands out; **(b)** a value
or refusal the code gets wrong against §2/§3; **(c)** the code is right and the test contradicts the
plan → Test Dispute.

| test | class | cause |
|---|---|---|
| `test_explain::*` (8 legs) | (a) | `Explanation.families`/`variations` are tuples; the suite reads them by name/label. `Family` has no `entry` record, its `placements` carry `(label, point)`, its links are rendered names; `Operation` has no `str()` line and renders its links |
| `test_records::test_ambient_entries_carry_…` | (a) + (c) | (a) the rider's links are `("mask", <Array>)`, the suite froze `("mask", None)` pairs from the root. (c) its `projected` leg registers the owner of `hf_up` at that projection, which §2.3 refuses |
| `test_records::test_ambient_entries_list_the_order_…` | (a) | `ambient_entries` reports the LIVE (adoption-expanded) list; the order the composition applies is the context's OWN list, whose head is the adopted operation |
| `test_records::test_ambient_entries_keep_their_slots_…` | (c) | reads `m57_ambient_values(session, graphed.weight(projected))`, and at a projection that is the bare adopted member: `graphed.labels(Array)` is refused (pre-m57, pinned by an extra test) |
| `test_lineage::*` (7 legs), `test_projection::test_naming_the_factor_that_owns_…`, `…a_node_that_is_both_…` | (b) | the central is `graphed.reindex_to(x, ctx)`: `_same_node` split on a projection (which keeps the identity) and had no way to see a node re-indexed across a mask |
| `test_projection::test_at_a_placed_label_…`, `…below_a_second_projection_…`, `…below_a_further_projection_…`, `…inside_an_overlays_universe_…`, `test_reads::test_an_ancestor_factor_join_after_the_overlay_…` | (b) | an ancestor-factor join REPLACED the adopted head by the operations it stands for, re-composing the adopted product into an equal value under a NEW node; §2.3's join "keeps the universe projected into" |
| `test_projection::test_a_parent_built_projected_nominal_…` | (c) | its control's central IS the node the projection adopted and recorded as its read (measured); §2.3's parenthetical that it is "not the adopted node" is false |
| `test_staleness::test_a_union_that_widens_…` | (b) | the widening refusal fires after `gather_members` minted the family's cross members |
| `test_projection::test_a_central_equal_to_the_owners_member_…`, `…at_the_nominal_projection_a_re_derived_central_…`, `test_reads::test_at_the_nominal_projection_a_child_factor_…`, `test_lineage::…widened_factor_is_refused_as_stale`, `test_placements::test_a_placement_at_a_point_…` | (b) | see iterations below |

## Iteration 1 — one identity predicate, plus the re-index peel (32 → 23)

`_same_node` no longer splits on a projection (§2.3: a projection re-indexes the composed product,
not the nodes a central names, so a central built above it or re-stamped at it names its factor),
and answers for a central the user carried down with `graphed.reindex_to` across a mask:
`_reindexed_onto` peels one `getitem` per mask link off the Session's op record — provenance the
re-index already wrote, so deciding still mints nothing. `_same_member` is gone: with the projection
test dropped it was `_same_node`, and both arms now ask the one predicate.

Closed: the whole `test_lineage` group but the stale leg, `test_naming_the_factor_that_owns_the_label_is_refused`,
`test_a_node_that_is_both_a_nominal_and_the_owners_member_joins_that_factor`.
Remaining: 23 (the (a) surfaces, the head-node class, the five singles, the four disputes).

## Iteration 2 — the adopted head is a CONTAINER a join extends, not a list it expands (23 → 19)

An ancestor-factor join rebuilt the child's entry list as the operations the head stands for, with
the joined one replaced, and re-composed it. The product it made is EQUAL to the adopted node but a
different node: `_compose` folds `_product_tree` over the nominals and `_product` LEFT over
`[rest, *parts]` per label, and the adopted head was built by the parent's own fold (measured: 43 vs
47, 33 vs 44, 45 vs 83 at the projected/masked nominal). §2.3's join keeps the universe projected
into, so the head KEEPS its slot and its node and gains one product per label the join MOVED
(`_headed`), while `EventContext._head_ops` records what it stands for (`_Stood`) for the next join.

Closed: `test_at_a_placed_label_…`, `…below_a_second_projection_…`, `…below_a_further_projection_…`,
`test_an_ancestor_factor_join_after_the_overlay_…`, `test_a_central_equal_to_the_owners_member_…`.

## Iteration 3 — `ambient_entries` reports the ambient this context COMPOSES (19 → 18)

It walked `_live_factors`, so an adoption showed the parent's operations expanded:
`[factor, factor, overlay, factor]` where the suite froze `[factor, overlay, factor]`. It now walks
the own list and collapses each product RUN the head stands for into ONE record (the head's slot,
the head's entry, the union of the run's families), while an overlay among them keeps its own line —
`test_explain_marks_a_fixed_overlay_…` reads a fixed overlay behind the head.

## Iteration 4 — the rider's links are root-relative `(kind, label)` pairs (18 → 13)

`Rider.links` carried the mask's `Array` payload (`(("mask", Array(node_id=27)),)`); the suite froze
`("mask", None)` and `("project", "hf_up")` pairs from the ROOT down. `_row_links` reads them off the
lineage and drops the payload. `_member_here`/`_owned_inside` are insensitive to the added outer
links because `rider.projected` gates the member arm (verified by the projection group).

## Iteration 5 — the explain records take the frozen suite's shape (13 → 12)

`Explanation.families` keyed by name, `.variations` keyed by label (nominal included), `Family` with
`kind/tags/links/entry/placements` and four relation frozensets, `Entry(kind, families, collections)`,
`Operation(position, slot, kind, families, links, nodes, fixed, fixed_at)` with its own `__str__`,
`Variation(label, origin, families, point)`. `Operation.families` is a Mapping (`"mu" in
fixed[0].families`). `_path_from_root` is gone — the links come from the rider.

## Iteration 6 — `composes_with` is the symmetric no-shared-point relation (12 → 11)

It was derived from `ambient_entries`, so the collapsed head made every family behind a projection
share one operation and `composes_with` went empty. `shared` now reads `_live_riders`/`_live_factors`
and composition is "the two register no point in common, in either direction" (measured: `lf_up`'s
point is `(('hf','up'),('mu','up'))`, so the tour's `pu` composes with `lf` as well).

## Iteration 7 — the widening refusal is decided before anything mints (11 → 10)

`_union_nominal` raised it, which is after `gather_members` has minted the family's labels and cross
members. `_check_widening` runs in `_vary_weight` before the mint, comparing LABEL SETS through
`_coordinates` (the first attempt compared `member_of(central, "nominal")` and did not raise at all).
`_crossed` also stopped adding a nominal prior at a non-nominal projection, where the expansion
re-stamped a projected member as a nominal identity and turned an owner refusal into a join
(measured: prior 45 == the central). `_carried_up` + `_lineage_reads` yielding each record's HOME
refuse a stale crossed handle (the record holds the ancestor's nodes, the handle the masked ones),
and `accessors.reindex_to` accepts a handle from a `vary` sibling row space (`_row_space_above`) —
m48's divergence legs use genuinely different row spaces (a second Session, mask-derived siblings).

## Iteration 8 — composition candidates come from the RIDERS, not the tag maps (10 → 9)

`test_at_the_nominal_projection_a_child_factor_makes_both_orders_agree`, class (b). At a projection
the adopted head is a bare member carrying no `_tags`, so `composed = frozenset(ctx._ambient_tags())`
was EMPTY there and `vary._foreign`'s spectator gate (`carrier_nuisances` empty → every nuisance
genuine) fanned the relative-delta's members out over the parent's `pu`/`hf` labels —
`mu_up__pu_up`, `mu_down__hf_up`, … — which the projected context does not compose. Registering the
child factor first made the gate see `{trig}` and drop them, so the two orders disagreed. The field
the decision failed to read is `Rider.families`: the head still composes the families its rider
names (a projection answers them with the universe projected into). `composed` is now the union over
`_live_riders(ctx)`, so the node test (`_reads_ambient`) decides in both orders and the label sets
and every value agree. m48/m53/m56 stay green (107 passed).

Remaining: 9, all measured dispute candidates — see `.graphed/m57/disputes/`.

## Iteration 9 — a handle read at an ancestor names its record in BOTH spellings (9 failing, unchanged)

The r21impl battery, not the suite, caught this: iteration 7 keyed the read match on the central
PEELED up to the record's row space, which is the crossed spelling only. A handle read at the parent
and passed at a masked child exactly as it was read stopped matching, so `mu` on it became a NEW
FACTOR and the ambient SQUARED (measured on the §2.3 program `w = weight(ctx)`, then a cut, then `mu`
on `w`: nominal 13.5 where v13 and the plan say 4.5; the strict-prefix handle fell through to the
factor arm, `('factor', 5)` for `('prefix', 5)`, against §2.1's read-first rule). `_carried_up` now
returns BOTH node tuples — the central as it stands and the peeled one — and the read loop asks
`members in keys`. R1/R3/R9's mask legs and `T1_v14_repairs` leg 4 witness it.

## Iteration 10 — the head composes only the operations it stands for (9 failing, unchanged)

Also from the battery: `_extend`'s ancestor branch composed `_expanded(ctx)`, which is the whole live
list — the head's operations AND whatever this context registered after the adoption. At a label the
join moved, the child's own factors were multiplied in twice (measured: `lf_up = 2.4 x nominal` where
the oracle says 1.2, the doubled factor being the child's `tr`). `_live_slots_behind` restricts the
re-composition to `_stood_for(ctx)`'s slots; the list keeps `[head, *ctx._factors[1:]]` as before, so
each operation is multiplied exactly once. R1's four legs and `T1_v14_repairs` leg 2 witness it.

## Iteration 11 — a projection's own universe is reported (9 failing, unchanged)

`explain` built `variations` from `labels_of(the ambient)`, and a row-space change hands its child ONE
composed member carrying no labels, so the record listed NO universe at a projection. The universe the
context is inside is its nominal, and that is now what it reports (`Q6`'s projection leg, `T1` leg 7).

## Iteration 7, corrected

`_check_widening` did not close `test_a_union_that_widens_…`: that test fails on its own argument's
mints (disputed). Measured by mutation — with the call replaced by `pass` the suite is still 59
passing, and the widening test fails DID NOT RAISE instead, so the early check is the ONLY path that
raises §2.1's covering-overlay refusal. What iteration 7 closed was the `_crossed` prior, the stale
crossed handle and the `vary`-sibling `reindex_to`.

## Final — 59 passing, 9 disputed

The nine are filed under `.graphed/m57/disputes/`, each with the measurement and a verified
correction. Every earlier frozen suite is green (m48-m56: 177 passed; the whole tree via
`scripts/run-tests.sh`).

## Iteration 12 — the implementation review's three blockers and five LOWs (68 passing, unchanged)

The review found every mechanical gate green and rejected on judgment. One shipped §2.1 defect:
`_check_widening` differenced the RAW central against the entry's nominal MEMBER, so after a nominal
projection re-indexed the head the two operands stood one level apart and a join whose union adds
nothing was refused in one of its two spellings. Both are now read at the level this context reads
them, peeling only the projections between a value's row space and here — a mask carries a label set
unchanged, and following one would mint inside a refusal that must leave no node behind.

The rest were frozen properties that cannot fail in the direction they guard: §2.3's covered-READ
refusal had no witness at all, and the two-mask identity chain's own leg spells its central
`reindex_to`, which matches with no prior, so it passed a build that squares the SF. Seven legs in
`tests/extra/awkward/m57/` close those and the three order/listing survivors; each kills a one-hunk
mutant the whole frozen tree survives, and the frozen suite kills the control mutant neither of them
would otherwise reach. The mask identity `_crossed` appends, measured dead over the frozen legs, has
an admitted member after all: a context-free central is never recorded as an ancestor's node
re-indexed, so peeling the mask off the op record cannot reach it and only that append answers.

What remained: nothing from the review. `docs/frontend/design.rst`'s weight-form example now imports
`explain` and runs verbatim.
