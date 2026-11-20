# Test Dispute — `tests/frozen/frontend/m49/test_impact_sets.py::test_member_of_is_exported_and_resolves_the_label_union`

Status: CLOSED — owner rulings via team-lead: #231 "readers return a detached proxy of the same node. The four `is` assertions become `.node_id ==`", and "refreezes are authorized"; refrozen in the #231 PR, tag freeze-m49-fixup5.

## The pin

`graphed.member_of(container, "jes_up") is graphed.universe(container, "jes_up")` and `graphed.member_of(container, "btag_up") is graphed.nominal(container)` compare reader results by object identity, where they mean "the same node".

## The clause it contradicts

graphed#231: assigning into a universe read from a container changed the container, because the readers returned the container's own member object. The ruling makes `nominal`, `universe` and `member_of` return a detached proxy of the same node, so two reads are two objects.

## Correction

Both assertions compare `.node_id`. `member_of(plain, "jes_up") is plain` stays: a plain value is returned as itself.
