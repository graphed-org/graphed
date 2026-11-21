# Test Dispute — `tests/frozen/awkward/m52/test_registration_projection.py::test_a_default_point_registration_reduces_same_family_but_fans_out_a_foreign_one` (two assertions) and `tests/frozen/awkward/m52/test_projection_resolution.py::test_the_fast_path_returns_the_containers_own_member_by_identity`

Status: CLOSED — owner rulings via team-lead: #231 "readers return a detached proxy of the same node. The four `is` assertions become `.node_id ==`; re-aim the identity-pinning fast-path test", and "refreezes are authorized"; refrozen in the #231 PR, tag freeze-m52-fixup3.

## The pins

- `graphed.universe(z_tag, "jes_up") is graphed.nominal(other_tag)` and `graphed.universe(z_family, "jes_up") is graphed.nominal(other_family)`: reader results compared by object identity, meaning "the same node".
- `graphed.member_of(container, label) is container._members[label]`: "Object identity pins that the fast path was kept".

## The clause it contradicts

graphed#231: the readers return a detached proxy of the member's node, so assigning into it leaves the container as it was. No reader result is a member object any more.

## Correction

- The two registration assertions compare `.node_id`.
- The fast-path test is renamed `test_the_fast_path_resolves_a_carried_label_to_its_own_members_node`. With `_key_for` (the §4.6 restriction) replaced by one that raises, every carried label still resolves, to the member's node and to an object that is not the member. A `_member_for` that always restricts fails it. README rows follow.
