# m53 implementer attempts

## Target
42 FAIL / 30 PASS (measured pre-m53). Make the 42 pass; survivors + m48 independent-union stay green.

## C1 — discriminator + joint generator + composes + mint (vary.py, loose overload)

C1 wires the discriminator (`_foreign`) + generator (`_fanout`) through all three overloads (the
tuple return couples them: `gather_members` -> `(one_at_a_time, joints)`, so context.py moves with
vary.py or mypy/runtime break). Two refinements the frozen suite forced beyond the plan's literal
`registered_points` discriminator:
 - STACKED weight: a foreign nuisance in `old._tags` composes into the union via `_two_level(old,)`,
   so it is excluded (`composed`). Distinguishes stacked (btag in _tags) from the §8-g ambient-carry
   (jes carried but _tags empty), which DOES fan out.
 - SPECTATOR: a carrier varied by some OTHER nuisance but not the member's foreign one collapses it
   (m48 test_vary_stacking, loose `vary(base_jes,"jer",up=inner_varied)`); a fresh carrier or one
   sharing the foreign nuisance fans out. Both member objects are byte-identical, so the signal is
   the CARRIER, not the member.
Result: 56 pass, 16 fail (all C3 points= prune + C4 guard). awkward/m48 green (regression fixed).
