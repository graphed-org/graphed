# m55 implementer attempts

## Target
Frozen `tests/frozen/awkward/m55` (19 tests, tag `freeze-m55`): 17 fail pre-m55 with
`collection '<name>' needs a {tag: record} mapping, got Varied`, 2 declared controls pass.

## Iteration 1 — `_unpack_varied` (context.py)
One normalisation loop in `_vary_shift` before `_check_lockstep`; refusals in order: placement
beside a Varied, labels beyond exactly the registered family, reindexed nominal != the context's
central node; then `{tag: member_of(v, f"{name}_{tag}")}`. `collections:` annotations admit
`Varied` (the merged mapping local is `dict[str, Any]` for mypy). Frozen 19/19 on the first run;
diff coverage from the frozen suite alone 19/19 lines, 10/10 branches; full runner green.
Docs example trap: the context's MET must already be the Type-1 MET of its central jets and the
propagated container must be built from the RAW MET, or its nominal is a fresh node (refused).

## Iteration 2 — review folds (wf_ab7b1e07-439: design/integrity/mutation/docs + refuters)
Standing: MUT-1 (`**tags` spelling with a Varied unguarded by the frozen suite), MUT-2 (a
refusal on a LATER collection leaves the first collection's shift-after-weight diagnostic behind,
and the frozen suite cannot see it because `vary`'s rollback restores the point registry only) —
the latter is pre-existing on the hand form, fixed by rolling the diagnostic registries back too;
both witnessed in `tests/extra/awkward/m55`. RULES-1/2/3: root prompt restructured (R24).
Refuted: INT-1, MUT-3 (messages now pinned in extra anyway), RULES-2 wording (folded).
