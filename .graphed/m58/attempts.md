# m58 — implementer iterations

Branch `m58-projection-declarations` on top of `049e3c7` (= the frozen suite, tag `freeze-m58`).
Run: `python -m pytest tests/frozen/awkward/m58 -q -p no:cacheprovider`.

## Iteration 0 — baseline (7 failing, 4 passing)

Matches the suite README's non-vacuity table exactly: H1 ×3 and H3 on `source.calls == 0`, H4 on the
declared order, E1 on `AxisError: axis=2 exceeds the depth of this array (2)`, E3 on
`AttributeError: no field named 'total'`. H2 ×2, H5, E2 pass (the regression controls).

## Iteration 1 — the hook + the declared stand-in (7 → 1)

`graphed/write.py::declared_columns(source, outputs)` — `getattr` for a callable `projected_columns`,
called once with `tuple(outputs)`, returning `tuple(answer)` or `None`. Used at the three driver
sites, each short-circuiting the driver's own computation (`... if declared is None else declared`
in `aggregate_plan`; `if columns is None:` before `_evaluation_columns` / `_evaluation_columns_union`
in `graphed/awkward/io.py`). The `PartitionedSource` Protocol gains no member.

`projection.py::_replay.on_external` returns a typetracer of `session.form_of(nid)` carrying the
backend's behavior when that form is an `AwkwardForm`, else today's `inputs[0]`. Premise confirmed
first: `graphed-histogram` records fills with `HistogramForm(spec_hash)` (a dataclass with
`describe()` only — no `tt`), so a non-awkward recorded form is real, not hypothetical.

Remaining: `test_plan_boundary.py::test_the_built_plan_carries_the_answer_and_no_worker_asks_again`.

## Iteration 2 — the remaining leg is unsatisfiable → Test Dispute, STOP

`pickle.dumps(plan)` raises `TypeError: cannot pickle 'builtins.GraphStore' object`: the plan's
`process.reader` IS the `DeclaringSource`, whose `outputs` (required by H1 to be the driver's own
`Array`s, by identity) reach `Session._store`. Clearing `outputs` makes the dump succeed; `seen`
then also travels and the shipped run appends to it (4 entries, not 2). Both are fixture recording
state. Filed
`.graphed/m58/disputes/test_the_built_plan_carries_the_answer_and_no_worker_asks_again.md` with a
one-line `__getstate__` correction, verified out of tree to make the test body pass verbatim, and
stopped: no docs, no `tests/extra/awkward/m58/`, no commit. `tests/frozen/**` unmodified.

## Iteration 3 — dispute CORRECTED, re-freeze `freeze-m58-2` → 12/12, then the rest

The amended fixture (`DeclaringSource.__getstate__` ships the answer + counter, drops `outputs`/`seen`)
makes H3 pass on iteration 1's source diff unchanged: `tests/frozen/awkward/m58` is 12/12. Iteration 1's
diff reviewed and kept as is — nothing to add on the source side.

Premise for the E guard re-confirmed on this tree:
`grep -n "class HistogramForm" -A 20 graphed-histogram/src/graphed_histogram/boost.py` → a frozen
dataclass with `spec_hash` + `describe()` and no `tt`, recorded at `form=HistogramForm(chash)` (two
fill sites). The `isinstance(form, AwkwardForm)` arm is a real case, not a defensive one.

Finished the partition:

* `tests/extra/awkward/m58/test_m58_declaration_branches.py` — six legs, each killing a one-hunk
  mutant the whole frozen tree survives: the three drivers' short-circuit (own computation hoisted
  out of its `is None` guard → the patched-to-raise own computation runs), an empty declaration read
  as no answer (`or None` → `('x','y') != ()`), `callable()` dropped from the lookup (`TypeError:
  'tuple' object is not callable`), and the `AwkwardForm` guard dropped (`AttributeError:
  '_ForeignForm' object has no attribute 'tt'`). Each driver leg carries its own control: the same
  patched site with a non-declaring source must raise.
* `pyproject.toml` — `tests/frozen/awkward/m58` on the pytest `pythonpath` and
  `m58_declaration_fixtures` in the bare-name mypy override, exactly as m57 is registered.
* docs — the optional capability in the `PartitionedSource` docstring (who calls it, when, that it
  replaces the driver's own list) and as a fourth detail in the awkward design page's partitioned
  write section. No example added, so nothing to execute.

Gates: frozen m58 12/12; `./scripts/run-tests.sh` green; diff coverage from the FROZEN suite alone
18/18 statements and 7/8 branch arms on changed lines (the one partial is the non-`AwkwardForm` arm,
which the extra suite closes); `precommit --fast` green. `sphinx -W` cannot run in this environment
(`myst_nb` is not installed in `m52/.venv`), so the docs change was measured by building a copy of
`docs/` with the notebook pages removed: rc=0, zero warnings from `awkward/design.rst`, and
`declared_columns` present in the generated `graphed.write` page — with the deliberately broken
notebook toctrees emitting warnings as the live control.

## Iteration 4 — REJECT answered: the write drivers' own sentinel, translated where it is produced

The reviewer's finding: a declared `()` reached `read_partition` as `None` at both write drivers,
because `_WritePart._chunk`/`_VariedWritePart._chunk` read `self.columns or None` — the drivers'
own `_evaluation_columns` family spells "everything" `()`, so the consumer was undoing the
declaration. The two `_chunk` methods now pass `self.columns` through, the two dataclass fields are
`tuple[str, ...] | None`, and each driver's OWN computation is translated at the point it is
produced (`... or None`). `aggregate_plan` needed nothing: `read_columns` already returns `None`
for "everything". Walked every consumer (`grep -n "columns" python/graphed/awkward/io.py`, plus
every `read_partition` call site in `python/`): three readers, all now verbatim. No frozen test pins
a write part's `columns == ()`. `python/graphed/numpy/io.py` keeps `self.columns or None` — that
driver has no declaration hook and its own list is a projection frozenset, a different vocabulary.

Also in this partition: `on_external` is the shorter equivalent `form.tt if isinstance(form,
AwkwardForm) else inputs[0]` (the `_with_behavior` re-wrap was a no-op — `eval_stage` re-wraps every
operand), dropping the `AwkwardBackend` import; `declared_columns`' docstring says the answer is a
sequence of names.

Five legs added to `tests/extra/awkward/m58`; every branch of the translation is now killed by a
one-hunk mutant (script kept out of tree, run with `PYTHONDONTWRITEBYTECODE=1` — same-size mutants
within one second otherwise hit a stale `__pycache__` and read as survivors):

| mutant | leg that fails |
|---|---|
| `_WritePart` / `_VariedWritePart` `self.columns` → `or None` | the two `ships_an_empty_declaration` legs |
| either driver `if columns is None:` → `if not columns:` | the same two |
| either driver's own computation with `or None` dropped | the two hook-less `whole_record` legs |
| `hook(tuple(outputs))` → `hook(outputs)` | `hands_the_hook_a_tuple` (varied driver, the only list) |
| `on_external` guard dropped | `non_awkward_recorded_form` |

Gates: frozen m58 12/12, extra 11/11; `./scripts/run-tests.sh` rc=0 over all 68 subtrees (Python
only — no `cargo test` leg); `precommit --fast` ok. `tests/frozen` byte-identical to `freeze-m58-2`.
