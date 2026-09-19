# m60 — implementer iterations

Branch `m60-integration-seams` on top of `720944b` (= the frozen suite, tag `freeze-m60`).
Run: `python -m pytest tests/frozen/frontend/m60 tests/frozen/awkward/m60 -q -p no:randomly`.

## Iteration 0 — baseline (27 + 3 failing)

Matches both suite READMEs' non-vacuity tables: frontend/m60 27 failing / 10 passing,
awkward/m60 3 failing / 3 passing.

## Iteration 1 — X + O (session identity)

One guard, `Session._mine(arrays, located=None)`, called from `serialized_ir`, `form`,
`provenance`, `walk` (which is how `materialize` reaches it), the four `record_*` entries and
`graphed.compile_ir`. The `record_*` family passes its `(op, provenance)` pair and gets the
`GraphedTypeError` `record_op` already raised; the readers pass nothing and get a `TypeError`.
`record_op`'s own inline check is deleted — the guard IS that check, hoisted.

One naming helper, `Session._fn_name(fn, name)`, called from `Array.map`, `graphed.apply` and
`graphed.numpy.apply_gufunc` (the three sites `grep -rn '"fn"' python/graphed` finds). It holds
`id(fn) -> (fn, unique)` — the object is kept so the id cannot be recycled — plus a per-derived-name
count, under a `threading.Lock`. `name=` is returned untouched.

The ordinal arm is an `if` statement, not a ternary (`# noqa: SIM108`): coverage.py does not split
the arms of a conditional expression, and both arms are certified by frozen legs. `_mine`'s
`(op, prov)` collapsed into one parameter after the two-parameter form left a `prov or capture()`
arm no caller reaches.

## Iteration 2 — V + P + E + D

`graphed.provenance.register_internal(prefix)` rebinds a module-global `_SKIP` tuple of
dot-terminated prefixes under the existing lock, through a set so a repeat registration cannot grow
it. `capture` reads the tuple once per walk and tests `(name + ".").startswith(skip)` — one
`str.startswith` call per frame however many libraries registered, and the trailing dot is what
makes a registered prefix match whole dotted components while leaving the built-in `graphed` rule
(a bare string prefix) unchanged.

`graphed.awkward.io._schema_form` writes `schema.empty_table()` through pyarrow to a
`TemporaryDirectory` and reads it back with `ak.from_parquet`; the `columns` selection still runs
on the resulting form (`select_columns`), so column semantics are exactly today's. The scratch file
is outside the dataset, which is what keeps the frozen P3 path-scoped guard green.

`expand` re-exported from `graphed/__init__.py` into `__all__`; its annotations mention no `Array`,
so m48's §2.3d walk demands no `VERB_DISPOSITIONS` entry.

Docs: `docs/architecture.rst`'s "serialized by value" bullet corrected — measured, the IR carries
only the payload descriptor and the wired evaluator fails ordinary `pickle` for a lambda, so on a
process-pool runner an External's callable must be module-level. `docs/frontend/design.rst` grew
"Wrapping graphed in a library of your own" (register_internal, expand, the `name=` rule, the
own-session refusal); `docs/awkward/design.rst`'s parquet section states the form keeps record
names and parameters. Every example executed (`<scratchpad>/m60/docex.py`, `mylib/`).

### Branch evidence

16 mutants, each run against the frozen suites in a scratch copy of `python/graphed`
(`<scratchpad>/m60/mutate.py`, `PYTHONDONTWRITEBYTECODE=1`). pytest's own `pythonpath` ini puts the
working tree's `python/` ahead of `$PYTHONPATH`, so the runner overrides that entry with `-o`;
without it every mutant "survived" against an unmutated import. A `node_count` mutant is carried as
the live-instrument control.

| mutant | killed by |
|---|---|
| CONTROL-instrument-live | `test_two_distinct_callables_are_two_nodes…[lambdas-map]` |
| X-guard-inverted | `test_two_distinct_callables_are_two_nodes…[lambdas-map]` |
| X-guard-removed | `test_every_entry_refuses_a_foreign_array_and_records_nothing[record_exchange]` |
| X-located-flipped | `test_every_entry_refuses_a_foreign_array_and_records_nothing[record_exchange]` |
| X-compile-ir-unguarded | `test_a_module_verb_refuses_a_foreign_array_and_records_nothing` |
| O-name-ignored | `test_an_explicit_name_is_the_callers_identity_declaration` |
| O-memo-ignored | `test_one_callable_object_recorded_twice_is_one_node_and_apply_interns_with_map` |
| O-always-ordinal | `test_the_second_object_of_a_derived_name_records_that_name_with_an_ordinal` |
| O-never-ordinal | `test_two_distinct_callables_are_two_nodes…[lambdas-map]` |
| O-map-unnamed | `test_two_distinct_callables_are_two_nodes…[lambdas-map]` |
| O-gufunc-unnamed | `test_two_distinct_callables_are_two_gufunc_nodes[lambdas]` |
| O-apply-unnamed | `test_two_distinct_callables_are_two_nodes…[lambdas-apply]` |
| V-no-component-rule | `test_a_registered_prefix_matches_whole_dotted_components` |
| V-registry-ignored | `test_a_registered_prefix_matches_whole_dotted_components` |
| V-register-noop | `test_a_registered_prefix_matches_whole_dotted_components` |
| E-expand-unexported | `test_expand_is_public_and_is_the_systematics_verb` |
| P-schema-only-form | `test_the_recorded_form_equals_what_eager_awkward_gives_the_same_file` |

## Iteration 3 — STOPPED, test dispute

`./scripts/run-tests.sh` over the whole tree reds exactly one earlier frozen leg:
`tests/frozen/preserve/m25/test_histogram_preservation.py::test_histogram_terminal_bundle_reproduces_bit_for_bit`
passes `session=_record()[0], value=_record()[1]` — two different sessions — into `build_bundle`,
which hands the foreign `Array` to `session.serialized_ir` and reads `session.sourcemap()` against
it. X1 refuses that by construction and no faithful implementation of X1 admits it; the leg is
green today only because the two recordings coincide.

Filed `.graphed/m60/disputes/m25-test_histogram_terminal_bundle_reproduces_bit_for_bit.md` with the
probe and a proposed correction, and stopped without committing, per §A.7 / §B.6.
