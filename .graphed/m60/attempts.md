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

## Iteration 4 — review repairs in `_fn_name` and `register_internal`

Iterations 1–3 committed unchanged as `58cea95` (X + O) and `5c08b15` (V + P + E + D). This
iteration repairs what the review found, each as a class rather than the coordinate it named.

**"The same callable, asked for again."** `obj.method` builds a fresh object with a fresh id on
every access, so an `id(fn)` memo could never hit it: `x.map(c.scale)` twice was two nodes, a
six-iteration loop was seven, and `apply(c.scale, x)` stopped interning with `x.map(c.scale)`. The
memo is now keyed on the CALLABLE — bound methods compare equal on `__self__` identity plus
`__func__`, functions and `functools.partial` on identity, which is the same answer. A callable
that cannot be hashed at all keys on its id instead, in a `try`/`except TypeError` statement with
its own leg; the value still holds `fn`, so an id key cannot be recycled.

**"A derived name equal to a name already handed to a different callable."** `name=` lived outside
the derived name space, so `x.map(other, name="q")` then `x.map(q)` silently gave the second
callable the first one's node and result. Every name handed out, declared or derived, now enters
one per-Session set, and a derived candidate advances (`q`, `q#1`, `q#2`, …) until free — which
also refuses the `name="q#1"` decoy and a third same-named callable. A declaration is still never
ordinaled: equal declared names intern (O4), in either order.

**`register_internal`.** `"mylib."` stored `"mylib.."` and matched nothing — a silent no-op in the
call that exists to fix silent wrong provenance. Trailing dots are stripped, and a prefix naming no
module (`""`, `"."`) raises `ValueError` rather than registering a rule that can match nothing.

`docs/awkward/design.rst` states the form's cost: one zero-row parquet round trip per source,
measured at ~1.1 ms median over seven runs.

Closing tests in `tests/extra/frontend/m60/` (toy-backed, awkward-free), with `m60x_lib` /
`m60x_libx` as the stand-in wrapping library and its sibling — a prefix no frozen leg registers.
Both bare names join the existing mypy unresolvable-helper override.

### Branch evidence

Every new/changed branch, one mutant each, in a scratch copy of `python/graphed`
(`<scratchpad>/m60/mutate2.py`, `PYTHONDONTWRITEBYTECODE=1`, the working tree's `python/` replaced
in pytest's `pythonpath`). `CONTROL(unmutated)` runs green first and a `node_count` mutant is the
live-instrument control.

| mutant | killed by |
|---|---|
| CONTROL-instrument-live | `test_one_bound_method_asked_for_again_is_one_node…` |
| O-memo-keyed-on-id | `test_one_bound_method_asked_for_again_is_one_node…` |
| O-unhashable-unhandled | `test_an_unhashable_callable_is_memoed_on_its_identity` |
| O-memo-ignored | `test_one_bound_method_asked_for_again_is_one_node…` |
| O-declared-name-not-taken | `test_a_derived_name_never_takes_a_declared_ones_node` |
| O-advance-once | `test_a_third_callable_of_a_declared_name_advances_to_the_next_free_ordinal` |
| O-never-advance | `test_a_third_callable_of_a_declared_name_advances_to_the_next_free_ordinal` |
| O-name-default-changed | `test_a_callable_without_a_name_records_the_lambda_literal` |
| V-no-rstrip | `test_a_trailing_dot_spelling_is_the_same_declaration` |
| V-no-empty-refusal | `test_a_prefix_naming_no_module_is_refused[]` |
| O-name-ignored (frozen) | `test_an_explicit_name_is_the_callers_identity_declaration` |

The two advance mutants were re-run under `-k` restricted to the ordinal legs, so the kill is
theirs and not an earlier test's.

## Iteration 3 — the two rejected lines of 8418070

**The identity rule.** `_fn_name`'s memo was keyed on the callable itself, so lookup ran `==`/`hash`
and two DIFFERENT callables whose class declares them equal collapsed to one node — the second got
the first's result (`got=(4.0, 4.0)`, truth `(4.0, 200.0)`). Stated once at the point of the
operation: a callable carrying a non-None `__self__` is identified by that owner's IDENTITY plus its
function (`__func__`, or `__name__` where there is none — builtin methods and method-wrappers);
every other callable by its OWN identity; never by `==`/`hash`. The memo entry still holds `fn`
(which holds its owner), so no id in a key is recycled. The `hash(fn)` try/except is gone and an
unhashable CALLABLE is now just "its own identity" — but an unhashable OWNER still reached a hash,
and the `__name__` fallback still merged; iteration 4 narrows the key and only then is it true that
nothing in a key is hashed but ints.
Written as `if` statements, not an `or`, so coverage certifies each arm.

**`register_internal`.** The `if not prefix` guard still registered prefixes no module name can
equal (`" "`, `" mylib"`, `"mylib "`, `"my lib"`, `"-"`, `"123"`). The class is "every dotted
component is an identifier": after `rstrip(".")`, refuse unless
`all(part.isidentifier() for part in prefix.split("."))`. This REPLACES the emptiness test —
`"".isidentifier()` is False, so `""`, `"."` and `"a..b"` fall out of the same check.

Exit-round items done: the `~1.1 ms` figure is out of `docs/awkward/design.rst` (the structural
claim — one zero-row round trip per source, never per partition — stays); `Array.map`, `apply`,
`apply_gufunc` and `_fn_name` now say a declared `name=` IS the identity, so declaring a name
another callable already wears means the same node, in either order.

### Branch evidence

Same harness as iteration 2: scratch copy of the tree, `PYTHONDONTWRITEBYTECODE=1`, pytest run with
its rootdir in the copy so the copy's `python/` is what `pythonpath` seats. Live-instrument control:
a test asserting `graphed.session.__file__` is under the copy passes there (it names the path it
found), and the unmutated control leg is green before each mutant.

| mutant | killed by |
|---|---|
| O-self-arm-flipped (`owner is None` → `is not None`) | `test_a_method_wrappers_owner_and_its_name_are_both_the_identity`, both `test_two_callables_their_class_calls_equal_are_still_two_nodes` params, `test_one_bound_method_asked_for_again…`, `test_an_unhashable_callable…` (+5 more) |
| O-func-fallback-removed (`__func__`→`__name__` leg deleted) | `test_a_method_wrappers_owner_and_its_name_are_both_the_identity` |
| V-identifier-test-reverted (`if not prefix`) | `test_a_prefix_naming_no_module_is_refused[ ]`, `[ m60x_lib]`, `[m60x_lib ]`, `[my lib]`, `[123]`, `[a..b]` |

Admitted ends kept green in the same files: a bound method / method-wrapper re-accessed is ONE node
(`c.scale`, `k.__mul__`), `x.map(k.__add__)` is a second on the same owner, `other.__mul__` a third;
`"m60x_lib."` still registers, and `"m60x_lib.sub"` / `"m60x_under_score_lib"` register without
error (the test restores `_SKIP`, registration being process-global with no inverse).

## Iteration 4 — the identity key, narrowed to Python's own definition

**THE KEY.** Two callable objects share a memo entry only where PYTHON ITSELF defines them as the
same call — a genuine `types.MethodType`, whose call IS `__func__(__self__, …)`, keyed
`(id(fn.__self__), id(fn.__func__))`; every other callable is its own identity, `id(fn)`. No
`__name__`, no `==`, no `hash` of anything but ints; the entry still holds `fn`, and through it the
owner and the function, so no id recycles. Written as `if isinstance(...)` / `else` statements.

Why this narrow — each widening past Python's definition admitted a neighbour: `id(fn)` alone minted
a node per bound-method access; `fn` itself (`==`/`hash`) merged two callables their class calls
equal; `(id(owner), __func__ or __name__)` for ANY `__self__` carrier merged every
`functools.partialmethod` on one owner (each access is a `partial` with a copied `__self__`, no
`__func__` and no `__name__`, so both fell into one `(id(owner), "lambda")` bucket: node ids
`1 1 SHARED`, materialized `4.0 4.0`, truth `4.0 200.0`).

Accepted ceiling: an exotic re-accessed callable that is not a `MethodType` — builtin methods,
method-wrappers, partials/partialmethods — mints a node per access. Losing CSE is never a WRONG
answer; merging two distinct calls is. The HEP case is unaffected: `correctionlib`'s pybind11
`evaluate` IS a `types.MethodType` (lead-measured), so it keeps its CSE when the correction object
is held (`cset["sf"]` mints a fresh owner per access: `cset["sf"].evaluate` twice is two nodes,
both correct).

`test_a_method_wrappers_owner_and_its_name_are_both_the_identity` pinned the abolished owner+name
rule and is replaced by that ceiling stated as behaviour: `k.__mul__` recorded twice, no count
pinned, every recorded node materializing to `k * DATUM` (both wrapper objects held in variables so
no id can recycle mid-test).

### Branch evidence

Same harness: `rsync` copy of the tree into the scratchpad, `PYTHONDONTWRITEBYTECODE=1`, pytest run
from the copy with `-o pythonpath=<copy>/python …` so the copy's `python/` is what imports.
Live-instrument control: a test printing and asserting `graphed.session.__file__` under the copy
(`SEATED: …/scratchpad/copy/python/graphed/session.py`). Legs run
`tests/extra/frontend/m60/test_m60_callable_identity_repairs.py` + `tests/frozen/frontend/m60`;
`CONTROL-unmutated` is green (rc=0) first.

| mutant | killed by |
|---|---|
| K-no-methodtype-arm (`key = id(fn)` always) | `test_one_bound_method_asked_for_again_is_one_node_and_applies_intern_with_it`, `test_a_loop_recording_one_bound_method_stays_two_nodes`, `test_a_bound_method_of_an_unhashable_owner_is_one_node`, `test_two_method_objects_fabricated_from_one_pair_are_one_node` |
| K-owner-object-and-func (`key = (owner, func)`) | `test_a_bound_method_of_an_unhashable_owner_is_one_node` (`TypeError: unhashable type: 'UnhashableOwner'`), `test_bound_methods_of_two_equal_owners_are_two_nodes`, `test_two_partialmethods_on_one_owner_are_two_nodes` |
| K-owner-id-and-func-or-name (iteration 3's key) | `test_two_partialmethods_on_one_owner_are_two_nodes` (`assert 1 != 1`) |
| K-func-dropped (`key = id(fn.__self__)`) | `test_two_methods_of_one_owner_are_two_nodes` |

Members at each end, both ends green unmutated. MERGED: a bound method re-accessed (+ `apply`
interning with `map`, + the 6-iteration loop), a bound method of a `__hash__ = None` owner,
`types.MethodType(func, obj)` built twice from one pair. DISTINCT, each with its OWN materialized
value: `c.scale`/`d.scale`, `c.scale`/`c.offset` (one owner, two `__func__`), two `==`-equal
equal-hash OWNERS (`Calib`, behaviour field `compare=False`), the two partialmethods (`4.0`/`200.0`),
the two `==`-equal callable INSTANCES (`Weight`/`Trigger`), and `c.scale` vs the plain `scale`.

## Iteration 5 — the predicate is the rule's, not the callable's

Reviewer (unit round on f5ad10b): `isinstance(fn, types.MethodType)` consults `fn.__class__`, which
a transparent proxy forwards; two proxies over one held method reached the method arm with one
`(id(__self__), id(__func__))` key and two different calls — node ids `1 1`, materialized
`4.0 4.0`, truth `4.0 200.0`. The rule was already "Python's own definition"; the predicate was
wider than it. Now `type(fn) is types.MethodType` (`types.MethodType` is not an acceptable base
type, so the narrowing excludes nothing genuine). Closing test
`test_a_callable_that_only_claims_to_be_a_method_is_its_own_identity`: fails on the `isinstance`
predicate with `assert 1 != 1`, passes on `type(fn) is`; the merged end stays
`test_two_method_objects_fabricated_from_one_pair_are_one_node`.

## Iteration 6 — whole-artifact pass: one false universal in the docs

Reviewer (delta on 1e4354a + whole artifact over `freeze-m60..HEAD`): the identity key is closed —
no member whose type IS `types.MethodType` could be built that merges wrongly (`__self__`/`__func__`
are read-only, the type is not subclassable, `__class__` is not assignable); 23 mutants, 22 killed,
1 equivalent. One design finding: the frontend design page called `graphed.expand` "the one every
built-in verb uses" — tuple-returning and metadata verbs map differently, and `expand` over a
tuple-returning verb answers ONE `Varied` of tuples. The sentence now scopes `expand` to a verb
answering one array (what `graphed.apply` itself calls) and names the tuple case as outside it; the
private `expand_tuple` is not advertised. Two docstring wordings tightened (`register_internal`'s
raises clause, `_fn_name`'s ceiling). The reviewer's proposed test is not added: it would pin the
answer shape of a misuse, and the frozen `test_expand_stays_outside_the_array_consuming_verb_surface`
already pins that `expand` is not the whole surface.

## Iteration 7 — the m25 dispute, resolved by the owner

The owner affirmed the re-freeze (2026-09-19). `test_histogram_terminal_bundle_reproduces_bit_for_bit`
now takes `session` and `value` from ONE `_record()` call — the dispute's correction verbatim, three
lines, nothing else under `tests/frozen/**` touched. Tag `freeze-m25-fixup`.

