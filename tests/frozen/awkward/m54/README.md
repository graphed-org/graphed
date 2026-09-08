# awkward/m54 — behavior METHODS with arguments (traceability)

Milestone m54. Authority: `behavior-methods-plan.md` §2, items 1–7. The headline: `a.deltaR(b)` and
`a.scaled(2.0, offset=1.0)` record a fusible `method` op instead of raising
`TypeError: 'Array' object is not callable`, and the value equals the eager awkward call.

Run: `python -m pytest tests/frozen/awkward/m54 -q` (its own process, per the awkward per-milestone
split). The `m54_` helper prefix is load-bearing under prepend import mode. Every m54-new surface
(`graphed.BoundMethod`, a method CALL) is reached only inside test bodies, so the tree COLLECTS
against a tree with no m54 implementation and fails at RUN time.

## Fixture — `m54_behavior_fixtures.py`

`JetArray` subclasses vector's `MomentumArray4D` and is registered under `("*", "Jet")` in a COPY of
vector's behavior dict, never in global `ak.behavior`; vector's `Momentum4D` UFUNC overloads are
re-keyed onto the `Jet` name in that same copy, so `a + a` also resolves through the backend alone.
Vector's own `deltaR`/`rotateZ` cover the globally registered half.

Its attributes span every branch of the classification rule and every argument shape the contract
distinguishes:

| attribute | what it pins |
|---|---|
| `scaled(k, *, offset, gain)` | a positional constant and TWO keyword constants, so keyword ORDER is testable |
| `near(other, *, threshold)` | an array passable positionally or by keyword, beside a constant |
| `blend(*, x, y)` | two ARRAY keywords, asymmetric, so swapping them is a different call |
| `mixed(parts)` | a dict CONSTANT holding arrays, asymmetric in its keys |
| `combo(coeffs)` | a sequence constant read by index — a list and a tuple are one call |
| `weighted(weights)` | a str-keyed dict — the accepted collection the refusals are contrasted with |
| `split()` | a TUPLE of arrays |
| `count_scalar()` | a Python scalar result |
| `combine(x, y)` | a `staticmethod` — callable, so a method |
| `spread(x, y)` | a `classmethod` — not callable, a descriptor, so a method |
| `doubled()` | a `functools.partialmethod` — not callable, a descriptor, so a method |
| `widen(k)` | a `functools.singledispatchmethod` — not callable, a descriptor, so a method |
| `heavy` | a backend-only `property` |
| `bulk` | a `functools.cached_property` — a closed-set property, read like a field |
| `calibrated`, `calibrate(k)` | a property and a method reading `self.attrs["calib"]` — a re-wrap that drops `attrs` answers half |
| `MUON_MASS` | a bare class constant — not a descriptor, so it records a `field` holding the value |

`EVENTS` is three collections from `numpy.default_rng` at fixed seeds, all named `Jet`; `Shadow`
carries a real leaf named `scaled`, shadowing the method. `jets_source` builds a session whose SOURCE
record is itself Jet-named, the shape in which a method applies directly to the source.
`touched_columns` is the projection oracle: an EAGER reporting typetracer over a source form,
independent of graphed's own projection. `read_columns` answers TOP-LEVEL source fields, not leaves,
so `top_level` maps the oracle's dotted leaves onto that granularity and the read-list gate is prefix
containment; leaf-level narrowing is asserted against `project`, which does answer in leaves.

| Contract (plan §2) | Test |
|---|---|
| 1 a method is a `BoundMethod`, a property an `Array` | `test_method_attribute.py::test_a_method_is_a_bound_method_and_a_property_is_an_array` |
| 1 the PROPERTY side is the closed set; staticmethod, classmethod, `partialmethod` and `singledispatchmethod` are all methods, a bare class constant records an `Array` | `test_method_attribute.py::test_the_property_side_is_the_closed_set` |
| 1 every descriptor kind of method calls through and equals the eager call | `test_method_attribute.py::test_every_descriptor_kind_of_method_calls_through_the_recorder` |
| 1 the repr names the method and the receiver | `test_method_attribute.py::test_a_bound_method_names_itself_and_its_receiver` |
| 1 not an operand: no `node_id`, not subscriptable, `+ 1` is a `TypeError` | `test_method_attribute.py::test_a_bound_method_is_not_an_operand` |
| 1 fields resolve FIRST (regression control) | `test_method_attribute.py::test_a_field_named_like_a_method_is_still_a_field` |
| 1 the numpy backend is untouched (regression control) | `test_method_attribute.py::test_the_numpy_backend_attribute_path_is_unchanged` |
| 2 positional constant, keyword constant | `test_method_call.py::test_a_positional_constant_evaluates_as_the_eager_call`, `::test_a_keyword_constant_evaluates_as_the_eager_call` |
| 2 an array argument, positionally and by KEYWORD | `test_method_call.py::test_an_array_argument_evaluates_the_same_positionally_and_by_keyword` |
| 2 an array and a constant in one call | `test_method_call.py::test_an_array_and_a_constant_mix_in_one_call` |
| 2 numpy scalars and 0-d arrays coerce through `.item()` | `test_method_call.py::test_a_numpy_scalar_constant_is_coerced_and_matches_the_python_scalar`, `test_method_refusals.py::test_a_zero_dimensional_numpy_array_is_coerced_not_refused` |
| 2 provenance points at the call line | `test_method_call.py::test_the_call_site_is_the_recorded_provenance` |
| 2 eleven refusals (ndarray, eager array, callable, NaN, inf, set, non-str dict keys, and the `{"$": …}` marker shape at top level, nested in a list, nested in a dict and under a keyword), each naming the method and `args[i]`/`kwargs['k']`, recording nothing | `test_method_refusals.py::test_a_non_json_argument_is_refused_before_any_node_is_recorded` |
| 2 a dict is refused only when the marker IS its whole shape | `test_method_refusals.py::test_a_dict_is_refused_only_when_the_marker_is_its_whole_shape` |
| 2 an `Array` of another Session is refused as a method argument | `test_method_refusals.py::test_an_array_from_another_session_is_refused_as_a_method_argument` |
| 2 the same guard fires for a binary op, since it lives in `Session.record_op` | `test_method_refusals.py::test_an_array_from_another_session_is_refused_by_a_binary_op` |
| 2 str-keyed dicts and an explicit `None` ARE constants | `test_method_refusals.py::test_the_json_representable_constants_are_accepted` |
| 3 equal calls intern to one node | `test_method_nodes.py::test_the_same_call_twice_is_one_node` |
| 3 unequal constants or argument arrays are distinct nodes; equality is JSON's, so `2` and `2.0` differ | `test_method_nodes.py::test_a_different_constant_or_argument_array_is_a_different_node` |
| 3 keyword spelling order is not identity | `test_method_nodes.py::test_the_keyword_spelling_order_does_not_change_the_node` |
| 3 two array keywords follow the sorted NAME order, not the spelling | `test_method_nodes.py::test_two_array_keywords_are_ordered_by_name_not_by_spelling` |
| 3 a list and a tuple of the same constants are one node; a REORDERED list is not | `test_method_nodes.py::test_a_list_and_a_tuple_of_the_same_constants_are_one_node` |
| 3 arrays nested in a dict constant are joined in sorted-key order | `test_method_nodes.py::test_arrays_nested_in_a_dict_constant_follow_the_sorted_key_order` |
| 3 two independent builds compile byte-identically | `test_method_nodes.py::test_two_independent_builds_compile_to_identical_ir` |
| 4 a tuple result is a tuple of `Array`s, one `method` node each | `test_method_nodes.py::test_a_tuple_result_is_a_tuple_of_arrays_one_node_each` |
| 4 a scalar result is refused, recording nothing | `test_method_nodes.py::test_a_scalar_returning_method_is_refused_and_records_nothing` |
| 5 a `Varied`'s method attribute is a `BoundMethod` | `test_method_varied.py::test_a_varied_receivers_method_is_a_bound_method` |
| 5 a `Varied` RECEIVER fans over the labels, per-universe values | `test_method_varied.py::test_a_varied_receiver_fans_the_call_over_its_labels`, `::test_a_constant_only_call_on_a_varied_receiver_varies_too` |
| 5 a `Varied` ARGUMENT fans likewise | `test_method_varied.py::test_a_varied_argument_fans_the_call_over_its_labels` |
| 5 a tuple over a `Varied` is a tuple of `Varied` | `test_method_varied.py::test_a_tuple_result_over_a_varied_receiver_is_a_tuple_of_varied` |
| 6 projection reports exactly the leaves the method reads | `test_method_projection.py::test_projection_reports_exactly_the_leaves_the_method_reads` |
| 6 a `method` op is fusible, not a boundary | `test_method_projection.py::test_a_method_does_not_end_a_stage` |
| 6 the behavior re-wrap carries `attrs`, for a property and a method, in both graph shapes | `test_method_projection.py::test_the_behavior_re_wrap_carries_the_arrays_attrs` |
| 6 a backend-only behavior resolves properties, methods AND operator overloads | `test_method_projection.py::test_a_backend_only_behavior_resolves_like_a_globally_registered_one` |
| 6 read list contains every touched leaf's top-level field, field-rooted, and projection narrows the `Jet.*` leaves exactly | `test_method_projection.py::test_the_read_list_never_under_reads_for_a_field_rooted_method` |
| 6 the same containment for a method ON the source record (`None` stays legal) | `test_method_projection.py::test_the_read_list_never_under_reads_for_a_method_on_the_source_record` |
| 7 an unknown keyword carries the call-site line | `test_method_refusals.py::test_an_unknown_keyword_is_refused_at_the_call_site` |
| 7 an argument the method rejects carries the call-site line | `test_method_refusals.py::test_an_argument_the_method_rejects_is_refused_at_the_call_site` |

The tuple test does not assert the `index` param's spelling — the contract is one node per element
with the element's value, which the node ids and the materialised arrays pin without naming a param.

## Non-vacuity (measured against `main` at 07d5172, no m54 implementation)

49 of 51 tests fail at RUN time; the failing set is identical across two runs. The reasons:

| reason | tests |
|---|---|
| `TypeError: 'Array' object is not callable` | 38 |
| `TypeError: 'Varied' object is not callable` | 3 |
| `AttributeError: module 'graphed' has no attribute 'BoundMethod'` | 3 |
| `Failed: DID NOT RAISE AttributeError` (today `a.scaled` records a node and has a `node_id`) | 1 |
| `Failed: DID NOT RAISE GraphedTypeError` (today a cross-Session binary op records) | 1 |
| `assert 'scaled' in 'Array(node_id=5)'` (today the attribute is an `Array`) | 1 |
| `GraphedTypeError: ill-typed op 'field'` (a bare source read carries no behavior today) | 2 |

The two that PASS are the live-harness positive controls named above — the shadowed field and the
numpy attribute path resolve the same before and after m54, so a test that failed on them would be
pinning the wrong thing. The projection oracle is instrumented for the same reason: an expression
that provably reads `Jet.mass` reports it through the identical helper, so `"Jet.mass" not in
columns` is a real absence and not a dead instrument. The source-record read-list test asserts its
own oracle is non-empty before the superset, and the `attrs` test reads the same source through a
behavior-free backend first, which equals eager today — so its value assertions are live.

## Two notes for the implementer

- **Backend-only behaviors already resolve properties today.** Plan §1 says a behavior registered on
  the backend alone "resolves neither properties nor methods". Measured on this tree, `JetArray`'s
  `heavy`, `bulk` and the re-keyed `Jet` ufunc overloads all resolve and materialize while
  `("*", "Jet")` and `"Jet"` are absent from `ak.behavior`. Only the method CALL is missing. A class
  deriving from a globally registered vector class may be why the two probes differ.
- **The projection test uses a `with_name`'d record, not a zipped one.** The tested chain is
  `source → field → with_name → firsts → method`, which the property path narrows today.
