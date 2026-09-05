# m52 frozen suite — `graphed` / `tests/frozen/frontend/m52`

Milestone m52 (nuisance POINTS: named points R2 + executable joint universes R1-c). Authority:
`systematics-design/nuisance-points-design.md` (FINAL) as grounded by `m52-decomposition.md` §3.1;
this tree carries C1 (`graphed/_points.py`), C2 (the Session label→point registry, the `points=`
keyword and `graphed.points()`) and C4 (coordinate reachability, and `variations()`' third kind).

**Import ceiling.** `ci.yml`'s REQUIRED `test-freethreaded` job collects `tests/frozen/frontend`
WHOLE under `pip install pytest hypothesis numpy` alone, so nothing reachable from this tree imports
`graphed.awkward`, `awkward`, `hist`, `boost_histogram`, `pyarrow` or `pandas` — at module level or
in a body. Event contexts are built directly, the `frontend/m49` idiom. C3's projection anchors need
awkward and live in `tests/frozen/awkward/m52`; C5's fill anchors live in `graphed-histogram`.

**Collection.** The whole `tests/frozen/frontend` tree is one pytest process under prepend import
mode, so the helper carries the `m52_` prefix and every basename here is unique against its
siblings.

**Freeze rule.** `tests/frozen/**` is read-only after the freeze tag: never edited, deleted,
`skip`ped, `xfail`ed or weakened, and `git diff m52-freeze -- tests/frozen/` must stay empty for the
life of the milestone. A frozen test that looks wrong is a Test Dispute at
`.graphed/m52/disputes/<test_id>.md`, never a repair in place.

Every m52-new symbol (`graphed._points` through `m52_point_fixtures.point_api`, `graphed.points`,
the `points=` keyword) is reached in a test BODY, never a module-level import, so the suite COLLECTS
against `origin/main` and fails at RUN time for the RIGHT reason.

| anchor | design § | test |
|---|---|---|
| `vary-m52-C1` | §4.2 numeric spellings, one value space | `test_point_value.py::test_numeric_spellings_of_one_coordinate_are_one_point` |
| `vary-m52-C1` | §4.2 exact decimal decomposition, never IEEE | `test_point_value.py::test_a_float_decomposes_exactly_and_not_through_ieee` |
| `vary-m52-C1` | §4.2 no finite decimal expansion, refused naming the value | `test_point_value.py::test_a_fraction_with_no_finite_decimal_is_refused_naming_the_value` |
| `vary-m52-C1` | §4.2 identifier tags are NOT promoted to numbers | `test_point_value.py::test_identifier_coordinates_are_not_promoted_to_numbers` |
| `vary-m52-C1` | §4.2 name-sorted pair tuple, order-independent | `test_point_value.py::test_a_point_is_name_sorted_and_order_independent` |
| `vary-m52-C1` | §4.2 explicit-map zero-drop; the origin | `test_point_value.py::test_an_explicit_map_zero_drops_and_an_all_zero_map_is_the_origin` |
| `vary-m52-C1` | §4.2 the zero ASYMMETRY — a default point is never zero-dropped | `test_point_value.py::test_a_default_point_is_never_zero_dropped` |
| `vary-m52-C1` | §4.2/§4.6 `restrict(point, axes)` | `test_point_value.py::test_restrict_drops_only_the_coordinates_off_the_axes` |
| `vary-m52-C1` | §4.10 rendering is BY VALUE | `test_point_value.py::test_rendering_is_by_value_so_two_equal_points_render_alike` |
| `vary-m52-C1` | §4.2 a nuisance name is a Python identifier (+ positive control) | `test_point_value.py::test_a_nuisance_name_must_be_a_python_identifier`, `::test_an_identifier_nuisance_name_is_accepted` |
| `vary-m52-C2` | §4.5 a point is minted for EVERY label — six existing shapes, literal maps | `test_point_registry.py::test_every_existing_shape_reports_its_default_points` |
| `vary-m52-C2` | §8-g the ambient weight's axes include the inherited shift family | `test_point_registry.py::test_the_ambient_weights_axes_include_the_inherited_shift_family` |
| `vary-m52-C2` | §4.4 `points=` on the loose / weight / shift overloads | `test_point_registry.py::test_points_is_accepted_on_the_loose_overload`, `::test_points_is_accepted_on_the_weight_overload`, `::test_points_is_accepted_on_the_shift_overload` |
| `vary-m52-C2` | §4.4 `points` leaves BOTH keyword namespaces | `test_point_registry.py::test_a_variation_tagged_points_still_registers_through_variations`, `::test_a_collection_named_points_still_registers_through_collections` |
| `vary-m52-C2` | §4.11-1 one label, one point (+ the idempotent re-mint) | `test_mint_refusals.py::test_one_label_under_two_points_is_refused_naming_both` |
| `vary-m52-C2` | §4.11-2 one point, one label | `test_mint_refusals.py::test_one_point_under_two_labels_is_refused_naming_both` |
| `vary-m52-C2` | §4.11-5 a `points=` key that is not a tag of the call, matched AFTER `canonical_tag` | `test_mint_refusals.py::test_a_points_key_that_is_not_a_tag_of_this_call_is_refused` |
| `vary-m52-C2` | §4.11-3 the origin entry, beside the zero TAG that still mints | `test_mint_refusals.py::test_an_origin_points_entry_is_refused_while_a_zero_tag_still_mints` |
| `vary-m52-C2` | §4.5 the mint is TRANSACTIONAL | `test_mint_refusals.py::test_a_failed_vary_leaves_the_registry_untouched_and_the_label_reregistrable` |
| `vary-m52-C2` | §4.10 `points()` refuses a result mapping | `test_points_introspection.py::test_points_refuses_a_result_mapping` |
| `vary-m52-C2` | §4.10/§5.3 sorted, and stable across `PYTHONHASHSEED` | `test_points_introspection.py::test_points_is_sorted_and_stable_across_hash_seeds` |
| `vary-m52-C4` | §4.4 R2 verbatim against numerically tagged families | `test_coordinate_reachability.py::test_r2_verbatim_against_numerically_tagged_families` |
| `vary-m52-C4` | §4.11-4/§8-i a typed coordinate naming no registered tag, naming what is | `test_coordinate_reachability.py::test_a_typed_coordinate_naming_no_registered_tag_is_refused_naming_what_is` |
| `vary-m52-C4` | §4.11-4 a nuisance registered nowhere | `test_coordinate_reachability.py::test_a_nuisance_registered_nowhere_is_refused` |
| `vary-m52-C4` | §4.11-4 a joint point registered BEFORE its axis exists | `test_coordinate_reachability.py::test_a_joint_point_registered_before_its_axis_exists_is_refused` |
| `vary-m52-C4` | §4.11-4 the carrier walk: ambient weight / `Varied` collections / selection | `test_coordinate_reachability.py::test_the_carrier_walk_covers_all_three_context_carriers`, `::test_a_nuisance_on_none_of_the_three_carriers_is_refused`, `::test_the_loose_forms_carrier_is_the_targets_own_tag_map` |
| `vary-m52-C4` | §4.7/§8-b an inherited label with partial coverage still falls back silently | `test_coordinate_reachability.py::test_an_inherited_label_with_partial_coverage_still_falls_back_silently` |
| `vary-m52-C4` | §4.10 `variations()`' third kind, and the m50 anchor guard | `test_variations_kind.py::test_a_family_registered_as_both_shift_and_weight_reports_both`, `::test_the_dual_kind_keeps_the_frozen_value_half`, `::test_disjoint_families_report_exactly_the_two_m50_kinds` |

## Spellings this freeze pins

* **`graphed._points`** exposes four names, and the tests touch no other: `Point(mapping)` —
  EXPLICIT construction, which zero-drops every coordinate whose `numeric_value` is 0;
  `default(name, tag)` — the default point, never zero-dropped; `restrict(point, axes)` where
  `axes` is a set of nuisance names; `render(point) -> dict[str, str]`, by VALUE. **The origin is
  spelled `Point({})`** — no separate constant is frozen, so one may be added freely.
* **`graphed.points(obj) -> dict[str, dict[str, str]]`** — top-level export, label-sorted, each
  inner map nuisance-sorted, `"nominal"` mapping to `{}`. Defined on a `Varied` and an event
  context; raises on a result mapping.
* **`points=`** — keyword-only on all three `vary` overloads, `{tag: {nuisance: coordinate}}`, keyed
  by tag exactly like `variations=`, and out of `**tags` so a variation or a collection genuinely
  named `points` still registers.
* **`variations()`' third kind word is `"both"`**, per (name, tag), with the frozen
  `{name: {tag: (kind, value | None)}}` shape and `numeric_value(tag)` still in the value half.
* Refusals raise `graphed.errors.GraphedError`. Message content is asserted only where the design
  says the message must NAME something (§4.11-1/2/4/5, §4.2's exact fraction).

## Fixture families (`m52_point_fixtures.py`)

* **The six existing shapes** `EXISTING_SHAPES` / `DEFAULT_POINT_MAPS` — loose, loose family
  extension, weight, shift, lockstep shift, stacked weight families. The literal point map of each
  is the fixture's, so the test asserts a map rather than re-deriving the rule under test.
* **`shift_then_weight_context`** — §8-g: the ambient weight CARRIES `jes_up` while its `_tags` has
  no `jes` key. The one shape that separates a registry-derived axis set from a `_tags`-derived one.
* **`two_axis_loose` / `two_axis_context`** — `jes` and `jer` registered, the smallest carrier of a
  legal two-coordinate point. Every passing `points=` map in this tree has at least two
  coordinates, because a one-coordinate explicit point is unregistrable by construction: §4.11-4
  requires its coordinate to be a registered tag of its nuisance, so the default label for that
  (nuisance, tag) already owns the point and §4.11-2 refuses a second label for it.
* **`numeric_families`** — `jes` (shift), `btag` and `jer` (weights), all tagged `1` / `-1`, each
  moving the ambient weight by a different factor. R2's literal `{"jes": 1, "btag": -1}` must reach
  these; `identifier_families` is the same programme tagged `up` / `down`, which it must not.
* **`ambient_only_context` / `collection_only_context` / `selection_only_context`** — §4.11-4's
  three carriers, each in isolation, each registering TWO families so a legal two-coordinate point
  exists whose every axis only that carrier supplies. The ambient-weight one is the discriminator:
  its weight's `_tags` is EMPTY while it carries the labels, so a `_tags`-derived walk refuses it.
* **`descendant_weight`** — the m48 row-space refusal, which `vary` raises AFTER `gather_members`
  has minted, for the transactional-mint anchor.
* **`dual_registered_family`** — one family name registered as both a shift and a weight, `up` dual
  and `down` shift-only, so a per-family verdict cannot pass.

## Non-vacuity (baseline: 43 failed / 2 passed against `origin/main`)

Every test fails against a tree with no m52 implementation for the RIGHT reason —
`ModuleNotFoundError: graphed._points`, `AttributeError: module 'graphed' has no attribute
'points'`, the `points=` map arriving as a variation member (`a variation member must be an Array
or a Varied, got dict` / `variation tag 'points' was given both as a keyword and in variations=` /
`collection 'points' was named twice`), a `DID NOT RAISE` on a refusal that does not exist yet, or
the measured wrong answer (`variations()` reporting `("shift", None)` where `("both", None)` is
required) — never a collection or import error.

The only two green tests are the backward-compatibility positive controls, and each has red
siblings in its own file: `test_an_inherited_label_with_partial_coverage_still_falls_back_silently`
(§8-b, which C4 must not turn into an error) and
`test_disjoint_families_report_exactly_the_two_m50_kinds` (the m50 anchor guard, which C4 must not
widen).
