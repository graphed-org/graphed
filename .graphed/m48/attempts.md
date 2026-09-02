# m48 (graphed arc) — implementer attempts

Frozen suite: `tests/frozen/frontend/m48/**` (28) + `tests/frozen/awkward/m48/**` (84) = 112 tests,
tag `m48-freeze` @ 37d86cf. Nothing under `tests/frozen/**` was edited, skipped, xfailed or
weakened at any point.

Environment: `.venv`; every pytest/mypy/ruff invocation is prefixed `PATH=$PWD/.venv/bin:$PATH`
(`scripts/run-tests.sh` invokes bare `python`, which otherwise dies on `--cov=graphed`).

## Iteration 1 — G-I1: vocabulary + container

Added `python/graphed/_tags.py` (§1.1 canonicaliser: exact-decimal e-form, the 32-char cap with
the two refusals split by cause), `varied.py` (`Varied`, the §2.4 label-aligned union, the
per-idiom container registry), `vary.py` (the three §2.1 overloads) and `numpy/varied.py`.

Gates: frontend m48 19/28. Rewrote `varied.py`'s method generation around a single
`expand(fn, args, kwargs)` primitive after the first shape (rebuilding args from a flattened
tuple) proved unreadable; added the pass-through when no container is among the operands, without
which an unvaried expression recorded as a one-member container.

## Iteration 2 — G-I2: event context + lineage seams

Added `context.py` (`EventContext`: lineage, the ambient weight registry, memoised pure
derivations and projections), `awkward/gnano.py` and `accessors.py` (§9.1). Rerouted all five
`self._array_cls(self, node_id)` sites in `session.py` through a new `_wrap` so the §2.3e handle
merge and the `_labels` union happen at one chokepoint (call sites before: 5, left: 0).

`compile_ir`'s unreached-label report cannot key on node reachability — F8's two containers intern
to identical node ids — which is why `_labels` is object provenance on the frontend wrapper.

## Iteration 3 — G-I3a: the gak classification table

Appended §2.3c's five-class table to `awkward/functions.py`: 65 classified / 65 discovered / 0
unclassified / 0 extra, plus `GakSlot`/`GakArgFixture` and 58 argument fixtures for the carrying
classes. `functools.wraps` on `_dispatched` keeps each verb's signature, which §2.3d's annotation
filter reads.

## Iteration 4 — G-I3b: making every carrying fixture callable

Emulating the frozen A5 gate over all 58 fixtures left two red:

- `onnx_inference` → `DecodeError: Wire format was corrupt`. `payloads.onnx_weights_descriptor`
  suppresses parse failures but `onnx_weights_hash` (called outside that suppression) does not.
  Fixed in the fixture: `b""` is a valid empty `ModelProto`, so the weights-identity path parses.
- `unflatten` → `ill-typed op 'ak.unflatten': counts must be integers`. The frozen operand
  vocabulary (`OPERAND_KINDS`, five kinds, all float/bool/record) carries no integer array, and
  five alternative slot pairings all failed. **Root cause, not the fixture**: `gak.unflatten`
  accepted only an `Array` counts, while `ak.unflatten`'s own error names the contract — "counts
  must be an integer or a one-dimensional array of integers". `gak.unflatten` now takes
  `Array | int`, an integer riding in params rather than as a graph edge, with the matching read
  in `awkward/_ops.py`. The fixture is then the ordinary `(jagged, 2, axis=1)` spelling.

Gates: awkward m48 73/84, frontend m48 19/28.

## Iteration 5 — G-I3c: the module-verb dispositions

`VERB_DISPOSITIONS` in `graphed`, `graphed.numpy` and `graphed.awkward` (12 / 6 / 2), the two
refusal contracts (`refuse_boundary` on repartition/pack_key/join/shuffle_plan/join_plan;
`refuse_container` naming `graphed.universe` on compile_ir/aggregate_plan), the `broadcasting` and
`expanding` decorators, `graphed.apply`'s expansion, and the `Varied` branches in
`Array._binary` / `__getitem__` / `filter`.

`read_columns` was not `None`-dominant across several graphs — a conservative graph beside one
that narrows returned the narrow set (measured: `read_columns([plain, opaque], nid)` gave
`('MET',)`). Restructured to answer per graph and combine with `None` dominating, which is what
its own docstring already promised.

Gates: 109/112.

## Iteration 6 — G-I4: the compile and aggregate seams

`CompiledGraph.unreached_labels` (additive field, sorted, default `()`), fed by the Session
registry minus the labels the marked outputs carry; `aggregate_plan(on_compiled=...)` firing once
on the compiled artifact with its return value carried onto `_PartitionReduce.variation_labels`.

The registry stores `(labels, weakref)` pairs rather than bare weak references: F8's third anchor
registers a container and immediately discards it, which is precisely the silent-cost case the
diagnostic exists to report, so the labels must outlive the container. The weak reference to the
container itself is §2.5's pinned registration mechanism and is unchanged.

Gates: 110/112.

## Iteration 7 — lint, types, format

`ruff check python tests` clean, `ruff format --check python` clean (78 files),
`mypy` clean (76 source files, `files=["python"]` unchanged).

Three `# type: ignore[no-any-return]` in `array.py`, each one line with its reason: the container
branches of `_binary`/`__getitem__`/`filter` answer with a `Varied`. Widening those three return
types to `Array | Varied` was tried and reverted — it rippled into `numpy/array.py` and three
`awkward/functions.py` callers for a union that only arises when the *key* is a container, so the
dynamic answer stays confined to the dispatch layer. `_is_context` became a `TypeGuard`, which
removed two `Returning Any` errors honestly rather than by ignore.

## Iteration 8 — the accessor shapes the frozen tree cannot reach

Frozen-suite-only coverage put `accessors.py` at 68%. Only PART of that gap is cross-repo. The
`{label: hist}` RESULT MAPPING and bare-histogram shapes are `graphed-histogram`'s anchors, in a
different distribution, and `tests/frozen/frontend` must stay importable under
`pytest hypothesis numpy` alone — those this repo's frozen suite genuinely cannot reach. The rest
— `labels`/`universe` on a plain `Array`, the two unknown-shape errors, `weight` on a non-context,
`reindex_to` to a context-free target, and `broadcast_like`'s body — need neither a histogram nor
awkward, and the awkward-free extra test added below covers exactly those, so a frozen anchor
could have too. Carried to the m49 frontend freeze.

Added `tests/extra/frontend/m48/test_accessor_input_shapes.py` (awkward-free, a duck-typed
histogram on the one attribute §2.2 reads) and, with `AwkwardBackend.broadcast_like` recording
`ak.broadcast_arrays`, `tests/extra/awkward/m48/test_broadcast_like_seam.py`. `run-tests.sh` picks
both up with no change: it already runs `tests/extra/<pkg>` for the split packages.

`accessors.py` 68% -> 99%; `awkward/backend.py` 98% -> 100%.

## Gate results at the dispute STOP (tip fdbc2ae, freeze `m48-freeze`)

| gate | result |
|---|---|
| m48 frozen | 110/112; the 2 red are the disputed pair |
| whole frozen suite | 7 failures = 5 pre-existing `awkward/m16` + the 2 disputed; no regressions |
| coverage (combined, `COV=1 ./scripts/run-tests.sh`) | TOTAL 93%, gate >=90 |
| coverage (frozen suite ALONE) | TOTAL 92%; every m48 file >=91% except `accessors.py` at 68% (above) |
| ruff | `check python tests` clean; `format --check` clean, 80 files |
| mypy | clean, 76 source files, `files=["python"]` unchanged |
| determinism | frozen F2 green |
| precommit `--fast` | toml-valid ok, integrity-scan ok; workflows-valid skipped (pyyaml absent); prek cannot launch here (`rustup could not choose a version of cargo-clippy`), so ruff and mypy were run directly |

## Iteration 9 — final gate pass on the re-freeze

The dispute was upheld. The test-author applied both adjudicated corrections in affd355 — link kind
(1) re-indexes `shifted.Photon.pt` (a collection the shift does not name, so the read is genuinely
unvaried), and the §6.1d(B) identity arm keeps `sel.MET.pt` and asserts the identity over HANDLES:
equal label lists plus per-label member node-id equality, which is strictly stronger than the single
`.node_id` §2.2 refuses on a container. Suite re-frozen at `m48-freeze2`.

**No implementation change was needed or made.** Every gate below ran against the source as it stood
at the STOP; the only commits since are the test-author's re-freeze and this log entry.

| gate | result | decisive output |
|---|---|---|
| m48 frozen | **112/112** | `pytest tests/frozen/awkward/m48 tests/frozen/frontend/m48` exit 0, zero FAILED |
| whole frozen suite | only the known-env reds | 5 FAILED, all `awkward/m16` `ak.var`/`ak.std` typetracer `MaybeNone` |
| coverage (combined) | green | `TOTAL 5905 314 1712 172 93%`; `coverage report --fail-under=90` printed no failure line |
| coverage (frozen suite ALONE) | `TOTAL … 92%` whole-package | every m48 file >=91% except `accessors.py` 68% (see iteration 8) |
| ruff | clean | `All checks passed!` / `84 files already formatted` |
| mypy `--strict` | clean | `Success: no issues found in 76 source files`, `files=["python"]` unchanged |
| determinism | green | frozen F2 passes inside the 112 |
| precommit `--fast` | unchanged | `toml-valid ok`, `integrity-scan ok` — but measured over a CLEAN worktree, so it scanned an empty change set; the live arc scan is in iteration 10. `workflows-valid --  pyyaml not installed`; `prek FAIL` on `rustup could not choose a version of cargo-clippy` |

Per-file coverage under the frozen suite alone: `_tags` 98, `varied` 95, `vary` 91, `numpy/varied`
100, `context` 92, `gnano` 100, `session` 96, `projection` 100, `execute` 93, `aggregate` 95,
`array` 95, `awkward/backend` 98, `accessors` 68 (99 with `tests/extra`).

## Decision: `graphed.labels(ctx)` on a ROOT context

**Answer: `("nominal",)`, not `()`.**

§2.2 defines the label list as "ordered: nominal first, then insertion order", and §2.4 makes
`"nominal"` always first in every union — a `Varied` cannot exist without it. A root context is
the union of three empty terms, and the union of nothing under a rule that always seats
`"nominal"` first is `("nominal",)`, not the empty tuple. It also keeps the verb total across its
four input shapes: `labels` of a bare unvaried histogram is `("nominal",)` by the same reasoning,
and a caller iterating `graphed.labels(ctx)` to fill per universe gets the one nominal fill rather
than silently filling nothing. The empty tuple would make "has this context any variations?" and
"which universes does it have?" the same question, which is the §2.5 silent-drop shape.

`graphed.labels` on a plain `Array` raises instead — an array carries no universes, and answering
`("nominal",)` there would let a dropped container read as a legitimate single-universe result.

## Test dispute — STOPPED on two tests (UPHELD; re-frozen at `m48-freeze2`)

`.graphed/m48/disputes/awkward-m48-test_lineage_seams.md`.

`tests/frozen/awkward/m48/test_lineage_seams.py::test_link_kind_1_a_mask_derivation_makes_an_
UNVARIED_value_varied` and `::test_reindex_to_is_the_identity_for_a_value_already_at_the_target_
or_context_free` both require a read through a shift-varied / `Varied`-mask-derived context to be
a plain `Array`. Plan §2.6b ("thereafter `events.<Collection>` is a `Varied`") and §2.6c ("its
collections READ as `Varied`") bind those reads to be containers, and §2.2's reserved-name rule
makes `.node_id` on a container an `AttributeError`. The dispute carries the measurement, with
`Photon` (a collection the shift does not name) as the positive control, and a proposed
correction for each test.

No implementation shortcut was taken to make either pass.

## Known-environment red (pre-existing, not m48)

`tests/frozen/awkward/m16` fails 5 tests in this environment (`ak.var`/`ak.std` typetracer
`MaybeNone`), independent of m48. Neither fixed nor masked.

Verified against a worktree at `m48-freeze` (37d86cf) running the same command: the two `-q --tb=line`
transcripts diff to ZERO lines once absolute paths and the one shifted `session.py` line number are
normalised — same failing ids, same progress bitmap `.........FF.............F.........................FF`,
same `GraphedTypeError: ill-typed op 'ak.std'/'ak.var' … Encountered unknown type MaybeNone`.
Regenerate with:

    git worktree add --detach <dir> m48-freeze && cp python/graphed/core/graphed_core.*.so <dir>/python/graphed/core/
    (cd <dir> && PYTHONPATH=<dir>/python python -m pytest tests/frozen/awkward/m16 -q --tb=line)

## Iteration 10 — impl-review REJECT: the six fixes and their witnesses

Three-lens review rejected the arc (1 HIGH, 5 MED, 5 LOW, 2 NIT). F4/F6 are m49 test-authoring
constraints and F12 is the stray `uv.lock`, both carried elsewhere. The rest:

**F1 (HIGH, `vary.py:_vary_loose`) — real defect.** Overload (a) re-indexed only the NEW members;
`existing` — the target as `"nominal"`, plus every inherited label — reached `rebuild` untouched,
so a container whose most-derived handle came from a new member advertised a handle its nominal
universe did not honour. Fixed with `_align`, which re-indexes an inherited member only across a
link that MOVES rows (`"mask"`/`"project"`). The blunt form — `reindex_to` over
`{**existing, **resolved}` — was measured and reds a frozen anchor, because `reindex_to` ends in a
`with_context` stamp even across a `vary` identity link.

**F2 (MED, `context.py:_vary_weight`) — real defect.** The weight form had neither sibling's
"already carried by this container" check, so a label produced under one `name` that collided with
an ambient label registered under a DIFFERENT `name` was composed as `old[L] * factor[L]`, leaving
a universe that differs from nominal in two knobs. Added the same check both siblings carry.

**F3 (MED, `context.py:_mask_key`) — guard gap, NOT a defect.** The key already covers every
member's node id (`tuple((label, member.node_id) for label, member in mask._members.items())`); no
code change was needed or made. The witness is what was missing.

**F5 (MED, `accessors.py:with_context`) — guard gap, NOT a defect.** `stamped._labels =
value._labels` is present and correct; verified by mutation, not by reading. No code change.

**F7 (LOW, `projection.py`)** — the walk's external callback never set `conservative`, so a graph
whose External consumed the source RECORD while separately reading a field narrowed to that field.
The callback now mirrors `on_op`'s non-field branch; `reads_source` was lifted out of `on_op` so
both use one predicate.

**F10 (LOW, `accessors.py:labels`)** — the Mapping branch seated `"nominal"` unconditionally,
claiming a label `universe` refuses on the same argument. Seated only when present.

**F13 (NIT, `context.py`)** — dropped the `provenance=` parameter (no caller anywhere) and the
`__weakref__` slot (nothing weak-references a context; §2.5's registry weakrefs `Varied`).

**F11 (LOW, `pyproject.toml`)** — the pythonpath comment claimed "nothing performs an explicit
`import conftest`"; three corpus/m05 tests do. Rewritten to the real guarantee: separate processes
per subtree plus prepend mode seating the test's own dir first.

### Witnesses, each shown discriminating

Mutate the named line, run the witness (`tests/extra/{awkward,frontend}/m48`):

| finding | mutation | witness |
|---|---|---|
| F1 | drop the `_align` map over `existing` | RED |
| F2 | drop the duplicate-label check | RED |
| F3 | `_mask_key` -> nominal's id alone | RED |
| F5 | drop `stamped._labels = value._labels` | RED |
| F7 | `external=lambda *_a: None` | RED |
| F10 | seat `"nominal"` unconditionally | RED |

All six green unmutated. F1, F2 and F7 also carry a positive control (the `vary`-identity link
that must NOT be re-indexed; the same-`name` family check and a legal fresh label; an External over
FIELDS that must still narrow), so none of the three fixes can pass by refusing everything.

### F9 — the live integrity scan

The `integrity-scan ok` in both earlier tables was measured on a clean worktree, i.e. over an empty
change set. Live scan over the arc, `git diff origin/main...HEAD | scan_diff`, measured on the tree
this iteration ships (see iteration 11 for why the figures below moved once):

    26 findings: frozen_test_modified 24, skip_or_xfail_added 1, type_ignore_flood 1

No `assertion_removed`, `tautological_assert`, `target_stubbed` or `except_pass_flood`. The 24
`frozen_test_modified` rows are the arc CREATING the frozen suite (`check_integrity`'s
new-vs-modified split downgrades brand-new frozen files to advisory).

Both remaining rows point at THIS FILE and are the scanner reading its own record:

- `skip_or_xfail_added` — detail is the positive-control sentence below, which names the marker
  the control plants. The scanner matches the added line, not its role, so the log's description
  of the control reads as the thing itself. Nothing under `tests/frozen/**` carries the marker.
- `type_ignore_flood` — the threshold-3 heuristic over 15 added `# type: ignore` lines: 7 in
  `python/` (4 `array.py`, 2 `varied.py`, 1 `awkward/backend.py`), 6 in the test-author's frozen
  files, 2 quoted inside this log. None is the blanket form §A.7 bans.

Neither is reworded to make the scanner quiet; both are recorded as the instrument reports them.

Positive control, same invocation with a planted `-assert result == 3` / `+pass` and a
`@pytest.mark.skip` appended to the diff: `assertion_removed` and `skip_or_xfail_added` both fire.
The clean result is therefore a measurement, not a dead instrument.

### F8 — the diff-coverage figure

Frozen-suite-alone DIFF coverage over the arc's added `python/` lines: **94.25% line+branch**
(1869/1983 — 795 executable added lines, 732 covered; 1188 branch arcs departing them, 1137
covered), or 92.08% line-only (732/795). Whole-package frozen-alone is 92%.

This does NOT reproduce the review's reported 90.24%; the method behind that number is not stated,
so both are recorded rather than one transcribed. Regenerate mine with:

    bash <scratch>/frozen_only.sh          # per-subtree frozen runs into .coverage.frozenonly
    git diff origin/main...HEAD            # added-line population, python/ only

### Gates after the fixes

| gate | result | decisive output |
|---|---|---|
| m48 frozen | 112/112 | zero FAILED |
| extra m48 witnesses | 21/21 | `tests/extra/{frontend,awkward}/m48` |
| whole frozen suite | only known-env reds | 5 FAILED, all `awkward/m16` `MaybeNone` |
| coverage (combined) | `TOTAL 5922 313 1722 171 93%` | gate >=90, no failure line |
| coverage (frozen alone) | whole-package 92%; diff 94.25% | above |
| ruff / format | clean | `All checks passed!` / `86 files already formatted` |
| mypy `--strict` | clean | `Success: no issues found in 76 source files` |
| determinism | green | frozen F2 inside the 112 |
| precommit `--fast` | unchanged | `prek FAIL` on `cargo-clippy`; integrity now measured live, above |

## Iteration 11 — delta re-review of 38eebf6: A-1, A-2, A-3, C2, A-4

Three MED findings on the iteration-10 fixes, plus one lead-issued class completion. FC-3 was
dropped by adjudication.

**A-1 (`context.py:_vary_weight`) — the F2 check keyed on the wrong population.** `old._members`
is the §2.4 UNION: after a shift, the ambient weight carries `jes_up`/`jes_down` as union members
even though nothing was ever REGISTERED as a weight under `jes`. The check therefore refused the
legal correlated pair (a `jes` shift, then a `jes` weight factor) — one knob per universe, exactly
what §2.1 permits. Now keyed on `{f"{n}_{t}" for n, ts in old._tags.items() for t in ts}`, the
labels actually registered as weight variations.

**A-2 (`vary.py:_vary_loose`) — guard order.** The duplicate-label check ran after the row-space
maps. A colliding label shadows its existing member in `{**existing, **resolved}`, so
`check_members` never sees that member's handle; with the shadowed member the only contexted one,
the container is left with no handle and `_align` dies on `AttributeError: 'NoneType' object has no
attribute '_links_below'` where the designed `GraphedError` belongs. The check moved above both
maps.

**C2 (`awkward/io.py:_syntactic_fields`) — the F7 class, other member.** F7 repaired the walk in
`projection.py`; the structurally identical walk here had the same gap, and it feeds
`_evaluation_columns`, i.e. the per-task parquet read list. An External handed the source RECORD
can replay against any column, so the list cannot narrow. `on_external` now applies `on_op`'s
non-field rule; `touches_source` was lifted so both callbacks share one predicate.

**A-4** — `_align`'s docstring stated the rule ("only across a link that moves rows") without the
reason a `vary` link is excluded. It now names it: a `vary` link is the identity in row space and
content, so re-indexing across it would re-stamp the handle and lose the parent identity §2.3e
pins on the member.

**A-3** — the F9 figures above were measured before this log existed and so were not true of the
tree they shipped in. Re-measured on this tree and corrected in place: 26 findings, not 25, the
extra row being `skip_or_xfail_added` against the F9 block's own positive-control sentence, and 15
added suppression comments, not 14. Both remaining rows are attributed there as the scanner reports
them; neither sentence was reworded to make the instrument quiet.

### Witnesses, each shown discriminating

New in `tests/extra/awkward/m48/`: `test_row_space_and_memoisation.py` gains three, and
`test_syntactic_fields_whole_record.py` is new (3 tests).

| finding | mutation | witness |
|---|---|---|
| A-1 | check keyed back on `old._members` | RED (both registration orders) |
| A-1 | drop the check entirely | RED (the cross-name control still refuses) |
| A-1c | `_align` returns `member` unconditionally | RED (varied-mask path) |
| A-2 | move the check back below the maps | RED (`AttributeError` at `vary.py:103`) |
| C2 | `external=lambda *_a: None` in `io.py` | RED |

A-1 carries its population-axis control: the cross-name collision must STILL be refused with a
shift present, or the narrowing degenerates into accepting everything. C2 carries the same control
F7 does — an External over FIELDS alone must still narrow to those fields.

A-1c is the varied-mask arm of `_align`: the F1 witness drives an UNVARIED mask link, so the
per-label re-indexing path was unwitnessed. Its nominal member is itself a container, which is why
the witness narrows recursively rather than reading `node_id` off it (§2.2 refuses that name).

### Gates

| gate | result | delta vs 38eebf6 |
|---|---|---|
| m48 frozen | 112/112 | none |
| extra m48 witnesses | 28/28 | +7 (3 A-1/A-2, 3 C2, +1 from cycle 1 recount) |
| whole frozen suite | only known-env reds | none: 5 FAILED, all `awkward/m16` `MaybeNone` |
| coverage (combined) | `TOTAL 5928 312 1724 170 93%` | +6 statements, -1 miss |
| ruff / format | clean | `All checks passed!` / `87 files already formatted` |
| mypy `--strict` | clean | `Success: no issues found in 76 source files` |
| determinism | green | frozen F2 inside the 112 |
| integrity scan (live) | 26 findings, all attributed above | delta over 38eebf6 alone: 0 findings |
