# Test dispute — a ONE-coordinate explicit `points=` map is unregistrable, so two spec rows'
# admitted members had to be widened

Filed by `vary-m52-ta-graphed` (test-author) during TEST_AUTHORING. **Not a blocker**: both rows are
written, with the admitted member widened by one coordinate. This records why, so the reviewer does
not read the widening as a weakened test.

## The conflict

Design §4.11-4 requires every coordinate of a `points=` entry to be a tag already registered under
its nuisance. Design §4.11-2 refuses minting a point that is already registered under a different
label. Together those close a door:

* if `{n: t}` passes §4.11-4, then `t` is a registered tag of `n`, so the label `f"{n}_{t}"` was
  minted, and §4.4's default rule gave it exactly the point `{n: t}`;
* so §4.11-2 refuses any other label claiming `{n: t}`.

**A single-coordinate explicit point is therefore always refused.** This is consistent with the
design's purpose (§4.8: `points=` is the mechanism for "a universe displaced on ≥2 axes at once"),
but two rows of `m52-decomposition.md` §3.1 spell an admitted member with one coordinate:

* **2.6** — "`points={"up":{"jes":1,"btag":0}}` succeeds and registers `{jes:1}` only". After the
  zero-drop the surviving point is `{jes: 1}`, which `jes_1` already owns.
* **4.2** (quoting design §8-i's *today* recipe) — "the SAME call with `{"jes":"up"}` succeeds and
  registers the joint universe". `{jes: up}` is `jes_up`'s point.

## What the frozen tests do instead

The property each row tests is preserved; only the admitted member gains a second coordinate, which
makes it a genuinely new point:

* `test_mint_refusals.py::test_an_origin_points_entry_is_refused_while_a_zero_tag_still_mints`
  registers `points={"up": {"jes": 1, "jer": 1, "btag": 0}}` and asserts the surviving point is
  `{"jer": "1", "jes": "1"}` — the zero coordinate still drops, and the drop still happens BEFORE
  §4.11-4's reachability check (`0` is not a registered tag of `btag`), which is the ordering
  decomposition row 1.6's discriminator depends on.
* `test_coordinate_reachability.py::test_a_typed_coordinate_naming_no_registered_tag_is_refused_naming_what_is`
  uses `points={"a": {"jes": "up", "btag": "up"}}` as the positive control and
  `points={"a": {"jes": 1, "btag": "up"}}` as the refusal, so exactly one coordinate differs between
  the admitted and the refused member.

## Proposed correction

Say it once in the design (§4.4 or §4.11-2): a `points=` entry that survives canonicalization with a
single coordinate is refused, because that point is the default point of an existing label. Then
re-spell the two decomposition rows' admitted members with two coordinates, as above.

No frozen test changes under this correction.

## ADJUDICATION (owner, 2026-09-04) — ACCEPTED
Correct: §4.11-4 + §4.11-2 jointly refuse any single-coordinate `points=` entry; `points=` earns a
new label only for a ≥2-coordinate universe (§4.8/§4.9). No frozen test changes (the authored tests
already use ≥2-coordinate admitted members). Folded into design §4.11-2 as an explicit corollary so
the implementer and reviewer read the widened members as intent, not a weakened test.
