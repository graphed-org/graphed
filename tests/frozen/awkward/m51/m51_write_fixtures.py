"""Fixtures + freeze spellings for the m51 awkward-idiom variation-aware write-out anchors (§6.4).

Every anchor here imports `graphed.awkward`, so none can live under `tests/frozen/frontend|numpy`
(a required free-threaded job collects those under `pytest hypothesis numpy` alone). The helper
carries an `m51_` prefix even though `run-tests.sh` gives `tests/frozen/awkward/m51` its own process:
prepend import mode binds a bare top-level helper name to whichever sibling dir imported first.

────────────────────────────────────────────────────────────────────────────────────────────────
FREEZE SPELLINGS (this test-author owns them; the plan defers the naming to m51 freeze — §6.4a/b/e,
§9.1; decomposition §4). They are LAW: the implementer conforms. Asserted consistently below.

  * `graphed.selection(ctx)`            — the §9.1 three-case bridge verb (top-level module export).
  * `graphed.awkward.read_varied(path)` — the symmetric awkward-idiom reader → {label: ak.Array}.
  * `to_parquet(record, select=…)`      — `select=` is a single `Varied` row mask (⇔ `{0: mask}`) OR
      a mapping keyed by  `int`  (bare depth, the RECORD'S OWN axis at that depth)  or
      `tuple[str, int]`  (a field-scoped `(field_name, depth≥1)` entry). §6.4a's key space.
  * on-disk VALUE-delta column   `__vary_{label}__{field_flat}`   (e.g. `__vary_murf_5em1__Jet_pt`);
    field path flattened per level with `_`  (`Jet.pt` → `Jet_pt`; a bare-field skim's `pt` → `pt`).
  * on-disk validity-MASK column `__vary_{label}__mask__{entry_flat}`  (`…__mask__0` for level 0,
    `…__mask__Jet_1` for a field-scoped `("Jet",1)`, `…__mask__1` for a bare depth-1 entry).
  * parquet KV metadata key      `b"graphed.variations"`  (greenfield — graphed writes none today).
  * MANIFEST (json, stored under that key):  a mapping whose TOP-LEVEL keys are every label
    (`"nominal"` included) PLUS the reserved key `"levels"`.  Each label maps to
      `{stored_column: {"representation": "base"|"xor"|"packbits", "field": <flat_field>|null,
                        "entry": <int | [flat_field, depth]>|null}}`
    (value deltas: representation `"xor"`, `field` set, `entry` null; masks: representation
    `"packbits"`, `field` null, `entry` the level).  `"levels"` is a LIST whose elements are an
    `int` depth or a two-element `[flat_field, depth]`, ordered by the key `(depth, flat_field or "")`
    — bare-depth before field-scoped of the same depth (§6.4e; `sorted()` cannot order the mixed
    list and `json.dumps(sort_keys=True)` never reorders a list, so the order is an explicit key).
    Serialized `json.dumps(manifest, sort_keys=True)` — sorted mapping keys, `PYTHONHASHSEED`-free.
  * numpy refusal (anchor L, other tree): `GraphedError` naming the awkward backend on a `Varied`
    first positional to `graphed.numpy.io.to_parquet`.

RECONSTRUCTION CONTRACT (freeze; §6.4c "post-selection values and row set"): `read_varied(path)[L]`
is the record for universe L, restricted to the rows passing L's stored level-0 mask AND, at every
supplied level ≥ 1, the objects passing L's stored mask — the analyst's selected data for that
universe. Bit-exact vs the in-memory varied run (XOR deltas are exact by construction).
────────────────────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import awkward as ak
import numpy as np
from graphed_corpus import make_events

import graphed
import graphed.awkward as ga
from graphed import Array, Session
from graphed.awkward import AwkwardBackend, AwkwardForm, from_awkward, gak
from graphed.core import Partition

# ---- freeze constants (see the module docstring) --------------------------------------------
MANIFEST_KEY = b"graphed.variations"
LEVELS_ENTRY = "levels"


def value_column(label: str, field_flat: str) -> str:
    return f"__vary_{label}__{field_flat}"


def mask_column(label: str, entry_flat: str) -> str:
    return f"__vary_{label}__mask__{entry_flat}"


#: one synthetic dataset for the whole tree (jagged Jet/Muon/… + a per-event MET record)
N_EVENTS = 60
SEED = 51
EVENTS = make_events(n_events=N_EVENTS, seed=SEED)


def awkward_session() -> tuple[Session, Array]:
    session = Session(AwkwardBackend())
    return session, from_awkward(session, "events", ak.Array(EVENTS))


def events_context() -> tuple[Session, Any]:
    """The §2.6 event context over the corpus events."""
    session, root = awkward_session()
    return session, ga.gnano.events(root)


# ---- INDEPENDENT eager references (§5.2a: never derived from the graphed graph under test) ----
def raw_events() -> ak.Array:
    return ak.Array(EVENTS)


def _scaled_jets(label: str) -> ak.Array:
    """Plain-awkward jets under the `jes` shift used across the object-migration fixtures."""
    jets = raw_events().Jet
    if label == "jes_up":
        return ak.with_field(jets, jets.pt * 1.05, "pt")
    if label == "jes_down":
        return ak.with_field(jets, jets.pt * 0.95, "pt")
    return jets


JES_LABELS = ("nominal", "jes_up", "jes_down")
#: level-0 event predicate and level-1 per-jet predicate, both MIGRATING under the shift
_EVT_PT = 30.0
_JET_PT = 25.0


def eager_evt_mask(label: str) -> ak.Array:
    return ak.any(_scaled_jets(label).pt > _EVT_PT, axis=1)


def eager_jet_mask(label: str) -> ak.Array:
    return _scaled_jets(label).pt > _JET_PT


def eager_superset_rows() -> ak.Array:
    """The written row set: level-0 OR over every universe's event mask (§6.4a)."""
    acc = eager_evt_mask("nominal")
    for label in JES_LABELS[1:]:
        acc = acc | eager_evt_mask(label)
    return acc


def eager_jet_universe(label: str) -> ak.Array:
    """The in-memory varied run for the bare-key single-collection Jet skim: universe `label`'s
    jets on the rows passing its event mask, inner-filtered by its per-jet mask."""
    jets, em, jm = _scaled_jets(label), eager_evt_mask(label), eager_jet_mask(label)
    kept = jets[em]
    return kept[jm[em]]


def eager_multifield_universe(label: str) -> ak.Array:
    """The in-memory varied run for the field-scoped `{Jet, MET}` record: same selection, but the
    Jet field is inner-filtered while MET rides the event mask only."""
    raw = raw_events()
    jets, em, jm = _scaled_jets(label), eager_evt_mask(label), eager_jet_mask(label)
    kept_jets = jets[em]
    return ak.zip({"Jet": kept_jets[jm[em]], "MET": raw.MET[em]}, depth_limit=1)


def eager_row_universe(label: str) -> ak.Array:
    """The weight-only per-event skim's universe `label`: `{met_pt, w}` on the superset rows."""
    raw = raw_events()
    w = ak.prod(1.0 + raw.Jet.btag, axis=1)
    factor = {"nominal": w, "murf_1": w, "murf_5em1": w * 0.5, "murf_2": w * 2.0}[label]
    mask = eager_row_mask()
    return ak.zip({"met_pt": raw.MET.pt[mask], "w": factor[mask]}, depth_limit=1)


ROW_LABELS = ("nominal", "murf_1", "murf_5em1", "murf_2")


def eager_row_mask() -> ak.Array:
    """The weight skim's (unvaried) level-0 selection — a plain per-event predicate."""
    raw = raw_events()
    return raw.MET.pt > float(np.median(np.asarray(raw.MET.pt)))


# ---- graphed varied-record builders (the write inputs) ---------------------------------------
def jes_context(events: Any) -> Any:
    """`events` with a `jes` shift on the Jet collection (per-jet pt scaled ±5%)."""
    jets = events.Jet
    up = gak.with_field(jets, jets.pt * 1.05, "pt")
    down = gak.with_field(jets, jets.pt * 0.95, "pt")
    return graphed.vary(events, "jes", collections={"Jet": {"up": up, "down": down}})


def jet_skim_inputs(events: Any) -> tuple[Any, Any, Any]:
    """(record, evt_mask, jet_mask) for the BARE-KEY single-collection Jet skim; all `Varied`."""
    ctx = jes_context(events)
    vjets = ctx.Jet
    evt_mask = gak.any(vjets.pt > _EVT_PT, axis=1)
    jet_mask = vjets.pt > _JET_PT
    return vjets, evt_mask, jet_mask


def multifield_skim_inputs(events: Any) -> tuple[Any, Any, Any]:
    """(record, evt_mask, jet_mask) for the FIELD-SCOPED `{Jet, MET}` record."""
    ctx = jes_context(events)
    vjets = ctx.Jet
    record = gak.zip({"Jet": vjets, "MET": events.MET}, depth_limit=1)
    evt_mask = gak.any(vjets.pt > _EVT_PT, axis=1)
    jet_mask = vjets.pt > _JET_PT
    return record, evt_mask, jet_mask


def superset_inputs(events: Any) -> tuple[Any, Any]:
    """(record, evt_mask) for the superset anchor: an UNVARIED per-event `{met_pt}` record (met_pt is
    distinct per event, an identity to trace) written under a VARIED level-0 event mask that migrates
    with the JES shift, so the universes' row sets genuinely differ and their union is the superset."""
    ctx = jes_context(events)
    vjets = ctx.Jet
    record = gak.zip({"met_pt": events.MET.pt}, depth_limit=1)
    evt_mask = gak.any(vjets.pt > _EVT_PT, axis=1)
    return record, evt_mask


def eager_universe_met(label: str) -> ak.Array:
    return raw_events().MET.pt[eager_evt_mask(label)]


def eager_superset_met() -> ak.Array:
    return raw_events().MET.pt[eager_superset_rows()]


def weight_collapse_context(events: Any) -> Any:
    """`events` with a μR/μF weight family whose `murf_1` member EQUALS nominal (all-zero delta),
    interning to nominal's node id (§7.2 collapse) — the anchor-B replication discriminator, and the
    e-canonical `murf_5em1` label carrier."""
    w = gak.prod(1.0 + events.Jet.btag, axis=1)
    return graphed.vary(events, "murf", w, is_weight=True, variations={"1": w, "0.5": w * 0.5, "2": w * 2.0})


def weight_skim_inputs(events: Any) -> tuple[Any, Any]:
    """(record, evt_mask) for the weight-only per-event skim: a `{met_pt, w}` record whose `w` field
    is the collapsing ambient weight; a plain (unvaried) level-0 event mask."""
    ctx = weight_collapse_context(events)
    amb = graphed.weight(ctx)
    record = gak.zip({"met_pt": events.MET.pt, "w": amb}, depth_limit=1)
    threshold = float(np.median(np.asarray(EVENTS.MET.pt)))
    evt_mask = ctx.MET.pt > threshold
    return record, evt_mask


def multiplicity_change_inputs(events: Any) -> tuple[Any, Any]:
    """(record, evt_mask) whose Jet field's per-label OFFSETS differ from nominal — a cleaning that
    drops a different number of jets per universe (§6.4d passes §2.1's TYPE check yet has no
    representable XOR delta). The offsets refusal is EXECUTION-time (§6.4a(1))."""
    jets = events.Jet
    up = jets[jets.pt > 10.0]
    down = jets[jets.pt > 5.0]
    ctx = graphed.vary(events, "clean", collections={"Jet": {"up": up, "down": down}})
    threshold = float(np.median(np.asarray(EVENTS.MET.pt)))
    return ctx.Jet, events.MET.pt > threshold


def _sf_context(events: Any) -> Any:
    w = gak.prod(1.0 + events.Jet.btag, axis=1)
    return graphed.vary(events, "sf", w, is_weight=True, up=w * 1.1, down=w * 0.9)


def rowspace_negative_inputs(events: Any) -> tuple[Any, Any]:
    """(record, evt_mask) storing `graphed.weight(sel)` for a SELECTION-derived `sel`: its per-label
    members live in `sel`'s post-selection row space (§2.6c), not the record's superset — §6.4b
    refuses this at the entry check naming the ROW SPACE, never offsets."""
    c = _sf_context(events)
    sel = c[gak.num(c.Jet) >= 2]
    record = gak.zip({"met": c.MET.pt, "w": graphed.weight(sel)}, depth_limit=1)
    threshold = float(np.median(np.asarray(EVENTS.MET.pt)))
    return record, c.MET.pt > threshold


def rowspace_positive_inputs(events: Any) -> tuple[Any, Any]:
    """The storable spelling: `graphed.weight(c)` for a `c` reached from the record's context across
    `vary` IDENTITY links only — same-length members in the record's own row space (§6.4b)."""
    c = _sf_context(events)
    record = gak.zip({"met": c.MET.pt, "w": graphed.weight(c)}, depth_limit=1)
    threshold = float(np.median(np.asarray(EVENTS.MET.pt)))
    return record, c.MET.pt > threshold


# ---- single-read witness (§5.2b): a PartitionedSource that COUNTS partition reads --------------
@dataclass
class CountingSource:
    """An awkward `PartitionedSource` that records every `read_partition` call — the m48 `ArraySource`
    shape. A varied write must read each partition ONCE (one pass for all universes, §7.1), never
    once per label."""

    data: ak.Array
    reads: list[tuple[int, int]] = field(default_factory=list)

    def __call__(self) -> ak.Array:
        raise AssertionError("the whole-dataset loader must never run during a varied write plan")

    def partitions(self, steps_per_file: int = 1) -> tuple[Partition, ...]:
        return tuple(Partition.blind("toy://events", "", s, steps_per_file) for s in range(steps_per_file))

    def read_partition(self, partition: Partition, columns: Any, resources: Any) -> ak.Array:
        part = partition.resolve(len(self.data))
        self.reads.append((part.entry_start, part.entry_stop))
        return self.data[part.entry_start : part.entry_stop]


def counting_source_events() -> tuple[Session, Any, CountingSource]:
    """(session, event context, the read-counting source) over a PARTITIONED awkward source."""
    data = ak.Array(EVENTS)
    tt = ak.Array(data.layout.to_typetracer(forget_length=True))
    source = CountingSource(data)
    session = Session(AwkwardBackend())
    root = session.source("events", form=AwkwardForm(tt), data=source)
    return session, ga.gnano.events(root), source


# ---- raw-file readers (schema + manifest, through the PUBLIC parquet surface) -----------------
def _pq() -> Any:
    import pyarrow.parquet as pq  # noqa: PLC0415

    return pq


def raw_manifest_bytes(path: str) -> bytes | None:
    """The graphed manifest KV bytes in a parquet part's file metadata, `None` when absent."""
    meta = _pq().ParquetFile(path).metadata.metadata or {}
    return meta.get(MANIFEST_KEY)


def raw_manifest(path: str) -> dict[str, Any]:
    blob = raw_manifest_bytes(path)
    assert blob is not None, f"no {MANIFEST_KEY!r} KV entry in {path}"
    return json.loads(blob)


def raw_schema_names(path: str) -> list[str]:
    """Every leaf column name in a parquet part (dotted for nested awkward records)."""
    return list(_pq().ParquetFile(path).schema_arrow.names)


def as_list(value: object) -> Any:
    return ak.to_list(value)


# ---- self-contained writer for the two-process determinism anchor (F) ------------------------
def emit_weight_manifest_hex(destination: str) -> str:
    """Write the weight skim to `destination` and RETURN the hex of its manifest KV bytes.

    Invoked in a fresh subprocess under a chosen `PYTHONHASHSEED` by the determinism anchor; kept a
    module function so the subprocess reproduces the identical write. Uses `select=`/the manifest —
    m51-new — so it fails loudly (nonzero exit) until the feature exists.
    """
    _session, events = events_context()
    record, evt_mask = weight_skim_inputs(events)
    paths = ga.to_parquet(record, destination, select={0: evt_mask})  # type: ignore[call-arg]
    blob = raw_manifest_bytes(paths[0])
    assert blob is not None
    return blob.hex()
