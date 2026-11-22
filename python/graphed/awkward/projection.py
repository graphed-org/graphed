"""Necessary-buffer (column) projection via a reporting typetracer (plan M5 + M10).

Builds a reporting typetracer per source from its form, replays the recorded stages symbolically
(no event data is read), and collects the touched buffer form-keys. Two granularities share the
replay:

- `project` (M5) maps touched LEAF buffers to column names — the column-level view.
- `project_buffers` (M10) keeps the full buffer-level information awkward already reports: every
  touched structural node (list offsets / index / option masks) is attributed to its dotted path,
  so a count-only access reports `{collection: OFFSETS}` instead of the under-specified empty set,
  and a reader can serve it from a counter branch (TTree) or an index column (RNTuple) without the
  payload. `project_buffers(...).to_projection()` collapses exactly to the M5 column view.

Opaque ops (`map`) honor the on-fail policy. Projection is correct whether or not the graph has
been reduced (it replays the ops, which a fused stage contains unchanged).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import awkward as ak
import numpy as np

from graphed import Array, Session
from graphed.projection import (
    CONSERVATIVE,
    BufferNeed,
    BufferProjection,
    Projection,
    handle_opaque,
)
from graphed.systematics.by_label import _reach
from graphed.systematics.varied import expanding

from .backend import AwkwardForm

_STRUCTURAL = ("is_list", "is_regular", "is_option", "is_indexed")


def _leaf_columns(form: object, path: tuple[str, ...] = ()) -> dict[str, str]:
    """Map each leaf buffer's form_key -> its dotted column path."""
    out: dict[str, str] = {}
    if getattr(form, "is_record", False):
        for field, content in zip(form.fields, form.contents, strict=True):  # type: ignore[attr-defined]
            out.update(_leaf_columns(content, (*path, field)))
    elif any(getattr(form, a, False) for a in _STRUCTURAL):
        out.update(_leaf_columns(form.content, path))  # type: ignore[attr-defined]
    elif getattr(form, "is_numpy", False):
        key = getattr(form, "form_key", None)
        if key is not None:
            out[key] = ".".join(path) if path else "<root>"
    return out


def _structural_paths(form: object, path: tuple[str, ...] = ()) -> dict[str, str]:
    """Map each STRUCTURAL buffer's form_key to the dotted path it shapes.

    Structural buffers are list offsets / index / option masks. The outermost event structure
    (empty path) carries no column to read, so it is deliberately absent.
    """
    out: dict[str, str] = {}
    if getattr(form, "is_record", False):
        for field, content in zip(form.fields, form.contents, strict=True):  # type: ignore[attr-defined]
            out.update(_structural_paths(content, (*path, field)))
    elif any(getattr(form, a, False) for a in _STRUCTURAL):
        key = getattr(form, "form_key", None)
        if key is not None and path:
            out[key] = ".".join(path)
        out.update(_structural_paths(form.content, path))  # type: ignore[attr-defined]
    return out


def _readers(session: Session, arrays: Sequence[Array]) -> Counter[int]:
    """Count, per node of ``arrays``' cones, the reads by other nodes of those cones."""
    cone = _reach(arrays)
    with session._lock:
        return Counter(i for nid in cone for i in session._inputs_of(nid))


@dataclass(frozen=True)
class _Summary:
    """What a memo entry's evaluated node did, by position in its inputs' buffer ids."""

    width: int  # how many input positions
    alias: np.ndarray  # (position, first position holding its buffer) where they differ
    derived: np.ndarray  # [start, stop) runs of positions holding no source buffer
    data: np.ndarray  # the positions whose data it touched
    shape: np.ndarray  # the positions whose shape it touched
    through: np.ndarray  # each output buffer -> the input position it is, -1 if derived

    def carries(self, flat: np.ndarray) -> bool:
        """Whether ``flat`` has the alias pattern and no source buffer where this had none."""
        if len(flat) != self.width or (flat[self.alias[:, 0]] != flat[self.alias[:, 1]]).any():
            return False
        if not len(self.derived):
            return True
        keyed = np.concatenate([[0], np.cumsum(flat >= 0)])
        return not (keyed[self.derived[:, 1]] - keyed[self.derived[:, 0]]).any()


_BUFFERS = {
    "NumpyArray": ("data",),
    "ListOffsetArray": ("offsets",),
    "ListArray": ("starts", "stops"),
    "IndexedArray": ("index",),
    "IndexedOptionArray": ("index",),
    "ByteMaskedArray": ("mask",),
    "BitMaskedArray": ("mask",),
    "UnionArray": ("tags", "index"),
}
_TRACED = (ak.Array, ak.typetracer.TypeTracerArray)  # a scalar's tracer is its one buffer


def _buffers_of(value: object) -> Iterator[Any]:
    """Yield the typetracer buffers of an array or scalar ``value``, preorder."""
    if not isinstance(value, ak.Array):
        yield value
        return
    stack = [value.layout]
    while stack:
        node = stack.pop()
        kind = type(node).__name__
        for name in _BUFFERS.get(kind, ()):
            buffer = getattr(node, name)
            yield buffer if kind == "NumpyArray" else buffer.data  # an Index wraps its buffer
        contents = getattr(node, "contents", None)
        if contents is not None:
            stack.extend(reversed(contents))
        elif hasattr(node, "content"):
            stack.append(node.content)


def _positions(found: set[int]) -> np.ndarray:
    out: np.ndarray = np.fromiter(sorted(found), dtype=np.int32, count=len(found))
    return out


class _Summaries:
    """The walk's op handler: a memo entry's later nodes replay its first node's summary.

    Each source buffer is an id; a summarized node carries its output's buffer ids (an int32
    array) instead of a tracer, and touches what the entry's first node touched at the same input
    positions. A node the summary does not carry to is rebuilt and evaluated alone.
    """

    def __init__(
        self,
        session: Session,
        reports: dict[int, tuple[object, object, str]],
        tracers: dict[int, ak.Array],
        real: set[int],
        learn: set[int],
        entries: Sequence[int],
    ) -> None:
        self.session, self.real, self.learn, self.entries = session, real, learn, entries
        self.ids: dict[str, int] = {}  # source buffer key -> id
        self.held: list[tuple[object, str]] = []  # id -> (report, source buffer key)
        self.log: list[tuple[str, str]] = []  # the source touches since the last clear
        for nid, tracer in tracers.items():
            report = reports[nid][0]
            for buffer in _buffers_of(tracer):
                self.ids[buffer.form_key] = len(self.held)
                self.held.append((report, buffer.form_key))
            for kind in ("touch_data", "touch_shape"):
                setattr(report, kind, self._logged(getattr(report, kind), kind))
        self.keys: dict[int, np.ndarray] = {}  # an evaluated node read by a summarized one
        self.summaries: dict[int, _Summary | None] = {}

    @classmethod
    def plan(
        cls,
        session: Session,
        arrays: Sequence[Array],
        reports: dict[int, tuple[object, object, str]],
        tracers: dict[int, ak.Array],
    ) -> _Summaries | None:
        """Return the handler, or ``None`` when every node must be evaluated anyway.

        Evaluated: each entry's first node in walk order, every node of no entry or whose
        typetracers are not arrays or scalars, and every ancestor of those. An External stands in
        its recorded form's tracer, so it needs no real input -- unless it stands in its first.
        """
        order: list[int] = []
        externals: list[int] = []
        dry: dict[int, object] = {}
        for array in arrays:
            session.walk(
                array,
                source=lambda nid: None,
                op=lambda nid, *_: order.append(nid),
                external=lambda nid, *_: externals.append(nid),
                memo=dry,
            )
        with session._lock:
            entries = session._entries[:]
            ins = {nid: session._inputs_of(nid) for nid in order}
            ins.update(
                (nid, session._inputs_of(nid)[:1])
                for nid in externals
                if not isinstance(session._forms[nid], AwkwardForm)
            )
            traced = {
                nid: isinstance(getattr(session._forms[nid], "tt", None), _TRACED)
                for nid in {*order, *(i for nid in order for i in ins[nid])}
            }
        first: dict[int, int] = {}
        seed = [
            nid
            for nid in order
            if _entry(entries, nid) < 0
            or not all(traced[i] for i in (nid, *ins[nid]))
            or first.setdefault(_entry(entries, nid), nid) == nid
        ]
        real: set[int] = set()
        while seed:
            nid = seed.pop()
            if nid not in real and nid in ins:
                real.add(nid)
                seed.extend(ins[nid])
        learn = {first[_entry(entries, nid)] for nid in order if nid not in real}
        return cls(session, reports, tracers, real, learn, entries) if learn else None

    def _logged(self, touch: Callable[[str], None], kind: str) -> Callable[[str], None]:
        def logged(key: str) -> None:
            self.log.append((kind, key))
            touch(key)

        return logged

    def __call__(
        self, nid: int, name: str, ins: list[object], params: Mapping[str, object]
    ) -> object:
        """Evaluate node ``nid``, or replay its entry's summary on its inputs' buffer ids."""
        if nid in self.real:
            self.log.clear()
            out = self.session.backend.eval_stage(name, ins, params)
            if nid in self.learn:
                self.summaries[_entry(self.entries, nid)] = self._summary(nid, ins, out)
            return out
        summary = self.summaries.get(_entry(self.entries, nid))
        flat = self._flat(nid, ins)
        if summary is None or not summary.carries(flat):
            return self._rebuilt(nid, name, ins, params)
        self.touch(flat[summary.data], "touch_data")
        self.touch(flat[summary.shape], "touch_shape")
        out = np.full(len(summary.through), -1, dtype=np.int32)
        passed = summary.through >= 0
        out[passed] = flat[summary.through[passed]]
        return out

    def _rebuilt(
        self, nid: int, name: str, ins: list[object], params: Mapping[str, object]
    ) -> np.ndarray:
        """Evaluate node ``nid`` on reporting tracers rebuilt from its summarized inputs."""
        with self.session._lock:
            ids = self.session._inputs_of(nid)
        rebuilt: dict[str, int] = {}  # a rebuilt buffer's key -> its source id
        reports, tracers = [], []
        for n, (i, value) in enumerate(zip(ids, ins, strict=True)):
            if not isinstance(value, np.ndarray):
                tracers.append(value)
                continue
            # a summarized input is an array: an op reading a scalar has no memo entry
            tracer, report, _ = _reporting(self.session, i, f"r{n}-{{id}}")
            tracers.append(tracer)
            reports.append(report)
            for buffer, source in zip(_buffers_of(tracer), value.tolist(), strict=True):
                rebuilt[buffer.form_key] = source
        out = self.session.backend.eval_stage(name, tracers, params)
        for report in reports:
            self.touch(_ids(report.data_touched, rebuilt), "touch_data")  # type: ignore[attr-defined]
            self.touch(_ids(report.shape_touched, rebuilt), "touch_shape")  # type: ignore[attr-defined]
        return self._ids(out, rebuilt)

    def _ids(self, value: object, rebuilt: Mapping[str, int]) -> np.ndarray:
        """Each buffer of ``value``: its source buffer's id, ``-1`` for a derived one."""
        keys = (buffer.form_key for buffer in _buffers_of(value))
        out: np.ndarray = np.fromiter(
            (self.ids.get(k, rebuilt.get(k, -1)) for k in keys),
            dtype=np.int32,
        )
        return out

    def _keys_of(self, nid: int, value: object) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value
        got = self.keys.get(nid)
        if got is None:
            got = self.keys[nid] = self._ids(value, {})
        return got

    def forget(self, nid: int) -> None:
        """Drop the buffer ids of evaluated node ``nid``, which no later node reads."""
        self.keys.pop(nid, None)

    def _flat(self, nid: int, ins: list[object]) -> np.ndarray:
        with self.session._lock:
            ids = self.session._inputs_of(nid)
        parts = [self._keys_of(i, value) for i, value in zip(ids, ins, strict=True)]
        return np.concatenate(parts) if parts else np.empty(0, dtype=np.int32)

    def _summary(self, nid: int, ins: list[object], out: object) -> _Summary | None:
        flat = self._flat(nid, ins)
        firsts: dict[int, int] = {}
        alias = []
        for position, key in enumerate(flat.tolist()):
            if key >= 0 and firsts.setdefault(key, position) != position:
                alias.append((position, firsts[key]))
        touched: dict[str, set[int]] = {"touch_data": set(), "touch_shape": set()}
        for kind, label in self.log:
            at = firsts.get(self.ids.get(label, -1))
            if at is None:
                return None  # a touch outside its inputs
            touched[kind].add(at)
        through = []
        for key in self._keys_of(nid, out).tolist():
            at = -1 if key < 0 else firsts.get(key)
            if at is None:
                return None  # a source buffer that none of its inputs holds
            through.append(at)
        edges = np.flatnonzero(np.diff(np.concatenate([[0], (flat < 0).view(np.int8), [0]])))
        return _Summary(
            width=len(flat),
            alias=np.array(alias, dtype=np.int32).reshape(-1, 2),
            derived=edges.astype(np.int32).reshape(-1, 2),
            data=_positions(touched["touch_data"]),
            shape=_positions(touched["touch_shape"]),
            through=np.array(through, dtype=np.int32),
        )

    def touch(self, found: np.ndarray, kind: str) -> None:
        """Touch ``kind`` of each source buffer ``found`` names."""
        if found.size:
            for i in np.unique(found[found >= 0]).tolist():
                report, key = self.held[i]
                getattr(report, kind)(key)


def _entry(entries: Sequence[int], nid: int) -> int:
    return entries[nid] if nid < len(entries) else -1


def _ids(keys: Iterable[str], rebuilt: Mapping[str, int]) -> np.ndarray:
    out: np.ndarray = np.fromiter((rebuilt[k] for k in keys), dtype=np.int32)
    return out


def _reporting(session: Session, nid: int, key: str) -> tuple[ak.Array, object, object]:
    """Return a reporting typetracer of node ``nid``'s form, its buffers keyed by ``key``."""
    form = session.form_of(nid).tt.layout.form_with_key(form_key=key)  # type: ignore[attr-defined]
    layout, report = ak.typetracer.typetracer_with_report(form, highlevel=False)
    return ak.Array(layout), report, form


def _sources(
    session: Session,
) -> tuple[dict[int, tuple[object, object, str]], dict[int, ak.Array]]:
    reports: dict[int, tuple[object, object, str]] = {}
    tracers: dict[int, ak.Array] = {}
    for nid in session.source_ids():
        tracers[nid], report, form = _reporting(session, nid, f"s{nid}-n{{id}}")
        reports[nid] = (report, form, session.source_name(nid))
    return reports, tracers


def _replay(
    arrays: Sequence[Array], on_fail: str
) -> tuple[dict[int, tuple[object, object, str]], bool]:
    """Run the recorded graph of every output on ONE set of reporting typetracers.

    So each report is the union of the outputs' touches. Returns per-source (report, form, name)
    keyed by source node id, plus whether an opaque op forced conservative projection. A node
    whose form came from a Session memo entry another node evaluates replays that node's summary
    (`_Summaries`).
    """
    if not arrays:
        raise ValueError("projection needs at least one output array")
    return _walked(arrays, on_fail, summarize=True)


def _walked(
    arrays: Sequence[Array], on_fail: str, *, summarize: bool
) -> tuple[dict[int, tuple[object, object, str]], bool]:
    """`_replay`; ``summarize=False`` evaluates every node, the result summaries must equal."""
    session = arrays[0].session
    backend = session.backend
    reports, tracers = _sources(session)
    summaries = _Summaries.plan(session, arrays, reports, tracers) if summarize else None
    conservative = False

    def read_fully(value: object) -> None:
        if isinstance(value, ak.Array):
            ak.typetracer.touch_data(value)
        elif summaries is not None and isinstance(value, np.ndarray):
            summaries.touch(value, "touch_data")  # a summarized array

    def on_external(nid: int, _fn: object, inputs: list[object]) -> object:
        nonlocal conservative
        if handle_opaque("map", on_fail) is CONSERVATIVE:
            conservative = True
        # an opaque op reads its inputs fully -> mark their data touched so the input columns are
        # reported (dask-awkward does the same). Then continue with a stand-in.
        for t in inputs:
            read_fully(t)
        # The stand-in is a typetracer of the node's RECORDED form (M23 `record_external(form=)`):
        # a package recording its own External family may declare a type its first input does not
        # have, and a downstream op is well-typed against that declaration alone. A recorded form
        # from another backend (graphed-histogram's HistogramForm) has no typetracer -> first
        # input. `eval_stage` re-wraps every operand with the backend's behavior, so the bare
        # typetracer is enough — a behavior property on the External's output still resolves.
        form = session.form_of(nid)
        return form.tt if isinstance(form, AwkwardForm) else inputs[0]

    memo: dict[int, object] = {}  # one memo, so a sub-DAG the outputs share is replayed once
    readers = _readers(session, [a for a in arrays if a.session is session])
    readers.update(a.node_id for a in arrays)  # an output is read by the caller, so it is kept

    def replayed(nid: int, value: object) -> object:
        # a tracer no later node of any output's cone reads is dropped as soon as that holds
        with session._lock:
            read = session._inputs_of(nid)
        for i in read:
            readers[i] -= 1
            if not readers[i]:
                del memo[i]
                if summaries is not None:
                    summaries.forget(i)
        return value

    def evaluate(nid: int, name: str, ins: list[object], params: Mapping[str, object]) -> object:
        return backend.eval_stage(name, ins, params)

    op = evaluate if summaries is None else summaries
    for array in arrays:  # `walk` refuses an array from another Session
        result = session.walk(
            array,
            source=lambda nid: tracers[nid],
            op=lambda nid, name, ins, params: replayed(nid, op(nid, name, ins, params)),
            external=lambda nid, fn, ins: replayed(nid, on_external(nid, fn, ins)),
            memo=memo,
        )
        # the output is materialized, so its own columns are read — touch them (covers a bare
        # field-access output that otherwise only touches shape, not leaf data).
        read_fully(result)
    return reports, conservative


@expanding
def project(array: Array, *, on_fail: str = "raise") -> Projection:
    """Compute the columns each source must read for ``array`` (metadata-only)."""
    reports, conservative = _replay([array], on_fail)
    read_columns: dict[str, frozenset[str]] = {}
    for report, form, name in reports.values():
        key_map = _leaf_columns(form)
        if conservative:
            cols = set(key_map.values())
        else:
            cols = {key_map[k] for k in report.data_touched if k in key_map}  # type: ignore[attr-defined]
        read_columns[name] = frozenset(cols)
    return Projection(read_columns)


@expanding
def project_buffers(array: Array, *, on_fail: str = "raise") -> BufferProjection:
    """Compute each source's needed columns with their :class:`graphed.projection.BufferNeed`.

    Metadata-only.

    ``DATA`` marks columns whose leaf values are read. ``OFFSETS`` marks paths whose list/option/
    index STRUCTURE alone is needed (a multiplicity, a mask) with no leaf data beneath them — the
    truthful, non-empty answer for a count-only analysis, which the column-level `project`
    necessarily reports as the empty set.
    """
    return _buffers(*_replay([array], on_fail))


@expanding
def project_buffers_many(arrays: Sequence[Array], *, on_fail: str = "raise") -> BufferProjection:
    """:func:`project_buffers` for several outputs of one Session at once.

    The union of their needs, from one reporting typetracer per source rather than one per source
    per output.
    """
    return _buffers(*_replay(arrays, on_fail))


def _buffers(
    reports: dict[int, tuple[object, object, str]], conservative: bool
) -> BufferProjection:
    read_buffers: dict[str, dict[str, BufferNeed]] = {}
    for report, form, name in reports.values():
        leaf_map = _leaf_columns(form)
        if conservative:
            read_buffers[name] = dict.fromkeys(leaf_map.values(), BufferNeed.DATA)
            continue
        needs: dict[str, BufferNeed] = {}
        for key in report.data_touched:  # type: ignore[attr-defined]
            if key in leaf_map:
                needs[leaf_map[key]] = BufferNeed.DATA
        struct_map = _structural_paths(form)
        touched = set(report.data_touched) | set(report.shape_touched)  # type: ignore[attr-defined]
        data_cols = set(needs)
        for key, path in struct_map.items():
            if key not in touched:
                continue
            # a DATA read at-or-under this path already brings its structure along
            covered = any(c == path or c.startswith(path + ".") for c in data_cols)
            if not covered:
                needs.setdefault(path, BufferNeed.OFFSETS)
        read_buffers[name] = needs
    return BufferProjection(read_buffers)
