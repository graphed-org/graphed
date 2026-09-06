"""Fixtures for the frontend m53b additive / off-grid ``points=`` suite.

The additive mode annotates a NAMED *independent* member with a multi-axis point over *foreign*
axes (the retired m52 mechanism, respelled in the unified m53 LIST form). The member does not
naturally carry those universes, so ``points=`` RE-POINTS its one-at-a-time label from the default
``{name: tag}`` to the foreign-only point, dropping the own axis.

Numpy-idiom and awkward-free: the required free-threaded CI job collects ``tests/frozen/frontend``
whole with only ``pytest hypothesis numpy`` installed, so nothing here may import
``graphed.awkward``, ``awkward``, ``hist``, ``boost_histogram``, ``pyarrow`` or ``pandas`` — at
module level or in a body. Every additive re-point is a record-time point-registry fact read back
through ``graphed.points()``; none needs backend materialization, so this increment stays entirely
on the frontend tree.

The ``m53b_`` prefix is load-bearing: the whole ``tests/frozen/frontend`` tree collects in one
process under prepend import mode, where a shared bare helper name binds to whichever sibling
directory imported first.

No additive-routing behavior is reached at import: ``points=`` appears only inside builders the
tests call, so the tree COLLECTS against a tree with no additive implementation and fails at RUN
time (probe "(b)" in the plan: an off-grid entry over an independent member is refused today with
"names no joint the fanout of '<name>' derives; the derived joints are []").
"""

from __future__ import annotations

from typing import Any

import numpy as np

import graphed
from graphed import Session, Varied
from graphed.context import EventContext
from graphed.numpy import NumpyBackend, from_record

#: the 1-D payload every fixture reads
VECTOR = np.arange(1.0, 13.0)
JES_UP, JES_DOWN = 1.10, 0.90


def _record(*, eta: bool = False) -> tuple[Session, Any]:
    session = Session(NumpyBackend())
    if eta:
        return session, from_record(session, "ev", pt=VECTOR, eta=VECTOR / 10.0)
    return session, from_record(session, "ev", pt=VECTOR)


# ---- two-axis carriers: jes + jer registered, an independent member re-pointed over both ----------
def two_axis_loose() -> tuple[Session, Varied]:
    """The loose form's carrier is ``target._tags``: a loose ``Varied`` over ``jes`` and ``jer``.
    An independent member is read off ``graphed.nominal(target)`` (it carries neither family)."""
    session, record = _record()
    pt = record["pt"]
    first = graphed.vary(pt, "jes", up=pt * JES_UP, down=pt * JES_DOWN)
    nominal = graphed.nominal(first)
    return session, graphed.vary(first, "jer", up=nominal * 1.01, down=nominal * 0.99)


def two_axis_context() -> tuple[Session, EventContext]:
    """``jes`` on the ``pt`` collection and ``jer`` on ``eta`` — the carriers a weight/shift-form
    additive point ``{jes: up, jer: up}`` reaches. An independent member is read off
    ``graphed.nominal(ctx["pt"])``."""
    session, record = _record(eta=True)
    pt = record["pt"]
    ctx = EventContext(session, pt, collections={"pt": pt, "eta": record["eta"]})
    shifted = graphed.vary(
        ctx, "jes", collections={"pt": {"up": pt * JES_UP, "down": pt * JES_DOWN}}
    )
    eta = shifted["eta"]
    return session, graphed.vary(
        shifted, "jer", collections={"eta": {"up": eta * 1.01, "down": eta * 0.99}}
    )


def independent_weight(carrier: Any, factor: float = 0.5) -> Any:
    """A weight read off the NOMINAL of a carrier's ``pt`` — independent of every registered
    family, so a plain ``vary`` mints it one-at-a-time and ``points=`` must RE-POINT it."""
    return graphed.nominal(carrier["pt"]) * factor


# ---- numeric grids: μRxμF (2 / 0.5) and the prescribed off-diagonal (1 / -1) ----------------------
def scale_grid() -> tuple[Session, EventContext, Any]:
    """``muR`` and ``muF`` as INDEPENDENT weight families (numeric tags ``2`` / ``0.5``), each
    one-at-a-time (no fanout), plus the plain ``pt`` a scale member is read off. The tour's μRxμF
    7-point grid is the 4 axis-aligned universes these mint, the nominal, and 2 additive diagonals."""
    session, record = _record()
    pt = record["pt"]
    ctx = EventContext(session, pt, collections={"pt": pt})
    mu_r = graphed.vary(ctx, "muR", pt * 0.5, is_weight=True, points={"2": pt * 0.6, "0.5": pt * 0.4})
    ambient = graphed.weight(mu_r)
    mu_f = graphed.vary(
        mu_r, "muF", ambient, is_weight=True, points={"2": ambient * 1.3, "0.5": ambient * 0.7}
    )
    return session, mu_f, pt


def offdiag_axes() -> tuple[Session, EventContext, Any]:
    """``jes`` and ``btag`` as INDEPENDENT weight families with numeric tags ``1`` / ``-1`` — the
    carriers the prescribed off-diagonal ``{jes: 1, btag: -1}`` reaches."""
    session, record = _record()
    pt = record["pt"]
    ctx = EventContext(session, pt, collections={"pt": pt})
    jes = graphed.vary(ctx, "jes", pt * 0.5, is_weight=True, points={"1": pt * 0.6, "-1": pt * 0.4})
    ambient = graphed.weight(jes)
    btag = graphed.vary(
        jes, "btag", ambient, is_weight=True, points={"1": ambient * 1.3, "-1": ambient * 0.7}
    )
    return session, btag, pt


def unregistered_context() -> tuple[Session, EventContext, Any]:
    """No foreign family registered — the context ``{jes: 1, btag: -1}`` must be refused against by
    carrier-reachability, naming what IS registered."""
    session, record = _record()
    pt = record["pt"]
    return session, EventContext(session, pt, collections={"pt": pt}), pt


def triple_numeric() -> tuple[Session, EventContext, Any]:
    """Three INDEPENDENT numeric weight families ``jes`` / ``btag`` / ``jer`` (tags ``1`` / ``-1``),
    so a zero coordinate can drop and still leave a legal TWO-coordinate additive point behind."""
    session, record = _record()
    pt = record["pt"]
    ctx = EventContext(session, pt, collections={"pt": pt})
    jes = graphed.vary(ctx, "jes", pt * 0.5, is_weight=True, points={"1": pt * 0.6, "-1": pt * 0.4})
    w1 = graphed.weight(jes)
    btag = graphed.vary(jes, "btag", w1, is_weight=True, points={"1": w1 * 1.3, "-1": w1 * 0.7})
    w2 = graphed.weight(btag)
    jer = graphed.vary(btag, "jer", w2, is_weight=True, points={"1": w2 * 1.1, "-1": w2 * 0.9})
    return session, jer, pt
