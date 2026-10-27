"""One Session's variation registries: what `graphed.vary` minted, and the record-time memos.

`Session.systematics` holds exactly one `SystematicsState`; the Session never interprets it. Every
read and write runs under the Session's `_lock`, and the `vary` transaction holds that lock for its
whole body, so a rollback restores only what its own call wrote and a reader never sees a mint a
rollback is about to undo.

The four rolled-back registries are PUBLISHED, never mutated: a writer replaces the mapping, so a
view is a snapshot a later mint cannot change under its reader, and a transaction's snapshot is the
references it held.
"""

from __future__ import annotations

import threading
import weakref
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

from .points import Point

#: one `_two_level` page: label -> the (outer, inner) member keys the label resolves through
Page = dict[str, tuple[str, str | None]]


class SystematicsState:
    """The variation registries of one Session (§2.5, §4.5, §5)."""

    def __init__(self, lock: threading.RLock) -> None:
        """Keep the registries under ``lock``, the owning Session's."""
        self._lock = lock
        # M52 §4.5: every label `vary` mints, with its POINT, so resolution stays a pure f(label).
        # Record-time only: points are not written to disk (§5.3).
        self._points: dict[str, Point] = {}
        # §4.11-2's inverse (point -> owning label), so "one point wears one label" is O(1)
        self._point_owners: dict[Point, str] = {}
        # §5's MINT EPOCH: moves at every extension of `_points` and every rollback of it. A
        # composed ambient weight is stamped with it; a later mint makes the next read remake it.
        self._epoch = 0
        # M49 §2.5: each weight factor's own member node ids, BY VALUE — the factor container is
        # rebuilt after `register`, so nothing Session-retained carries a collection name
        self._weight_factors: tuple[tuple[str, tuple[int, ...]], ...] = ()
        # `(family, collection)` -> the offending factor's member ids, so `compile_ir` ships only
        # the pairs whose factor reaches the artifact it compiled
        self._shift_after_weight: dict[tuple[str, str], frozenset[int]] = {}
        # M48 §2.5: every container's labels (by value) beside a weak reference to it, so
        # `compile_ir` can report a label that reaches no marked output
        self._varied: list[tuple[tuple[str, ...], weakref.ref[Any]]] = []
        # `ambient._two_level`'s memo, by container id. Keys and never members, so it pins
        # nothing; a page is stored only where no later mint can move it, and a rollback drops all
        self._universes: dict[int, Page] = {}
        #: `ambient._mul_form`'s memo, (left, right) form descriptions -> product form. Per Session
        #: because the answer depends on this backend instance.
        self.mul_forms: dict[tuple[str, str], Any] = {}

    # ---- read views ----------------------------------------------------------------------------
    @property
    def points(self) -> Mapping[str, Point]:
        """Return the registry, label to point."""
        with self._lock:
            return self._points

    @property
    def point_owners(self) -> Mapping[Point, str]:
        """Return the registry's inverse, point to label."""
        with self._lock:
            return self._point_owners

    @property
    def epoch(self) -> int:
        """Return the mint epoch."""
        with self._lock:
            return self._epoch

    @property
    def weight_factors(self) -> Sequence[tuple[str, tuple[int, ...]]]:
        """Return every registered weight factor as `(family, member node ids)`."""
        with self._lock:
            return self._weight_factors

    @property
    def shift_after_weight(self) -> Mapping[tuple[str, str], frozenset[int]]:
        """Return §2.5's `(family, collection) -> factor node ids` report."""
        with self._lock:
            return self._shift_after_weight

    @property
    def universes(self) -> Mapping[int, Mapping[str, tuple[str, str | None]]]:
        """Return a copy of `ambient._two_level`'s memo pages, by container id."""
        with self._lock:
            return {key: dict(page) for key, page in self._universes.items()}

    def registered_labels(self) -> frozenset[str]:
        """Return every label a registered container carries."""
        with self._lock:
            return frozenset(label for labels, _ref in self._varied for label in labels)

    # ---- writers -------------------------------------------------------------------------------
    def bind(self, minted: Mapping[str, Point]) -> None:
        """Register `{label: point}`, its inverse and a new epoch together."""
        with self._lock:
            owners = dict(self._point_owners)
            for label, point in minted.items():
                owners[point] = label
            self._points = {**self._points, **minted}
            self._point_owners = owners
            self._epoch += 1

    def record_varied(self, labels: tuple[str, ...], container: object) -> None:
        """Record a container's labels for `compile_ir`'s unreached-label report."""
        with self._lock:
            self._varied.append((labels, weakref.ref(container)))

    def record_weight_factor(self, name: str, nodes: tuple[int, ...]) -> None:
        """Record weight family `name`'s own member node ids."""
        with self._lock:
            self._weight_factors = (*self._weight_factors, (name, nodes))

    def report_shift_after_weight(self, pair: tuple[str, str], nodes: frozenset[int]) -> None:
        """Report `(family, collection)`, merging `nodes` into any earlier report of it."""
        with self._lock:
            held = self._shift_after_weight.get(pair, frozenset())
            self._shift_after_weight = {**self._shift_after_weight, pair: held | nodes}

    def universe_page(self, container: object) -> Page:
        """Return `container`'s memo page, made (and dropped with the container) on first use."""
        key = id(container)
        with self._lock:
            page = self._universes.get(key)
            if page is None:
                page = self._universes[key] = {}
                # lock-free: a finalizer runs on whichever thread collects, mid-anything
                weakref.finalize(container, self._universes.pop, key, None)
            return page

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Hold the lock for the body, and undo the body's registrations if it raises (§4.5).

        A failed `vary` must leave no binding behind, or one refusal poisons a label for the life
        of the Session.
        """
        with self._lock:
            saved = (
                self._points,
                self._point_owners,
                self._shift_after_weight,
                self._weight_factors,
            )
            try:
                yield
            except BaseException:
                (
                    self._points,
                    self._point_owners,
                    self._shift_after_weight,
                    self._weight_factors,
                ) = saved
                # the one place a resolution can move BACKWARDS: the epoch moves as for a mint, and
                # the `_two_level` memo, which assumes the registry only grows, is dropped
                self._epoch += 1
                self._universes.clear()
                raise
