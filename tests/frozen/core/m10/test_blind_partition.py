"""M10 — first-class blind partitions (finding C.9): no more negative-entry_stop sentinel.

The uproot integration encoded "step k of n, range unknown until the file is opened" by smuggling
``entry_stop = -n_steps`` through a plain Partition — any consumer unaware of the convention
silently misread the range. `Partition.blind` makes the deferral explicit, resolvable, picklable,
and durable through the DurablePlan codec, while non-blind partitions (and their plan bytes /
content-addressed task ids) are byte-for-byte unchanged.
"""

from __future__ import annotations

import pickle

import pytest

from graphed_core import Partition
from graphed_core.plan import _partition_bytes, _partition_from_json, _partition_json


def test_blind_constructor_and_flags() -> None:
    p = Partition.blind("f.root", "events", 2, 5)
    assert p.is_blind and p.blind_step == 2 and p.blind_n_steps == 5
    assert not Partition("f.root", "events", 0, 10).is_blind


def test_blind_validation() -> None:
    with pytest.raises(ValueError):
        Partition.blind("f.root", "t", 5, 5)
    with pytest.raises(ValueError):
        Partition.blind("f.root", "t", -1, 5)
    with pytest.raises(ValueError):
        Partition.blind("f.root", "t", 0, 0)


def test_resolve_covers_every_entry_exactly_once() -> None:
    for num_entries in (0, 1, 7, 100, 101):
        for n_steps in (1, 2, 3, 7):
            ranges = [Partition.blind("f", "t", k, n_steps).resolve(num_entries) for k in range(n_steps)]
            covered: list[int] = []
            for r in ranges:
                assert not r.is_blind
                covered.extend(range(r.entry_start, r.entry_stop))
            assert covered == list(range(num_entries)), f"{num_entries=} {n_steps=}"


def test_resolve_of_a_concrete_partition_is_identity() -> None:
    p = Partition("f", "t", 3, 9)
    assert p.resolve(1000) == p


def test_blind_partition_pickles() -> None:
    p = Partition.blind("f.root", "events", 1, 4)
    assert pickle.loads(pickle.dumps(p)) == p


def test_blind_n_entries_is_zero_until_resolved() -> None:
    assert Partition.blind("f", "t", 0, 2).n_entries == 0


def test_durable_codec_round_trips_blind_partitions() -> None:
    p = Partition.blind("f.root", "events", 3, 8)
    assert _partition_from_json(_partition_json(p)) == p


def test_non_blind_plan_bytes_are_unchanged() -> None:
    # the M8 determinism pin: adding blind support must not move a single byte of existing plans
    p = Partition("f.root", "events", 0, 100)
    assert _partition_bytes(p) == b'{"entry_start":0,"entry_stop":100,"tree":"events","uri":"f.root"}'
