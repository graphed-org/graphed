"""m51 anchor J — `to_parquet`'s §2.3d table entry becomes *accepting* (plan §10, §2.3d).

Until m51 `to_parquet` carries no disposition; with §6.4's `select=` landed it enters the §2.3d table
as *accepting*: a `Varied` RECORD and/or a `Varied` `select=` is consumed INTERNALLY and NO per-label
result is returned to the caller (unlike a *broadcasting* verb, which answers with a `Varied`, or a
*refusing* one, which raises). The write returns its ordinary list of part paths.

The disposition is BEHAVIORAL, not table-registered: the frozen m48 gate
(`tests/frozen/awkward/m48/test_module_verb_dispositions.py::test_to_parquet_carries_no_disposition_until_m51`)
hard-asserts `to_parquet` stays out of `VERB_DISPOSITIONS` and out of the discovered surface (its
first parameter is annotated `Any`), so the accepting behaviour must be proven by CALLING it, never by
adding a table row — which is exactly what this anchor does.
"""

from __future__ import annotations

from typing import Any

import pytest
from m51_write_fixtures import events_context, jes_context

import graphed
import graphed.awkward as ga
from graphed.awkward import gak

pytest.importorskip("pyarrow")


def _is_paths(result: Any) -> bool:
    return isinstance(result, list) and all(isinstance(p, str) for p in result)


def test_a_varied_record_and_varied_select_are_consumed_and_return_paths(tmp_path) -> None:  # type: ignore[no-untyped-def]
    _s, events = events_context()
    ctx = jes_context(events)
    vjets = ctx.Jet  # a Varied RECORD
    varied_mask = gak.any(vjets.pt > 30.0, axis=1)  # a Varied select=
    assert isinstance(vjets, graphed.Varied) and isinstance(varied_mask, graphed.Varied)

    result = ga.to_parquet(vjets, str(tmp_path / "j"), select={0: varied_mask, 1: vjets.pt > 25.0})  # type: ignore[call-arg]
    assert _is_paths(result)  # accepting: an ordinary path list, NOT a {label: result} mapping
    assert not isinstance(result, (dict, graphed.Varied))
