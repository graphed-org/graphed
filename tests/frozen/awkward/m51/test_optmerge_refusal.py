"""m51 anchor I — §6.4f binds §7.2's optimizer-merge refusal onto the WRITE path.

§7.2: the frontend owns `(output, label) → node id` and derives positions from the DEDUPLICATED
marked-id list; `mark_output` de-dups on the reduced store and `evaluate_ir` returns one value per
DISTINCT compiled output, so the deduplicated frontend list matches the value list element for
element on every program m48 admits — the two disagree ONLY when the M4 OPTIMIZER merges distinct
record ids. §6.4f settles that the varied `to_parquet` is a varied unpack path in §7.2's sense, so
the same distinct-outputs-vs-distinct-marked-ids shortfall check runs there, RECORD-TIME (the call
already compiles, `compile_ir` runs in `awkward/io.py`), with §7.2's message and workaround.

`w * 1.0` is §1.1-legal via `variations={tag: w * float(tag)}` and the M4 identity tokens merge it
with nominal's `w` (confirmed in `tests/frozen/frontend/m48/test_label_out_of_identity.py`:
`compile_ir(s, b, b * 1.0)` yields ONE output), so its two distinct marked ids collapse to one
compiled output — the exact shortfall. The UNVARIED write path is unchanged (§6.3), the positive
control.
"""

from __future__ import annotations

from typing import Any

import pytest
from m51_write_fixtures import events_context

import graphed
import graphed.awkward as ga
from graphed.awkward import gak
from graphed.errors import GraphedError

pytest.importorskip("pyarrow")


def _optimizer_mergeable_record(events: Any) -> tuple[Any, Any]:
    """A weight family with a `w * 1.0` member: distinct record id, but the M4 optimizer merges it
    with nominal's `w` into ONE compiled output."""
    w = gak.prod(1.0 + events.Jet.btag, axis=1)
    ctx = graphed.vary(events, "sys", w, is_weight=True, variations={"one": w * 1.0})
    record = gak.zip({"met": events.MET.pt, "w": graphed.weight(ctx)}, depth_limit=1)
    return record, events.MET.pt > 0.0


def test_optimizer_mergeable_label_is_refused_at_the_call(tmp_path) -> None:  # type: ignore[no-untyped-def]
    _s, events = events_context()
    record, evt_mask = _optimizer_mergeable_record(events)
    # record-time: the shortfall is caught at the call, so even compute=False raises
    with pytest.raises(GraphedError, match=r"merge|optimiz"):
        ga.to_parquet(record, str(tmp_path / "merge"), select={0: evt_mask}, compute=False)  # type: ignore[call-arg]


def test_unvaried_write_path_is_unchanged(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """§6.3 scope control: an unvaried write compiles and writes exactly as today — the merge check
    is varied-only, so a program whose outputs the optimizer merges must keep working."""
    _s, events = events_context()
    paths = ga.to_parquet(events.Jet, str(tmp_path / "plain"))
    assert len(paths) == 1
