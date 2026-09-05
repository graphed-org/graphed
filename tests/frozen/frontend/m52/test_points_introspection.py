"""C2 / design §4.10, §5.3: what `graphed.points()` is defined on, and its determinism.

The subprocess child is the `frontend/m48` two-seed idiom — two FRESH interpreters under differing
`PYTHONHASHSEED`, which is the only form that can see a set-ordered mapping at all.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
from m52_point_fixtures import two_axis_context, weight_context

import graphed
from graphed.errors import GraphedError

_CHILD = """
import sys
sys.path.insert(0, {helpers!r})
import graphed
from m52_point_fixtures import two_axis_context

_session, ctx = two_axis_context()
weight = ctx["pt"] * 0.5
registered = graphed.vary(ctx, "corr", weight, is_weight=True,
                          variations={{"a": weight * 3.0}},
                          points=[{{"corr": "a", "jes": "up"}}])
ambient = graphed.weight(registered)
print(repr(graphed.points(ambient)))
print(",".join(graphed.labels(ambient)))
print(hash("graphed"))
"""


def _child(seed: str) -> tuple[str, str, str]:
    env = {**os.environ, "PYTHONHASHSEED": seed}
    program = _CHILD.format(helpers=os.path.dirname(os.path.abspath(__file__)))
    done = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, env=env, check=False
    )
    assert done.returncode == 0, done.stderr
    rendered, labels, salt = done.stdout.splitlines()
    return rendered, labels, salt


def test_points_refuses_a_result_mapping() -> None:
    """§4.10: points are a record-time fact. Answering `{label: {}}` for an executed result asserts
    that every executed universe is the origin."""
    _session, ctx = weight_context()

    result_shaped = dict.fromkeys(graphed.labels(ctx), ctx)
    with pytest.raises(GraphedError) as caught:
        graphed.points(result_shaped)
    assert "point" in str(caught.value).lower()

    # the positive control: the record-time shape the same call answers
    assert graphed.points(ctx)["btag_up"] == {"btag": "up"}


def test_points_is_sorted_and_stable_across_hash_seeds() -> None:
    _session, ctx = two_axis_context()
    weight = ctx["pt"] * 0.5
    registered = graphed.vary(
        ctx,
        "corr",
        weight,
        is_weight=True,
        variations={"a": weight * 3.0},
        points=[{"corr": "a", "jes": "up"}],
    )

    reported = graphed.points(graphed.weight(registered))
    assert list(reported) == sorted(reported)
    assert reported["nominal"] == {}
    assert reported["corr_a__jes_up"] == {"corr": "a", "jes": "up"}
    for point in reported.values():
        assert list(point) == sorted(point)

    one = _child("1")
    two = _child("424242")
    assert one[2] != two[2], (
        "the two children salted their string hashes identically, so the instrument is dead and a "
        "set-ordered points mapping would pass this test unseen"
    )
    assert one[0] == two[0]
    assert one[1] == two[1]
