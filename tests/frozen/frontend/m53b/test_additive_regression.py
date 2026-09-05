"""vary-m53b plan §7.7 / §7.8: the additive increment must not disturb the two behaviors it sits
beside, and its registry is deterministic.

Two are POSITIVE CONTROLS (live on the current tree, proving the harness is real, not merely failing):
(a) a GENUINELY dependent member still auto-mints its compound universe with NO ``points=`` (§1b — a
    2+-foreign universe from real member dependence already works and needs no additive code);
(b) a pure ``points=[{prune}]`` selection is byte-identical to the frozen m53 prune behavior.

The determinism witness FAILS on the current tree for feature-absence (its additive build is refused),
and PASSES under the feature: two interpreters at different ``PYTHONHASHSEED`` produce a byte-identical
``graphed.points()`` registry.
"""

from __future__ import annotations

import os
import subprocess
import sys

from m53b_offgrid_fixtures import VECTOR

import graphed
from graphed import Session
from graphed.context import EventContext
from graphed.numpy import NumpyBackend, from_record


def test_a_genuinely_dependent_member_still_auto_mints_its_compound() -> None:
    """§1b, the measured carve-out: jer members read off the jes-varied pt genuinely depend on jes,
    so a plain vary mints the compound ``jer_hi__jes_up`` → ``{jer: hi, jes: up}`` with NO points=."""
    session = Session(NumpyBackend())
    record = from_record(session, "ev", pt=VECTOR)
    pt = record["pt"]
    ctx = EventContext(session, pt, collections={"pt": pt})
    shifted = graphed.vary(ctx, "jes", collections={"pt": {"up": pt * 1.05, "down": pt * 0.95}})
    varied_pt = shifted["pt"]  # jes-varied
    compound = graphed.vary(
        shifted, "jer", collections={"pt": {"hi": varied_pt * 1.02, "lo": varied_pt * 0.98}}
    )

    points = graphed.points(compound)
    assert points["jer_hi__jes_up"] == {"jer": "hi", "jes": "up"}
    assert points["jer_lo__jes_down"] == {"jer": "lo", "jes": "down"}


def test_a_pure_prune_selection_is_byte_identical_to_the_frozen_behavior() -> None:
    """§7.7(b): the additive router leaves the frozen prune path untouched — a pure prune list yields
    exactly the m53-freeze registry."""
    session = Session(NumpyBackend())
    record = from_record(session, "ev", pt=VECTOR, eta=VECTOR / 10.0)
    pt = record["pt"]
    ctx = EventContext(session, pt, collections={"pt": pt, "eta": record["eta"]})
    shifted = graphed.vary(ctx, "jes", collections={"pt": {"up": pt * 1.10, "down": pt * 0.90}})
    varied = shifted["pt"]  # jes-varied → the btag family fans out over jes
    factors = {"hf_up": 1.03, "hf_down": 0.97, "lf_up": 1.05, "lf_down": 0.95}
    members = {tag: varied * factor for tag, factor in factors.items()}
    registered = graphed.vary(
        shifted,
        "btag",
        varied * 1.0,
        is_weight=True,
        variations=[*members.items(), {"btag": "hf_up", "jes": "up"}, {"btag": "lf_down", "jes": "down"}],
    )

    assert graphed.points(graphed.weight(registered)) == {
        "nominal": {},
        "jes_up": {"jes": "up"},
        "jes_down": {"jes": "down"},
        "btag_hf_up": {"btag": "hf_up"},
        "btag_hf_down": {"btag": "hf_down"},
        "btag_lf_up": {"btag": "lf_up"},
        "btag_lf_down": {"btag": "lf_down"},
        "btag_hf_up__jes_up": {"btag": "hf_up", "jes": "up"},  # the two named joints kept
        "btag_lf_down__jes_down": {"btag": "lf_down", "jes": "down"},
    }


_CHILD = """
import sys
sys.path.insert(0, {helpers!r})
import graphed
from m53b_offgrid_fixtures import two_axis_context, independent_weight

_session, ctx = two_axis_context()
factor = independent_weight(ctx)
registered = graphed.vary(ctx, "corr", factor, is_weight=True,
                          variations=[("a", factor * 3.0), {{"corr": "a", "jes": "up", "jer": "up"}}])
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


def test_the_additive_registry_is_byte_identical_across_hash_seeds() -> None:
    one = _child("1")
    two = _child("424242")
    assert one[2] != two[2], (
        "the two children salted their string hashes identically, so the instrument is dead and a "
        "set-ordered points mapping would pass this test unseen"
    )
    assert one[0] == two[0]  # byte-identical points() registry
    assert one[1] == two[1]  # byte-identical label order
