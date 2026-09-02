"""§3.2: a variation-expanded graph compiles byte-identically, and `graphed.labels` has one order.

The strong R22.3 form — FRESH processes under differing `PYTHONHASHSEED`, not two compiles in one
interpreter, which cannot see a set-ordered label list at all.
"""

from __future__ import annotations

import os
import subprocess
import sys

from vary_fixtures import loose_varied, vector_source

import graphed

_CHILD = """
import hashlib, sys
sys.path.insert(0, {helpers!r})
import graphed
from vary_fixtures import loose_varied, sibling_outputs, vector_source

session, x = vector_source()
varied = loose_varied(x)
print(hashlib.sha256(graphed.compile_ir(session, *sibling_outputs(varied)).ir).hexdigest())
print(",".join(graphed.labels(varied)))
print(hash("graphed"))
"""


def _child(seed: str) -> tuple[str, str, str]:
    env = {**os.environ, "PYTHONHASHSEED": seed}
    program = _CHILD.format(helpers=os.path.dirname(os.path.abspath(__file__)))
    done = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, env=env, check=False
    )
    assert done.returncode == 0, done.stderr
    digest, labels, salt = done.stdout.split()
    return digest, labels, salt


def test_two_fresh_processes_compile_the_same_bytes() -> None:
    one = _child("1")
    two = _child("424242")
    assert one[2] != two[2], (
        "the two children salted their string hashes identically, so the instrument is dead and a "
        "set-ordered label list would pass this test unseen"
    )
    assert one[0] == two[0], "the variation expansion is hash-order dependent (§3.2)"
    assert one[1] == two[1] == "nominal,jes_up,jes_down"


def test_label_order_is_nominal_first_then_insertion_order() -> None:
    _s, x = vector_source()
    varied = graphed.vary(x, "mur", variations={"2": x * 2.0, "0.5": x * 0.5, "0.25": x * 0.25})
    assert list(graphed.labels(varied)) == ["nominal", "mur_2", "mur_5em1", "mur_25em2"]


def test_stacking_puts_inherited_labels_before_new_ones() -> None:
    _s, x = vector_source()
    stacked = graphed.vary(loose_varied(x), "jer", up=x * 1.1, down=x * 0.9)
    assert list(graphed.labels(stacked)) == ["nominal", "jes_up", "jes_down", "jer_up", "jer_down"]
