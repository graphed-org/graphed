"""C3 / design §4.7's frozen §2.4 invariant and §5.3: projection changes WHICH member a label gets,
never the union ORDER, and the joint program is byte-deterministic.

The determinism child is the `frontend/m48` two-seed idiom — two FRESH interpreters under differing
`PYTHONHASHSEED`, which is the only form that can see a set-ordered label list at all.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys

from m52_projection_fixtures import JOINT_LABEL, joint_weight_program

import graphed

_CHILD = """
import hashlib, sys
sys.path.insert(0, {helpers!r})
import graphed
from m52_projection_fixtures import joint_weight_program

program = joint_weight_program()
outputs = [graphed.universe(program.ambient, label) for label in graphed.labels(program.ambient)]
print(hashlib.sha256(program.session.serialized_ir(*outputs)).hexdigest())
print(repr(graphed.points(program.ambient)))
print(",".join(graphed.labels(program.ambient)))
print(hash("graphed"))
"""


def _child(seed: str) -> tuple[str, str, str, str]:
    env = {**os.environ, "PYTHONHASHSEED": seed}
    program = _CHILD.format(helpers=os.path.dirname(os.path.abspath(__file__)))
    done = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, env=env, check=False
    )
    assert done.returncode == 0, done.stderr
    digest, rendered, labels, salt = done.stdout.splitlines()
    return digest, rendered, labels, salt


def test_the_union_order_is_unchanged() -> None:
    program = joint_weight_program()
    first, second = program.ambient, program.two_axis

    first_labels = list(graphed.labels(first))
    expected = first_labels + [
        label for label in graphed.labels(second) if label not in first_labels
    ]
    assert expected[0] == "nominal"
    assert expected != first_labels  # the second operand really does contribute a new label

    combined = first * second
    assert list(graphed.labels(combined)) == expected

    # and projection IS engaged on that union: the joint label reads the second operand's shifted
    # member, not its nominal one
    assert (
        graphed.member_of(second, JOINT_LABEL).node_id
        == graphed.universe(second, "jes_up").node_id
    )
    assert (
        graphed.member_of(second, JOINT_LABEL).node_id != graphed.nominal(second).node_id
    )


def test_the_joint_program_is_byte_deterministic_across_two_runs() -> None:
    program = joint_weight_program()
    outputs = [
        graphed.universe(program.ambient, label) for label in graphed.labels(program.ambient)
    ]
    assert hashlib.sha256(program.session.serialized_ir(*outputs)).hexdigest()

    one = _child("1")
    two = _child("424242")
    assert one[3] != two[3], (
        "the two children salted their string hashes identically, so the instrument is dead and a "
        "set-ordered axis union would pass this test unseen"
    )
    assert one[0] == two[0], "the joint program's IR is hash-order dependent (§5.3)"
    assert one[1] == two[1]
    assert one[2] == two[2]
