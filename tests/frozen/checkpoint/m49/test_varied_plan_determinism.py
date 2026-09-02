"""§8.2(i) — plan bytes and `task_id`s are seed-independent with `variation_labels` POPULATED.

The plan-level twin of §3.2's IR-level anchor. `_PartitionReduce` is embedded BY VALUE
(`OpSpec.from_callable`, §7.3), so the field's pickled representation reaches `DurablePlan.to_bytes`
and `OpSpec.identity()` -> `task_id`. Everything in the field must therefore be ORDERED and SORTED:
a `frozenset` pickles in hash order and would make cross-run checkpoint reuse `PYTHONHASHSEED`-
dependent. The fixture supplies the (beta) hook, since m48's `None` default pickles seed-independently
and would freeze this anchor green against the very shape it exists to ban.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import m49_analyses as A

SEEDS = ("1", "2", "99")
_HERE = str(Path(__file__).resolve().parent)


def _digests(kind: str) -> list[list[str]]:
    """Run the fixture in fresh child processes, one per seed; each prints digest + `task_id`s."""
    out = []
    for seed in SEEDS:
        env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": _HERE}
        proc = subprocess.run(
            [sys.executable, "-c", f"import m49_analyses as A; A.emit_plan_digest({kind!r})"],
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )
        lines = proc.stdout.split()
        assert len(lines) == 1 + A.N_PARTITIONS, proc.stderr
        out.append(lines)
    return out


def test_a_populated_sorted_payload_is_seed_independent() -> None:
    """Byte-identical `DurablePlan.to_bytes()` and identical per-partition `task_id` across seeds."""
    first, *others = _digests("sorted")
    for other in others:
        assert other == first


def test_a_frozenset_payload_is_seed_dependent() -> None:
    """The live control for the instrument above: the ban has teeth only if the bytes can move."""
    digests = [lines[0] for lines in _digests("frozenset")]
    assert len(set(digests)) == len(SEEDS), "the seeds did not perturb the plan bytes at all"


def test_the_payload_reaches_the_plan_bytes() -> None:
    """§7.3's BY-VALUE clause: under `OpSpec.from_ref` the field is not in the bytes and the two
    assertions above would hold for a plan that never carried it."""
    built = [
        A.durable(*A.build_plan(on_compiled=payload)[:2])
        for payload in (A.sorted_payload, lambda compiled: A.sorted_payload(compiled)[:-1])
    ]
    assert built[0].to_bytes() != built[1].to_bytes()
    assert built[0].task_id(built[0].partitions[0]) != built[1].task_id(built[1].partitions[0])
