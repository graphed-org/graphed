"""m51 anchor F — the manifest is `PYTHONHASHSEED`-independent (§6.4e; §8.2(i)'s closure hazard).

A content-only manifest anchor is satisfied by an unsorted, hash-seed-dependent serialization, so
this anchor pins DETERMINISM directly: the same varied write performed in two FRESH processes under
DIFFERING `PYTHONHASHSEED` yields byte-identical manifest bytes. Minimally (and observable in one
process): the serialized MAPPING keys are sorted, and the `levels` LIST is in §6.4e's bound order —
"sorted" is not a computable predicate over that list's heterogeneous elements, and
`json.dumps(sort_keys=True)` never reorders list elements, so the order is an explicit key.

The two-process leg is the live instrument; the in-process leg is its positive control — it proves
the manifest carries real, non-empty content (a null manifest would compare equal across seeds and
prove nothing).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest
from m51_write_fixtures import (
    events_context,
    raw_manifest_bytes,
    weight_skim_inputs,
)

pytest.importorskip("pyarrow")

_M51_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_M51_DIR, "..", "..", "..", ".."))


def _manifest_hex_in_subprocess(seed: str, destination: str) -> str:
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = seed
    env["PYTHONPATH"] = os.pathsep.join(
        [os.path.join(_REPO_ROOT, "python"), os.path.join(_REPO_ROOT, "tests", "_corpus"), _M51_DIR]
    )
    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys, m51_write_fixtures as f; sys.stdout.write(f.emit_weight_manifest_hex(sys.argv[1]))",
         destination],
        capture_output=True, text=True, env=env, check=False,
    )
    assert proc.returncode == 0, f"seed={seed} write failed:\n{proc.stderr}"
    return proc.stdout.strip()


def test_manifest_bytes_are_identical_across_two_hash_seeds(tmp_path) -> None:  # type: ignore[no-untyped-def]
    hex0 = _manifest_hex_in_subprocess("0", str(tmp_path / "seed0"))
    hex1 = _manifest_hex_in_subprocess("1", str(tmp_path / "seed1"))
    assert hex0 and hex1  # positive control: the instrument actually produced manifest bytes
    assert hex0 == hex1  # byte-identical manifest under differing PYTHONHASHSEED


def test_manifest_mapping_keys_are_sorted_and_levels_list_is_ordered(tmp_path) -> None:  # type: ignore[no-untyped-def]
    _s, events = events_context()
    record, evt_mask = weight_skim_inputs(events)
    import graphed.awkward as ga  # noqa: PLC0415

    paths = ga.to_parquet(record, str(tmp_path / "det"), select={0: evt_mask})  # type: ignore[call-arg]
    blob = raw_manifest_bytes(paths[0])
    assert blob is not None and b"murf_5em1" in blob  # instrument live: real, non-empty content
    manifest = json.loads(blob)

    # json.loads preserves file key order, so this is a direct, separator-agnostic sortedness proof
    assert list(manifest) == sorted(manifest)
    for label, columns in manifest.items():
        if label != "levels":
            assert list(columns) == sorted(columns), label
    assert manifest["levels"] == [0]  # the levels entry's own bound order
