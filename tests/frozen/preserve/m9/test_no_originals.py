"""M9 acceptance — reproduce on a "machine B" with NO access to the originals.

Plan M9: reproduces "bit-for-bit on machine B that has NO access to the original user code,
environment, author, or input files (inputs resolved only via the bundle's content-addressed
references)." We approximate machine B with a fresh interpreter whose working directory and import
path contain none of this analysis, after deleting the original payload files — it can succeed only
by resolving everything from the bundle's own content-addressed store.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import agc
import numpy as np


def test_reproduce_with_no_original_code_or_inputs(tmp_path) -> None:  # type: ignore[no-untyped-def]
    bundle, reference = agc.build_agc(tmp_path / "build")

    # destroy the originals: the source payload files and the analysis-side working area
    shutil.rmtree(tmp_path / "build" / "payloads")
    elsewhere = tmp_path / "machine_b"
    elsewhere.mkdir()

    child = (
        "import sys, json; from graphed_preserve import Bundle, reproduce;"
        "assert __import__('importlib.util', fromlist=['x']).find_spec('agc') is None;"  # originals unreachable
        "out = reproduce(Bundle.open(sys.argv[1]));"
        "print(json.dumps([float(x) for x in out]))"
    )
    # inherit the OS env (so the interpreter starts on every platform) but drop the import-path vars,
    # so the analysis source (agc.py, only ever on sys.path via the test runner) is unreachable —
    # the child can resolve nothing but installed packages + the bundle's own store.
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    proc = subprocess.run(
        [sys.executable, "-c", child, str(bundle.root)],
        cwd=elsewhere,  # not the build dir; this analysis's source (agc.py) is unreachable
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    counts = np.asarray(json.loads(proc.stdout), dtype="float64")
    assert np.array_equal(counts, reference), "machine-B reproduce must match bit-for-bit"
