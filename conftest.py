"""Repo-root pytest configuration.

`pyproject.toml`'s `pythonpath` puts the vendored `graphed_corpus` mirror and the cross-dir helper
providers on `sys.path` for the pytest process ITSELF. A test that re-runs a recorded program in a
FRESH interpreter — the two-seed `PYTHONHASHSEED` determinism anchors — starts that child from
`os.environ` alone, so without this the child cannot import what its parent could. Export the same
roots.

The two clean-machine anchors that must NOT see this tree scrub `PYTHONPATH` themselves
(`preserve/m9/test_no_originals.py`, `checkpoint/m8/test_no_source.py`), so exporting it here
leaves their isolation intact.
"""

from __future__ import annotations

import os

import pytest


def pytest_configure(config: pytest.Config) -> None:
    roots = [str(config.rootpath / entry) for entry in config.getini("pythonpath")]
    existing = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = os.pathsep.join([*roots, *([existing] if existing else [])])
