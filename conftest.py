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


_skipped_collections: list[str] = []


def pytest_collectreport(report: pytest.CollectReport) -> None:
    if report.skipped:
        _skipped_collections.append(report.nodeid)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int | pytest.ExitCode) -> None:
    # A subtree whose every module importorskip'd (pyarrow has no win_arm64 wheel) is not "no tests
    # collected"; an empty or mistyped path, which skips nothing, still exits 5.
    if exitstatus == pytest.ExitCode.NO_TESTS_COLLECTED and _skipped_collections and not session.testsfailed:
        session.exitstatus = pytest.ExitCode.OK
