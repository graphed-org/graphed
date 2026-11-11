"""The README's python blocks run and print the output their comments show."""

from __future__ import annotations

import contextlib
import io
import pathlib
import re

import awkward as ak
import pytest
import uproot

import graphed

README = pathlib.Path(__file__).parents[3] / "README.md"
BLOCKS = re.findall(r"```python\n(.*?)```", README.read_text(), re.S)


def test_the_readme_has_python_blocks() -> None:
    assert len(BLOCKS) >= 3
    assert 'uproot.graphed("data.root:Events")' in BLOCKS[0]


@pytest.fixture
def data_root(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Write a two-dimuon ``data.root`` in the working directory, for the quick-look block."""
    mu = ak.Array(
        [
            [[45.0, 0.1, 0.0], [45.0, -0.1, 3.14159]],
            [[30.0, 1.0, 1.0]],
            [[50.0, 0.5, 0.5], [40.0, -0.5, -2.6]],
        ]
    )
    monkeypatch.chdir(tmp_path)
    with uproot.recreate("data.root") as file:
        file["Events"] = {
            "Muon_pt": mu[..., 0],
            "Muon_eta": mu[..., 1],
            "Muon_phi": mu[..., 2],
            "Muon_mass": mu[..., 0] * 0 + 0.1057,
        }


@pytest.mark.parametrize("block", BLOCKS)
def test_a_readme_block_prints_what_it_shows(block: str, data_root: None) -> None:
    shown = [line[2:] for line in block.splitlines() if re.match(r"# [\[(\w]", line)]
    printed = io.StringIO()
    previous = graphed.set_session(None)
    try:
        with contextlib.redirect_stdout(printed):
            exec(compile(block, str(README), "exec"), {})
    finally:
        graphed.set_session(previous)
    assert printed.getvalue().splitlines() == shown
