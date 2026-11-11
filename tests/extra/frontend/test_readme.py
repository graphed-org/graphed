"""The README's python blocks run and print the output their comments show."""

from __future__ import annotations

import contextlib
import io
import pathlib
import re

import pytest

README = pathlib.Path(__file__).parents[3] / "README.md"
BLOCKS = re.findall(r"```python\n(.*?)```", README.read_text(), re.S)


def test_the_readme_has_python_blocks() -> None:
    assert len(BLOCKS) >= 3
    assert "uproot.graphed(" in BLOCKS[0]


@pytest.mark.parametrize("block", BLOCKS, ids=lambda block: block.splitlines()[-1])
def test_a_readme_block_prints_what_it_shows(
    block: str, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)  # the quick look writes its stand-in ROOT file here
    shown = [line[2:] for line in block.splitlines() if re.match(r"# [\[(\w]", line)]
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        exec(compile(block, str(README), "exec"), {})
    assert printed.getvalue().splitlines() == shown
