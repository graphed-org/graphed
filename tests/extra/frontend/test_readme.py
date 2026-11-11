"""The README's python blocks run and print the output their comments show."""

from __future__ import annotations

import contextlib
import io
import pathlib
import re

import pytest

import graphed

README = pathlib.Path(__file__).parents[3] / "README.md"
BLOCKS = re.findall(r"```python\n(.*?)```", README.read_text(), re.S)


def test_the_readme_has_python_blocks() -> None:
    assert len(BLOCKS) >= 2


@pytest.mark.parametrize("block", BLOCKS)
def test_a_readme_block_prints_what_it_shows(block: str) -> None:
    shown = [line[2:] for line in block.splitlines() if re.match(r"# [\[(\w]", line)]
    printed = io.StringIO()
    previous = graphed.set_session(None)
    try:
        with contextlib.redirect_stdout(printed):
            exec(compile(block, str(README), "exec"), {})
    finally:
        graphed.set_session(previous)
    assert printed.getvalue().splitlines() == shown
