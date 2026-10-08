"""A part under a ``dir::`` destination lands at the wrapped URL's path."""

from __future__ import annotations

from pathlib import Path

import pytest

from graphed.write import part_path, prepare_part


def test_dir_part_lands_under_the_wrapped_destination(tmp_path: Path) -> None:
    pytest.importorskip("fsspec")
    out = tmp_path / "out"
    fs, where = prepare_part(part_path(f"dir::{out.as_uri()}", 0, suffix=".bin"))
    fs.pipe_file(where, b"part")
    assert (out / "part-00000.bin").read_bytes() == b"part"
