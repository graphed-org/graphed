"""Fixtures for the awkward m60 frozen suite — a parquet source keeps awkward's record parameters.

The `m60_` prefix is load-bearing: the frozen awkward tree runs one process per milestone dir, and
under prepend import mode a bare helper name binds to whichever sibling dir imported it first.

Every value is a dyadic rational with a small denominator, so each asserted sum below is exact in
binary floating point. Eager awkward is the ORACLE: `ak.from_parquet` of the same file is what a
recorded form and an executed value must equal.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import awkward as ak
import pyarrow as pa
import pyarrow.parquet as pq

#: the record name the behavior below is keyed on — known to the backend's dict, never to global
#: `ak.behavior`, so a resolution here can only come from the RECORDED form
RECORD_NAME = "m60pair"

_INNER = ak.with_parameter(
    ak.with_name(
        ak.Array([[{"px": 1.0, "py": 2.0}, {"px": 4.0, "py": 8.0}], [{"px": 0.5, "py": 0.25}]]),
        RECORD_NAME,
    ),
    "m60note",
    "kept",
)

#: a named record nested in a list, a custom parameter on an inner node, an option and a regular —
#: the four things an arrow schema alone cannot say
DATA = ak.Array(
    {
        "p": _INNER,
        "o": ak.Array([[1, None], [2]]),
        "r": ak.to_regular(ak.Array([[1.0, 2.0], [4.0, 8.0]])),
    }
)

#: `PairArray.total` over `DATA.p`
TOTALS = [[3.0, 12.0], [0.75]]

#: what today's schema-only route answers for a file pyarrow wrote without awkward's metadata
FOREIGN_FORM = "## * {x: option[var * ?float64], n: ?int64}"


class PairArray(ak.Array):
    """A behavior class the backend's dict alone knows; `total` exists only on the named record."""

    @property
    def total(self) -> Any:
        return self.px + self.py


BEHAVIOR: dict[Any, Any] = {("*", RECORD_NAME): PairArray}


def awkward_dataset(tmp_path: Path, parts: int = 1) -> tuple[str, ...]:
    """`parts` identical parquet files written by awkward itself (metadata and all)."""
    paths = []
    for index in range(parts):
        path = str(tmp_path / f"part{index}.parquet")
        ak.to_parquet(DATA, path)
        paths.append(path)
    return tuple(paths)


def pyarrow_dataset(tmp_path: Path) -> str:
    """The no-metadata control: one file written through bare pyarrow."""
    path = str(tmp_path / "foreign.parquet")
    pq.write_table(
        pa.table({"x": pa.array([[1.0, 2.0], [4.0]]), "n": pa.array([1, None])}), path
    )
    return path


def eager(path: str) -> ak.Array:
    """What awkward's own reader gives the whole file — the oracle for form AND values."""
    return ak.from_parquet(path)


def tracer(array: ak.Array, behavior: Any = None) -> ak.Array:
    """The metadata-only counterpart of `array`: what a recorded form is inferred on."""
    return ak.Array(array.layout.to_typetracer(forget_length=True), behavior=behavior)


def read_guard(monkeypatch: Any, banned: Sequence[str]) -> list[str]:
    """Ban every EVENT-DATA read of `banned` and return the live list of paths opened.

    Metadata stays readable (a schema read must still work) and so does every OTHER file, because
    the contract's own route writes a zero-row file of the dataset's schema and reads it back with
    `ak.from_parquet`. Path-scoped and not blanket, so the instrument cannot ban the answer.
    """
    opened: list[str] = []
    forbidden = {str(path) for path in banned}

    original_init = pq.ParquetFile.__init__

    def init(self: Any, source: Any, *args: Any, **kwargs: Any) -> None:
        where = str(source) if isinstance(source, str | os.PathLike) else ""
        self._m60_source = where
        opened.append(where)
        original_init(self, source, *args, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "__init__", init)

    def refuse(name: str) -> None:
        original = getattr(pq.ParquetFile, name)

        def guarded(self: Any, *args: Any, **kwargs: Any) -> Any:
            if getattr(self, "_m60_source", "") in forbidden:
                raise AssertionError(f"event data read at record time: ParquetFile.{name}")
            return original(self, *args, **kwargs)

        monkeypatch.setattr(pq.ParquetFile, name, guarded)

    for name in ("read", "read_row_group", "read_row_groups", "iter_batches"):
        refuse(name)

    original_read_table = pq.read_table

    def read_table(source: Any, *args: Any, **kwargs: Any) -> Any:
        if str(source) in forbidden:
            raise AssertionError("event data read at record time: pyarrow.parquet.read_table")
        return original_read_table(source, *args, **kwargs)

    monkeypatch.setattr(pq, "read_table", read_table)

    original_from_parquet = ak.from_parquet

    def from_parquet(path: Any, *args: Any, **kwargs: Any) -> Any:
        items = [path] if isinstance(path, str | os.PathLike) else list(path)
        if any(str(item) in forbidden for item in items):
            raise AssertionError("event data read at record time: ak.from_parquet")
        return original_from_parquet(path, *args, **kwargs)

    monkeypatch.setattr(ak, "from_parquet", from_parquet)
    return opened
