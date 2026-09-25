"""Parts two writes would share are refused at build, inside one plan and across collated plans."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from m72_multiout_fixtures import (
    DATA_FILES,
    MC_FILES,
    FileSource,
    add,
    json_codec,
    no_paths,
    paths_only,
    read_part,
    record,
)

import graphed
from graphed.core import Partition
from graphed.core.execution import SequentialRunner
from graphed.debug import replay
from graphed.write import PartWrite


def by_uri_step(p: Partition) -> str:
    """Distinct within one plan; `mc-a` is in both fixtures' file sets, under different trees."""
    return f"{p.uri}-{p.blind_step}.json"


def _plan(files: dict[str, Any], tree: str, destination: Path) -> tuple[Any, Any, FileSource]:
    _s, x, src = record(files, tree=tree)
    write = PartWrite(array=x, destination=str(destination), name=by_uri_step, codec=json_codec)
    plan = graphed.aggregate_plan(x.sum(), reduce=paths_only, combine=add, empty=no_paths, writes=[write])
    return plan, x, src


def test_collated_plans_writing_one_part_are_refused(tmp_path: Path) -> None:
    out = tmp_path / "out"
    mc, _, mc_src = _plan(MC_FILES, "Events", out)
    data, _, data_src = _plan(DATA_FILES, "Runs", out)
    with pytest.raises(ValueError, match=r"write the same part .*mc-a-0\.json"):
        graphed.collate({"mc": mc, "data": data})
    with pytest.raises(ValueError, match="write the same part"):
        graphed.collate({"nested": graphed.collate({"mc": mc}), "data": data})
    assert mc_src.reads == [] and data_src.reads == []
    assert not out.exists()


def test_collated_plans_with_distinct_parts_build_and_write_each(tmp_path: Path) -> None:
    mc, _, _ = _plan(MC_FILES, "Events", tmp_path / "mc")
    data, _, _ = _plan(DATA_FILES, "Runs", tmp_path / "data")
    value = SequentialRunner().run(graphed.collate({"mc": graphed.collate({"mc": mc}), "data": data})).value
    assert value == {
        "mc": {"mc": [str(tmp_path / "mc" / f"{uri}-0.json") for uri in MC_FILES]},
        "data": [str(tmp_path / "data" / f"{uri}-0.json") for uri in DATA_FILES],
    }
    assert read_part(value["data"][0])["values"] == DATA_FILES["mc-a"].tolist()
    assert read_part(value["mc"]["mc"][0])["values"] == MC_FILES["mc-a"].tolist()


def test_replay_refuses_a_plan_with_writes(tmp_path: Path) -> None:
    plan, x, src = _plan(MC_FILES, "Events", tmp_path / "out")
    with pytest.raises(TypeError, match="does not re-run a plan's writes"):
        replay(plan, 0, x.sum(), x)
    assert src.reads == []
    assert not (tmp_path / "out").exists()
