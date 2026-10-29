"""Awkward-backend guards refuse with their own messages."""

from __future__ import annotations

import awkward as ak
import pytest

from graphed.awkward import read_varied
from graphed.awkward._templates import _parse_template
from graphed.awkward.join import join_form
from graphed.errors import GraphedError


def test_a_call_template_entry_that_is_no_slot_group_or_constant_is_refused() -> None:
    with pytest.raises(TypeError, match="is not a slot, group, or allowed constant"):
        _parse_template([object()], 1, constants=False, groups=True)


def test_a_call_template_that_is_not_a_list_is_refused() -> None:
    with pytest.raises(TypeError, match="must be a list"):
        _parse_template("$0", 1, constants=True, groups=False)


def test_a_call_template_slot_past_the_inputs_is_refused() -> None:
    with pytest.raises(ValueError, match=r"slot \$1 out of range \(1 inputs\)"):
        _parse_template(["$1"], 1, constants=True, groups=False)


def test_a_group_for_a_callee_without_groups_is_refused() -> None:
    with pytest.raises(TypeError, match="not stacked groups"):
        _parse_template([["$0"]], 1, constants=True, groups=False)


def test_the_join_form_refuses_a_shared_non_key_column() -> None:
    left = ak.Array([{"k": 1, "v": 2}])
    right = ak.Array([{"k": 1, "v": 3}])
    with pytest.raises(ValueError, match=r"shared non-key column\(s\) \['v'\]"):
        join_form([left, right], {"on": "k"})


def test_read_varied_refuses_a_parquet_without_a_manifest(tmp_path) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("pyarrow")
    path = str(tmp_path / "plain.parquet")
    ak.to_parquet(ak.Array({"x": [1, 2]}), path)
    with pytest.raises(GraphedError, match="not a variation-aware skim"):
        read_varied(path)
