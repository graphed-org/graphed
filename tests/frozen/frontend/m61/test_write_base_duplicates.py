"""m61 — `file_bases` refuses a key listed twice: a worker derives its part index from the
partition alone (R15.9), so two partitions of one file and step would overwrite each other."""

from __future__ import annotations

import pytest

from graphed import write as gw


def test_file_bases_refuses_duplicate_keys() -> None:
    assert gw.file_bases(["a", "b"], 3) == {"a": 0, "b": 3}
    with pytest.raises(ValueError, match="duplicate input 'a'"):
        gw.file_bases(["a", "b", "a"], 3)
    with pytest.raises(ValueError, match=r"duplicate input \('a', 't'\)"):
        gw.file_bases([("a", "t"), ("a", "t")], 2)
    assert gw.file_bases([("a", "t1"), ("a", "t2")], 2) == {("a", "t1"): 0, ("a", "t2"): 2}
