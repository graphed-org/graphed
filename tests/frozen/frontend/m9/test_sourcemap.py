"""M9 support — `Session.sourcemap()` exposes per-node user-source provenance for `inspect`."""

from __future__ import annotations

from backends import ListBackend, from_list

from graphed import Session


def test_sourcemap_has_an_entry_per_recorded_node_pointing_at_this_file() -> None:
    s = Session(ListBackend())
    a = from_list(s, "a", [1, 2, 3])
    b = from_list(s, "b", [4, 5, 6])
    c = a + b

    sm = s.sourcemap()
    assert set(sm) == {a.node_id, b.node_id, c.node_id}
    entry = sm[c.node_id]
    assert entry["filename"].endswith("test_sourcemap.py")
    assert isinstance(entry["lineno"], int) and entry["lineno"] > 0
    assert "a + b" in str(entry["source"])  # the sub-expression text, for a faithful inspect view
