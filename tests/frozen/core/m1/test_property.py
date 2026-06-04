"""Property: after any op sequence, node_count == number of distinct structural keys (M1)."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

import graphed_core as gc

# Restrict params to int/str/bool here so the Python-side reference key exactly mirrors Rust
# without float-canonicalization concerns (floats are covered in test_interning).
_param_val = st.one_of(st.integers(-5, 5), st.text(min_size=0, max_size=3), st.booleans())
_params = st.dictionaries(st.sampled_from(["a", "b", "c"]), _param_val, max_size=3)
_names = st.sampled_from(["pt", "eta", "phi", "cut", "add", "mul"])


@st.composite
def _program(draw: st.DrawFn) -> list[tuple[str, str, dict[str, object]]]:
    n = draw(st.integers(min_value=1, max_value=40))
    ops: list[tuple[str, str, dict[str, object]]] = []
    for _ in range(n):
        kind = draw(st.sampled_from(["source", "op"]))
        ops.append((kind, draw(_names), draw(_params)))
    return ops


def _key(kind: str, name: str, params: dict[str, object], inputs: tuple[int, ...]) -> tuple[object, ...]:
    # bool is a subclass of int in Python; tag the type to mirror Rust's typed ParamValue
    items = tuple(sorted((k, type(v).__name__, v) for k, v in params.items()))
    return (kind, name, items, inputs)


@settings(max_examples=150, deadline=None)
@given(program=_program())
def test_node_count_equals_distinct_keys(program: list[tuple[str, str, dict[str, object]]]) -> None:
    store = gc.GraphStore()
    keys: set[tuple[object, ...]] = set()
    made: list[int] = []
    for kind, name, params in program:
        if kind == "source" or not made:
            nid = store.add_source(name, params)
            keys.add(_key("source", name, params, ()))
        else:
            # reference up to two previously-made nodes deterministically
            inputs = (made[len(made) // 2],) if len(made) == 1 else (made[0], made[-1])
            nid = store.add_op(name, list(inputs), params)
            keys.add(_key("op", name, params, tuple(inputs)))
        made.append(nid)
    assert store.node_count() == len(keys)
