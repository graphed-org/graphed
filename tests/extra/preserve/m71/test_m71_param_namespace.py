"""User plugin params named like the form directives stay plain params, as in 0.0.6."""

import hashlib
from typing import Any

import awkward as ak
import numpy as np
import pytest

from graphed import GraphedTypeError, Session
from graphed.awkward import AwkwardBackend, from_awkward
from graphed.preserve.externals import ExternalPlugin, record_external


def _plugin(output_dtype: object = None) -> ExternalPlugin:
    return ExternalPlugin(
        kind=f"m71-namespace-{output_dtype}",
        content_hash=lambda b: hashlib.sha256(b).hexdigest(),
        evaluate=lambda resource, params, inputs: inputs[0],
        samples=lambda: [],
        framework="x",
        output_dtype=output_dtype,
    )


def _x() -> tuple[Session, Any]:
    s = Session(AwkwardBackend())
    return s, from_awkward(s, "e", ak.zip({"x": np.arange(3, dtype=np.float32)}, depth_limit=1)).x


def _params(s: Session, a: Any) -> dict[str, Any]:
    return dict(next(n for n in s._store.nodes() if n["id"] == a.node_id)["params"])


@pytest.mark.parametrize(
    "user", [{"output_type": "probabilities"}, {"output_type": "f4"}, {"output_dtype": "int8"}]
)
def test_directive_named_params_leave_form_and_params(user: dict[str, str]) -> None:
    s, x = _x()
    a = record_external(s, _plugin(), b"m", [x], params=user)
    assert s.form(a).describe() == "## * float32"
    assert {k: _params(s, a)[k] for k in user} == user


def test_plugin_output_dtype_wins_over_a_user_param() -> None:
    s, x = _x()
    a = record_external(s, _plugin("float64"), b"m", [x], params={"output_dtype": "int8"})
    assert s.form(a).describe() == "## * float64"


def test_declaration_colliding_with_a_user_param_is_refused() -> None:
    s, x = _x()
    with pytest.raises(GraphedTypeError, match="collides"):
        record_external(s, _plugin(), b"m", [x], params={"output_type": "p"}, output_type="bool")
