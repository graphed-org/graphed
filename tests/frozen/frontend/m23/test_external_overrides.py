"""M23 (graphed): caller-supplied External descriptors and forms (the graphed-histogram seam).

A package recording its own External family (histogram fills, like M3's correctionlib/ONNX nodes)
supplies the `PayloadDescriptor` and output form ITSELF — `record_external(descriptor=, form=)`
skips backend consultation entirely, so backends stay free of domain content (§A.4). Everything
downstream is the existing machinery: hash-consed identity, materialize calling the evaluator
with every input, `evaluate_ir` resolving by content hash.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from m10_toy import CountingListBackend, from_list

from graphed import GraphedTypeError, Session, compile_ir, evaluate_ir
from graphed.core import PayloadDescriptor


@dataclass(frozen=True)
class OpaqueForm:
    label: str

    def describe(self) -> str:
        return self.label


def _descriptor(tag: str = "h1") -> PayloadDescriptor:
    return PayloadDescriptor(
        kind="histogram",
        content_hash=f"sha256:{tag}",
        framework="boost_histogram",
        version="1",
        io_schema="uhi",
        preprocessing_ref=None,
    )


def _weighted_sum(values: object, weights: object) -> float:
    assert isinstance(values, list) and isinstance(weights, list)
    return float(sum(v * w for v, w in zip(values, weights, strict=True)))


def test_without_overrides_unknown_external_ops_still_fail() -> None:
    s = Session(CountingListBackend())
    x = from_list(s, "x", [1.0, 2.0])
    with pytest.raises(GraphedTypeError):  # the toy backend has no "histogram" payload
        s.record_external("histogram", _weighted_sum, [x])


def test_overrides_record_without_consulting_the_backend() -> None:
    s = Session(CountingListBackend())
    x = from_list(s, "x", [1.0, 2.0, 3.0])
    w = from_list(s, "w", [2.0, 2.0, 2.0])
    node = s.record_external(
        "histogram",
        _weighted_sum,
        [x, w],
        {"spec": "reg:3"},
        descriptor=_descriptor(),
        form=OpaqueForm("histogram[reg:3]"),
    )
    assert s.form(node).describe() == "histogram[reg:3]"  # the caller's form, verbatim
    assert s.materialize(node) == 12.0  # the evaluator gets EVERY input


def test_compiled_path_resolves_by_the_callers_content_hash() -> None:
    s = Session(CountingListBackend())
    x = from_list(s, "x", [1.0, 2.0, 3.0])
    w = from_list(s, "w", [1.0, 0.0, 1.0])
    node = s.record_external(
        "histogram",
        _weighted_sum,
        [x, w],
        {"spec": "reg:3"},
        descriptor=_descriptor("abc"),
        form=OpaqueForm("h"),
    )
    compiled = compile_ir(s, node)
    got = evaluate_ir(
        compiled,
        CountingListBackend(),
        {"x": [1.0, 2.0, 3.0], "w": [1.0, 0.0, 1.0]},
        externals={"sha256:abc": _weighted_sum},
    )
    assert got == [4.0]
    with pytest.raises(Exception, match="sha256:abc"):  # unresolved hash still fails loudly
        evaluate_ir(compiled, CountingListBackend(), {"x": [1.0], "w": [1.0]})


def test_descriptor_identity_is_hash_consed() -> None:
    s = Session(CountingListBackend())
    x = from_list(s, "x", [1.0])
    a = s.record_external(
        "histogram",
        _weighted_sum,
        [x, x],
        {"spec": "reg:3"},
        descriptor=_descriptor("same"),
        form=OpaqueForm("h"),
    )
    b = s.record_external(
        "histogram",
        _weighted_sum,
        [x, x],
        {"spec": "reg:3"},
        descriptor=_descriptor("same"),
        form=OpaqueForm("h"),
    )
    c = s.record_external(
        "histogram",
        _weighted_sum,
        [x, x],
        {"spec": "reg:9"},
        descriptor=_descriptor("same"),
        form=OpaqueForm("h"),
    )
    assert a.node_id == b.node_id  # same descriptor + params + inputs -> one interned node
    assert a.node_id != c.node_id  # different params -> different node
