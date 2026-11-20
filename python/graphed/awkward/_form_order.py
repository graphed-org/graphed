"""Forms with every parameter mapping in key order, for the encoders that hash or ship them (#238).

``ak.concatenate`` returns an array's parameters in set order, so their order depends on
``PYTHONHASHSEED``; a form serialised from such an array names the same type in different bytes.
"""

from __future__ import annotations

import json
from typing import Any

import awkward as ak


def ordered_form(form: ak.forms.Form) -> ak.forms.Form:
    """Return ``form`` with the parameters of every node in key order."""
    return ak.forms.from_dict(_ordered(form.to_dict()))


def ordered_array(array: ak.Array) -> ak.Array:
    """Return ``array`` over the same buffers, its form's parameters in key order."""
    if _in_order(array.layout.form):
        return array
    form, length, container = ak.to_buffers(array)
    # `array.attrs` would fill an unset attrs with {} on the caller's array
    return ak.from_buffers(
        ordered_form(form), length, container, behavior=array.behavior, attrs=array._attrs
    )


def ordered_record(record: ak.Record) -> ak.Record:
    """Return ``record`` over a packed copy of its row, its form's parameters in key order."""
    if _in_order(record.layout.array.form):
        return record
    packed = ak.to_packed(record, highlevel=False)
    array = ordered_array(ak.Array(packed.array, behavior=record.behavior, attrs=record._attrs))
    return ak.Record(
        ak.record.Record(array.layout, packed.at), behavior=record.behavior, attrs=record._attrs
    )


def _in_order(form: ak.forms.Form) -> bool:
    # rebuilding an array costs more than pickling a small one, and most are already in order
    node = form.to_dict()
    return json.dumps(node) == json.dumps(_ordered(node))


def _ordered(node: Any) -> Any:
    if isinstance(node, list):
        return [_ordered(item) for item in node]
    if not isinstance(node, dict):
        return node
    # a form dict lists a record's fields and contents, so `parameters` is always a node's own
    return {
        key: dict(sorted(value.items())) if key == "parameters" else _ordered(value)
        for key, value in node.items()
    }
