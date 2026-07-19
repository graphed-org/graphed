"""M40 iteration-3 — a NON-INNER join's KEY (``on``) columns must stay NON-OPTION and agree across
backends (plan §3.3: the ``on`` key on a miss row is COALESCED from the present side -- SQL COALESCE --
so it is NEVER null; only the ABSENT side's NON-key fields become option-typed).

The iteration-2 non-inner suite pins the RELATIONAL VALUES (``test_noninner_backend_independence.py``:
the merged rows equal ``pandas.merge`` with the orphan key coalesced) but NOT the TYPE of the key
columns, and not that the RECORDED form agrees with the MATERIALIZED form. Two gaps, both frozen here
through the public recorded surface (``graphed.join`` + ``Session.form``/``Session.materialize``):

  1. **op_form-vs-materialized (awkward).** ``Session.form`` declares the key columns non-option (e.g.
     ``event: int64``), but MATERIALIZING the same recorded left/right/outer join yields an OPTION key
     (``event: ?int64``) — the recorded form and the produced block disagree on the key's type. Under
     §3.3 the key is coalesced and never null, so its type MUST be non-option AND MUST equal what
     op_form declared.
  2. **Cross-backend key-optionality parity.** The SAME recorded join must give the key columns the
     SAME optionality on BOTH backends. On HEAD they DIVERGE: awkward materializes the key as an option
     (``?int64``), numpy as a plain non-option ``int64`` (no ``__valid_<key>__`` companion) — so a
     downstream consumer sees a different key type depending on the backend.

Optionality is an INDEPENDENT property of the block/form, not read from the join kernel: awkward option
== ``?`` in the type string; numpy option == a ``__valid_<field>__`` companion mask column (the plan's
§3.3 null carrier) or a genuinely-masked field. Expected values follow only from SQL/pandas COALESCE
semantics (key never null => non-option), never from the code under test.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest
from shuffle_backends import ON, REAL_BACKENDS, skim_tables

import graphed
from graphed import Session

_CASES = {c.name: c for c in REAL_BACKENDS}


def _leaf_type(arr: object) -> str:
    """The per-column leaf type without the length/``##`` prefix: ``11 * ?int64`` -> ``?int64``."""
    return str(arr.type).split("*")[-1].strip()  # type: ignore[attr-defined]


def _numpy_field_names(block: object) -> list[str]:
    if isinstance(block, Mapping):
        return list(block.keys())
    return list(np.ma.asanyarray(block).dtype.names or ())


def _awkward_key_is_option(block: object, key: str) -> bool:
    return "?" in _leaf_type(block[key])  # type: ignore[index]


def _numpy_key_is_option(block: object, key: str) -> bool:
    """numpy option carrier (§3.3): a ``__valid_<key>__`` companion column, or a genuinely masked
    field. A plain non-option key has neither."""
    names = _numpy_field_names(block)
    if f"__valid_{key}__" in names:
        return True
    col = block[key] if isinstance(block, Mapping) else np.ma.asanyarray(block)[key]  # type: ignore[index]
    return bool(np.ma.isMaskedArray(col) and np.ma.getmaskarray(col).any())


def _key_is_option(case_name: str, block: object, key: str) -> bool:
    return _awkward_key_is_option(block, key) if case_name == "awkward" else _numpy_key_is_option(block, key)


@pytest.mark.parametrize("how", ["left", "right", "outer"])
def test_awkward_noninner_key_form_matches_materialized_and_is_nonoption(how: str) -> None:
    """(gap 1) On the awkward backend the RECORDED op_form and the MATERIALIZED block must agree on the
    key columns' type, and that type must be NON-option (keys are coalesced). Fails on HEAD: op_form
    declares ``event: int64`` but materialize produces ``event: ?int64`` (an option key)."""
    case = _CASES["awkward"]
    left_cols, right_cols = skim_tables()
    s = Session(case.make_backend())
    left = case.make_source(s, "left", left_cols)
    right = case.make_source(s, "right", right_cols)
    j = graphed.join(left, right, on=ON, how=how)

    form_tt = s.form(j).tt  # the typetracer array behind the recorded op_form
    block = s.materialize(j)
    for k in ON:
        declared = _leaf_type(form_tt[k])
        produced = _leaf_type(block[k])
        assert "?" not in produced, (
            f"how={how}: materialized key column {k!r} is OPTION ({produced}); a coalesced join key is "
            f"never null and must be non-option"
        )
        assert "?" not in declared, f"how={how}: op_form key column {k!r} is unexpectedly option ({declared})"
        assert declared == produced, (
            f"how={how}: recorded op_form and materialized block disagree on key {k!r}: "
            f"op_form={declared} vs materialized={produced}"
        )


@pytest.mark.parametrize("how", ["left", "right", "outer"])
def test_noninner_key_optionality_agrees_across_backends(how: str) -> None:
    """(gap 2) The SAME recorded non-inner join must give the key columns the SAME optionality on the
    awkward and numpy backends, and both must be NON-option. Fails on HEAD: awkward key is option,
    numpy key is non-option -> divergence."""
    left_cols, right_cols = skim_tables()
    optionality: dict[str, dict[str, bool]] = {}
    for name in ("awkward", "numpy"):
        case = _CASES[name]
        s = Session(case.make_backend())
        left = case.make_source(s, "left", left_cols)
        right = case.make_source(s, "right", right_cols)
        block = s.materialize(graphed.join(left, right, on=ON, how=how))
        optionality[name] = {k: _key_is_option(name, block, k) for k in ON}

    for k in ON:
        ak_opt, np_opt = optionality["awkward"][k], optionality["numpy"][k]
        assert not ak_opt, f"how={how}: awkward key {k!r} is OPTION; a coalesced join key must be non-option"
        assert not np_opt, f"how={how}: numpy key {k!r} is OPTION; a coalesced join key must be non-option"
        assert ak_opt == np_opt, (
            f"how={how}: key {k!r} optionality DIVERGES across backends: awkward option={ak_opt}, "
            f"numpy option={np_opt}"
        )
