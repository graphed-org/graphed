"""One rule at the deferred/eager boundary: a conversion refuses, naming the explicit choices."""

from __future__ import annotations

import pickle
from collections.abc import Callable
from typing import Any

import awkward as ak
import numpy as np
import pytest

import graphed
import graphed.numpy as gnp
from graphed import Session
from graphed.awkward import AwkwardBackend, AwkwardForm, from_awkward, gak
from graphed.context import EventContext
from graphed.errors import DeferredTypeError, GraphedTypeError, MissingFieldError
from graphed.numpy import NumpyBackend


def _unread() -> ak.Array:
    raise AssertionError("a refused conversion read the source")


def _events(behavior: dict[Any, Any] | None = None) -> Any:
    data = ak.Array({"Jet_pt": [[1.0, 2.0], []], "MET_pt": [1.0, 2.0]})
    form = AwkwardForm(ak.Array(data.layout.to_typetracer(forget_length=True)))
    return Session(AwkwardBackend(behavior=behavior)).source("Events", form=form, data=_unread)


def _plt_hist(x: Any) -> object:
    plt = pytest.importorskip("matplotlib.pyplot")
    return plt.hist(x)


def _hist_fill(x: Any) -> object:
    hist = pytest.importorskip("hist")
    return hist.Hist.new.Reg(10, 0, 100, name="pt").Double().fill(pt=x)


CONVERSIONS: dict[str, Callable[[Any], object]] = {
    "ak.to_list": ak.to_list,
    "ak.to_numpy": ak.to_numpy,
    "ak.Array": ak.Array,
    "np.asarray": np.asarray,
    "np.histogram": np.histogram,
    ".tolist": lambda x: x.tolist(),
    ".to_list": lambda x: x.to_list(),
    ".to_numpy": lambda x: x.to_numpy(),
    ".show": lambda x: x.show(),
    "float": lambda x: float(gak.sum(x, axis=None)),
    "int": lambda x: int(gak.sum(x, axis=None)),
    "len": len,
    "list": list,
    "plt.hist": _plt_hist,
    "hist.Hist.fill": _hist_fill,
    ".compute": lambda x: x.compute(),
}
MIXING: dict[str, Callable[[Any], object]] = {
    "numpy": lambda x: x * np.ones(2),
    "reflected numpy": lambda x: np.ones(2) * x,
    "awkward": lambda x: x + ak.Array([1.0, 2.0]),
    "list": lambda x: x * [1.0, 2.0],
    "gak argument": lambda x: gak.where(x > 0, np.ones(2), x),
    "with_field": lambda x: gak.with_field(x, np.ones(2), "w"),
}


@pytest.mark.parametrize("convert", CONVERSIONS.values(), ids=CONVERSIONS)
def test_a_conversion_refuses_at_the_users_line_and_reads_nothing(
    convert: Callable[[Any], object],
) -> None:
    x = _events().MET_pt
    with pytest.raises(GraphedTypeError) as refused:
        convert(x)
    text = str(refused.value)
    assert "graphed.materialize" in text
    assert not [word for word in ("np.where", "typetracer", "nplike") if word in text]
    assert refused.value.provenance.filename == __file__


def test_a_refused_ak_function_records_nothing() -> None:
    x = _events().MET_pt
    nodes = x.session.node_count()
    with pytest.raises(GraphedTypeError):
        ak.to_list(x)
    assert x.session.node_count() == nodes


@pytest.mark.parametrize("mix", MIXING.values(), ids=MIXING)
def test_mixing_with_an_eager_array_names_both_fixes_at_the_users_line(
    mix: Callable[[Any], object],
) -> None:
    with pytest.raises(GraphedTypeError) as refused:
        mix(_events().MET_pt)
    assert "from_awkward" in str(refused.value)
    assert "graphed.materialize" in str(refused.value)
    assert refused.value.provenance.filename == __file__


@pytest.mark.parametrize(
    "sniff",
    [len, float, int, iter, lambda x: x * np.ones(2)],
    ids=["len", "float", "int", "iter", "operand"],
)
def test_a_protocol_sniffer_still_reads_a_type_error(sniff: Callable[[Any], object]) -> None:
    with pytest.raises(TypeError):
        sniff(_events().MET_pt)


def test_asarray_is_no_type_error_a_library_could_fall_back_on() -> None:
    with pytest.raises(GraphedTypeError) as refused:
        np.asarray(_events().MET_pt)
    assert not isinstance(refused.value, TypeError)
    assert "hist.graphed.Hist" in str(refused.value)


@pytest.mark.parametrize(
    ("convert", "kind"),
    [
        (len, DeferredTypeError),
        (bool, GraphedTypeError),
        (np.asarray, GraphedTypeError),
        (ak.to_list, GraphedTypeError),
    ],
    ids=["len", "bool", "asarray", "ak.to_list"],
)
def test_a_refusal_crosses_a_process_boundary(
    convert: Callable[[Any], object], kind: type[GraphedTypeError]
) -> None:
    with pytest.raises(kind) as refused:
        convert(_events().MET_pt)
    back = pickle.loads(pickle.dumps(refused.value))
    assert type(back) is kind
    assert str(back) == str(refused.value)


def test_bool_keeps_its_elementwise_advice() -> None:
    with pytest.raises(GraphedTypeError, match=r"np\.where.*gak\.where") as refused:
        bool(_events().MET_pt)
    assert "graphed.materialize" in str(refused.value)


def test_len_names_how_to_get_the_length() -> None:
    with pytest.raises(TypeError, match=r"graphed\.materialize\(gak\.num\(x, axis=0\)\)"):
        len(_events())


def test_numpy_scalars_stay_operands() -> None:
    x = from_awkward("x", ak.Array([1.0, 2.0]))
    for scalar in (np.float64(2.0), np.array(2.0), np.int32(2), 2.0):
        assert ak.to_list(graphed.materialize(x * scalar)) == [2.0, 4.0]


@pytest.mark.parametrize("operand", [object(), b"ab", {"a": 1}, {1.0}], ids=repr)
def test_what_is_no_array_keeps_the_scalar_refusal(operand: object) -> None:
    with pytest.raises(TypeError, match="unsupported scalar operand") as refused:
        _events().MET_pt * operand
    assert not isinstance(refused.value, GraphedTypeError)


def test_a_deferred_value_is_no_scalar_operand() -> None:
    ev = from_awkward("ev", ak.Array({"shape": [1.0, 2.0], "x": [3.0, 4.0]}))
    with pytest.raises(TypeError, match="unsupported scalar operand"):
        gak.fill_none(ev.x, ev)


def test_a_field_named_like_a_verb_reads_the_field() -> None:
    fields = {"compute": [1.0, 2.0], "sum": [3.0, 4.0], "to_list": [5.0, 6.0]}
    ev = from_awkward("ev", ak.Array(fields))
    assert ak.to_list(graphed.materialize(ev.compute)) == [1.0, 2.0]
    assert ak.to_list(graphed.materialize(ev.sum)) == [3.0, 4.0]
    assert ak.to_list(graphed.materialize(ev.to_list)) == [5.0, 6.0]


def test_a_missing_compute_still_reads_as_absent() -> None:
    x = _events().MET_pt
    assert not hasattr(x, "compute")
    with pytest.raises(
        MissingFieldError, match=r"compute\(\): needs values.*no field named 'compute'"
    ):
        _ = x.compute


def test_another_missing_field_gets_no_compute_advice() -> None:
    with pytest.raises(MissingFieldError) as refused:
        _ = _events().nope
    assert "compute" not in str(refused.value)


def test_a_conversion_reads_as_absent_to_a_sniffing_library() -> None:
    assert not hasattr(_events().MET_pt, "to_numpy")


def test_a_behaviors_own_method_of_a_conversions_name_still_records() -> None:
    class Jets(ak.Array):  # type: ignore[misc]
        def to_list(self, scale: float) -> Any:
            return self.pt * scale

    behavior: dict[Any, Any] = {("*", "Jet"): Jets}
    data = ak.with_name(ak.Array([{"pt": 1.0}, {"pt": 2.0}]), "Jet")
    jets = from_awkward("j", data, session=Session(AwkwardBackend(behavior=behavior)))
    assert ak.to_list(graphed.materialize(jets.to_list(2.0))) == [2.0, 4.0]


def test_a_record_refuses_like_an_array() -> None:
    ev = _events()
    with pytest.raises(GraphedTypeError, match="needs values"):
        ev[0].to_list()


def test_gak_to_list_evaluates_and_is_deprecated() -> None:
    x = from_awkward("x", ak.Array([1.0, 2.0]))
    with pytest.warns(
        DeprecationWarning, match=r"ak\.to_list\(graphed\.materialize\(x\)\)"
    ) as caught:
        assert gak.to_list(x) == [1.0, 2.0]
    assert caught[0].filename == __file__


def test_the_deprecation_pointer_works_on_a_scalar() -> None:
    total = gak.sum(from_awkward("x", ak.Array([1.0, 2.0])), axis=None)
    assert ak.to_list(graphed.materialize(total)) == 3.0


@pytest.mark.parametrize(
    "convert",
    [
        len,
        float,
        int,
        list,
        np.asarray,
        lambda v: v.to_numpy(),
        ak.to_list,
        lambda v: v.compute(),
    ],
    ids=["len", "float", "int", "list", "asarray", "to_numpy", "ak.to_list", "compute"],
)
def test_a_varied_refuses_by_the_same_rule(convert: Callable[[Any], object]) -> None:
    x = from_awkward("x", ak.Array([1.0, 2.0]))
    with pytest.raises(GraphedTypeError, match=r"graphed\.materialize") as refused:
        convert(graphed.vary(x, "sf", up=x * 2, down=x / 2))
    assert refused.value.provenance.filename == __file__


def test_a_numpy_backed_array_refuses_by_the_same_rule() -> None:
    x = gnp.from_array("x", np.arange(3.0))
    for convert in (len, float, int):
        with pytest.raises(TypeError, match=r"graphed\.materialize"):
            convert(x)
    with pytest.raises(TypeError, match=r"graphed\.numpy\.from_array"):
        x * np.ones(3)


@pytest.mark.parametrize("verb", ["to_list", "tolist", "to_numpy", "show", "compute"])
@pytest.mark.parametrize(
    "make",
    [
        lambda: gnp.from_array("x", np.arange(3.0), session=Session(NumpyBackend())),
        lambda: gak.sum(from_awkward("x", ak.Array([1.0, 2.0])), axis=None),
    ],
    ids=["numpy-backed", "awkward-scalar"],
)
def test_a_conversion_verb_refuses_on_any_backend_and_shape(
    make: Callable[[], Any], verb: str
) -> None:
    value = make()
    nodes = value.session.node_count()
    with pytest.raises(GraphedTypeError, match=r"graphed\.materialize") as refused:
        getattr(value, verb)()
    assert refused.value.provenance.filename == __file__
    assert value.session.node_count() == nodes


@pytest.mark.parametrize(
    "convert",
    [len, float, int, bool, np.asarray],
    ids=["len", "float", "int", "bool", "asarray"],
)
def test_an_event_context_refuses_by_the_same_rule(convert: Callable[[Any], object]) -> None:
    session = Session(NumpyBackend())
    pt = gnp.from_array("pt", np.arange(3.0), session=session)
    with pytest.raises(GraphedTypeError, match=r"graphed\.materialize"):
        convert(EventContext(session, pt, collections={"pt": pt}))
