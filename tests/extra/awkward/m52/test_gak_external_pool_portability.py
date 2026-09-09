"""`gak.apply_correction` / `gak.onnx_inference` plans must survive a STDLIB pickle.

Both recorders used to hand `Session.record_external` a local closure (`apply_correction.<locals>._fn`).
`graphed-executors` ships a plan's process callable with stdlib `pickle`, so every plan reading a
templated correction or ONNX score died on submission with

    AttributeError: Can't get local object 'apply_correction.<locals>._fn'

— `pickle.dumps(plan.process)` and `ProcessPoolExecutor.run(plan)` alike. The recorded evaluator is now
a module-level `_TemplateExternal`, which rebuilds the resource from the payload bytes via the preserve
plugin. `apply_correction` records that rebuild for EVERY backend, in-process included, so the two
cannot diverge; `onnx_inference` still keeps the user's `runner` in-process, because an inference
session carries provider/device configuration the plugin's CPU-only load would not reproduce.
Nothing else moves — same template semantics, same content descriptor, same output form.
"""

from __future__ import annotations

import ast
import json
import pickle
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import pytest
from graphed_corpus import make_events

from graphed import Array, Session
from graphed.aggregate import aggregate_plan
from graphed.awkward import AwkwardBackend, AwkwardForm, from_awkward, gak
from graphed.awkward.payloads import correctionlib_contents_hash, onnx_weights_hash
from graphed.core import Partition
from graphed.core.execution import SequentialRunner, WorkerResources
from graphed.preserve.errors import PreserveError

EVENTS = make_events(n_events=1_500, seed=52)
SYSTEMATICS = ("nominal", "up", "down")

#: Captured from `SequentialRunner` BEFORE the fix (the closure path), so the fix is held to the
#: numbers the user-supplied `cset[...].evaluate` produced, bit-for-bit.
GOLDEN_SF_SUMS = [1500.0, 1587.9500000000003, 1412.0499999999997]

_SF_EDGES = [0.0, 4.0, 5.0, 6.0, 100.0]
_SF_CONTENT = {
    "nominal": [1.0, 1.0, 1.0, 1.0],
    "up": [1.05, 1.10, 1.15, 1.20],
    "down": [0.95, 0.90, 0.85, 0.80],
}


def _correctionlib_json() -> bytes:
    cset = {
        "schema_version": 2,
        "corrections": [
            {
                "name": "btag_sf",
                "version": 1,
                "inputs": [{"name": "systematic", "type": "string"}, {"name": "x", "type": "real"}],
                "output": {"name": "sf", "type": "real"},
                "data": {
                    "nodetype": "category",
                    "input": "systematic",
                    "content": [
                        {
                            "key": syst,
                            "value": {
                                "nodetype": "binning",
                                "input": "x",
                                "edges": _SF_EDGES,
                                "content": vals,
                                "flow": "clamp",
                            },
                        }
                        for syst, vals in _SF_CONTENT.items()
                    ],
                },
            }
        ],
    }
    return json.dumps(cset, sort_keys=True).encode("utf-8")


CSET = _correctionlib_json()


def _compound_correctionlib_json() -> bytes:
    """The JEC L1L2L3 shape in miniature: two plain corrections plus a `compound_corrections` stack
    that multiplies them. The stack members differ, so their product identifies both."""

    def _level(name: str, content: list[float]) -> dict[str, Any]:
        return {
            "name": name,
            "version": 1,
            "inputs": [{"name": "pt", "type": "real"}],
            "output": {"name": "c", "type": "real"},
            "data": {
                "nodetype": "binning",
                "input": "pt",
                "edges": [0.0, 40.0, 1000.0],
                "content": content,
                "flow": "clamp",
            },
        }

    return json.dumps(
        {
            "schema_version": 2,
            "corrections": [_level("l1", [0.9, 1.1]), _level("l2", [2.0, 3.0])],
            "compound_corrections": [
                {
                    "name": "jec",
                    "inputs": [{"name": "pt", "type": "real"}],
                    "output": {"name": "c", "type": "real"},
                    "inputs_update": [],
                    "input_op": "*",
                    "output_op": "*",
                    "stack": ["l1", "l2"],
                }
            ],
        },
        sort_keys=True,
    ).encode("utf-8")


COMPOUND_CSET = _compound_correctionlib_json()
#: pt values straddling the 40 GeV edge, jagged with an empty middle list
_COMPOUND_EVENTS = ak.Array({"Jet": [[{"pt": 30.0}, {"pt": 50.0}], [], [{"pt": 80.0}]]})


def _unusable(*call: Any) -> Any:
    """An `evaluator` that fails loudly if the template path ever consults the caller's callable."""
    raise AssertionError("the template path called the user's evaluator")


def _onnx_model() -> bytes:
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper  # noqa: PLC0415

    w = numpy_helper.from_array(np.array([[0.5], [0.25]], dtype=np.float32), name="W")
    kin = helper.make_tensor_value_info("kin", TensorProto.FLOAT, [None, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, 1])
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["kin", "W"], ["y"])], "m", [kin], [y], initializer=[w]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=9)
    onnx.checker.check_model(model)
    return model.SerializeToString()  # type: ignore[no-any-return]


@dataclass
class CorpusEvents:
    """A `graphed.write.PartitionedSource` over the corpus events."""

    data: ak.Array
    part_reads: list[tuple[int, int]] = field(default_factory=list)

    def __call__(self) -> ak.Array:
        return self.data

    def partitions(self, steps_per_file: int = 1) -> tuple[Partition, ...]:
        return tuple(Partition.blind("corpus://events", "", s, steps_per_file) for s in range(steps_per_file))

    def read_partition(self, partition: Partition, columns: Any, resources: WorkerResources) -> ak.Array:
        part = partition.resolve(len(self.data))
        self.part_reads.append((part.entry_start, part.entry_stop))
        return self.data[part.entry_start : part.entry_stop]


def _events(session: Session) -> Array:
    form = AwkwardForm(ak.Array(EVENTS.layout.to_typetracer(forget_length=True)))
    return session.source("events", form=form, data=CorpusEvents(EVENTS))


def _correction_universes() -> tuple[Session, list[Array]]:
    """Three SF universes recorded through the REAL user surface: `gak.apply_correction` with a
    template whose first entry is the systematic CONSTANT."""
    import correctionlib  # noqa: PLC0415

    cset = correctionlib.CorrectionSet.from_string(CSET.decode())
    session = Session(AwkwardBackend())
    events = _events(session)
    njet = gak.num(events.Jet[events.Jet.pt > 25], axis=1)
    outs = [
        gak.apply_correction(CSET, "btag_sf", [njet], cset["btag_sf"].evaluate, args=[syst, "$0"])
        for syst in SYSTEMATICS
    ]
    return session, outs


def _legacy_scale(njet: object) -> object:
    """A picklable single-input 'correction' for the legacy (``args=None``) recording path."""
    arr = np.asarray(ak.to_numpy(ak.Array(njet)), dtype="float64")
    return ak.Array(arr * 1.05)


def _legacy_correction() -> tuple[Session, list[Array]]:
    """The legacy (``args=None``) recorder — the other end of the class. origin/main wrapped the
    callable in an ``apply_correction.<locals>.<lambda>`` that no stdlib pickle could carry; the
    recorder now hands the user's callable straight to ``record_external``."""
    path = Path(tempfile.gettempdir()) / "graphed_m52_legacy_btag_sf.json"
    path.write_bytes(CSET)  # the legacy path derives the descriptor by hashing this file at build
    session = Session(AwkwardBackend())
    events = _events(session)
    njet = gak.num(events.Jet[events.Jet.pt > 25], axis=1)
    out = gak.apply_correction(str(path), "btag_sf", [njet], _legacy_scale)
    return session, [out]


def _onnx_score() -> tuple[Session, list[Array]]:
    ort = pytest.importorskip("onnxruntime")
    payload = _onnx_model()
    ort_session = ort.InferenceSession(payload, providers=["CPUExecutionProvider"])

    def runner(x: Any) -> Any:
        out = ort_session.run(None, {"kin": np.asarray(x, dtype="float32")})[0].reshape(-1)
        return ak.Array(np.asarray(out, dtype="float64"))

    session = Session(AwkwardBackend())
    events = _events(session)
    njet = gak.num(events.Jet, axis=1)
    ht = gak.sum(events.Jet.pt, axis=1)
    score = gak.onnx_inference(payload, [njet, ht], runner, args=[["$0", "$1"]])
    return session, [score]


def _sums(vals: list[object]) -> np.ndarray:
    sums: np.ndarray = np.array([float(ak.sum(v)) for v in vals], dtype="float64")
    return sums


def _add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    total: np.ndarray = a + b
    return total


def _zeros_for(n: int) -> Any:
    return _Zeros(n)


@dataclass(frozen=True)
class _Zeros:
    """Picklable `empty` factory (a lambda would defeat the very thing under test)."""

    n: int

    def __call__(self) -> np.ndarray:
        return np.zeros(self.n)


def _plan(build: Any) -> Any:
    _session, outs = build()
    return aggregate_plan(*outs, reduce=_sums, combine=_add, empty=_zeros_for(len(outs)), steps_per_file=4)


# ---------------------------------- 1. pool portability -------------------------------------------
def test_apply_correction_plan_pickles_and_runs_in_a_process_pool() -> None:
    pytest.importorskip("correctionlib")
    ProcessPoolExecutor = pytest.importorskip("graphed_executors.local").ProcessPoolExecutor

    pickle.dumps(_plan(_correction_universes).process)  # pre-fix: AttributeError on the local closure
    seq = SequentialRunner().run(_plan(_correction_universes)).value
    par = ProcessPoolExecutor(max_workers=2).run(_plan(_correction_universes)).value
    assert np.array_equal(par, seq)  # bit-for-bit, not merely close
    assert par.tolist() == GOLDEN_SF_SUMS


def test_legacy_untemplated_apply_correction_plan_pickles_and_runs_in_a_process_pool() -> None:
    """The other end of the class: `args=None` recorded a wrapping lambda pre-fix and would not
    pickle; now a picklable evaluator yields a picklable, pool-runnable plan."""
    ProcessPoolExecutor = pytest.importorskip("graphed_executors.local").ProcessPoolExecutor

    pickle.dumps(_plan(_legacy_correction).process)  # pre-fix: Can't pickle <locals>.<lambda>
    seq = SequentialRunner().run(_plan(_legacy_correction)).value
    par = ProcessPoolExecutor(max_workers=2).run(_plan(_legacy_correction)).value
    # allclose, not array_equal: seq and pool combine the per-partition partials in different orders,
    # so a float sum differs by ~1 ULP across executors (measured 4.5e-13) — expected, not a defect.
    assert np.allclose(par, seq)
    # recompute njet from the corpus directly (not through graphed): proves _legacy_scale (x1.05) ran
    ref_njet = ak.num(EVENTS.Jet[EVENTS.Jet.pt > 25], axis=1)
    assert par.tolist() == pytest.approx([float(ak.sum(ref_njet)) * 1.05])


def test_a_recorded_kind_with_no_plugin_fails_with_a_clear_message() -> None:
    """A pickled `_TemplateExternal` whose kind has no registered plugin cannot be rebuilt in a
    worker — the reconstruction path says so, rather than an obscure NoneType deeper in the cache."""
    from graphed.awkward.functions import _TemplateExternal  # noqa: PLC0415

    orphan = _TemplateExternal("no_such_kind", b"{}", {"content_hash": "sha256:0"}, None)
    with pytest.raises(RuntimeError, match="no registered plugin"):
        orphan(ak.Array([1, 2, 3]))


def test_onnx_inference_plan_pickles_and_runs_in_a_process_pool() -> None:
    ProcessPoolExecutor = pytest.importorskip("graphed_executors.local").ProcessPoolExecutor

    pickle.dumps(_plan(_onnx_score).process)
    seq = SequentialRunner().run(_plan(_onnx_score)).value
    par = ProcessPoolExecutor(max_workers=2).run(_plan(_onnx_score)).value
    assert np.array_equal(par, seq)
    assert float(seq[0]) != 0.0  # the model actually scored something


def test_the_recorded_evaluator_round_trips_through_stdlib_pickle() -> None:
    """The evaluator itself: picklable, and the restored copy — which has NO access to the user's
    callable — evaluates to the same numbers by rebuilding the correction set from the payload."""
    pytest.importorskip("correctionlib")
    from graphed.awkward.functions import _TemplateExternal  # noqa: PLC0415  (absent pre-fix)

    session, outs = _correction_universes()
    # the "up" universe, NOT nominal: nominal's SF is all-ones, so fn(x) == restored(x) there for
    # any input even if the template were dropped. "up" varies by bin, so equality discriminates.
    fn = session._externals[outs[1].node_id][0]
    assert isinstance(fn, _TemplateExternal)
    restored = pickle.loads(pickle.dumps(fn))
    assert restored.call is None  # the live handle does not ride the pickle
    x = ak.Array([3, 5, 7])  # njet -> "up" content bins [0,4), [5,6), [6,100)
    assert ak.to_list(ak.Array(fn(x))) == [1.05, 1.15, 1.20]  # non-trivial: not identity, not ones
    assert ak.to_list(ak.Array(restored(x))) == ak.to_list(ak.Array(fn(x)))


def test_the_template_path_evaluates_in_process_through_the_plugin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`evaluator` is not what runs on the template path — the correctionlib plugin is, in-process
    exactly as in a worker. So a wrapping/scaling `evaluator` can no longer make one backend
    disagree with another, and the in-process call gets the plugin's flat-buffer evaluation rather
    than correctionlib's per-call `ak.transform` wrapper. The evaluator passed here fails loudly if
    it is still consulted; the values pin that the real correction answered instead."""
    correctionlib = pytest.importorskip("correctionlib")
    import graphed.preserve.externals.correctionlib_external as clx  # noqa: PLC0415

    session = Session(AwkwardBackend())
    events = _events(session)
    njet = gak.num(events.Jet[events.Jet.pt > 25], axis=1)
    # "up", not nominal: nominal's SF is all-ones, which any broken evaluation could also produce
    sf = gak.apply_correction(CSET, "btag_sf", [njet], _unusable, args=["up", "$0"])

    answered: list[Any] = []
    real = clx._flat_buffer_fast_path

    def spy(evaluate: Any, call: list[Any]) -> Any:
        out = real(evaluate, call)
        answered.append(out)
        return out

    monkeypatch.setattr(clx, "_flat_buffer_fast_path", spy)
    got = ak.Array(session.materialize(sf))

    # ENGAGED, not merely consulted: a None return is a silent fall back to correctionlib's wrapper
    assert len(answered) == 1 and answered[0] is not None
    ref_njet = ak.num(EVENTS.Jet[EVENTS.Jet.pt > 25], axis=1)
    ref = correctionlib.CorrectionSet.from_string(CSET.decode())["btag_sf"].evaluate("up", ref_njet)
    assert ak.to_list(got) == ak.to_list(ref)  # bit-for-bit, and not the all-ones nominal
    assert set(ak.to_list(got)) == set(_SF_CONTENT["up"])  # every bin of the varied SF was read


def test_a_compound_correction_resolves_out_of_cset_compound() -> None:
    """A *compound* correction — the JEC L1L2L3 shape — is keyed in `cset.compound`, not in `cset`,
    and correctionlib's `__getitem__` raises `IndexError` on the miss (not `KeyError`). Now that the
    plugin owns in-process evaluation, resolving only `cset[name]` would turn every compound payload
    into an opaque `IndexError: map::at: key not found` where the caller's own `evaluate` answered
    before. The two stack members have different contents, so the product pins that both applied."""
    correctionlib = pytest.importorskip("correctionlib")
    session = Session(AwkwardBackend())
    ev = from_awkward(session, "events", _COMPOUND_EVENTS)
    jec = gak.apply_correction(COMPOUND_CSET, "jec", [ev.Jet.pt], _unusable, args=["$0"])

    got = ak.Array(session.materialize(jec))
    flat_pt = ak.to_numpy(ak.flatten(_COMPOUND_EVENTS.Jet.pt))
    ref = correctionlib.CorrectionSet.from_string(COMPOUND_CSET.decode()).compound["jec"].evaluate(flat_pt)
    assert ak.num(got, axis=1).tolist() == [2, 0, 1]  # jagged structure survives the compound call
    assert ak.to_list(ak.flatten(got)) == list(ref)  # bit-for-bit against correctionlib itself
    assert set(ak.to_list(ak.flatten(got))) == {0.9 * 2.0, 1.1 * 3.0}  # neither factor on its own


def test_an_unknown_correction_name_names_both_key_sets() -> None:
    """The miss is an `IndexError: map::at: key not found` out of pybind11 otherwise — it names
    neither the correction asked for nor what the payload does hold, plain or compound."""
    pytest.importorskip("correctionlib")
    session = Session(AwkwardBackend())
    ev = from_awkward(session, "events", _COMPOUND_EVENTS)
    sf = gak.apply_correction(COMPOUND_CSET, "l3", [ev.Jet.pt], _unusable, args=["$0"])

    with pytest.raises(PreserveError, match=r"no correction named 'l3'.*\['l1', 'l2'\].*\['jec'\]"):
        session.materialize(sf)


def test_a_missing_framework_is_attributed_to_the_plugin() -> None:
    """Rebuilding from the payload bytes means the framework must be importable wherever evaluation
    happens. A bare `ImportError: No module named 'correctionlib'` out of a worker names neither the
    External that needed it nor the payload kind; the branded error names both."""
    from graphed.preserve.externals._base import _PluginEvaluator  # noqa: PLC0415
    from graphed.preserve.externals.correctionlib_external import CORRECTIONLIB_PLUGIN  # noqa: PLC0415

    def _no_lib(payload: bytes, params: Any) -> Any:
        raise ImportError("No module named 'correctionlib'")

    def _corrupt(payload: bytes, params: Any) -> Any:
        raise RuntimeError("payload is not a correction set")

    # content_hashes no real load ever cached, so `_RESOURCE_CACHE` cannot answer instead
    def _params(tag: str) -> dict[str, Any]:
        return {"name": "btag_sf", "content_hash": f"sha256:{tag}", "args": '["up", "$0"]'}

    with pytest.raises(PreserveError, match=r"'correctionlib' plugin.*correctionlib is not importable"):
        _PluginEvaluator(replace(CORRECTIONLIB_PLUGIN, load=_no_lib), CSET, _params("no-lib"))(ak.Array([3]))

    # only a missing import is a missing framework: any other load failure keeps its own type
    with pytest.raises(RuntimeError, match="not a correction set"):
        _PluginEvaluator(replace(CORRECTIONLIB_PLUGIN, load=_corrupt), CSET, _params("corrupt"))(
            ak.Array([3])
        )


# ---------------------------------- 2. numeric identity -------------------------------------------
def test_sequential_numbers_match_the_pre_fix_golden() -> None:
    """Routing in-process evaluation through the plugin changed no number: the SFs the user's
    `cset[...].evaluate` produced pre-fix are still the SFs that come out."""
    pytest.importorskip("correctionlib")
    got = SequentialRunner().run(_plan(_correction_universes)).value
    assert got.tolist() == GOLDEN_SF_SUMS
    assert got[1] > got[0] > got[2]  # up/down bracket nominal: the SF was really read per universe


# ---------------------------------- 3. preservation intact ----------------------------------------
def test_content_identity_and_plan_bytes_are_unchanged_and_deterministic() -> None:
    pytest.importorskip("correctionlib")
    session, outs = _correction_universes()
    nodes = {n["id"]: n for n in session._store.nodes()}
    for out, syst in zip(outs, SYSTEMATICS, strict=True):
        node = nodes[out.node_id]
        assert node["descriptor"]["kind"] == "correctionlib"
        assert node["descriptor"]["content_hash"] == correctionlib_contents_hash(CSET)
        assert "path" not in node["params"]
        assert json.loads(str(node["params"]["args"])) == [syst, "$0"]
    assert bytes(_plan(_correction_universes).process.ir) == bytes(_plan(_correction_universes).process.ir)


def test_onnx_content_identity_is_unchanged() -> None:
    pytest.importorskip("onnxruntime")
    session, (score,) = _onnx_score()
    node = next(n for n in session._store.nodes() if n["id"] == score.node_id)
    assert node["descriptor"]["kind"] == "onnx_model"
    assert node["descriptor"]["content_hash"] == onnx_weights_hash(_onnx_model())
    assert "path" not in node["params"]


# ---------------------------------- 4. no cloudpickle ---------------------------------------------
def _imports_cloudpickle(module: Any) -> bool:
    tree = ast.parse(Path(str(module.__file__)).read_text())
    return any(
        (isinstance(n, ast.Import) and any(a.name.split(".")[0] == "cloudpickle" for a in n.names))
        or (isinstance(n, ast.ImportFrom) and (n.module or "").split(".")[0] == "cloudpickle")
        for n in ast.walk(tree)
    )


def test_the_recording_path_does_not_reach_for_cloudpickle() -> None:
    """§A.3.1: a correctionlib/ONNX payload is preservable, so it must never need by-value pickling."""
    from graphed.awkward import functions  # noqa: PLC0415
    from graphed.preserve.externals import _base  # noqa: PLC0415

    assert _imports_cloudpickle(_base)  # positive control: a module that DOES import it
    assert not _imports_cloudpickle(functions)
