"""`gak.apply_correction` / `gak.onnx_inference` plans must survive a STDLIB pickle.

Both recorders used to hand `Session.record_external` a local closure (`apply_correction.<locals>._fn`).
`graphed-executors` ships a plan's process callable with stdlib `pickle`, so every plan reading a
templated correction or ONNX score died on submission with

    AttributeError: Can't get local object 'apply_correction.<locals>._fn'

— `pickle.dumps(plan.process)` and `ProcessPoolExecutor.run(plan)` alike. The recorded evaluator is now
a module-level `_TemplateExternal`: in-process it still calls the user's `evaluator`/`runner` through the
template, and across a pickle it rebuilds the resource from the payload bytes via the preserve plugin.
Nothing else moves — same template semantics, same content descriptor, same output form.
"""

from __future__ import annotations

import ast
import json
import pickle
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import pytest
from graphed_corpus import make_events

from graphed import Array, Session
from graphed.aggregate import aggregate_plan
from graphed.awkward import AwkwardBackend, AwkwardForm, gak
from graphed.awkward.payloads import correctionlib_contents_hash, onnx_weights_hash
from graphed.core import Partition
from graphed.core.execution import SequentialRunner, WorkerResources

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
    return np.array([float(ak.sum(v)) for v in vals], dtype="float64")


def _add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a + b


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


# ---------------------------------- 2. numeric identity -------------------------------------------
def test_sequential_numbers_match_the_pre_fix_golden() -> None:
    """In-process the user's `cset[...].evaluate` is still what runs, through the same template."""
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
