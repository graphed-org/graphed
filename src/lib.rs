//! graphed-core: thread-safe interned graph IR (Rust+PyO3), plans M1 + M4.
//!
//! The graph lives in Rust; this crate does not (and must not) depend on awkward. The M4 optimizer
//! (DCE/CSE + equality-saturation stage fusion via egg) lives in `optimizer`, behind a
//! `RewriteEngine` trait so the engine is swappable (Phase-2 egglog).

mod node;
mod optimizer;
mod param;
mod serialize;
mod store;

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict};

use std::sync::{Mutex, MutexGuard};

use crate::node::{parse_op_token, NodeId, PayloadDescriptor};
use crate::optimizer::{EggEngine, FusionMode, IncrementalReducer, ReductionReport};
use crate::param::{ParamMap, ParamValue};
use crate::serialize::{DecodeError, EncodeError};
use crate::store::{BadNodeId, GraphStore};

fn fusion_mode(maximal_fusion: bool) -> FusionMode {
    if maximal_fusion {
        FusionMode::Maximal
    } else {
        FusionMode::SingleUse
    }
}

impl From<BadNodeId> for PyErr {
    fn from(e: BadNodeId) -> Self {
        PyValueError::new_err(e.to_string())
    }
}

impl From<DecodeError> for PyErr {
    fn from(e: DecodeError) -> Self {
        PyValueError::new_err(e.to_string())
    }
}

impl From<EncodeError> for PyErr {
    fn from(e: EncodeError) -> Self {
        PyValueError::new_err(e.to_string())
    }
}

fn paramvalue_from_py(v: &Bound<'_, PyAny>) -> PyResult<ParamValue> {
    // bool must be checked before int (Python bool is a subclass of int).
    if v.is_instance_of::<PyBool>() {
        Ok(ParamValue::Bool(v.extract()?))
    } else if let Ok(i) = v.extract::<i64>() {
        Ok(ParamValue::Int(i))
    } else if let Ok(f) = v.extract::<f64>() {
        Ok(ParamValue::Float(f))
    } else if let Ok(s) = v.extract::<String>() {
        Ok(ParamValue::Str(s))
    } else {
        Err(PyTypeError::new_err(
            "param values must be int, float, bool, or str",
        ))
    }
}

fn params_from_py(obj: Option<&Bound<'_, PyAny>>) -> PyResult<ParamMap> {
    let mut entries: Vec<(String, ParamValue)> = Vec::new();
    if let Some(d) = obj {
        let dict = d
            .cast::<PyDict>()
            .map_err(|_| PyTypeError::new_err("params must be a dict or None"))?;
        for (k, v) in dict.iter() {
            entries.push((k.extract::<String>()?, paramvalue_from_py(&v)?));
        }
    }
    Ok(ParamMap::new(entries))
}

fn descriptor_from_py(obj: &Bound<'_, PyAny>) -> PyResult<PayloadDescriptor> {
    if let Ok(pd) = obj.extract::<PyPayloadDescriptor>() {
        return Ok(pd.inner);
    }
    let dict = obj
        .cast::<PyDict>()
        .map_err(|_| PyTypeError::new_err("descriptor must be a PayloadDescriptor or dict"))?;
    let req = |key: &str| -> PyResult<String> {
        match dict.get_item(key)? {
            Some(v) => v.extract::<String>(),
            None => Err(PyValueError::new_err(format!(
                "descriptor missing required field '{key}'"
            ))),
        }
    };
    let preprocessing_ref = match dict.get_item("preprocessing_ref")? {
        Some(v) if !v.is_none() => Some(v.extract::<String>()?),
        _ => None,
    };
    Ok(PayloadDescriptor {
        kind: req("kind")?,
        content_hash: req("content_hash")?,
        framework: req("framework")?,
        version: req("version")?,
        io_schema: req("io_schema")?,
        preprocessing_ref,
    })
}

/// Reproducibility metadata an External node carries (participates in the structural hash).
#[pyclass(name = "PayloadDescriptor", frozen, from_py_object)]
#[derive(Clone)]
struct PyPayloadDescriptor {
    inner: PayloadDescriptor,
}

#[pymethods]
impl PyPayloadDescriptor {
    #[new]
    #[pyo3(signature = (*, kind, content_hash, framework, version, io_schema, preprocessing_ref=None))]
    fn new(
        kind: String,
        content_hash: String,
        framework: String,
        version: String,
        io_schema: String,
        preprocessing_ref: Option<String>,
    ) -> Self {
        Self {
            inner: PayloadDescriptor {
                kind,
                content_hash,
                framework,
                version,
                io_schema,
                preprocessing_ref,
            },
        }
    }

    #[getter]
    fn kind(&self) -> &str {
        &self.inner.kind
    }
    #[getter]
    fn content_hash(&self) -> &str {
        &self.inner.content_hash
    }
    #[getter]
    fn framework(&self) -> &str {
        &self.inner.framework
    }
    #[getter]
    fn version(&self) -> &str {
        &self.inner.version
    }
    #[getter]
    fn io_schema(&self) -> &str {
        &self.inner.io_schema
    }
    #[getter]
    fn preprocessing_ref(&self) -> Option<&str> {
        self.inner.preprocessing_ref.as_deref()
    }
}

/// Thread-safe interned graph store (frozen pyclass: interior mutability via a Mutex, safe to
/// share across threads under the GIL and free-threaded 3.14t).
#[pyclass(name = "GraphStore", frozen)]
struct PyGraphStore {
    store: GraphStore,
}

#[pymethods]
impl PyGraphStore {
    #[new]
    fn new() -> Self {
        Self {
            store: GraphStore::new(),
        }
    }

    #[pyo3(signature = (name, params=None))]
    fn add_source(&self, name: String, params: Option<&Bound<'_, PyAny>>) -> PyResult<NodeId> {
        let p = params_from_py(params)?;
        Ok(self.store.add_source(name, p))
    }

    #[pyo3(signature = (name, inputs, params=None))]
    fn add_op(
        &self,
        name: String,
        inputs: Vec<NodeId>,
        params: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<NodeId> {
        let p = params_from_py(params)?;
        Ok(self.store.add_op(name, inputs, p)?)
    }

    #[pyo3(signature = (name, inputs, params=None))]
    fn add_reduction(
        &self,
        name: String,
        inputs: Vec<NodeId>,
        params: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<NodeId> {
        let p = params_from_py(params)?;
        Ok(self.store.add_reduction(name, inputs, p)?)
    }

    #[pyo3(signature = (descriptor, inputs, params=None))]
    fn add_external(
        &self,
        descriptor: &Bound<'_, PyAny>,
        inputs: Vec<NodeId>,
        params: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<NodeId> {
        let d = descriptor_from_py(descriptor)?;
        let p = params_from_py(params)?;
        Ok(self.store.add_external(d, inputs, p)?)
    }

    /// Record an `Exchange` boundary (plan M39 §2.1). No `name` — an Exchange's identity is its
    /// `scheme` `ParamMap` (`params`) + its one logical input; `nodes()[i]["kind"] == "exchange"`.
    #[pyo3(signature = (inputs, params=None))]
    fn add_exchange(
        &self,
        inputs: Vec<NodeId>,
        params: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<NodeId> {
        let p = params_from_py(params)?;
        Ok(self.store.add_exchange(inputs, p)?)
    }

    /// Record a `Join` boundary (plan M40 §2.1). Two inputs `[left, right]`; identity is its
    /// `scheme` `ParamMap` (`params`) + its ordered inputs; `nodes()[i]["kind"] == "join"`.
    #[pyo3(signature = (inputs, params=None))]
    fn add_join(&self, inputs: Vec<NodeId>, params: Option<&Bound<'_, PyAny>>) -> PyResult<NodeId> {
        let p = params_from_py(params)?;
        Ok(self.store.add_join(inputs, p)?)
    }

    fn node_count(&self) -> usize {
        self.store.node_count()
    }

    fn to_dot(&self) -> String {
        self.store.to_dot()
    }

    /// Serialize the IR to the canonical, versioned, byte-identical durable form (plan M8 / A.3.1:
    /// the serializable IR — not cloudpickle — is the canonical durable representation). With
    /// `outputs=` the bytes flag EXACTLY that set, ignoring stored marks (M22: outputs are a
    /// property of the compile request); the default keeps the marks-based behavior.
    #[pyo3(signature = (outputs=None))]
    fn serialize<'py>(
        &self,
        py: Python<'py>,
        outputs: Option<Vec<NodeId>>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = match outputs {
            None => serialize::serialize(&self.store),
            Some(outs) => {
                let n = self.store.node_count() as NodeId;
                if let Some(&bad) = outs.iter().find(|&&o| o >= n) {
                    return Err(BadNodeId(bad).into());
                }
                serialize::serialize_with(&self.store, &outs)
            }
        }?;
        Ok(PyBytes::new(py, &bytes))
    }

    /// Rebuild a store from canonical bytes. A round trip reproduces the same node ids, so
    /// `serialize` of the result is byte-identical to the input (plan M8 determinism gate).
    #[staticmethod]
    fn deserialize(data: &[u8]) -> PyResult<Self> {
        Ok(Self {
            store: serialize::deserialize(data)?,
        })
    }

    /// Structured, read-only view of every node in id order (plan M9: `inspect` renders the IR and
    /// `reproduce` interprets it). Each entry is a dict with `id`, `kind`
    /// (source|op|reduction|external|stage), `name`, `params`, `inputs`, `output` (bool), plus a
    /// `descriptor` dict for external nodes and `n_members` for stage nodes.
    fn nodes<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let (nodes, outputs) = self.store.snapshot();
        let mut out = Vec::with_capacity(nodes.len());
        for (id, nk) in nodes.iter().enumerate() {
            let d = PyDict::new(py);
            d.set_item("id", id as u64)?;
            d.set_item("output", outputs.contains(&(id as NodeId)))?;
            d.set_item("inputs", nk.inputs().to_vec())?;
            match nk {
                node::NodeKey::Source { name, params } => {
                    d.set_item("kind", "source")?;
                    d.set_item("name", name)?;
                    d.set_item("params", params_to_py(py, params)?)?;
                }
                node::NodeKey::Op { name, params, .. } => {
                    d.set_item("kind", "op")?;
                    d.set_item("name", name)?;
                    d.set_item("params", params_to_py(py, params)?)?;
                }
                node::NodeKey::Reduction { name, params, .. } => {
                    d.set_item("kind", "reduction")?;
                    d.set_item("name", name)?;
                    d.set_item("params", params_to_py(py, params)?)?;
                }
                node::NodeKey::External {
                    descriptor, params, ..
                } => {
                    d.set_item("kind", "external")?;
                    d.set_item("name", "")?;
                    d.set_item("params", params_to_py(py, params)?)?;
                    d.set_item("descriptor", descriptor_to_py(py, descriptor)?)?;
                }
                node::NodeKey::Exchange { scheme, .. } => {
                    d.set_item("kind", "exchange")?;
                    d.set_item("name", "")?;
                    d.set_item("params", params_to_py(py, scheme)?)?;
                }
                node::NodeKey::Join { scheme, .. } => {
                    d.set_item("kind", "join")?;
                    d.set_item("name", "")?;
                    d.set_item("params", params_to_py(py, scheme)?)?;
                }
                node::NodeKey::Stage { members, .. } => {
                    d.set_item("kind", "stage")?;
                    d.set_item("name", "")?;
                    d.set_item("params", PyDict::new(py))?;
                    d.set_item("n_members", members.len() as u64)?;
                    // the fused op-DAG, decoded to executable (name, params) pairs — what lets an
                    // executor evaluate the REDUCED IR directly (one dispatch per fused op, no
                    // re-recording). Each member input is ("input", slot) or ("member", index).
                    let mut ms = Vec::with_capacity(members.len());
                    for m in members {
                        let (kind, name, params) = parse_op_token(&m.token).ok_or_else(|| {
                            PyValueError::new_err(format!(
                                "unparseable stage member token {:?}",
                                m.token
                            ))
                        })?;
                        let md = PyDict::new(py);
                        md.set_item("kind", kind)?;
                        md.set_item("name", name)?;
                        md.set_item("params", params_to_py(py, &params)?)?;
                        let refs: Vec<(&str, usize)> = m
                            .inputs
                            .iter()
                            .map(|r| match r {
                                node::StageRef::Input(i) => ("input", *i),
                                node::StageRef::Member(i) => ("member", *i),
                            })
                            .collect();
                        md.set_item("inputs", refs)?;
                        ms.push(md);
                    }
                    d.set_item("members", ms)?;
                }
            }
            out.push(d);
        }
        Ok(out)
    }

    /// The marked output node ids, in mark order (what an IR evaluator returns, in order).
    fn outputs(&self) -> Vec<NodeId> {
        self.store.outputs()
    }

    /// §8.2(i): for each node of the arena this store was REDUCED FROM, where it landed here —
    /// `(reduced_node_id, member_index | None)`, or `None` if DCE dropped it. Indexed by that
    /// arena's node id. Empty for a store that did not come out of a reduction.
    fn node_map(&self) -> Vec<Option<(NodeId, Option<usize>)>> {
        self.store.node_map()
    }

    /// `opt_level=0` (M6): the 1:1 cone of `outputs` — DCE without the rewrites — as a fresh store
    /// marked with `outputs`; its `node_map` sends each kept id to `(cone id, None)`.
    #[pyo3(signature = (*, outputs))]
    #[allow(
        clippy::needless_pass_by_value,
        reason = "pyo3 extracts a Python sequence only into an owned Vec"
    )]
    fn cone(&self, outputs: Vec<NodeId>) -> PyResult<Self> {
        let store = self.store.cone(&outputs)?;
        Ok(Self { store })
    }

    /// Reduce via the M4 optimizer (DCE + CSE + equality-saturation stage fusion behind
    /// `RewriteEngine`). Returns the reduced store and a report dict. `maximal_fusion=True` opts in
    /// to fusing fan-out ops whose consumers all land in one stage (the default single-use rule is
    /// pinned by the frozen M4 suite).
    #[pyo3(signature = (*, maximal_fusion=false, outputs=None))]
    fn reduce(
        &self,
        maximal_fusion: bool,
        outputs: Option<Vec<NodeId>>,
    ) -> PyResult<(Self, std::collections::HashMap<String, usize>)> {
        let (reduced, report) = match outputs {
            // M22: an explicit output set scopes the reduction to the compile request, ignoring
            // stored marks — compiling is read-only and sequential compiles never cross-talk
            Some(outs) => self.store.reduce_with_outputs(
                &outs,
                &EggEngine::default(),
                fusion_mode(maximal_fusion),
            )?,
            None => self
                .store
                .reduce_with(&EggEngine::default(), fusion_mode(maximal_fusion)),
        };
        Ok((Self { store: reduced }, report_to_map(&report)))
    }

    /// One-shot incremental reduction of the current graph (same result as `reduce`). For genuine
    /// step-by-step reduction while building, use `IncrementalReducer`.
    #[pyo3(signature = (*, outputs=None))]
    fn reduce_incremental(
        &self,
        outputs: Option<Vec<NodeId>>,
    ) -> PyResult<(Self, std::collections::HashMap<String, usize>)> {
        let (reduced, report) = match outputs {
            Some(outs) => self.store.reduce_with_outputs(
                &outs,
                &EggEngine::default(),
                FusionMode::SingleUse,
            )?,
            None => self.store.reduce_incremental(&EggEngine::default()),
        };
        Ok((Self { store: reduced }, report_to_map(&report)))
    }

    /// The reduction report for the current graph, without keeping the reduced store.
    #[pyo3(signature = (*, outputs=None))]
    fn reduction_report(
        &self,
        outputs: Option<Vec<NodeId>>,
    ) -> PyResult<std::collections::HashMap<String, usize>> {
        let (_, report) = match outputs {
            Some(outs) => self.store.reduce_with_outputs(
                &outs,
                &EggEngine::default(),
                FusionMode::SingleUse,
            )?,
            None => self.store.reduce(&EggEngine::default()),
        };
        Ok(report_to_map(&report))
    }
}

fn params_to_py<'py>(py: Python<'py>, params: &ParamMap) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    for (k, v) in params.entries() {
        match v {
            ParamValue::Int(i) => d.set_item(k, *i)?,
            ParamValue::Float(f) => d.set_item(k, *f)?,
            ParamValue::Bool(b) => d.set_item(k, *b)?,
            ParamValue::Str(s) => d.set_item(k, s)?,
        }
    }
    Ok(d)
}

fn descriptor_to_py<'py>(
    py: Python<'py>,
    desc: &PayloadDescriptor,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("kind", &desc.kind)?;
    d.set_item("content_hash", &desc.content_hash)?;
    d.set_item("framework", &desc.framework)?;
    d.set_item("version", &desc.version)?;
    d.set_item("io_schema", &desc.io_schema)?;
    d.set_item("preprocessing_ref", desc.preprocessing_ref.as_deref())?;
    Ok(d)
}

fn report_to_map(r: &ReductionReport) -> std::collections::HashMap<String, usize> {
    [
        ("input_nodes", r.input_nodes),
        ("reachable_nodes", r.reachable_nodes),
        ("canonical_nodes", r.canonical_nodes),
        ("stages", r.stages),
        ("reduced_nodes", r.reduced_nodes),
        ("boundary_nodes", r.boundary_nodes),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect()
}

/// Genuinely incremental reduction (plan §A.1): feed it a store as the graph is being built; each
/// `step` canonicalizes ONLY the nodes recorded since the last step (identity elimination,
/// commutativity dedup, hash-consing — the same sound rules the `EggEngine` runs), so a concise
/// canonical form is maintained while building and `finalize` does one linear pass instead of a
/// whole-history optimization. `total_work()` counts nodes processed across all steps — it equals
/// the node count of the store, no matter how many steps fed it (the incrementality witness).
#[pyclass(name = "IncrementalReducer", frozen)]
struct PyIncrementalReducer {
    inner: Mutex<IncrementalReducer>,
}

impl PyIncrementalReducer {
    #[allow(
        clippy::expect_used,
        reason = "nothing panics while the lock is held, so it is never poisoned"
    )]
    fn reducer(&self) -> MutexGuard<'_, IncrementalReducer> {
        self.inner.lock().expect("incremental reducer poisoned")
    }
}

#[pymethods]
impl PyIncrementalReducer {
    #[new]
    fn new() -> Self {
        Self {
            inner: Mutex::new(IncrementalReducer::new()),
        }
    }

    /// Consume the nodes recorded in `store` since the last step. Returns how many were processed
    /// (the delta size). The reducer must always be fed the same store.
    fn step(&self, store: &PyGraphStore) -> PyResult<usize> {
        let mut r = self.reducer();
        let delta = store.store.snapshot_from(r.watermark());
        Ok(r.step(&delta)?)
    }

    /// How many of the store's nodes have been consumed so far.
    fn watermark(&self) -> usize {
        self.reducer().watermark()
    }

    /// Cumulative nodes processed across all steps (== watermark: each node touched once).
    fn total_work(&self) -> usize {
        self.reducer().total_work()
    }

    /// Size of the maintained canonical form.
    fn canonical_count(&self) -> usize {
        self.reducer().canonical_count()
    }

    /// Finish: consume any remaining delta, then reduce the maintained canonical form against the
    /// store's marked outputs — or, with `outputs=`, against EXACTLY that set (M22: stored marks
    /// ignored). Returns the reduced store + report, exactly like `GraphStore.reduce`.
    #[pyo3(signature = (store, *, maximal_fusion=false, outputs=None))]
    fn finalize(
        &self,
        store: &PyGraphStore,
        maximal_fusion: bool,
        outputs: Option<Vec<NodeId>>,
    ) -> PyResult<(PyGraphStore, std::collections::HashMap<String, usize>)> {
        let outs = match outputs {
            Some(outs) => {
                let n = store.store.node_count() as NodeId;
                if let Some(&bad) = outs.iter().find(|&&o| o >= n) {
                    return Err(BadNodeId(bad).into());
                }
                outs
            }
            None => store.store.outputs(),
        };
        let mut r = self.reducer();
        let delta = store.store.snapshot_from(r.watermark());
        r.step(&delta)?;
        let red = r.finalize(&outs, &EggEngine::default(), fusion_mode(maximal_fusion))?;
        let (reduced, report) = GraphStore::from_reduced(red);
        Ok((PyGraphStore { store: reduced }, report_to_map(&report)))
    }
}

#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pymodule(gil_used = false)]
fn graphed_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGraphStore>()?;
    m.add_class::<PyPayloadDescriptor>()?;
    m.add_class::<PyIncrementalReducer>()?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    Ok(())
}

#[cfg(test)]
#[cfg(not(loom))]
mod tests {
    use super::*;
    use std::ffi::CStr;

    /// Run Python `code` with a freshly initialized `graphed_core` module bound to `gc`.
    fn run(code: &CStr) {
        Python::initialize();
        Python::attach(|py| {
            let m = PyModule::new(py, "graphed_core")?;
            graphed_core(&m)?;
            let globals = PyDict::new(py);
            globals.set_item("gc", m)?;
            py.run(code, Some(&globals), None)
        })
        .unwrap();
    }

    #[test]
    fn an_encode_error_raises_value_error() {
        Python::initialize();
        Python::attach(|py| {
            let err = PyErr::from(EncodeError(1));
            assert!(err.is_instance_of::<PyValueError>(py));
            assert!(err.to_string().contains("does not fit"));
        });
    }

    #[test]
    fn records_every_node_kind_and_validates_arguments() {
        run(c"
def raises(exc, f, *a, **k):
    try:
        f(*a, **k)
    except exc as e:
        return str(e)
    raise AssertionError(f'{f} did not raise {exc}')

assert gc.version()
s = gc.GraphStore()
src = s.add_source('events', {'i': 1, 'f': 2.5, 'b': True, 's': 'x'})
assert s.add_source('events', {'b': True, 'i': 1, 's': 'x', 'f': 2.5}) == src
pt = s.add_op('pt', [src])
red = s.add_reduction('sum', [pt], None)
pd = gc.PayloadDescriptor(kind='onnx', content_hash='h', framework='ort', version='1',
                          io_schema='f->f', preprocessing_ref='pre')
assert (pd.kind, pd.content_hash, pd.framework, pd.version, pd.io_schema,
        pd.preprocessing_ref) == ('onnx', 'h', 'ort', '1', 'f->f', 'pre')
ext = s.add_external(pd, [pt], {'k': 1})
as_dict = {'kind': 'onnx', 'content_hash': 'h', 'framework': 'ort', 'version': '1',
           'io_schema': 'f->f', 'preprocessing_ref': 'pre'}
assert s.add_external(as_dict, [pt], {'k': 1}) == ext
ext2 = s.add_external({**as_dict, 'preprocessing_ref': None}, [pt])
assert ext2 != ext
exch = s.add_exchange([pt], {'parts': 2})
join = s.add_join([exch, exch], {'how': 'inner'})
assert s.node_count() == 7
assert s.to_dot().startswith('digraph')

assert 'no node with id 99' in raises(ValueError, s.add_op, 'x', [99])
raises(ValueError, s.add_reduction, 'x', [99])
raises(ValueError, s.add_external, pd, [99])
raises(ValueError, s.add_exchange, [99])
raises(ValueError, s.add_join, [99])
assert 'int, float, bool, or str' in raises(TypeError, s.add_source, 'e', {'v': [1]})
assert 'dict or None' in raises(TypeError, s.add_source, 'e', [1])
assert 'PayloadDescriptor or dict' in raises(TypeError, s.add_external, 1, [pt])
assert \"'version'\" in raises(ValueError, s.add_external, {'kind': 'onnx', 'content_hash': 'h',
                                            'framework': 'ort', 'io_schema': 's'}, [pt])

kinds = [n['kind'] for n in s.nodes()]
assert kinds == ['source', 'op', 'reduction', 'external', 'external', 'exchange', 'join'], kinds
by_kind = {n['kind']: n for n in s.nodes()}
assert by_kind['source']['params'] == {'i': 1, 'f': 2.5, 'b': True, 's': 'x'}
assert by_kind['external']['descriptor']['kind'] == 'onnx'
assert by_kind['exchange']['params'] == {'parts': 2} and by_kind['exchange']['name'] == ''
assert by_kind['join']['inputs'] == [exch, exch]
");
    }

    #[test]
    fn serializes_reduces_and_reports_through_python() {
        run(c"
import struct

def raises(exc, f, *a, **k):
    try:
        f(*a, **k)
    except exc as e:
        return str(e)
    raise AssertionError(f'{f} did not raise {exc}')

s = gc.GraphStore()
src = s.add_source('events')
pt = s.add_op('pt', [src])
neg = s.add_op('neg', [pt])
dead = s.add_op('eta', [src])
out = s.add_reduction('sum', [neg])

assert s.outputs() == [] and s.node_map() == []
blob = s.serialize(outputs=[out])
back = gc.GraphStore.deserialize(blob)
assert back.outputs() == [out] and back.serialize() == blob
assert s.serialize() != blob
raises(ValueError, s.serialize, outputs=[99])
assert 'bad magic' in raises(ValueError, gc.GraphStore.deserialize, b'XXXX')

cone = s.cone(outputs=[out])
assert cone.outputs() == [3] and cone.node_map()[dead] is None
raises(ValueError, s.cone, outputs=[99])

for kw in ({}, {'maximal_fusion': True}):
    red, rep = s.reduce(outputs=[out], **kw)
    assert rep['input_nodes'] == 5 and rep['reachable_nodes'] == 4, rep
    stage = [n for n in red.nodes() if n['kind'] == 'stage'][0]
    assert stage['n_members'] == 2 and stage['params'] == {}
    assert [m['name'] for m in stage['members']] == ['pt', 'neg']
    assert stage['members'][0]['inputs'] == [('input', 0)]
    assert stage['members'][1]['inputs'] == [('member', 0)]
    assert red.node_map()[dead] is None
raises(ValueError, s.reduce, outputs=[99])

red2, rep2 = back.reduce()
assert rep2 == rep
assert back.reduce_incremental()[1] == rep
assert s.reduce_incremental(outputs=[out])[1] == rep
raises(ValueError, s.reduce_incremental, outputs=[99])
assert back.reduction_report() == rep and s.reduction_report(outputs=[out]) == rep
raises(ValueError, s.reduction_report, outputs=[99])

bad_member = (b'GIR1' + struct.pack('<IBII', 1, 4, 0, 1) + struct.pack('<I', 5) + b'src|x'
              + struct.pack('<II', 0, 0))
assert 'unparseable stage member' in raises(ValueError, gc.GraphStore.deserialize(bad_member).nodes)
");
    }

    #[test]
    fn incremental_reducer_matches_one_shot_reduce() {
        run(c"
def raises(exc, f, *a, **k):
    try:
        f(*a, **k)
    except exc as e:
        return str(e)
    raise AssertionError(f'{f} did not raise {exc}')

s = gc.GraphStore()
a = s.add_source('a')
b = s.add_source('b')
r = gc.IncrementalReducer()
assert r.step(s) == 2 and r.watermark() == 2
ab = s.add_op('add', [a, b])
ba = s.add_op('add', [b, a])
out = s.add_op('mul', [ab, ba])
assert r.step(s) == 3
assert r.watermark() == r.total_work() == s.node_count() == 5
assert r.canonical_count() == 4
for kw in ({}, {'maximal_fusion': True}):
    inc, rep = r.finalize(s, outputs=[out], **kw)
    one_shot, one_rep = s.reduce(outputs=[out], **kw)
    assert inc.serialize() == one_shot.serialize()
    assert rep['input_nodes'] == 4 and one_rep['input_nodes'] == 5
assert r.finalize(s)[0].nodes() == []
raises(ValueError, r.finalize, s, outputs=[99])
");
    }
}
