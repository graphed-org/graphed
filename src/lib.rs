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

use std::sync::Mutex;

use crate::node::{parse_op_token, NodeId, PayloadDescriptor};
use crate::optimizer::{EggEngine, FusionMode, IncrementalReducer, ReductionReport};
use crate::param::{ParamMap, ParamValue};
use crate::store::{BadNodeId, GraphStore};

fn fusion_mode(maximal_fusion: bool) -> FusionMode {
    if maximal_fusion {
        FusionMode::Maximal
    } else {
        FusionMode::SingleUse
    }
}

fn map_err(e: BadNodeId) -> PyErr {
    PyValueError::new_err(e.to_string())
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
        PyPayloadDescriptor {
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
        PyGraphStore {
            store: GraphStore::new(),
        }
    }

    #[pyo3(signature = (name, params=None))]
    fn add_source(&self, name: String, params: Option<Bound<'_, PyAny>>) -> PyResult<NodeId> {
        let p = params_from_py(params.as_ref())?;
        Ok(self.store.add_source(name, p))
    }

    #[pyo3(signature = (name, inputs, params=None))]
    fn add_op(
        &self,
        name: String,
        inputs: Vec<NodeId>,
        params: Option<Bound<'_, PyAny>>,
    ) -> PyResult<NodeId> {
        let p = params_from_py(params.as_ref())?;
        self.store.add_op(name, inputs, p).map_err(map_err)
    }

    #[pyo3(signature = (name, inputs, params=None))]
    fn add_reduction(
        &self,
        name: String,
        inputs: Vec<NodeId>,
        params: Option<Bound<'_, PyAny>>,
    ) -> PyResult<NodeId> {
        let p = params_from_py(params.as_ref())?;
        self.store.add_reduction(name, inputs, p).map_err(map_err)
    }

    #[pyo3(signature = (descriptor, inputs, params=None))]
    fn add_external(
        &self,
        descriptor: Bound<'_, PyAny>,
        inputs: Vec<NodeId>,
        params: Option<Bound<'_, PyAny>>,
    ) -> PyResult<NodeId> {
        let d = descriptor_from_py(&descriptor)?;
        let p = params_from_py(params.as_ref())?;
        self.store.add_external(d, inputs, p).map_err(map_err)
    }

    fn mark_output(&self, node_id: NodeId) -> PyResult<()> {
        self.store.mark_output(node_id).map_err(map_err)
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
        match outputs {
            None => Ok(PyBytes::new(py, &serialize::serialize(&self.store))),
            Some(outs) => {
                let n = self.store.node_count() as NodeId;
                if let Some(&bad) = outs.iter().find(|&&o| o >= n) {
                    return Err(map_err(BadNodeId(bad)));
                }
                Ok(PyBytes::new(
                    py,
                    &serialize::serialize_with(&self.store, &outs),
                ))
            }
        }
    }

    /// Rebuild a store from canonical bytes. A round trip reproduces the same node ids, so
    /// `serialize` of the result is byte-identical to the input (plan M8 determinism gate).
    #[staticmethod]
    fn deserialize(data: &[u8]) -> PyResult<PyGraphStore> {
        let store =
            serialize::deserialize(data).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyGraphStore { store })
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

    /// Reduce via the M4 optimizer (DCE + CSE + equality-saturation stage fusion behind
    /// RewriteEngine). Returns the reduced store and a report dict. `maximal_fusion=True` opts in
    /// to fusing fan-out ops whose consumers all land in one stage (the default single-use rule is
    /// pinned by the frozen M4 suite).
    #[pyo3(signature = (*, maximal_fusion=false, outputs=None))]
    fn reduce(
        &self,
        maximal_fusion: bool,
        outputs: Option<Vec<NodeId>>,
    ) -> PyResult<(PyGraphStore, std::collections::HashMap<String, usize>)> {
        let (reduced, report) = match outputs {
            // M22: an explicit output set scopes the reduction to the compile request, ignoring
            // stored marks — compiling is read-only and sequential compiles never cross-talk
            Some(outs) => self
                .store
                .reduce_with_outputs(&outs, &EggEngine::default(), fusion_mode(maximal_fusion))
                .map_err(map_err)?,
            None => self
                .store
                .reduce_with(&EggEngine::default(), fusion_mode(maximal_fusion)),
        };
        Ok((PyGraphStore { store: reduced }, report_to_map(&report)))
    }

    /// One-shot incremental reduction of the current graph (same result as `reduce`). For genuine
    /// step-by-step reduction while building, use `IncrementalReducer`.
    fn reduce_incremental(&self) -> (PyGraphStore, std::collections::HashMap<String, usize>) {
        let (reduced, report) = self.store.reduce_incremental(&EggEngine::default());
        (PyGraphStore { store: reduced }, report_to_map(&report))
    }

    /// The reduction report for the current graph, without keeping the reduced store.
    fn reduction_report(&self) -> std::collections::HashMap<String, usize> {
        let (_, report) = self.store.reduce(&EggEngine::default());
        report_to_map(&report)
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
/// commutativity dedup, hash-consing — the same sound rules the EggEngine runs), so a concise
/// canonical form is maintained while building and `finalize` does one linear pass instead of a
/// whole-history optimization. `total_work()` counts nodes processed across all steps — it equals
/// the node count of the store, no matter how many steps fed it (the incrementality witness).
#[pyclass(name = "IncrementalReducer", frozen)]
struct PyIncrementalReducer {
    inner: Mutex<IncrementalReducer>,
}

#[pymethods]
impl PyIncrementalReducer {
    #[new]
    fn new() -> Self {
        PyIncrementalReducer {
            inner: Mutex::new(IncrementalReducer::new()),
        }
    }

    /// Consume the nodes recorded in `store` since the last step. Returns how many were processed
    /// (the delta size). The reducer must always be fed the same store.
    fn step(&self, store: &PyGraphStore) -> PyResult<usize> {
        let mut r = self.inner.lock().expect("incremental reducer poisoned");
        let delta = store.store.snapshot_from(r.watermark());
        r.step(&delta).map_err(map_err)
    }

    /// How many of the store's nodes have been consumed so far.
    fn watermark(&self) -> usize {
        self.inner
            .lock()
            .expect("incremental reducer poisoned")
            .watermark()
    }

    /// Cumulative nodes processed across all steps (== watermark: each node touched once).
    fn total_work(&self) -> usize {
        self.inner
            .lock()
            .expect("incremental reducer poisoned")
            .total_work()
    }

    /// Size of the maintained canonical form.
    fn canonical_count(&self) -> usize {
        self.inner
            .lock()
            .expect("incremental reducer poisoned")
            .canonical_count()
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
                    return Err(map_err(BadNodeId(bad)));
                }
                outs
            }
            None => store.store.outputs(),
        };
        let mut r = self.inner.lock().expect("incremental reducer poisoned");
        let delta = store.store.snapshot_from(r.watermark());
        r.step(&delta).map_err(map_err)?;
        let red = r
            .finalize(&outs, &EggEngine::default(), fusion_mode(maximal_fusion))
            .map_err(map_err)?;
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
