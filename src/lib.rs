//! graphed-core: thread-safe interned graph IR (Rust+PyO3), plan M1.
//!
//! The graph lives in Rust; this crate does not (and must not) depend on awkward. No optimization
//! here — that is M4.

mod node;
mod param;
mod store;

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict};

use crate::node::{NodeId, PayloadDescriptor};
use crate::param::{ParamMap, ParamValue};
use crate::store::{BadNodeId, GraphStore};

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
}

#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pymodule(gil_used = false)]
fn graphed_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGraphStore>()?;
    m.add_class::<PyPayloadDescriptor>()?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    Ok(())
}
