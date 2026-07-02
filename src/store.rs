//! The thread-safe interned graph store (plan M1).
//!
//! # Locking discipline
//!
//! Interning is a read-modify-write on shared state: look up the structural key in the intern
//! table and, only if absent, push a new node and record its id. To keep that atomic and
//! race-free, the entire inner state (arena + intern table + outputs) sits behind a single
//! [`std::sync::Mutex`]. A `RwLock` would not help — interning writes on the common path — and
//! sharding is deferred because a node's `inputs` reference ids in the shared arena, so a sharded
//! design would need cross-shard coordination to validate them. The single mutex is therefore the
//! documented discipline; it is correct under the GIL and under free-threaded 3.14t. The critical
//! section is short (a hash lookup + two pushes), so contention is acceptable for the MVP; finer
//! sharding is a tracked improvement. A `loom` model of this critical section lives in the
//! `loom_model` test module below (run with `RUSTFLAGS="--cfg loom" cargo test --lib loom_model`).

#[cfg(not(loom))]
use std::sync::{Mutex, MutexGuard};

#[cfg(loom)]
use loom::sync::{Mutex, MutexGuard};

use std::collections::HashMap;

use crate::node::{NodeId, NodeKey, PayloadDescriptor};
use crate::optimizer::{self, ReductionReport, RewriteEngine};
use crate::param::ParamMap;

/// Error returned when a referenced node id is not in the arena.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BadNodeId(pub NodeId);

impl std::fmt::Display for BadNodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "no node with id {}", self.0)
    }
}

struct Inner {
    nodes: Vec<NodeKey>,
    intern: HashMap<NodeKey, NodeId>,
    outputs: Vec<NodeId>,
}

pub struct GraphStore {
    inner: Mutex<Inner>,
}

impl Default for GraphStore {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphStore {
    pub fn new() -> Self {
        GraphStore {
            inner: Mutex::new(Inner {
                nodes: Vec::new(),
                intern: HashMap::new(),
                outputs: Vec::new(),
            }),
        }
    }

    fn lock(&self) -> MutexGuard<'_, Inner> {
        // Poisoning only happens if a thread panics mid-update; our critical sections do not panic.
        self.inner
            .lock()
            .expect("graphed-core store mutex poisoned")
    }

    /// Intern a key, validating any input ids in the same critical section.
    fn intern(&self, key: NodeKey) -> Result<NodeId, BadNodeId> {
        let mut g = self.lock();
        let len = g.nodes.len() as NodeId;
        for &i in key.inputs() {
            if i >= len {
                return Err(BadNodeId(i));
            }
        }
        if let Some(&id) = g.intern.get(&key) {
            return Ok(id);
        }
        let id = g.nodes.len() as NodeId;
        g.nodes.push(key.clone());
        g.intern.insert(key, id);
        Ok(id)
    }

    pub fn add_source(&self, name: String, params: ParamMap) -> NodeId {
        // sources have no inputs -> interning cannot fail
        self.intern(NodeKey::Source { name, params })
            .expect("source has no inputs")
    }

    pub fn add_op(
        &self,
        name: String,
        inputs: Vec<NodeId>,
        params: ParamMap,
    ) -> Result<NodeId, BadNodeId> {
        self.intern(NodeKey::Op {
            name,
            params,
            inputs,
        })
    }

    pub fn add_reduction(
        &self,
        name: String,
        inputs: Vec<NodeId>,
        params: ParamMap,
    ) -> Result<NodeId, BadNodeId> {
        self.intern(NodeKey::Reduction {
            name,
            params,
            inputs,
        })
    }

    pub fn add_external(
        &self,
        descriptor: PayloadDescriptor,
        inputs: Vec<NodeId>,
        params: ParamMap,
    ) -> Result<NodeId, BadNodeId> {
        self.intern(NodeKey::External {
            descriptor,
            params,
            inputs,
        })
    }

    /// Record an `Exchange` boundary (plan M39): its `scheme` ParamMap is its identity, its one
    /// logical `input` is the partition it repartitions.
    pub fn add_exchange(&self, inputs: Vec<NodeId>, scheme: ParamMap) -> Result<NodeId, BadNodeId> {
        self.intern(NodeKey::Exchange { scheme, inputs })
    }

    pub fn mark_output(&self, id: NodeId) -> Result<(), BadNodeId> {
        let mut g = self.lock();
        if id >= g.nodes.len() as NodeId {
            return Err(BadNodeId(id));
        }
        if !g.outputs.contains(&id) {
            g.outputs.push(id);
        }
        Ok(())
    }

    pub fn node_count(&self) -> usize {
        self.lock().nodes.len()
    }

    /// Intern a pre-built `NodeKey` (used when rebuilding the reduced graph).
    pub fn add_key(&self, key: NodeKey) -> Result<NodeId, BadNodeId> {
        self.intern(key)
    }

    /// Snapshot the arena + outputs for the optimizer (clone under one lock).
    pub fn snapshot(&self) -> (Vec<NodeKey>, Vec<NodeId>) {
        let g = self.lock();
        (g.nodes.clone(), g.outputs.clone())
    }

    /// Snapshot only the nodes with id >= `start` (the delta an `IncrementalReducer` consumes).
    pub fn snapshot_from(&self, start: usize) -> Vec<NodeKey> {
        let g = self.lock();
        g.nodes[start.min(g.nodes.len())..].to_vec()
    }

    /// The marked output ids, in mark order.
    pub fn outputs(&self) -> Vec<NodeId> {
        self.lock().outputs.clone()
    }

    /// Rebuild a `Reduced` form into a fresh interned store.
    pub(crate) fn from_reduced(red: optimizer::Reduced) -> (GraphStore, ReductionReport) {
        let store = GraphStore::new();
        let mut map: Vec<NodeId> = Vec::with_capacity(red.nodes.len());
        for key in &red.nodes {
            let remapped: Vec<NodeId> = key.inputs().iter().map(|&i| map[i as usize]).collect();
            let id = store
                .add_key(key.with_inputs(remapped))
                .expect("reduced graph references only earlier nodes");
            map.push(id);
        }
        for &o in &red.outputs {
            store
                .mark_output(map[o as usize])
                .expect("reduced output is valid");
        }
        (store, red.report)
    }

    /// Reduce the graph via the M4 pipeline (DCE + CSE + equality-saturation stage fusion) into a
    /// fresh interned store. Returns the reduced store and the reduction report.
    pub fn reduce(&self, engine: &dyn RewriteEngine) -> (GraphStore, ReductionReport) {
        self.reduce_with(engine, optimizer::FusionMode::SingleUse)
    }

    /// `reduce` with an explicit stage-fusion mode (see `optimizer::FusionMode`).
    pub fn reduce_with(
        &self,
        engine: &dyn RewriteEngine,
        mode: optimizer::FusionMode,
    ) -> (GraphStore, ReductionReport) {
        let (nodes, outputs) = self.snapshot();
        GraphStore::from_reduced(optimizer::reduce_with_mode(&nodes, &outputs, engine, mode))
    }

    /// `reduce_with` against an EXPLICIT output set — the compile request's — ignoring stored
    /// marks entirely (M22: outputs are a property of the compile request, not store state, so
    /// compiling is a READ-ONLY operation and sequential compiles never cross-talk).
    pub fn reduce_with_outputs(
        &self,
        outputs: &[NodeId],
        engine: &dyn RewriteEngine,
        mode: optimizer::FusionMode,
    ) -> Result<(GraphStore, ReductionReport), BadNodeId> {
        let nodes = { self.lock().nodes.clone() };
        for &o in outputs {
            if o >= nodes.len() as NodeId {
                return Err(BadNodeId(o));
            }
        }
        Ok(GraphStore::from_reduced(optimizer::reduce_with_mode(
            &nodes, outputs, engine, mode,
        )))
    }

    /// One-shot incremental reduction of the current graph — same result as `reduce`. For genuine
    /// step-by-step reduction while building, use `optimizer::IncrementalReducer`, which processes
    /// only the delta per step (and whose work counter proves it).
    pub fn reduce_incremental(&self, engine: &dyn RewriteEngine) -> (GraphStore, ReductionReport) {
        self.reduce(engine)
    }

    /// Byte-stable graphviz rendering: nodes in id order, edges in (node, input-position) order.
    pub fn to_dot(&self) -> String {
        use std::fmt::Write as _;
        let g = self.lock();
        let mut out = String::from("digraph graphed {\n");
        for (id, node) in g.nodes.iter().enumerate() {
            let shape = if g.outputs.contains(&(id as NodeId)) {
                ", shape=doublecircle"
            } else {
                ""
            };
            // writing to a String is infallible
            let _ = writeln!(out, "  n{id} [label=\"{}\"{shape}];", escape(&node.label()));
        }
        for (id, node) in g.nodes.iter().enumerate() {
            for &src in node.inputs() {
                let _ = writeln!(out, "  n{src} -> n{id};");
            }
        }
        out.push_str("}\n");
        out
    }
}

fn escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;

    fn pm(entries: Vec<(&str, crate::param::ParamValue)>) -> ParamMap {
        ParamMap::new(
            entries
                .into_iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect(),
        )
    }

    #[test]
    fn identical_structure_interns() {
        let s = GraphStore::new();
        let src = s.add_source("e".into(), pm(vec![]));
        let a = s.add_op("pt".into(), vec![src], pm(vec![])).unwrap();
        let b = s.add_op("pt".into(), vec![src], pm(vec![])).unwrap();
        assert_eq!(a, b);
        assert_eq!(s.node_count(), 2);
    }

    #[test]
    fn nan_canonicalizes_zero_signed_distinct() {
        use crate::param::ParamValue::Float;
        let s = GraphStore::new();
        let src = s.add_source("e".into(), pm(vec![]));
        let nan1 = s
            .add_op("k".into(), vec![src], pm(vec![("v", Float(f64::NAN))]))
            .unwrap();
        let nan2 = s
            .add_op("k".into(), vec![src], pm(vec![("v", Float(-f64::NAN))]))
            .unwrap();
        let pos = s
            .add_op("k".into(), vec![src], pm(vec![("v", Float(0.0))]))
            .unwrap();
        let neg = s
            .add_op("k".into(), vec![src], pm(vec![("v", Float(-0.0))]))
            .unwrap();
        assert_eq!(nan1, nan2, "all NaNs intern to one node");
        assert_ne!(pos, neg, "0.0 and -0.0 are distinct");
        assert_ne!(nan1, pos);
    }

    #[test]
    fn output_scoped_reduce_ignores_marks_and_matches_single_mark_store() {
        use crate::optimizer::EggEngine;
        let build = || {
            let s = GraphStore::new();
            let src = s.add_source("events".into(), pm(vec![]));
            let a = s.add_op("pt".into(), vec![src], pm(vec![])).unwrap();
            let b = s.add_op("eta".into(), vec![src], pm(vec![])).unwrap();
            (s, a, b)
        };
        let (s, a, b) = build();
        s.mark_output(a).unwrap(); // stale state from an earlier compile
        let (scoped, _) = s
            .reduce_with_outputs(
                &[b],
                &EggEngine::default(),
                optimizer::FusionMode::SingleUse,
            )
            .unwrap();
        let (reference, _) = {
            let (s2, _a2, b2) = build();
            s2.mark_output(b2).unwrap();
            (s2.reduce(&EggEngine::default()).0, ())
        };
        assert_eq!(
            crate::serialize::serialize(&scoped),
            crate::serialize::serialize(&reference),
            "explicit-outputs reduce is byte-identical to a fresh single-mark store"
        );
        assert_eq!(s.outputs(), vec![a], "compiling wrote no store state");
        // invalid ids are rejected, not silently dropped
        assert!(s
            .reduce_with_outputs(
                &[10_000],
                &EggEngine::default(),
                optimizer::FusionMode::SingleUse
            )
            .is_err());
    }

    #[test]
    fn serialize_with_scopes_the_output_flags() {
        let s = GraphStore::new();
        let src = s.add_source("events".into(), pm(vec![]));
        let a = s.add_op("pt".into(), vec![src], pm(vec![])).unwrap();
        let b = s.add_op("eta".into(), vec![src], pm(vec![])).unwrap();
        s.mark_output(a).unwrap();
        let scoped = crate::serialize::deserialize(&crate::serialize::serialize_with(&s, &[b]))
            .expect("round trip");
        assert_eq!(scoped.outputs(), vec![b], "exactly the requested set");
        let legacy =
            crate::serialize::deserialize(&crate::serialize::serialize(&s)).expect("round trip");
        assert_eq!(
            legacy.outputs(),
            vec![a],
            "the default stays the marks path"
        );
    }

    #[test]
    fn bad_input_rejected() {
        let s = GraphStore::new();
        assert_eq!(
            s.add_op("x".into(), vec![99], pm(vec![])),
            Err(BadNodeId(99))
        );
        assert_eq!(s.mark_output(99), Err(BadNodeId(99)));
    }

    #[test]
    fn external_descriptor_participates_in_identity() {
        let s = GraphStore::new();
        let src = s.add_source("e".into(), pm(vec![]));
        let d = |hash: &str| PayloadDescriptor {
            kind: "onnx".into(),
            content_hash: hash.into(),
            framework: "ort".into(),
            version: "1".into(),
            io_schema: "x".into(),
            preprocessing_ref: None,
        };
        let a = s.add_external(d("h1"), vec![src], pm(vec![])).unwrap();
        let b = s.add_external(d("h1"), vec![src], pm(vec![])).unwrap();
        let c = s.add_external(d("h2"), vec![src], pm(vec![])).unwrap();
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn concurrent_interning_counts_exactly() {
        use std::sync::Arc;
        use std::thread;
        let s = Arc::new(GraphStore::new());
        let src = s.add_source("e".into(), pm(vec![]));
        let mut handles = vec![];
        for _ in 0..16 {
            let s = Arc::clone(&s);
            handles.push(thread::spawn(move || {
                for i in 0..100 {
                    // every thread builds the SAME 100 ops -> heavy intern contention
                    s.add_op(
                        "op".into(),
                        vec![src],
                        pm(vec![("i", crate::param::ParamValue::Int(i))]),
                    )
                    .unwrap();
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(s.node_count(), 1 + 100);
    }

    #[test]
    fn to_dot_is_byte_stable() {
        let build = || {
            let s = GraphStore::new();
            let src = s.add_source(
                "e".into(),
                pm(vec![("uri", crate::param::ParamValue::Str("f".into()))]),
            );
            let pt = s.add_op("pt".into(), vec![src], pm(vec![])).unwrap();
            s.mark_output(pt).unwrap();
            s.to_dot()
        };
        assert_eq!(build(), build());
        assert!(build().starts_with("digraph"));
    }
}

/// loom model of the intern critical section (plan M1: "a `loom` test of the locking discipline").
///
/// Run with: `RUSTFLAGS="--cfg loom" cargo test --lib loom_model`. loom exhaustively explores the
/// thread interleavings around the store's `Mutex` (which is `loom::sync::Mutex` under `--cfg loom`)
/// and asserts that two threads interning the same structural key produce exactly one node and the
/// same id — i.e. the single-mutex discipline is free of races and lost updates.
#[cfg(all(test, loom))]
mod loom_model {
    use super::*;
    use crate::param::ParamMap;
    use loom::sync::Arc;

    #[test]
    fn concurrent_intern_of_same_key_creates_one_node() {
        loom::model(|| {
            let s = Arc::new(GraphStore::new());
            let src = s.add_source("e".into(), ParamMap::new(vec![]));
            let handle = {
                let s = Arc::clone(&s);
                loom::thread::spawn(move || {
                    s.add_op("pt".into(), vec![src], ParamMap::new(vec![]))
                        .unwrap()
                })
            };
            let id_main = s
                .add_op("pt".into(), vec![src], ParamMap::new(vec![]))
                .unwrap();
            let id_other = handle.join().unwrap();
            assert_eq!(id_main, id_other);
            assert_eq!(s.node_count(), 2); // source + exactly one interned op
        });
    }
}
