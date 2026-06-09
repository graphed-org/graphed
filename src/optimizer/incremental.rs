//! Genuinely incremental reduction (plan §A.1: "reduce the graph to a concise set of stage-nodes
//! incrementally as the user builds it, so a large un-reduced graph never exists").
//!
//! The reducer consumes the interned arena *delta-by-delta*: each `step` processes only the nodes
//! recorded since the previous step, exactly once, applying the SAME sound rule set the
//! `EggEngine` runs (the shared `SYMMETRIC_OPS` / `IDENTITY_TOKENS` constants) as a *local,
//! constructor-time* canonicalization:
//!
//! - identity ops (`x+0`, `x*1`) collapse to their input the moment they are recorded;
//! - a symmetric op (`add`/`mul`/`and`/...) recorded with swapped arguments dedups onto the
//!   first-recorded orientation — the same earliest-representative choice the engine's O(N)
//!   extraction makes;
//! - everything else hash-conses against the canonical arena.
//!
//! Because every rule is constructor-local (an op's canonical form depends only on its inputs'
//! already-canonical forms), one pass per node reaches the same fixpoint equality saturation does
//! for this rule set — `finalize` therefore produces the same reduced graph as a one-shot
//! `reduce`, while the per-step work is **proportional to the delta**, never to the history. The
//! cumulative-work counter exposes that property so the frozen suite can assert incrementality
//! (the aliased implementation this replaces could not pass such a test).

use std::collections::HashMap;

use crate::node::{NodeId, NodeKey};
use crate::store::BadNodeId;

use super::engine::{IDENTITY_TOKENS, SYMMETRIC_OPS};
use super::{FusionMode, Reduced, RewriteEngine};

#[derive(Default)]
pub struct IncrementalReducer {
    /// The canonical arena: identity-eliminated, symmetry-deduped, hash-consed nodes whose inputs
    /// are canonical ids. Grows monotonically; never rescanned by `step`.
    canon_nodes: Vec<NodeKey>,
    canon_table: HashMap<NodeKey, NodeId>,
    /// original arena id -> canonical id (`len()` is the watermark into the original arena).
    map: Vec<NodeId>,
    /// cumulative nodes processed across all steps — the incrementality witness: equals the
    /// number of original nodes, no matter how many steps they arrived in.
    total_work: usize,
}

impl IncrementalReducer {
    pub fn new() -> Self {
        Self::default()
    }

    /// How many original-arena nodes have been consumed so far.
    pub fn watermark(&self) -> usize {
        self.map.len()
    }

    /// Cumulative nodes processed across every `step` (each original node is touched once).
    pub fn total_work(&self) -> usize {
        self.total_work
    }

    /// Size of the maintained canonical (concise) form.
    pub fn canonical_count(&self) -> usize {
        self.canon_nodes.len()
    }

    /// Consume the nodes recorded since the last step (`delta` = original arena `[watermark..]`).
    /// Returns the number of nodes processed — always exactly `delta.len()`.
    pub fn step(&mut self, delta: &[NodeKey]) -> Result<usize, BadNodeId> {
        for key in delta {
            let mut inputs = Vec::with_capacity(key.inputs().len());
            for &i in key.inputs() {
                inputs.push(self.map.get(i as usize).copied().ok_or(BadNodeId(i))?);
            }
            let canon = self.canonicalize(key.with_inputs(inputs));
            self.map.push(canon);
        }
        self.total_work += delta.len();
        Ok(delta.len())
    }

    fn canonicalize(&mut self, key: NodeKey) -> NodeId {
        if let NodeKey::Op {
            name,
            params,
            inputs,
        } = &key
        {
            // identity elimination — the engine's identity rules, applied at record time.
            if IDENTITY_TOKENS.contains(&key.token().as_str()) {
                return inputs[0];
            }
            // commutativity — the engine's commute rules: both orientations of a symmetric op are
            // ONE canonical node; the first-recorded orientation is the representative (matching
            // the engine's earliest-member extraction).
            if inputs.len() == 2 && params.is_empty() && SYMMETRIC_OPS.contains(&name.as_str()) {
                if let Some(&id) = self.canon_table.get(&key) {
                    return id;
                }
                let swapped = key.with_inputs(vec![inputs[1], inputs[0]]);
                if let Some(&id) = self.canon_table.get(&swapped) {
                    return id;
                }
            }
        }
        if let Some(&id) = self.canon_table.get(&key) {
            return id;
        }
        let id = self.canon_nodes.len() as NodeId;
        self.canon_nodes.push(key.clone());
        self.canon_table.insert(key, id);
        id
    }

    /// Finish the reduction for the given original-arena outputs: translate them into the
    /// canonical arena and run the standard pipeline (DCE + engine + CSE + stage fusion) over the
    /// *canonical* nodes. Cost is O(canonical size) once — independent of how many steps fed it.
    pub fn finalize(
        &self,
        outputs: &[NodeId],
        engine: &dyn RewriteEngine,
        mode: FusionMode,
    ) -> Result<Reduced, BadNodeId> {
        let mut outs = Vec::with_capacity(outputs.len());
        for &o in outputs {
            outs.push(self.map.get(o as usize).copied().ok_or(BadNodeId(o))?);
        }
        Ok(super::reduce_with_mode(
            &self.canon_nodes,
            &outs,
            engine,
            mode,
        ))
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;
    use crate::optimizer::EggEngine;
    use crate::param::{ParamMap, ParamValue};
    use crate::store::GraphStore;

    fn empty() -> ParamMap {
        ParamMap::new(vec![])
    }

    /// Drive a store and a reducer together, stepping after every record.
    fn reduce_incrementally(store: &GraphStore) -> IncrementalReducer {
        let mut r = IncrementalReducer::new();
        let (nodes, _) = store.snapshot();
        r.step(&nodes).unwrap();
        r
    }

    #[test]
    fn finalize_matches_one_shot_reduce() {
        let s = GraphStore::new();
        let a = s.add_source("a".into(), empty());
        let b = s.add_source("b".into(), empty());
        let ab = s.add_op("add".into(), vec![a, b], empty()).unwrap();
        let ba = s.add_op("add".into(), vec![b, a], empty()).unwrap(); // commuted twin
        let one = s
            .add_op(
                "mul".into(),
                vec![ab],
                ParamMap::new(vec![
                    ("scalar".into(), ParamValue::Float(1.0)),
                    ("side".into(), ParamValue::Str("r".into())),
                ]),
            )
            .unwrap(); // identity
        let out = s.add_op("mul".into(), vec![one, ba], empty()).unwrap();
        s.mark_output(out).unwrap();

        let r = reduce_incrementally(&s);
        let (outputs, engine) = (s.snapshot().1, EggEngine::default());
        let inc = r
            .finalize(&outputs, &engine, FusionMode::SingleUse)
            .unwrap();
        let full = crate::optimizer::reduce_with_mode(
            &s.snapshot().0,
            &outputs,
            &engine,
            FusionMode::SingleUse,
        );
        // identical reduced structure, node for node
        assert_eq!(inc.nodes, full.nodes);
        assert_eq!(inc.outputs, full.outputs);
    }

    #[test]
    fn per_step_work_is_the_delta_not_the_history() {
        let s = GraphStore::new();
        let src = s.add_source("x".into(), empty());
        let mut cur = src;
        let mut r = IncrementalReducer::new();
        let mut seen = 0usize;
        for i in 0..100 {
            cur = s
                .add_op(
                    "inc".into(),
                    vec![cur],
                    ParamMap::new(vec![("i".into(), ParamValue::Int(i))]),
                )
                .unwrap();
            let (nodes, _) = s.snapshot();
            let did = r.step(&nodes[seen..]).unwrap();
            seen = nodes.len();
            assert!(did <= 2, "a step processes only the delta, got {did}");
        }
        // every node touched exactly once across the whole build
        assert_eq!(r.total_work(), s.node_count());
    }
}
