//! Genuinely incremental reduction (plan §A.1: "reduce the graph to a concise set of stage-nodes
//! incrementally as the user builds it, so a large un-reduced graph never exists").
//!
//! The reducer consumes the interned arena *delta-by-delta*: each `step` processes only the nodes
//! recorded since the previous step, exactly once, applying the SAME sound rule set the
//! `EggEngine` runs (the shared `SYMMETRIC_OPS` / `IDENTITY_TOKENS` constants) as a *local,
//! constructor-time* canonicalization:
//!
//! - identity ops (`x+0`, `x*1`) collapse to their input the moment they are recorded;
//! - a symmetric op (`add`/`mul`/`and`/...) takes its operands in structural-key order, as the
//!   one-shot path does, so both orientations dedup onto one node whichever came first;
//! - everything else hash-conses against the canonical arena.
//!
//! Because every rule is constructor-local (an op's canonical form depends only on its inputs'
//! already-canonical forms), one pass per node reaches the same fixpoint equality saturation does
//! for this rule set — `finalize` therefore produces the same reduced graph as a one-shot
//! `reduce`, while the per-step work is **proportional to the delta**, never to the history. The
//! cumulative-work counter exposes that property so the frozen suite can assert incrementality
//! (the aliased implementation this replaces could not pass such a test).

use std::collections::HashMap;

use crate::node::{index, NodeId, NodeKey};
use crate::store::BadNodeId;

use super::engine::IDENTITY_TOKENS;
use super::{orient, structural_key, FusionMode, Reduced, RewriteEngine};

#[derive(Default)]
pub(crate) struct IncrementalReducer {
    /// The canonical arena: identity-eliminated, symmetry-deduped, hash-consed nodes whose inputs
    /// are canonical ids. Grows monotonically; never rescanned by `step`.
    canon_nodes: Vec<NodeKey>,
    canon_table: HashMap<NodeKey, NodeId>,
    /// each canonical node's `structural_key`, by canonical id.
    canon_keys: Vec<u128>,
    /// `canonicalize`'s input-key buffer, kept to avoid an allocation per recorded node.
    scratch_keys: Vec<u128>,
    /// original arena id -> canonical id (`len()` is the watermark into the original arena).
    map: Vec<NodeId>,
    /// cumulative nodes processed across all steps — the incrementality witness: equals the
    /// number of original nodes, no matter how many steps they arrived in.
    total_work: usize,
}

impl IncrementalReducer {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// How many original-arena nodes have been consumed so far.
    pub(crate) fn watermark(&self) -> usize {
        self.map.len()
    }

    /// Cumulative nodes processed across every `step` (each original node is touched once).
    pub(crate) fn total_work(&self) -> usize {
        self.total_work
    }

    /// Size of the maintained canonical (concise) form.
    pub(crate) fn canonical_count(&self) -> usize {
        self.canon_nodes.len()
    }

    /// Consume the nodes recorded since the last step (`delta` = original arena `[watermark..]`).
    /// Returns the number of nodes processed — always exactly `delta.len()`.
    pub(crate) fn step(&mut self, delta: &[NodeKey]) -> Result<usize, BadNodeId> {
        for key in delta {
            let mut inputs = Vec::with_capacity(key.inputs().len());
            for &i in key.inputs() {
                inputs.push(self.map.get(index(i)).copied().ok_or(BadNodeId(i))?);
            }
            let canon = self.canonicalize(key.with_inputs(inputs));
            self.map.push(canon);
        }
        self.total_work += delta.len();
        Ok(delta.len())
    }

    #[allow(
        clippy::indexing_slicing,
        reason = "inputs are canonical ids, and every canonical node has its key"
    )]
    fn canonicalize(&mut self, key: NodeKey) -> NodeId {
        // identity elimination — the engine's identity rules (arity one, as their `(tok ?x)`
        // patterns are), applied at record time.
        let mut token = None;
        if let NodeKey::Op { inputs, .. } = &key {
            if let [only] = inputs.as_slice() {
                let t = key.token();
                if IDENTITY_TOKENS.contains(&t.as_str()) {
                    return *only;
                }
                token = Some(t);
            }
        }
        // commutativity — the engine's commute rules: both orientations of a symmetric op orient
        // to ONE canonical node by their operands' structural keys, as the one-shot path orients
        // them, so which orientation was recorded first never shows in the result.
        let mut input_keys = std::mem::take(&mut self.scratch_keys);
        input_keys.clear();
        input_keys.extend(key.inputs().iter().map(|&i| self.canon_keys[index(i)]));
        let key = orient(key, &input_keys);
        if let Some(&id) = self.canon_table.get(&key) {
            self.scratch_keys = input_keys;
            return id;
        }
        let id = self.canon_nodes.len() as NodeId;
        let token = token.unwrap_or_else(|| key.token());
        self.canon_keys
            .push(structural_key(&key, &token, &input_keys));
        self.scratch_keys = input_keys;
        self.canon_nodes.push(key.clone());
        self.canon_table.insert(key, id);
        id
    }

    /// Finish the reduction for the given original-arena outputs: translate them into the
    /// canonical arena and run the standard pipeline (DCE + engine + CSE + stage fusion) over the
    /// *canonical* nodes. Cost is O(canonical size) once — independent of how many steps fed it.
    #[allow(
        clippy::indexing_slicing,
        reason = "the reduction's node_map covers the canonical arena that self.map points into"
    )]
    pub(crate) fn finalize(
        &self,
        outputs: &[NodeId],
        engine: &dyn RewriteEngine,
        mode: FusionMode,
    ) -> Result<Reduced, BadNodeId> {
        let mut outs = Vec::with_capacity(outputs.len());
        for &o in outputs {
            outs.push(self.map.get(index(o)).copied().ok_or(BadNodeId(o))?);
        }
        // `canonicalize` oriented every symmetric op as it recorded it
        let mut red = super::reduce_oriented(&self.canon_nodes, &outs, None, engine, mode);
        // §8.2(i): the four passes answered in CANONICAL-arena ids, so compose this reducer's own
        // original->canonical map in front. Without it the accessor silently mis-keys every
        // `Session(incremental=True)` program.
        let canonical = std::mem::take(&mut red.node_map);
        red.node_map = self.map.iter().map(|&c| canonical[index(c)]).collect();
        Ok(red)
    }
}

#[cfg(test)]
#[cfg(not(loom))]
#[allow(
    clippy::many_single_char_names,
    reason = "test graphs name their nodes a, b, c as the plan's examples do"
)]
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
        // §8.2(i): and the same RECORD-keyed correspondence — the incremental path composes its
        // own canonical map in front of the four passes so both answer in original-arena ids.
        assert_eq!(inc.node_map.len(), s.node_count());
        assert_eq!(inc.node_map, full.node_map);
        assert_eq!(inc.node_map[index(one)], inc.node_map[index(ab)]);
    }

    #[test]
    fn identity_tokens_collapse_only_at_arity_one_like_the_engine() {
        let zero = ParamMap::new(vec![
            ("scalar".into(), ParamValue::Float(0.0)),
            ("side".into(), ParamValue::Str("r".into())),
        ]);
        for arity in [0, 2] {
            let s = GraphStore::new();
            let a = s.add_source("a".into(), empty());
            let b = s.add_source("b".into(), empty());
            let inputs = [a, b][..arity].to_vec();
            let out = s.add_op("add".into(), inputs, zero.clone()).unwrap();
            let engine = EggEngine::default();
            let inc = reduce_incrementally(&s)
                .finalize(&[out], &engine, FusionMode::SingleUse)
                .unwrap();
            let full = crate::optimizer::reduce_with_mode(
                &s.snapshot().0,
                &[out],
                &engine,
                FusionMode::SingleUse,
            );
            assert_eq!(inc.nodes, full.nodes, "arity {arity}");
            assert_eq!(inc.outputs, full.outputs, "arity {arity}");
        }
    }

    #[test]
    fn nodes_equal_after_canonicalizing_their_inputs_share_one_canonical_node() {
        let s = GraphStore::new();
        let a = s.add_source("a".into(), empty());
        let b = s.add_source("b".into(), empty());
        s.add_op("add".into(), vec![a, b], empty()).unwrap();
        let zero = ParamMap::new(vec![
            ("scalar".into(), ParamValue::Float(0.0)),
            ("side".into(), ParamValue::Str("r".into())),
        ]);
        let x = s.add_op("add".into(), vec![a], zero).unwrap(); // identity: x is a
        s.add_op("add".into(), vec![x, b], empty()).unwrap(); // the symmetric add(a, b) again
        s.add_op("neg".into(), vec![a], empty()).unwrap();
        s.add_op("neg".into(), vec![x], empty()).unwrap(); // the plain neg(a) again
        let r = reduce_incrementally(&s);
        assert_eq!(r.total_work(), 7);
        assert_eq!(r.canonical_count(), 4, "a, b, add(a, b), neg(a)");
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

    #[test]
    fn a_commuted_twin_recorded_first_does_not_reorient_a_later_output() {
        // `(a + b) * b`, alone or after an unrelated `(b + a) * a` that records the other orientation
        let build = |twin_first: bool| {
            let s = GraphStore::new();
            let a = s.add_source("a".into(), empty());
            let b = s.add_source("b".into(), empty());
            if twin_first {
                let ba = s.add_op("add".into(), vec![b, a], empty()).unwrap();
                s.add_op("sub".into(), vec![ba, a], empty()).unwrap();
            }
            let ab = s.add_op("add".into(), vec![a, b], empty()).unwrap();
            let out = s.add_op("sub".into(), vec![ab, b], empty()).unwrap();
            let inc = reduce_incrementally(&s)
                .finalize(&[out], &EggEngine::default(), FusionMode::SingleUse)
                .unwrap();
            let (one_shot, _) = s
                .reduce_with_outputs(&[out], &EggEngine::default(), FusionMode::SingleUse)
                .unwrap();
            (
                crate::serialize::serialize(&GraphStore::from_reduced(inc).0).unwrap(),
                crate::serialize::serialize(&one_shot).unwrap(),
            )
        };
        let (fresh_inc, fresh_one_shot) = build(false);
        let (inc, one_shot) = build(true);
        assert_eq!(one_shot, fresh_one_shot, "one-shot control");
        assert_eq!(inc, fresh_inc);
        assert_eq!(inc, one_shot);
    }

    #[test]
    fn structural_key_is_fnv1a_and_invariant_under_the_engine_rules() {
        let s = GraphStore::new();
        let a = s.add_source("a".into(), empty());
        let b = s.add_source("b".into(), empty());
        let ab = s.add_op("add".into(), vec![a, b], empty()).unwrap();
        let ba = s.add_op("add".into(), vec![b, a], empty()).unwrap();
        let sub = s.add_op("sub".into(), vec![a, b], empty()).unwrap();
        let bsub = s.add_op("sub".into(), vec![b, a], empty()).unwrap();
        let ident = s
            .add_op(
                "add".into(),
                vec![a],
                ParamMap::new(vec![
                    ("scalar".into(), ParamValue::Float(0.0)),
                    ("side".into(), ParamValue::Str("r".into())),
                ]),
            )
            .unwrap();
        let (nodes, _) = s.snapshot();
        let mut keys: Vec<u128> = Vec::new();
        for n in &nodes {
            let ins: Vec<u128> = n.inputs().iter().map(|&i| keys[index(i)]).collect();
            keys.push(structural_key(n, &n.token(), &ins));
        }
        let k = |id: NodeId| keys[index(id)];
        // FNV-1a 128 of the source token `src|a` behind its u64 length: a release-stable value
        assert_eq!(k(a), 0x6168_9e05_c825_02fc_5e5e_1696_7b11_fd3d);
        assert_eq!(k(ab), k(ba), "commuted twins share a key");
        assert_ne!(k(sub), k(bsub), "an asymmetric op keeps its operand order");
        assert_eq!(k(ident), k(a), "an identity op is its input");
        assert_ne!(k(a), k(b));
    }
}
