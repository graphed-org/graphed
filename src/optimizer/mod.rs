//! The M4 reduction pipeline: DCE + CSE (outside the engine) and stage fusion, around the
//! equality-saturation `RewriteEngine` (egg).
//!
//! ```text
//! reduce(graph):
//!   reachable = dead_code_elimination(graph, outputs)   # plain reachability pass
//!   canonical = engine.canonicalize(reachable)          # genuine equality saturation (egg)
//!   deduped   = cse(canonical)                           # hash-consing (M1 property, re-asserted)
//!   stages    = stage_fusion(deduped)                    # maximal op-runs between boundaries
//!   rebuild stages into a fresh interned GraphStore      # interning gives CSE again
//! ```

mod engine;

use std::collections::HashMap;

pub use engine::{EggEngine, EngineGraph, EngineNode, RewriteEngine};

use crate::node::{NodeId, NodeKey, StageOp, StageRef};

/// Stats reported for one reduction (plan M4 `reduction_report`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ReductionReport {
    pub input_nodes: usize,
    pub reachable_nodes: usize,
    pub canonical_nodes: usize,
    pub stages: usize,
    pub reduced_nodes: usize,
    pub boundary_nodes: usize,
}

/// The reduced graph: interned NodeKeys (topological), remapped outputs, and the report.
pub struct Reduced {
    pub nodes: Vec<NodeKey>,
    pub outputs: Vec<NodeId>,
    pub report: ReductionReport,
}

/// Run the full reduction. `nodes` is the interned arena (topological by id); `outputs` are marked
/// output ids. Returns reduced NodeKeys whose inputs reference indices *within the returned list*.
pub fn reduce(nodes: &[NodeKey], outputs: &[NodeId], engine: &dyn RewriteEngine) -> Reduced {
    // 1. DCE — keep only nodes reachable from the outputs.
    let (reachable_keys, reachable_outputs) = dead_code_elimination(nodes, outputs);

    // operator-token -> a NodeKey template, for reconstructing boundary nodes after canonicalization
    let mut templates: HashMap<String, NodeKey> = HashMap::new();
    for key in &reachable_keys {
        templates.entry(key.token()).or_insert_with(|| key.clone());
    }

    // 2. canonicalize via the engine (equality saturation behind RewriteEngine).
    let eg = to_engine_graph(&reachable_keys, &reachable_outputs);
    let canonical = engine.canonicalize(&eg);

    // 3. CSE — hash-cons identical (token, inputs) nodes (a plain pass, not the engine).
    let deduped = cse(&canonical);

    // 4. stage fusion + rebuild.
    let mut out = stage_fusion(&deduped, &templates);
    out.report.input_nodes = nodes.len();
    out.report.reachable_nodes = reachable_keys.len();
    out.report.canonical_nodes = deduped.nodes.len();
    out
}

/// Reachability from the outputs (plan M4: DCE = reachability; never drops a node on a path to an
/// output). Returns the reachable nodes compacted into topological order + remapped outputs.
pub fn dead_code_elimination(nodes: &[NodeKey], outputs: &[NodeId]) -> (Vec<NodeKey>, Vec<usize>) {
    let mut keep = vec![false; nodes.len()];
    let mut stack: Vec<usize> = outputs.iter().map(|&o| o as usize).collect();
    while let Some(i) = stack.pop() {
        if keep[i] {
            continue;
        }
        keep[i] = true;
        for &inp in nodes[i].inputs() {
            stack.push(inp as usize);
        }
    }
    // compact, preserving topological (ascending-id) order
    let mut remap = vec![usize::MAX; nodes.len()];
    let mut kept: Vec<NodeKey> = Vec::new();
    for (i, node) in nodes.iter().enumerate() {
        if keep[i] {
            remap[i] = kept.len();
            let new_inputs: Vec<NodeId> = node
                .inputs()
                .iter()
                .map(|&x| remap[x as usize] as NodeId)
                .collect();
            kept.push(node.with_inputs(new_inputs));
        }
    }
    let out_idx = outputs.iter().map(|&o| remap[o as usize]).collect();
    (kept, out_idx)
}

fn to_engine_graph(keys: &[NodeKey], outputs: &[usize]) -> EngineGraph {
    let nodes = keys
        .iter()
        .map(|k| EngineNode {
            token: k.token(),
            boundary: k.is_boundary(),
            inputs: k.inputs().iter().map(|&i| i as usize).collect(),
        })
        .collect();
    EngineGraph {
        nodes,
        outputs: outputs.to_vec(),
    }
}

/// Hash-cons identical (token, inputs) nodes — CSE as a plain pass.
fn cse(graph: &EngineGraph) -> EngineGraph {
    let mut remap = vec![0usize; graph.nodes.len()];
    let mut seen: HashMap<(String, Vec<usize>), usize> = HashMap::new();
    let mut nodes: Vec<EngineNode> = Vec::new();
    for (i, node) in graph.nodes.iter().enumerate() {
        let inputs: Vec<usize> = node.inputs.iter().map(|&x| remap[x]).collect();
        let key = (node.token.clone(), inputs.clone());
        let idx = *seen.entry(key).or_insert_with(|| {
            nodes.push(EngineNode {
                token: node.token.clone(),
                boundary: node.boundary,
                inputs,
            });
            nodes.len() - 1
        });
        remap[i] = idx;
    }
    let outputs = graph.outputs.iter().map(|&o| remap[o]).collect();
    EngineGraph { nodes, outputs }
}

// ---- union-find over op nodes ------------------------------------------------
struct Dsu {
    parent: Vec<usize>,
}
impl Dsu {
    fn new(n: usize) -> Self {
        Dsu {
            parent: (0..n).collect(),
        }
    }
    fn find(&mut self, x: usize) -> usize {
        let mut r = x;
        while self.parent[r] != r {
            r = self.parent[r];
        }
        let mut c = x;
        while self.parent[c] != c {
            let n = self.parent[c];
            self.parent[c] = r;
            c = n;
        }
        r
    }
    fn union(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            // keep the smaller root so head selection (max index) is independent of union order
            self.parent[ra.max(rb)] = ra.min(rb);
        }
    }
}

/// Group maximal runs of ops between boundaries into `Stage` nodes (fusion never crosses a
/// boundary). An op fuses into its consumer only when it has exactly one use and that use is an op.
fn stage_fusion(graph: &EngineGraph, templates: &HashMap<String, NodeKey>) -> Reduced {
    let n = graph.nodes.len();
    // consumers + output marks
    let mut consumers: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (i, node) in graph.nodes.iter().enumerate() {
        for &inp in &node.inputs {
            consumers[inp].push(i);
        }
    }
    let mut is_output = vec![false; n];
    for &o in &graph.outputs {
        is_output[o] = true;
    }

    // union ops whose single use is another op
    let mut dsu = Dsu::new(n);
    for (i, node) in graph.nodes.iter().enumerate() {
        if node.boundary {
            continue;
        }
        if consumers[i].len() == 1 && !is_output[i] {
            let c = consumers[i][0];
            if !graph.nodes[c].boundary {
                dsu.union(i, c);
            }
        }
    }

    // component id per node (ops -> dsu root; boundaries -> themselves)
    let comp_of = |dsu: &mut Dsu, i: usize| {
        if graph.nodes[i].boundary {
            i
        } else {
            dsu.find(i)
        }
    };
    let comp_id: Vec<usize> = (0..n).map(|i| comp_of(&mut dsu, i)).collect();

    // build comp -> member node indices
    let mut comp_members: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, &c) in comp_id.iter().enumerate() {
        comp_members.entry(c).or_default().push(i);
    }

    // topological order of comps (by max member index = the comp's head/output)
    let mut comps: Vec<usize> = comp_members.keys().copied().collect();
    comps.sort_by_key(|c| *comp_members[c].iter().max().unwrap());

    let mut reduced: Vec<NodeKey> = Vec::new();
    let mut comp_to_reduced: HashMap<usize, NodeId> = HashMap::new();
    let mut stages = 0usize;
    let mut boundary_nodes = 0usize;

    for &comp in &comps {
        let members = &comp_members[&comp];
        if graph.nodes[comp].boundary {
            // a single boundary node -> reduced as itself
            let node = &graph.nodes[comp];
            let inputs: Vec<NodeId> = node
                .inputs
                .iter()
                .map(|&i| comp_to_reduced[&comp_id[i]])
                .collect();
            let template = &templates[&node.token];
            reduced.push(template.with_inputs(inputs));
            boundary_nodes += 1;
        } else {
            // an op-component -> a fused Stage
            let mut ops: Vec<usize> = members.clone();
            ops.sort_unstable();
            let local: HashMap<usize, usize> =
                ops.iter().enumerate().map(|(k, &v)| (v, k)).collect();
            let mut input_slots: Vec<usize> = Vec::new(); // external comps, in first-seen order
            let mut slot_of: HashMap<usize, usize> = HashMap::new();
            let mut stage_ops: Vec<StageOp> = Vec::new();
            for &op in &ops {
                let refs = graph.nodes[op]
                    .inputs
                    .iter()
                    .map(|&inp| {
                        if local.contains_key(&inp) {
                            StageRef::Member(local[&inp])
                        } else {
                            let c = comp_id[inp];
                            let slot = *slot_of.entry(c).or_insert_with(|| {
                                input_slots.push(c);
                                input_slots.len() - 1
                            });
                            StageRef::Input(slot)
                        }
                    })
                    .collect();
                stage_ops.push(StageOp {
                    token: graph.nodes[op].token.clone(),
                    inputs: refs,
                });
            }
            let inputs: Vec<NodeId> = input_slots.iter().map(|&c| comp_to_reduced[&c]).collect();
            reduced.push(NodeKey::Stage {
                inputs,
                members: stage_ops,
            });
            stages += 1;
        }
        comp_to_reduced.insert(comp, (reduced.len() - 1) as NodeId);
    }

    let outputs: Vec<NodeId> = graph
        .outputs
        .iter()
        .map(|&o| comp_to_reduced[&comp_id[o]])
        .collect();
    let reduced_nodes = reduced.len();
    Reduced {
        nodes: reduced,
        outputs,
        report: ReductionReport {
            input_nodes: 0,
            reachable_nodes: 0,
            canonical_nodes: 0,
            stages,
            reduced_nodes,
            boundary_nodes,
        },
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;
    use crate::node::StageRef;
    use crate::param::{ParamMap, ParamValue};
    use crate::store::GraphStore;
    use std::collections::HashMap;

    fn pm(entries: Vec<(&str, ParamValue)>) -> ParamMap {
        ParamMap::new(
            entries
                .into_iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect(),
        )
    }
    fn empty() -> ParamMap {
        ParamMap::new(vec![])
    }

    // ---- a tiny deterministic interpreter (toy integer semantics) ------------
    fn apply_op(name: &str, vals: &[i64]) -> i64 {
        match name {
            "add" => vals.iter().sum(),
            "mul" => vals.iter().product(),
            "neg" => -vals[0],
            "inc" => vals[0] + 1,
            _ => vals[0], // identity for any other unary op
        }
    }
    fn op_name(token: &str) -> &str {
        token
            .strip_prefix("op|")
            .unwrap_or(token)
            .split('|')
            .next()
            .unwrap()
    }
    fn eval(nodes: &[NodeKey], outputs: &[NodeId], seeds: &HashMap<String, i64>) -> Vec<i64> {
        let mut vals = vec![0i64; nodes.len()];
        for (i, node) in nodes.iter().enumerate() {
            vals[i] = match node {
                NodeKey::Source { name, .. } => seeds[name],
                NodeKey::Op { name, inputs, .. } => {
                    let v: Vec<i64> = inputs.iter().map(|&x| vals[x as usize]).collect();
                    apply_op(name, &v)
                }
                NodeKey::Reduction { inputs, .. } | NodeKey::External { inputs, .. } => {
                    vals[inputs[0] as usize]
                }
                NodeKey::Stage { inputs, members } => {
                    let inv: Vec<i64> = inputs.iter().map(|&x| vals[x as usize]).collect();
                    let mut mv: Vec<i64> = Vec::new();
                    for op in members {
                        let ins: Vec<i64> = op
                            .inputs
                            .iter()
                            .map(|r| match r {
                                StageRef::Input(k) => inv[*k],
                                StageRef::Member(j) => mv[*j],
                            })
                            .collect();
                        mv.push(apply_op(op_name(&op.token), &ins));
                    }
                    *mv.last().unwrap()
                }
            };
        }
        outputs.iter().map(|&o| vals[o as usize]).collect()
    }

    #[test]
    fn chain_reduces_to_constant_stages_regardless_of_length() {
        for n in [10usize, 100, 1000] {
            let s = GraphStore::new();
            let src = s.add_source("x".into(), empty());
            let mut cur = src;
            for _ in 0..n {
                cur = s.add_op("inc".into(), vec![cur], empty()).unwrap();
            }
            s.mark_output(cur).unwrap();
            let (reduced, report) = s.reduce(&EggEngine::default());
            // source + exactly one fused stage, independent of n
            assert_eq!(reduced.node_count(), 2, "n={n}");
            assert_eq!(report.stages, 1);
            assert!(report.stages < 10);
        }
    }

    #[test]
    fn reduced_graph_matches_unreduced_semantics() {
        let s = GraphStore::new();
        let a = s.add_source("a".into(), empty());
        let b = s.add_source("b".into(), empty());
        let ab = s.add_op("add".into(), vec![a, b], empty()).unwrap();
        let inc = s.add_op("inc".into(), vec![ab], empty()).unwrap();
        let out = s.add_op("mul".into(), vec![inc, a], empty()).unwrap();
        s.mark_output(out).unwrap();

        let (nodes, outs) = s.snapshot();
        let (reduced, _) = s.reduce(&EggEngine::default());
        let (rnodes, routs) = reduced.snapshot();

        let seeds: HashMap<String, i64> = [("a".to_string(), 3i64), ("b".to_string(), 5i64)].into();
        // (a+b)+1)*a = (3+5+1)*3 = 27
        assert_eq!(eval(&nodes, &outs, &seeds), vec![27]);
        assert_eq!(eval(&rnodes, &routs, &seeds), eval(&nodes, &outs, &seeds));
    }

    #[test]
    fn dce_drops_unreachable_but_never_on_a_path_to_output() {
        let s = GraphStore::new();
        let a = s.add_source("a".into(), empty());
        let used = s.add_op("inc".into(), vec![a], empty()).unwrap();
        let _dead = s.add_op("neg".into(), vec![a], empty()).unwrap(); // never marked output
        s.mark_output(used).unwrap();
        let (nodes, outs) = s.snapshot();
        let (kept, _kout) = dead_code_elimination(&nodes, &outs);
        // a + used kept; dead dropped
        assert_eq!(kept.len(), 2);
        // every node on the path to the output survives
        let (reduced, _) = s.reduce(&EggEngine::default());
        assert_eq!(reduced.node_count(), 2); // source + stage(inc)
    }

    #[test]
    fn fusion_never_crosses_a_boundary() {
        let s = GraphStore::new();
        let src = s.add_source("x".into(), empty());
        let pre = s.add_op("inc".into(), vec![src], empty()).unwrap();
        let red = s.add_reduction("sum".into(), vec![pre], empty()).unwrap(); // boundary
        let post = s.add_op("inc".into(), vec![red], empty()).unwrap();
        s.mark_output(post).unwrap();
        let (reduced, report) = s.reduce(&EggEngine::default());
        // source, stage(pre), reduction, stage(post) -> 4 nodes; the reduction splits the stages
        assert_eq!(report.stages, 2);
        assert_eq!(reduced.node_count(), 4);
    }

    #[test]
    fn commuted_forms_merge_via_equality_saturation() {
        let s = GraphStore::new();
        let a = s.add_source("a".into(), empty());
        let b = s.add_source("b".into(), empty());
        let ab = s.add_op("add".into(), vec![a, b], empty()).unwrap();
        let ba = s.add_op("add".into(), vec![b, a], empty()).unwrap();
        assert_ne!(ab, ba); // distinct in the un-reduced graph
        s.mark_output(ab).unwrap();
        s.mark_output(ba).unwrap();
        let (reduced, report) = s.reduce(&EggEngine::default());
        // egg commutativity + CSE collapse a+b and b+a into ONE stage
        assert_eq!(report.stages, 1, "commuted adds should merge");
        assert_eq!(reduced.node_count(), 3); // a, b, one stage
    }

    #[test]
    fn additive_identity_is_simplified_away() {
        let s = GraphStore::new();
        let x = s.add_source("x".into(), empty());
        let zero = s
            .add_op(
                "add".into(),
                vec![x],
                pm(vec![
                    ("scalar", ParamValue::Float(0.0)),
                    ("side", ParamValue::Str("r".into())),
                ]),
            )
            .unwrap();
        s.mark_output(zero).unwrap();
        let (_reduced, report) = s.reduce(&EggEngine::default());
        // x + 0 -> x : no op remains, the output is the source itself (zero stages)
        assert_eq!(report.stages, 0);
    }

    #[test]
    fn reduction_is_deterministic() {
        let build = || {
            let s = GraphStore::new();
            let a = s.add_source("a".into(), empty());
            let b = s.add_source("b".into(), empty());
            let c = s.add_op("add".into(), vec![a, b], empty()).unwrap();
            let d = s.add_op("inc".into(), vec![c], empty()).unwrap();
            s.mark_output(d).unwrap();
            s.reduce(&EggEngine::default()).0.to_dot()
        };
        assert_eq!(build(), build());
    }

    // ---- complex topologies (diamond / star / nested) — the dask-optimizer failure points -------

    fn seeds(pairs: &[(&str, i64)]) -> HashMap<String, i64> {
        pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    #[test]
    fn diamond_shares_apex_and_round_trips() {
        // x -> a; a fans out to two DISTINCT branches that re-converge at out (a classic diamond).
        let s = GraphStore::new();
        let x = s.add_source("x".into(), empty());
        let a = s.add_op("inc".into(), vec![x], empty()).unwrap(); // apex, out-degree 2
        let l = s.add_op("inc".into(), vec![a], empty()).unwrap();
        let r = s.add_op("neg".into(), vec![a], empty()).unwrap();
        let out = s.add_op("add".into(), vec![l, r], empty()).unwrap();
        s.mark_output(out).unwrap();
        assert_eq!(s.node_count(), 5); // referencing `a` twice interns to ONE node (CSE), not two

        let (nodes, outs) = s.snapshot();
        let (reduced, report) = s.reduce(&EggEngine::default());
        let (rn, ro) = reduced.snapshot();
        let sd = seeds(&[("x", 5)]);
        // (x+1)+1 + -(x+1) = (x+2) - (x+1) = 1
        assert_eq!(eval(&nodes, &outs, &sd), vec![1]);
        assert_eq!(eval(&rn, &ro, &sd), eval(&nodes, &outs, &sd));
        // the fan-out apex is NOT duplicated into both branches: it stays its own stage
        assert_eq!(report.stages, 2); // stage(a) + stage(l, r, out)
        assert_eq!(reduced.node_count(), 3); // source + apex-stage + branch-stage
    }

    #[test]
    fn dce_keeps_a_node_reachable_via_two_paths() {
        // a diamond plus a genuinely dead branch off the apex; DCE keeps the whole diamond (the apex
        // is reachable via BOTH branches) and drops only the dead op.
        let s = GraphStore::new();
        let x = s.add_source("x".into(), empty());
        let a = s.add_op("inc".into(), vec![x], empty()).unwrap();
        let l = s.add_op("inc".into(), vec![a], empty()).unwrap();
        let r = s.add_op("neg".into(), vec![a], empty()).unwrap();
        let out = s.add_op("add".into(), vec![l, r], empty()).unwrap();
        let _dead = s.add_op("mul".into(), vec![a, x], empty()).unwrap(); // off the apex, not an output
        s.mark_output(out).unwrap();
        let (nodes, outs) = s.snapshot();
        let (kept, _) = dead_code_elimination(&nodes, &outs);
        assert_eq!(kept.len(), 5); // x, a, l, r, out kept; the dead mul dropped
        let sd = seeds(&[("x", 5)]);
        let (reduced, _) = s.reduce(&EggEngine::default());
        let (rn, ro) = reduced.snapshot();
        assert_eq!(eval(&rn, &ro, &sd), eval(&nodes, &outs, &sd));
    }

    #[test]
    fn star_fan_out_and_fan_in_round_trip_with_a_shared_hub() {
        // one hub fans out to N distinct consumers (each takes a distinct extra source), and they all
        // fan in to a single add. The hub is interned once and never duplicated.
        let n = 16usize;
        let s = GraphStore::new();
        let x = s.add_source("x".into(), empty());
        let hub = s.add_op("inc".into(), vec![x], empty()).unwrap();
        let names: Vec<String> = (0..n).map(|i| format!("s{i}")).collect();
        let mut leaves = Vec::new();
        for nm in &names {
            let si = s.add_source(nm.clone(), empty());
            leaves.push(s.add_op("add".into(), vec![hub, si], empty()).unwrap());
        }
        let out = s.add_op("add".into(), leaves.clone(), empty()).unwrap();
        s.mark_output(out).unwrap();

        let mut pairs = vec![("x", 2i64)];
        for nm in &names {
            pairs.push((nm.as_str(), 1));
        }
        let sd = seeds(&pairs);
        let (nodes, outs) = s.snapshot();
        let (reduced, _) = s.reduce(&EggEngine::default());
        let (rn, ro) = reduced.snapshot();
        // hub = x+1 = 3; each leaf = hub + 1 = 4; out = sum = n*4
        let expect = vec![(n as i64) * 4];
        assert_eq!(eval(&nodes, &outs, &sd), expect);
        assert_eq!(eval(&rn, &ro, &sd), expect);
        // the hub is a single shared node feeding all N leaves
        assert_eq!(
            nodes.iter().filter(|k| k.inputs().contains(&hub)).count(),
            n
        );
    }

    #[test]
    fn nested_stacked_diamonds_collapse_to_constant() {
        // D diamonds stacked: step(v) = (v+1) + -(v) = 1, so the tower is 1 for any x and any D>=1.
        // Stresses deep nesting + many fan-out apexes; must stay correct AND grow at most linearly.
        for d in [1usize, 4, 16, 64] {
            let s = GraphStore::new();
            let x = s.add_source("x".into(), empty());
            let mut v = x;
            for _ in 0..d {
                let l = s.add_op("inc".into(), vec![v], empty()).unwrap();
                let r = s.add_op("neg".into(), vec![v], empty()).unwrap();
                v = s.add_op("add".into(), vec![l, r], empty()).unwrap();
            }
            s.mark_output(v).unwrap();
            let (nodes, outs) = s.snapshot();
            let (reduced, _) = s.reduce(&EggEngine::default());
            let (rn, ro) = reduced.snapshot();
            let sd = seeds(&[("x", 7)]);
            assert_eq!(eval(&nodes, &outs, &sd), vec![1], "d={d}");
            assert_eq!(eval(&rn, &ro, &sd), vec![1], "d={d}");
            assert!(reduced.node_count() <= 3 * d + 2, "d={d}: no blow-up");
        }
    }

    #[test]
    fn complex_mixed_topology_round_trips_and_is_deterministic() {
        let build = || {
            let s = GraphStore::new();
            let a = s.add_source("a".into(), empty());
            let b = s.add_source("b".into(), empty());
            let hub = s.add_op("add".into(), vec![a, b], empty()).unwrap(); // a shared hub (fan-out 3)
            let l = s.add_op("inc".into(), vec![hub], empty()).unwrap();
            let r = s.add_op("neg".into(), vec![hub], empty()).unwrap();
            let d = s.add_op("add".into(), vec![l, r], empty()).unwrap(); // diamond off the hub
            let p = s.add_op("inc".into(), vec![hub], empty()).unwrap();
            let red = s.add_reduction("sum".into(), vec![p], empty()).unwrap(); // a boundary in a branch
            let post = s.add_op("mul".into(), vec![d, red], empty()).unwrap();
            s.mark_output(post).unwrap();
            s
        };
        let s = build();
        let (nodes, outs) = s.snapshot();
        let (reduced, _) = s.reduce(&EggEngine::default());
        let (rn, ro) = reduced.snapshot();
        let sd = seeds(&[("a", 3), ("b", 4)]);
        // hub=7; d=(7+1)+-(7)=1; p=8; red=8 (toy reduction passthrough); post=d*red=8
        let expected = eval(&nodes, &outs, &sd);
        assert_eq!(expected, vec![8]);
        assert_eq!(eval(&rn, &ro, &sd), expected);
        assert_eq!(
            build().reduce(&EggEngine::default()).0.to_dot(),
            build().reduce(&EggEngine::default()).0.to_dot()
        );
    }
}
