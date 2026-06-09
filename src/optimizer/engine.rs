//! The `RewriteEngine` boundary and its `egg` implementation (plan M4).
//!
//! The trait inputs/outputs are egg-free (`EngineGraph`), so no `egg` types leak past it — this is
//! what lets Phase 2 swap in `egglog`. Canonicalization is *genuine equality saturation*: the IR is
//! loaded into an e-graph and SOUND rewrite rules run to (bounded) saturation, computing the
//! equivalence classes of the IR. DCE/CSE are NOT here — they are plain passes outside the engine.
//!
//! Extraction note: egg's recursive `Extractor` is O(depth × nodes) and blows up to O(N²) on the
//! deep chains a systematics graph produces (caught by the M4 super-linear benchmark). Since our
//! sound rules only ever make an *existing, earlier* node the canonical form (commutativity:
//! `add(a,b)`/`add(b,a)`; identity: `x+0`≡`x`), we extract in O(N) by quotienting the IR by the
//! e-graph's equivalence and keeping the earliest node of each class — a cost-based choice that is
//! both topologically safe and prefers the simpler form.

use std::collections::HashMap;

use egg::{EGraph, Id, Pattern, Rewrite, Runner, Symbol, SymbolLang};

/// The symmetric binary ops in the frontend vocabulary (argument order is semantically
/// irrelevant). ONE list shared by the egg rule set and the incremental canonicalizer, so the two
/// reduction paths agree node-for-node. Asymmetric ops (sub/div/lt/...) are deliberately absent.
pub const SYMMETRIC_OPS: &[&str] = &["add", "mul", "and", "or", "eq", "ne", "maximum", "minimum"];

/// Identity-elimination tokens: an op with this exact token is equivalent to its single input
/// (`x + 0.0` / `0.0 + x` / `x * 1.0` / `1.0 * x`). Shared with the incremental canonicalizer.
pub const IDENTITY_TOKENS: &[&str] = &[
    "op|add|scalar=f0000000000000000;side=sr",
    "op|add|scalar=f0000000000000000;side=sl",
    "op|mul|scalar=f3ff0000000000000;side=sr",
    "op|mul|scalar=f3ff0000000000000;side=sl",
];

/// An egg-free node view: an operator `token`, whether it is a boundary, and input indices.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EngineNode {
    pub token: String,
    pub boundary: bool,
    pub inputs: Vec<usize>,
}

/// A topologically-ordered DAG passed across the engine boundary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EngineGraph {
    pub nodes: Vec<EngineNode>,
    pub outputs: Vec<usize>,
}

/// The swappable optimizer engine boundary (egg today, egglog in Phase 2).
pub trait RewriteEngine {
    /// Canonicalize the DAG via equality saturation.
    fn canonicalize(&self, graph: &EngineGraph) -> EngineGraph;
}

fn boundary_from_token(token: &str) -> bool {
    !token.starts_with("op|")
}

/// The MVP engine: `egg` over `SymbolLang`, with a SOUND rule set only.
pub struct EggEngine {
    rules: Vec<Rewrite<SymbolLang, ()>>,
    /// Deterministic saturation budget (no wall-clock limit — that would be non-deterministic).
    iter_limit: usize,
}

impl Default for EggEngine {
    fn default() -> Self {
        // Only provably-sound rewrites: commutativity of every symmetric op in the vocabulary, and
        // the additive/multiplicative identities. Domain-dependent rules (mask-fusion,
        // field-collapse) are intentionally excluded. Generated from the shared SYMMETRIC_OPS /
        // IDENTITY_TOKENS constants so the incremental canonicalizer provably uses the same rules.
        let mut rules: Vec<Rewrite<SymbolLang, ()>> = Vec::new();
        for op in SYMMETRIC_OPS {
            let lhs: Pattern<SymbolLang> = format!("(op|{op} ?a ?b)").parse().unwrap();
            let rhs: Pattern<SymbolLang> = format!("(op|{op} ?b ?a)").parse().unwrap();
            rules.push(Rewrite::new(format!("commute-{op}"), lhs, rhs).unwrap());
        }
        for (i, tok) in IDENTITY_TOKENS.iter().enumerate() {
            let lhs: Pattern<SymbolLang> = format!("({tok} ?x)").parse().unwrap();
            let rhs: Pattern<SymbolLang> = "?x".parse().unwrap();
            rules.push(Rewrite::new(format!("identity-{i}"), lhs, rhs).unwrap());
        }
        EggEngine {
            rules,
            iter_limit: 12,
        }
    }
}

impl RewriteEngine for EggEngine {
    fn canonicalize(&self, graph: &EngineGraph) -> EngineGraph {
        // 1. load the DAG into an e-graph, recording each node's eclass (topo order).
        let mut egraph: EGraph<SymbolLang, ()> = EGraph::default();
        let mut node_eclass: Vec<Id> = Vec::with_capacity(graph.nodes.len());
        for node in &graph.nodes {
            let children: Vec<Id> = node.inputs.iter().map(|&i| node_eclass[i]).collect();
            node_eclass
                .push(egraph.add(SymbolLang::new(Symbol::from(node.token.as_str()), children)));
        }

        // 2. saturate with a deterministic budget (egg merges equivalent eclasses).
        let runner = Runner::default()
            .with_iter_limit(self.iter_limit)
            .with_node_limit(graph.nodes.len().saturating_mul(4) + 1024)
            .with_egraph(egraph)
            .run(&self.rules);

        // 3. extract in O(N): quotient the IR by the e-graph equivalence, keeping the EARLIEST node
        //    of each class (topologically safe + prefers the simpler form for our rules).
        let canon: Vec<Id> = node_eclass
            .iter()
            .map(|&id| runner.egraph.find(id))
            .collect();
        let mut rep_of_class: HashMap<Id, usize> = HashMap::new();
        for (i, &c) in canon.iter().enumerate() {
            rep_of_class.entry(c).or_insert(i);
        }

        let mut class_to_new: HashMap<Id, usize> = HashMap::new();
        let mut nodes: Vec<EngineNode> = Vec::new();
        for (i, node) in graph.nodes.iter().enumerate() {
            if rep_of_class[&canon[i]] != i {
                continue; // a duplicate of an earlier-equivalent node
            }
            let inputs: Vec<usize> = node
                .inputs
                .iter()
                .map(|&j| class_to_new[&canon[j]])
                .collect();
            class_to_new.insert(canon[i], nodes.len());
            nodes.push(EngineNode {
                token: node.token.clone(),
                boundary: boundary_from_token(&node.token),
                inputs,
            });
        }
        let outputs = graph
            .outputs
            .iter()
            .map(|&o| class_to_new[&canon[o]])
            .collect();
        EngineGraph { nodes, outputs }
    }
}
