//! The IR node and its structural key (plan M1).
//!
//! A node *is* its structural key: two nodes with equal `NodeKey` are the same node and intern to
//! one `NodeId`. `External` carries a full `PayloadDescriptor` so external inputs are reproducibly
//! identified, and that descriptor participates in the structural identity (plan A.3.1 / M1).

use std::fmt;

use crate::param::ParamMap;

pub type NodeId = u64;

/// Reproducibility metadata an `External` node carries (plan A.6 "Payload descriptor").
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PayloadDescriptor {
    pub kind: String,
    pub content_hash: String,
    pub framework: String,
    pub version: String,
    pub io_schema: String,
    pub preprocessing_ref: Option<String>,
}

/// The structural identity of a node. Equality/Hash here define interning.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum NodeKey {
    Source {
        name: String,
        params: ParamMap,
    },
    Op {
        name: String,
        params: ParamMap,
        inputs: Vec<NodeId>,
    },
    Reduction {
        name: String,
        params: ParamMap,
        inputs: Vec<NodeId>,
    },
    External {
        descriptor: PayloadDescriptor,
        params: ParamMap,
        inputs: Vec<NodeId>,
    },
    // The Stage variant is part of the M1 Node enum per the Acceptance Contract; it is
    // *constructed* by the M4 optimizer (stage fusion), not by any M1 builder.
    #[allow(dead_code)]
    Stage {
        inputs: Vec<NodeId>,
        members: Vec<NodeId>,
    },
}

impl NodeKey {
    pub fn inputs(&self) -> &[NodeId] {
        match self {
            NodeKey::Source { .. } => &[],
            NodeKey::Op { inputs, .. }
            | NodeKey::Reduction { inputs, .. }
            | NodeKey::External { inputs, .. }
            | NodeKey::Stage { inputs, .. } => inputs,
        }
    }

    /// Deterministic graphviz label (used by `to_dot`).
    pub fn label(&self) -> String {
        match self {
            NodeKey::Source { name, params } => fmt_label("Source", name, params),
            NodeKey::Op { name, params, .. } => fmt_label("Op", name, params),
            NodeKey::Reduction { name, params, .. } => fmt_label("Reduction", name, params),
            NodeKey::External {
                descriptor, params, ..
            } => {
                let base = format!(
                    "External {}:{} [{} {}]",
                    descriptor.kind,
                    descriptor.content_hash,
                    descriptor.framework,
                    descriptor.version
                );
                if params.is_empty() {
                    base
                } else {
                    format!("{base} {params}")
                }
            }
            NodeKey::Stage { members, .. } => format!("Stage[{} members]", members.len()),
        }
    }
}

fn fmt_label(kind: &str, name: &str, params: &ParamMap) -> String {
    if params.is_empty() {
        format!("{kind} {name}")
    } else {
        format!("{kind} {name} {params}")
    }
}

impl fmt::Display for NodeKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.label())
    }
}
