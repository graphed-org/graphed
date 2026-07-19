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

/// A reference inside a fused `Stage`: either one of the stage's external inputs or an earlier
/// member op. This makes a Stage a self-contained, executable, hashable mini-DAG.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum StageRef {
    Input(usize),
    Member(usize),
}

/// One fused op inside a `Stage` (the M4 optimizer groups a maximal run of ops into a stage).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct StageOp {
    pub token: String,
    pub inputs: Vec<StageRef>,
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
    /// A fused run of ops between boundaries, produced by the M4 optimizer. `members` is the
    /// internal op-DAG (last member is the stage's output); `inputs` are the boundary nodes it
    /// consumes.
    Stage {
        inputs: Vec<NodeId>,
        members: Vec<StageOp>,
    },
    /// A pure data-movement boundary (plan M39 §2.1): its one logical input's rows are
    /// repartitioned according to `scheme` (hash|range|coalesce|split, key, parts, seed, …). Its
    /// identity IS its `scheme` ParamMap — it is NOT an `Op`, so it interns/CSEs like every node,
    /// ends a stage (fusion never crosses it), and survives reduction as itself.
    Exchange {
        scheme: ParamMap,
        inputs: Vec<NodeId>,
    },
    /// A local co-partitioned relational combine (plan M40 §2.1): its two inputs `[left, right]`
    /// (both co-partitioned `Exchange`es on the same key + `parts`) are joined per `scheme`
    /// (`how=inner|left|outer`, `on=<field>`). Like `Exchange` it is NOT an `Op`, so it interns/CSEs,
    /// ends a stage (fusion never crosses it), and survives reduction as itself. `inputs` is a flat
    /// `Vec` by the fixed `[left, right]` convention, which `with_inputs` preserves in order.
    Join {
        scheme: ParamMap,
        inputs: Vec<NodeId>,
    },
}

impl NodeKey {
    pub fn inputs(&self) -> &[NodeId] {
        match self {
            NodeKey::Source { .. } => &[],
            NodeKey::Op { inputs, .. }
            | NodeKey::Reduction { inputs, .. }
            | NodeKey::External { inputs, .. }
            | NodeKey::Stage { inputs, .. }
            | NodeKey::Exchange { inputs, .. }
            | NodeKey::Join { inputs, .. } => inputs,
        }
    }

    /// True for boundary nodes (source / reduction / external / stage) — fusion never merges
    /// across these (plan A.6 boundary op).
    pub fn is_boundary(&self) -> bool {
        !matches!(self, NodeKey::Op { .. })
    }

    /// The "operator" identity of a node, excluding inputs — the egg symbol and the key into the
    /// reconstruction table. Two structurally identical operators share a token; inputs are
    /// children in the e-graph, not part of the token.
    pub fn token(&self) -> String {
        use crate::param::escape_token as esc;
        match self {
            NodeKey::Source { name, params } => with_params(format!("src|{}", esc(name)), params),
            NodeKey::Op { name, params, .. } => with_params(format!("op|{}", esc(name)), params),
            NodeKey::Reduction { name, params, .. } => {
                with_params(format!("red|{}", esc(name)), params)
            }
            NodeKey::External {
                descriptor, params, ..
            } => with_params(
                format!(
                    "ext|{}|{}|{}|{}|{}|{}",
                    descriptor.kind,
                    descriptor.content_hash,
                    descriptor.framework,
                    descriptor.version,
                    descriptor.io_schema,
                    descriptor.preprocessing_ref.as_deref().unwrap_or("-")
                ),
                params,
            ),
            NodeKey::Stage { members, .. } => format!("stage|{}", members.len()),
            // a non-`op|` prefix, so the optimizer's `boundary_from_token` treats it as a boundary;
            // the scheme is in the token so distinct schemes are distinct templates on reconstruction.
            NodeKey::Exchange { scheme, .. } => with_params("exch".to_string(), scheme),
            // a non-`op|` prefix so `boundary_from_token` treats it as a boundary; the two ordered
            // inputs are e-graph children (build vs probe asymmetry preserved), the scheme is in the
            // token so distinct join types/keys are distinct templates on reconstruction.
            NodeKey::Join { scheme, .. } => with_params("join".to_string(), scheme),
        }
    }

    /// Clone this node's operator identity with a fresh input list (used when rebuilding the
    /// reduced graph). Sources ignore inputs.
    pub fn with_inputs(&self, new_inputs: Vec<NodeId>) -> NodeKey {
        match self {
            NodeKey::Source { name, params } => NodeKey::Source {
                name: name.clone(),
                params: params.clone(),
            },
            NodeKey::Op { name, params, .. } => NodeKey::Op {
                name: name.clone(),
                params: params.clone(),
                inputs: new_inputs,
            },
            NodeKey::Reduction { name, params, .. } => NodeKey::Reduction {
                name: name.clone(),
                params: params.clone(),
                inputs: new_inputs,
            },
            NodeKey::External {
                descriptor, params, ..
            } => NodeKey::External {
                descriptor: descriptor.clone(),
                params: params.clone(),
                inputs: new_inputs,
            },
            NodeKey::Stage { members, .. } => NodeKey::Stage {
                inputs: new_inputs,
                members: members.clone(),
            },
            NodeKey::Exchange { scheme, .. } => NodeKey::Exchange {
                scheme: scheme.clone(),
                inputs: new_inputs,
            },
            NodeKey::Join { scheme, .. } => NodeKey::Join {
                scheme: scheme.clone(),
                inputs: new_inputs,
            },
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
            NodeKey::Exchange { scheme, .. } => {
                if scheme.is_empty() {
                    "Exchange".to_string()
                } else {
                    format!("Exchange {scheme}")
                }
            }
            NodeKey::Join { scheme, .. } => {
                if scheme.is_empty() {
                    "Join".to_string()
                } else {
                    format!("Join {scheme}")
                }
            }
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

/// Append a compact, injective, whitespace-free param encoding to a token prefix.
fn with_params(prefix: String, params: &ParamMap) -> String {
    if params.is_empty() {
        prefix
    } else {
        format!("{prefix}|{}", params.token())
    }
}

/// Decode an op-shaped token (`op|<name>` / `red|<name>`, optionally `|<params>`) back into its
/// `(kind, name, params)` parts — the inverse of `token()` for the tokens a fused `Stage` member
/// carries. This is what lets `GraphStore.nodes()` expose stage members as executable
/// (name, params) pairs for IR-driven evaluation, without changing the stage's structural identity
/// or the serialized form. Returns `None` for non-op tokens (sources/externals never fuse).
pub fn parse_op_token(token: &str) -> Option<(String, String, ParamMap)> {
    use crate::param::unescape_token;
    let mut parts = token.splitn(3, '|');
    let kind = parts.next()?;
    if kind != "op" && kind != "red" {
        return None;
    }
    let name = unescape_token(parts.next()?);
    let params = match parts.next() {
        Some(p) => ParamMap::from_token(p)?,
        None => ParamMap::new(vec![]),
    };
    Some((kind.to_string(), name, params))
}

impl fmt::Display for NodeKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.label())
    }
}
