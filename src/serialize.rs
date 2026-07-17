//! Versioned, deterministic, byte-identical serialization of the interned IR (plan M8).
//!
//! The **canonical durable form of a graph is this serialized IR** (plan A.3.1: "the serializable
//! IR — not cloudpickle — is the canonical durable representation"). The encoding is a compact
//! length-prefixed binary format with a version header. Determinism is structural: nodes are
//! written in interned-id order (already canonical from M1 hash-consing) and `ParamMap` entries are
//! key-sorted, so **identical graphs serialize to byte-identical output** (the M8 determinism gate)
//! and a round trip rebuilds a structurally identical store (same node count, same `to_dot`, same
//! bytes).

use crate::node::{NodeId, NodeKey, PayloadDescriptor, StageOp, StageRef};
use crate::param::{ParamMap, ParamValue};
use crate::store::GraphStore;

/// Format magic + version. Bumping the trailing byte is the versioning hook (plan M8: "versioned
/// deterministic Plan serialization"); a reader rejects any other magic.
const MAGIC: &[u8; 4] = b"GIR1";

// Node tags.
const T_SOURCE: u8 = 0;
const T_OP: u8 = 1;
const T_REDUCTION: u8 = 2;
const T_EXTERNAL: u8 = 3;
const T_STAGE: u8 = 4;
// M39: the Exchange boundary variant. Tag APPENDED (magic stays GIR1) so tags 0..4 encode
// byte-identically and the M8 determinism gate is untouched.
const T_EXCHANGE: u8 = 5;

// Param value tags (mirror ParamValue order).
const P_INT: u8 = 0;
const P_FLOAT: u8 = 1;
const P_BOOL: u8 = 2;
const P_STR: u8 = 3;

// StageRef tags.
const R_INPUT: u8 = 0;
const R_MEMBER: u8 = 1;

/// Error from decoding a malformed or wrong-version byte string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeError {
    BadMagic,
    Truncated,
    BadTag(&'static str, u8),
    BadUtf8,
    BadNodeRef(u64),
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecodeError::BadMagic => write!(f, "not a graphed IR blob (bad magic/version)"),
            DecodeError::Truncated => write!(f, "truncated graphed IR blob"),
            DecodeError::BadTag(what, t) => write!(f, "invalid {what} tag {t}"),
            DecodeError::BadUtf8 => write!(f, "invalid utf-8 in graphed IR blob"),
            DecodeError::BadNodeRef(i) => write!(f, "node reference {i} out of range"),
        }
    }
}

// ---- writer -------------------------------------------------------------------------------------

fn put_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn put_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn put_str(out: &mut Vec<u8>, s: &str) {
    put_u32(out, s.len() as u32);
    out.extend_from_slice(s.as_bytes());
}

fn put_params(out: &mut Vec<u8>, p: &ParamMap) {
    let entries = p.entries();
    put_u32(out, entries.len() as u32);
    for (k, v) in entries {
        put_str(out, k);
        match v {
            ParamValue::Int(i) => {
                out.push(P_INT);
                put_u64(out, *i as u64);
            }
            ParamValue::Float(x) => {
                out.push(P_FLOAT);
                // raw bits: a round trip reproduces the exact stored value (NaN canonicalization is
                // M1 interning's job, not the codec's).
                put_u64(out, x.to_bits());
            }
            ParamValue::Bool(b) => {
                out.push(P_BOOL);
                out.push(u8::from(*b));
            }
            ParamValue::Str(s) => {
                out.push(P_STR);
                put_str(out, s);
            }
        }
    }
}

fn put_inputs(out: &mut Vec<u8>, inputs: &[u64]) {
    put_u32(out, inputs.len() as u32);
    for &i in inputs {
        put_u64(out, i);
    }
}

/// Serialize the store's nodes (in id order) + its MARKED outputs into the canonical byte form.
pub fn serialize(store: &GraphStore) -> Vec<u8> {
    let outputs = store.outputs();
    serialize_with(store, &outputs)
}

/// `serialize` against an EXPLICIT output set (M22): the bytes flag exactly the requested
/// outputs, ignoring stored marks. The byte FORMAT is identical — `serialize` is this with the
/// marked set.
pub fn serialize_with(store: &GraphStore, outputs: &[NodeId]) -> Vec<u8> {
    let (nodes, _) = store.snapshot();
    let mut out = Vec::new();
    out.extend_from_slice(MAGIC);
    put_u32(&mut out, nodes.len() as u32);
    for node in &nodes {
        match node {
            NodeKey::Source { name, params } => {
                out.push(T_SOURCE);
                put_str(&mut out, name);
                put_params(&mut out, params);
            }
            NodeKey::Op {
                name,
                params,
                inputs,
            } => {
                out.push(T_OP);
                put_str(&mut out, name);
                put_params(&mut out, params);
                put_inputs(&mut out, inputs);
            }
            NodeKey::Reduction {
                name,
                params,
                inputs,
            } => {
                out.push(T_REDUCTION);
                put_str(&mut out, name);
                put_params(&mut out, params);
                put_inputs(&mut out, inputs);
            }
            NodeKey::External {
                descriptor,
                params,
                inputs,
            } => {
                out.push(T_EXTERNAL);
                put_str(&mut out, &descriptor.kind);
                put_str(&mut out, &descriptor.content_hash);
                put_str(&mut out, &descriptor.framework);
                put_str(&mut out, &descriptor.version);
                put_str(&mut out, &descriptor.io_schema);
                match &descriptor.preprocessing_ref {
                    Some(s) => {
                        out.push(1);
                        put_str(&mut out, s);
                    }
                    None => out.push(0),
                }
                put_params(&mut out, params);
                put_inputs(&mut out, inputs);
            }
            NodeKey::Exchange { scheme, inputs } => {
                out.push(T_EXCHANGE);
                put_params(&mut out, scheme);
                put_inputs(&mut out, inputs);
            }
            NodeKey::Stage { inputs, members } => {
                out.push(T_STAGE);
                put_inputs(&mut out, inputs);
                put_u32(&mut out, members.len() as u32);
                for m in members {
                    put_str(&mut out, &m.token);
                    put_u32(&mut out, m.inputs.len() as u32);
                    for r in &m.inputs {
                        match r {
                            StageRef::Input(i) => {
                                out.push(R_INPUT);
                                put_u32(&mut out, *i as u32);
                            }
                            StageRef::Member(i) => {
                                out.push(R_MEMBER);
                                put_u32(&mut out, *i as u32);
                            }
                        }
                    }
                }
            }
        }
    }
    put_u32(&mut out, outputs.len() as u32);
    for &o in outputs {
        put_u64(&mut out, o);
    }
    out
}

// ---- reader -------------------------------------------------------------------------------------

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Reader { buf, pos: 0 }
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8], DecodeError> {
        let end = self.pos.checked_add(n).ok_or(DecodeError::Truncated)?;
        if end > self.buf.len() {
            return Err(DecodeError::Truncated);
        }
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8, DecodeError> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> Result<u32, DecodeError> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn u64(&mut self) -> Result<u64, DecodeError> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }
    fn string(&mut self) -> Result<String, DecodeError> {
        let n = self.u32()? as usize;
        let b = self.take(n)?;
        std::str::from_utf8(b)
            .map(|s| s.to_string())
            .map_err(|_| DecodeError::BadUtf8)
    }
    fn params(&mut self) -> Result<ParamMap, DecodeError> {
        let n = self.u32()? as usize;
        let mut entries = Vec::with_capacity(n);
        for _ in 0..n {
            let k = self.string()?;
            let tag = self.u8()?;
            let v = match tag {
                P_INT => ParamValue::Int(self.u64()? as i64),
                P_FLOAT => ParamValue::Float(f64::from_bits(self.u64()?)),
                P_BOOL => ParamValue::Bool(self.u8()? != 0),
                P_STR => ParamValue::Str(self.string()?),
                other => return Err(DecodeError::BadTag("param", other)),
            };
            entries.push((k, v));
        }
        Ok(ParamMap::new(entries))
    }
    fn inputs(&mut self, max: u64) -> Result<Vec<u64>, DecodeError> {
        let n = self.u32()? as usize;
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            let i = self.u64()?;
            if i >= max {
                return Err(DecodeError::BadNodeRef(i));
            }
            v.push(i);
        }
        Ok(v)
    }
}

/// Decode the canonical byte form back into a freshly interned store. Because nodes are written in
/// dependency order and re-interned in that order, the rebuilt store has identical ids, so a
/// re-serialize is byte-identical to the input (asserted by the round-trip tests).
pub fn deserialize(data: &[u8]) -> Result<GraphStore, DecodeError> {
    let mut r = Reader::new(data);
    if r.take(4)? != MAGIC {
        return Err(DecodeError::BadMagic);
    }
    let n_nodes = r.u32()? as u64;
    let store = GraphStore::new();
    for idx in 0..n_nodes {
        let tag = r.u8()?;
        let key = match tag {
            T_SOURCE => {
                let name = r.string()?;
                let params = r.params()?;
                NodeKey::Source { name, params }
            }
            T_OP => {
                let name = r.string()?;
                let params = r.params()?;
                let inputs = r.inputs(idx)?;
                NodeKey::Op {
                    name,
                    params,
                    inputs,
                }
            }
            T_REDUCTION => {
                let name = r.string()?;
                let params = r.params()?;
                let inputs = r.inputs(idx)?;
                NodeKey::Reduction {
                    name,
                    params,
                    inputs,
                }
            }
            T_EXTERNAL => {
                let kind = r.string()?;
                let content_hash = r.string()?;
                let framework = r.string()?;
                let version = r.string()?;
                let io_schema = r.string()?;
                let preprocessing_ref = if r.u8()? != 0 {
                    Some(r.string()?)
                } else {
                    None
                };
                let params = r.params()?;
                let inputs = r.inputs(idx)?;
                NodeKey::External {
                    descriptor: PayloadDescriptor {
                        kind,
                        content_hash,
                        framework,
                        version,
                        io_schema,
                        preprocessing_ref,
                    },
                    params,
                    inputs,
                }
            }
            T_EXCHANGE => {
                let scheme = r.params()?;
                let inputs = r.inputs(idx)?;
                NodeKey::Exchange { scheme, inputs }
            }
            T_STAGE => {
                let inputs = r.inputs(idx)?;
                let n_members = r.u32()? as usize;
                let mut members = Vec::with_capacity(n_members);
                for m_idx in 0..n_members {
                    let token = r.string()?;
                    let n_refs = r.u32()? as usize;
                    let mut refs = Vec::with_capacity(n_refs);
                    for _ in 0..n_refs {
                        let rtag = r.u8()?;
                        let v = r.u32()? as usize;
                        refs.push(match rtag {
                            R_INPUT => StageRef::Input(v),
                            R_MEMBER => {
                                if v >= m_idx {
                                    return Err(DecodeError::BadNodeRef(v as u64));
                                }
                                StageRef::Member(v)
                            }
                            other => return Err(DecodeError::BadTag("stage-ref", other)),
                        });
                    }
                    members.push(StageOp {
                        token,
                        inputs: refs,
                    });
                }
                NodeKey::Stage { inputs, members }
            }
            other => return Err(DecodeError::BadTag("node", other)),
        };
        store
            .add_key(key)
            .map_err(|e| DecodeError::BadNodeRef(e.0))?;
    }
    let n_out = r.u32()? as usize;
    for _ in 0..n_out {
        let o = r.u64()?;
        store
            .mark_output(o)
            .map_err(|e| DecodeError::BadNodeRef(e.0))?;
    }
    Ok(store)
}

#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;
    use crate::param::ParamValue::{Float, Int, Str};

    fn pm(entries: Vec<(&str, ParamValue)>) -> ParamMap {
        ParamMap::new(
            entries
                .into_iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect(),
        )
    }

    fn sample() -> GraphStore {
        let s = GraphStore::new();
        let src = s.add_source("events".into(), pm(vec![("uri", Str("f.root".into()))]));
        let pt = s.add_op("pt".into(), vec![src], pm(vec![])).unwrap();
        let cut = s
            .add_op("gt".into(), vec![pt], pm(vec![("thr", Float(30.0))]))
            .unwrap();
        let d = PayloadDescriptor {
            kind: "onnx".into(),
            content_hash: "abc".into(),
            framework: "ort".into(),
            version: "1.17".into(),
            io_schema: "f32->f32".into(),
            preprocessing_ref: Some("pre".into()),
        };
        let nn = s
            .add_external(d, vec![cut], pm(vec![("k", Int(2))]))
            .unwrap();
        let red = s.add_reduction("sum".into(), vec![nn], pm(vec![])).unwrap();
        s.mark_output(red).unwrap();
        s
    }

    #[test]
    fn roundtrip_is_byte_identical() {
        let g = sample();
        let bytes = serialize(&g);
        let g2 = deserialize(&bytes).unwrap();
        assert_eq!(g.node_count(), g2.node_count());
        assert_eq!(g.to_dot(), g2.to_dot());
        assert_eq!(bytes, serialize(&g2), "re-serialize must be byte-identical");
    }

    #[test]
    fn identical_graphs_serialize_identically() {
        assert_eq!(serialize(&sample()), serialize(&sample()));
    }

    #[test]
    fn stage_nodes_roundtrip() {
        // a reduced graph carries Stage nodes; they must survive the codec too.
        let g = sample();
        let (reduced, _) = g.reduce(&crate::optimizer::EggEngine::default());
        let bytes = serialize(&reduced);
        let back = deserialize(&bytes).unwrap();
        assert_eq!(reduced.to_dot(), back.to_dot());
        assert_eq!(bytes, serialize(&back));
    }

    #[test]
    fn rejects_bad_magic_and_truncation() {
        assert_eq!(deserialize(b"XXXX").err(), Some(DecodeError::BadMagic));
        let mut bytes = serialize(&sample());
        bytes.truncate(bytes.len() - 1);
        assert_eq!(deserialize(&bytes).err(), Some(DecodeError::Truncated));
    }
}
