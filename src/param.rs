//! Deterministic parameter values with total-order float hashing (plan M1).
//!
//! Floats use a canonical bit key: every NaN collapses to one key (so NaN interns to itself),
//! while `0.0` and `-0.0` stay distinct (canonicalizing them numerically is M4's job, not M1's).

use std::fmt;
use std::hash::{Hash, Hasher};

/// Canonical quiet-NaN bit pattern; all NaNs hash/compare equal to this.
const CANON_NAN: u64 = 0x7ff8_0000_0000_0000;

#[derive(Clone, Debug)]
pub enum ParamValue {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
}

impl ParamValue {
    fn tag(&self) -> u8 {
        match self {
            ParamValue::Int(_) => 0,
            ParamValue::Float(_) => 1,
            ParamValue::Bool(_) => 2,
            ParamValue::Str(_) => 3,
        }
    }

    /// IEEE-aware canonical bits used for both equality and hashing.
    fn float_key(x: f64) -> u64 {
        if x.is_nan() {
            CANON_NAN
        } else {
            x.to_bits()
        }
    }

    /// Type-tagged token mirroring equality/hash identity (floats by canonical bits, so 0.0/-0.0
    /// stay distinct and all NaNs collapse, exactly as interning does). String payloads are
    /// escaped so the encoding is injective even when a value contains a separator character.
    pub fn token(&self) -> String {
        match self {
            ParamValue::Int(a) => format!("i{a}"),
            ParamValue::Float(a) => format!("f{:016x}", Self::float_key(*a)),
            ParamValue::Bool(a) => format!("b{a}"),
            ParamValue::Str(a) => format!("s{}", escape_token(a)),
        }
    }

    /// Decode one tagged value token (the inverse of `token`).
    pub fn from_token(tok: &str) -> Option<ParamValue> {
        let (tag, rest) = tok.split_at(tok.char_indices().nth(1).map_or(tok.len(), |(i, _)| i));
        match tag {
            "i" => rest.parse::<i64>().ok().map(ParamValue::Int),
            "f" => u64::from_str_radix(rest, 16)
                .ok()
                .map(|bits| ParamValue::Float(f64::from_bits(bits))),
            "b" => match rest {
                "true" => Some(ParamValue::Bool(true)),
                "false" => Some(ParamValue::Bool(false)),
                _ => None,
            },
            "s" => Some(ParamValue::Str(unescape_token(rest))),
            _ => None,
        }
    }
}

/// Escape the separator characters (`%`, `;`, `=`, `|`) so param tokens are injective and
/// losslessly parseable. `%` first so unescaping is its exact inverse.
pub fn escape_token(s: &str) -> String {
    s.replace('%', "%25")
        .replace(';', "%3B")
        .replace('=', "%3D")
        .replace('|', "%7C")
}

/// Inverse of [`escape_token`].
pub fn unescape_token(s: &str) -> String {
    s.replace("%7C", "|")
        .replace("%3D", "=")
        .replace("%3B", ";")
        .replace("%25", "%")
}

impl PartialEq for ParamValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (ParamValue::Int(a), ParamValue::Int(b)) => a == b,
            (ParamValue::Float(a), ParamValue::Float(b)) => {
                Self::float_key(*a) == Self::float_key(*b)
            }
            (ParamValue::Bool(a), ParamValue::Bool(b)) => a == b,
            (ParamValue::Str(a), ParamValue::Str(b)) => a == b,
            _ => false,
        }
    }
}

impl Eq for ParamValue {}

impl Hash for ParamValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.tag().hash(state);
        match self {
            ParamValue::Int(a) => a.hash(state),
            ParamValue::Float(a) => Self::float_key(*a).hash(state),
            ParamValue::Bool(a) => a.hash(state),
            ParamValue::Str(a) => a.hash(state),
        }
    }
}

impl fmt::Display for ParamValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParamValue::Int(a) => write!(f, "{a}"),
            ParamValue::Float(a) => write!(f, "{a}"),
            ParamValue::Bool(a) => write!(f, "{a}"),
            ParamValue::Str(a) => write!(f, "{a:?}"),
        }
    }
}

/// A parameter map with keys kept in a total order so its hash is deterministic (plan M1).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ParamMap(Vec<(String, ParamValue)>);

impl ParamMap {
    pub fn new(mut entries: Vec<(String, ParamValue)>) -> Self {
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        ParamMap(entries)
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// The sorted (key, value) entries, for canonical serialization (plan M8).
    pub fn entries(&self) -> &[(String, ParamValue)] {
        &self.0
    }

    /// A compact, injective, whitespace-free encoding for optimizer tokens (M4). Type-tagged so
    /// int/float/bool/str with the same printed form stay distinct, matching interning identity.
    /// Keys and string payloads are separator-escaped, so the encoding is genuinely injective and
    /// invertible via [`ParamMap::from_token`].
    pub fn token(&self) -> String {
        self.0
            .iter()
            .map(|(k, v)| format!("{}={}", escape_token(k), v.token()))
            .collect::<Vec<_>>()
            .join(";")
    }

    /// Decode a `token()` string back into a `ParamMap` (used by `GraphStore.nodes()` to expose
    /// fused stage members for IR-driven execution). Returns `None` on a malformed token.
    pub fn from_token(s: &str) -> Option<ParamMap> {
        if s.is_empty() {
            return Some(ParamMap::new(vec![]));
        }
        let mut entries = Vec::new();
        for part in s.split(';') {
            let (k, v) = part.split_once('=')?;
            entries.push((unescape_token(k), ParamValue::from_token(v)?));
        }
        Some(ParamMap::new(entries))
    }
}

impl fmt::Display for ParamMap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let parts: Vec<String> = self.0.iter().map(|(k, v)| format!("{k}={v}")).collect();
        write!(f, "{{{}}}", parts.join(", "))
    }
}
