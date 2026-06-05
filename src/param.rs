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
    /// stay distinct and all NaNs collapse, exactly as interning does).
    pub fn token(&self) -> String {
        match self {
            ParamValue::Int(a) => format!("i{a}"),
            ParamValue::Float(a) => format!("f{:016x}", Self::float_key(*a)),
            ParamValue::Bool(a) => format!("b{a}"),
            ParamValue::Str(a) => format!("s{a}"),
        }
    }
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

    /// A compact, injective, whitespace-free encoding for optimizer tokens (M4). Type-tagged so
    /// int/float/bool/str with the same printed form stay distinct, matching interning identity.
    pub fn token(&self) -> String {
        self.0
            .iter()
            .map(|(k, v)| format!("{k}={}", v.token()))
            .collect::<Vec<_>>()
            .join(";")
    }
}

impl fmt::Display for ParamMap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let parts: Vec<String> = self.0.iter().map(|(k, v)| format!("{k}={v}")).collect();
        write!(f, "{{{}}}", parts.join(", "))
    }
}
