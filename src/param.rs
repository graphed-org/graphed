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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn token_roundtrips_every_variant() {
        for v in [
            ParamValue::Int(-7),
            ParamValue::Float(30.5),
            ParamValue::Bool(true),
            ParamValue::Bool(false),
            ParamValue::Str("plain".into()),
        ] {
            assert_eq!(ParamValue::from_token(&v.token()).unwrap(), v);
        }
    }

    #[test]
    fn nan_collapses_but_signed_zero_stays_distinct() {
        // any NaN payload interns to the same key, matching the doc comment's contract.
        let a = ParamValue::Float(f64::NAN);
        let b = ParamValue::Float(-f64::NAN);
        assert_eq!(a.token(), b.token());
        assert_eq!(a, b);
        // 0.0 and -0.0 are numerically equal but must stay distinct keys (M4's job, not M1's).
        let pz = ParamValue::Float(0.0);
        let nz = ParamValue::Float(-0.0);
        assert_ne!(pz.token(), nz.token());
        assert_ne!(pz, nz);
    }

    #[test]
    fn hash_agrees_with_eq_for_nan() {
        let mut set: HashSet<ParamValue> = HashSet::new();
        set.insert(ParamValue::Float(f64::NAN));
        set.insert(ParamValue::Float(-f64::NAN)); // same canonical key -> no growth
        assert_eq!(set.len(), 1);
        set.insert(ParamValue::Float(1.0));
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn hash_covers_every_tag_arm() {
        // one member of a HashSet per variant witnesses tag()/hash()'s Int/Float/Bool/Str arms.
        let set: HashSet<ParamValue> = HashSet::from([
            ParamValue::Int(1),
            ParamValue::Float(1.0),
            ParamValue::Bool(true),
            ParamValue::Str("s".into()),
        ]);
        assert_eq!(set.len(), 4);
    }

    #[test]
    fn cross_variant_equality_is_false() {
        assert_ne!(ParamValue::Int(1), ParamValue::Bool(true));
        assert_ne!(ParamValue::Str("1".into()), ParamValue::Int(1));
    }

    #[test]
    fn from_token_rejects_malformed_input() {
        assert!(ParamValue::from_token("").is_none()); // no recognized tag at all
        assert!(ParamValue::from_token("ixyz").is_none()); // bad int digits
        assert!(ParamValue::from_token("fzz").is_none()); // bad hex payload
        assert!(ParamValue::from_token("btrue2").is_none()); // unrecognized bool spelling
        assert!(ParamValue::from_token("q1").is_none()); // unknown tag
    }

    #[test]
    fn escape_unescape_roundtrips_and_is_order_correct() {
        let raw = "a%b;c=d|e";
        let escaped = escape_token(raw);
        assert_eq!(escaped, "a%25b%3Bc%3Dd%7Ce");
        assert_eq!(unescape_token(&escaped), raw);
        // `%` escapes first, so a literal "%3B" in the source must survive as literal text, not
        // be mistaken for an escaped `;` — this is the order the doc comment promises.
        let tricky = "%3B";
        assert_eq!(unescape_token(&escape_token(tricky)), tricky);
    }

    #[test]
    fn display_formats_every_variant() {
        assert_eq!(ParamValue::Int(5).to_string(), "5");
        assert_eq!(ParamValue::Float(2.5).to_string(), "2.5");
        assert_eq!(ParamValue::Bool(true).to_string(), "true");
        assert_eq!(ParamValue::Str("hi".into()).to_string(), "\"hi\"");
    }

    #[test]
    fn parammap_sorts_and_roundtrips() {
        let m = ParamMap::new(vec![
            ("b".into(), ParamValue::Int(2)),
            ("a".into(), ParamValue::Str("x;y".into())),
        ]);
        assert_eq!(m.entries()[0].0, "a");
        assert_eq!(m.entries()[1].0, "b");
        assert!(!m.is_empty());
        assert_eq!(ParamMap::from_token(&m.token()).unwrap(), m);
        assert_eq!(m.to_string(), "{a=\"x;y\", b=2}");
    }

    #[test]
    fn parammap_empty_token_roundtrips_to_empty_map() {
        let m = ParamMap::new(vec![]);
        assert!(m.is_empty());
        assert_eq!(m.token(), "");
        assert_eq!(ParamMap::from_token(""), Some(m));
    }

    #[test]
    fn parammap_from_token_rejects_malformed_entries() {
        assert!(ParamMap::from_token("nokeyvalue").is_none()); // missing '='
        assert!(ParamMap::from_token("k=zbogus").is_none()); // unparseable value token
    }
}
