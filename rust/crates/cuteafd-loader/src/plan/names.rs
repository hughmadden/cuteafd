//! Small helpers for walking dotted tensor names without regular expressions.

/// If `name` starts with `prefix` followed by a decimal index and a dot,
/// returns the index and the remainder: `("layers.", "layers.3.attn.x")`
/// yields `(3, "attn.x")`.
pub fn indexed<'a>(name: &'a str, prefix: &str) -> Option<(usize, &'a str)> {
    let rest = name.strip_prefix(prefix)?;
    let (index, rest) = rest.split_once('.')?;
    Some((index.parse().ok()?, rest))
}

/// Like [`indexed`] but tolerates a trailing index with no remainder.
pub fn indexed_tail<'a>(name: &'a str, prefix: &str) -> Option<(usize, &'a str)> {
    let rest = name.strip_prefix(prefix)?;
    match rest.split_once('.') {
        Some((index, rest)) => Some((index.parse().ok()?, rest)),
        None => Some((rest.parse().ok()?, "")),
    }
}

pub fn starts_with_any(name: &str, prefixes: &[&str]) -> bool {
    prefixes.iter().any(|prefix| name.starts_with(prefix))
}
