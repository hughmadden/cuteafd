//! Prefix-cache namespaces for weight variants (contract §5): a variant's cached state differs
//! from base from its first edited layer on, so the same tokens under two variants must never
//! share a snapshot, a host page or a content id.
//!
//! The namespace is part of the key. Base (`tag` 0) keys are the tokens themselves, so base
//! entries, host pages and chain ids are byte-identical to a build without variants. A variant
//! (`tag` 1..=7, its `--variant` index) keys its first token as
//! `0x8000_0000 | tag << 24 | token`: every tier matches keys from position 0, so a variant key
//! never shares even one row with a base key or another variant's key, and the device radix,
//! the host index and the host page chain all separate with no other change. Base first tokens
//! are vocabulary ids (< 2^31); media keys are only ever later positions in practice and are
//! verified by their full SHA-256 anyway. The tag is the variant's index inside this process;
//! the host tier lives in process memory, and a process's variant tables never change.
//!
//! **Process-local only.** An index means nothing to another process, or after a restart with a
//! different `--variant` order or changed tables. If a host tier is ever shared across processes
//! or persisted (e.g. a 512 GB L2 tier on another host), key variants by
//! `VariantPlan.namespace` (the SHA-256 of their tables) instead, for example as a second
//! namespaced token, and keep base keys unchanged.
use std::borrow::Cow;

/// Variant tags fit in seven bits; first tokens must fit below the tag field.
pub const MAX_NAMESPACE_TAG: u8 = 0x7f;
const TOKEN_BITS: u32 = 24;

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum NamespaceError {
    #[error("prefix namespace tag {0} is outside 0..=127")]
    Tag(u8),
    #[error("first token {0} does not fit a namespaced prefix key (24 bits)")]
    Token(u32),
}

/// The prefix-cache key of `tokens` in namespace `tag` (0 = base: borrowed, unchanged).
pub fn namespaced(tokens: &[u32], tag: u8) -> Result<Cow<'_, [u32]>, NamespaceError> {
    if tag == 0 || tokens.is_empty() { return Ok(Cow::Borrowed(tokens)); }
    if tag > MAX_NAMESPACE_TAG { return Err(NamespaceError::Tag(tag)); }
    let first = tokens[0];
    if first >> TOKEN_BITS != 0 { return Err(NamespaceError::Token(first)); }
    let mut key = tokens.to_vec();
    key[0] = 0x8000_0000 | (u32::from(tag) << TOKEN_BITS) | first;
    Ok(Cow::Owned(key))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prefix::page_chain;

    #[test]
    fn the_tag_is_the_process_local_variant_index() {
        // The key carries exactly the index and the first token: nothing derived from the
        // tables, so it is valid only inside the process that assigned the indices.
        for tag in 1..=7u8 {
            let key = namespaced(&[0x00ab_cdef, 9], tag).unwrap();
            assert_eq!(key[0], 0x8000_0000 | (u32::from(tag) << 24) | 0x00ab_cdef);
            assert_eq!((key[0] >> 24) & 0x7f, u32::from(tag));
        }
    }

    #[test]
    fn base_keys_and_chains_are_unchanged_and_variants_separate() {
        let tokens: Vec<u32> = (0..256).map(|i| 1000 + i * 7).collect();
        assert!(matches!(namespaced(&tokens, 0).unwrap(), Cow::Borrowed(t) if t == tokens.as_slice()));
        let base = page_chain(&tokens, 64);
        // The pinned base chain value (unchanged from before namespaces existed).
        assert_eq!(base, page_chain(namespaced(&tokens, 0).unwrap().as_ref(), 64));
        let a = namespaced(&tokens, 1).unwrap();
        let b = namespaced(&tokens, 2).unwrap();
        assert_eq!(&a[1..], &tokens[1..]);
        assert_ne!(a[0], tokens[0]);
        assert_ne!(a[0], b[0]);
        let (ca, cb) = (page_chain(&a, 64), page_chain(&b, 64));
        for page in 0..base.len() {
            assert!(base[page] != ca[page] && base[page] != cb[page] && ca[page] != cb[page], "page {page}");
        }
        assert_eq!(namespaced(&[0x0100_0000], 1).unwrap_err(), NamespaceError::Token(0x0100_0000));
        assert_eq!(namespaced(&[1], 200).unwrap_err(), NamespaceError::Tag(200));
        assert!(namespaced(&[], 3).unwrap().is_empty());
    }
}
