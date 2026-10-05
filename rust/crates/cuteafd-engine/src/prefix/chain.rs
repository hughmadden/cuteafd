//! Host page identity by content: a page's rows depend only on the tokens up to its end, so a
//! hash chain over `page_rows`-token blocks names it (hughmadden/glm53f-afd `page_chain`).
//! Identical prefixes from different requests then share host pages even after their device
//! pages were freed. 64 bits: the host tier holds tens of thousands of pages, so a collision is
//! ~1e-10 likely over a whole cache.
use cuteafd_hostcache::snapshot::DevicePageId;
use cuteafd_core::MediaSpan;
use std::hash::{Hash, Hasher};

/// Marks a content identity in [`DevicePageId::compressor`], so it never equals a device
/// identity (whose class byte is below 0x80).
pub const CONTENT_CLASS: u8 = 0x80;

/// `ids[i]` identifies `tokens[..page_rows * (i + 1)]`; a trailing partial block has no id.
pub fn page_chain(tokens: &[u32], page_rows: usize) -> Vec<u64> {
    let mut prev = 0u64;
    tokens
        .chunks_exact(page_rows)
        .map(|block| {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            0x6375_7465_6166_6470u64.hash(&mut h);
            prev.hash(&mut h);
            block.hash(&mut h);
            prev = h.finish();
            prev
        })
        .collect()
}

/// Salt image-containing and subsequent page identities with the full SHA-256 keys.
/// Full-key restore verification alone is insufficient: the host page deduper must not share
/// bytes from two colliding 31-bit radix sequences. Text-only page chains stay unchanged.
pub fn page_chain_media(tokens: &[u32], media: &[MediaSpan], page_rows: usize) -> Vec<u64> {
    if media.is_empty() { return page_chain(tokens, page_rows); }
    let mut prev = 0u64;
    tokens.chunks_exact(page_rows).enumerate().map(|(i, block)| {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        0x6375_7465_6166_6470u64.hash(&mut h);
        prev.hash(&mut h);
        block.hash(&mut h);
        let start = i * page_rows;
        let end = start + page_rows;
        for span in media.iter().filter(|s| s.start < end && s.checked_end().is_some_and(|e| e > start)) {
            span.start.hash(&mut h);
            span.len.hash(&mut h);
            span.key.hash(&mut h);
        }
        prev = h.finish();
        prev
    }).collect()
}

/// The host tier's identity of a full page with content id `id` in page class `class`.
pub fn content_id(class: u8, id: u64) -> DevicePageId {
    DevicePageId { compressor: CONTENT_CLASS | class, page: (id >> 32) as u32, generation: id as u32 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chain_names_prefixes_not_blocks() {
        for rows in [64usize, 256] {
            let a: Vec<u32> = (0..(4 * rows - 24) as u32).collect();
            let mut b = a.clone();
            b[2 * rows + 10] = 5;
            let (ca, cb) = (page_chain(&a, rows), page_chain(&b, rows));
            assert_eq!(ca.len(), 3);
            assert_eq!(ca[..2], cb[..2]);
            assert_ne!(ca[2], cb[2]);
            let mut c = a.clone();
            c[3] = 9;
            assert!(page_chain(&c, rows).iter().zip(&ca).all(|(x, y)| x != y));
            assert!(page_chain(&a[..rows - 1], rows).is_empty());
        }
        let id = content_id(0, 0x1234_5678_9abc_def0);
        assert_eq!((id.compressor, id.page, id.generation), (0x80, 0x1234_5678, 0x9abc_def0));
    }
}
