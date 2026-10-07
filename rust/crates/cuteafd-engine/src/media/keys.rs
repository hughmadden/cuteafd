use super::{ImageKey, MediaError, MediaKey, MediaSpan};

/// Native tokens remain owned by the caller; only this copy goes to prefix tiers.
#[derive(Clone, Debug)]
pub struct MediaKeys {
    tokens: Vec<u32>,
    spans: Vec<MediaSpan>,
}

impl MediaKeys {
    pub fn new(native: &[u32], vocabulary: u32, spans: &[MediaSpan]) -> Result<Self, MediaError> {
        if vocabulary == 0 || vocabulary >= 0x8000_0000 || native.iter().any(|&t| t >= vocabulary) {
            return Err(MediaError::Vocabulary);
        }
        validate_spans(spans)?;
        if spans
            .last()
            .is_some_and(|s| s.checked_end().unwrap() > native.len())
        {
            return Err(MediaError::Spans);
        }
        let mut tokens = native.to_vec();
        for span in spans {
            for row in 0..span.len {
                tokens[span.start + row] = media_token_id(span.key, row);
            }
        }
        Ok(Self {
            tokens,
            spans: spans.to_vec(),
        })
    }
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }
    pub fn spans(&self) -> &[MediaSpan] {
        &self.spans
    }
}

/// Hugh Madden's mimo26f-afd v1.3.0, crates/mimo26-coordinator/src/api.rs
/// `image_token_id`, with a fold of the full SHA-256 key instead of a source hash.
/// This is a radix hint, not an identity: restores must verify all 256 key bits.
pub fn image_token_id(key: ImageKey, row: usize) -> u32 {
    media_token_id(key.into(), row)
}
pub fn media_token_id(key: MediaKey, row: usize) -> u32 {
    let hash = key
        .bytes()
        .chunks_exact(8)
        .enumerate()
        .fold(0u64, |h, (i, word)| {
            h ^ u64::from_le_bytes(word.try_into().unwrap()).rotate_left((i * 13) as u32)
        });
    let domain = if key.is_audio() { 0x6175_6469_6f76_3031 } else { 0 };
    let mut x = hash ^ domain ^ (row as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    x ^= x >> 33;
    0x8000_0000 | (x as u32 & 0x7fff_ffff)
}

pub(crate) fn validate_spans(spans: &[MediaSpan]) -> Result<(), MediaError> {
    let mut previous_end = 0;
    for span in spans {
        let end = span.checked_end().ok_or(MediaError::Spans)?;
        if span.len == 0 || span.start < previous_end {
            return Err(MediaError::Spans);
        }
        previous_end = end;
    }
    Ok(())
}

/// Safe points include either endpoint but never a row strictly inside an image.
pub fn round_frontier(frontier: usize, spans: &[MediaSpan]) -> usize {
    spans
        .iter()
        .find(|s| s.start < frontier && s.checked_end().is_some_and(|e| frontier < e))
        .map_or(frontier, |s| s.start)
}

pub fn snapshot_media(frontier: usize, spans: &[MediaSpan]) -> Vec<MediaSpan> {
    spans
        .iter()
        .filter(|s| s.checked_end().is_some_and(|e| e <= frontier))
        .copied()
        .collect()
}

/// Compare span geometry as well as identity, in both directions. Reject a partial image.
pub fn verify_media(frontier: usize, saved: &[MediaSpan], request: &[MediaSpan]) -> bool {
    if round_frontier(frontier, saved) != frontier || round_frontier(frontier, request) != frontier
    {
        return false;
    }
    saved
        .iter()
        .filter(|s| s.start < frontier)
        .eq(request.iter().filter(|s| s.start < frontier))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn keys_do_not_mutate_native_ids_and_are_stable() {
        let native = vec![1, 2, 3, 3, 4];
        let span = MediaSpan {
            start: 2,
            len: 2,
            key: ImageKey([7; 32]).into(),
        };
        let keys = MediaKeys::new(&native, 10, &[span]).unwrap();
        assert_eq!(
            keys.tokens(),
            MediaKeys::new(&native, 10, &[span]).unwrap().tokens()
        );
        assert_eq!(native, [1, 2, 3, 3, 4]);
        assert!(keys.tokens()[2..4].iter().all(|&t| t >= 0x8000_0000));
        assert_eq!(keys.tokens()[..2], native[..2]);
        assert_eq!(keys.tokens()[4], native[4]);
    }
    #[test]
    fn validation_and_frontiers() {
        let span = MediaSpan {
            start: 2,
            len: 3,
            key: ImageKey([7; 32]).into(),
        };
        assert!(MediaKeys::new(&[1; 4], 10, &[span]).is_err());
        assert!(MediaKeys::new(&[0x8000_0000], u32::MAX, &[]).is_err());
        assert!(validate_spans(&[span, span]).is_err());
        for p in 0..8 {
            assert_eq!(
                round_frontier(p, &[span]),
                if (3..5).contains(&p) { 2 } else { p }
            );
        }
        assert!(verify_media(2, &[span], &[]));
        assert!(!verify_media(3, &[span], &[span]));
        assert!(!verify_media(5, &[span], &[]));
        assert!(verify_media(5, &[span], &[span]));
    }
}
