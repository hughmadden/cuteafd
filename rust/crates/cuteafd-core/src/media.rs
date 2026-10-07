//! Shared media identity types. Native model tokens never contain these keys.

use serde::{Deserialize, Serialize};

/// The full SHA-256 identity of an image, including encoder and preprocessing.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct ImageKey(pub [u8; 32]);

/// The full SHA-256 identity of canonical PCM, preprocessing and its audio tower.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct AudioKey(pub [u8; 32]);

/// A contiguous span of image feature rows in the expanded native prompt.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct MediaSpan {
    pub start: usize,
    pub len: usize,
    pub key: ImageKey,
}

impl MediaSpan {
    pub fn checked_end(&self) -> Option<usize> {
        self.start.checked_add(self.len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_key_and_span_round_trip() {
        let span = MediaSpan {
            start: 7,
            len: 4,
            key: ImageKey([0xab; 32]),
        };
        let json = serde_json::to_vec(&span).unwrap();
        assert_eq!(serde_json::from_slice::<MediaSpan>(&json).unwrap(), span);
        assert_eq!(span.checked_end(), Some(11));
    }

    #[test]
    fn overflowing_span_is_not_a_valid_frontier() {
        let span = MediaSpan {
            start: usize::MAX,
            len: 1,
            key: ImageKey([0; 32]),
        };
        assert_eq!(span.checked_end(), None);
    }
}
