//! Shared media identity types. Native model tokens never contain these keys.

use serde::{Deserialize, Serialize};

/// The full SHA-256 identity of an image, including encoder and preprocessing.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct ImageKey(pub [u8; 32]);

/// The full SHA-256 identity of canonical PCM, preprocessing and its audio tower.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct AudioKey(pub [u8; 32]);

/// Modality is part of cache and prefix identity, even when digest bytes coincide.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum MediaKey {
    Image(ImageKey),
    Audio(AudioKey),
}
impl MediaKey {
    pub fn bytes(&self) -> &[u8; 32] {
        match self {
            Self::Image(key) => &key.0,
            Self::Audio(key) => &key.0,
        }
    }
    pub fn is_audio(self) -> bool { matches!(self, Self::Audio(_)) }
}
impl From<ImageKey> for MediaKey {
    fn from(key: ImageKey) -> Self { Self::Image(key) }
}
impl From<AudioKey> for MediaKey {
    fn from(key: AudioKey) -> Self { Self::Audio(key) }
}
impl std::hash::Hash for MediaKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Preserve existing image page identities; audio alone adds a domain tag.
        match self {
            Self::Image(key) => key.hash(state),
            Self::Audio(key) => {
                b"cuteafd.audio.v1".hash(state);
                key.hash(state);
            }
        }
    }
}
#[derive(Serialize, Deserialize)]
#[serde(untagged)]
enum SerializedMediaKey {
    Image(ImageKey),
    Audio { audio: AudioKey },
}
impl Serialize for MediaKey {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Image(key) => SerializedMediaKey::Image(*key),
            Self::Audio(key) => SerializedMediaKey::Audio { audio: *key },
        }.serialize(serializer)
    }
}
impl<'de> Deserialize<'de> for MediaKey {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(match SerializedMediaKey::deserialize(deserializer)? {
            SerializedMediaKey::Image(key) => Self::Image(key),
            SerializedMediaKey::Audio { audio } => Self::Audio(audio),
        })
    }
}

/// A contiguous span of media feature rows in the expanded native prompt.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct MediaSpan {
    pub start: usize,
    pub len: usize,
    pub key: MediaKey,
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
            key: ImageKey([0xab; 32]).into(),
        };
        let json = serde_json::to_vec(&span).unwrap();
        assert_eq!(serde_json::from_slice::<MediaSpan>(&json).unwrap(), span);
        assert_eq!(span.checked_end(), Some(11));
    }

    #[test]
    fn image_wire_identity_stays_legacy_and_audio_is_domain_separated() {
        use std::hash::{Hash, Hasher};
        let image = ImageKey([0xab; 32]);
        let audio = MediaKey::Audio(AudioKey(image.0));
        assert_eq!(serde_json::to_vec(&MediaKey::Image(image)).unwrap(), serde_json::to_vec(&image).unwrap());
        assert_eq!(serde_json::from_slice::<MediaKey>(&serde_json::to_vec(&audio).unwrap()).unwrap(), audio);
        let hash = |value: &dyn Fn(&mut std::collections::hash_map::DefaultHasher)| {
            let mut h = std::collections::hash_map::DefaultHasher::new(); value(&mut h); h.finish()
        };
        assert_eq!(hash(&|h| image.hash(h)), hash(&|h| MediaKey::Image(image).hash(h)));
        assert_ne!(hash(&|h| audio.hash(h)), hash(&|h| image.hash(h)));
    }

    #[test]
    fn overflowing_span_is_not_a_valid_frontier() {
        let span = MediaSpan {
            start: usize::MAX,
            len: 1,
            key: ImageKey([0; 32]).into(),
        };
        assert_eq!(span.checked_end(), None);
    }
}
