//! CPU-side multimodal admission. Encoding waiters own no KV pages or state slots.
mod admission;
mod cache;
mod encoder;
pub(crate) mod keys;
mod request;
mod stats;

pub use admission::{MediaAdmission, MediaPoll, MediaReady, MediaWaiter};
pub use cache::{EmbeddingCache, EmbeddingLease};
pub use cuteafd_core::{ImageKey, MediaSpan};
pub use encoder::{EncodeJob, EncodeOutput, EncoderClient, EncoderTicket, FakeEncoder};
pub use keys::{image_token_id, round_frontier, snapshot_media, verify_media, MediaKeys};
pub use request::{MediaChunk, RequestMedia};
pub use stats::MediaStats;
use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum MediaError {
    #[error("vocabulary and native tokens must be below 2^31")]
    Vocabulary,
    #[error("media spans must be nonempty, ordered, disjoint and within the prompt")]
    Spans,
    #[error("media feature geometry or byte length is invalid")]
    Features,
    #[error("media features are not ready for {0:?}")]
    NotReady(ImageKey),
    #[error("embedding cache budget exhausted: need {needed} bytes, {free} free of {capacity}")]
    CacheFull {
        needed: usize,
        free: usize,
        capacity: usize,
    },
    #[error("media pending queue is full")]
    QueueFull,
    #[error("media request needs {images} new images / {tokens} new image tokens (limit {max_images} / {max_tokens})")]
    EncodeLimit {
        images: usize,
        tokens: usize,
        max_images: usize,
        max_tokens: usize,
    },
    #[error("vision encoder unavailable: {0}")]
    Encoder(String),
}
