//! Qwen 3.8 Flash Next (qwen4_exp) family: configuration and the PLE n-gram
//! hashing for the generic engine.
pub mod config;
pub mod ngram;
pub mod resident;
pub mod rope;
pub use rope::{ImageSpan, RopeError, RopePositions};
pub use config::{Qwen4Attention, Qwen4Config};
pub use ngram::{NgramHasher, NgramHistory};
