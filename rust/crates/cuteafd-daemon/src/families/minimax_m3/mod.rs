//! Native MiniMax M3 target on the shared continuous scheduler.
mod admission;
mod backend;
mod cache;
mod execution;
mod ffn;
mod head;
mod media;
mod prefix;
mod routed;
mod scratch;
pub(crate) mod serve;
mod transaction;
mod weights;

#[derive(Debug, thiserror::Error)]
pub(super) enum MiniMaxError {
    #[error("MiniMax unsupported: {0}")]
    Unsupported(&'static str),
    #[error("MiniMax invalid {0}")]
    Invalid(&'static str),
    #[error("MiniMax allocation overflow: {0}")]
    Overflow(&'static str),
}
