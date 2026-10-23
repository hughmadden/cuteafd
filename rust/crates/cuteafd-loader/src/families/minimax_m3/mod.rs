//! MiniMax M3 (`MiniMaxM3SparseForConditionalGeneration`, `model_type=minimax_m3_vl`)
//! checkpoint readers: MSA block-sparse GQA, sigmoid-routed SwiGLU-OAI MoE on
//! compressed-tensors NVFP4 experts, MXFP8 (1x32 UE8M0) dense projections, a
//! bundled CLIP-style vision tower and a bundled DSpark drafter.
pub mod config;
pub mod reference;

/// The release's compiled context (`max_position_embeddings`).
pub const COMPILED_CONTEXT: usize = 1 << 20;
