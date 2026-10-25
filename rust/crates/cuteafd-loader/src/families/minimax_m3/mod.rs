//! MiniMax M3 (`MiniMaxM3SparseForConditionalGeneration`, `model_type=minimax_m3_vl`)
//! checkpoint readers: MSA block-sparse GQA, sigmoid-routed SwiGLU-OAI MoE on
//! compressed-tensors NVFP4 experts, MXFP8 (1x32 UE8M0) dense projections, a
//! bundled CLIP-style vision tower and a bundled DSpark drafter; a DFlash2
//! drafter checkpoint runs on the shared drafter (`dflash`).
pub mod cache;
pub mod config;
pub mod dflash;
pub mod dspark;

/// The release's compiled context (`max_position_embeddings`).
pub const COMPILED_CONTEXT: usize = 1 << 20;
