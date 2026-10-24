//! CPU reference for the MiniMax M3 sigmoid router, and the per-token cache
//! record sizes the planner charges. The arithmetic oracle (SwiGLU-OAI, Gemma
//! norms, partial RoPE, MSA selection) is the Python reference the goldens use,
//! `python/reference/families/minimax_m3/model.py`.
use super::config::MiniMaxM3Config;

/// The router: sigmoid scores, top-k by `score + bias` (selection only), the
/// unbiased scores of the selection renormalized and times `scale`. Returns
/// `(expert, weight)` in descending selection score; ties keep the lower id.
pub fn route(logits: &[f32], bias: &[f32], topk: usize, scale: f32) -> Vec<(usize, f32)> {
    assert_eq!(logits.len(), bias.len());
    let scores: Vec<f32> = logits.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect();
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&a, &b| (scores[b] + bias[b]).total_cmp(&(scores[a] + bias[a])).then(a.cmp(&b)));
    order.truncate(topk);
    let sum: f32 = order.iter().map(|&e| scores[e]).sum();
    order.into_iter().map(|e| (e, scores[e] / sum * scale)).collect()
}

/// Bytes of one token's main KV across every layer: K and V for every KV head.
pub fn kv_bytes_per_token(cfg: &MiniMaxM3Config, element_bytes: usize) -> usize {
    cfg.layers * 2 * cfg.kv_heads * cfg.head_dim * element_bytes
}

/// Bytes of one token's MSA index key across the sparse layers, in the paged
/// indexer layout (FP8 E4M3 row plus one FP32 scale).
pub fn index_bytes_per_token(cfg: &MiniMaxM3Config) -> usize {
    cfg.sparse.iter().filter(|s| **s).count() * (cfg.msa.index_dim + 4)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::families::minimax_m3::config::tests::release_config;

    #[test]
    fn router_selects_with_bias_and_weights_without() {
        let logits = [0.0, 2.0, 1.0, -1.0];
        let bias = [5.0, 0.0, 0.0, 0.0];
        let routed = route(&logits, &bias, 2, 2.0);
        assert_eq!(routed.iter().map(|r| r.0).collect::<Vec<_>>(), [0, 1]);
        let s = |x: f32| 1.0 / (1.0 + (-x).exp());
        assert!((routed[0].1 - s(0.0) / (s(0.0) + s(2.0)) * 2.0).abs() < 1e-6);
        assert!((routed.iter().map(|r| r.1).sum::<f32>() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn release_cache_record_sizes() {
        let cfg = MiniMaxM3Config::from_hf(&release_config()).unwrap();
        assert_eq!(kv_bytes_per_token(&cfg, 1), 61_440);
        assert_eq!(kv_bytes_per_token(&cfg, 2), 122_880);
        assert_eq!(index_bytes_per_token(&cfg), 7_920);
    }
}
