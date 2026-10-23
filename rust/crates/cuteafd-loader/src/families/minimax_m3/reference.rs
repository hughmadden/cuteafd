//! CPU reference contracts for MiniMax M3 arithmetic the native programs must
//! reproduce: the SwiGLU-OAI FFN, the sigmoid router, Gemma RMSNorm, partial
//! NeoX RoPE and MSA block selection. FP32/FP64 throughout; goldens and kernel
//! tests compare against these, never the other way round.
use super::config::{MiniMaxM3Config, MsaConfig};

/// `gate.clamp(max=L) * sigmoid(alpha * gate.clamp(max=L)) * (up.clamp(-L, L) + 1)`.
pub fn swiglu_oai(gate: f32, up: f32, alpha: f32, limit: f32) -> f32 {
    let gate = gate.min(limit);
    let up = up.clamp(-limit, limit);
    gate / (1.0 + (-alpha * gate).exp()) * (up + 1.0)
}

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

/// Gemma RMSNorm: `x / rms(x) * (1 + w)`, in FP32.
pub fn gemma_rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    assert_eq!(x.len(), weight.len());
    let mean = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (mean + eps).sqrt();
    x.iter().zip(weight).map(|(v, w)| v * inv * (1.0 + w)).collect()
}

/// NeoX RoPE on the first `rotary` channels of one head (`rotate_half` over
/// those channels: pairs `(i, i + rotary/2)`); the rest pass through.
pub fn rope_partial(head: &mut [f32], position: u64, rotary: usize, theta: f64) {
    let half = rotary / 2;
    for i in 0..half {
        let freq = theta.powf(-((2 * i) as f64) / rotary as f64);
        let (sin, cos) = (position as f64 * freq).sin_cos();
        let (a, b) = (head[i] as f64, head[i + half] as f64);
        head[i] = (a * cos - b * sin) as f32;
        head[i + half] = (b * cos + a * sin) as f32;
    }
}

/// MSA block selection for one query row and one index head, in the
/// attention kernel's contract: ascending block ids, no duplicates, the local
/// blocks included, `-1` padding to `topk_blocks`.
///
/// `scores[t]` is the index score of key `t` for keys `0..=position` (the
/// causal prefix). A block scores the max of its keys; the `local_blocks`
/// blocks ending at the query's own block always win a slot. Ties prefer the
/// lower block id.
pub fn msa_select(msa: &MsaConfig, scores: &[f32], position: usize) -> Vec<i32> {
    assert!(scores.len() > position, "a score per causal key");
    let blocks = position / msa.block + 1;
    let own = position / msa.block;
    let mut block_scores: Vec<f32> = (0..blocks).map(|b| {
        let end = ((b + 1) * msa.block).min(position + 1);
        scores[b * msa.block..end].iter().copied().fold(f32::NEG_INFINITY, f32::max)
    }).collect();
    for local in 0..msa.local_blocks {
        if let Some(b) = own.checked_sub(local) { block_scores[b] = f32::INFINITY; }
    }
    let mut order: Vec<usize> = (0..blocks).collect();
    order.sort_by(|&a, &b| block_scores[b].total_cmp(&block_scores[a]).then(a.cmp(&b)));
    order.truncate(msa.topk_blocks);
    order.sort_unstable();
    let mut out: Vec<i32> = order.into_iter().map(|b| b as i32).collect();
    out.resize(msa.topk_blocks, -1);
    out
}

/// Keys one query row attends after selection (the visible prefix of the
/// selected blocks), for capacity and parity checks.
pub fn msa_attended(msa: &MsaConfig, selection: &[i32], position: usize) -> usize {
    selection.iter().filter(|b| **b >= 0).map(|&b| {
        let start = b as usize * msa.block;
        ((b as usize + 1) * msa.block).min(position + 1) - start
    }).sum()
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

    fn msa() -> MsaConfig { MiniMaxM3Config::from_hf(&release_config()).unwrap().msa }

    #[test]
    fn swiglu_oai_clamps_like_the_reference() {
        let (a, l) = (1.702, 7.0);
        assert_eq!(swiglu_oai(0.0, 3.0, a, l), 0.0);
        // gate clamps above only; up clamps both ways.
        assert_eq!(swiglu_oai(9.0, 0.0, a, l), swiglu_oai(7.0, 0.0, a, l));
        assert_eq!(swiglu_oai(2.0, -20.0, a, l), swiglu_oai(2.0, -7.0, a, l));
        assert!(swiglu_oai(-9.0, 0.0, a, l) != swiglu_oai(-7.0, 0.0, a, l));
        let expected = 2.0 / (1.0 + (-1.702f32 * 2.0).exp()) * 1.5;
        assert!((swiglu_oai(2.0, 0.5, a, l) - expected).abs() < 1e-6);
    }

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
    fn gemma_norm_adds_one() {
        let out = gemma_rms_norm(&[3.0, 4.0], &[0.0, 1.0], 0.0);
        let rms = (12.5f32).sqrt();
        assert!((out[0] - 3.0 / rms).abs() < 1e-6 && (out[1] - 8.0 / rms).abs() < 1e-6);
    }

    #[test]
    fn rope_rotates_leading_channels_only() {
        let mut head: Vec<f32> = (0..128).map(|i| i as f32).collect();
        let original = head.clone();
        rope_partial(&mut head, 0, 64, 1e7);
        assert_eq!(head, original);
        rope_partial(&mut head, 5, 64, 1e7);
        assert_eq!(&head[64..], &original[64..]);
        // Channel 0 pairs with 32 at the highest frequency.
        let (sin, cos) = (5.0f64).sin_cos();
        assert!((head[0] as f64 - (0.0 * cos - 32.0 * sin)).abs() < 1e-4);
        assert!((head[32] as f64 - (32.0 * cos + 0.0 * sin)).abs() < 1e-4);
    }

    #[test]
    fn short_contexts_select_every_block() {
        let msa = msa();
        let scores = vec![0.0; 2048];
        assert_eq!(msa_select(&msa, &scores, 0)[..2], [0, -1]);
        let all = msa_select(&msa, &scores, 2047);
        assert_eq!(all, (0..16).collect::<Vec<i32>>());
        assert_eq!(msa_attended(&msa, &all, 2047), 2048);
    }

    #[test]
    fn long_contexts_keep_the_local_block_and_the_best_maxima() {
        let msa = msa();
        let position = 128 * 40 + 5;
        let mut scores = vec![-1.0; position + 1];
        // One hot key wins its block regardless of the block's other keys.
        for b in [3usize, 7, 11, 13, 17, 19, 23, 29, 31, 33, 35, 36, 37, 38, 39] { scores[b * 128 + 77] = 10.0 + b as f32; }
        // The local block (40) scores lowest but is forced in.
        let selection = msa_select(&msa, &scores, position);
        assert_eq!(selection, [3, 7, 11, 13, 17, 19, 23, 29, 31, 33, 35, 36, 37, 38, 39, 40]);
        assert_eq!(msa_attended(&msa, &selection, position), 15 * 128 + 6);
        // A future key never counts: the reference masks it before pooling.
        assert_eq!(msa_select(&msa, &scores[..=position], position), selection);
    }

    #[test]
    fn release_cache_record_sizes() {
        let cfg = MiniMaxM3Config::from_hf(&release_config()).unwrap();
        assert_eq!(kv_bytes_per_token(&cfg, 1), 61_440);
        assert_eq!(kv_bytes_per_token(&cfg, 2), 122_880);
        assert_eq!(index_bytes_per_token(&cfg), 7_920);
    }
}
