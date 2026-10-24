//! CPU reference for the MiniMax M3 sigmoid router. The arithmetic oracle
//! (SwiGLU-OAI, Gemma norms, partial RoPE, MSA selection) is the Python
//! reference the goldens use, `python/reference/families/minimax_m3/model.py`.

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

#[cfg(test)]
mod tests {
    use super::*;

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
}
