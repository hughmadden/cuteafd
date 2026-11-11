//! MiniMax's MSA instance of the shared sparse-selection contract.
use crate::shared::sparse_indexer::{
    Activation, ForcedWindows, Granularity, HeadReduction, IndexerGeometry, Norm, Pool, Rope,
    RopeStyle,
};

use super::MiniMaxM3Config;

impl MiniMaxM3Config {
    pub fn sparse_indexer(&self) -> IndexerGeometry {
        IndexerGeometry {
            query_heads: self.msa.index_heads,
            key_heads: 1,
            head_dim: self.msa.index_dim,
            rope: Some(Rope { dim: self.rotary_dim, style: RopeStyle::NeoX }),
            query_norm: Norm::GemmaRmsNorm,
            key_norm: Norm::GemmaRmsNorm,
            activation: Activation::Identity,
            head_reduction: HeadReduction::None,
            selection_heads: self.msa.index_heads,
            granularity: Granularity::Block { tokens: self.msa.block, pool: Pool::Max },
            topk: self.msa.topk_blocks,
            initial_blocks: 0,
            local_blocks: self.msa.local_blocks,
            forced_windows: ForcedWindows::WithinTopK,
            index_values: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn release() -> MiniMaxM3Config {
        MiniMaxM3Config::from_hf(&serde_json::from_str(include_str!("fixtures/config.json")).unwrap()).unwrap()
    }

    #[test]
    fn native_msa_geometry_and_exact_owner_payloads() -> anyhow::Result<()> {
        let g = release().sparse_indexer();
        g.validate()?;
        assert_eq!((g.query_heads, g.key_heads, g.head_dim, g.selection_heads), (4, 1, 128, 4));
        assert_eq!(g.rope, Some(Rope { dim: 64, style: RopeStyle::NeoX }));
        assert_eq!((g.query_norm, g.key_norm), (Norm::GemmaRmsNorm, Norm::GemmaRmsNorm));
        assert_eq!(g.granularity, Granularity::Block { tokens: 128, pool: Pool::Max });
        assert_eq!((g.topk, g.initial_blocks, g.local_blocks), (16, 0, 1));
        assert_eq!(g.forced_windows, ForcedWindows::WithinTopK);
        assert!(!g.index_values);
        assert_eq!(g.selection_width()?, 16);
        assert_eq!(g.selection_bytes(8192)?, 8192 * 4 * 16 * 4);
        assert_eq!(g.key_cache_bytes(8192, 1)?, 8192 * 128);
        Ok(())
    }

    #[test]
    fn signed_heads_remain_independent() -> anyhow::Result<()> {
        let g = release().sparse_indexer();
        let mut query = vec![0.; 4 * 128];
        for (head, value) in [1., -1., 2., -2.].into_iter().enumerate() {
            query[head * 128] = value;
        }
        let mut keys = vec![0.; 3 * 128];
        for (token, value) in [-3., 2., -1.].into_iter().enumerate() {
            keys[token * 128] = value;
        }
        assert_eq!(g.scores(&query, &keys, None)?, vec![
            vec![-3., 2., -1.], vec![3., -2., 1.],
            vec![-6., 4., -2.], vec![6., -4., 2.],
        ]);
        Ok(())
    }

    #[test]
    fn selection_matches_independent_reference_vectors() -> anyhow::Result<()> {
        let g = release().sparse_indexer();
        let fixture: Value = serde_json::from_str(include_str!("fixtures/msa-vectors.json"))?;
        for case in fixture["cases"].as_array().unwrap() {
            let position = case["position"].as_u64().unwrap() as usize;
            let scores: Vec<Vec<f32>> = case["block_scores"].as_array().unwrap().iter().map(|head| {
                head.as_array().unwrap().iter().flat_map(|score|
                    std::iter::repeat_n(score.as_f64().unwrap() as f32, 128)).collect()
            }).collect();
            let mut scores = scores;
            for update in case["overrides"].as_array().unwrap() {
                scores[update[0].as_u64().unwrap() as usize][update[1].as_u64().unwrap() as usize] =
                    update[2].as_f64().unwrap() as f32;
            }
            let expected: Vec<Vec<i32>> = serde_json::from_value(case["selected"].clone())?;
            assert_eq!(g.select(&scores, position)?, expected, "{}", case["name"]);
        }
        Ok(())
    }
}
