//! Exact paged cache storage for MiniMax M3 on one coordinator GPU or two that
//! split the KV heads: per layer, K and V pages of `PAGE_ROWS` tokens (the b12x
//! MSA attention's 64-row pages; a 128-token MSA block is two consecutive
//! logical pages) and the MSA index-key pages in the paged indexer layout
//! (`uint8[pages, 64 * (128 + 4)]`: 64 E4M3 rows, then one FP32 scale per row).
//! The index key is one shared head, so every rank keeps all of it.
//!
//! Pure paged state: no recurrent or sliding state, so an exact prefix restore
//! is a page-reference copy and marks hold nothing. This is storage, not serving
//! admission: weights, workspaces, graphs and transport reserve separately.
use super::config::MiniMaxM3Config;
use crate::{families::qwen4::Qwen4KvCache,
    serving_capacity::{CacheGeometryError, FamilyCacheGeometry, KvPlacement, RankCacheGeometry}};

/// Tokens per physical page.
pub const PAGE_ROWS: u64 = 64;
/// Pages per MSA selection block (`block / PAGE_ROWS`).
pub const PAGES_PER_BLOCK: u64 = 2;
/// One E4M3 index row plus its FP32 scale.
pub const INDEX_ROW_BYTES: u64 = 128 + 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RankPages {
    /// KV heads this rank owns, `first..first + heads`.
    pub first_kv_head: usize,
    pub kv_heads: usize,
    /// One K (or V) page of one layer: `[PAGE_ROWS, kv_heads, head_dim]`.
    pub kv_page_bytes: u64,
    /// One index-key page of one layer.
    pub index_page_bytes: u64,
    /// Every layer's K, V and index pages for one page of tokens.
    pub page_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MiniMaxCacheLayout {
    pub kv_format: Qwen4KvCache,
    pub layers: usize,
    pub ranks: Vec<RankPages>,
    /// Requested context; pages per sequence cover it in whole MSA blocks.
    pub context: u64,
}

fn invalid(what: &'static str) -> CacheGeometryError {
    CacheGeometryError::Unsupported { family: "minimax_m3", what }
}

impl MiniMaxCacheLayout {
    pub fn new(cfg: &MiniMaxM3Config, ranks: usize, context: u64, kv_format: Qwen4KvCache) -> Result<Self, CacheGeometryError> {
        if !(1..=2).contains(&ranks) || cfg.kv_heads % ranks != 0 || cfg.msa.block as u64 != PAGE_ROWS * PAGES_PER_BLOCK
            || cfg.msa.index_dim as u64 + 4 != INDEX_ROW_BYTES {
            return Err(invalid("one GPU or two splitting KV heads, 128-token MSA blocks over 64-row pages"));
        }
        if context == 0 || context > cfg.max_position as u64 {
            return Err(invalid("context outside the checkpoint extent"));
        }
        let element = match kv_format { Qwen4KvCache::Fp8 => 1, Qwen4KvCache::Bf16 => 2 };
        let heads = cfg.kv_heads / ranks;
        let kv_page_bytes = PAGE_ROWS * (heads * cfg.head_dim * element) as u64;
        let index_page_bytes = PAGE_ROWS * INDEX_ROW_BYTES;
        let sparse = cfg.sparse.iter().filter(|s| **s).count() as u64;
        let page_bytes = (cfg.layers as u64).checked_mul(2 * kv_page_bytes)
            .and_then(|kv| kv.checked_add(sparse * index_page_bytes))
            .ok_or(CacheGeometryError::Overflow("MiniMax page bytes"))?;
        Ok(Self { kv_format, layers: cfg.layers, context, ranks: (0..ranks).map(|rank| RankPages {
            first_kv_head: rank * heads, kv_heads: heads, kv_page_bytes, index_page_bytes, page_bytes }).collect() })
    }

    /// Pages one sequence of `tokens` occupies: whole MSA blocks, so a block's
    /// two pages are always both present.
    pub fn sequence_pages(&self, tokens: u64) -> u64 {
        tokens.div_ceil(PAGE_ROWS * PAGES_PER_BLOCK) * PAGES_PER_BLOCK
    }

    /// Bytes per token on `rank` (for pool sizing and reports).
    pub fn token_bytes(&self, rank: usize) -> u64 { self.ranks[rank].page_bytes / PAGE_ROWS }

    pub fn geometry(&self) -> FamilyCacheGeometry {
        FamilyCacheGeometry {
            logical_unit_rows: PAGE_ROWS * PAGES_PER_BLOCK,
            placement: if self.ranks.len() == 1 { KvPlacement::SingleDevice } else { KvPlacement::PartitionedHeads },
            ranks: self.ranks.iter().map(|rank| RankCacheGeometry {
                persistent_unit_bytes: rank.page_bytes * PAGES_PER_BLOCK, ..Default::default() }).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::families::minimax_m3::{config::tests::release_config, reference};

    #[test]
    fn release_page_bytes() {
        let cfg = MiniMaxM3Config::from_hf(&release_config()).unwrap();
        let one = MiniMaxCacheLayout::new(&cfg, 1, 1 << 20, Qwen4KvCache::Fp8).unwrap();
        assert_eq!(one.ranks[0].kv_page_bytes, 64 * 4 * 128);
        assert_eq!(one.ranks[0].index_page_bytes, 64 * 132);
        assert_eq!(one.token_bytes(0), (reference::kv_bytes_per_token(&cfg, 1) + reference::index_bytes_per_token(&cfg)) as u64);
        assert_eq!(one.token_bytes(0), 69_360);
        let two = MiniMaxCacheLayout::new(&cfg, 2, 1 << 20, Qwen4KvCache::Bf16).unwrap();
        assert_eq!((two.ranks[1].first_kv_head, two.ranks[1].kv_heads), (2, 2));
        assert_eq!(two.token_bytes(1), 60 * 2 * 2 * 128 * 2 + 60 * 132);
        assert_eq!(one.sequence_pages(1), 2);
        assert_eq!(one.sequence_pages(129), 4);
        assert_eq!(one.geometry().ranks[0].persistent_unit_bytes, 128 * 69_360);
        assert!(MiniMaxCacheLayout::new(&cfg, 3, 1 << 20, Qwen4KvCache::Fp8).is_err());
        assert!(MiniMaxCacheLayout::new(&cfg, 1, (1 << 20) + 1, Qwen4KvCache::Fp8).is_err());
    }
}
