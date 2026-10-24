//! Exact paged cache storage for MiniMax M3 on its one coordinator GPU: per layer, K and V
//! pages of `PAGE_ROWS` tokens (the b12x MSA attention's 64-row pages; a 128-token MSA block is
//! two consecutive logical pages) and the MSA index-key pages in the paged indexer layout
//! (`uint8[pages, 64 * (128 + 4)]`: 64 E4M3 rows, then one FP32 scale per row).
//!
//! Pure paged state: no recurrent or sliding state, so an exact prefix restore is a
//! page-reference copy and marks hold nothing. This is storage, not serving admission: weights,
//! workspaces, graphs and transport reserve separately.
use super::config::MiniMaxM3Config;
use crate::{families::qwen4::Qwen4KvCache,
    serving_capacity::{CacheGeometryError, FamilyCacheGeometry, KvPlacement, RankCacheGeometry}};

/// Tokens per physical page.
pub const PAGE_ROWS: u64 = 64;
/// Pages per MSA selection block (`block / PAGE_ROWS`).
pub const PAGES_PER_BLOCK: u64 = 2;
/// One E4M3 index row plus its FP32 scale.
pub const INDEX_ROW_BYTES: u64 = 128 + 4;
pub const INDEX_PAGE_BYTES: u64 = 8448;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MiniMaxCacheLayout {
    pub layers: usize,
    /// One K (or V) page of one layer: `[PAGE_ROWS, kv_heads, head_dim]` E4M3.
    pub kv_page_bytes: u64,
    /// One index-key page of one layer.
    pub index_page_bytes: u64,
    /// Every layer's K, V and index pages for one page of tokens.
    pub page_bytes: u64,
    /// Requested context.
    pub context: u64,
}

fn invalid(what: &'static str) -> CacheGeometryError {
    CacheGeometryError::Unsupported { family: "minimax_m3", what }
}

impl MiniMaxCacheLayout {
    /// The native ABI: separate E4M3 K/V planes and packed index pages.
    pub fn new(cfg: &MiniMaxM3Config, context: u64, kv_format: Qwen4KvCache) -> Result<Self, CacheGeometryError> {
        if kv_format != Qwen4KvCache::Fp8 { return Err(invalid("MiniMax native MSA requires FP8 KV; BF16 KV has no export")); }
        if cfg.msa.block as u64 != PAGE_ROWS * PAGES_PER_BLOCK || cfg.msa.index_dim as u64 + 4 != INDEX_ROW_BYTES {
            return Err(invalid("128-token MSA blocks over 64-row pages"));
        }
        if context == 0 || context > cfg.max_position as u64 {
            return Err(invalid("context outside the checkpoint extent"));
        }
        let kv_page_bytes = PAGE_ROWS * (cfg.kv_heads * cfg.head_dim) as u64;
        let index_page_bytes = PAGE_ROWS * INDEX_ROW_BYTES;
        let sparse = cfg.sparse.iter().filter(|s| **s).count() as u64;
        let page_bytes = (cfg.layers as u64).checked_mul(2 * kv_page_bytes)
            .and_then(|kv| kv.checked_add(sparse * index_page_bytes))
            .ok_or(CacheGeometryError::Overflow("MiniMax page bytes"))?;
        Ok(Self { layers: cfg.layers, kv_page_bytes, index_page_bytes, page_bytes, context })
    }

    /// Bytes per token (for pool sizing and reports).
    pub fn token_bytes(&self) -> u64 { self.page_bytes / PAGE_ROWS }

    pub fn geometry(&self) -> FamilyCacheGeometry {
        FamilyCacheGeometry {
            logical_unit_rows: PAGE_ROWS * PAGES_PER_BLOCK,
            placement: KvPlacement::SingleDevice,
            ranks: vec![RankCacheGeometry { persistent_unit_bytes: self.page_bytes * PAGES_PER_BLOCK, ..Default::default() }],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::families::minimax_m3::config::tests::release_config;

    #[test]
    fn release_page_bytes() {
        let cfg = MiniMaxM3Config::from_hf(&release_config()).unwrap();
        let cache = MiniMaxCacheLayout::new(&cfg, 1 << 20, Qwen4KvCache::Fp8).unwrap();
        assert_eq!(cache.kv_page_bytes, 64 * 4 * 128);
        assert_eq!(cache.index_page_bytes, INDEX_PAGE_BYTES);
        // K and V for every KV head of 60 layers, plus one 132-byte index row per layer.
        assert_eq!(cache.token_bytes(), 60 * 2 * 4 * 128 + 60 * 132);
        assert_eq!(cache.token_bytes(), 69_360);
        assert_eq!(cache.geometry().ranks[0].persistent_unit_bytes, 128 * 69_360);
        assert!(MiniMaxCacheLayout::new(&cfg, (1 << 20) + 1, Qwen4KvCache::Fp8).is_err());
        assert!(MiniMaxCacheLayout::new(&cfg, 1024, Qwen4KvCache::Bf16).is_err());
    }
}
