//! Separate E4M3 K/V planes and packed index pages, with no positional marks.
//!
//! Host K/V placement: each layer's K and V plane is one virtual range whose
//! pages `[0, resident)` are device memory and `[resident, pages)` pinned host
//! memory (CUDA VMM), so every kernel, copy and snapshot addresses a page the
//! same way wherever it lives. Index pages stay device memory: every step's
//! block selection scans them all.
use super::{execution::{KvStaging, LayerCache}, MiniMaxError};
use anyhow::{Context, Result};
use cuteafd_ffi::{minimax::{kv_stage_scratch_bytes, Buffer}, CuteafdDeviceBuffer, NativeLibrary};
use cuteafd_loader::families::minimax_m3::cache::{MiniMaxCacheLayout, TIER_GRANULE_BYTES};
use cuteafd_engine::prefix::TailCopy;

pub(super) const PAGE_ROWS: usize = 64;
pub(super) const KV_PAGE_BYTES: usize = PAGE_ROWS * 4 * 128;
pub(super) const INDEX_PAGE_BYTES: usize = PAGE_ROWS * 132;

pub(super) struct Cache<'a> {
    pub layers: Vec<LayerCache<'a>>,
    pub pages: usize,
    /// K/V pages below this are device memory (`pages` without host placement).
    pub resident: usize,
    pub context: usize,
    pub sequences: usize,
    pub tail: Buffer<'a>,
    pub library: &'a NativeLibrary,
    pub device: i32,
}

/// Admitted bytes of one cache: GPU (index pages, resident K/V and the tail
/// slot) and pinned host (non-resident K/V).
#[derive(Debug, Clone, Copy)]
pub(super) struct CacheBudget { pub gpu: u64, pub host: u64 }

/// Offsets for one partial page, including the index scales' separate suffix.
pub(super) fn copy_spans(copy: TailCopy, pages: usize)
    -> std::result::Result<Vec<(usize, usize, usize, usize)>, MiniMaxError> {
    if copy.owner.is_some() || copy.rows > PAGE_ROWS || copy.from == copy.to
        || copy.from as usize >= pages || copy.to as usize >= pages {
        return Err(MiniMaxError::Invalid("cache tail copy"));
    }
    let (from, to, rows) = (copy.from as usize, copy.to as usize, copy.rows);
    Ok(vec![
        (0, from * KV_PAGE_BYTES, to * KV_PAGE_BYTES, rows * 512),
        (1, from * KV_PAGE_BYTES, to * KV_PAGE_BYTES, rows * 512),
        (2, from * INDEX_PAGE_BYTES, to * INDEX_PAGE_BYTES, rows * 128),
        (2, from * INDEX_PAGE_BYTES + 8192, to * INDEX_PAGE_BYTES + 8192, rows * 4),
    ])
}

impl<'a> Cache<'a> {
    /// `resident` K/V pages of `pages` on the GPU, the rest in pinned host memory
    /// staged per attention launch through a translated page table of
    /// `table_bytes` (the size of the execution's page table).
    #[allow(clippy::too_many_arguments)]
    pub fn new(library: &'a NativeLibrary, device: i32, layout: &MiniMaxCacheLayout, pages: usize,
        resident: usize, sequences: usize, table_bytes: usize, admitted: CacheBudget) -> Result<Self> {
        if layout.ranks.len() != 1 || layout.ranks[0].kv_page_bytes != KV_PAGE_BYTES as u64
            || layout.ranks[0].index_page_bytes != INDEX_PAGE_BYTES as u64
            || pages == 0 || pages > i32::MAX as usize || resident > pages || !(1..=64).contains(&sequences) {
            return Err(MiniMaxError::Invalid("native cache geometry").into());
        }
        let (gpu, host) = if resident == pages {
            (layout.ranks[0].page_bytes.checked_mul(pages as u64).ok_or(MiniMaxError::Overflow("cache"))?, 0)
        } else {
            let (gpu, host) = layout.tiered_bytes(0, pages as u64, resident as u64)?;
            (gpu + layout.staging_bytes(0, pages as u64, resident as u64) + table_bytes as u64, host)
        };
        let required = gpu.checked_add(256).ok_or(MiniMaxError::Overflow("cache"))?;
        if required > admitted.gpu || host > admitted.host { return Err(MiniMaxError::Invalid("cache admission").into()); }
        let (keys, values, stage) = if resident == pages {
            let plane = || (0..layout.layers).map(|_| Buffer::new(library, device, pages * KV_PAGE_BYTES)).collect::<Result<Vec<_>>>();
            (plane()?, plane()?, None)
        } else {
            let granule = { library.cuda_set_device(device)?; library.tiered_granularity()? };
            if granule == 0 || TIER_GRANULE_BYTES as usize % granule != 0 {
                return Err(MiniMaxError::Unsupported("tiered page granularity").into());
            }
            let unit = layout.tier_unit_pages(0) as usize;
            let base = resident + (pages - resident).div_ceil(unit) * unit;
            // CUTEAFD_MINIMAX_KV_STAGE=off: attention reads host pages in place
            // (diagnostic; the admission still reserves the staging bytes).
            let staged = !matches!(std::env::var("CUTEAFD_MINIMAX_KV_STAGE").as_deref(), Ok("off" | "0"));
            let capacity = if staged { layout.staging_pages(0, pages as u64, resident as u64) as usize } else { 0 };
            let plane = || Buffer::tiered(library, device, layout.layers, resident * KV_PAGE_BYTES,
                (base - resident) * KV_PAGE_BYTES, capacity * KV_PAGE_BYTES).context("MiniMax host K/V placement");
            let stage = staged.then(|| -> Result<_> {
                let stage = KvStaging { resident, pool_pages: pages, base, capacity,
                    scratch: Buffer::new(library, device, kv_stage_scratch_bytes(pages, capacity))?,
                    stats: Buffer::new(library, device, 256)?, table: Buffer::new(library, device, table_bytes)? };
                stage.stats.upload(&[0; 256])?;
                Ok(stage)
            }).transpose()?;
            (plane()?, plane()?, stage)
        };
        let mut layers = Vec::with_capacity(layout.layers);
        for (key, value) in keys.into_iter().zip(values) {
            layers.push(LayerCache { key, value, index: Buffer::new(library, device, pages * INDEX_PAGE_BYTES)?,
                stage: stage.clone() });
        }
        let tail = Buffer::new(library, device, 256)?;
        tail.upload(&[0; 256])?;
        if resident < pages {
            tracing::info!(pages, resident, host_pages = pages - resident, staging_pages = stage.as_ref().map_or(0, |s| s.capacity),
                gpu_bytes = gpu, host_bytes = host, "MiniMax host K/V placement: index pages and resident K/V on the GPU, \
                the rest pinned host memory staged per attention launch (0 staging pages: read in place)");
        }
        Ok(Self { layers, pages, resident, context: layout.context as usize, sequences, tail, library, device })
    }

    /// Host placement's cumulative staging counters: (head slices copied from
    /// host pages, selected host-page reads left in place for lack of staging room).
    pub fn stage_stats(&self) -> Result<Option<(u64, u64)>> {
        let Some(stage) = self.layers.first().and_then(|l| l.stage.as_ref()) else { return Ok(None) };
        let mut bytes = [0u8; 16];
        self.library.cuda_set_device(self.device)?;
        self.library.copy_d2h(&mut bytes, stage.stats.view(0, 16)?.raw())?;
        let word = |i: usize| u64::from_le_bytes(bytes[i * 8..i * 8 + 8].try_into().expect("8 bytes"));
        Ok(Some((word(0), word(1))))
    }

    pub fn buffers(&self) -> Vec<CuteafdDeviceBuffer> {
        self.layers.iter().flat_map(|l| [l.key.raw(), l.value.raw(), l.index.raw()])
            .chain([self.tail.raw()]).collect()
    }

    pub fn copy_rows(&self, copy: TailCopy) -> Result<()> {
        let spans = copy_spans(copy, self.pages)?;
        self.library.cuda_set_device(self.device)?;
        for layer in &self.layers {
            let planes = [&layer.key, &layer.value, &layer.index];
            for &(plane, from, to, bytes) in &spans {
                if bytes == 0 { continue; }
                self.library.copy_d2d(planes[plane].view(to, bytes)?.raw(),
                    planes[plane].view(from, bytes)?.raw(), bytes)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn minimax_tail_copy_preserves_every_visible_byte_and_nothing_else() {
        for rows in [0, 1, 17, 63, 64] {
            let sizes = [KV_PAGE_BYTES, KV_PAGE_BYTES, INDEX_PAGE_BYTES];
            let mut planes: Vec<_> = sizes.iter().map(|&n| (0..n * 3)
                .map(|i| ((i * 7 + i / n) % 251) as u8).collect::<Vec<_>>()).collect();
            let before = planes.clone();
            let spans = copy_spans(TailCopy { from: 1, to: 2, rows, owner: None }, 3).unwrap();
            for &(p, from, to, n) in &spans { planes[p].copy_within(from..from+n, to); }
            for p in 0..3 { for i in 0..planes[p].len() {
                let source = spans.iter().find(|&&(q, _, to, n)| q == p && (to..to+n).contains(&i));
                assert_eq!(planes[p][i], source.map_or(before[p][i], |&(_, from, to, _)| before[p][from+i-to]));
            }}
        }
        assert!(copy_spans(TailCopy { from: 0, to: 1, rows: 65, owner: None }, 2).is_err());
        assert!(copy_spans(TailCopy { from: 0, to: 2, rows: 1, owner: None }, 2).is_err());
    }

    #[test]
    fn minimax_staging_ledger_matches_native_scratch() {
        use cuteafd_loader::families::{minimax_m3::cache::RankPages, qwen4::Qwen4KvCache};
        let layout = MiniMaxCacheLayout { kv_format: Qwen4KvCache::Fp8, layers: 60, context: 1 << 20,
            ranks: vec![RankPages { first_kv_head: 0, kv_heads: 4, kv_page_bytes: KV_PAGE_BYTES as u64,
                index_page_bytes: INDEX_PAGE_BYTES as u64, page_bytes: 60 * (2 * KV_PAGE_BYTES + INDEX_PAGE_BYTES) as u64 }] };
        for (pages, resident, staging) in [(16_384u64, 1_664u64, 4096usize), (2048, 0, 2048), (1280, 64, 1216), (100, 64, 64)] {
            assert_eq!(layout.staging_pages(0, pages, resident) as usize, staging);
            assert_eq!(layout.staging_bytes(0, pages, resident) as usize,
                2 * staging * KV_PAGE_BYTES + kv_stage_scratch_bytes(pages as usize, staging) + 256);
        }
    }
}
