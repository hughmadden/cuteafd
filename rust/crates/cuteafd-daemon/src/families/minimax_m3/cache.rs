//! Separate E4M3 K/V planes and packed index pages, with no positional marks.
//!
//! Host K/V placement: each layer's K and V plane is one virtual range whose
//! pages `[0, resident)` are device memory and `[resident, pages)` pinned host
//! memory (CUDA VMM), so every kernel, copy and snapshot addresses a page the
//! same way wherever it lives. Index pages stay device memory: every step's
//! block selection scans them all.
use super::{execution::LayerCache, MiniMaxError};
use anyhow::{Context, Result};
use cuteafd_ffi::{minimax::Buffer, CuteafdDeviceBuffer, NativeLibrary};
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
    /// `resident` K/V pages of `pages` on the GPU, the rest in pinned host memory.
    pub fn new(library: &'a NativeLibrary, device: i32, layout: &MiniMaxCacheLayout,
        pages: usize, resident: usize, sequences: usize, admitted: CacheBudget) -> Result<Self> {
        if layout.ranks.len() != 1 || layout.ranks[0].kv_page_bytes != KV_PAGE_BYTES as u64
            || layout.ranks[0].index_page_bytes != INDEX_PAGE_BYTES as u64
            || pages == 0 || pages > i32::MAX as usize || resident > pages || !(1..=64).contains(&sequences) {
            return Err(MiniMaxError::Invalid("native cache geometry").into());
        }
        let (gpu, host) = if resident == pages {
            (layout.ranks[0].page_bytes.checked_mul(pages as u64).ok_or(MiniMaxError::Overflow("cache"))?, 0)
        } else { layout.tiered_bytes(0, pages as u64, resident as u64)? };
        let required = gpu.checked_add(256).ok_or(MiniMaxError::Overflow("cache"))?;
        if required > admitted.gpu || host > admitted.host { return Err(MiniMaxError::Invalid("cache admission").into()); }
        let (keys, values) = if resident == pages {
            let plane = || (0..layout.layers).map(|_| Buffer::new(library, device, pages * KV_PAGE_BYTES)).collect::<Result<Vec<_>>>();
            (plane()?, plane()?)
        } else {
            let granule = { library.cuda_set_device(device)?; library.tiered_granularity()? };
            if granule == 0 || TIER_GRANULE_BYTES as usize % granule != 0 {
                return Err(MiniMaxError::Unsupported("tiered page granularity").into());
            }
            let unit = layout.tier_unit_pages(0) as usize;
            let (device_bytes, host_bytes) = (resident * KV_PAGE_BYTES, (pages - resident).div_ceil(unit) * unit * KV_PAGE_BYTES);
            let plane = || Buffer::tiered(library, device, layout.layers, device_bytes, host_bytes, 0)
                .context("MiniMax host K/V placement");
            (plane()?, plane()?)
        };
        let mut layers = Vec::with_capacity(layout.layers);
        for (key, value) in keys.into_iter().zip(values) {
            layers.push(LayerCache { key, value, index: Buffer::new(library, device, pages * INDEX_PAGE_BYTES)? });
        }
        let tail = Buffer::new(library, device, 256)?;
        tail.upload(&[0; 256])?;
        if resident < pages {
            tracing::info!(pages, resident, host_pages = pages - resident, gpu_bytes = gpu, host_bytes = host,
                "MiniMax host K/V placement: index pages and resident K/V on the GPU, the rest pinned host memory");
        }
        Ok(Self { layers, pages, resident, context: layout.context as usize, sequences, tail, library, device })
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
}
