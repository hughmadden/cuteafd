//! Separate E4M3 K/V planes and packed index pages, with no positional marks.
use super::{execution::LayerCache, MiniMaxError};
use anyhow::Result;
use cuteafd_ffi::{minimax::Buffer, CuteafdDeviceBuffer, NativeLibrary};
use cuteafd_loader::families::minimax_m3::cache::MiniMaxCacheLayout;
use cuteafd_engine::prefix::TailCopy;

pub(super) const PAGE_ROWS: usize = 64;
pub(super) const KV_PAGE_BYTES: usize = PAGE_ROWS * 4 * 128;
pub(super) const INDEX_PAGE_BYTES: usize = PAGE_ROWS * 132;

pub(super) struct Cache<'a> {
    pub layers: Vec<LayerCache<'a>>,
    pub pages: usize,
    pub context: usize,
    pub sequences: usize,
    pub tail: Buffer<'a>,
    pub library: &'a NativeLibrary,
    pub device: i32,
}

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
    pub fn new(library: &'a NativeLibrary, device: i32, layout: &MiniMaxCacheLayout,
        pages: usize, sequences: usize, admitted: u64) -> Result<Self> {
        if layout.ranks.len() != 1 || layout.ranks[0].kv_page_bytes != KV_PAGE_BYTES as u64
            || layout.ranks[0].index_page_bytes != INDEX_PAGE_BYTES as u64
            || pages == 0 || pages > i32::MAX as usize || !(1..=64).contains(&sequences) {
            return Err(MiniMaxError::Invalid("native cache geometry").into());
        }
        let required = layout.ranks[0].page_bytes.checked_mul(pages as u64)
            .and_then(|n| n.checked_add(256)).ok_or(MiniMaxError::Overflow("cache"))?;
        if required > admitted { return Err(MiniMaxError::Invalid("cache admission").into()); }
        let mut layers = Vec::with_capacity(layout.layers);
        for _ in 0..layout.layers {
            layers.push(LayerCache {
                key: Buffer::new(library, device, pages * KV_PAGE_BYTES)?,
                value: Buffer::new(library, device, pages * KV_PAGE_BYTES)?,
                index: Buffer::new(library, device, pages * INDEX_PAGE_BYTES)?,
            });
        }
        let tail = Buffer::new(library, device, 256)?;
        tail.upload(&[0; 256])?;
        Ok(Self { layers, pages, context: layout.context as usize, sequences, tail, library, device })
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
