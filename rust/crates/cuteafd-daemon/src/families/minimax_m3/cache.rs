//! Separate E4M3 K/V planes and packed index pages, with no positional marks.
use super::{execution::LayerCache, MiniMaxError};
use crate::shared::prefix::PagedPlane;
use anyhow::Result;
use cuteafd_ffi::{minimax::Buffer, CuteafdDeviceBuffer, NativeLibrary};
use cuteafd_loader::families::minimax_m3::cache::{self as layout, MiniMaxCacheLayout};
use cuteafd_engine::prefix::TailCopy;

pub(super) const PAGE_ROWS: usize = layout::PAGE_ROWS as usize;
/// One K or V page: 64 rows of 4 KV heads x 128 E4M3 bytes.
pub(super) const KV_PAGE_BYTES: usize = PAGE_ROWS * 4 * 128;
pub(super) use crate::shared::prefix::INDEX_PAGE_BYTES;
const _: () = assert!(INDEX_PAGE_BYTES as u64 == layout::INDEX_PAGE_BYTES);
const KV_SPANS: &[(usize, usize)] = &[(0, KV_PAGE_BYTES / PAGE_ROWS)];

/// One layer's K, V and index planes, in the order the host tier stores them.
fn planes(key: CuteafdDeviceBuffer, value: CuteafdDeviceBuffer, index: CuteafdDeviceBuffer) -> [PagedPlane; 3] {
    let kv = |buffer| PagedPlane { buffer, page_bytes: KV_PAGE_BYTES, spans: KV_SPANS };
    [kv(key), kv(value), PagedPlane::index(index)]
}

/// A tail copy's `(from, to, rows)` inside a pool of `pages` (one GPU, so no owner).
fn tail_pages(copy: TailCopy, pages: usize) -> std::result::Result<(usize, usize, usize), MiniMaxError> {
    if copy.owner.is_some() || copy.rows > PAGE_ROWS || copy.from == copy.to
        || copy.from as usize >= pages || copy.to as usize >= pages {
        return Err(MiniMaxError::Invalid("cache tail copy"));
    }
    Ok((copy.from as usize, copy.to as usize, copy.rows))
}

pub(super) struct Cache<'a> {
    pub layers: Vec<LayerCache<'a>>,
    pub pages: usize,
    pub context: usize,
    pub sequences: usize,
    pub tail: Buffer<'a>,
    pub library: &'a NativeLibrary,
    pub device: i32,
}

impl<'a> Cache<'a> {
    pub fn new(library: &'a NativeLibrary, device: i32, layout: &MiniMaxCacheLayout,
        pages: usize, sequences: usize, admitted: u64) -> Result<Self> {
        if layout.kv_page_bytes != KV_PAGE_BYTES as u64 || layout.index_page_bytes != INDEX_PAGE_BYTES as u64
            || pages == 0 || pages > i32::MAX as usize || !(1..=64).contains(&sequences) {
            return Err(MiniMaxError::Invalid("native cache geometry").into());
        }
        let required = layout.page_bytes.checked_mul(pages as u64)
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

    /// Every layer's planes; page `p` of the prefix cache is page `p` of each.
    pub fn planes(&self) -> impl Iterator<Item = PagedPlane> + '_ {
        self.layers.iter().flat_map(|l| planes(l.key.raw(), l.value.raw(), l.index.raw()))
    }

    pub fn copy_rows(&self, copy: TailCopy) -> Result<()> {
        let (from, to, rows) = tail_pages(copy, self.pages)?;
        self.library.cuda_set_device(self.device)?;
        for plane in self.planes() {
            for (dst, src) in plane.tail(from, to, rows)? { self.library.copy_d2d(dst, src, src.bytes)?; }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimax_tail_copy_preserves_every_visible_byte_and_nothing_else() {
        let pages = 3;
        let sizes = [KV_PAGE_BYTES, KV_PAGE_BYTES, INDEX_PAGE_BYTES];
        let bases = [0x1000_0000usize, 0x2000_0000, 0x3000_0000];
        let buffer = |p: usize| CuteafdDeviceBuffer { ptr: bases[p] as *mut std::ffi::c_void,
            bytes: pages * sizes[p], device_id: 0, flags: 0 };
        // The rows a page holds: K/V rows of 4 heads x 128 bytes; index keys, then scales at 64 x 128.
        let visible = |p: usize, offset: usize, rows: usize| if p < 2 { offset < rows * 512 }
            else { offset < rows * 128 || (8192..8192 + rows * 4).contains(&offset) };
        for rows in [0, 1, 17, 63, 64] {
            let (from, to, n) = tail_pages(TailCopy { from: 1, to: 2, rows, owner: None }, pages).unwrap();
            for (p, plane) in planes(buffer(0), buffer(1), buffer(2)).iter().enumerate() {
                let size = sizes[p];
                let before: Vec<u8> = (0..pages * size).map(|i| ((i * 7 + i / size) % 251) as u8).collect();
                let mut after = before.clone();
                for (dst, src) in plane.tail(from, to, n).unwrap() {
                    let (d, s) = (dst.ptr as usize - bases[p], src.ptr as usize - bases[p]);
                    after.copy_within(s..s + src.bytes, d);
                }
                for (i, &byte) in after.iter().enumerate() {
                    let (page, offset) = (i / size, i % size);
                    let expected = if page == to && visible(p, offset, rows) { before[from * size + offset] } else { before[i] };
                    assert_eq!(byte, expected, "plane {p} byte {i} after a {rows}-row tail copy");
                }
            }
        }
        for bad in [TailCopy { from: 0, to: 1, rows: 65, owner: None }, TailCopy { from: 0, to: 3, rows: 1, owner: None },
            TailCopy { from: 1, to: 1, rows: 1, owner: None }, TailCopy { from: 0, to: 1, rows: 1, owner: Some(0) }] {
            assert!(tail_pages(bad, pages).is_err(), "{bad:?}");
        }
    }
}
