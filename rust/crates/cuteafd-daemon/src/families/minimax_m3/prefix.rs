use super::{cache::{Cache, PAGE_ROWS}, MiniMaxError};
use cuteafd_core::MediaSpan;
use cuteafd_engine::{media::RequestMedia, prefix::{BoxError, FamilyLayout, MarkSlot, MarkStore,
    PageOwners, PrefixFamily, ReuseRule, TailCopy}};
use cuteafd_hostcache::copy::DeviceRange;

#[derive(Clone)]
pub(super) struct Placement {
    pub pages: Vec<u32>,
    pub sequence: usize,
    pub committed: usize,
    pub media: RequestMedia,
}
impl crate::shared::qwen::scheduler::Placement for Placement {
    fn len(&self) -> usize { self.committed }
    fn set_len(&mut self, len: usize) { self.committed = len; }
    fn units(&self) -> &[u32] { &self.pages }
    fn slot(&self) -> i32 { self.sequence as i32 }
    fn spans(&self) -> &[MediaSpan] { self.media.spans() }
}

pub(super) struct MiniMaxPrefix<'e, 'a> { pub cache: &'e Cache<'a> }
impl PrefixFamily for MiniMaxPrefix<'_, '_> {
    type Placement = Placement;
    fn layout(&self) -> FamilyLayout {
        FamilyLayout { page_rows: PAGE_ROWS, pages: self.cache.pages,
            page_bytes: self.cache.planes().map(|plane| plane.page_bytes).sum(),
            mark_bytes: 0, draft_bytes: 0, rule: ReuseRule::EXACT,
            mark_store: MarkStore::Arena, page_owners: PageOwners::Uniform }
    }
    fn record_format(&self) -> &'static str { "minimax_msa_fp8_p64_index8448_v1" }
    fn pages<'p>(&self, placement: &'p Placement) -> &'p [u32] { &placement.pages }
    fn commit_point(&self, placement: &Placement) -> usize { placement.committed }
    fn capture(&self, _: MarkSlot, _: &Placement, _: usize) -> Result<(), BoxError> {
        Err(MiniMaxError::Invalid("positional mark: cache consists only of pages").into())
    }
    fn restore(&self, mark: Option<MarkSlot>, placement: &mut Placement, len: usize) -> Result<(), BoxError> {
        if mark.is_some() || len == 0 || len > self.cache.context || len.div_ceil(PAGE_ROWS) > placement.pages.len()
            || placement.pages.iter().any(|&p| p as usize >= self.cache.pages) {
            return Err(MiniMaxError::Invalid("prefix frontier").into());
        }
        placement.committed = len;
        Ok(())
    }
    fn copy_rows(&self, copy: TailCopy) -> Result<(), BoxError> { self.cache.copy_rows(copy).map_err(Into::into) }
    fn drain(&self) -> Result<(), BoxError> { Ok(()) }
    fn page_segments(&self, page: u32) -> Vec<DeviceRange> {
        if page as usize >= self.cache.pages { return Vec::new(); }
        self.cache.planes().map(|plane| plane.page(page as usize)).collect()
    }
    fn mark_segments(&self, _: MarkSlot) -> Vec<DeviceRange> { Vec::new() }
    fn host_tail(&self) -> Vec<DeviceRange> {
        vec![DeviceRange { addr: self.cache.tail.raw().ptr as u64, bytes: 1 }]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::cache::{INDEX_PAGE_BYTES, KV_PAGE_BYTES};
    use cuteafd_engine::prefix::{After, PrefixCache, PrefixConfig, SnapshotKind};
    use cuteafd_hostcache::copy::StubCopyEngine;
    use cuteafd_loader::families::minimax_m3::cache::MiniMaxCacheLayout;

    #[test]
    #[ignore = "CPU-only native fixture; requires an explicit build slot and fixture directory"]
    fn minimax_prefix_fixture_shares_pages_and_copies_exact_tail() -> Result<(), BoxError> {
        let fixture = cuteafd_ffi::native_library_lifetime_fixture::Fixture::build()?;
        let library = fixture.load()?;
        fixture.immediate_copies(&library)?;
        let layout = MiniMaxCacheLayout { layers: 2, context: 256, kv_page_bytes: KV_PAGE_BYTES as u64,
            index_page_bytes: INDEX_PAGE_BYTES as u64, page_bytes: 2 * (2 * KV_PAGE_BYTES + INDEX_PAGE_BYTES) as u64 };
        let cache = Cache::new(&library, 0, &layout, 16, 2, layout.page_bytes * 16 + 256)?;
        let family = MiniMaxPrefix { cache: &cache };
        let mut shared = PrefixCache::<StubCopyEngine>::new(family.layout(), PrefixConfig {
            entries: 4, mark_slots: 0, keep_logits: false, min_tokens: 1 }, None)?;
        let placement = |pages| Placement { pages, sequence: 0, committed: 0,
            media: RequestMedia::new(Vec::new(), 6144, 0).unwrap() };
        for len in [1usize, 63, 64, 65, 127, 128, 129] {
            shared.clear(&family)?;
            let tokens: Vec<_> = (0..len as u32).collect();
            let mut source = shared.admit(&family, &tokens, len + 1, false, placement)?.placement;
            source.committed = len;
            for layer in &cache.layers {
                for (plane, bytes) in [(&layer.key, KV_PAGE_BYTES), (&layer.value, KV_PAGE_BYTES), (&layer.index, INDEX_PAGE_BYTES)] {
                    for (logical, &page) in source.pages.iter().enumerate() {
                        let data: Vec<_> = (0..bytes).map(|i| ((i * 7 + logical * 13) % 251) as u8).collect();
                        library.copy_h2d(plane.view(page as usize * bytes, bytes)?.raw(), &data)?;
                    }
                }
            }
            assert!(shared.capture(&family, SnapshotKind::Prompt, &tokens, &source, After::default())?);
            let mut longer = tokens.clone(); longer.push(99);
            let restored = shared.admit(&family, &longer, len + 1, false, placement)?;
            assert_eq!(restored.resume, len);
            assert_eq!(restored.placement.committed, len);
            for logical in 0..len.div_ceil(64) {
                let (from, to) = (source.pages[logical], restored.placement.pages[logical]);
                let rows = (len - logical * 64).min(64);
                if rows == 64 { assert_eq!(from, to, "complete pages share storage"); }
                else { assert_ne!(from, to, "append gets a private partial page"); }
                for layer in &cache.layers {
                    for (plane, page_bytes, spans) in [
                        (&layer.key, KV_PAGE_BYTES, vec![(0, rows * 512)]),
                        (&layer.value, KV_PAGE_BYTES, vec![(0, rows * 512)]),
                        (&layer.index, INDEX_PAGE_BYTES, vec![(0, rows * 128), (8192, rows * 4)]),
                    ] {
                        for (offset, bytes) in spans {
                            let (mut a, mut b) = (vec![0; bytes], vec![0; bytes]);
                            library.copy_d2h(&mut a, plane.view(from as usize * page_bytes + offset, bytes)?.raw())?;
                            library.copy_d2h(&mut b, plane.view(to as usize * page_bytes + offset, bytes)?.raw())?;
                            assert_eq!(a, b, "restored KV and index scales match cold state");
                        }
                    }
                }
            }
            shared.release(&family, &source.pages)?;
            shared.release(&family, &restored.placement.pages)?;
        }
        shared.clear(&family)?;
        Ok(())
    }
}
