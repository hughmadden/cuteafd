//! Dense-summary Step-5 storage adapter for the shared exact prefix pool.
//! The executor still owns admission, page-table publication and committed lengths.
use super::cache::CacheRank;
use anyhow::{ensure, Result};
use cuteafd_engine::prefix::{BoxError, FamilyLayout, Mark, MarkSlot, MarkStore, PageOwners,
    PrefixFamily, RestoreContext, RestorePlan, ReuseRule, TailCopy};
use cuteafd_hostcache::copy::DeviceRange;
use std::cell::RefCell;

#[derive(Debug, Clone)]
pub(super) struct Placement {
    pub pages: Vec<u32>,
    pub sequence: u64,
    pub committed: usize,
}

/// Carried by both device and host entries. Actual stage lengths are in the mark.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct Frontier {
    committed: usize,
}

pub(super) struct Step5Prefix<'e, 'a> {
    ranks: Vec<&'e RefCell<CacheRank<'a>>>,
    layout: FamilyLayout,
    sequences: u64,
    context: u64,
}

fn range(buffer: cuteafd_ffi::CuteafdDeviceBuffer) -> DeviceRange {
    DeviceRange { addr: buffer.ptr as u64, bytes: buffer.bytes }
}

impl<'e, 'a> Step5Prefix<'e, 'a> {
    pub fn new(ranks: Vec<&'e RefCell<CacheRank<'a>>>) -> Result<Self> {
        ensure!(!ranks.is_empty(), "Step-5 prefix requires cache ranks");
        let first = ranks[0].borrow();
        let shape = first.layout().clone();
        ensure!(ranks.len() == shape.ranks.len(), "Step-5 prefix rank count");
        let sequences = first.sequences();
        let slots = first.slots();
        let pages = first.page_count();
        let context = shape.context;
        drop(first);
        let mut page_bytes = 0usize;
        let mut mark_bytes = 0usize;
        for (rank, owner) in ranks.iter().enumerate() {
            let cache = owner.borrow();
            ensure!(cache.rank() == rank && cache.layout() == &shape,
                "Step-5 prefix ranks disagree on layout");
            ensure!(cache.sequences() == sequences && cache.slots() == slots && cache.page_count() == pages,
                "Step-5 prefix ranks disagree on capacity");
            let geometry = &shape.geometry.ranks[rank];
            page_bytes = page_bytes.checked_add(usize::try_from(geometry.persistent_unit_bytes)?)
                .ok_or_else(|| anyhow::anyhow!("Step-5 page size overflow"))?;
            mark_bytes = mark_bytes.checked_add(usize::try_from(shape.ranks[rank].bytes)?)
                .ok_or_else(|| anyhow::anyhow!("Step-5 mark size overflow"))?;
        }
        ensure!(pages <= u32::MAX as u64 && slots <= u32::MAX as u64,
            "Step-5 shared pool index overflow");
        let layout = FamilyLayout { page_rows: 64, pages: usize::try_from(pages)?, page_bytes,
            mark_bytes, draft_bytes: 0, rule: ReuseRule::EXACT,
            mark_store: MarkStore::Arena, page_owners: PageOwners::Uniform };
        Ok(Self { ranks, layout, sequences, context })
    }

    fn validate(&self, placement: &Placement, len: usize) -> Result<()> {
        ensure!(placement.sequence < self.sequences && len as u64 <= self.context,
            "Step-5 placement outside admitted capacity");
        ensure!(len.div_ceil(64) <= placement.pages.len(), "Step-5 placement lacks CSA pages");
        ensure!(placement.pages.iter().all(|p| (*p as usize) < self.layout.pages),
            "Step-5 placement page outside pool");
        Ok(())
    }

    fn restore_mark(&self, slot: MarkSlot, placement: &mut Placement, len: usize) -> Result<()> {
        self.validate(placement, len)?;
        for owner in &self.ranks {
            owner.borrow().restore_retained(placement.sequence, u64::from(slot.0), len as u64)?;
        }
        placement.committed = len;
        Ok(())
    }
}

impl PrefixFamily<Frontier> for Step5Prefix<'_, '_> {
    type Placement = Placement;

    fn layout(&self) -> FamilyLayout { self.layout }
    fn record_format(&self) -> &'static str { "step5_bf16_csa64_dense_e4m3_marks_v1" }
    fn pages<'p>(&self, placement: &'p Placement) -> &'p [u32] { &placement.pages }
    fn commit_point(&self, placement: &Placement) -> usize { placement.committed }

    fn bind(&self, placement: &mut Placement) -> Result<(), BoxError> {
        self.validate(placement, placement.committed)?;
        Ok(())
    }

    fn capture_meta(&self, placement: &Placement, len: usize) -> Result<Frontier, BoxError> {
        self.validate(placement, len)?;
        if len != placement.committed {
            return Err("Step-5 capture must be at the committed frontier".into());
        }
        Ok(Frontier { committed: len })
    }

    fn capture(&self, slot: MarkSlot, placement: &Placement, len: usize) -> Result<(), BoxError> {
        self.capture_meta(placement, len)?;
        for owner in &self.ranks {
            owner.borrow_mut().capture(placement.sequence, u64::from(slot.0), len as u64)?;
        }
        Ok(())
    }

    fn restore(&self, mark: Option<MarkSlot>, placement: &mut Placement, len: usize) -> Result<(), BoxError> {
        let slot = mark.ok_or("Step-5 exact restore requires a positional mark")?;
        self.validate(placement, len)?;
        // Direct callers use local capture bookkeeping. Host-tier restores use
        // restore_with and the entry's self-describing frontier instead.
        for owner in &self.ranks {
            owner.borrow_mut().restore(placement.sequence, u64::from(slot.0), len as u64)?;
        }
        placement.committed = len;
        Ok(())
    }

    fn restore_with(&self, mark: Option<&Mark>, saved: Option<&Frontier>, placement: &mut Placement,
        plan: &RestorePlan, _context: &RestoreContext<'_>) -> Result<(), BoxError> {
        if !plan.exact() || plan.resume() != plan.snapshot_end {
            return Err("Step-5 does not reconstruct approximate prefix state".into());
        }
        if !saved.is_some_and(|f| f.committed == plan.resume()) {
            return Err("Step-5 retained mark frontier mismatch".into());
        }
        let Some(Mark::Slot(slot)) = mark else {
            return Err("Step-5 exact restore requires an arena mark".into());
        };
        self.restore_mark(*slot, placement, plan.resume())?;
        Ok(())
    }

    fn copy_rows(&self, copy: TailCopy) -> Result<(), BoxError> {
        if copy.owner.is_some() { return Err("Step-5 uses uniform head-split pages".into()); }
        for owner in &self.ranks {
            owner.borrow().copy_page_rows(copy.from, copy.to, copy.rows)?;
        }
        Ok(())
    }

    fn drain(&self) -> Result<(), BoxError> {
        for owner in &self.ranks { owner.borrow().stream.drain()?; }
        Ok(())
    }

    fn page_segments(&self, page: u32) -> Vec<DeviceRange> {
        self.ranks.iter().flat_map(|owner| owner.borrow().page_views(page)
            .expect("shared prefix pool passed an invalid Step-5 page").into_iter().map(range)).collect()
    }

    fn mark_segments(&self, slot: MarkSlot) -> Vec<DeviceRange> {
        self.ranks.iter().map(|owner| range(owner.borrow().mark_view(slot.0)
            .expect("shared prefix arena passed an invalid Step-5 slot"))).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::memory::device::cache_test_device;
    use cuteafd_engine::prefix::{After, PrefixCache, PrefixConfig, SnapshotKind};
    use cuteafd_hostcache::copy::StubCopyEngine;

    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIB and CUDA"]
    fn native_shared_prefix_restores_marks_and_csa_partial_pages() -> std::result::Result<(), BoxError> {
        // SAFETY: the selected library implements the native CUDA ABI.
        let library = unsafe { cuteafd_ffi::NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        let device = cache_test_device(&library)?;
        let shape = super::super::cache::tests::layout_at(66);
        let owner = RefCell::new(CacheRank::new(device, shape.clone(), 0, 2, 2, 8, usize::MAX)?);
        let family = Step5Prefix::new(vec![&owner])?;
        let config = PrefixConfig { entries: 2, mark_slots: 2, keep_logits: false, min_tokens: 1 };
        let mut shared = PrefixCache::<StubCopyEngine, Frontier>::new(family.layout(), config, None)?;
        let tokens: Vec<u32> = (0..65).collect();
        let mut source = shared.admit(&family, &tokens, 65, false,
            |pages| Placement { pages, sequence: 0, committed: 0 })?.placement;
        source.committed = 65;
        assert!(family.capture(MarkSlot(0), &source, 64).is_err());
        assert!(family.restore(None, &mut source, 65).is_err());
        let first = source.pages[0];
        let tail = source.pages[1];
        let set = |buffer: cuteafd_ffi::CuteafdDeviceBuffer, byte: u8| -> Result<()> {
            device.run(|| library.copy_h2d(buffer, &vec![byte; buffer.bytes]))
        };
        for (i, region) in shape.ranks[0].regions.iter().enumerate() {
            let (key, value) = owner.borrow().csa_planes(region.layer)?;
            assert_eq!(key.bytes, 8 * 64 * 4 * 192 * 2);
            assert_eq!(value.bytes, key.bytes);
            assert_ne!(key.ptr, value.ptr);
            assert_eq!(i, shape.ranks[0].regions.iter().position(|r| r.layer == region.layer).unwrap());
        }
        for page in 0..8 {
            for plane in owner.borrow().page_views(page)? { set(plane, 0xa5)?; }
        }
        for rows in [0, 1, 7, 8, 63, 64] {
            for plane in owner.borrow().page_views(7)? { set(plane, 0xa5)?; }
            for plane in owner.borrow().page_views(first)? { set(plane, 0x19)?; }
            family.copy_rows(TailCopy { from: first, to: 7, rows, owner: None })?;
            for plane in owner.borrow().page_views(7)? {
                let mut bytes = vec![0; plane.bytes];
                device.run(|| library.copy_d2h(&mut bytes, plane))?;
                let retained = rows * plane.bytes / 64;
                assert!(bytes[..retained].iter().all(|b| *b == 0x19));
                assert!(bytes[retained..].iter().all(|b| *b == 0xa5));
            }
        }
        for plane in owner.borrow().page_views(tail)? { set(plane, 0x29)?; }
        let spans = shape.snapshot_spans(0, 65)?;
        let components = |span: &cuteafd_loader::serving_capacity::step5::ByteSpan| {
            let state = &shape.ranks[0];
            std::iter::once(state.length).chain(state.mtp_lengths.iter().copied()).chain(state.mtp_hidden)
                .chain(state.sliding.iter().flat_map(|r| [r.key, r.value]))
                .chain(state.regions.iter().flat_map(|r| [Some(r.pending_key), Some(r.pending_z), r.summaries].into_iter().flatten()))
                .find(|s| span.offset >= s.offset && span.offset + span.bytes <= s.offset + s.bytes).unwrap()
        };
        for span in &spans {
            let full = components(span);
            set(owner.borrow().component(full)?, 0x39)?;
        }
        assert!(shared.capture(&family, SnapshotKind::Prompt, &tokens, &source, After::default())?);
        // Simulate host-tier promotion into a different arena slot: its bytes
        // and entry metadata are valid, but no local capture ever used slot 1.
        let mark = owner.borrow().mark_view(0)?;
        let mut saved = vec![0; mark.bytes];
        device.run(|| library.copy_d2h(&mut saved, mark))?;
        device.run(|| library.copy_h2d(owner.borrow().mark_view(1)?, &saved))?;
        let plan = RestorePlan { snapshot_end: 65, target_end: 66,
            lag: cuteafd_engine::prefix::LaggedState { source_end: 65, replay_start: 65 },
            fidelity: cuteafd_engine::prefix::RestoreFidelity::Exact };
        let context = RestoreContext { native_tokens: &tokens, media: &[] };
        let mut promoted = source.clone();
        promoted.sequence = 1;
        assert!(family.restore(Some(MarkSlot(1)), &mut promoted, 65).is_err());
        assert!(family.restore_with(Some(&Mark::Slot(MarkSlot(1))), Some(&Frontier { committed: 64 }),
            &mut promoted, &plan, &context).is_err());
        family.restore_with(Some(&Mark::Slot(MarkSlot(1))), Some(&Frontier { committed: 65 }),
            &mut promoted, &plan, &context)?;
        // Only the source's partial page is mutable; the full page remains shared.
        for plane in owner.borrow().page_views(tail)? { set(plane, 0x49)?; }
        for span in &spans {
            let full = components(span);
            set(owner.borrow().component(full)?, 0xa5)?;
        }
        let mut continuation = tokens.clone();
        continuation.push(66);
        let resumed = shared.admit(&family, &continuation, 66, false,
            |pages| Placement { pages, sequence: 1, committed: 0 })?;
        assert_eq!(resumed.resume, 65);
        assert_eq!(resumed.placement.committed, 65);
        assert_eq!(resumed.placement.pages[0], first);
        assert_ne!(resumed.placement.pages[1], tail);
        for plane in owner.borrow().page_views(resumed.placement.pages[1])? {
            let mut bytes = vec![0; plane.bytes];
            device.run(|| library.copy_d2h(&mut bytes, plane))?;
            assert!(bytes[..plane.bytes / 64].iter().all(|b| *b == 0x29));
        }
        for span in &spans {
            let full = components(span);
            let plane = owner.borrow().component(full)?;
            let mut bytes = vec![0; plane.bytes];
            device.run(|| library.copy_d2h(&mut bytes, plane))?;
            let stride = full.bytes as usize;
            assert!(bytes[..stride].iter().all(|b| *b == 0xa5));
            let offset = (span.offset - full.offset) as usize;
            assert!(bytes[stride + offset..stride + offset + span.bytes as usize].iter().all(|b| *b == 0x39));
        }
        assert!(owner.borrow().page_views(8).is_err());
        assert!(owner.borrow().copy_page_rows(0, 1, 65).is_err());
        shared.release(&family, &source.pages)?;
        shared.release(&family, &resumed.placement.pages)?;
        Ok(())
    }
}
