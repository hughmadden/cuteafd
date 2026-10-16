//! Exact dense Qwen frontiers: plain 64-row GQA pages plus FP32 GDN state.
use super::*;
use crate::shared::{prefix::view, qwen::MarkLayout};
use cuteafd_engine::prefix::{BoxError, FamilyLayout, MarkSlot, MarkStore, PrefixFamily, ReuseRule, TailCopy};
use cuteafd_hostcache::copy::DeviceRange;
use std::{cell::RefCell, rc::Rc};

#[derive(Clone)]
pub(crate) struct Placement {
    pub units: Vec<u32>,
    pub slot: usize,
    pub len: usize,
    pub state_len: usize,
}

impl crate::shared::qwen::scheduler::Placement for Placement {
    fn len(&self) -> usize { self.len }
    fn set_len(&mut self, len: usize) { self.len = len; }
    fn units(&self) -> &[u32] { &self.units }
    fn slot(&self) -> i32 { self.slot as i32 }
    fn spans(&self) -> &[cuteafd_core::MediaSpan] { &[] }
}

pub(crate) struct DensePrefix<'e, 'a> {
    engine: &'e RefCell<DenseEngine<'a>>,
    layout: FamilyLayout,
    mark: MarkLayout,
}

impl<'e, 'a> DensePrefix<'e, 'a> {
    pub fn new(engine: &'e RefCell<DenseEngine<'a>>) -> Result<Self> {
        let target = engine.borrow();
        let regions = Self::state(&target, 0);
        let mark = MarkLayout::new(&regions.iter().map(|region| (region.device_id, region.bytes)).collect::<Vec<_>>())?;
        ensure!(mark.bytes > 0 && !target.kv.iter().flatten().collect::<Vec<_>>().is_empty(),
            "dense prefix requires recurrent and full-attention layers");
        ensure!(target.prefix_arena.as_ref().map_or(0, |arena| arena.buffer.bytes)
            == target.prefix_marks * mark.bytes, "dense prefix marks differ from admission");
        let layout = FamilyLayout { page_rows: 64, pages: target.pool_tokens / 64,
            page_bytes: target.kv.iter().flatten().count() * 64 * 2 * target.cfg.kv_heads * target.cfg.head_dim * 2,
            mark_bytes: mark.bytes, draft_bytes: 0, rule: ReuseRule::EXACT,
            mark_store: MarkStore::Arena, page_owners: Default::default() };
        drop(target);
        Ok(Self { engine, layout, mark })
    }

    pub fn slots(&self) -> usize { self.engine.borrow().prefix_marks }

    pub fn snapshot_owners(&self) -> Vec<Rc<Allocation<'a>>> {
        let target = self.engine.borrow();
        target.kv.iter().flatten().cloned().chain(target.prefix_arena.iter().cloned()).collect()
    }

    fn state(target: &DenseEngine<'_>, slot: usize) -> Vec<CuteafdDeviceBuffer> {
        (0..target.homes.gdn_layers[0]).flat_map(|ordinal| target.bank.slot_regions(ordinal, slot)).collect()
    }

    fn mark_buffers(&self, slot: MarkSlot) -> Result<Vec<CuteafdDeviceBuffer>> {
        let target = self.engine.borrow();
        ensure!((slot.0 as usize) < target.prefix_marks, "dense prefix mark outside admitted arena");
        let arena = target.prefix_arena.as_ref().context("dense prefix arena disabled")?;
        self.mark.regions.iter().map(|&(_, offset, bytes)|
            view(arena.buffer, slot.0 as usize * self.mark.bytes + offset, bytes)).collect()
    }

    fn move_mark(&self, mark: MarkSlot, state: usize, capture: bool) -> Result<()> {
        let marks = self.mark_buffers(mark)?;
        let target = self.engine.borrow();
        ensure!(state < target.slots.len(), "dense prefix state outside admitted slots");
        for (live, saved) in Self::state(&target, state).into_iter().zip(marks) {
            let (dst, src) = if capture { (saved, live) } else { (live, saved) };
            // SAFETY: admitted arena and live bank outlive the shared prefix cache's drain.
            unsafe { target.stream.device.library.copy_d2d_async(dst, src, src.bytes, target.stream.raw)?; }
        }
        Ok(())
    }

    fn page_buffers(&self, page: u32, rows: usize) -> Result<Vec<CuteafdDeviceBuffer>> {
        ensure!((page as usize) < self.layout.pages && rows <= 64, "dense prefix page/rows outside admitted pool");
        let target = self.engine.borrow();
        let record = 2 * target.cfg.kv_heads * target.cfg.head_dim * 2;
        target.kv.iter().flatten().map(|pool| view(pool.buffer, page as usize * 64 * record, rows * record)).collect()
    }
}

impl PrefixFamily for DensePrefix<'_, '_> {
    type Placement = Placement;

    fn record_format(&self) -> &'static str { "qwen_dense_bf16_gqa_gdn_f32_v1" }
    fn layout(&self) -> FamilyLayout { self.layout }
    fn pages<'p>(&self, placement: &'p Placement) -> &'p [u32] { &placement.units }
    fn commit_point(&self, placement: &Placement) -> usize { placement.len.min(placement.state_len) }

    fn bind(&self, placement: &mut Placement) -> Result<(), BoxError> {
        let mut target = self.engine.borrow_mut();
        target.reset_slot(placement.slot)?;
        target.bind_slot(placement.slot, &placement.units)?;
        placement.state_len = 0;
        Ok(())
    }

    fn discard(&self, placement: &mut Placement) -> Result<(), BoxError> {
        self.engine.borrow_mut().reset_slot(placement.slot)?;
        placement.len = 0;
        placement.state_len = 0;
        Ok(())
    }

    fn capture(&self, slot: MarkSlot, placement: &Placement, len: usize) -> Result<(), BoxError> {
        let target = self.engine.borrow();
        if !(placement.slot < target.slots.len() && len == placement.len && len == placement.state_len
            && len == target.slots[placement.slot].position && placement.units == target.slots[placement.slot].pages) {
            return Err("dense prefix capture requires exact committed frontier".into());
        }
        if !target.healthy || target.verification.pending.is_some() {
            return Err("dense prefix capture during uncommitted verify".into());
        }
        drop(target);
        self.move_mark(slot, placement.slot, true)?;
        Ok(())
    }

    fn restore(&self, mark: Option<MarkSlot>, placement: &mut Placement, len: usize) -> Result<(), BoxError> {
        let mark = mark.context("dense prefix restores require exact GDN marks")?;
        if len == 0 || len > self.engine.borrow().cfg.max_context || len.div_ceil(64) > placement.units.len() {
            return Err("dense prefix restore outside assigned pages/context".into());
        }
        // Validate the mark before changing any live state.
        self.mark_buffers(mark)?;
        {
            let mut target = self.engine.borrow_mut();
            target.bind_slot(placement.slot, &placement.units)?;
            target.healthy = false;
        }
        self.move_mark(mark, placement.slot, false)?;
        let mut target = self.engine.borrow_mut();
        target.stream.drain()?;
        target.slots[placement.slot].position = len;
        target.position = target.slots[0].position;
        target.healthy = true;
        placement.len = len;
        placement.state_len = len;
        Ok(())
    }

    fn copy_rows(&self, copy: TailCopy) -> Result<(), BoxError> {
        let source = self.page_buffers(copy.from, copy.rows)?;
        let destination = self.page_buffers(copy.to, copy.rows)?;
        let target = self.engine.borrow();
        for (dst, src) in destination.into_iter().zip(source) {
            // SAFETY: shared pool owns both page views until this family drains.
            unsafe { target.stream.device.library.copy_d2d_async(dst, src, src.bytes, target.stream.raw)?; }
        }
        Ok(())
    }

    fn drain(&self) -> Result<(), BoxError> { Ok(self.engine.borrow().stream.drain()?) }
    fn page_segments(&self, page: u32) -> Vec<DeviceRange> {
        self.page_buffers(page, 64).unwrap_or_default().into_iter()
            .map(|buffer| DeviceRange { addr: buffer.ptr as u64, bytes: buffer.bytes }).collect()
    }
    fn mark_segments(&self, slot: MarkSlot) -> Vec<DeviceRange> {
        self.mark_buffers(slot).unwrap_or_default().into_iter()
            .map(|buffer| DeviceRange { addr: buffer.ptr as u64, bytes: buffer.bytes }).collect()
    }
}
