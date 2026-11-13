//! DeepSeek V4.1 as a prefix-cache family (`cuteafd_engine::prefix`).
//!
//! **Pages.** One engine page is a 512-token unit: one page (256 source rows) of each ratio-two
//! source (layers 2, 8, 14) and two pages of the ratio-one source (layer 20), 5 x 91,136 B =
//! 455,680 B (~890 B/token). A request's units are allocated for its whole declared lifetime at
//! admission (V4.1 admits by declared lifetime) and bound to every source's page table, so the
//! decode path never allocates; a restore shares the snapshot's full units and copies its
//! partial tail unit once (one eager copy per restore and per capture, PLAN decision 4).
//!
//! **Mark.** Arena slots hold the positional state: every window's ring rows and each source's
//! odd-frontier carry (`BackboneMark::BYTES`, 2,720,064 B, on the prefix-copy device) and, with
//! dSpark, the three drafter rings (`DRAFT_MARK_BYTES`, 202,752 B, on the drafter's
//! device): one combined mark per snapshot, so a restored request drafts warm (MiMo
//! precedent). [`V41Meta`] describes it as plain data: window spans, the frontier, the Engram
//! lookback and the drafter's frontier.
//!
//! **Restores are exact** (`ReuseRule::EXACT`, AGENTS "Exact prefix-cache restores"): only a
//! retained frontier that is a prefix of the prompt is selected, and its mark restores
//! byte-exactly. An exact ancestor followed by a suffix of at least 128 tokens resumes as an
//! encoder continuation (encoder rings and carry restored, decoder rings rebuilt by the suffix's
//! final replay, which covers the whole decoder window): still the state a cold prefill reaches.
//!
//! **Copy stream.** Captures, restores and tail copies run on the backbone's private prefix
//! stream ([`BackboneCache::prefix_drain`]), not on `shared::prefix::PrefixCopies`. That is safe
//! by ownership rather than by a reader fence: a capture reads only a committed request's
//! private ring/carry slots and its committed (immutable) source rows, a restore and a tail copy
//! write only a fresh placement's private slots and freshly forked pages that no kernel reads
//! until the restore drains, and the engine drains before it releases or stores anything
//! (`tests::prefix_stream_reads_only_committed_private_state` checks the argument's source;
//! moving onto the shared copy stream is #1882).
pub(crate) mod points;

use super::*;
use super::scores::RetainedScores;
use crate::families::deepseek_v41::v41_backbone_cache::{BackboneMark, CacheLease, UNIT_BYTES, UNIT_TOKENS};
use crate::families::deepseek_v41::v41_requests::RequestMark;
use crate::shared::memory::Tag;
use crate::shared::memory::device::{Allocation, Device};
use crate::shared::prefix::{cuda_copy::range, CudaCopyEngine};
use cuteafd_engine::prefix::{
    After, BoxError, CaptureTicket, Captured, Eviction, FamilyLayout, Mark, MarkSlot, MarkStore,
    PrefixCache, PrefixConfig, PrefixFamily, RestoreCandidate, RestoreContext, RestoreFidelity, RestorePlan, ReuseRule, TailCopy,
};
use cuteafd_hostcache::copy::DeviceRange;
use serde::{Deserialize, Serialize};
use std::cell::{Cell, RefCell};
pub(crate) use cuteafd_engine::media::MediaSpan;
/// A prompt's media-keyed token copy for the radix and its image spans (engine `MediaKeys`).
pub(crate) type ImageKeys = cuteafd_engine::media::MediaKeys;
pub(crate) use cuteafd_engine::prefix::SnapshotKind;

/// A V4.1 snapshot's metadata: what its mark's bytes mean. Plain, serializable data.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct V41Meta {
    pub request: RequestMark,
    /// The drafter rings' frontier, when the mark carries them.
    pub draft_end: Option<u64>,
}

/// One admitted request's place in the cache: its units and its lease.
pub(crate) struct V41Placement {
    pub units: Vec<u32>,
    pub lease: CacheLease,
    /// Draft request identity when dSpark runs.
    pub draft: Option<u64>,
    /// Tokens the placement holds committed state for (the commit point).
    pub len: usize,
    /// The restore plan that produced this placement, if any.
    pub plan: Option<RestorePlan>,
}

/// The family's device side, borrowing the request bank and the drafter for one call.
pub(crate) struct V41Prefix<'r, 'a, 'w, C: DraftChain<'a>> {
    requests: RefCell<&'r mut Requests<'a>>,
    draft: RefCell<Option<&'r mut DraftRuntime<'w, 'a, C>>>,
    arena: &'r Arenas<'a>,
    units: usize,
    /// Capture lane of the next queued capture (`queue_capture`'s ticket is the lane).
    lane: Cell<usize>,
}

/// The mark arenas, allocated before serving: target marks on the backbone's prefix-copy
/// device, draft marks on the drafter's.
pub(crate) struct Arenas<'a> {
    target: Option<std::rc::Rc<Allocation<'a>>>,
    draft: Option<std::rc::Rc<Allocation<'a>>>,
    slots: usize,
}

impl<'a> Arenas<'a> {
    pub fn new(target: Device<'a>, draft: Option<Device<'a>>, slots: usize) -> Result<Self> {
        let _scope = cuteafd_ffi::memory_ledger::scope("prefix/snapshot");
        let target = (slots > 0).then(|| Allocation::new(target, slots * BackboneMark::BYTES, Tag::prefix("v41/prefix-snapshots")).map(std::rc::Rc::new))
            .transpose()?;
        let draft = match draft {
            Some(device) if slots > 0 => Some(std::rc::Rc::new(Allocation::new(device, slots * super::speculative::DRAFT_MARK_BYTES, Tag::prefix("v41/prefix-snapshots"))?)),
            _ => None,
        };
        Ok(Self { target, draft, slots })
    }
    /// Draft marks allocated after admission placed the drafter (two-RTX
    /// dSpark loads after the solve, so its marks are admitted, not sampled).
    pub fn with_draft(mut self, draft: Device<'a>) -> Result<Self> {
        ensure!(self.draft.is_none(), "draft mark arena already allocated");
        if self.slots > 0 {
            let _scope = cuteafd_ffi::memory_ledger::scope("prefix/snapshot");
            self.draft = Some(std::rc::Rc::new(Allocation::new(draft, self.slots * super::speculative::DRAFT_MARK_BYTES, Tag::prefix("v41/prefix-snapshots"))?));
        }
        Ok(self)
    }
    pub fn draft_mark_bytes(slots: usize) -> usize { slots * super::speculative::DRAFT_MARK_BYTES }
    pub fn device_bytes(slots: usize, draft: bool) -> usize {
        slots * (BackboneMark::BYTES + if draft { super::speculative::DRAFT_MARK_BYTES } else { 0 })
    }
    fn target(&self, slot: MarkSlot) -> Result<cuteafd_ffi::CuteafdDeviceBuffer> {
        let arena = self.target.as_ref().context("no prefix mark arena")?;
        ensure!((slot.0 as usize) < self.slots, "mark slot {} of {}", slot.0, self.slots);
        Ok(view(arena.buffer, slot.0 as usize * BackboneMark::BYTES, BackboneMark::BYTES))
    }
    fn draft(&self, slot: MarkSlot) -> Result<Option<cuteafd_ffi::CuteafdDeviceBuffer>> {
        Ok(match &self.draft {
            Some(arena) => {
                ensure!((slot.0 as usize) < self.slots, "mark slot {} of {}", slot.0, self.slots);
                Some(view(arena.buffer, slot.0 as usize * super::speculative::DRAFT_MARK_BYTES, super::speculative::DRAFT_MARK_BYTES))
            }
            None => None,
        })
    }
    pub fn slots(&self) -> usize { self.slots }
    /// Mark slots for the launch: the shared `MarkRule` (`--prefix-cache-mark-mib`, host tier)
    /// over one complete mark, as the planner reserves them.
    pub fn slots_for(args: &crate::cli::NativeServeArgs) -> usize {
        args.prefix.mark_rule(args.concurrency as usize).slots(mark_bytes(args.dspark))
    }
    pub fn owners(&self) -> Vec<std::rc::Rc<Allocation<'a>>> {
        self.target.iter().chain(&self.draft).cloned().collect()
    }
}

fn view(buffer: cuteafd_ffi::CuteafdDeviceBuffer, offset: usize, bytes: usize) -> cuteafd_ffi::CuteafdDeviceBuffer {
    debug_assert!(offset + bytes <= buffer.bytes);
    cuteafd_ffi::CuteafdDeviceBuffer { ptr: unsafe { buffer.ptr.cast::<u8>().add(offset).cast() }, bytes, ..buffer }
}

fn boxed(error: anyhow::Error) -> BoxError {
    format!("{error:#}").into()
}

/// One snapshot's mark bytes: the backbone mark plus, with dSpark, the drafter rings.
pub(crate) fn mark_bytes(draft: bool) -> usize {
    BackboneMark::BYTES + if draft { super::speculative::DRAFT_MARK_BYTES } else { 0 }
}

/// V4.1's geometry for the prefix engine over `units` 512-token units: exact frontiers only.
pub(crate) fn layout(units: usize, draft: bool) -> FamilyLayout {
    FamilyLayout {
        page_rows: UNIT_TOKENS,
        pages: units,
        page_bytes: UNIT_BYTES,
        mark_bytes: mark_bytes(draft),
        draft_bytes: 0,
        rule: ReuseRule::EXACT,
        mark_store: MarkStore::Arena,
        page_owners: cuteafd_engine::prefix::PageOwners::Uniform,
    }
}

/// The restore route and its accounting share the decoder-window admission rule.
fn plan(hit: RestoreCandidate<'_>) -> RestorePlan {
    let mut plan = RestorePlan::under(ReuseRule::EXACT, hit);
    if plan.exact() && plan.target_end.saturating_sub(plan.snapshot_end) >= 128 {
        plan.fidelity = RestoreFidelity::ExactEncoderContinuation;
    }
    plan
}

impl<'r, 'a, 'w, C: DraftChain<'a>> V41Prefix<'r, 'a, 'w, C> {
    pub fn new(requests: &'r mut Requests<'a>, draft: Option<&'r mut DraftRuntime<'w, 'a, C>>, arena: &'r Arenas<'a>)
        -> Self {
        let units = requests.cache().unit_capacity();
        Self { requests: RefCell::new(requests), draft: RefCell::new(draft), arena, units, lane: Cell::new(0) }
    }
    /// Capture lane for the next queued capture.
    pub fn on_lane(&self, lane: usize) -> &Self {
        self.lane.set(lane);
        self
    }
    fn bind_units(&self, placement: &V41Placement, rows: usize) -> Result<()> {
        let mut requests = self.requests.borrow_mut();
        let pages = requests.cache().unit_pages(&placement.units);
        let rows = requests.cache().source_rows(rows);
        let pages = [&pages[0][..], &pages[1][..], &pages[2][..], &pages[3][..]];
        requests.bind_units(placement.lease, placement.units.clone(), pages, rows)
    }
}

impl<'a, C: DraftChain<'a>> PrefixFamily<V41Meta> for V41Prefix<'_, 'a, '_, C> {
    type Placement = V41Placement;

    fn layout(&self) -> FamilyLayout {
        layout(self.units, self.arena.draft.is_some())
    }
    fn eviction(&self) -> Eviction {
        Eviction::FreesPages
    }
    fn plan_restore(&self, hit: RestoreCandidate<'_>) -> RestorePlan { plan(hit) }
    fn refused_partial_rule(&self) -> Option<ReuseRule> { Some(ReuseRule::V41) }
    fn capture_meta(&self, placement: &V41Placement, len: usize) -> Result<V41Meta, BoxError> {
        let requests = self.requests.borrow();
        let backbone = requests.cache().mark_meta(placement.lease).map_err(boxed)?;
        if backbone.end as usize != len {
            return Err(format!("V4.1 captures its committed frontier only ({} != {len})", backbone.end).into());
        }
        let (position, recent) = requests.engram_lookback(placement.lease).map_err(boxed)?;
        let draft_end = match (placement.draft, self.draft.borrow().as_ref()) {
            (Some(id), Some(draft)) => Some(draft.committed_end(id).map_err(boxed)?),
            _ => None,
        };
        Ok(V41Meta { request: RequestMark { backbone, engram_position: position, engram_recent: recent }, draft_end })
    }
    fn bind(&self, placement: &mut V41Placement) -> Result<(), BoxError> {
        self.bind_units(placement, 0).map_err(boxed)
    }
    fn discard(&self, placement: &mut V41Placement) -> Result<(), BoxError> {
        // The engine releases the placement's units itself: take them off the request so its
        // release (here or by the scheduler) does not unreference them a second time. A
        // failed restore may already have released the lease; then they were queued for
        // reclaim instead, and are taken back from there.
        let mut requests = self.requests.borrow_mut();
        if requests.take_units(placement.lease).is_empty() {
            requests.forget_released_units(&placement.units);
        }
        Ok(())
    }
    fn restore_with(&self, mark: Option<&Mark>, saved: Option<&V41Meta>, placement: &mut V41Placement,
        plan: &RestorePlan, context: &RestoreContext<'_>) -> Result<(), BoxError> {
        if !plan.exact() {
            return Err("V4.1 restores exact snapshot frontiers only".into());
        }
        let (Some(Mark::Slot(slot)), Some(saved)) = (mark, saved) else {
            return Err("an exact V4.1 restore needs its arena mark and metadata".into());
        };
        let _ = context;
        // The forked rows through the snapshot's frontier are initialized in the bound pages.
        self.bind_units(placement, plan.lag.source_end).map_err(boxed)?;
        let mut requests = self.requests.borrow_mut();
        // Tail copies of forked units were enqueued on the prefix stream; restores below read
        // the pages only through later kernels on the same stream or after its drain.
        let continuation = (plan.fidelity == RestoreFidelity::ExactEncoderContinuation).then_some(plan.target_end as u64);
        let target = self.arena.target(*slot).map_err(boxed)?;
        requests.restore_mark(placement.lease, &saved.request, target, continuation).map_err(boxed)?;
        if continuation.is_none() {
            let mut draft = self.draft.borrow_mut();
            match (placement.draft, draft.as_deref_mut(), saved.draft_end, self.arena.draft(*slot).map_err(boxed)?) {
                (Some(id), Some(draft), Some(end), Some(ring)) => draft.restore_mark(id, end, ring).map_err(boxed)?,
                (None, None, None, _) | (None, None, _, None) => {}
                _ => return Err("retained execution mode differs".into()),
            }
        }
        placement.len = plan.snapshot_end;
        placement.plan = Some(*plan);
        Ok(())
    }
    fn pages<'p>(&self, placement: &'p V41Placement) -> &'p [u32] {
        &placement.units
    }
    /// Extends every source's page table (the slot keeps its committed rows); runs at a drained
    /// boundary, so no append is pending on the slot.
    fn extend_pages(&self, placement: &mut V41Placement, units: &[u32]) -> Result<(), BoxError> {
        let committed = self.requests.borrow().cache().committed_end(placement.lease).map_err(boxed)? as usize;
        placement.units.extend_from_slice(units);
        if let Err(error) = self.bind_units(placement, committed) {
            placement.units.truncate(placement.units.len() - units.len());
            return Err(boxed(error));
        }
        Ok(())
    }
    fn commit_point(&self, placement: &V41Placement) -> usize {
        placement.len
    }
    fn capture(&self, slot: MarkSlot, placement: &V41Placement, len: usize) -> Result<(), BoxError> {
        let target = self.arena.target(slot).map_err(boxed)?;
        let saved = self.requests.borrow_mut().capture_mark(placement.lease, target).map_err(boxed)?;
        if saved.backbone.end as usize != len {
            return Err("V4.1 captures its committed frontier only".into());
        }
        if let (Some(id), Some(draft), Some(ring)) = (placement.draft, self.draft.borrow_mut().as_deref_mut(),
            self.arena.draft(slot).map_err(boxed)?) {
            draft.capture_mark(id, len as u64, ring).map_err(boxed)?;
        }
        Ok(())
    }
    fn queue_capture(&self, slot: MarkSlot, placement: &V41Placement, len: usize, tail: Option<TailCopy>)
        -> Result<Option<CaptureTicket>, BoxError> {
        let lane = self.lane.get();
        // The tail copy runs on the prefix stream; `queue_mark` orders the lane's copies after
        // it on the device.
        if let Some(tail) = tail { self.copy_rows(tail)?; }
        let target = self.arena.target(slot).map_err(boxed)?;
        let saved = self.requests.borrow_mut().queue_mark(lane, placement.lease, target).map_err(boxed)?;
        if saved.backbone.end as usize != len {
            let _ = self.requests.borrow_mut().abort_mark(lane);
            return Err("V4.1 captures its committed frontier only".into());
        }
        if let (Some(id), Some(draft), Some(ring)) = (placement.draft, self.draft.borrow_mut().as_deref_mut(),
            self.arena.draft(slot).map_err(boxed)?) {
            if let Err(error) = draft.queue_mark(lane, id, len as u64, ring) {
                let _ = self.requests.borrow_mut().abort_mark(lane);
                return Err(boxed(error));
            }
        }
        Ok(Some(CaptureTicket(lane as u32)))
    }
    fn capture_ready(&self, ticket: CaptureTicket) -> Result<bool, BoxError> {
        let lane = ticket.0 as usize;
        let mut requests = self.requests.borrow_mut();
        let mut draft = self.draft.borrow_mut();
        // Poll both before publishing: a capture is complete only when target and drafter
        // copies both landed.
        let target = if requests.mark_pending(lane) { requests.mark_ready(lane).map_err(boxed)? } else { true };
        let rings = match draft.as_deref_mut() {
            Some(draft) if draft.mark_pending(lane) => draft.mark_ready(lane).map_err(boxed)?,
            _ => true,
        };
        Ok(target && rings)
    }
    fn abort_capture(&self, ticket: CaptureTicket) -> Result<(), BoxError> {
        let lane = ticket.0 as usize;
        let target = self.requests.borrow_mut().abort_mark(lane);
        let rings = self.draft.borrow_mut().as_deref_mut().map_or(Ok(()), |draft| draft.abort_mark(lane));
        target.and(rings).map_err(boxed)
    }
    fn host_restored(&self, pages: &[u32]) -> Result<(), BoxError> {
        self.requests.borrow().cache().publish_units(pages).map_err(boxed)
    }
    fn restore(&self, _: Option<MarkSlot>, _: &mut V41Placement, _: usize) -> Result<(), BoxError> {
        Err("V4.1 restores through its restore plan".into())
    }
    fn copy_rows(&self, copy: TailCopy) -> Result<(), BoxError> {
        let requests = self.requests.borrow();
        requests.cache().copy_unit_rows(copy.from, copy.to, copy.rows).map_err(boxed)?;
        // Replicated layouts mirror the copied tail before any placement reads it.
        requests.cache().publish_units(&[copy.to]).map_err(boxed)
    }
    fn drain(&self) -> Result<(), BoxError> {
        self.requests.borrow().cache().prefix_drain().map_err(boxed)
    }
    fn page_segments(&self, page: u32) -> Vec<DeviceRange> {
        self.requests.borrow().cache().unit_segments(page).into_iter().map(range).collect()
    }
    fn mark_segments(&self, slot: MarkSlot) -> Vec<DeviceRange> {
        let mut segments = Vec::with_capacity(2);
        if let Ok(target) = self.arena.target(slot) { segments.push(range(target)); }
        if let Ok(Some(ring)) = self.arena.draft(slot) { segments.push(range(ring)); }
        segments
    }
}

/// The scheduler's prefix cache: the engine's [`PrefixCache`] with V4.1's metadata, plus the
/// family's arenas and the capture lanes' pending tickets.
pub(crate) struct V41Cache<'a> {
    cache: PrefixCache<CudaCopyEngine<'a>, V41Meta>,
    arenas: Arenas<'a>,
    pending: [Option<CaptureTicket>; 2],
    capture_session: Option<String>,
    pub points: points::Stats,
}

/// What a V4.1 admission restored: the resume point and, for an exact full-prompt hit, the
/// retained scores of the first token.
pub(crate) struct Restored {
    pub source: Option<cuteafd_engine::prefix::Source>,
    pub cached: usize,
    pub scores: Option<RetainedScores>,
    pub plan: Option<RestorePlan>,
}

impl<'a> V41Cache<'a> {
    pub fn new(arenas: Arenas<'a>, config: PrefixConfig,
        host: Option<(cuteafd_hostcache::config::Config, CudaCopyEngine<'a>)>, layout: FamilyLayout) -> Result<Self> {
        ensure!(layout.rule == ReuseRule::EXACT, "V4.1 restores exact snapshot frontiers only");
        let cache = PrefixCache::new(layout, config, host).map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(Self { cache, arenas, pending: [None, None], capture_session: None, points: Default::default() })
    }
    pub fn arenas(&self) -> &Arenas<'a> { &self.arenas }
    /// A cache that retains nothing (component fixtures).
    #[cfg(test)]
    pub fn disabled() -> Self {
        Self::configured(0)
    }
    /// A cache of `entries` per bank with no arena or host tier (CPU fixtures: nothing is
    /// captured).
    #[cfg(test)]
    pub fn configured(entries: usize) -> Self {
        let config = PrefixConfig { entries, mark_slots: 0, keep_logits: true, min_tokens: 1 };
        Self::new(Arenas { target: None, draft: None, slots: 0 }, config, None, layout(1, false))
            .expect("a fixture prefix cache")
    }
    pub fn engine(&self) -> &PrefixCache<CudaCopyEngine<'a>, V41Meta> { &self.cache }
    pub fn family<'r, 'w, C: DraftChain<'a>>(&'r self, requests: &'r mut Requests<'a>,
        draft: Option<&'r mut DraftRuntime<'w, 'a, C>>) -> V41Prefix<'r, 'a, 'w, C> {
        V41Prefix::new(requests, draft, &self.arenas)
    }
    pub fn capture_session(&mut self, session: Option<String>) {
        self.capture_session = session.clone();
        self.cache.capture_session(session);
    }
    pub fn restored_session(&self) -> Option<&str> { self.cache.restored_session() }
    pub fn turn_bank_enabled(&self) -> bool { self.cache.turn_bank_enabled() }
    pub fn host_config(&self) -> Option<&cuteafd_hostcache::config::Config> { self.cache.host_config() }
    pub fn host_metrics(&self) -> Option<cuteafd_hostcache::metrics::Snapshot> { self.cache.serving_stats().host }
    pub fn stats(&self) -> cuteafd_engine::prefix::PrefixStats { self.cache.stats() }
    pub fn serving_stats(&self) -> cuteafd_engine::prefix::PrefixStats { self.cache.serving_stats() }
    pub fn prefill_hold(&mut self) -> Result<()> { self.cache.prefill_hold() }

    /// Unreference released requests' units (after the family drained) and poll the host tier.
    pub fn tick<C: DraftChain<'a>>(&mut self, requests: &mut Requests<'a>, draft: Option<&mut DraftRuntime<'_, 'a, C>>)
        -> Result<()> {
        self.reclaim(requests, draft)?;
        self.cache.tick();
        Ok(())
    }
    /// Return released requests' units to the pool, so [`V41Cache::room`] sees them.
    pub fn reclaim<C: DraftChain<'a>>(&mut self, requests: &mut Requests<'a>, draft: Option<&mut DraftRuntime<'_, 'a, C>>)
        -> Result<()> {
        let released = requests.take_released_units();
        if released.is_empty() { return Ok(()); }
        let family = V41Prefix::new(requests, draft, &self.arenas);
        for units in released {
            self.cache.release(&family, &units).map_err(|e| anyhow::anyhow!("{e}"))?;
        }
        Ok(())
    }

    /// Admit a request leased at `lease` (fresh, its draft request admitted as `draft_id`):
    /// look `keys` up (media-keyed copy of `native`), allocate units for `capacity` tokens
    /// (its reservation: prompt plus a decode chunk), restore. Pool pressure is
    /// `SourcePoolExhausted`; a restore that fails is a miss. Internal staging
    /// may require exactly `boundary`; external admission leaves selection unchanged.
    #[allow(clippy::too_many_arguments)]
    pub fn admit<C: DraftChain<'a>>(&mut self, requests: &mut Requests<'a>, mut draft: Option<&mut DraftRuntime<'_, 'a, C>>,
        lease: CacheLease, draft_id: Option<u64>, keys: &[u32], native: &[u32], media: &[MediaSpan], capacity: usize,
        cold: bool, boundary: Option<usize>) -> Result<Restored> {
        self.reclaim(requests, draft.as_deref_mut())?;
        let family = V41Prefix::new(requests, draft, &self.arenas);
        let build = |units: Vec<u32>| V41Placement { units, lease, draft: draft_id, len: 0, plan: None };
        let admitted = if cold || !self.cache.enabled() {
            self.cache.admit_cold(&family, keys.len(), capacity, build)
        } else if let Some(boundary) = boundary {
            // Internal staging restores precisely the frontier just captured,
            // even if a peer retained a deeper ancestor while this stage ran.
            let compatible = |meta: &V41Meta, _: usize| meta.request.backbone.end as usize == boundary;
            self.cache.admit_native_compatible(&family, keys, native, media, capacity, true, Some(&compatible), build)
        } else {
            self.cache.admit_native(&family, keys, native, media, capacity, true, build)
        };
        let admitted = match admitted {
            Ok(admitted) => admitted,
            Err(cuteafd_engine::prefix::PrefixError::Pages(pages)) => {
                return Err(crate::families::deepseek_v41::v41_compressor::SourcePoolExhausted {
                    work_index: 0, needed: pages.needed, available: pages.free }.into());
            }
            Err(error) => return Err(anyhow::anyhow!("{error}")),
        };
        let scores = match &admitted.after {
            Some(after) => {
                let logits = after.logits.as_ref().context("exact prefix has no retained logits")?;
                let bytes = logits.iter().flat_map(|v| v.to_ne_bytes()).collect();
                Some(RetainedScores::new(super::scores::VOCAB, bytes)?)
            }
            None => None,
        };
        let plan = admitted.placement.plan;
        Ok(Restored { cached: admitted.resume, scores, plan, source: admitted.source })
    }

    /// Grow a live request's units to cover `rows` tokens (a decode reservation top-up), at a
    /// drained boundary. `Ok(false)` when the pool cannot supply them now: the request stalls.
    pub fn extend<C: DraftChain<'a>>(&mut self, requests: &mut Requests<'a>, draft: Option<&mut DraftRuntime<'_, 'a, C>>,
        lease: CacheLease, rows: usize) -> Result<bool> {
        self.reclaim(requests, None::<&mut DraftRuntime<'_, 'a, C>>)?;
        let units = requests.units(lease)?.to_vec();
        let len = requests.cache().committed_end(lease)? as usize;
        let mut placement = V41Placement { units, lease, draft: None, len, plan: None };
        let family = V41Prefix::new(requests, draft, &self.arenas);
        self.cache.extend(&family, &mut placement, rows).map_err(|e| anyhow::anyhow!("{e}"))
    }

    /// Units a reservation may still take: free units and units only retained snapshots hold.
    /// Capacity a reservation may take. Call [`V41Cache::reclaim`] after retiring requests first:
    /// a released request's units are free only once reclaimed.
    pub fn room(&self) -> cuteafd_engine::reserve::Room { self.cache.target_room() }
    /// No capture is still queued (its units return as an evictable snapshot when it lands).
    pub fn settled(&self) -> bool { self.cache.settled() }

    /// Retain a snapshot of `lease` at its committed frontier (synchronously).
    #[allow(clippy::too_many_arguments)]
    pub fn retain<C: DraftChain<'a>>(&mut self, requests: &mut Requests<'a>, draft: Option<&mut DraftRuntime<'_, 'a, C>>,
        kind: SnapshotKind, keys: &[u32], media: &[MediaSpan], next: &RetainedScores, lease: CacheLease, draft_id: Option<u64>)
        -> Result<()> {
        self.retain_frontier(requests, draft, kind, keys, media, next, lease, draft_id).map(drop)
    }

    /// The synchronous capture result, used by staged prefill to fall back cold
    /// when no ordinary complete snapshot could be retained.
    #[allow(clippy::too_many_arguments)]
    pub fn retain_frontier<C: DraftChain<'a>>(&mut self, requests: &mut Requests<'a>, draft: Option<&mut DraftRuntime<'_, 'a, C>>,
        kind: SnapshotKind, keys: &[u32], media: &[MediaSpan], next: &RetainedScores, lease: CacheLease, draft_id: Option<u64>)
        -> Result<bool> {
        if !self.cache.enabled() { return Ok(false); }
        let end = requests.cache().committed_end(lease)? as usize;
        ensure!(end > 0 && end <= keys.len(), "retained token frontier differs");
        let units = requests.units(lease)?.to_vec();
        let placement = V41Placement { units, lease, draft: draft_id, len: end, plan: None };
        let after = After::from_logits(&next.logits()?, true);
        let family = V41Prefix::new(requests, draft, &self.arenas);
        self.cache.capture_media(&family, kind, &keys[..end], media, &placement, after)
            .map_err(|e| anyhow::anyhow!("{e}"))
    }
    /// [`V41Cache::retain`] with its copies queued on capture lane `lane`; poll it with
    /// [`V41Cache::poll_retain`]. `Ok(false)` when nothing was queued.
    #[allow(clippy::too_many_arguments)]
    pub fn queue_retain<C: DraftChain<'a>>(&mut self, lane: usize, requests: &mut Requests<'a>,
        draft: Option<&mut DraftRuntime<'_, 'a, C>>, kind: SnapshotKind, keys: &[u32], media: &[MediaSpan],
        next: &RetainedScores, lease: CacheLease, draft_id: Option<u64>) -> Result<bool> {
        ensure!(self.pending.get(lane).context("invalid retention lane")?.is_none(), "retention lane occupied");
        if !self.cache.enabled() { return Ok(false); }
        let end = requests.cache().committed_end(lease)? as usize;
        ensure!(end > 0 && end <= keys.len(), "retained token frontier differs");
        let units = requests.units(lease)?.to_vec();
        let placement = V41Placement { units, lease, draft: draft_id, len: end, plan: None };
        let after = After::from_logits(&next.logits()?, true);
        let family = V41Prefix::new(requests, draft, &self.arenas);
        family.on_lane(lane);
        match self.cache.queue_capture(&family, kind, &keys[..end], media, &placement, after)
            .map_err(|e| anyhow::anyhow!("{e}"))? {
            Captured::Queued(ticket) => {
                self.pending[lane] = Some(ticket);
                Ok(true)
            }
            Captured::Done | Captured::Skipped => Ok(false),
        }
    }
    pub fn poll_retain<C: DraftChain<'a>>(&mut self, lane: usize, requests: &mut Requests<'a>,
        draft: Option<&mut DraftRuntime<'_, 'a, C>>) -> Result<bool> {
        let ticket = self.pending.get(lane).copied().flatten().context("retention is not pending")?;
        let family = V41Prefix::new(requests, draft, &self.arenas);
        let done = self.cache.poll_capture(&family, ticket).map_err(|e| anyhow::anyhow!("{e}"))?;
        if done { self.pending[lane] = None; }
        Ok(done)
    }
    pub fn abort_retain<C: DraftChain<'a>>(&mut self, lane: usize, requests: &mut Requests<'a>,
        draft: Option<&mut DraftRuntime<'_, 'a, C>>) -> Result<()> {
        let Some(ticket) = self.pending.get_mut(lane).and_then(Option::take) else { return Ok(()) };
        let family = V41Prefix::new(requests, draft, &self.arenas);
        self.cache.abort_capture(&family, ticket).map_err(|e| anyhow::anyhow!("{e}"))
    }
}

/// The radix key of a request's `tokens` (its prompt then its generated tokens): the prompt's
/// media-keyed copy, then the generated tokens' native ids (images occur only in prompts).
pub(crate) fn keyed<'t>(keys: &ImageKeys, tokens: &'t [u32]) -> std::borrow::Cow<'t, [u32]> {
    if keys.spans().is_empty() { return std::borrow::Cow::Borrowed(tokens); }
    let prompt = keys.tokens();
    let mut out = prompt[..prompt.len().min(tokens.len())].to_vec();
    out.extend_from_slice(&tokens[out.len()..]);
    std::borrow::Cow::Owned(out)
}

/// The draft request identity of request `id` when a drafter runs.
pub(crate) fn draft_id<T>(draft: &Option<T>, id: u64) -> Option<u64> {
    draft.is_some().then_some(id)
}

pub(crate) fn exhausted(error: &anyhow::Error) -> bool {
    error.downcast_ref::<crate::families::deepseek_v41::v41_compressor::SourcePoolExhausted>().is_some()
}

/// The engine's media spans of a V4.1 prompt's images (full 256-bit identities) and the prompt's
/// media-keyed token copy for the radix.
pub(crate) fn media_keys(tokens: &[u32], images: &[cuteafd_loader::V41ImageSpan])
    -> Result<cuteafd_engine::media::MediaKeys> {
    let spans: Vec<MediaSpan> = images.iter().map(|span| MediaSpan {
        start: span.start,
        len: span.image.grid().tokens(),
        key: cuteafd_core::ImageKey(*span.image.identity()).into(),
    }).collect();
    cuteafd_engine::media::MediaKeys::new(tokens, super::scores::VOCAB as u32, &spans)
        .map_err(|e| anyhow::anyhow!("image cache keys: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    use cuteafd_engine::prefix::{RestoreCandidate, RestoreFidelity};

    /// Every hit the radix can select under V4.1's rule, planned the way the engine plans it.
    fn selected(frontiers: &[&[u32]], prompt: &[u32]) -> Option<RestorePlan> {
        let rule = layout(8, true).rule;
        let mut banks = cuteafd_core::prefix::Retention::with_rule(8, rule);
        for (i, frontier) in frontiers.iter().enumerate() {
            banks.bank_mut(SnapshotKind::Prompt).insert(frontier, i);
        }
        let (common, snapshot_end, _) = banks.lookup_reusable(prompt)?;
        let hit = RestoreCandidate { common, snapshot_end, target_end: prompt.len(),
            resume: rule.skipped(common, snapshot_end), media: &[], saved_media: &[] };
        Some(plan(hit))
    }

    #[test]
    fn layout_reuses_exact_frontiers_only() {
        for draft in [false, true] {
            let layout = layout(4, draft);
            assert_eq!(layout.rule, ReuseRule::EXACT);
            assert!(!layout.rule.partial());
            assert_eq!(layout.mark_bytes, mark_bytes(draft));
        }
        // A partial match saves nothing; only an exact ancestor is reusable.
        for (common, frontier) in [(1001, 1500), (1000, 1002), (128, 4096), (1, 2)] {
            assert_eq!(ReuseRule::EXACT.skipped(common, frontier), 0, "{common}/{frontier}");
        }
        assert_eq!(ReuseRule::EXACT.skipped(777, 777), 777);
    }

    #[test]
    fn each_stage_applies_the_existing_128_token_restore_rule() {
        let tokens: Vec<u32> = (0..4096).collect();
        for boundary in [512, 513, 1024, 1025] {
            for suffix in [1, 37, 127, 128, 129] {
                let route = selected(&[&tokens[..boundary]], &tokens[..boundary + suffix]).unwrap();
                assert_eq!(route.snapshot_end, boundary);
                assert_eq!(route.fidelity, if suffix < 128 { RestoreFidelity::Exact }
                    else { RestoreFidelity::ExactEncoderContinuation });
                // The same snapshot can be a short first stage even if P is far away.
                let first = selected(&[&tokens[..boundary]], &tokens[..boundary + 37]).unwrap();
                assert_eq!(first.fidelity, RestoreFidelity::Exact);
            }
        }
    }

    #[test]
    fn an_exact_restore_never_selects_approximate_replay() {
        let tokens: Vec<u32> = (0..2048).collect();
        let mut branch = tokens[..1500].to_vec();
        branch[1001] = 99_999;
        branch.extend(3000..3100);
        // Prompt end, turn end and a long continuation all restore the exact frontier.
        for (frontiers, prompt, end) in [
            (vec![&tokens[..1001]], &tokens[..1001], 1001),
            (vec![&tokens[..1001]], &tokens[..1038], 1001),
            (vec![&tokens[..777]], &tokens[..2000], 777),
            (vec![&tokens[..512], &tokens[..1500]], &branch[..], 512),
        ] {
            let plan = selected(&frontiers, prompt).expect("an exact ancestor is selected");
            assert_eq!(plan.fidelity, if prompt.len() - end >= 128 {
                RestoreFidelity::ExactEncoderContinuation
            } else { RestoreFidelity::Exact });
            assert_eq!((plan.snapshot_end, plan.resume(), plan.lag.source_end), (end, end, end));
            assert!(plan.check(end).is_ok());
        }
        // A branch inside the only retained frontier is a miss, not an approximate replay.
        assert!(selected(&[&tokens[..1500]], &branch).is_none());
        assert!(selected(&[&tokens[..1500]], &tokens[..1499]).is_none());
    }

    #[test]
    fn the_cache_refuses_a_partial_reuse_rule() {
        let config = PrefixConfig { entries: 4, mark_slots: 0, keep_logits: true, min_tokens: 64 };
        let mut partial = layout(1, false);
        partial.rule = ReuseRule { align: 2, replay: Some(128) };
        assert!(V41Cache::new(Arenas { target: None, draft: None, slots: 0 }, config, None, partial).is_err());
        assert!(V41Cache::new(Arenas { target: None, draft: None, slots: 0 }, config, None, layout(1, false)).is_ok());
    }

    #[test]
    fn exact_continuation_fidelity_tracks_the_decoder_window_threshold() {
        let tokens: Vec<u32> = (0..1024).collect();
        for end in [512, 513] {
            for suffix in [0, 1, 127, 128, 129] {
                let plan = selected(&[&tokens[..end]], &tokens[..end + suffix]).unwrap();
                assert!(plan.exact());
                assert_eq!(plan.resume(), end);
                assert_eq!(plan.fidelity, if suffix >= 128 {
                    RestoreFidelity::ExactEncoderContinuation
                } else { RestoreFidelity::Exact });
            }
        }
    }

    #[test]
    fn serving_refuses_partial_on_before_native_startup() {
        use clap::Parser;
        for value in [None, Some("off"), Some("on")] {
            let mut argv = vec!["cuteafd", "serve-native", "--snapshot", "/model", "--native-lib", "/native.so"];
            if let Some(value) = value { argv.extend(["--prefix-partial", value]); }
            let crate::cli::Commands::ServeNative(mut args) = crate::cli::Cli::try_parse_from(argv).unwrap().command else {
                panic!("expected serve-native");
            };
            let result = super::super::normalize_serving(&mut args);
            if value == Some("on") {
                assert!(result.unwrap_err().to_string().contains("exact prefix frontiers only"));
            } else {
                result.unwrap();
            }
        }
    }

    /// The private prefix stream is safe by ownership (module docs): every function that queues
    /// work on it, or on a capture lane, reads only a committed request's private slots and
    /// immutable rows, or writes only a fresh placement. A new use must state its guard here.
    #[test]
    fn prefix_stream_reads_only_committed_private_state() {
        /// Each method of `source`: its name and its text up to its closing `    }` line.
        fn bodies(source: &str) -> Vec<(&str, &str)> {
            let mut out = Vec::new();
            for (at, _) in source.match_indices("\n    ") {
                let line = &source[at + 5..];
                let Some(fn_at) = ["pub fn ", "fn ", "pub unsafe fn ", "unsafe fn "].iter()
                    .find_map(|p| line.starts_with(p).then_some(p.len())) else { continue };
                let name = &line[fn_at..fn_at + line[fn_at..].find(['(', '<']).unwrap()];
                let end = line.find("\n    }\n").map_or(line.len(), |e| e + 6);
                out.push((name, &line[..end]));
            }
            out
        }
        let prefix = include_str!("../v41_backbone_cache/prefix.rs");
        let production = &prefix[..prefix.find("#[cfg(test)]").expect("test section")];
        // (function, the guard it must keep)
        let allowed: &[(&str, &[&str])] = &[
            ("diagnostic_state", &["self.capture_mark(lease"]),
            // Captures: a committed frontier with nothing unpublished (immutable rows).
            ("capture_mark", &["self.copy_mark(lease"]),
            ("queue_mark", &["self.copy_mark(lease", "cuda_stream_wait_event(stream, self.prefix_queued.0)"]),
            ("copy_mark", &["request.phase == CachePhase::Full && request.publication.is_empty()"]),
            // Restores: only a fresh admission, drained before it is observable.
            ("restore_retained_phase", &["request.end == 0", "request.version == 0", "cuda_stream_synchronize(self.prefix_stream.raw)"]),
            ("prefix_drain", &["cuda_stream_synchronize(self.prefix_stream.raw)"]),
            // Tail copies: into a fresh fork's private page, drained before release.
            ("copy_unit_rows", &["`to` belongs to the caller's fresh fork and no reader"]),
            ("publish_units", &["self.prefix_drain()?"]),
        ];
        for (name, body) in bodies(production) {
            // Work is queued through a raw stream handle (the prefix stream, a capture lane's) or a
            // stream parameter; polls and lane aborts only query or drain.
            let queues = body.contains(["prefix_stream", ".raw"].concat().as_str())
                || body.contains("copies.stream.raw") || body.contains("stream: *mut");
            if !queues { continue; }
            let guards = allowed.iter().find(|(allowed, _)| *allowed == name)
                .unwrap_or_else(|| panic!("{name} queues prefix-stream work without a stated ownership guard"));
            for guard in guards.1 {
                assert!(body.contains(guard), "{name} lost its ownership guard {guard:?}");
            }
        }
        // The request layer captures only a request with no pending prefill or history.
        let requests = include_str!("../v41_requests.rs");
        for name in ["pub fn capture_mark", "pub fn queue_mark"] {
            let body = &requests[requests.find(name).expect(name)..];
            let body = &body[..body.find("\n    pub fn ").unwrap_or(body.len())];
            assert!(body.contains("request.prefill.is_none() && request.history.position() == self.cache.committed_end(lease)?"),
                "{name} captures a request with pending history");
        }
        // Nothing outside the backbone cache reaches the private stream.
        for (file, source) in [("v41_backbone_cache.rs", include_str!("../v41_backbone_cache.rs")),
            ("v41_native_serve/prefix.rs", include_str!("prefix.rs"))] {
            let production = &source[..source.find("#[cfg(test)]").unwrap_or(source.len())];
            let uses = production.matches(["prefix_stream", ".raw"].concat().as_str()).count();
            assert_eq!(uses, 0, "{file} queues on the private prefix stream outside its guarded owners");
        }
        assert!(!include_str!("prefix.rs").contains(["Approximate", "Replay"].concat().as_str()));
    }

    #[test]
    fn metadata_is_plain_serializable_data() {
        let meta = V41Meta {
            request: RequestMark { backbone: BackboneMark { end: 513, windows: vec![(385, 513); 40] },
                engram_position: 513, engram_recent: vec![Some(5), None, Some(7)] },
            draft_end: Some(513),
        };
        let text = serde_json::to_string(&meta).unwrap();
        assert_eq!(serde_json::from_str::<V41Meta>(&text).unwrap(), meta);
    }
}
