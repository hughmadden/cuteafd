//! Physical Context owners for h64 DFlash. Producers and weights stay on the
//! lead; the ordinary target exchange carries native records and query rows.
use super::{draft_view, MlaAttention, MlaWorkspace};
use crate::shared::{draft::timing::DraftTiming, memory::device::{Allocation, Event, Stream},
    peer_split::{PeerExchange, DIRECT}};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{CuteafdDeviceBuffer as Buffer, mla_draft::{MlaDraftCombine, MlaDraftCombineBatch,
    MlaDraftPartial, MlaDraftPartialBatch, MlaDraftWindow}, peer_exchange::PeerPlane, programs::Programs};
use cuteafd_loader::formats::dflash::{MlaDraftCacheBatch, MlaDraftRuntimeLayout};
use std::{collections::BTreeMap, ffi::c_void, rc::Rc};
use crate::shared::memory::Tag;

pub(crate) struct ContextOwner<'a> {
    buffers: [BTreeMap<String, Allocation<'a>>; 2],
    partial: [[MlaDraftPartial<'a>; 2]; 2],
    combine: MlaDraftCombine<'a>,
    // Events and borrowed exchange streams outlive all their queued readers.
    ready: Event<'a>,
    complete: Event<'a>,
    streams: [Rc<Stream<'a>>; 2],
    peer: Rc<PeerExchange<'a>>,
    lead_stream: *mut c_void,
    record: usize,
}

impl<'a> ContextOwner<'a> {
    pub(crate) fn allocate(programs: &'a Programs<'a>, layout: &MlaDraftRuntimeLayout,
        streams: [Rc<Stream<'a>>; 2], peer: Rc<PeerExchange<'a>>, lead_stream: *mut c_void) -> Result<Self> {
        let inventory = layout.context_layout()?;
        let mut buffers: [BTreeMap<String, Allocation<'a>>; 2] = std::array::from_fn(|_| BTreeMap::new());
        for rank in 0..2 {
            ensure!(streams[rank].raw == peer.stream(rank) && streams[rank].device.id == peer.device(rank),
                "draft Context exchange streams differ from retained target exchange");
            for buffer in &inventory.buffers[rank] {
                // Lead cache halves are moved into the common layer owners.
                if rank == 1 && buffer.name.starts_with("layer.") { continue; }
                let allocation = streams[rank].device.run(|| {
                    let _scope = cuteafd_ffi::memory_ledger::scope("drafter/mla/context");
                    // The recipe states each buffer's kind (MlaDraftBuffer::kind_of).
                    Allocation::new(streams[rank].device, buffer.bytes,
                        crate::shared::memory::tag(buffer.kind, &format!("shared/mla-context/{rank}/{}", buffer.name)))
                })?;
                buffers[rank].insert(buffer.name.clone(), allocation);
            }
        }
        let load = |rank: usize| streams[rank].device.run(|| Ok([
            MlaDraftPartial::load(programs, "kimik3", MlaDraftWindow::Full, layout.cache_format())?,
            MlaDraftPartial::load(programs, "kimik3", MlaDraftWindow::Sliding4096, layout.cache_format())?]));
        let partial = [load(0)?, load(1)?];
        let combine = streams[1].device.run(|| Ok(MlaDraftCombine::load(programs, "kimik3")?))?;
        let ready = Event::new(streams[1].device)?;
        let complete = Event::new(streams[1].device)?;
        Ok(Self { buffers, partial, combine, ready, complete, streams, peer, lead_stream,
            record: layout.paged_cache().record_bytes })
    }

    pub(super) fn buffer(&self, rank: usize, name: &str) -> Result<Buffer> {
        Ok(self.buffers[rank].get(name).with_context(|| format!("draft Context owner {rank}/{name} missing"))?.buffer)
    }

    pub(super) fn drain(&self) -> Result<()> {
        // Request/turn barriers retire both physical owners, including final peer acks.
        for stream in &self.streams { stream.drain()?; }
        Ok(())
    }

    pub(super) fn exchange_streams(&self) -> impl Iterator<Item = &Stream<'a>> {
        self.streams.iter().map(Rc::as_ref)
    }

    pub(super) fn buffers(&self) -> impl Iterator<Item = (usize, &str, Buffer)> {
        self.buffers.iter().enumerate().flat_map(|(rank, buffers)|
            buffers.iter().map(move |(name, owner)| (rank, name.as_str(), owner.buffer)))
    }

    pub(super) fn cache(&self, layer: usize, head: Buffer, rank: usize) -> Result<Buffer> {
        if rank == 1 { Ok(head) } else { self.buffer(0, &format!("layer.{layer}.cache")) }
    }

    fn checked<T>(&self, work: impl FnOnce() -> Result<T>) -> Result<T> {
        let result = work();
        if result.is_err() { self.peer.publish_abort()?; }
        result
    }

    // One producer edge and one completion edge per Context transaction. The
    // lead stream waits on the event, never on a host acknowledgement.
    fn begin_lead(&self) -> Result<()> {
        self.streams[1].device.run(|| unsafe {
            self.streams[1].device.library.cuda_event_record(self.ready.raw, self.lead_stream)?;
            self.streams[1].device.library.cuda_stream_wait_event(self.peer.stream(1), self.ready.raw)
        })
    }

    fn begin(&self) -> Result<()> {
        self.begin_lead()?;
        self.peer.signal(1, DIRECT)?;
        self.peer.wait(0, DIRECT)
    }

    fn end(&self) -> Result<()> {
        self.streams[1].device.run(|| unsafe {
            self.streams[1].device.library.cuda_event_record(self.complete.raw, self.peer.stream(1))?;
            self.streams[1].device.library.cuda_stream_wait_event(self.lead_stream, self.complete.raw)
        })
    }

    fn put(&self, rank: usize, name: &str, bytes: &[u8]) -> Result<()> {
        let buffer = self.buffer(rank, name)?;
        ensure!(bytes.len() <= buffer.bytes, "Context draft metadata exceeds {name}");
        self.streams[rank].device.run(|| unsafe {
            self.streams[rank].device.library.copy_h2d_ordered(buffer, bytes, self.peer.stream(rank))
        })
    }

    pub(super) fn producer(&self, mut batch: cuteafd_ffi::mla_draft::MlaDraftBatch)
        -> Result<cuteafd_ffi::mla_draft::MlaDraftBatch> {
        batch.cache = self.buffer(1, "producer_cache")?;
        batch.cache_pages = 1;
        batch.slots = self.buffer(1, "producer_slots")?;
        Ok(batch)
    }

    pub(super) fn producer_slots(&self, rows: usize) -> Result<()> {
        ensure!((1..=64).contains(&rows), "Context draft producer rows");
        let slots: Vec<i64> = (0..rows as i64).collect();
        let buffer = self.buffer(1, "producer_slots")?;
        self.streams[1].device.run(|| unsafe {
            self.streams[1].device.library.copy_h2d_ordered(buffer, super::bytes_of(&slots), self.lead_stream)
        })
    }

    // Serial publication on DIRECT is intentional: <=4 aligned planes per
    // transaction, peer wait + device ack before the next destination reuse.
    fn remote_planes(&self, from: usize, planes: &[PeerPlane]) -> Result<()> {
        for planes in planes.chunks(cuteafd_ffi::peer_exchange::PEER_MAX_PLANES) {
            self.peer.push_planes(from, DIRECT, planes)?;
            self.peer.wait(1 - from, DIRECT)?;
            self.peer.signal(1 - from, DIRECT)?;
            self.peer.wait(from, DIRECT)?;
        }
        Ok(())
    }

    fn scatter(&self, index: usize, head: Buffer, slots: &[i64], query_follows: bool) -> Result<()> {
        let source = self.buffer(1, "producer_cache")?;
        let mut remote: Vec<PeerPlane> = Vec::new();
        for (row, &slot) in slots.iter().enumerate() {
            if slot == -1 { continue; }
            ensure!(slot >= 0 && row < 64, "Context scatter physical slot");
            let rank = slot as usize / 64 % 2;
            let local = (slot as usize / 128) * 64 + slot as usize % 64;
            let destination = draft_view(self.cache(index, head, rank)?, local * self.record, self.record)?;
            let source = draft_view(source, row * self.record, self.record)?;
            if rank == 1 {
                self.streams[1].device.run(|| unsafe {
                    self.streams[1].device.library.copy_d2d_async(destination, source, self.record, self.peer.stream(1))
                })?;
            } else {
                // Adjacent staging rows targeting adjacent compact records are
                // one plane, including runs crossing consecutive owner pages.
                if let Some(last) = remote.last_mut().filter(|last|
                    last.destination as usize + last.row_bytes as usize == destination.ptr as usize
                        && last.source as usize + last.row_bytes as usize == source.ptr as usize) {
                    last.row_bytes += self.record as u64;
                    last.destination_pitch = last.row_bytes;
                    last.source_pitch = last.row_bytes;
                } else {
                    remote.push(PeerPlane { destination: destination.ptr, source: source.ptr,
                        rows: 1, row_bytes: self.record as u64, destination_pitch: self.record as u64,
                        source_pitch: self.record as u64 });
                }
            }
        }
        if query_follows {
            // Query publication follows every scatter on stream 1. Its wait on
            // stream 0 orders cache reads; the partial reply retires all copies.
            for planes in remote.chunks(cuteafd_ffi::peer_exchange::PEER_MAX_PLANES) {
                self.peer.push_planes(1, DIRECT, planes)?;
                self.peer.wait(0, DIRECT)?;
            }
            Ok(())
        } else {
            self.remote_planes(1, &remote)
        }
    }

    pub(super) fn scatter_update(&self, index: usize, head: Buffer, slots: &[i64], timing: &DraftTiming) -> Result<()> {
        self.checked(|| {
            self.begin()?;
            let span = timing.begin("context_scatter", Some(index), slots.len(), self.peer.stream(1))?;
            self.scatter(index, head, slots, false)?;
            timing.end(span, self.peer.stream(1))?;
            self.end()
        })
    }

    pub(super) fn preflight(&self, index: usize, layer: &MlaAttention<'a>, workspace: &MlaWorkspace<'a>,
        metadata: &MlaDraftCacheBatch, window: Option<usize>, cache_pages: usize, scale: f32) -> Result<()> {
        let rows = metadata.positions.len();
        let shards = metadata.context_shards(window, cache_pages)?;
        for rank in 0..2 {
            let cache = self.cache(index, layer.cache.buffer, rank)?;
            let query = if rank == 0 { self.buffer(0, "query")? } else { workspace.query.buffer };
            self.partial[rank][usize::from(window.is_some())].check(MlaDraftPartialBatch {
                rows, page_table_width: shards[rank].page_table_width,
                cache_pages: cache.bytes / (64 * self.record), active_splits: layer.program.splits(),
                attention_scale: scale, query, cache, page_table: self.buffer(rank, "page_table")?,
                intervals: self.buffer(rank, "intervals")?, query_offsets: self.buffer(rank, "query_offsets")?,
                attention: self.buffer(rank, "partial")?, lse: self.buffer(rank, "lse")?,
                scratch: self.buffer(rank, "scratch")? })?;
            for (name, bytes) in [("page_table", shards[rank].page_table.len() * 4),
                ("intervals", shards[rank].intervals.len() * 4),
                ("query_offsets", shards[rank].query_offsets.len() * 4)] {
                ensure!(bytes <= self.buffer(rank, name)?.bytes, "Context metadata exceeds {name}");
            }
        }
        self.combine.check(MlaDraftCombineBatch {
            rows, partial: [self.buffer(1, "received_partial")?, self.buffer(1, "partial")?],
            lse: [self.buffer(1, "received_lse")?, self.buffer(1, "lse")?],
            sink: self.buffer(1, "sink")?, output: workspace.attention.buffer })?;
        Ok(())
    }

    pub(super) fn attend(&self, index: usize, layer: &MlaAttention<'a>, workspace: &MlaWorkspace<'a>,
        metadata: &MlaDraftCacheBatch, window: Option<usize>, cache_pages: usize,
        scale: f32, timing: &DraftTiming) -> Result<()> {
        let rows = metadata.positions.len();
        let shards = metadata.context_shards(window, cache_pages)?;
        self.checked(|| {
            self.begin_lead()?;
            let scatter = timing.begin("scatter", Some(index), rows, self.peer.stream(1))?;
            self.scatter(index, layer.cache.buffer, &metadata.slots, true)?;
            timing.end(scatter, self.peer.stream(1))?;
            let broadcast = timing.begin("query_broadcast", Some(index), rows, self.peer.stream(1))?;
            let peer_query = self.buffer(0, "query")?;
            self.peer.push_to(1, DIRECT, workspace.query.buffer.ptr, peer_query.ptr, rows * 64 * 576 * 2)?;
            timing.end(broadcast, self.peer.stream(1))?;
            self.peer.wait(0, DIRECT)?;
            for rank in 0..2 {
                self.put(rank, "page_table", super::bytes_of(&shards[rank].page_table))?;
                self.put(rank, "intervals", super::bytes_of(&shards[rank].intervals))?;
                self.put(rank, "query_offsets", super::bytes_of(&shards[rank].query_offsets))?;
                let cache = self.cache(index, layer.cache.buffer, rank)?;
                let query = if rank == 0 { peer_query } else { workspace.query.buffer };
                let batch = MlaDraftPartialBatch { rows, page_table_width: shards[rank].page_table_width,
                    cache_pages: cache.bytes / (64 * self.record), active_splits: layer.program.splits(),
                    attention_scale: scale, query, cache, page_table: self.buffer(rank, "page_table")?,
                    intervals: self.buffer(rank, "intervals")?, query_offsets: self.buffer(rank, "query_offsets")?,
                    attention: self.buffer(rank, "partial")?, lse: self.buffer(rank, "lse")?,
                    scratch: self.buffer(rank, "scratch")? };
                let span = if rank == 1 {
                    timing.begin("context_local_partial", Some(index), rows, self.peer.stream(1))?
                } else { None };
                self.streams[rank].device.run(|| Ok(self.partial[rank][usize::from(window.is_some())]
                    .launch(batch, self.peer.stream(rank))?))?;
                if rank == 1 { timing.end(span, self.peer.stream(1))?; }
            }
            // Peer query consumption and partial publication share one reply.
            let partial = self.buffer(0, "partial")?;
            let lse = self.buffer(0, "lse")?;
            let received = self.buffer(1, "received_partial")?;
            let received_lse = self.buffer(1, "received_lse")?;
            self.peer.push_planes(0, DIRECT, &[
                PeerPlane { destination: received.ptr, source: partial.ptr, rows: 1,
                    row_bytes: (rows * 64 * 512 * 2) as u64,
                    destination_pitch: (rows * 64 * 512 * 2) as u64, source_pitch: (rows * 64 * 512 * 2) as u64 },
                PeerPlane { destination: received_lse.ptr, source: lse.ptr, rows: 1,
                    row_bytes: (rows * 64 * 4) as u64,
                    destination_pitch: (rows * 64 * 4) as u64, source_pitch: (rows * 64 * 4) as u64 }])?;
            let reply = timing.begin("context_reply_join", Some(index), rows, self.peer.stream(1))?;
            self.peer.wait(1, DIRECT)?;
            timing.end(reply, self.peer.stream(1))?;
            let combine = timing.begin("context_combine", Some(index), rows, self.peer.stream(1))?;
            self.streams[1].device.run(|| Ok(self.combine.launch(MlaDraftCombineBatch {
                rows, partial: [received, self.buffer(1, "partial")?],
                lse: [received_lse, self.buffer(1, "lse")?], sink: self.buffer(1, "sink")?,
                output: workspace.attention.buffer }, self.peer.stream(1))?))?;
            timing.end(combine, self.peer.stream(1))?;
            // Received partial/query buffers cannot be reused before consumers.
            self.peer.signal(1, DIRECT)?;
            self.peer.wait(0, DIRECT)?;
            self.end()
        })
    }

    pub(super) fn segments(&self, index: usize, head: Buffer, rows: std::ops::Range<usize>)
        -> Result<Vec<Buffer>> {
        let mut result = Vec::new();
        for rank in 0..2 {
            let mut first = None;
            let mut count = 0;
            let mut start = rows.start;
            while start < rows.end {
                let end = rows.end.min((start / 64 + 1) * 64);
                if start / 64 % 2 == rank {
                    first.get_or_insert(start / 128 * 64 + start % 64);
                    count += end - start;
                }
                start = end;
            }
            if let Some(first) = first {
                result.push(draft_view(self.cache(index, head, rank)?, first * self.record, count * self.record)?);
            }
        }
        Ok(result)
    }

    pub(super) fn zero(&self, buffers: &[Buffer]) -> Result<()> {
        self.checked(|| {
            self.begin()?;
            for &buffer in buffers {
                let rank = (0..2).find(|&rank| self.peer.device(rank) == buffer.device_id)
                    .context("Context zero owner absent")?;
                self.streams[rank].device.run(|| unsafe {
                    self.streams[rank].device.library.cuda_zero_bytes_async(buffer, buffer.bytes, self.peer.stream(rank))
                })?;
            }
            self.peer.signal(0, DIRECT)?;
            self.peer.wait(1, DIRECT)?;
            self.end()
        })
    }

    pub(super) fn copy(&self, destination: Buffer, source: Buffer) -> Result<()> {
        ensure!(destination.bytes == source.bytes, "Context draft exact copy extent");
        self.checked(|| {
            self.begin()?;
            let rank = |buffer: Buffer| -> Result<usize> {
                (0..2).find(|&rank| self.peer.device(rank) == buffer.device_id)
                    .context("Context draft copy references an absent GPU")
            };
            let (to, from) = (rank(destination)?, rank(source)?);
            if from == to {
                self.streams[from].device.run(|| unsafe {
                    self.streams[from].device.library.copy_d2d_async(destination, source, source.bytes, self.peer.stream(from))
                })?;
                if from == 0 {
                    self.peer.signal(0, DIRECT)?;
                    self.peer.wait(1, DIRECT)?;
                }
            } else {
                // When GPU0 produces the copy, make it wait for the lead's
                // preceding restore/canonicalization before touching its cache.
                if from == 0 {
                    self.peer.signal(1, DIRECT)?;
                    self.peer.wait(0, DIRECT)?;
                }
                self.remote_planes(from, &[PeerPlane { destination: destination.ptr, source: source.ptr,
                    rows: 1, row_bytes: source.bytes as u64,
                    destination_pitch: source.bytes as u64, source_pitch: source.bytes as u64 }])?;
                if from == 0 {
                    self.peer.signal(0, DIRECT)?;
                    self.peer.wait(1, DIRECT)?;
                }
            }
            self.end()
        })
    }
}

impl Drop for ContextOwner<'_> {
    fn drop(&mut self) {
        for stream in &self.streams { stream.drain_or_abort("draft Context owners"); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::{fp8_linear::{Fp8Rows, Fp8Scales}, memory::device::Device};
    use cuteafd_loader::{formats::dflash::MlaDflashConfig,
        families::glm5::draft_representation::{GlmDraftCapacity, GlmDraftRepresentation}};

    #[test]
    #[ignore = "requires sealed h64 exports, native library, two RTX GPUs and CUTEAFD_KIMI_DRAFT_SNAPSHOT"]
    fn context_prefix_restore_cross_owner_cow_is_byte_exact_on_both_gpus() -> Result<()> {
        let env = |name| std::env::var(name).with_context(|| format!("Context COW proof needs {name}"));
        let snapshot = std::path::PathBuf::from(env("CUTEAFD_KIMI_DRAFT_SNAPSHOT")?);
        // SAFETY: golden-run supplies the sealed library and every borrower drops before it.
        let library = unsafe { cuteafd_ffi::NativeLibrary::load(env("CUTEAFD_NATIVE_LIB")?)? };
        let manifest = std::path::PathBuf::from(env("CUTEAFD_PROGRAM_MANIFEST")?);
        let programs = library.programs()?.with_manifest(&manifest)?;
        let devices = [0, 1].map(|id| Device { library: &library, id });
        ensure!(crate::shared::peer_split::probed_device(&library, 0, Some(1))? == Some(1), "Context COW needs P2P");
        let streams = [Rc::new(Stream::new(devices[0])?), Rc::new(Stream::new(devices[1])?)];
        let peer = Rc::new(devices[0].run(|| PeerExchange::new_abortable(&library,
            [0, 1].map(|rank| crate::shared::peer_split::RankDevice {
                device: devices[rank].id, stream: streams[rank].raw }), 1, 65536, 65536))?);
        let lead = Stream::new(devices[1])?;
        let native = MlaDflashConfig::read(&snapshot)?;
        for cache_format in [cuteafd_core::MlaCacheFormat::Bf16, cuteafd_core::MlaCacheFormat::Fp8] {
            let layout = native.runtime_layout(GlmDraftRepresentation::Fp8Only,
                GlmDraftCapacity::new(2, 2, native.block)?, 8192, 16384,
                devices[1].run(|| library.sm_count())?, Fp8Rows::W8a16.code() as u8, cache_format)?
                .with_context_split(true)?;
            let split = ContextOwner::allocate(&programs, &layout, streams.clone(), peer.clone(), lead.raw)?;
            let draft = devices[1].own(|| super::super::DflashDrafter::load_mla(&library, &programs, crate::shared::speculation::dflash::TargetProjection::Native,
                "kimik3", &snapshot, std::fs::read(snapshot.join("model.safetensors"))?, lead.raw,
                layout, GlmDraftRepresentation::Fp8Only, Fp8Scales::Amax, Fp8Rows::W8a16, Some(split)))?;
            let iterations = if cache_format == cuteafd_core::MlaCacheFormat::Fp8 {
                std::env::var("CUTEAFD_KIMI_DRAFT_COW_ITERATIONS").ok()
                    .map(|value| value.parse::<usize>()).transpose()?.unwrap_or(1)
            } else { 1 };
            ensure!((1..=1000).contains(&iterations), "Context COW iterations must be 1..=1000");
            for iteration in 0..iterations {
                // Exercise partials/combine and staging reuse on every iteration.
                draft.startup_primed.set(false);
                devices[1].run(|| draft.prime_remote())?;
                devices[1].run(|| draft.drain_mla())?;
                let super::super::DraftContext::Mla(context) = &draft.context else { unreachable!() };
                // Priming recycles arbitrary page IDs. Start a fresh, fully drained
                // host pool so capture 1->2 and restore 2->3 prove both directions.
                *context.state.borrow_mut() = super::super::MlaPagedState::new(&context.layout)?;
                devices[1].run(|| draft.reset_context(0, 0))?;
                let rows = 67;
                let taps: Vec<u8> = (0..rows * draft.cfg.taps.len() * draft.cfg.hidden)
                    .flat_map(|i| (0x3d00u16 + (i % 127) as u16).to_le_bytes()).collect();
                let fill = |slot| devices[1].run(|| {
                    draft.put_taps(&taps)?;
                    draft.update(&(0..rows).map(|position| super::super::ContextRow {
                        tap_row: position, slot, position }).collect::<Vec<_>>())
                });
                fill(0)?;
                let saved = devices[1].run(|| draft.capture_mla_pages(0, rows, rows)
                    .map_err(|error| anyhow::anyhow!("{error}")))?;
                let super::super::DraftContext::Mla(context) = &draft.context else { unreachable!() };
                let source_tail = context.state.borrow().pool.committed_page_table(
                    context.state.borrow().bindings[0].unwrap())?[1] as usize;
                let saved_tail = saved.pages[1].index as usize;
                assert_ne!(source_tail % 2, saved_tail % 2, "capture must cross physical owners");
                let before = devices[1].run(|| draft.mla_readback(0, 0))?;
                let marks = devices[1].run(|| draft.mla_window_buffers(0)?.iter()
                    .map(|buffer| Allocation::new(devices[1], buffer.bytes, crate::shared::memory::TEST)).collect::<Result<Vec<_>>>())?;
                let mark_buffers = marks.iter().map(|owner| owner.buffer).collect::<Vec<_>>();
                devices[1].run(|| draft.copy_mla_windows(0, &mark_buffers, true))?;
                devices[1].run(|| draft.copy_mla_windows(1, &mark_buffers, false))?;
                devices[1].run(|| draft.restore_mla_pages(1, &saved, rows)
                    .map_err(|error| anyhow::anyhow!("{error}")))?;
                let restored_tail = context.state.borrow().pool.committed_page_table(
                    context.state.borrow().bindings[1].unwrap())?[1] as usize;
                // Readback above may recycle a temporary snapshot page; ownership
                // crossing is mandatory, not merely an unverified allocator hope.
                assert_ne!(saved_tail % 2, restored_tail % 2, "restore tail must cross physical owners");
                let restored = devices[1].run(|| draft.mla_readback(1, 0))?;
                assert_eq!(before, restored, "both owner caches and odd-slot window marks restore exactly");
                devices[1].run(|| draft.reset_context(1, 0))?;
                fill(1)?;
                let fresh = devices[1].run(|| draft.mla_readback(1, 0))?;
                assert_eq!(restored, fresh, "cross-owner restore equals fresh committed state byte for byte");
                devices[1].run(|| draft.release_mla_pages(&saved))?;
                devices[1].run(|| draft.drain_mla())?;
                eprintln!("Context COW {cache_format:?} iteration {} exact", iteration + 1);
            }
        }
        Ok(())
    }
}
