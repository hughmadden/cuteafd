//! One lane's learned index query and retained source-20 candidates.
use crate::families::deepseek_v41::v41_attention_query::AttentionQueryOutput;
use crate::families::deepseek_v41::v41_backbone_cache::{CacheAttention, CachePlacement};
use crate::shared::memory::device::{Device, DeviceOwner};
mod placement;
use crate::families::deepseek_v41::v41_index_query::{IndexQueryStorage, IndexQueryWave, IndexQueryWeights};
use crate::families::deepseek_v41::v41_index_selection::{IndexSelectionOutput, IndexSelectionStorage, IndexSelectionWave};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::NativeLibrary;
use cuteafd_loader::OfficialV41Catalog;
use std::rc::Rc;
use crate::shared::memory::row_window::RowWindow;

const LAYERS: [usize; 8] = [2, 8, 14, 20, 24, 28, 32, 36];
pub(crate) struct IndexLaneWeights<'a> {
    library: &'a NativeLibrary,
    weights: Vec<DeviceOwner<'a, IndexQueryWeights<'a>>>,
    placement: Option<CachePlacement>,
}
impl<'a> IndexLaneWeights<'a> {
    pub fn device_bytes(library: &NativeLibrary, catalog: &OfficialV41Catalog) -> Result<usize> {
        LAYERS.into_iter().try_fold(0usize, |total, layer| {
            total
                .checked_add(IndexQueryWeights::device_bytes(library, catalog, layer)?)
                .context("index weight budget overflow")
        })
    }
    pub fn load(
        library: &'a NativeLibrary,
        catalog: &OfficialV41Catalog,
        budget: usize,
        staging: usize,
    ) -> Result<Self> {
        ensure!(
            Self::device_bytes(library, catalog)? <= budget,
            "index lane weights exceed budget"
        );
        Self::load_placed(library, catalog, staging, None)
    }
}

/// One per GPU: query row planes and selection output planes span the reserve;
/// query/selection hook-local scratch spans only the unit cap. Source and
/// reindex have distinct retained planes but share one selection stream and
/// scratch. All phase, bindings and pinned metadata remain in each slot's waves.
pub(crate) struct IndexLaneStorage<'a> {
    query: Rc<IndexQueryStorage<'a>>,
    source: Rc<IndexSelectionStorage<'a>>,
    reindex: Option<Rc<IndexSelectionStorage<'a>>>,
    device: Option<i32>,
}
pub(crate) struct IndexLane<'w, 'a> {
    weights: &'w IndexLaneWeights<'a>,
    source: IndexSelectionWave<'a>,
    reindex: Option<IndexSelectionWave<'a>>,
    query: IndexQueryWave<'w, 'a>,
    next: usize,
    device: Option<i32>,
    ready: Option<usize>,
    invalid: bool,
}
impl<'w, 'a> IndexLane<'w, 'a> {
    pub fn workspace_bytes(library: &NativeLibrary, capacity: u32) -> Result<[usize; 3]> {
        Ok([
            IndexQueryWave::device_bytes(library, capacity)?,
            IndexSelectionWave::device_bytes(capacity as usize)?,
            // The reindex wave shares the source's scores and top-k scratch
            // (they run one after the other in a lane), as the placed lanes do.
            IndexSelectionWave::shared_device_bytes(capacity as usize)?,
        ])
    }
    pub fn new(weights: &'w IndexLaneWeights<'a>, capacity: u32, budget: usize) -> Result<Self> {
        Self::new_inner(weights, capacity, budget, None)
    }
    fn new_inner(weights: &'w IndexLaneWeights<'a>, capacity: u32, budget: usize,
        device: Option<i32>) -> Result<Self> {
        let bytes = match device {
            Some(gpu) => Self::placed_workspace_bytes(weights.library,
                weights.placement.context("placed index weights absent")?,capacity,gpu as usize)?,
            None => Self::workspace_bytes(weights.library, capacity)?,
        };
        let total = bytes.iter().try_fold(0usize, |n, &b| {
            n.checked_add(b).context("index workspace budget overflow")
        })?;
        ensure!(total <= budget, "index lane workspace exceeds budget");
        ensure!(
            weights.weights.len() == 8,
            "index lane requires all eight weight owners"
        );
        ensure!(weights.placement.is_some() == device.is_some(), "index workspace placement differs from weights");
        let first = Self::next_owned(weights, device, 0);
        ensure!(first < LAYERS.len(), "GPU has no index producers");
        Self::on_storage(weights, Self::storage_inner(weights, capacity, capacity, device)?)
    }
    pub fn storage_bytes(library: &NativeLibrary, rows: u32, unit: u32) -> Result<[usize; 3]> {
        Ok([IndexQueryWave::storage_bytes(library, rows, unit)?,
            IndexSelectionWave::storage_bytes(rows as usize, unit as usize)?,
            rows as usize * (2048 + 8192)])
    }
    pub fn storage(weights: &'w IndexLaneWeights<'a>, rows: u32, unit: u32) -> Result<Rc<IndexLaneStorage<'a>>> {
        ensure!(weights.placement.is_none(), "unplaced index storage requires unplaced weights");
        Self::storage_inner(weights, rows, unit, None)
    }
    fn storage_inner(weights: &'w IndexLaneWeights<'a>, rows: u32, unit: u32,
        device: Option<i32>) -> Result<Rc<IndexLaneStorage<'a>>> {
        ensure!(weights.weights.len() == LAYERS.len() && weights.placement.is_some() == device.is_some(),
            "index storage weight placement differs");
        let first = Self::next_owned(weights, device, 0);
        ensure!(first < LAYERS.len(), "GPU has no index producers");
        let source = IndexSelectionWave::storage(weights.library, rows as usize, unit as usize)?;
        let reindex = if device.is_none_or(|gpu| weights.placement.is_some_and(|p| p.attention(20).ok() == Some(gpu as usize))) {
            Some(IndexSelectionWave::follower_storage(&source)?)
        } else { None };
        Ok(Rc::new(IndexLaneStorage {
            query: weights.weights[first].storage(rows, unit)?, source, reindex, device,
        }))
    }
    pub fn on_storage(weights: &'w IndexLaneWeights<'a>, storage: Rc<IndexLaneStorage<'a>>) -> Result<Self> {
        ensure!(weights.weights.len() == LAYERS.len() && weights.placement.is_some() == storage.device.is_some(),
            "index lane storage placement differs");
        let first = Self::next_owned(weights, storage.device, 0);
        ensure!(first < LAYERS.len(), "GPU has no index producers");
        Ok(Self {
            weights, query: weights.weights[first].on_storage(storage.query.clone())?,
            source: IndexSelectionWave::on_storage(storage.source.clone())?,
            reindex: storage.reindex.as_ref().map(|s| IndexSelectionWave::on_storage(s.clone())).transpose()?,
            next: first, device: storage.device, ready: None, invalid: false,
        })
    }
    /// Select at rest, after the previous unit and all its readers have drained.
    pub(crate) fn select_window(&mut self, window: RowWindow) -> Result<()> {
        ensure!(!self.invalid && self.ready.is_none(), "index lane window changes only at rest");
        self.query.select_window(window.clone())?;
        self.source.select_window(window.clone())?;
        if let Some(reindex) = &mut self.reindex { reindex.select_window(window)?; }
        Ok(())
    }
    fn next_owned(weights: &IndexLaneWeights<'_>, device: Option<i32>, start: usize) -> usize {
        if device.is_none() { return start; }
        (start..LAYERS.len()).find(|&i| device.is_none_or(|id| weights.weights[i].device.id == id))
            .unwrap_or(LAYERS.len())
    }
    fn advance(&mut self) {
        self.next = Self::next_owned(self.weights, self.device, self.next + 1);
    }
    /// Begin a new batch after all consumers finish; also recovers failed work.
    pub fn enable_small_graph_shapes(&mut self) {
        self.query.enable_small_graph_shapes();
        self.source.enable_small_graph_shapes();
        if let Some(reindex) = &mut self.reindex { reindex.enable_small_graph_shapes(); }
    }
    pub fn restart(&mut self) -> Result<()> {
        self.restart_at(0)
    }
    /// Replay begins at decoder source 20; earlier encoder selections are absent.
    pub fn restart_decoder(&mut self) -> Result<()> {
        self.restart_at(3)
    }
    fn restart_at(&mut self, first: usize) -> Result<()> {
        self.invalid = true;
        self.ready = None;
        self.source.finish_audit()?;
        if let Some(reindex) = &mut self.reindex { reindex.finish_audit()?; }
        self.source.restart()?;
        if let Some(reindex) = &mut self.reindex { reindex.restart()?; }
        let first = Self::next_owned(self.weights, self.device, first);
        if first < LAYERS.len() { self.query.rebind(&self.weights.weights[first])?; }
        self.next = first;
        self.invalid = false;
        Ok(())
    }
    /// # Safety
    /// Query and cache belong to the same admitted batch. All producers have
    /// completed and no external writes race these owners. Call once per index
    /// producer in layer order; intermediate attention layers reuse output().
    pub unsafe fn select(
        &mut self,
        query: &AttentionQueryOutput<'_>,
        cache: &CacheAttention<'_>,
    ) -> Result<()> {
        if crate::shared::memory::chain::active() {
            // Stream-ordered: the queued projection and selection finish on the
            // chain immediately, so the host never waits here.
            unsafe { self.enqueue_projection(query)?; }
            ensure!(self.poll_projection()?, "chained index projection still pending");
            unsafe { self.enqueue_selection(cache)?; }
            ensure!(self.poll_selection()?, "chained index selection still pending");
            return Ok(());
        }
        let valid = !std::mem::replace(&mut self.invalid, true);
        self.ready = None;
        ensure!(
            valid && LAYERS.get(self.next) == Some(&query.layer)
                && self.device.is_none_or(|id| query.hidden.device_id == id),
            "index producer order differs; restart lane"
        );
        // The direct index path drains its own streams and uses legacy-stream
        // uploads; order it after any chained producers on the host.
        crate::shared::memory::chain::settle(self.weights.library)?;
        self.query.rebind(&self.weights.weights[self.next])?;
        let requests = cache.selection_requests()?;
        let projected = unsafe { self.query.execute_attention(query)? };
        if query.layer <= 20 {
            unsafe {
                self.source.execute(&projected, &requests, None)?;
            }
        } else {
            let candidates = self.source.output()?;
            unsafe {
                self.reindex.as_mut().context("decoder reindex workspace absent")?
                    .execute(&projected, &requests, Some(&candidates))?;
            }
        }
        self.ready = Some(query.layer);
        self.advance();
        self.invalid = false;
        Ok(())
    }
    /// # Safety
    /// Retain main query/cache inputs until selection completes or abort drains.
    pub unsafe fn enqueue_projection(&mut self, query: &AttentionQueryOutput<'_>) -> Result<()> {
        let valid = !std::mem::replace(&mut self.invalid, true);
        self.ready = None;
        ensure!(valid && LAYERS.get(self.next) == Some(&query.layer)
                && self.device.is_none_or(|id| query.hidden.device_id == id), "queued index order differs");
        self.query.rebind(&self.weights.weights[self.next])?;
        unsafe { self.query.enqueue_attention(query) }
    }
    pub fn poll_projection(&mut self) -> Result<bool> { self.query.poll_pending() }
    /// # Safety
    /// Preserve completed projection and cache proposal storage through selection.
    pub unsafe fn enqueue_selection(&mut self, cache: &CacheAttention<'_>) -> Result<()> {
        ensure!(self.invalid, "queued index projection absent");
        let query = self.query.output()?;
        ensure!(LAYERS.get(self.next) == Some(&query.layer), "queued selection layer differs");
        let requests = cache.selection_requests()?;
        if query.layer <= 20 {
            unsafe { self.source.enqueue_selection(&query, &requests, None) }
        } else {
            let shared = self.source.output()?;
            unsafe { self.reindex.as_mut().context("decoder reindex workspace absent")?.enqueue_selection(&query, &requests, Some(&shared)) }
        }
    }
    pub fn poll_selection(&mut self) -> Result<bool> {
        ensure!(self.invalid, "queued index selection absent");
        let layer = *LAYERS.get(self.next).context("queued index order exhausted")?;
        let ready = if layer <= 20 { self.source.poll_pending()? } else { self.reindex.as_mut().context("decoder reindex workspace absent")?.poll_pending()? };
        if ready { self.ready = Some(layer); self.advance(); self.invalid = false; }
        Ok(ready)
    }
    pub fn abort_pending(&mut self) -> Result<()> {
        // Drain consumers before their projected inputs can be recycled.
        let reindex = self.reindex.as_mut().map_or(Ok(()), |wave| wave.abort_pending());
        let result = reindex.and(self.source.abort_pending()).and(self.query.abort_pending());
        self.invalid = true; self.ready = None;
        result
    }
    /// Revalidate exact proposal snapshots and row order for the attention layer.
    pub fn output(
        &self,
        layer: usize,
        cache: &CacheAttention<'_>,
    ) -> Result<IndexSelectionOutput<'_>> {
        ensure!(!self.invalid, "index lane requires restart");
        let producer = self.ready.context("index lane output unavailable")?;
        let output = if producer <= 20 {
            self.source.output()?
        } else {
            self.reindex.as_ref().context("decoder reindex workspace absent")?.output()?
        };
        let requests = cache.selection_requests()?;
        let bindings = requests
            .iter()
            .flat_map(|r| r.positions.iter().map(|&p| (r.proposal.binding(), p)))
            .collect::<Vec<_>>();
        output.validate_attention(layer, &bindings)?;
        Ok(output)
    }
}

impl Drop for IndexLane<'_, '_> {
    fn drop(&mut self) {
        if let Err(error) = self.abort_pending() { tracing::error!(%error, "draining index lane"); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn official_index_lane_budgets_reject_before_allocation() -> Result<()> {
        let Some(path) = std::env::var_os("CUTEAFD_LANE_PLAN_LIBRARY") else {
            eprintln!("skip index planning test: CUTEAFD_LANE_PLAN_LIBRARY unset");
            return Ok(());
        };
        let model = std::env::var_os("CUTEAFD_LANE_PLAN_MODEL")
            .context("CUTEAFD_LANE_PLAN_MODEL required")?;
        let library = unsafe { NativeLibrary::load(path)? };
        let catalog = cuteafd_loader::read_official_v41_catalog(std::path::Path::new(&model),
        )?;
        let bytes = IndexLaneWeights::device_bytes(&library, &catalog)?;
        assert_eq!(bytes, 45_916_160);
        assert!(
            IndexLaneWeights::load(&library, &catalog, bytes - 1, 1024 * 1024)
                .err()
                .unwrap()
                .to_string()
                .contains("weights exceed budget")
        );
        let empty = IndexLaneWeights {
            library: &library,
            weights: Vec::new(),
            placement: None,
        };
        for capacity in [1, 80, 4096] {
            let groups = IndexLane::workspace_bytes(&library, capacity)?;
            let total = groups.iter().sum::<usize>();
            assert!(IndexLane::new(&empty, capacity, total - 1)
                .err()
                .unwrap()
                .to_string()
                .contains("workspace exceeds budget"));
            assert!(IndexLane::new(&empty, capacity, total)
                .err()
                .unwrap()
                .to_string()
                .contains("all eight weight owners"));
            eprintln!("index lane capacity={capacity} workspace_groups={groups:?} total={total}");
        }
        for capacity in [0, 4097, u32::MAX] {
            assert!(IndexLane::workspace_bytes(&library, capacity).is_err());
        }
        Ok(())
    }
}

#[cfg(test)]
impl IndexLaneWeights<'_> {
    pub(crate) fn visit(&self, f: &mut dyn FnMut(String, cuteafd_ffi::CuteafdDeviceBuffer)) {
        for (i, w) in self.weights.iter().enumerate() { w.visit(&format!("gpu{}.index{i}.", w.device.id), f); }
    }
}

#[cfg(test)]
mod shared_storage_tests {
    use super::*;
    use crate::families::deepseek_v41::v41_attention_query::AttentionQueryWeights;
    use crate::families::deepseek_v41::v41_compressor::{CompressorChunk, CompressorState, CompressorWeights};
    use crate::families::deepseek_v41::v41_index_selection::SelectionRequest;
    use crate::shared::memory::{chain, LoadStream};

    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIB, CUTEAFD_SNAPSHOT and a GPU"]
    fn windowed_index_units_match_private_runs_behind_a_gated_stream() -> Result<()> {
        eprintln!("index shared-storage gate: preparing weights and inputs");
        // Distinct unit widths/positions poison both host metadata and row planes.
        // Each unit has its own stage chain, just as in-flight prefill slots do.
        let library = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        let catalog = cuteafd_loader::read_official_v41_catalog(std::path::Path::new(
            &std::env::var("CUTEAFD_SNAPSHOT")?))?;
        let weights = IndexLaneWeights::load(&library, &catalog, usize::MAX, 1 << 20)?;
        let shared = IndexLane::storage(&weights, 48, 16)?;
        let mut slots = [IndexLane::on_storage(&weights, shared.clone())?,
            IndexLane::on_storage(&weights, shared.clone())?];
        let counts = [13usize, 11];
        let context = [1024usize, 896];
        let tokens = std::array::from_fn::<_, 2, _>(|i|
            ((context[i] - counts[i]) as u64..context[i] as u64).collect::<Vec<_>>());
        for (i, slot) in slots.iter_mut().enumerate() {
            slot.select_window(RowWindow::rows(i * 16..i * 16 + counts[i]))?;
        }
        let input = |seed: usize, rows: usize| -> Vec<u8> {
            (0..rows * 5120).flat_map(|i| {
                let x = ((i * 7 + seed * 11) % 29) as f32 / 32.0 - 0.4;
                ((x.to_bits() >> 16) as u16).to_ne_bytes()
            }).collect()
        };
        let cw = CompressorWeights::load(&library, &catalog, 20, usize::MAX, 1 << 20)?;
        let mut state = CompressorState::new(&library, 20, 2, 2, usize::MAX)?;
        let leases = [state.begin_request(0, 1)?, state.begin_request(1, 2)?];
        let mut compressors = [cw.wave(context[0], usize::MAX)?, cw.wave(context[1], usize::MAX)?];
        for i in 0..2 {
            library.copy_h2d(compressors[i].input(), &input(i, context[i]))?;
            // SAFETY: initialized private input and live state; execution drains.
            unsafe { compressors[i].execute(&state, &[CompressorChunk {
                lease: leases[i], position: 0, tokens: context[i] as u32,
            }])?; }
        }
        let proposals = [compressors[0].index_proposal(&state, leases[0])?,
            compressors[1].index_proposal(&state, leases[1])?];
        let query_weights = [20usize, 24].into_iter().map(|layer| {
            AttentionQueryWeights::load(&library, &catalog, layer, usize::MAX, 1 << 20)
        }).collect::<Result<Vec<_>>>()?;
        let mut queries = Vec::new();
        for (layer, weight) in query_weights.iter().enumerate() {
            let mut units = Vec::new();
            for i in 0..2 {
                let mut query = weight.wave(16, usize::MAX)?;
                library.copy_h2d(query.input(), &input(i + layer * 7, counts[i]))?;
                // SAFETY: finite private input; tokens match rows.
                unsafe { query.execute_tokens(&tokens[i])?; }
                units.push(query);
            }
            queries.push(units);
        }
        // First-use module/handle setup may synchronize the CUDA context. Warm
        // exact calls on these same streams before closing the gate, using the
        // otherwise unused tail of each retained row plane for throwaway output.
        eprintln!("index shared-storage gate: warming projection, selection and reindex");
        let runtime = tokio::runtime::Builder::new_current_thread().build()?;
        for i in 0..2 {
            let slot = &mut slots[i];
            slot.select_window(RowWindow::rows(32..32 + counts[i]))?;
            let requests = [SelectionRequest { proposal: &proposals[i], positions: &tokens[i] }];
            let warm_chain = chain::StageChain::new(&library)?;
            warm_chain.set_device_order(true);
            for layer in 0..2 {
                runtime.block_on(warm_chain.handle().scope(async {
                    slot.query.rebind(&weights.weights[3 + layer])?;
                    // SAFETY: upstream queries and warm-up outputs are retained.
                    unsafe { slot.query.enqueue_attention(&queries[layer][i].output()?)?; }
                    ensure!(slot.query.poll_pending()?, "warm projection blocked");
                    let query = slot.query.output()?;
                    if layer == 0 {
                        unsafe { slot.source.enqueue_selection(&query, &requests, None)?; }
                        ensure!(slot.source.poll_pending()?, "warm selection blocked");
                    } else {
                        let source = slot.source.output()?;
                        let follower = slot.reindex.as_mut().unwrap();
                        unsafe { follower.enqueue_selection(&query, &requests, Some(&source))?; }
                        ensure!(follower.poll_pending()?, "warm reindex blocked");
                    }
                    anyhow::Ok(())
                }))?;
            }
            warm_chain.drain()?;
            slot.restart()?;
            slot.select_window(RowWindow::rows(i * 16..i * 16 + counts[i]))?;
        }
        let chains = [chain::StageChain::new(&library)?, chain::StageChain::new(&library)?];
        for chain in &chains { chain.set_device_order(true); }
        let gate_stream = LoadStream { library: &library, raw: library.cuda_stream_create()? };
        // SAFETY: live stream retained until all chains drain; gate release is RAII.
        let gate = unsafe { cuteafd_ffi::test_support::CudaStreamGate::new(&library, gate_stream.raw)? };
        eprintln!("index shared-storage gate: closed");
        for chain in &chains {
            runtime.block_on(chain.handle().scope(async {
                // SAFETY: mark an initial dependency behind the still-closed gate.
                unsafe { chain::join(&library, gate_stream.raw)?;
                    chain::finish(&library, gate_stream.raw)?; }
                anyhow::Ok(())
            }))?;
        }
        for layer in 0..2 {
            // Both units project before either selection is submitted: selection
            // reads another stream's outputs, not the next unit's scratch.
            for i in 0..2 {
                eprintln!("index shared-storage gate: enqueue projection layer={layer} unit={i}");
                runtime.block_on(chains[i].handle().scope(async {
                    slots[i].query.rebind(&weights.weights[3 + layer])?;
                    // SAFETY: upstream queries are complete and retained.
                    unsafe { slots[i].query.enqueue_attention(&queries[layer][i].output()?)?; }
                    ensure!(slots[i].query.poll_pending()?, "projection blocked on the host");
                    anyhow::Ok(())
                }))?;
            }
            for i in 0..2 {
                eprintln!("index shared-storage gate: enqueue selection layer={layer} unit={i}");
                runtime.block_on(chains[i].handle().scope(async {
                    let slot = &mut slots[i];
                    let query = slot.query.output()?;
                    let requests = [SelectionRequest { proposal: &proposals[i], positions: &tokens[i] }];
                    if layer == 0 {
                        // SAFETY: query and proposals survive both queued units.
                        unsafe { slot.source.enqueue_selection(&query, &requests, None)?; }
                        ensure!(slot.source.poll_pending()?, "source selection blocked on the host");
                    } else {
                        let source = slot.source.output()?;
                        let follower = slot.reindex.as_mut().unwrap();
                        // SAFETY: source candidates are unit-disjoint retained planes.
                        unsafe { follower.enqueue_selection(&query, &requests, Some(&source))?; }
                        ensure!(follower.poll_pending()?, "reindex blocked on the host");
                    }
                    anyhow::Ok(())
                }))?;
            }
        }
        eprintln!("index shared-storage gate: releasing");
        drop(gate);
        for chain in &chains { chain.drain()?; }
        eprintln!("index shared-storage gate: drained; comparing private references");
        assert_ne!(slots[0].query.output()?.packed.ptr, slots[1].query.output()?.packed.ptr);
        assert_ne!(slots[0].source.output()?.selected.ptr, slots[1].source.output()?.selected.ptr);
        assert_ne!(slots[0].source.output()?.candidate_blocks().unwrap().ptr,
            slots[1].source.output()?.candidate_blocks().unwrap().ptr);
        let read = |buffer: cuteafd_ffi::CuteafdDeviceBuffer| -> Result<Vec<u8>> {
            let mut bytes = vec![0; buffer.bytes]; library.copy_d2h(&mut bytes, buffer)?; Ok(bytes)
        };
        for i in 0..2 {
            let mut private = IndexLane::new(&weights, 16, usize::MAX)?;
            let requests = [SelectionRequest { proposal: &proposals[i], positions: &tokens[i] }];
            for layer in 0..2 {
                private.query.rebind(&weights.weights[3 + layer])?;
                // SAFETY: all inputs complete and private storage is exclusive.
                unsafe { private.query.execute_attention(&queries[layer][i].output()?)?; }
                if layer == 0 {
                    unsafe { private.source.execute(&private.query.output()?, &requests, None)?; }
                } else {
                    unsafe { private.reindex.as_mut().unwrap().execute(&private.query.output()?,
                        &requests, Some(&private.source.output()?))?; }
                }
            }
            for (actual, expected) in [
                (slots[i].query.output()?.packed, private.query.output()?.packed),
                (slots[i].query.output()?.scales, private.query.output()?.scales),
                (slots[i].query.output()?.head_weights, private.query.output()?.head_weights),
                (slots[i].source.output()?.selected, private.source.output()?.selected),
                (slots[i].source.output()?.candidate_blocks().unwrap(), private.source.output()?.candidate_blocks().unwrap()),
                (slots[i].reindex.as_ref().unwrap().output()?.selected, private.reindex.as_ref().unwrap().output()?.selected),
            ] {
                assert_eq!(read(actual)?, read(expected)?, "unit {i} shared/private mismatch");
            }
        }
        Ok(())
    }
}
