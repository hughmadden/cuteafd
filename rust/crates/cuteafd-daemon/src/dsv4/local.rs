//! Complete routed-expert layers resident on the coordinator GPU.
//!
//! Layers `0..count` skip the Spark exchange: their experts run on the RTX
//! through the geometry's `rtx_backbone` kernels (`cuteafd_{family}_local_*`),
//! reading the same FP8 K32 wire rows and host routes the Sparks would get,
//! and the local reducer adds the shared expert.
use crate::v41_experts::{ExpertLayer, ExpertWeights};
use crate::v41_memory::DeviceAllocation;
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{NativeLibrary, V41ExpertKernel, V41ExpertLaunchArgs, V41LocalExpertReducer, V41_EXPERT_POINTER_COUNT};
use cuteafd_loader::OfficialV41Catalog;
use std::ffi::c_void;

/// Capacities the exporter compiles (rows per launch).
const CAPACITIES: [u32; 6] = [1, 16, 80, 256, 1024, 4096];

struct State<'a> {
    kernel: V41ExpertKernel<'a>,
    slots: [*mut c_void; V41_EXPERT_POINTER_COUNT],
}

pub(crate) struct LocalExperts<'a> {
    library: &'a NativeLibrary,
    layers: Vec<ExpertWeights<'a>>,
    states: Vec<State<'a>>,
    _scratch: DeviceAllocation<'a>,
    reducer: V41LocalExpertReducer<'a>,
    ids: DeviceAllocation<'a>,
    routing: DeviceAllocation<'a>,
    pub output: DeviceAllocation<'a>,
    topk: usize,
}

impl<'a> LocalExperts<'a> {
    /// Loads layers `0..` while they fit in `budget` bytes (leaving room for
    /// the kernels' workspace), up to `max_layers`.
    pub fn load(
        library: &'a NativeLibrary,
        catalog: &OfficialV41Catalog,
        max_layers: usize,
        max_rows: usize,
        budget: usize,
        stream: *mut c_void,
    ) -> Result<Option<Self>> {
        if max_layers == 0 {
            return Ok(None);
        }
        let shape = *catalog.routed_experts();
        let capacities: Vec<u32> = CAPACITIES.iter().copied().filter(|&c| c as usize <= max_rows.max(1))
            .chain(CAPACITIES.iter().copied().find(|&c| c as usize >= max_rows)).collect();
        let mut states = Vec::new();
        let mut scratch_bytes = 0usize;
        for &capacity in &capacities {
            let kernel = library.v41_local_expert_kernel(capacity)
                .context("coordinator expert kernels for this geometry are not in the library (CUTEAFD_*_EXPERT_FAMILIES=<family>:rtx_backbone)")?;
            scratch_bytes = scratch_bytes.max(usize::try_from(kernel.info().scratch_bytes)?);
            states.push(State { kernel, slots: [std::ptr::null_mut(); V41_EXPERT_POINTER_COUNT] });
        }
        let workspace = scratch_bytes + max_rows * (shape.hidden * 2 + shape.topk * 8);
        ensure!(budget > workspace, "local experts need {workspace} workspace bytes, budget is {budget}");
        let mut remaining = budget - workspace;
        tracing::info!(budget, workspace, scratch_bytes, "loading coordinator expert layers");
        let mut layers = Vec::new();
        for layer in 0..max_layers.min(shape.layers) {
            let plan = ExpertWeights::plan(library, catalog, ExpertLayer::BackboneFull { layer })?;
            if plan.peak_device_bytes()? > remaining {
                break;
            }
            let weights = ExpertWeights::load(library, catalog, ExpertLayer::BackboneFull { layer }, remaining)
                .with_context(|| format!("coordinator expert layer {layer} with {remaining} bytes left"))?;
            remaining -= weights.budget().resident_bytes;
            tracing::debug!(layer, resident = weights.budget().resident_bytes, remaining, "coordinator expert layer resident");
            layers.push(weights);
        }
        if layers.is_empty() {
            return Ok(None);
        }
        let scratch = DeviceAllocation::new(library, scratch_bytes.max(256))?;
        for state in &mut states {
            // SAFETY: the arena is exclusively owned and sized for every variant.
            unsafe {
                state.kernel.bind_scratch(scratch.buffer.ptr, scratch.buffer.bytes as u64, &mut state.slots)?;
                state.kernel.initialize_scratch(scratch.buffer.ptr, scratch.buffer.bytes as u64, stream)?;
                library.cuda_stream_synchronize(stream)?;
            }
        }
        Ok(Some(Self {
            library,
            layers,
            states,
            _scratch: scratch,
            reducer: library.v41_local_expert_reducer()?,
            ids: DeviceAllocation::new(library, max_rows * shape.topk * 4)?,
            routing: DeviceAllocation::new(library, max_rows * shape.topk * 4)?,
            output: DeviceAllocation::new(library, max_rows * shape.hidden * 2)?,
            topk: shape.topk,
        }))
    }

    pub fn layers(&self) -> usize {
        self.layers.len()
    }

    /// Runs layer `layer`'s experts for `rows` wire rows with host routes and
    /// writes routed + shared into [`Self::output`].
    ///
    /// # Safety
    /// `wire` holds `rows` FP8 K32 rows and `shared` `rows` BF16 rows on this
    /// device, both complete in stream order and unchanged until it drains.
    pub unsafe fn run(
        &mut self,
        layer: usize,
        rows: usize,
        wire: *mut c_void,
        ids: &[u32],
        weights: &[f32],
        shared: *mut c_void,
        stream: *mut c_void,
    ) -> Result<()> {
        ensure!(layer < self.layers.len() && ids.len() == rows * self.topk && weights.len() == ids.len(),
            "local expert layer {layer} or routes out of range");
        let bytes = |values: &[u32]| values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>();
        let weight_bits: Vec<u32> = weights.iter().map(|w| w.to_bits()).collect();
        self.library.copy_h2d(cuteafd_ffi::CuteafdDeviceBuffer { bytes: ids.len() * 4, ..self.ids.buffer }, &bytes(ids))?;
        self.library.copy_h2d(cuteafd_ffi::CuteafdDeviceBuffer { bytes: ids.len() * 4, ..self.routing.buffer }, &bytes(&weight_bits))?;
        let state = self.states.iter_mut().find(|s| s.kernel.info().capacity_rows as usize >= rows)
            .context("no local expert capacity for this many rows")?;
        self.layers[layer].bind(&state.kernel, &mut state.slots)?;
        state.slots[0] = wire;
        state.slots[1] = self.ids.buffer.ptr;
        state.slots[2] = self.routing.buffer.ptr;
        let info = state.kernel.info();
        let args = V41ExpertLaunchArgs {
            tensors: state.slots,
            num_tokens: rows as i32,
            max_rows: info.max_rows,
            scatter_rows: (rows * self.topk) as i32,
            rows_padded: info.rows_padded,
            max_tasks: info.max_tasks,
            max_phys_tiles: info.max_phys_tiles,
            max_active_clusters: info.max_active_clusters,
            stream,
        };
        // SAFETY: slots are bound to resident weights, the initialized arena
        // and this call's inputs; the stream orders everything.
        unsafe {
            state.kernel.launch(&args)?;
            self.reducer.finish(state.slots[41].cast(), shared.cast(), self.output.buffer.ptr.cast(),
                rows as u32, state.kernel.accumulates_tokens(), stream)
        }
    }
}
