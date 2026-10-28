//! Shared shifted mHC boundaries for backbone and dSpark sublayers.
use crate::shared::memory::{DeviceAllocation, LoadStream};
use crate::families::deepseek_v41::v41_tensors::NativeRtxTensors;
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary, V41Hc};
use std::ffi::c_void;

pub(crate) struct HcBinding<'w, 'a> {
    weights: &'w NativeRtxTensors<'a>,
    names: [String; 4],
}

/// One mHC boundary's stream and row planes (`shared::memory::row_window`
/// contract: every plane is cross-hook state, so units own disjoint rows; there
/// is no scratch). Prefill slots share one per GPU; decode lanes own theirs.
pub(crate) struct HcStorage<'library> {
    stream: LoadStream<'library>,
    residual: DeviceAllocation<'library>,
    incoming_pre: DeviceAllocation<'library>,
    next_pre: DeviceAllocation<'library>,
    post: DeviceAllocation<'library>,
    comb: DeviceAllocation<'library>,
    collapsed: DeviceAllocation<'library>,
    normalized: DeviceAllocation<'library>,
    sublayer: DeviceAllocation<'library>,
    output: DeviceAllocation<'library>,
    rows: usize,
}
impl<'library> HcStorage<'library> {
    pub(crate) fn new(library: &'library NativeLibrary, rows: usize) -> Result<std::rc::Rc<Self>> {
        ensure!((1..=crate::shared::memory::row_window::MAX_RESERVE_ROWS).contains(&rows), "invalid mHC storage rows {rows}");
        Ok(std::rc::Rc::new(Self {
            stream: LoadStream { library, raw: library.cuda_stream_create()? },
            residual: DeviceAllocation::new(library, rows * 40960)?,
            incoming_pre: DeviceAllocation::new(library, rows * 16)?,
            next_pre: DeviceAllocation::new(library, rows * 16)?,
            post: DeviceAllocation::new(library, rows * 16)?,
            comb: DeviceAllocation::new(library, rows * 64)?,
            collapsed: DeviceAllocation::new(library, rows * 10240)?,
            normalized: DeviceAllocation::new(library, rows * 10240)?,
            sublayer: DeviceAllocation::new(library, rows * 10240)?,
            output: DeviceAllocation::new(library, rows * 40960)?,
            rows,
        }))
    }
    /// Bytes per row of every plane together.
    pub(crate) const ROW_BYTES: usize = 112752;
}
impl Drop for HcStorage<'_> {
    fn drop(&mut self) {
        // SAFETY: the stream is owned here; its queued work reads only these planes.
        if let Err(error) = unsafe { self.stream.library.cuda_stream_synchronize(self.stream.raw) } {
            tracing::error!(%error, "draining mHC storage");
        }
    }
}

pub(crate) struct HcSublayer<'weights, 'library> {
    storage: std::rc::Rc<HcStorage<'library>>,
    kernel: V41Hc<'library>,
    weights: &'weights NativeRtxTensors<'library>,
    retained_weights: Vec<&'weights NativeRtxTensors<'library>>,
    names: [String; 4],
    /// This unit's rows in every plane (`RowWindow::NONE`: from row 0).
    window: crate::shared::memory::row_window::RowWindow,
    capacity: usize,
    begun: Option<usize>,
    ready: Option<usize>,
    pending: Option<usize>,
}
impl<'weights, 'library> HcSublayer<'weights, 'library> {
    pub fn new(
        library: &'library NativeLibrary,
        weights: &'weights NativeRtxTensors<'library>,
        names: [String; 4],
        capacity: usize,
        budget: usize,
    ) -> Result<Self> {
        ensure!(
            Self::device_bytes(capacity)? <= budget,
            "mHC sublayer exceeds budget"
        );
        for name in &names {
            weights.get(name)?;
        }
        Self::on_storage(library, weights, names, HcStorage::new(library, capacity)?, capacity)
    }
    /// A slot's boundary over shared `storage` (one per GPU for prefill slots):
    /// units of at most `unit` rows, each in the window `select_window` sets.
    pub(crate) fn on_storage(
        library: &'library NativeLibrary,
        weights: &'weights NativeRtxTensors<'library>,
        names: [String; 4],
        storage: std::rc::Rc<HcStorage<'library>>,
        unit: usize,
    ) -> Result<Self> {
        ensure!(unit >= 1 && unit <= storage.rows, "mHC unit of {unit} rows exceeds its {}-row storage", storage.rows);
        for name in &names {
            weights.get(name)?;
        }
        Ok(HcSublayer {
            storage,
            kernel: library.v41_hc()?,
            weights,
            retained_weights: vec![weights],
            names,
            window: crate::shared::memory::row_window::RowWindow::NONE,
            capacity: unit,
            begun: None,
            ready: None,
            pending: None,
        })
    }
    /// Validate and drain before changing either boundary in a reusable block.
    pub(crate) fn prepare_binding(
        &self,
        weights: &'weights NativeRtxTensors<'library>,
        names: [String; 4],
    ) -> Result<HcBinding<'weights, 'library>> {
        ensure!(self.begun.is_none() && self.pending.is_none(),
            "cannot rebind an unfinished mHC boundary");
        ensure!(self.retained_weights.len() < 40
            || self.retained_weights.iter().any(|old| std::ptr::eq(*old, weights)),
            "mHC binding bank exceeds 40 weight owners");
        for (name, old) in names.iter().zip(&self.names) {
            let value = weights.get(name)?;
            ensure!(value.bytes == self.weights.get(old)?.bytes
                && value.device_id == self.storage.residual.buffer.device_id,
                "mHC rebound weight extent or device differs");
        }
        // A stage chain orders all later enqueues after the queued boundary work;
        // rebinding changes only the weights those later enqueues will use.
        if !crate::shared::memory::chain::active() { self.storage.stream.require_complete()?; }
        Ok(HcBinding { weights, names })
    }
    /// # Safety
    /// External streams using this boundary are drained. Captured external
    /// graphs must be recaptured before use with the new weight binding.
    pub(crate) unsafe fn install_binding(&mut self, binding: HcBinding<'weights, 'library>) {
        self.invalidate();
        if !self.retained_weights.iter().any(|old| std::ptr::eq(*old, binding.weights)) {
            self.retained_weights.push(binding.weights);
        }
        self.weights = binding.weights;
        self.names = binding.names;
    }
}
impl HcSublayer<'_, '_> {
    pub fn device_bytes(capacity: usize) -> Result<usize> {
        ensure!((1..=4096).contains(&capacity), "invalid mHC capacity");
        Ok(capacity * 112752)
    }
    /// Stable BF16 residual [capacity,4,5120] and incoming FP32 pre [capacity,4].
    /// The first stage starts with one-hot [1,0,0,0]; subsequent boundaries receive
    /// the preceding sublayer's next_pre. Never free or retain after owner drop.
    pub fn inputs(&self) -> [CuteafdDeviceBuffer; 2] {
        [self.window.apply(self.storage.residual.buffer, 40960), self.window.apply(self.storage.incoming_pre.buffer, 16)]
    }
    /// Destination for the attention/FFN result; finish producer writes before finish.
    pub fn sublayer_result(&self) -> CuteafdDeviceBuffer {
        self.window.apply(self.storage.sublayer.buffer, 10240)
    }
    /// A containing owner may enqueue a boundary chain on this stream, provided
    /// it drains the stream before releasing any participating sublayer borrow.
    pub(crate) fn stream_raw(&self) -> *mut c_void {
        self.storage.stream.raw
    }
    pub(crate) async fn wait_chain(&self) -> Result<()> {
        unsafe { crate::shared::memory::chain::finish_cooperative(&self.storage.stream).await }
    }
    fn synchronize(&self) -> Result<()> {
        unsafe { self.storage.stream.library.cuda_stream_synchronize(self.storage.stream.raw) }
    }
    /// # Safety
    /// Both inputs must be initialized on this device with completed producer
    /// writes. Serialize use; do not overwrite residual/coefficient storage until finish.
    pub unsafe fn begin(&mut self, rows: usize) -> Result<CuteafdDeviceBuffer> {
        let launched = unsafe { self.enqueue_begin(rows, None, self.storage.stream.raw) };
        let drained = self.synchronize();
        match launched.and_then(|output| drained.map(|()| output)) {
            Ok(output) => Ok(output),
            Err(error) => {
                self.invalidate();
                Err(error)
            }
        }
    }
    /// Selects this slot's unit rows in the shared planes (row_window
    /// contract), at rest: `RowWindow::NONE` uses rows from 0.
    pub(crate) fn select_window(&mut self, window: crate::shared::memory::row_window::RowWindow) -> Result<()> {
        crate::shared::memory::row_window::require_rest(self.begun.is_some() || self.pending.is_some(), "mHC sublayer", "work is begun or pending")?;
        if let Some(range) = window.range() {
            ensure!(range.end <= self.storage.rows && range.len() <= self.capacity,
                "mHC window {range:?} outside {} rows or wider than the {}-row unit", self.storage.rows, self.capacity);
        }
        self.window = window;
        Ok(())
    }
    pub(crate) fn invalidate(&mut self) {
        self.begun = None;
        self.ready = None;
        self.pending = None;
    }
    /// Caller owns the external stream and must drain it before releasing this
    /// borrow, including on error; an external normalized output must be distinct.
    pub(crate) unsafe fn enqueue_begin(
        &mut self,
        rows: usize,
        normalized: Option<CuteafdDeviceBuffer>,
        stream: *mut c_void,
    ) -> Result<CuteafdDeviceBuffer> {
        self.invalidate();
        let normalized = normalized.unwrap_or(self.window.apply(self.storage.normalized.buffer, 10240));
        ensure!(
            !normalized.ptr.is_null() && normalized.bytes >= rows.saturating_mul(10240),
            "mHC normalized output is too small"
        );
        ensure!(
            rows > 0 && rows <= self.capacity && self.window.admits(rows),
            "mHC rows exceed capacity or the unit's window"
        );
        unsafe {
            if !self.kernel.try_begin(
                self.window.apply(self.storage.residual.buffer, 40960),
                self.weights.get(&self.names[0])?,
                self.weights.get(&self.names[1])?,
                self.weights.get(&self.names[2])?,
                self.window.apply(self.storage.incoming_pre.buffer, 16),
                self.weights.get(&self.names[3])?,
                self.window.apply(self.storage.next_pre.buffer, 16),
                self.window.apply(self.storage.post.buffer, 16),
                self.window.apply(self.storage.comb.buffer, 64),
                normalized,
                self.window.apply(self.storage.collapsed.buffer, 10240),
                rows,
                stream,
            )? {
                // Mixes finish reading partials before pre writes collapsed on this
                // same stream. Reuse its storage without adding a serving allocation.
                self.kernel.mixes_workspace(
                    self.window.apply(self.storage.residual.buffer, 40960),
                    self.weights.get(&self.names[0])?,
                    self.weights.get(&self.names[1])?,
                    self.weights.get(&self.names[2])?,
                    self.window.apply(self.storage.next_pre.buffer, 16),
                    self.window.apply(self.storage.post.buffer, 16),
                    self.window.apply(self.storage.comb.buffer, 64),
                    self.window.apply(self.storage.collapsed.buffer, 10240),
                    rows,
                    stream,
                )?;
                // The newly generated pre belongs to the NEXT sublayer.
                self.kernel.pre(
                    self.window.apply(self.storage.residual.buffer, 40960),
                    self.window.apply(self.storage.incoming_pre.buffer, 16),
                    self.window.apply(self.storage.collapsed.buffer, 10240),
                    rows,
                    stream,
                )?;
                self.storage.stream.library.cuda_ds4_rmsnorm_bf16_rne_async(
                    self.window.apply(self.storage.collapsed.buffer, 10240),
                    self.weights.get(&self.names[3])?,
                    normalized,
                    rows as i32,
                    5120,
                    1e-20,
                    stream,
                )?;
            }
        }
        self.begun = Some(rows);
        let mut result = normalized;
        result.bytes = rows * 10240;
        Ok(result)
    }
    /// # Safety
    /// The BF16 sublayer result must be initialized for the rows returned by begin
    /// and all producer writes complete. Residual/coefficients must remain unchanged.
    pub unsafe fn finish(&mut self) -> Result<[CuteafdDeviceBuffer; 2]> {
        let launched = unsafe { self.enqueue_finish(None, self.storage.stream.raw) };
        let drained = self.synchronize();
        if let Err(error) = launched.and(drained) {
            self.invalidate();
            return Err(error);
        }
        unsafe { self.complete() }
    }
    /// External result producer must be ordered on stream; drain before release.
    pub(crate) unsafe fn enqueue_finish(
        &mut self,
        result: Option<CuteafdDeviceBuffer>,
        stream: *mut c_void,
    ) -> Result<()> {
        self.ready = None;
        self.pending = None;
        let rows = self.begun.take().context("mHC sublayer was not begun")?;
        let result = result.unwrap_or(self.window.apply(self.storage.sublayer.buffer, 10240));
        unsafe {
            self.kernel.post(
                result,
                self.window.apply(self.storage.residual.buffer, 40960),
                self.window.apply(self.storage.post.buffer, 16),
                self.window.apply(self.storage.comb.buffer, 64),
                self.window.apply(self.storage.output.buffer, 40960),
                rows,
                stream,
            )?;
        }
        self.pending = Some(rows);
        Ok(())
    }
    /// # Safety
    /// The stream used by enqueue_finish must have completed successfully.
    pub(crate) unsafe fn complete(&mut self) -> Result<[CuteafdDeviceBuffer; 2]> {
        self.ready = Some(self.pending.take().context("mHC post was not submitted")?);
        self.output()
    }
    /// # Safety
    /// A graph containing this boundary's post for exactly rows has completed.
    pub(crate) unsafe fn complete_replay(
        &mut self,
        rows: usize,
    ) -> Result<[CuteafdDeviceBuffer; 2]> {
        ensure!(rows > 0 && rows <= self.capacity && !self.window.is_windowed(), "invalid mHC replay rows");
        self.pending = Some(rows);
        unsafe { self.complete() }
    }
    #[cfg(test)]
    pub(crate) fn rearm_finish_for_test(&mut self, rows: usize) {
        self.begun = Some(rows);
        self.pending = None;
        self.ready = None;
    }
    /// # Safety
    /// A containing graph executes the same begin operations for these rows.
    /// No pending GPU work may reuse this owner from another stream.
    pub(crate) unsafe fn graph_begin_state(&mut self, rows: usize) -> Result<()> {
        ensure!(rows > 0 && rows <= self.capacity && !self.window.is_windowed(), "invalid mHC graph rows");
        self.invalidate();
        self.begun = Some(rows);
        Ok(())
    }
    /// # Safety
    /// The containing graph executes this owner's post for exactly these rows.
    pub(crate) unsafe fn graph_post_state(&mut self, rows: usize) -> Result<()> {
        ensure!(
            self.begun == Some(rows),
            "mHC graph post has no matching begin"
        );
        self.begun = None;
        self.ready = None;
        self.pending = Some(rows);
        Ok(())
    }
    pub(crate) fn graph_identity(&self) -> [usize; 3] {
        [
            self.weights as *const _ as usize,
            self.storage.residual.buffer.ptr as usize,
            self.storage.normalized.buffer.ptr as usize,
        ]
    }
    pub(crate) fn normalized_storage(&self, rows: usize) -> CuteafdDeviceBuffer {
        let mut value = self.window.apply(self.storage.normalized.buffer, 10240);
        value.bytes = rows * 10240;
        value
    }
    /// Raw destinations for a containing owner that serializes GPU consumers.
    pub(crate) fn output_storage(&self) -> [CuteafdDeviceBuffer; 2] {
        [self.window.apply(self.storage.output.buffer, 40960), self.window.apply(self.storage.next_pre.buffer, 16)]
    }
    /// BF16 residual [rows,4,5120] and FP32 next_pre [rows,4], borrowed until reuse.
    pub fn output(&self) -> Result<[CuteafdDeviceBuffer; 2]> {
        let rows = self.ready.context("mHC sublayer output is incomplete")?;
        let mut residual = self.window.apply(self.storage.output.buffer, 40960);
        residual.bytes = rows * 40960;
        let mut pre = self.window.apply(self.storage.next_pre.buffer, 16);
        pre.bytes = rows * 16;
        Ok([residual, pre])
    }
}
impl Drop for HcSublayer<'_, '_> {
    fn drop(&mut self) {
        if let Err(error) = self.synchronize() {
            tracing::error!(%error,"draining mHC sublayer");
        }
    }
}
