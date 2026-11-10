//! Shared native FP8 SwiGLU execution for backbone and dSpark weights.
use crate::shared::memory::{DeviceAllocation, LoadStream};
use crate::families::deepseek_v41::v41_tensors::NativeRtxTensors;
use anyhow::{ensure, Result};
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary, V41Fp8Plan, V41SharedSwiGlu};
use std::ffi::c_void;
use std::rc::Rc;

pub(crate) struct SharedFfn<'weights, 'library> {
    _weights: &'weights NativeRtxTensors<'library>,
    scales: &'weights [DeviceAllocation<'library>],
    tensors: [CuteafdDeviceBuffer; 3],
    storage: Rc<SharedFfnStorage<'library>>,

}
/// Plain unit-cap scratch, used entirely on the containing shared wave's
/// single shared stream. No intermediate survives the FFN hook. Standalone
/// dSpark owners retain private storage and their existing external stream.
pub(crate) struct SharedFfnStorage<'library> {
    up_kernel: V41Fp8Plan<'library>,
    down_kernel: V41Fp8Plan<'library>,
    activation: V41SharedSwiGlu<'library>,
    up_scratch: DeviceAllocation<'library>,
    down_scratch: DeviceAllocation<'library>,
    alpha: DeviceAllocation<'library>,
    gate: DeviceAllocation<'library>,
    up: DeviceAllocation<'library>,
    intermediate: DeviceAllocation<'library>,
    capacity: u32,
}
impl<'library> SharedFfnStorage<'library> {
    fn allocation_sizes(up: usize, down: usize, unit: u32) -> [usize; 6] {
        [up, down, 4, unit as usize * 4608, unit as usize * 4608, unit as usize * 4608]
    }
    pub(crate) fn storage_bytes(library: &NativeLibrary, unit: u32) -> Result<usize> {
        ensure!((1..=4096).contains(&unit), "invalid shared FFN unit");
        Ok(Self::allocation_sizes(
            library.v41_fp8_matrix_plan_info(unit, 5120, 2304)?.scratch_bytes as usize,
            library.v41_fp8_matrix_plan_info(unit, 2304, 5120)?.scratch_bytes as usize, unit)
            .into_iter().sum())
    }
    pub(crate) fn new(library: &'library NativeLibrary, unit: u32) -> Result<Rc<Self>> {
        Self::storage_bytes(library, unit)?;
        let up_kernel = library.v41_fp8_matrix_plan(unit, 5120, 2304)?;
        let down_kernel = library.v41_fp8_matrix_plan(unit, 2304, 5120)?;
        let [up, down, alpha, gate, intermediate_up, intermediate] = Self::allocation_sizes(
            up_kernel.info().scratch_bytes as usize, down_kernel.info().scratch_bytes as usize, unit);
        let value = Self {
            up_scratch: DeviceAllocation::new(library, up)?,
            down_scratch: DeviceAllocation::new(library, down)?,
            up_kernel, down_kernel,
            activation: library.v41_shared_swiglu()?,
            alpha: DeviceAllocation::new(library, alpha)?,
            gate: DeviceAllocation::new(library, gate)?,
            up: DeviceAllocation::new(library, intermediate_up)?,
            intermediate: DeviceAllocation::new(library, intermediate)?,
            capacity: unit,
        };
        let stream = LoadStream { library, raw: library.cuda_stream_create()? };
        // SAFETY: initialization only touches owned buffers; drain on failure too.
        let initialized = unsafe {
            value.up_kernel.initialize_scratch(value.up_scratch.buffer, value.alpha.buffer, stream.raw)
                .and_then(|()| value.down_kernel.initialize_scratch(value.down_scratch.buffer, value.alpha.buffer, stream.raw))
        };
        initialized.and(unsafe { library.cuda_stream_synchronize(stream.raw) })?;
        Ok(Rc::new(value))
    }
}
impl<'weights, 'library> SharedFfn<'weights, 'library> {
    /// # Safety
    /// All consumers of this workspace are drained. Any cached external graphs
    /// retain their original weights/scales and are selected by that binding.
    pub(crate) unsafe fn rebind(
        &mut self,
        weights: &'weights NativeRtxTensors<'library>,
        prefix: &str,
        scales: &'weights [DeviceAllocation<'library>],
    ) -> Result<()> {
        ensure!(scales.len() == 3, "shared FFN requires three packed scales");
        let tensors = [
            weights.get(&format!("{prefix}.w1.weight"))?,
            weights.get(&format!("{prefix}.w3.weight"))?,
            weights.get(&format!("{prefix}.w2.weight"))?,
        ];
        for i in 0..3 {
            ensure!(
                tensors[i].bytes == self.tensors[i].bytes
                    && tensors[i].device_id == self.storage.gate.buffer.device_id
                    && scales[i].buffer.bytes == self.scales[i].buffer.bytes
                    && scales[i].buffer.device_id == self.storage.gate.buffer.device_id,
                "shared FFN rebound weight extent or device differs"
            );
        }
        self._weights = weights;
        self.scales = scales;
        self.tensors = tensors;
        Ok(())
    }
    pub fn new(
        library: &'library NativeLibrary,
        weights: &'weights NativeRtxTensors<'library>,
        prefix: &str,
        scales: &'weights [DeviceAllocation<'library>],
        capacity: u32,
        budget: usize,
    ) -> Result<Self> {
        ensure!(scales.len() == 3, "shared FFN requires three packed scales");
        ensure!(
            Self::device_bytes(library, capacity)? <= budget,
            "shared FFN exceeds budget"
        );
        Self::on_storage(library, weights, prefix, scales, SharedFfnStorage::new(library, capacity)?)
    }
    pub(crate) fn on_storage(
        _library: &'library NativeLibrary,
        weights: &'weights NativeRtxTensors<'library>,
        prefix: &str,
        scales: &'weights [DeviceAllocation<'library>],
        storage: Rc<SharedFfnStorage<'library>>,
    ) -> Result<Self> {
        ensure!(scales.len() == 3, "shared FFN requires three packed scales");
        Ok(Self { _weights: weights, scales,
            tensors: [weights.get(&format!("{prefix}.w1.weight"))?,
                weights.get(&format!("{prefix}.w3.weight"))?,
                weights.get(&format!("{prefix}.w2.weight"))?], storage })
    }
}
impl SharedFfn<'_, '_> {
    pub fn device_bytes(library: &NativeLibrary, capacity: u32) -> Result<usize> {
        SharedFfnStorage::storage_bytes(library, capacity)
    }
    pub fn intermediates(&self) -> [CuteafdDeviceBuffer; 3] {
        [self.storage.gate.buffer, self.storage.up.buffer, self.storage.intermediate.buffer]
    }
    /// Enqueue on the owning expert wave stream; the caller must drain even on
    /// failure before releasing this exclusive borrow of scratch.
    pub(crate) unsafe fn enqueue(
        &mut self,
        input: CuteafdDeviceBuffer,
        output: CuteafdDeviceBuffer,
        rows: u32,
        stream: *mut c_void,
    ) -> Result<()> {
        ensure!(
            rows > 0 && rows <= self.storage.capacity,
            "shared FFN rows exceed capacity"
        );
        unsafe { self.enqueue_up(input, rows, stream)?; self.enqueue_down(output, rows, stream) }
    }
    pub(crate) unsafe fn enqueue_up(&self, input: CuteafdDeviceBuffer, rows: u32, stream: *mut c_void) -> Result<()> {
        ensure!(rows > 0 && rows <= self.storage.capacity, "shared FFN rows exceed capacity");
        let scales = self.scales;
        unsafe {
            self.storage.up_kernel.launch(
                input,
                self.tensors[0],
                scales[0].buffer,
                self.storage.up_scratch.buffer,
                self.storage.alpha.buffer,
                self.storage.gate.buffer,
                rows,
                stream,
            )?;
            self.storage.up_kernel.launch(
                input,
                self.tensors[1],
                scales[1].buffer,
                self.storage.up_scratch.buffer,
                self.storage.alpha.buffer,
                self.storage.up.buffer,
                rows,
                stream,
            )?;
            self.storage.activation.launch(
                self.storage.gate.buffer,
                self.storage.up.buffer,
                self.storage.intermediate.buffer,
                rows,
                stream,
            )
        }
    }
    pub(crate) unsafe fn enqueue_down(&self, output: CuteafdDeviceBuffer, rows: u32, stream: *mut c_void) -> Result<()> {
        ensure!(rows > 0 && rows <= self.storage.capacity, "shared FFN rows exceed capacity");
        let scales = self.scales;
        unsafe {
            self.storage.down_kernel.launch(
                self.storage.intermediate.buffer,
                self.tensors[2],
                scales[2].buffer,
                self.storage.down_scratch.buffer,
                self.storage.alpha.buffer,
                output,
                rows,
                stream,
            )
        }
    }
}

#[cfg(test)]
mod storage_bytes_tests {
    use super::*;
    #[test]
    fn shared_ffn_storage_bytes_equal_planned_allocations() {
        for unit in [4096, 2752, 2048] {
            let sizes = SharedFfnStorage::allocation_sizes(123, 456, unit);
            assert_eq!(sizes.into_iter().sum::<usize>(), 123 + 456 + 4 + unit as usize * 13824);
        }
    }
}
