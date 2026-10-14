//! Native draft expert halves, with device-scoped ownership and fixed scratch.
use super::Layout;
use crate::shared::experts::layer::{ExpertLayer, ExpertWeights};
use crate::shared::memory::device::{Allocation, Device, DeviceOwner, Stream};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary, V41ExpertKernel, V41ExpertLaunchArgs};
use cuteafd_loader::OfficialV41Catalog;
use std::{ffi::c_void, rc::Rc};

pub(crate) struct Weights<'a> {
    pub device: Device<'a>,
    pub layout: Layout,
    stages: DeviceOwner<'a, Vec<ExpertWeights<'a>>>,
}
impl<'a> Weights<'a> {
    /// Alternate rank loads within each stage to reuse checkpoint pages. Every
    /// partially loaded rank retains its device owner on an early return.
    pub fn load_pair(
        devices: [Device<'a>; 2],
        catalog: &OfficialV41Catalog,
        budgets: [usize; 2],
        layout: Layout,
    ) -> Result<[Rc<Self>; 2]> {
        layout.validate()?;
        ensure!(
            devices[0].id != devices[1].id && std::ptr::eq(devices[0].library, devices[1].library),
            "invalid draft TP2 device pair"
        );
        // Admit the installed ABI before loading any checkpoint payload.
        Self::validate_info(devices[0].library.v41_dspark_tp2_expert_info(1)?, layout)?;
        let mut pair = [
            Self {
                device: devices[0],
                layout,
                stages: devices[0].own(|| Ok(Vec::with_capacity(layout.stages)))?,
            },
            Self {
                device: devices[1],
                layout,
                stages: devices[1].own(|| Ok(Vec::with_capacity(layout.stages)))?,
            },
        ];
        let mut remaining = budgets;
        for stage in 0..layout.stages {
            for rank in 0..2 {
                devices[rank].run(|| {
                    let weight = ExpertWeights::load(
                        devices[rank].library,
                        catalog,
                        ExpertLayer::DsparkTp2 { stage, rank },
                        remaining[rank],
                    )?;
                    remaining[rank] = remaining[rank]
                        .checked_sub(weight.budget().resident_bytes)
                        .context("draft TP2 weight budget overflow")?;
                    pair[rank].stages.push(weight);
                    Ok(())
                })?;
            }
        }
        Ok(pair.map(Rc::new))
    }
    fn validate_info(info: cuteafd_ffi::V41ExpertInfo, layout: Layout) -> Result<()> {
        let g = layout.geometry;
        ensure!(
            info.hidden_size == g.hidden
                && info.experts == g.experts
                && info.logical_intermediate == g.intermediate / 2
                && info.topk == g.topk
                && info.role == 4
                && info.input_dtype == 7,
            "installed draft kernel differs from caller geometry"
        );
        Ok(())
    }
    pub fn resident_bytes(&self) -> usize {
        self.stages.iter().map(|w| w.budget().resident_bytes).sum()
    }
}

struct State<'a> {
    kernel: V41ExpertKernel<'a>,
    slots: [*mut c_void; 44],
}
/// One lane's workspace for one rank. Enqueue borrows an external stream so the
/// containing draft chain can capture both ranks in its single CUDA graph.
/// That chain must drain its streams and destroy graphs before dropping this.
pub(crate) struct Wave<'a> {
    states: Vec<State<'a>>,
    _scratch: Allocation<'a>,
    weights: Rc<Weights<'a>>,
    capacity: u32,
}
impl<'a> Wave<'a> {
    fn capacities(capacity: u32) -> Result<Vec<u32>> {
        ensure!((1..=4096).contains(&capacity), "invalid draft TP2 capacity");
        let maximum = [1, 16, 80, 256, 1024, 4096]
            .into_iter()
            .find(|&v| v >= capacity)
            .unwrap();
        Ok([1, 16, 80, 256, 1024, 4096]
            .into_iter()
            .filter(|&v| v <= maximum)
            .collect())
    }
    pub fn device_bytes(library: &NativeLibrary, capacity: u32, layout: Layout) -> Result<usize> {
        layout.validate()?;
        Self::capacities(capacity)?
            .into_iter()
            .map(|c| {
                let info = library.v41_dspark_tp2_expert_info(c)?;
                Weights::validate_info(info, layout)?;
                Ok(usize::try_from(info.scratch_bytes)?)
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .max()
            .context("missing draft TP2 capacity")
    }
    pub fn new(weights: Rc<Weights<'a>>, capacity: u32) -> Result<Self> {
        let device = weights.device;
        let bytes = Self::device_bytes(device.library, capacity, weights.layout)?;
        let scratch = Allocation::new(device, bytes)?;
        let stream = Stream::new(device)?;
        let mut states = Vec::new();
        for cap in Self::capacities(capacity)? {
            let kernel = device.run(|| device.library.v41_dspark_tp2_expert_kernel(cap))?;
            let mut slots = [std::ptr::null_mut(); 44];
            let initialized = device.run(|| unsafe {
                kernel.bind_scratch(scratch.buffer.ptr, bytes as u64, &mut slots)?;
                kernel.initialize_scratch(scratch.buffer.ptr, bytes as u64, stream.raw)
            });
            let drained = stream.drain();
            initialized.and(drained)?;
            states.push(State { kernel, slots });
        }
        Ok(Self {
            states,
            _scratch: scratch,
            weights,
            capacity,
        })
    }
    /// Inputs and stream belong to this rank and remain alive until completion.
    /// Output borrows shared scratch and must be consumed before the next stage.
    /// The caller drains external streams on cancellation or partial enqueue.
    pub unsafe fn enqueue(
        &mut self,
        stage: usize,
        rows: u32,
        inputs: [CuteafdDeviceBuffer; 3],
        stream: *mut c_void,
    ) -> Result<CuteafdDeviceBuffer> {
        ensure!(
            stage < self.weights.layout.stages && rows > 0 && rows <= self.capacity,
            "draft TP2 stage or rows out of range"
        );
        let device = self.weights.device;
        for (buffer, width) in inputs.iter().zip(self.weights.layout.input_bytes()) {
            ensure!(
                buffer.device_id == device.id && buffer.bytes >= rows as usize * width,
                "draft TP2 input device or extent differs"
            );
        }
        device.run(|| unsafe {
            let state = self
                .states
                .iter_mut()
                .find(|s| s.kernel.info().capacity_rows >= rows)
                .unwrap();
            self.weights.stages[stage].bind(&state.kernel, &mut state.slots)?;
            for (slot, input) in state.slots[..3].iter_mut().zip(inputs) {
                *slot = input.ptr;
            }
            let info = state.kernel.info();
            state.kernel.launch(&V41ExpertLaunchArgs {
                tensors: state.slots,
                num_tokens: rows as i32,
                max_rows: info.max_rows,
                scatter_rows: rows as i32 * self.weights.layout.geometry.topk as i32,
                rows_padded: info.rows_padded,
                max_tasks: info.max_tasks,
                max_phys_tiles: info.max_phys_tiles,
                max_active_clusters: info.max_active_clusters,
                stream,
            })?;
            let mut output = self._scratch.buffer;
            output.ptr = state.slots[41];
            output.bytes = rows as usize * self.weights.layout.route_bytes();
            Ok(output)
        })
    }
}
