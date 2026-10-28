//! Native EXLR8 routed halves on the shared RTX paired-expert executor.
use super::native::RankCompletion;
use super::{ExpertInput, PartialDtype, Routes, RtxExpertLayer, RtxShard};
use crate::shared::experts::exlr8::{grouped_decode_from_row, view, Exlr8Execution, Exlr8PrefillActivations, Exlr8Weights};
use crate::shared::memory::device::{Allocation, Device, DeviceOwner, Event};
use anyhow::{bail, ensure, Context, Result};
use cuteafd_ffi::{programs::Programs, CuteafdDeviceBuffer};
use cuteafd_loader::plan::exlr8::residency::ExpertSet;
use std::{ffi::c_void, rc::Rc};

const DECODE_ROWS: usize = 64;
const TP2_UNITS: usize = 8;

#[cfg(test)]
mod copy_tests {
    #[test]
    fn all_decode_operands_use_the_shared_sm_copy_route() {
        use crate::shared::memory::device::sm_peer_copy_for;
        for rows in [1, 16, DECODE_ROWS] {
            for bytes in [rows * 6144 * 33 / 32, rows * 8 * 4, rows * 6144 * 2] {
                assert!(sm_peer_copy_for(true, bytes, DECODE_ROWS * 6144 * 2));
            }
        }
        assert!(!sm_peer_copy_for(true, DECODE_ROWS * 6144 * 2 + 1, DECODE_ROWS * 6144 * 2));
        assert!(!sm_peer_copy_for(true, 4096 * 6144 * 2, DECODE_ROWS * 6144 * 2));
        let source = include_str!("exlr8.rs");
        let enqueue = source.split("unsafe fn enqueue(").nth(1).unwrap();
        assert_eq!(enqueue.matches("copy_d2d_async(").count(), 1);
        assert!(enqueue.contains("sm_peer_copy_for(true, input.bytes, DECODE_ROWS * 6144 * 2)"));
        assert!(enqueue.contains("self.sm_copy.launch_rows("), "same-device SM copies require the local-capable entry point");
        assert!(enqueue.contains("copy(view(self.output.buffer, 0, bytes)?,"));
    }
    use super::DECODE_ROWS;
}

pub(crate) struct Exlr8Tp2Weights<'a> {
    device: Device<'a>,
    rank: u8,
    layers: Vec<usize>,
    programs: &'a Programs<'a>,
    max_rows: usize,
    activations: Exlr8PrefillActivations,
    workspace_bytes: usize,
    grouped_from_row: usize,
    weights: Rc<DeviceOwner<'a, Vec<Exlr8Weights<'a>>>>,
}

pub(crate) struct Exlr8Tp2<'a> {
    // Drain every consumer before releasing graph-visible scratch or weights.
    completion: RankCompletion<Event<'a>>,
    owner: Exlr8Tp2Weights<'a>,
    executions: DeviceOwner<'a, Vec<Exlr8Execution<'a>>>,
    output: Allocation<'a>,
    sm_copy: cuteafd_ffi::V41PeerCopy<'a>,
}

impl<'a> Exlr8Tp2Weights<'a> {
    pub(crate) fn workspace_bytes(programs: &'a Programs<'a>, max_rows: usize,
        activations: Exlr8PrefillActivations) -> Result<usize> {
        ensure!((DECODE_ROWS..=4096).contains(&max_rows), "EXLR8 RTX row capacity");
        let compact = Exlr8Execution::decode_compact_workspace_bytes(programs, DECODE_ROWS, TP2_UNITS)?;
        let grouped = Exlr8Execution::decode_grouped_workspace_bytes(programs, DECODE_ROWS, TP2_UNITS)?;
        let prefill = if max_rows > DECODE_ROWS {
            Exlr8Execution::workspace_bytes_with_limit(programs, max_rows, ExpertSet::Routed, TP2_UNITS, activations)?
        } else { 0 };
        compact.checked_add(grouped).and_then(|bytes| bytes.checked_add(prefill))
            .and_then(|bytes| bytes.checked_add(max_rows * 6144 * 2))
            .context("EXLR8 RTX workspace overflow")
    }

    pub(crate) fn pushed_pair(devices: [Device<'a>; 2], programs: &'a Programs<'a>,
        layers: Vec<usize>, max_rows: usize, budgets: [usize; 2],
        activations: Exlr8PrefillActivations, pushed: [Vec<Exlr8Weights<'a>>; 2]) -> Result<[Self; 2]> {
        ensure!(!layers.is_empty() && layers.windows(2).all(|ids| ids[0] < ids[1]),
            "EXLR8 RTX needs ordered resident layer IDs");
        let workspace_bytes = Self::workspace_bytes(programs, max_rows, activations)?;
        let mut ranks = Vec::new();
        for (rank, (device, weights)) in devices.into_iter().zip(pushed).enumerate() {
            let info = device.library.cuda_device_info(device.id)?;
            let grouped_from_row = grouped_decode_from_row(TP2_UNITS,
                (info.compute_capability_major, info.compute_capability_minor));
            ensure!(weights.len() == layers.len(), "EXLR8 pushed layer count differs from admission");
            let resident = weights.iter().zip(&layers).try_fold(0usize, |sum, (weight, &layer)| {
                ensure!(weight.layout.layer == layer && weight.layout.world == 2 && weight.layout.rank == rank
                    && weight.layout.set == ExpertSet::Routed && weight.layout.compact_unit_limit() <= TP2_UNITS,
                    "EXLR8 pushed TP2 selection differs from admission");
                sum.checked_add(weight.budget.resident_bytes).context("EXLR8 RTX residency overflow")
            })?;
            ensure!(resident.checked_add(workspace_bytes).is_some_and(|bytes| bytes <= budgets[rank]),
                "EXLR8 RTX weights and execution exceed admitted peak");
            let weights = device.own(|| Ok(weights))?;
            ranks.push(Self { device, rank: rank as u8, layers: layers.clone(), programs,
                max_rows, activations, workspace_bytes, grouped_from_row, weights: Rc::new(weights) });
        }
        ranks.try_into().map_err(|_| anyhow::anyhow!("EXLR8 needs two RTX owners"))
    }

    pub(crate) fn execution(self) -> Result<Exlr8Tp2<'a>> {
        let device = self.device;
        let executions = device.own(|| {
            let mut executions = vec![
                Exlr8Execution::new_decode_compact(device.library, self.programs, DECODE_ROWS,
                    self.workspace_bytes, TP2_UNITS)?,
                Exlr8Execution::new_decode_grouped(device.library, self.programs, DECODE_ROWS,
                    self.workspace_bytes, TP2_UNITS)?,
            ];
            if self.max_rows > DECODE_ROWS {
                executions.push(Exlr8Execution::new_with_limit(device.library, self.programs,
                    self.max_rows, self.workspace_bytes, ExpertSet::Routed, TP2_UNITS, self.activations)?);
            }
            for execution in &mut executions { execution.seal_graph_storage()?; }
            Ok(executions)
        })?;
        let completion = RankCompletion::new(Event::new(device)?);
        let output = Allocation::new(device, self.max_rows * 6144 * 2)?;
        let sm_copy = device.run(|| device.library.v41_peer_copy())?;
        Ok(Exlr8Tp2 { completion, owner: self, executions, output, sm_copy })
    }
}

impl RtxExpertLayer for Exlr8Tp2<'_> {
    fn shard(&self) -> RtxShard { RtxShard::Tp2 { rank: self.owner.rank } }
    fn partial(&self) -> PartialDtype { PartialDtype::Bf16 }
    fn uses_fp8_input(&self) -> bool { true }
    fn layers(&self) -> &[usize] { &self.owner.layers }
    fn device(&self) -> i32 { self.owner.device.id }
    fn workspace_bytes(&self) -> usize { self.owner.workspace_bytes }
    fn output(&self) -> *mut c_void { self.output.buffer.ptr }
    fn group_rows(&self, rows: usize) -> Result<usize> {
        ensure!(rows <= self.owner.max_rows, "EXLR8 grouping rows exceed admission");
        let name = if rows == 0 || rows > DECODE_ROWS { "exlr8_a16_grouped" }
            else { "exlr8_a16_serving" };
        Ok(cuteafd_ffi::exlr8::group_rows(self.owner.programs.spec(name)?)?)
    }
    fn weight_read_bytes(&self) -> Result<usize> {
        self.owner.weights.iter().map(|weights| weights.budget.resident_bytes / weights.layout.experts)
            .max().context("EXLR8 local weights missing")
    }
    unsafe fn enqueue(&mut self, layer: usize, rows: usize, input: ExpertInput, routes: Routes,
        stream: *mut c_void, abort: &mut dyn FnMut()) -> Result<()> {
        let result = (|| {
            ensure!((1..=self.owner.max_rows).contains(&rows), "EXLR8 local rows exceed admission");
            let index = super::resident_index(&self.owner.layers, layer)?;
            let ExpertInput::Fp8K32(wire) = input else { bail!("EXLR8 RTX requires FP8 K32 wire input") };
            ensure!(!wire.is_null() && !routes.ids.is_null() && !routes.weights.is_null(), "null EXLR8 RTX input");
            let selected = if rows > DECODE_ROWS { 2 }
                else { usize::from(rows >= self.owner.grouped_from_row) };
            let execution = &self.executions[selected];
            let device = self.owner.device;
            let source = |ptr, bytes| CuteafdDeviceBuffer { ptr, bytes, device_id: device.id, flags: 0 };
            device.run(|| {
                let copy = |target, input: CuteafdDeviceBuffer| -> Result<()> {
                    // SAFETY: rank-local operands outlive this stream's completion;
                    // decode copies execute on SMs, never behind a parked copy engine.
                    unsafe {
                        if crate::shared::memory::device::sm_peer_copy_for(true, input.bytes, DECODE_ROWS * 6144 * 2) {
                            self.sm_copy.launch_rows(target, input, input.bytes, 1, input.bytes, input.bytes, stream)
                        } else {
                            device.library.copy_d2d_async(target, input, input.bytes, stream)
                        }
                    }
                };
                for (target, input) in [
                    (execution.input()?, source(wire, rows * 6144 * 33 / 32)),
                    (execution.ids()?, source(routes.ids, rows * 8 * 4)),
                    (execution.route_weights()?, source(routes.weights, rows * 8 * 4)),
                ] {
                    copy(view(target, 0, input.bytes)?, input)?;
                }
                // SAFETY: immutable weights and sealed arena remain owned until the completion event drains.
                unsafe {
                    execution.run(&self.owner.weights[index], rows, stream)?;
                    let bytes = rows * 6144 * 2;
                    copy(view(self.output.buffer, 0, bytes)?, view(execution.output()?, 0, bytes)?)?;
                }
                Ok(())
            })
        })();
        self.completion.finish(result, stream, abort)
    }
}
