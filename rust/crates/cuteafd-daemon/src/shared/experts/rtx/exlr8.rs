//! Native EXLR8 routed experts on one RTX or the shared paired-expert executor.
use super::native::RankCompletion;
use super::{ExpertInput, PartialDtype, Routes, RtxExpertLayer, RtxShard};
use crate::shared::experts::exlr8::{grouped_decode_from_row, view, Exlr8Execution, Exlr8PrefillActivations, Exlr8Weights};
use crate::shared::memory::device::{Allocation, Device, DeviceOwner, Event};
use anyhow::{bail, ensure, Context, Result};
use cuteafd_ffi::{exlr8::KernelFamily, programs::Programs, CuteafdDeviceBuffer};
use cuteafd_loader::plan::exlr8::residency::ExpertSet;
use std::{ffi::c_void, rc::Rc};

const DECODE_ROWS: usize = 64;
const TP2_UNITS: usize = 8;

fn native_shard(world: usize, rank: usize) -> Result<RtxShard> {
    ensure!((1..=2).contains(&world) && rank < world, "EXLR8 RTX requires world 1 or 2 and a valid rank");
    Ok(if world == 1 { RtxShard::Whole } else { RtxShard::Tp2 { rank: rank as u8 } })
}

#[cfg(test)]
mod copy_tests {
    #[test]
    fn native_rtx_shards_follow_actual_world_and_rank() {
        use super::{native_shard, RtxShard};
        assert_eq!(native_shard(1, 0).unwrap(), RtxShard::Whole);
        for rank in 0..2 { assert_eq!(native_shard(2, rank).unwrap(), RtxShard::Tp2 { rank: rank as u8 }); }
        for (world, rank) in [(0, 0), (1, 1), (2, 2), (3, 0), (usize::MAX, 0)] {
            assert!(native_shard(world, rank).is_err());
        }
    }

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
        assert!(enqueue.contains("sm_peer_copy_for(true, input.bytes, DECODE_ROWS * self.owner.hidden * 2)"));
        assert!(enqueue.contains("self.sm_copy.launch_rows("), "same-device SM copies require the local-capable entry point");
        assert!(enqueue.contains("copy(view(self.output.buffer, 0, bytes)?,"));
    }
    use super::DECODE_ROWS;
}

pub(crate) struct Exlr8RtxWeights<'a> {
    device: Device<'a>,
    shard: RtxShard,
    unit_limit: usize,
    layers: Vec<usize>,
    programs: &'a Programs<'a>,
    max_rows: usize,
    family: KernelFamily,
    hidden: usize,
    activations: Exlr8PrefillActivations,
    workspace_bytes: usize,
    grouped_from_row: usize,
    weights: Rc<DeviceOwner<'a, Vec<Exlr8Weights<'a>>>>,
}

pub(crate) struct Exlr8Rtx<'a> {
    // Drain every consumer before releasing graph-visible scratch or weights.
    completion: RankCompletion<Event<'a>>,
    owner: Exlr8RtxWeights<'a>,
    executions: DeviceOwner<'a, Vec<Exlr8Execution<'a>>>,
    output: Allocation<'a>,
    sm_copy: cuteafd_ffi::V41PeerCopy<'a>,
}

impl<'a> Exlr8RtxWeights<'a> {
    pub(crate) fn workspace_bytes(programs: &'a Programs<'a>, family: KernelFamily, max_rows: usize,
        activations: Exlr8PrefillActivations) -> Result<usize> {
        Self::workspace_bytes_with_limit(programs, family, max_rows, TP2_UNITS, activations)
    }

    pub(crate) fn workspace_bytes_with_limit(programs: &'a Programs<'a>, family: KernelFamily,
        max_rows: usize, unit_limit: usize, activations: Exlr8PrefillActivations) -> Result<usize> {
        ensure!((DECODE_ROWS..=4096).contains(&max_rows), "EXLR8 RTX row capacity");
        ensure!((1..=family.geometry().units).contains(&unit_limit), "EXLR8 RTX unit capacity");
        let hidden = Exlr8Execution::routed_hidden(programs, family)?;
        let compact = Exlr8Execution::decode_compact_workspace_bytes(programs, family, DECODE_ROWS, unit_limit)?;
        let grouped = Exlr8Execution::decode_grouped_workspace_bytes(programs, family, DECODE_ROWS, unit_limit)?;
        let prefill = if max_rows > DECODE_ROWS {
            Exlr8Execution::workspace_bytes_with_limit(programs, family, max_rows, ExpertSet::Routed, unit_limit, activations)?
        } else { 0 };
        compact.checked_add(grouped).and_then(|bytes| bytes.checked_add(prefill))
            .and_then(|bytes| max_rows.checked_mul(hidden)?.checked_mul(2)?.checked_add(bytes))
            .context("EXLR8 RTX workspace overflow")
    }

    pub(crate) fn pushed_pair(devices: [Device<'a>; 2], programs: &'a Programs<'a>,
        layers: Vec<usize>, max_rows: usize, budgets: [usize; 2],
        activations: Exlr8PrefillActivations, pushed: [Vec<Exlr8Weights<'a>>; 2]) -> Result<[Self; 2]> {
        let [left, right] = pushed;
        Ok([Self::pushed_rank(devices[0], programs, layers.clone(), max_rows, budgets[0], activations, 2, 0, left)?,
            Self::pushed_rank(devices[1], programs, layers, max_rows, budgets[1], activations, 2, 1, right)?])
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn pushed_rank(device: Device<'a>, programs: &'a Programs<'a>, layers: Vec<usize>,
        max_rows: usize, budget: usize, activations: Exlr8PrefillActivations,
        world: usize, rank: usize, weights: Vec<Exlr8Weights<'a>>) -> Result<Self> {
        let shard = native_shard(world, rank)?;
        ensure!(!layers.is_empty() && layers.windows(2).all(|ids| ids[0] < ids[1]),
            "EXLR8 RTX needs ordered resident layer IDs");
        let first = &weights.first().context("EXLR8 RTX has no pushed weights")?.layout;
        let expected = (first.hidden, first.units, first.experts);
        let family = KernelFamily::from_geometry(expected.0, expected.1, 8, expected.2)?;
        let hidden = Exlr8Execution::routed_hidden(programs, family)?;
        let unit_limit = first.compact_unit_limit();
        let workspace_bytes = Self::workspace_bytes_with_limit(programs, family, max_rows, unit_limit, activations)?;
        let info = device.library.cuda_device_info(device.id)?;
        let grouped_from_row = grouped_decode_from_row(unit_limit,
            (info.compute_capability_major, info.compute_capability_minor));
        ensure!(weights.len() == layers.len(), "EXLR8 pushed layer count differs from admission");
        let resident = weights.iter().zip(&layers).try_fold(0usize, |sum, (weight, &layer)| {
            ensure!(weight.layout.layer == layer && weight.layout.world == world && weight.layout.rank == rank
                && weight.layout.hidden == hidden && weight.layout.experts == expected.2
                && weight.layout.units == expected.1 && weight.layout.set == ExpertSet::Routed
                && weight.layout.compact_unit_limit() == unit_limit,
                "EXLR8 pushed RTX selection differs from admission");
            sum.checked_add(weight.budget.resident_bytes).context("EXLR8 RTX residency overflow")
        })?;
        ensure!(resident.checked_add(workspace_bytes).is_some_and(|bytes| bytes <= budget),
            "EXLR8 RTX weights and execution exceed admitted peak");
        let weights = device.own(|| Ok(weights))?;
        Ok(Self { device, shard, unit_limit, layers, programs, max_rows, family, hidden,
            activations, workspace_bytes, grouped_from_row, weights: Rc::new(weights) })
    }

    pub(crate) fn execution(self) -> Result<Exlr8Rtx<'a>> {
        let device = self.device;
        let executions = device.own(|| {
            let mut executions = vec![
                Exlr8Execution::new_decode_compact(device.library, self.programs, self.family, DECODE_ROWS,
                    self.workspace_bytes, self.unit_limit)?,
                Exlr8Execution::new_decode_grouped(device.library, self.programs, self.family, DECODE_ROWS,
                    self.workspace_bytes, self.unit_limit)?,
            ];
            if self.max_rows > DECODE_ROWS {
                executions.push(Exlr8Execution::new_with_limit(device.library, self.programs, self.family,
                    self.max_rows, self.workspace_bytes, ExpertSet::Routed, self.unit_limit, self.activations)?);
            }
            for execution in &mut executions { execution.seal_graph_storage()?; }
            Ok(executions)
        })?;
        let completion = RankCompletion::new(Event::new(device)?);
        let output = Allocation::new(device, self.max_rows * self.hidden * 2)?;
        let sm_copy = device.run(|| device.library.v41_peer_copy())?;
        Ok(Exlr8Rtx { completion, owner: self, executions, output, sm_copy })
    }
}

impl RtxExpertLayer for Exlr8Rtx<'_> {
    fn shard(&self) -> RtxShard { self.owner.shard }
    fn partial(&self) -> PartialDtype { PartialDtype::Bf16 }
    fn uses_fp8_input(&self) -> bool { true }
    fn layers(&self) -> &[usize] { &self.owner.layers }
    fn device(&self) -> i32 { self.owner.device.id }
    fn workspace_bytes(&self) -> usize { self.owner.workspace_bytes }
    fn output(&self) -> *mut c_void { self.output.buffer.ptr }
    fn group_rows(&self, rows: usize) -> Result<usize> {
        ensure!(rows <= self.owner.max_rows, "EXLR8 grouping rows exceed admission");
        let kind = if rows == 0 || rows > DECODE_ROWS { cuteafd_ffi::exlr8::KernelKind::GroupedA16 }
            else { cuteafd_ffi::exlr8::KernelKind::Routed };
        let name = self.owner.family.program(kind);
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
                        if crate::shared::memory::device::sm_peer_copy_for(true, input.bytes, DECODE_ROWS * self.owner.hidden * 2) {
                            self.sm_copy.launch_rows(target, input, input.bytes, 1, input.bytes, input.bytes, stream)
                        } else {
                            device.library.copy_d2d_async(target, input, input.bytes, stream)
                        }
                    }
                };
                for (target, input) in [
                    (execution.input()?, source(wire, rows * self.owner.hidden * 33 / 32)),
                    (execution.ids()?, source(routes.ids, rows * 8 * 4)),
                    (execution.route_weights()?, source(routes.weights, rows * 8 * 4)),
                ] {
                    copy(view(target, 0, input.bytes)?, input)?;
                }
                // SAFETY: immutable weights and sealed arena remain owned until the completion event drains.
                unsafe {
                    execution.run(&self.owner.weights[index], rows, stream)?;
                    let bytes = rows * self.owner.hidden * 2;
                    copy(view(self.output.buffer, 0, bytes)?, view(execution.output()?, 0, bytes)?)?;
                }
                Ok(())
            })
        })();
        self.completion.finish(result, stream, abort)
    }
}
