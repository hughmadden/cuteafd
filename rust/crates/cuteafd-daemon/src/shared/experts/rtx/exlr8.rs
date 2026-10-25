//! Native EXLR8 routed halves on the shared RTX paired-expert executor.
use super::native::RankCompletion;
use super::{ExpertInput, PartialDtype, Routes, RtxExpertLayer, RtxShard};
use crate::shared::experts::exlr8::{grouped_decode_from_row, view, Exlr8Execution, Exlr8PrefillActivations, Exlr8Weights};
use crate::shared::memory::device::{Allocation, Device, DeviceOwner, Event};
use anyhow::{bail, ensure, Context, Result};
use cuteafd_ffi::{programs::Programs, CuteafdDeviceBuffer};
use cuteafd_loader::plan::exlr8::residency::{ExpertSet, Residency};
use std::{ffi::c_void, path::Path, rc::Rc};

const DECODE_ROWS: usize = 64;
const TP2_UNITS: usize = 8;

pub(crate) struct Exlr8Tp2Weights<'a> {
    device: Device<'a>,
    rank: u8,
    layers: Vec<usize>,
    programs: &'a Programs<'a>,
    max_rows: usize,
    activations: Exlr8PrefillActivations,
    workspace_bytes: usize,
    weights: Rc<DeviceOwner<'a, Vec<Exlr8Weights<'a>>>>,
}

pub(crate) struct Exlr8Tp2<'a> {
    // Drain every consumer before releasing graph-visible scratch or weights.
    completion: RankCompletion<Event<'a>>,
    owner: Exlr8Tp2Weights<'a>,
    executions: DeviceOwner<'a, Vec<Exlr8Execution<'a>>>,
    output: Allocation<'a>,
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

    pub(crate) fn load_pair(devices: [Device<'a>; 2], snapshot: &Path, selected: &str,
        programs: &'a Programs<'a>, layers: Vec<usize>, max_rows: usize,
        budgets: [usize; 2], activations: Exlr8PrefillActivations) -> Result<[Self; 2]> {
        ensure!(!layers.is_empty() && layers.windows(2).all(|ids| ids[0] < ids[1]),
            "EXLR8 RTX needs ordered resident layer IDs");
        let workspace_bytes = Self::workspace_bytes(programs, max_rows, activations)?;
        let mut ranks = Vec::new();
        for (rank, device) in devices.into_iter().enumerate() {
            let plans = layers.iter().map(|&layer| Residency::plan(snapshot, selected, layer,
                2, rank, ExpertSet::Routed)).collect::<std::result::Result<Vec<_>, _>>()?;
            let resident = plans.iter().try_fold(0usize, |sum, plan| sum.checked_add(plan.resident_bytes)
                .context("EXLR8 RTX residency overflow"))?;
            ensure!(resident.checked_add(workspace_bytes).is_some_and(|bytes| bytes <= budgets[rank]),
                "EXLR8 RTX weights and execution exceed admitted peak");
            let retained_host = plans.iter().try_fold(0usize, |bytes, plan| bytes.checked_add(plan.host_bytes())
                .context("EXLR8 RTX host residency overflow"))?;
            let transient_host = plans.iter().map(|plan| {
                let budget = Exlr8Weights::plan(plan)?;
                budget.pinned_host_bytes.checked_add(budget.read_scratch_bytes.saturating_sub(plan.host_bytes()))
                    .context("EXLR8 RTX host staging overflow")
            }).collect::<Result<Vec<_>>>()?.into_iter().max().unwrap_or(0);
            let (_, available) = cuteafd_loader::formats::mapped_preload::host_memory()?;
            ensure!(retained_host.checked_add(transient_host).is_some_and(|bytes| bytes as u64 <= available),
                "EXLR8 RTX host residency and pinned load staging exceed available memory");
            let weights = device.own(|| plans.into_iter().map(|plan| {
                ensure!(plan.compact_unit_limit() <= TP2_UNITS, "EXLR8 TP2 unit ownership exceeds admission");
                Exlr8Weights::load(device.library, plan, resident)
            }).collect::<Result<Vec<_>>>())?;
            ranks.push(Self { device, rank: rank as u8, layers: layers.clone(), programs,
                max_rows, activations, workspace_bytes, weights: Rc::new(weights) });
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
        Ok(Exlr8Tp2 { completion, owner: self, executions, output })
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
        usize::try_from(self.owner.programs.spec(name)?.geometry["group_rows"].as_u64()
            .context("EXLR8 grouping metadata missing")?).context("EXLR8 group width exceeds address space")
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
                else { usize::from(rows >= grouped_decode_from_row(TP2_UNITS)) };
            let execution = &self.executions[selected];
            let device = self.owner.device;
            let source = |ptr, bytes| CuteafdDeviceBuffer { ptr, bytes, device_id: device.id, flags: 0 };
            device.run(|| {
                for (target, input) in [
                    (execution.input()?, source(wire, rows * 6144 * 33 / 32)),
                    (execution.ids()?, source(routes.ids, rows * 8 * 4)),
                    (execution.route_weights()?, source(routes.weights, rows * 8 * 4)),
                ] {
                    // SAFETY: caller retains rank-local input/routes and this owner through stream drainage.
                    unsafe { device.library.copy_d2d_async(view(target, 0, input.bytes)?, input, input.bytes, stream)?; }
                }
                // SAFETY: immutable weights and sealed arena remain owned until the completion event drains.
                unsafe {
                    execution.run(&self.owner.weights[index], rows, stream)?;
                    let bytes = rows * 6144 * 2;
                    device.library.copy_d2d_async(view(self.output.buffer, 0, bytes)?,
                        view(execution.output()?, 0, bytes)?, bytes, stream)?;
                }
                Ok(())
            })
        })();
        self.completion.finish(result, stream, abort)
    }
}
