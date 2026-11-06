//! Selected native attention storage and load/push records, independent of family.
use crate::shared::{experts::rtx::dense::{Job, Rect}, fp8_load_quant::Target, memory::DeviceAllocation};
use anyhow::{ensure, Context, Result};
use cuteafd_core::DType;
use cuteafd_ffi::CuteafdDeviceBuffer;

pub(crate) struct NativeAttentionWeights<'a> {
    pub layout: cuteafd_loader::formats::native_attention::NativeLayout,
    pub values: DeviceAllocation<'a>,
    pub scales: Option<DeviceAllocation<'a>>,
}

/// Shared mount/push recipe, inspectable on CPU without creating a CUDA owner.
/// Native FP8 lands verbatim; only selected BF16 in fp8-compact is quantized.
pub(crate) enum AttentionRecord {
    Operand(crate::shared::experts::rtx::dense::Operand),
    Step(crate::shared::experts::rtx::dense::Step),
}

pub(crate) fn attention_records(snapshot: &std::path::Path,
    selected: &cuteafd_loader::plan::checkpoint::SelectedProjection,
    mode: cuteafd_loader::plan::checkpoint::AttentionMode,
    layout: &cuteafd_loader::formats::native_attention::NativeLayout,
    rule: crate::shared::fp8_linear::Fp8Scales, values: CuteafdDeviceBuffer,
    scales: Option<CuteafdDeviceBuffer>) -> Result<Vec<AttentionRecord>> {
    use cuteafd_loader::{formats::trellis_core::CoreSplit,
        plan::checkpoint::{AttentionMode, ProjectionStorage}};
    use crate::shared::experts::rtx::dense::{Operand, Step, windows};
    let ProjectionStorage::Native { dtype, shape } = &selected.storage else {
        anyhow::bail!("{}: native recipe received trellis", layout.stem);
    };
    ensure!(shape.len() == 2 && matches!(dtype, DType::Bf16 | DType::F8E4M3), "native attention source geometry");
    let fp8 = *dtype == DType::F8E4M3 || mode == AttentionMode::Fp8Compact;
    ensure!(fp8 == layout.fp8 && layout.ranks > 0 && layout.rank < layout.ranks,
        "native attention recipe differs from selected mode");
    let [n, k] = layout.shape;
    let (rows, cols) = match layout.split {
        CoreSplit::Output => (layout.rank*n..(layout.rank+1)*n, 0..k),
        CoreSplit::Input => (0..n, layout.rank*k..(layout.rank+1)*k),
        CoreSplit::Whole => (0..n, 0..k),
    };
    ensure!(rows.end <= shape[0] && cols.end <= shape[1]
        && rows.len() == n && cols.len() == k, "native attention source slice");
    let source = |name: &str, rows: std::ops::Range<usize>, cols: std::ops::Range<usize>, width: usize| -> Result<Rect> {
        let t = selected.tensors.iter().find(|t| t.meta.name == name).context("native attention selected tensor missing")?;
        ensure!(t.meta.shape.len() == 2 && rows.end <= t.meta.shape[0] && cols.end <= t.meta.shape[1],
            "{name}: native rectangle outside selected tensor");
        Ok(Rect { name: name.into(), path: snapshot.join(&t.shard), base: t.meta.byte_offset,
            row_bytes: t.meta.shape[1]*width, rows, cols: cols.start*width..cols.end*width })
    };
    ensure!(values.bytes >= n*k*if fp8 { 1 } else { 2 }, "native attention values extent");
    let weight = format!("{}.weight", layout.stem);
    let device = values.device_id;
    if !fp8 {
        ensure!(scales.is_none(), "BF16 attention has scale allocation");
        return Ok(vec![AttentionRecord::Operand(Operand { device, buffer: values,
            lands: vec![(source(&weight, rows, cols, 2)?, 0)], bytes: n*k*2 })]);
    }
    ensure!(n%128 == 0 && k%128 == 0, "native attention whole FP8 blocks");
    let grid = scales.context("native attention FP8 scale allocation missing")?;
    ensure!(grid.device_id == device && grid.bytes >= n/128*(k/128)*4, "native attention scale extent");
    if *dtype == DType::F8E4M3 {
        return Ok(vec![AttentionRecord::Operand(Operand { device, buffer: values,
            lands: vec![(source(&weight, rows.clone(), cols.clone(), 1)?, 0)], bytes: n*k }),
            AttentionRecord::Operand(Operand { device, buffer: grid, lands: vec![(source(&format!("{weight}_scale_inv"),
                rows.start/128..rows.end/128, cols.start/128..cols.end/128, 4)?, 0)], bytes: n/128*(k/128)*4 })]);
    }
    let rule = if mode == AttentionMode::Fp8Compact { crate::shared::fp8_linear::Fp8Scales::Amax } else { rule };
    let whole = source(&weight, rows, cols, 2)?;
    windows(&[whole], 128)?.into_iter().map(|(rects, rows)| {
        let target = Target {
            q: CuteafdDeviceBuffer { ptr: values.ptr.wrapping_byte_add(rows.start*k), bytes: rows.len()*k, ..values },
            q_stride: k,
            scales: CuteafdDeviceBuffer { ptr: grid.ptr.wrapping_byte_add(rows.start/128*(k/128)*4),
                bytes: rows.len()/128*(k/128)*4, ..grid }, kmajor: None,
        };
        Ok(AttentionRecord::Step(Step { device, rects, jobs: vec![Job::Quant { rows: 0..rows.len(), pitch: k,
            col0: 0, cols: k, layout: crate::shared::fp8_load_quant::Layout::Block, rule, target }] }))
    }).collect()
}

