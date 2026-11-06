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
    if *dtype == DType::F8E4M3 && mode == AttentionMode::Fp8Compact {
        return Err(cuteafd_loader::formats::native_attention::UnsupportedFp8Conversion {
            tensor: format!("{}.weight", layout.stem), found: "E4M3".into(),
        }.into());
    }
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
        let scale = cuteafd_loader::formats::native_attention::fp8_scale(selected, &layout.stem)?;
        return Ok(vec![AttentionRecord::Operand(Operand { device, buffer: values,
            lands: vec![(source(&weight, rows.clone(), cols.clone(), 1)?, 0)], bytes: n*k }),
            AttentionRecord::Operand(Operand { device, buffer: grid, lands: vec![(source(&scale.meta.name,
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

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_loader::{formats::{native_attention::NativeLayout, trellis_core::CoreSplit},
        plan::checkpoint::{AttentionMode, CheckpointTensor, ProjectionStorage, SelectedProjection}, SafetensorsTensorMetadata};

    #[test]
    fn fp8_compact_records_refuse_already_fp8_before_allocations_or_sources() {
        let selected = SelectedProjection { storage: ProjectionStorage::Native {
            dtype: DType::F8E4M3, shape: vec![256, 256] }, source_bytes: 0, tensors: Vec::new() };
        let layout = NativeLayout { stem: "layers.0.attention.wq".into(), shape: [256, 256],
            fp8: true, rank: 0, ranks: 1, split: CoreSplit::Whole };
        let error = attention_records(std::path::Path::new("/unreadable"), &selected,
            AttentionMode::Fp8Compact, &layout, crate::shared::fp8_linear::Fp8Scales::Amax,
            CuteafdDeviceBuffer::default(), None).err().expect("already-FP8 must refuse");
        let refusal = error.downcast_ref::<cuteafd_loader::formats::native_attention::UnsupportedFp8Conversion>()
            .expect("conversion refusal must retain its typed error");
        assert_eq!(refusal.tensor, "layers.0.attention.wq.weight");
        assert_eq!(refusal.found, "E4M3");
    }

    #[test]
    fn native_fp8_records_land_the_declared_scale_name_and_tp_slice() -> Result<()> {
        for suffix in ["weight_scale_inv", "qscale_weight"] {
            let stem = "layers.0.attention.wq";
            let tensor = |name, dtype, shape: Vec<usize>, width| CheckpointTensor {
                shard: "selected.safetensors".into(), meta: SafetensorsTensorMetadata { name, dtype,
                    byte_length: shape.iter().product::<usize>() as u64 * width, shape, byte_offset: 128 } };
            let mut selected = SelectedProjection { storage: ProjectionStorage::Native { dtype: DType::F8E4M3,
                shape: vec![512, 256] }, source_bytes: 131104, tensors: vec![
                    tensor(format!("{stem}.weight"), DType::F8E4M3, vec![512, 256], 1),
                    tensor(format!("{stem}.{suffix}"), DType::F32, vec![4, 2], 4)] };
            let layout = NativeLayout { stem: stem.into(), shape: [256, 256], fp8: true, rank: 1, ranks: 2, split: CoreSplit::Output };
            let values = CuteafdDeviceBuffer { device_id: 0, bytes: 65536, ..Default::default() };
            let scales = CuteafdDeviceBuffer { device_id: 0, bytes: 256, ..Default::default() };
            let records = attention_records(std::path::Path::new("/selected"), &selected, AttentionMode::Hqa,
                &layout, crate::shared::fp8_linear::Fp8Scales::Amax, values, Some(scales))?;
            let AttentionRecord::Operand(scale) = &records[1] else { panic!("native scale must land verbatim"); };
            assert_eq!(scale.lands[0].0.name, format!("{stem}.{suffix}"));
            assert_eq!(scale.lands[0].0.rows, 2..4);
            assert_eq!(scale.lands[0].0.cols, 0..8);
            selected.tensors.pop();
            let error = attention_records(std::path::Path::new("/selected"), &selected, AttentionMode::Hqa,
                &layout, crate::shared::fp8_linear::Fp8Scales::Amax, values, Some(scales)).err().unwrap();
            assert!(error.is::<cuteafd_loader::formats::native_attention::InvalidFp8Scale>());
        }
        Ok(())
    }
}

