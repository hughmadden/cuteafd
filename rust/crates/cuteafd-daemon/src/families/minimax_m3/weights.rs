//! Unchanged MXFP8 payloads through the shared dense records, the shared
//! gate/up and K/V fused layout, and load-time scale arrangement.
use super::native::Native;
use crate::shared::experts::rtx::dense::{DenseLoad, Job, Operand, Rect, Step, STEP_BYTES};
use crate::shared::memory::device::{Allocation, Device};
use anyhow::{ensure, Context, Result};
use cuteafd_core::DType;
use cuteafd_ffi::{programs::Scalar, CuteafdDeviceBuffer};
use cuteafd_loader::formats::native_dense::{element_bytes, RankWeightLayout, WeightOperand, WeightSlice};
use cuteafd_loader::plan::{checkpoint::Checkpoint, families::minimax::coordinator};
use std::{collections::BTreeMap, path::Path};

pub(super) const EMBEDDING: &str = "language_model.model.embed_tokens.weight";
pub(super) struct Weights<'a> { operands: BTreeMap<String, Allocation<'a>> }

impl<'a> Weights<'a> {
    pub fn load(native: &Native<'a>, checkpoint: &Checkpoint) -> Result<Self> {
        let layout = coordinator::resident_weights(checkpoint, false)?;
        Self::load_layout(native, checkpoint, &layout)
    }
    pub fn draft_layout(contract: &cuteafd_loader::families::minimax_m3::dspark::DsparkContract) -> Result<RankWeightLayout> {
        coordinator::single_gpu(&contract.weights)
    }
    pub fn load_layout(native: &Native<'a>, checkpoint: &Checkpoint, layout: &RankWeightLayout) -> Result<Self> {
        let device = native.device();
        let mut operands = BTreeMap::new();
        for operand in &layout.operands {
            if operand.name == EMBEDDING { continue; }
            cuteafd_ffi::memory_ledger::tensor(&operand.name);
            let source = Allocation::new(device, usize::try_from(operand.allocation_bytes)?)?;
            load_records(device, &checkpoint.snapshot, layout, operand, source.buffer)?;
            let resident = if operand.name.ends_with(".weight_scale_inv") {
                let arranged = Allocation::new(device, operand.allocation_bytes as usize)?;
                native.launch("minimax_scale_prep", &[source.buffer,arranged.buffer], &[
                    Scalar::I32(i32::try_from(operand.shape[0])?), Scalar::I32(i32::try_from(operand.shape[1])?)])?;
                native.drain()?;
                arranged
            } else { source };
            operands.insert(operand.name.clone(), resident);
        }
        Ok(Self { operands })
    }
    pub fn get(&self, name: &str) -> Result<CuteafdDeviceBuffer> {
        self.operands.get(name).map(|a| a.buffer).with_context(|| format!("MiniMax missing native operand {name}"))
    }
    pub fn layer(&self, layer: usize, name: &str) -> Result<CuteafdDeviceBuffer> {
        self.get(&format!("language_model.model.layers.{layer}.{name}"))
    }
}

/// Where a native dense operand's bytes come from, in allocation order: each
/// source's checkpoint rectangle (the whole tensor, the rank's rows or the
/// rank's column window, in source order) at its operand offset, flagged when
/// an F32 rectangle narrows to BF16 into half its bytes. Also returns the
/// packed bytes, which the header admission must match.
fn operand_lands(snapshot: &Path, layout: &RankWeightLayout, operand: &WeightOperand)
    -> Result<(Vec<(Rect, usize, bool)>, usize)> {
    let elem = element_bytes(&operand.dtype).with_context(|| format!("{}: unsupported precision", operand.name))? as usize;
    let mut lands = Vec::with_capacity(operand.sources.len());
    let mut at = 0usize;
    for source in &operand.sources {
        let meta = &source.meta;
        let narrow = meta.dtype != operand.dtype;
        ensure!(!narrow || (operand.dtype == DType::Bf16 && meta.dtype == DType::F32),
            "{}: native parameter precision changed", meta.name);
        let total = if meta.shape.len() >= 2 { meta.shape[0] } else { 1 };
        let bytes = usize::try_from(meta.byte_length)?;
        ensure!(total > 0 && bytes % total == 0, "{}: rows do not divide the tensor", meta.name);
        let whole = Rect { name: meta.name.clone(), path: snapshot.join(&source.shard), base: meta.byte_offset,
            row_bytes: bytes / total, rows: 0..total, cols: 0..bytes / total };
        let rect = match operand.slice {
            WeightSlice::Replicated => whole,
            WeightSlice::Rows | WeightSlice::Cols => {
                ensure!(operand.sources.len() == 1 && meta.shape.len() == 2, "{}: split source layout", meta.name);
                let (rank, ranks) = (layout.rank, layout.ranks);
                // Narrowing is elementwise: a source element's width selects the window.
                let elem = if narrow { 4 } else { elem };
                if operand.slice == WeightSlice::Rows {
                    ensure!(total % ranks == 0, "{}: rows do not split over {ranks} GPUs", meta.name);
                    whole.sub(rank * total / ranks..(rank + 1) * total / ranks)
                } else {
                    ensure!(meta.shape[1] % ranks == 0, "{}: columns do not split over {ranks} GPUs", meta.name);
                    let width = meta.shape[1] / ranks * elem;
                    Rect { cols: rank * width..(rank + 1) * width, ..whole }
                }
            }
        };
        let landed = if narrow { rect.bytes() / 2 } else { rect.bytes() };
        lands.push((rect, at, narrow));
        at += landed;
    }
    let expected = operand.shape.iter().try_fold(elem, |n: usize, dim| n.checked_mul(*dim))
        .context("packed size overflow")?;
    ensure!(at == expected && expected.max(256) as u64 == operand.allocation_bytes,
        "{}: packed storage disagrees with header admission", operand.name);
    Ok((lands, at))
}

/// `operand`'s bytes as dense records into `buffer` (allocated as the mount
/// allocates it): its [`operand_lands`], the unchanged rectangles as one
/// operand and each narrowed one as a staged step on the GPU. Filled through
/// the mount at once, or recorded for the coordinator push group.
fn load_records(device: Device<'_>, snapshot: &Path, layout: &RankWeightLayout, operand: &WeightOperand,
    buffer: CuteafdDeviceBuffer) -> Result<()> {
    let (lands, bytes) = operand_lands(snapshot, layout, operand)?;
    ensure!(buffer.bytes >= bytes, "{}: allocation smaller than its packed bytes", operand.name);
    let dense = DenseLoad::mount(device.library);
    let mut unchanged = Vec::with_capacity(lands.len());
    for (rect, at, narrow) in lands {
        if !narrow { unchanged.push((rect, at)); continue; }
        // F32 -> BF16 on the GPU: one tensor per step (CSA indexer tensors).
        let n = rect.bytes();
        ensure!(n <= STEP_BYTES, "{}: F32 narrowing step beyond the staging bound", rect.name);
        // The lands keep the narrowed values inside the packed bytes.
        let dst = CuteafdDeviceBuffer { ptr: buffer.ptr.wrapping_byte_add(at), bytes: n / 2, ..buffer };
        device.run(|| dense.step(Step { device: device.id, rects: vec![rect],
            jobs: vec![Job::NarrowF32Bf16 { at: 0, bytes: n, dst }] }, std::ptr::null_mut()))?;
    }
    if !unchanged.is_empty() {
        device.run(|| dense.operand(Operand { device: device.id, buffer, lands: unchanged, bytes }))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_loader::plan::testing;
    #[test] fn minimax_draft_fusion_preserves_native_headers_and_allocation_bytes() -> Result<()> {
        use cuteafd_loader::families::minimax_m3::dspark::{DsparkContract,PREFIX};
        let dir=tempfile::tempdir()?;
        testing::write_minimax_m3_snapshot(dir.path(),false,true);
        let contract=DsparkContract::open(dir.path())?;
        let layout=Weights::draft_layout(&contract)?;
        assert_eq!(layout.operands.len(),77);
        assert_eq!(layout.resident_bytes,2_301_852_928);
        assert_eq!(layout.resident_bytes,layout.operands.iter().map(|w|w.allocation_bytes).sum::<u64>());
        for operand in &layout.operands {
            assert_eq!(operand.allocation_bytes,operand.sources.iter().map(|s|s.meta.byte_length).sum::<u64>().max(256));
            assert!(operand.sources.iter().all(|s|s.meta.dtype==operand.dtype));
            assert_eq!(operand.slice,WeightSlice::Replicated);
        }
        for layer in 0..5 { for suffix in ["weight","weight_scale_inv"] {
            for (stem,first,second,rows) in [("self_attn.kv_proj",".k_proj.",".v_proj.",1024),
                ("mlp.gate_up_proj",".gate_proj.",".up_proj.",24576)] {
                let operand=layout.operands.iter().find(|w|w.name==format!("{PREFIX}.layers.{layer}.decoder_layer.{stem}.{suffix}")).unwrap();
                assert_eq!(operand.sources.len(),2);
                assert!(operand.sources[0].meta.name.contains(first));
                assert!(operand.sources[1].meta.name.contains(second));
                assert_eq!(operand.shape[0],rows);
            }
        } }
        Ok(())
    }
    #[test] fn minimax_fused_kv_and_ffn_preserve_checkpoint_order_and_bytes() -> Result<()> {
        let dir = tempfile::tempdir()?;
        testing::write_minimax_m3_snapshot(dir.path(), true, true);
        let checkpoint = Checkpoint::open(dir.path())?;
        let layout = coordinator::resident_weights(&checkpoint,false)?;
        assert_eq!(layout.resident_bytes,layout.operands.iter().map(|w|w.allocation_bytes).sum::<u64>());
        for suffix in ["weight","weight_scale_inv"] {
            let operand = layout.operands.iter().find(|w|w.name == format!("language_model.model.layers.0.self_attn.kv_proj.{suffix}")).unwrap();
            assert_eq!(operand.sources.len(),2);
            assert!(operand.sources[0].meta.name.contains(".k_proj."));
            assert!(operand.sources[1].meta.name.contains(".v_proj."));
            assert_eq!(operand.shape[0],1024);
            let gate = layout.operands.iter().find(|w|w.name == format!("language_model.model.layers.0.mlp.gate_up_proj.{suffix}")).unwrap();
            assert!(gate.sources[0].meta.name.contains(".gate_proj."));
            assert!(gate.sources[1].meta.name.contains(".up_proj."));
        }
        // Every operand lands its sources whole and unchanged, back to back in checkpoint order.
        for operand in &layout.operands {
            let (lands, bytes) = operand_lands(&checkpoint.snapshot, &layout, operand)?;
            let mut at = 0;
            for ((rect, offset, narrow), source) in lands.iter().zip(&operand.sources) {
                assert_eq!((rect.name.as_str(), rect.path.clone(), rect.base, *offset, *narrow),
                    (source.meta.name.as_str(), checkpoint.snapshot.join(&source.shard), source.meta.byte_offset, at, false));
                assert_eq!((rect.rows.start, rect.cols.start, rect.bytes() as u64), (0, 0, source.meta.byte_length));
                at += rect.bytes();
            }
            assert_eq!((lands.len(), bytes, (bytes as u64).max(256)), (operand.sources.len(), at, operand.allocation_bytes));
        }
        Ok(())
    }
}
