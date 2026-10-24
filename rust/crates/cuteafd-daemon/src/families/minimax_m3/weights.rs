//! Unchanged MXFP8 payloads through the shared dense records, the shared
//! gate/up and K/V fused layout, and load-time scale arrangement.
use super::native::Native;
use crate::shared::{memory::device::Allocation, experts::rtx::dense::load_records};
use anyhow::{Context, Result};
use cuteafd_ffi::{programs::Scalar, CuteafdDeviceBuffer};
use cuteafd_loader::formats::native_dense::RankWeightLayout;
use cuteafd_loader::plan::{checkpoint::Checkpoint, families::minimax::coordinator};
use std::collections::BTreeMap;

pub(super) const EMBEDDING: &str = "language_model.model.embed_tokens.weight";
pub(super) struct Weights<'a> { operands: BTreeMap<String, Allocation<'a>> }

impl<'a> Weights<'a> {
    pub fn load(native: &Native<'a>, checkpoint: &Checkpoint) -> Result<Self> {
        let layout = coordinator::RankWeightLayout::from_checkpoint(checkpoint, 0, 1)?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::experts::rtx::dense::operand_lands;
    use cuteafd_loader::formats::native_dense::WeightSlice;
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
        let layout = coordinator::RankWeightLayout::from_checkpoint(&checkpoint,0,1)?;
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
