//! Unchanged MXFP8 payloads, load-time scale arrangement, and gate-first fusion.
use crate::shared::{memory::device::Device, experts::rtx::dense::{DenseLoad, Operand, Rect}};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{minimax::{Buffer, Commands}, programs::Scalar};
use cuteafd_loader::plan::{checkpoint::Checkpoint, families::minimax::coordinator::{RankWeightLayout, WeightOperand, WeightSlice}};
use std::collections::BTreeMap;

pub(super) const EMBEDDING: &str = "language_model.model.embed_tokens.weight";
pub(super) struct Weights<'a> { operands: BTreeMap<String, Buffer<'a>> }

fn fuse_kv(layout: &RankWeightLayout) -> Result<Vec<WeightOperand>> {
    let mut operands = Vec::new();
    for operand in &layout.operands {
        if operand.name.contains(".self_attn.v_proj.") { continue; }
        let mut operand = operand.clone();
        if operand.name.contains(".self_attn.k_proj.") {
            let value_name = operand.name.replace(".k_proj.", ".v_proj.");
            let value = layout.operands.iter().find(|w| w.name == value_name).context("MiniMax missing V projection")?;
            ensure!(operand.shape == value.shape && operand.dtype == value.dtype, "MiniMax K/V layout mismatch");
            operand.name = operand.name.replace(".k_proj.", ".kv_proj.");
            operand.shape[0] *= 2;
            operand.allocation_bytes += value.allocation_bytes;
            operand.sources.extend(value.sources.clone());
        }
        operands.push(operand);
    }
    Ok(operands)
}

fn load_operand<'a>(device: Device<'a>, checkpoint: &Checkpoint, operand: &WeightOperand) -> Result<Buffer<'a>> {
    ensure!(operand.slice == WeightSlice::Replicated, "MiniMax head split is unsupported");
    let buffer = Buffer::new(device.library, device.id, usize::try_from(operand.allocation_bytes)?)?;
    let mut lands = Vec::new();
    let mut offset = 0;
    for source in &operand.sources {
        let meta = &source.meta;
        ensure!(meta.dtype == operand.dtype, "MiniMax native source dtype changed");
        let rows = if meta.shape.len() == 2 { meta.shape[0] } else { 1 };
        let bytes = usize::try_from(meta.byte_length)?;
        ensure!(bytes % rows == 0, "MiniMax source rows do not divide bytes");
        lands.push((Rect { name: meta.name.clone(), path: checkpoint.snapshot.join(&source.shard),
            base: meta.byte_offset, row_bytes: bytes/rows, rows: 0..rows, cols: 0..bytes/rows }, offset));
        offset += bytes;
    }
    ensure!(offset.max(256) == operand.allocation_bytes as usize, "MiniMax operand extent disagrees with admission");
    device.run(|| DenseLoad::mount(device.library).operand(Operand {
        device: device.id, buffer: buffer.raw(), lands, bytes: offset }))?;
    Ok(buffer)
}

impl<'a> Weights<'a> {
    pub fn load(device: Device<'a>, checkpoint: &Checkpoint, commands: &Commands<'a>) -> Result<Self> {
        let layout = RankWeightLayout::from_checkpoint(checkpoint, 0, 1)?;
        let mut operands = BTreeMap::new();
        for operand in fuse_kv(&layout)? {
            if operand.name == EMBEDDING { continue; }
            cuteafd_ffi::memory_ledger::tensor(&operand.name);
            let source = load_operand(device, checkpoint, &operand)?;
            let resident = if operand.name.ends_with(".weight_scale_inv") {
                let arranged = Buffer::new(device.library, device.id, operand.allocation_bytes as usize)?;
                commands.launch("minimax_scale_prep", &[&source,&arranged], &[
                    Scalar::I32(i32::try_from(operand.shape[0])?), Scalar::I32(i32::try_from(operand.shape[1])?)])?;
                commands.drain()?;
                arranged
            } else { source };
            operands.insert(operand.name, resident);
        }
        Ok(Self { operands })
    }
    pub fn get(&self, name: &str) -> Result<&Buffer<'a>> {
        self.operands.get(name).with_context(|| format!("MiniMax missing native operand {name}"))
    }
    pub fn layer(&self, layer: usize, name: &str) -> Result<&Buffer<'a>> {
        self.get(&format!("language_model.model.layers.{layer}.{name}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_loader::plan::testing;
    #[test] fn minimax_fused_kv_and_ffn_preserve_checkpoint_order_and_bytes() -> Result<()> {
        let dir = tempfile::tempdir()?;
        testing::write_minimax_m3_snapshot(dir.path(), true, true);
        let checkpoint = Checkpoint::open(dir.path())?;
        let layout = RankWeightLayout::from_checkpoint(&checkpoint,0,1)?;
        let fused = fuse_kv(&layout)?;
        assert_eq!(layout.resident_bytes,fused.iter().map(|w|w.allocation_bytes).sum::<u64>());
        for suffix in ["weight","weight_scale_inv"] {
            let operand = fused.iter().find(|w|w.name == format!("language_model.model.layers.0.self_attn.kv_proj.{suffix}")).unwrap();
            assert_eq!(operand.sources.len(),2);
            assert!(operand.sources[0].meta.name.contains(".k_proj."));
            assert!(operand.sources[1].meta.name.contains(".v_proj."));
            assert_eq!(operand.shape[0],1024);
            let gate = fused.iter().find(|w|w.name == format!("language_model.model.layers.0.mlp.gate_up_proj.{suffix}")).unwrap();
            assert!(gate.sources[0].meta.name.contains(".gate_proj."));
            assert!(gate.sources[1].meta.name.contains(".up_proj."));
        }
        Ok(())
    }
}
