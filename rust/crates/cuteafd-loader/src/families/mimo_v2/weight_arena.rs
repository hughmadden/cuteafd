//! Final MiMo operand requests, shared by packing and the header-only inventory.
use super::resident::MimoResidentLayout;
use crate::{plan::checkpoint::Checkpoint, serving_capacity::WeightArenaLayout};
use anyhow::Result;

#[derive(Debug)]
pub struct MimoArenaLayouts {
    pub target: Vec<WeightArenaLayout>,
    pub mtp: Option<WeightArenaLayout>,
}

/// Concatenated sources keep the same identity in mount and recorded delivery.
pub fn operand_key(names: impl IntoIterator<Item = impl AsRef<str>>) -> String {
    let mut names = names.into_iter().map(|name| name.as_ref().to_owned()).collect::<Vec<_>>();
    names.dedup();
    names.join("+")
}

impl MimoArenaLayouts {
    pub fn new(checkpoint: &Checkpoint, resident: &MimoResidentLayout) -> Result<Self> {
        let mut target = Vec::new();
        let mut mtp = Vec::new();
        for (rank, requests) in resident.ranks.iter().enumerate() {
            let mut core = Vec::new();
            for request in requests {
                let key = final_key(checkpoint, &request.name);
                let entry = (key, usize::try_from(request.bytes)?);
                if request.name.starts_with("model.mtp.") { mtp.push(entry); }
                else { core.push(entry); }
            }
            target.push(WeightArenaLayout::new(format!("mimo.target.rank{rank}"), core)?);
        }
        Ok(Self { target, mtp: if mtp.is_empty() { None } else {
            Some(WeightArenaLayout::new("mimo.mtp.rank0", mtp)?)
        } })
    }
}

fn final_key(checkpoint: &Checkpoint, name: &str) -> String {
    if name == "model.embed_tokens.weight" { return "embedding.table".into(); }
    if let Some((sink, _)) = name.rsplit_once(".rank") {
        if sink.ends_with("attention_sink_bias") { return sink.into(); }
    }
    for suffix in [".fp8", ".row_scale", ".k_scale", ".bf16"] {
        if let Some(stem) = name.strip_suffix(suffix) {
            let source = if stem == "lm_head" { "lm_head.weight".into() }
            else if let Some(p) = stem.strip_suffix(".qkv") {
                let fused = format!("{p}.self_attn.qkv_proj.weight");
                if checkpoint.contains_tensor(&fused) { fused } else {
                    operand_key([format!("{p}.self_attn.q_proj.weight"),
                        format!("{p}.self_attn.k_proj.weight"), format!("{p}.self_attn.v_proj.weight")])
                }
            } else if let Some(p) = stem.strip_suffix(".gate_up") {
                operand_key([format!("{p}.mlp.gate_proj.weight"), format!("{p}.mlp.up_proj.weight")])
            } else if let Some(p) = stem.strip_suffix(".down") { format!("{p}.mlp.down_proj.weight") }
            else if let Some(p) = stem.strip_suffix(".o_proj") { format!("{p}.self_attn.o_proj.weight") }
            else { unreachable!("unexpected MiMo resident operand {name}") };
            return if suffix == ".bf16" { source } else { format!("{source}{suffix}") };
        }
    }
    name.into()
}
