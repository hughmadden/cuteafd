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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::families::mimo_v2::{MimoV2Config, qkv::checkpoint_tp, projection::MimoProjectionRepresentation as R,
        resident::MimoResidentOptions, weight_policy::{default_policy, MimoDefaultPolicy}};

    const MIB: u64 = 1 << 20;
    const SM120_GRANULE: u64 = 2 * MIB;

    /// Header-only before/after inventory of the selected `mimo26-flash-max`
    /// layout (TP2 head split, native MTP, GPU embedding): per-tensor requests
    /// with their SM120 2 MiB rounding versus one packed arena per scope.
    #[test]
    #[ignore = "requires MiMo-V2.6-Flash-MOPD@2479e2d0 in CUTEAFD_CHECKPOINT_TEST_HUB or the Hugging Face cache"]
    fn flash_max_inventory() -> Result<()> {
        let hub = std::env::var_os("CUTEAFD_CHECKPOINT_TEST_HUB").map(std::path::PathBuf::from)
            .unwrap_or_else(|| crate::default_hf_home().join("hub"));
        let snapshot = hub.join("models--XiaomiMiMo--MiMo-V2.6-Flash-MOPD/snapshots/2479e2d0029eca9a34cc7e7f55a121925f81908e");
        let checkpoint = Checkpoint::open(&snapshot)?;
        let cfg = MimoV2Config::from_hf(&checkpoint.config)?;
        let qualified = default_policy(&checkpoint, &cfg) == MimoDefaultPolicy::Fp8;
        let source = |name: &str| checkpoint.tensors.iter().find(|t| t.meta.name == name)
            .map(|t| if t.meta.dtype == cuteafd_core::DType::Bf16 && !qualified { R::Bf16 } else { R::Fp8 });
        let stages = checkpoint.tensors.iter().filter_map(|t|
            t.meta.name.strip_prefix("model.mtp.layers.")?.split('.').next()?.parse::<usize>().ok())
            .max().map_or(0, |stage| stage + 1);
        for ranks in [1, 2] {
            let resident = MimoResidentLayout::new(&checkpoint, &cfg, &MimoResidentOptions {
                layers: cfg.layers, coordinator_ranks: ranks, checkpoint_tp: checkpoint_tp(&snapshot)?,
                native_mtp_layers: stages, gpu_embedding: true,
                head_format: source("lm_head.weight").unwrap_or(R::Bf16),
                output_formats: checkpoint.tensors.iter()
                    .filter(|t| t.meta.name.ends_with(".self_attn.o_proj.weight"))
                    .filter_map(|t| source(&t.meta.name).map(|f| (t.meta.name.clone(), f))).collect(),
            })?;
            let layouts = MimoArenaLayouts::new(&checkpoint, &resident)?;
            for (rank, requests) in resident.ranks.iter().enumerate() {
                for (scope, mtp) in [("target", false), ("mtp", true)] {
                    let requests = requests.iter().filter(|r| r.name.starts_with("model.mtp.") == mtp)
                        .map(|r| r.bytes).collect::<Vec<_>>();
                    let layout = if mtp { if rank == 0 { layouts.mtp.as_ref() } else { None } }
                        else { Some(&layouts.target[rank]) };
                    let Some(layout) = layout else { assert!(requests.is_empty()); continue };
                    let large = requests.iter().filter(|&&b| b > MIB).collect::<Vec<_>>();
                    let rounding = large.iter().map(|&&b| b.div_ceil(SM120_GRANULE) * SM120_GRANULE - b).sum::<u64>();
                    let bytes = layout.bytes() as u64;
                    let tail = if bytes > MIB { bytes.div_ceil(SM120_GRANULE) * SM120_GRANULE - bytes } else { 0 };
                    assert_eq!(layout.views().len(), requests.len(), "unique final keys");
                    assert_eq!(layout.payload_bytes() as u64, requests.iter().sum::<u64>());
                    eprintln!("tp{ranks} rank{rank} {scope}: before n={} sum={} large n={} sum={} sm120_rounding={} \
                        | after arena={} padding={} tail={} saved={}",
                        requests.len(), requests.iter().sum::<u64>(), large.len(), large.iter().copied().sum::<u64>(),
                        rounding, bytes, layout.padding_bytes(), tail, rounding as i64 - (layout.padding_bytes() as u64 + tail) as i64);
                }
            }
        }
        Ok(())
    }
}
