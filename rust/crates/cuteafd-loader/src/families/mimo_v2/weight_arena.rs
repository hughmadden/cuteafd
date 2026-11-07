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

    fn fixture_options(ranks: usize, tp: usize, format: R) -> MimoResidentOptions {
        MimoResidentOptions {
            layers: 2, coordinator_ranks: ranks, checkpoint_tp: tp, native_mtp_layers: 0, gpu_embedding: true,
            head_format: format,
            output_formats: (0..2).map(|l| (format!("model.layers.{l}.self_attn.o_proj.weight"), format)).collect(),
        }
    }

    /// Every resident request has one 256-aligned, nonoverlapping view under a
    /// source key the loader derives from checkpoint names; no staging or
    /// transient representation enters the arena.
    fn check(checkpoint: &Checkpoint, resident: &MimoResidentLayout) -> MimoArenaLayouts {
        let layouts = MimoArenaLayouts::new(checkpoint, resident).unwrap();
        assert_eq!(layouts.target.len(), resident.ranks.len());
        for (rank, layout) in layouts.target.iter().enumerate() {
            assert_eq!(layout.id(), format!("mimo.target.rank{rank}"));
            assert_eq!(layout.views().len(), resident.ranks[rank].len(), "unique final keys");
            assert_eq!(layout.payload_bytes() as u64, resident.ranks[rank].iter().map(|r| r.bytes).sum::<u64>());
            let mut views = layout.views().iter().map(|(name, v)| (v.offset(), v.bytes(), name)).collect::<Vec<_>>();
            views.sort();
            for pair in views.windows(2) {
                assert!(pair[0].0 + pair[0].1 <= pair[1].0, "{} overlaps {}", pair[0].2, pair[1].2);
            }
            for (offset, _, name) in &views {
                assert_eq!(offset % 256, 0, "{name}");
                assert!(!name.contains("staging") && !name.ends_with("_scale_inv") && !name.contains(".bf16"), "{name}");
                assert!(!name.starts_with("model.mtp."), "{name}: MTP stays out of the target arena");
            }
            assert!(views.last().map_or(0, |v| v.0 + v.1) <= layout.bytes());
        }
        layouts
    }

    #[test]
    fn target_layouts_cover_selected_tp1_tp2_formats_with_source_keys() {
        use crate::plan::testing::{mimo_flash_config, mimo_flash_tensors, mimo_pro_config, mimo_pro_tensors, write_snapshot};
        for (config, tensors, tp) in [(mimo_flash_config(), mimo_flash_tensors(), 1), (mimo_pro_config(), mimo_pro_tensors(), 8)] {
            let dir = tempfile::tempdir().unwrap();
            write_snapshot(dir.path(), &config, &tensors, Some(tp));
            let checkpoint = Checkpoint::open(dir.path()).unwrap();
            let cfg = MimoV2Config::from_hf(&checkpoint.config).unwrap();
            for ranks in [1, 2] {
                for format in [R::Bf16, R::Fp8] {
                    let resident = MimoResidentLayout::new(&checkpoint, &cfg, &fixture_options(ranks, tp, format)).unwrap();
                    let layouts = check(&checkpoint, &resident);
                    assert!(layouts.mtp.is_none());
                    let lead = &layouts.target[0];
                    let head = if format == R::Fp8 { "lm_head.weight.fp8" } else { "lm_head.weight" };
                    assert!(lead.view(head).is_some(), "{head}");
                    assert!(lead.view("embedding.table").is_some());
                    // Both retained scale layouts of every FP8 linear (the FP8 head keeps one bank).
                    for name in lead.views().keys().filter(|n| n.ends_with(".row_scale") && !n.starts_with("lm_head")) {
                        assert!(lead.view(&name.replace(".row_scale", ".k_scale")).is_some(), "{name}");
                    }
                    let qkv = if checkpoint.contains_tensor("model.layers.0.self_attn.qkv_proj.weight") {
                        "model.layers.0.self_attn.qkv_proj.weight.fp8".to_string()
                    } else {
                        format!("{}.fp8", operand_key(["q", "k", "v"].map(|p| format!("model.layers.0.self_attn.{p}_proj.weight"))))
                    };
                    for layout in &layouts.target { assert!(layout.view(&qkv).is_some(), "{qkv}"); }
                    if ranks == 2 {
                        assert!(layouts.target[1].views().keys().all(|n| !n.starts_with("lm_head") && n != "embedding.table"));
                    }
                }
            }
        }
    }

    #[test]
    fn mtp_requests_get_their_own_rank0_arena() {
        use cuteafd_core::serving_capacity::MemoryReservation;
        let resident = MimoResidentLayout { ranks: vec![vec![
            MemoryReservation { name: "model.norm.weight".into(), bytes: 8192 },
            MemoryReservation { name: "model.mtp.layers.0.enorm.weight".into(), bytes: 8192 },
            MemoryReservation { name: "model.mtp.layers.0.gate_up.fp8".into(), bytes: 3 << 20 },
        ]], loading_temporary_rank_bytes: vec![0] };
        let dir = tempfile::tempdir().unwrap();
        use crate::plan::testing::{mimo_flash_config, mimo_flash_tensors, write_snapshot};
        write_snapshot(dir.path(), &mimo_flash_config(), &mimo_flash_tensors(), Some(1));
        let checkpoint = Checkpoint::open(dir.path()).unwrap();
        let layouts = MimoArenaLayouts::new(&checkpoint, &resident).unwrap();
        let mtp = layouts.mtp.unwrap();
        assert_eq!(mtp.id(), "mimo.mtp.rank0");
        assert_eq!(mtp.views().keys().cloned().collect::<Vec<_>>(), [
            "model.mtp.layers.0.enorm.weight".to_string(),
            operand_key(["model.mtp.layers.0.mlp.gate_proj.weight", "model.mtp.layers.0.mlp.up_proj.weight"]) + ".fp8"]);
        assert_eq!(layouts.target[0].views().keys().collect::<Vec<_>>(), ["model.norm.weight"]);
    }

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
