//! Exact native K6 storage beside the Flash execution slots, and the staged KDA of KX7 layers
//! (exact Huffman-packed BF16 q/k/v/o, `--kda-exact kx7`), which runs the same stage programs.
use super::{core::CoreInventory, GlmNextAttention, GlmNextConfig};
use crate::serving_capacity::GlmfKdaState;
use crate::formats::{dense_trellis::TrellisSide, trellis_core::{CoreWeightLayout, ProjectionScratchLayout}};
use anyhow::{ensure, Context, Result};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoreStageLayout { pub extents: BTreeMap<&'static str, usize> }
impl CoreStageLayout {
    pub fn new(cfg: &GlmNextConfig, rows: usize, ranks: usize, owned: bool, scratch: usize) -> Result<Self> {
        ensure!((1..=4096).contains(&rows) && matches!(ranks, 1 | 2), "Flash K6 stage geometry");
        let d = cfg.kda_heads * cfg.kda_head_dim / ranks;
        let heads = cfg.heads / if owned { 1 } else { ranks };
        let inter = cfg.dense_intermediate.max(cfg.moe_intermediate) / ranks;
        let bf16 = |n: usize| rows.checked_mul(n).and_then(|n| n.checked_mul(2)).context("Flash K6 stage overflow");
        Ok(Self { extents: BTreeMap::from([
            ("q", bf16(d.max(heads * cfg.qk_nope_dim).max(cfg.q_lora_rank))?),
            ("k", bf16(d.max(cfg.kv_lora_rank))?), ("v", bf16(d)?),
            ("values", bf16(d.max(heads * cfg.v_head_dim))?),
            ("gate", bf16(inter)?), ("up", bf16(inter)?), ("hidden", bf16(inter)?),
            ("scratch", scratch.max(256)),
        ]) })
    }
    /// The stages of KX7 KDA layers alone: their q, k and v projections, the core's values and its
    /// program scratch (no MLA or FFN stage runs staged beside them).
    pub fn kda(cfg: &GlmNextConfig, rows: usize, scratch: usize) -> Result<Self> {
        ensure!((1..=4096).contains(&rows), "Flash KX7 stage geometry");
        let d = cfg.kda_heads * cfg.kda_head_dim;
        let bf16 = || rows.checked_mul(d).and_then(|n| n.checked_mul(2)).context("Flash KX7 stage overflow");
        Ok(Self { extents: BTreeMap::from([("q", bf16()?), ("k", bf16()?), ("v", bf16()?), ("values", bf16()?),
            ("scratch", scratch.max(256))]) })
    }
    pub fn bytes(&self) -> Result<usize> {
        self.extents.values().try_fold(0usize, |n, &v| n.checked_add(v).context("Flash K6 stage sum overflow"))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoreRankLayout {
    pub body: Vec<CoreWeightLayout>,
    pub stages: Option<[CoreStageLayout; 2]>,
    pub decode: Option<ProjectionScratchLayout>,
    pub prefill: Option<ProjectionScratchLayout>,
    pub target_head: Option<ProjectionScratchLayout>,
    pub draft_head: Option<ProjectionScratchLayout>,
    pub retained_rows: usize,
    pub receive_suffix: usize,
    /// KX7 layers' one decode workspace (the largest decoded projection), reused in order on the
    /// rank's stream by every KX7 linear; 0 without KX7 layers.
    pub kx7_workspace: usize,
}
impl CoreRankLayout {
    pub fn bytes(&self, hidden: usize) -> Result<u64> {
        let mut bytes = self.receive_suffix.checked_add(self.kx7_workspace).context("Flash KX7 workspace overflow")?;
        for layout in self.decode.iter().chain(&self.prefill).chain(&self.target_head).chain(&self.draft_head) {
            bytes = bytes.checked_add(layout.bytes()?).context("Flash K6 scratch sum overflow")?;
        }
        if let Some(stages) = &self.stages { for stage in stages {
            bytes = bytes.checked_add(stage.bytes()?).context("Flash K6 stage sum overflow")?;
        } }
        let retained = self.retained_rows.checked_mul(hidden).and_then(|n| n.checked_mul(8))
            .context("Flash K6 retained rows overflow")?;
        Ok(bytes.checked_add(retained).context("Flash K6 allocation total overflow")? as u64)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoreCharges { pub ranks: Vec<CoreRankLayout>, pub programs: Vec<String> }

/// Whole MLA stages exist only when an admitted layer actually contains MLA.
pub fn owned_mla(cfg: &GlmNextConfig, layers: usize, ranks: usize, whole: bool) -> bool {
    ranks == 2 && whole && cfg.attention[..layers].contains(&super::GlmNextAttention::Mla)
}

pub fn receive_suffix_bytes(rows: usize, hidden: usize, waves: usize, output_shard: bool) -> Result<usize> {
    (if output_shard { 6usize } else { 4 }).checked_mul(waves)
        .and_then(|n| n.checked_mul(rows)).and_then(|n| n.checked_mul(hidden))
        .and_then(|n| n.checked_mul(2)).context("Flash K6 receive overflow")
}
impl CoreCharges {
    #[allow(clippy::too_many_arguments)]
    pub fn from_package(cfg: &GlmNextConfig, core: &CoreInventory, layers: usize, ranks: usize,
        owners: Option<&[u8]>, unit: usize, slots: usize, decode_rows: usize,
        draft_rows: Option<usize>, output_shard: bool, manifest: &serde_json::Value) -> Result<Self> {
        ensure!(layers > 0 && layers <= cfg.layers && matches!(ranks, 1 | 2)
            && (1..=4096).contains(&unit) && (1..=4).contains(&slots)
            && matches!(decode_rows, 64 | 128) && (ranks == 1 || decode_rows == 64)
            && owners.is_none_or(|o| ranks == 2 && o.len() == layers), "Flash K6 admission geometry");
        let owned = owned_mla(cfg, layers, ranks, owners.is_some());
        let mut bodies = vec![Vec::new(); ranks];
        let mut native_names = vec![[BTreeSet::new(), BTreeSet::new()]; ranks];
        let mut bf16_shapes = BTreeSet::new();
        for layer in 0..layers {
            let owner = owners.and_then(|o| (cfg.attention[layer] == super::GlmNextAttention::Mla).then_some(o[layer]));
            for (rank, weights) in core.layer_weights(cfg, layer, ranks, owner)?.into_iter().enumerate() {
                bodies[rank].extend(weights.into_values());
            }
            for (rank, weights) in core.layer_native(cfg, layer, ranks, owner)?.into_iter().enumerate() {
                for weight in weights.values() {
                    if !weight.fp8 {
                        bf16_shapes.insert((weight.shape[1], weight.shape[0]));
                        continue;
                    }
                    let prefix = if ranks == 1 || owner.is_some() { "glmf_stage" } else { "glmf2_stage" };
                    for (index, caps) in [vec![64, decode_rows], vec![4096]].into_iter().enumerate() {
                        for cap in caps {
                            native_names[rank][index].insert(fp8_projection_name(prefix, weight.shape, cap));
                        }
                    }
                }
            }
        }
        let head = core.get_weight("lm_head").map(|p| CoreWeightLayout::new(p, TrellisSide::Output, 0, 1)).transpose()?;
        let mut programs = BTreeSet::new();
        for (k, n) in bf16_shapes {
            programs.insert(cuteafd_ffi::draft_projection::validate_package_geometry(
                manifest, k, n, unit.max(decode_rows))?);
        }
        for weight in bodies.iter().flatten().chain(head.iter()) {
            let book = weight.codebook.marker();
            programs.extend(cuteafd_ffi::dense_trellis::validate_package_geometry(manifest,
                weight.shape[1], weight.shape[0], if weight.side == TrellisSide::Input { 2 } else { 1 }, book,
                if weight.stem == "lm_head" { decode_rows.max(draft_rows.unwrap_or(1)) } else { unit.max(decode_rows) },
                if weight.stem == "lm_head" { cuteafd_ffi::dense_trellis::TrellisDType::F32 }
                else { cuteafd_ffi::dense_trellis::TrellisDType::Bf16 })?);
        }
        let entries = manifest["programs"].as_array().context("Flash K6 manifest programs")?;
        let lookup = |name: &str| -> Result<usize> {
            let matches = entries.iter().filter(|p| p["name"] == name).collect::<Vec<_>>();
            ensure!(matches.len() == 1, "Flash K6 requires one {name}");
            Ok(matches[0]["scratch_bytes_at_capacity"]["scratch"].as_u64().unwrap_or(0).try_into()?)
        };
        let scratch = |rank: usize, decode: bool| -> Result<usize> {
            let mut bytes = 0;
            for name in route_stage_names(cfg, ranks, owned, decode, decode_rows).iter()
                .chain(native_names[rank][usize::from(!decode)].iter()) {
                bytes = bytes.max(lookup(name)?);
            }
            Ok(bytes)
        };
        if bodies.iter().any(|b| !b.is_empty()) {
            for decode in [true, false] {
                for name in route_stage_names(cfg, ranks, owned, decode, decode_rows) {
                    lookup(&name)?; programs.insert(name);
                }
            }
        }
        for name in native_names.iter().flatten().flatten() { lookup(name)?; programs.insert(name.clone()); }
        let projection = |body: &[CoreWeightLayout], rows| -> Result<Option<ProjectionScratchLayout>> {
            let mut result: Option<ProjectionScratchLayout> = None;
            for w in body {
                let layout = ProjectionScratchLayout::new(w.shape[1], w.shape[0], rows)?;
                if let Some(result) = &mut result { result.merge(&layout)?; } else { result = Some(layout); }
            }
            Ok(result)
        };
        let head_scratch = |rows| head.as_ref().map(|h| ProjectionScratchLayout::new(h.shape[1], h.shape[0], rows)).transpose();
        let split = bodies.iter().flatten().any(|w| w.side == TrellisSide::Input);
        let retained_rows = decode_rows.checked_add(unit.checked_mul(slots).context("Flash K6 live rows overflow")?)
            .context("Flash K6 live rows overflow")?;
        let receive = receive_suffix_bytes(unit.max(decode_rows), cfg.hidden, slots, output_shard)?;
        let mut layouts = Vec::new();
        for (rank, body) in bodies.into_iter().enumerate() {
            layouts.push(CoreRankLayout {
                stages: if body.is_empty() { None } else { Some([
                    CoreStageLayout::new(cfg, decode_rows, ranks, owned, scratch(rank, true)?)?,
                    CoreStageLayout::new(cfg, unit, ranks, owned, scratch(rank, false)?)?]) },
                decode: projection(&body, decode_rows)?, prefill: projection(&body, unit)?,
                target_head: if rank == 0 { head_scratch(decode_rows)? } else { None },
                draft_head: if rank == 0 { draft_rows.map(head_scratch).transpose()?.flatten() } else { None },
                retained_rows: if body.is_empty() { 0 } else { retained_rows },
                receive_suffix: if split { receive } else { 0 }, body, kx7_workspace: 0,
            });
        }
        let result = Self { ranks: layouts, programs: programs.into_iter().collect() };
        result.bytes(cfg.hidden)?;
        Ok(result)
    }
    pub fn bytes(&self, hidden: usize) -> Result<Vec<u64>> { self.ranks.iter().map(|r| r.bytes(hidden)).collect() }

    /// The staged KDA of KX7 layers (`--kda-exact kx7`) on one GPU, `None` without a KDA layer
    /// in the first `layers`: no trellis owner, the KDA stages of `decode_rows` and `unit` rows
    /// with the largest scratch of their core programs (`scratch(name)`, the manifest's
    /// `scratch_bytes_at_capacity`), and one decode workspace of the largest KDA projection.
    pub fn kx7_kda(cfg: &GlmNextConfig, layers: usize, ranks: usize, unit: usize, decode_rows: usize,
        state: GlmfKdaState, scratch: impl Fn(&str) -> Result<usize>) -> Result<Option<Self>> {
        ensure!(ranks == 1, "--kda-exact kx7 serves one coordinator GPU (the two-GPU head split is not supported yet)");
        ensure!(layers > 0 && layers <= cfg.layers && (1..=4096).contains(&unit) && matches!(decode_rows, 64 | 128),
            "Flash KX7 admission geometry");
        if !cfg.attention[..layers].contains(&GlmNextAttention::Kda) { return Ok(None); }
        let mut stages = Vec::new();
        let mut programs = Vec::new();
        for (decode, rows) in [(true, decode_rows), (false, unit)] {
            let mut bytes = 0;
            for name in kx7_stage_names(state, decode, decode_rows) {
                bytes = bytes.max(scratch(&name)?);
                programs.push(name);
            }
            stages.push(CoreStageLayout::kda(cfg, rows, bytes)?);
        }
        let d = cfg.kda_heads * cfg.kda_head_dim;
        let workspace = cuteafd_core::kx7::decoded_bytes(d, cfg.hidden).max(cuteafd_core::kx7::decoded_bytes(cfg.hidden, d));
        let stages: [CoreStageLayout; 2] = stages.try_into().map_err(|_| anyhow::anyhow!("Flash KX7 stages"))?;
        programs.sort(); programs.dedup();
        let result = Self { ranks: vec![CoreRankLayout { body: vec![], stages: Some(stages), decode: None, prefill: None,
            target_head: None, draft_head: None, retained_rows: 0, receive_suffix: 0,
            kx7_workspace: usize::try_from(workspace)? }], programs };
        result.bytes(cfg.hidden)?;
        Ok(Some(result))
    }

    /// Whether these charges stage KX7 KDA layers (no trellis owner).
    pub fn kx7(&self) -> bool { self.ranks.iter().any(|r| r.kx7_workspace > 0) }
}

/// The staged KDA core program, without its `glmf_stage_` prefix, for a KX7 layer's recurrent
/// state at capacity `cap` (`m64`, `m128`, `m4096`): the F32 state's `kda_core`, or the BF16
/// state's `kda_core_s16` (`kda_core_s16t_m4096` for the tile-rounded prefill), as the fused
/// `kda` programs are named (`GlmfKdaState::program`).
pub fn kx7_kda_core(state: GlmfKdaState, cap: &str) -> String {
    match (state, cap) {
        (GlmfKdaState::F32, _) => format!("kda_core_{cap}"),
        (GlmfKdaState::Bf16Tile, "m4096") => format!("kda_core_s16t_{cap}"),
        _ => format!("kda_core_s16_{cap}"),
    }
}

/// Every staged KDA core a KX7 decode (`m64`, and `m128` past 64 rows) or prefill (`m4096`)
/// step can launch, with its `glmf_stage_` prefix.
pub fn kx7_stage_names(state: GlmfKdaState, decode: bool, decode_rows: usize) -> Vec<String> {
    let caps: &[&str] = if !decode { &["m4096"] } else if decode_rows > 64 { &["m64", "m128"] } else { &["m64"] };
    caps.iter().map(|cap| format!("glmf_stage_{}", kx7_kda_core(state, cap))).collect()
}

/// `scratch_bytes_at_capacity.scratch` of the one manifest program `name`, for KX7 charges.
pub fn kx7_manifest_scratch(manifest: &serde_json::Value, name: &str) -> Result<usize> {
    let entries = manifest["programs"].as_array().context("PROGRAMS.json has no program list")?;
    let matches = entries.iter().filter(|p| p["name"] == name).collect::<Vec<_>>();
    ensure!(matches.len() == 1, "--kda-exact kx7 runs the staged KDA core {name}, which this build does not export \
        (a build exporting the glmf_stage programs does)");
    Ok(matches[0]["scratch_bytes_at_capacity"]["scratch"].as_u64().unwrap_or(0).try_into()?)
}

pub fn fp8_projection_name(prefix: &str, [n, k]: [usize; 2], cap: usize) -> String {
    format!("{prefix}_fp8_linear_k{k}_n{n}_m{cap}")
}

/// Decode and prefill stages differ even when their live row counts coincide.
pub fn route_stage_names(cfg: &GlmNextConfig, ranks: usize, owned: bool, decode: bool, decode_rows: usize) -> Vec<String> {
    let mut names = stage_names(cfg, ranks, owned, if decode { "m64" } else { "m4096" });
    if decode && decode_rows > 64 { names.extend(stage_names(cfg, ranks, owned, "m128")); }
    names.sort(); names.dedup(); names
}

/// Stage package names are separate from the existing fused programs.
fn stage_names(cfg: &GlmNextConfig, ranks: usize, owned: bool, cap: &str) -> Vec<String> {
    let mut names = Vec::new();
    for prefix in if ranks == 1 { vec!["glmf_stage"] } else if owned { vec!["glmf_stage", "glmf2_stage"] }
        else { vec!["glmf2_stage"] } {
        for stage in ["kda_core", "rank_norm", "absorb", "values"] {
            names.push(format!("{prefix}_{stage}_{cap}"));
        }
    }
    let prefix = if ranks == 2 { "glmf2_stage" } else { "glmf_stage" };
    for inter in [cfg.dense_intermediate / ranks, cfg.moe_intermediate / ranks] {
        names.push(format!("{prefix}_swiglu_i{inter}_{cap}"));
    }
    names
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn prefill_uses_its_stage_route_even_at_decode_width() -> Result<()> {
        let cfg = GlmNextConfig::from_hf(&crate::plan::testing::glm5_flash_config(45))?;
        let decode = route_stage_names(&cfg, 2, true, true, 64);
        let prefill = route_stage_names(&cfg, 2, true, false, 64);
        assert!(decode.iter().all(|n| n.ends_with("_m64")));
        assert!(prefill.iter().all(|n| n.ends_with("_m4096")));
        assert!(prefill.iter().any(|n| n.starts_with("glmf_stage_rank_norm")));
        assert!(prefill.iter().any(|n| n.starts_with("glmf2_stage_kda_core")));
        let wide = route_stage_names(&cfg, 1, false, true, 128);
        assert!(wide.iter().any(|n| n.ends_with("_m64")));
        assert!(wide.iter().any(|n| n.ends_with("_m128")));
        let full = CoreStageLayout::new(&cfg, 64, 2, true, 256)?;
        assert!(full.extents["values"] >= 64 * cfg.heads * cfg.v_head_dim * 2);
        Ok(())
    }

    #[test]
    fn exact_native_bytes_include_every_owner_and_no_retained_alias() -> Result<()> {
        let p = ProjectionScratchLayout::new(4096, 8192, 64)?;
        let cfg = GlmNextConfig::from_hf(&crate::plan::testing::glm5_flash_config(45))?;
        let stage = CoreStageLayout::new(&cfg, 64, 2, false, 4096)?;
        let rank = CoreRankLayout { body: vec![], stages: Some([stage.clone(), stage.clone()]),
            decode: Some(p.clone()), prefill: Some(p.clone()), target_head: Some(p.clone()), draft_head: Some(p.clone()),
            retained_rows: 64 + 3 * 4096, receive_suffix: 4 * 3 * 4096 * 4096 * 2, kx7_workspace: 0 };
        assert_eq!(rank.bytes(4096)?, (4 * p.bytes()? + 2 * stage.bytes()? + rank.receive_suffix
            + rank.retained_rows * 4096 * 8) as u64);
        let mut overflow = rank; overflow.retained_rows = usize::MAX;
        assert!(overflow.bytes(4096).is_err());
        Ok(())
    }

    #[test]
    fn kx7_charges_stage_kda_alone_with_one_decode_workspace() -> Result<()> {
        let cfg = GlmNextConfig::from_hf(&crate::plan::testing::glm5_flash_config(45))?;
        let scratch = |name: &str| Ok(if name.ends_with("_m4096") { 9000 } else if name.ends_with("_m128") { 700 } else { 500 });
        for (state, decode_rows, decode, prefill) in [
            (GlmfKdaState::F32, 64, vec!["glmf_stage_kda_core_m64"], vec!["glmf_stage_kda_core_m4096"]),
            (GlmfKdaState::F32, 128, vec!["glmf_stage_kda_core_m128", "glmf_stage_kda_core_m64"], vec!["glmf_stage_kda_core_m4096"]),
            (GlmfKdaState::Bf16, 64, vec!["glmf_stage_kda_core_s16_m64"], vec!["glmf_stage_kda_core_s16_m4096"]),
            (GlmfKdaState::Bf16Tile, 64, vec!["glmf_stage_kda_core_s16_m64"], vec!["glmf_stage_kda_core_s16t_m4096"])] {
            let charges = CoreCharges::kx7_kda(&cfg, cfg.layers, 1, 4096, decode_rows, state, scratch)?.unwrap();
            assert!(charges.kx7());
            let mut names: Vec<_> = decode.iter().chain(&prefill).map(|n| n.to_string()).collect();
            names.sort();
            assert_eq!(charges.programs, names);
            let rank = &charges.ranks[0];
            let [d, p] = rank.stages.clone().unwrap();
            let width = cfg.kda_heads * cfg.kda_head_dim;
            assert_eq!(d.extents.keys().copied().collect::<Vec<_>>(), ["k", "q", "scratch", "v", "values"]);
            assert_eq!((d.extents["q"], d.extents["values"]), (decode_rows * width * 2, decode_rows * width * 2));
            assert_eq!((p.extents["v"], p.extents["scratch"]), (4096 * width * 2, 9000));
            assert_eq!(d.extents["scratch"], if decode_rows > 64 { 700 } else { 500 });
            assert_eq!(rank.kx7_workspace, 64 << 20);
            assert!(rank.body.is_empty() && rank.retained_rows == 0 && rank.decode.is_none() && rank.prefill.is_none());
            assert_eq!(rank.bytes(cfg.hidden)?, (d.bytes()? + p.bytes()? + (64 << 20)) as u64);
        }
        assert!(CoreCharges::kx7_kda(&cfg, cfg.layers, 2, 4096, 64, GlmfKdaState::F32, scratch).unwrap_err()
            .to_string().contains("one coordinator GPU"));
        let missing = CoreCharges::kx7_kda(&cfg, cfg.layers, 1, 4096, 64, GlmfKdaState::Bf16,
            |name| kx7_manifest_scratch(&serde_json::json!({"programs": [{"name": "glmf_stage_kda_core_m64"}]}), name));
        assert!(missing.unwrap_err().to_string().contains("glmf_stage_kda_core_s16_m64"));
        if cfg.attention[0] == GlmNextAttention::Mla {
            assert!(CoreCharges::kx7_kda(&cfg, 1, 1, 4096, 64, GlmfKdaState::F32, scratch)?.is_none());
        }
        let manifest = serde_json::json!({"programs": [
            {"name": "glmf_stage_kda_core_m64", "scratch_bytes_at_capacity": {"scratch": 1234}}]});
        assert_eq!(kx7_manifest_scratch(&manifest, "glmf_stage_kda_core_m64")?, 1234);
        Ok(())
    }
}
