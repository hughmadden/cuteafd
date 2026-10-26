//! Kimi K3 (`kimi_k3` wrapping `text_config.model_type=kimi_linear`) checkpoint readers.
pub mod allocation;
pub mod config;
pub mod core;
pub mod kda;
pub mod mla;
pub mod reference;
pub mod w8;
pub mod weight_layout;

pub const COMPILED_CONTEXT: usize = 1_048_576;
/// Default context for a BF16 (K2-class) core: one full-length conversation
/// would leave little pool beside the BF16 core weights.
pub const DEFAULT_CONTEXT: usize = 524_288;

/// The default API/allocation context from the checkpoint's own core: a
/// header-native K6 core frees the memory for the full compiled 1M extent with
/// the full pool and onboard experts (TJ, 2026-10-22); BF16 cores keep 512K.
pub fn default_context(core: &core::CoreInventory) -> usize {
    if core.projections.is_empty() { DEFAULT_CONTEXT } else { COMPILED_CONTEXT }
}
pub const SPARK_DEFAULT_BUDGET_BYTES: u64 = 108 << 30;

/// Offline ceiling only; the first full worker load/steady census qualifies it.
pub const SPARK_ESTIMATED_WORKSPACE_BYTES: u64 = 512 << 20;
pub const SPARK_ESTIMATED_RESERVE_BYTES: u64 =
    cuteafd_core::memory_layout::SPARK_RUNTIME_RESIDUAL_BYTES + SPARK_ESTIMATED_WORKSPACE_BYTES;

/// Target taps for one prefill call (`rows` lane rows) while the common MLA
/// drafter consumes 2048-row slices.
pub fn draft_bridge_demands(layout: &crate::formats::dflash::MlaDraftRuntimeLayout, rows: usize)
    -> anyhow::Result<Vec<crate::placement::Demand>> {
    use anyhow::Context;
    let mut demands = crate::placement::draft::remote_dflash_bridge_demands(
        layout.target_hidden() as u64, layout.target_vocab() as u64,
        layout.target_taps().len() as u64, layout.capacity().block_rows as u64, 1)?.demands;
    let taps = demands.iter_mut().find(|d| d.group == "DFlash bridge taps")
        .context("Kimi bridge tap inventory missing")?;
    taps.bytes = (rows * layout.target_taps().len() * layout.target_hidden() * 2) as u64;
    Ok(demands)
}

/// The shared probe retains one GPU0 route arena; its pinned mirror is host memory.
pub fn draft_probe_demand(layers: usize, topk: usize, rows: usize) -> anyhow::Result<crate::placement::Demand> {
    use anyhow::Context;
    use crate::placement::Demand;
    use cuteafd_core::memory_layout::{Basis, Category};
    let bytes = layers.checked_mul(topk).and_then(|n| n.checked_mul(rows))
        .and_then(|n| n.checked_mul(4)).context("Kimi draft probe extent overflow")?;
    Ok(Demand::new(0, Category::Workspace, "draft policy route arena", bytes.max(256) as u64, Basis::Formula))
}

/// Validate sealed role artifacts at runtime, never during checkpoint planning.
pub fn validate_expert_inventory(path: &std::path::Path, root: &std::path::Path) -> anyhow::Result<()> {
    use crate::plan::checkpoint::read_json;
    use anyhow::{ensure, Context};
    let programs = read_json(path)?;
    ensure!(programs["schema"] == 1 && programs["programs"].is_array(), "invalid PROGRAMS.json inventory");
    let package = read_json(&root.join("manifest.json"))?;
    ensure!(package["schema"] == "cuteafd.exl3-package.v1" && package["geometry"] == "kimik3",
        "Kimi K3 EXL3 package schema/geometry mismatch");
    let revision = package["sparkinfer_revision"].as_str().filter(|s| s.len() == 40)
        .context("Kimi K3 package lacks source revision")?;
    let lock: serde_json::Value = serde_json::from_str(include_str!(concat!(env!("CARGO_MANIFEST_DIR"),
        "/../../../third_party/sparkinfer.lock.json")))?;
    ensure!(lock["revision"] == revision, "Kimi K3 package/source revision mismatch");
    let (layouts, output) = match package["role"].as_str() {
        Some("spark") => ((0..6).map(|r| (format!("tp6-rank{r}"), 512)).collect::<Vec<_>>(), "bf16"),
        Some("coordinator") => {
            ensure!(programs["sparkinfer_revision"] == revision, "Kimi K3 program/package revision mismatch");
            let inventory = programs["programs"].as_array().unwrap();
            for name in ["kimik3_router_scores", "kimik3_expert_input_quant", "kimik3_latent_down",
                "kimik3_latent_up", "kimik3_rmsnorm_h3584"] {
                let entries = inventory.iter().filter(|p| p["name"] == name).collect::<Vec<_>>();
                ensure!(entries.len() == 1 && entries[0]["object_sha256"].as_str().is_some_and(|s| s.len() == 64),
                    "Kimi K3 program inventory requires one sealed {name}");
                let geometry = &entries[0]["geometry"];
                match name {
                    "kimik3_router_scores" => ensure!(geometry["hidden"] == 7168 && geometry["experts"] == 896
                        && geometry["top_k"] == 16, "Kimi router geometry mismatch"),
                    "kimik3_latent_down" => ensure!(geometry["width"] == 7168 && geometry["outputs"] == 3584,
                        "Kimi latent down geometry mismatch"),
                    "kimik3_latent_up" => ensure!(geometry["width"] == 3584 && geometry["outputs"] == 7168,
                        "Kimi latent up geometry mismatch"),
                    _ => ensure!(geometry["hidden"] == 3584, "Kimi latent input geometry mismatch"),
                }
            }
            (vec![("rtx-tp1".into(), 3072), ("rtx-tp2".into(), 1536)], "fp32")
        }
        _ => anyhow::bail!("Kimi K3 EXL3 package must declare coordinator or spark role"),
    };
    let variants = package["variants"].as_array().context("Kimi K3 package lacks variants")?;
    for (layout, width) in layouts {
        for capacity in cuteafd_core::Exl3RowPolicy::for_geometry(cuteafd_core::ExpertGeometry::KIMI_K3).capacities(4096)? {
            let directory = format!("{layout}/m{capacity}");
            let entries = variants.iter().filter(|v| v["directory"] == directory).collect::<Vec<_>>();
            ensure!(entries.len() == 1, "Kimi K3 package requires one {directory}");
            let meta = read_json(&root.join(&directory).join("v41_exl3.json"))?;
            for entry in [entries[0], &meta] {
                ensure!(entry["intermediate"] == width && entry["experts"] == 896 && entry["top_k"] == 16
                    && entry["capacity"] == capacity && entry["bits"] == serde_json::json!([2, 3])
                    && entry["output_dtype"] == output, "Kimi K3 package geometry mismatch: {directory}");
            }
            ensure!(meta["sparkinfer_revision"] == revision && meta["hidden"] == 3584
                && meta["activation"] == "situ" && meta["descriptor_bits"] == 10
                && meta["descriptor_rows"].as_u64().is_none_or(|n| n == 3)
                && meta["native_info_version"].as_u64().is_none_or(|n| n == 2)
                && meta["decode_schedule"].is_null(), "Kimi K3 EXL3 native contract mismatch: {directory}");
            for artifact in ["libcuteafd_exl3.so", "routes/libv41_exl3_routes.so", "trellis_lut.bin"] {
                ensure!(root.join(&directory).join(artifact).is_file(),
                    "Kimi K3 package missing {directory}/{artifact}");
            }
        }
    }
    Ok(())
}
