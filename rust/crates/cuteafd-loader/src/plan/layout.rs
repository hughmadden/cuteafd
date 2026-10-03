//! Placement planner core: a per-device memory layout for a checkpoint on a
//! hardware inventory (1-2 coordinator GPUs, 0-8 Spark ranks).
//!
//! Weights come from the checkpoint plan (source bytes per component) mapped
//! to each family's resident representation and head-split ownership; KV from
//! the family cache geometry; workspaces, runtime overhead and Spark buffers
//! from per-family costs calibrated against the allocation ledger
//! (`scripts/bench/memory-audit.py`, PLAN.md Phase 6 audit). The KV pool takes
//! what the tightest KV-owning device has left, capped at the target.
use super::{Component, ExpertPlacement, Owner, PlanReport, Status};
use crate::serving_capacity::{CacheOptions, KvPlacement};
use cuteafd_core::memory_layout::{size_pool, Basis, Category, DeviceKind, DeviceLayout, Item, MemoryLayout, Waste};

const GIB: u64 = 1 << 30;
const MIB: u64 = 1 << 20;

/// The hardware and serving shape to lay out.
#[derive(Debug, Clone)]
pub struct LayoutOptions {
    /// Usable bytes of each coordinator GPU (1 or 2).
    pub rtx_bytes: Vec<u64>,
    /// Usable bytes of one Spark rank (unified memory).
    pub spark_bytes: u64,
    /// Two coordinator GPUs split attention heads (generic families) rather
    /// than V4.1's layer ranges.
    pub head_split: bool,
    /// Prefill rows per step (workspace shape).
    pub prefill_rows: u64,
    /// Spark wave capacity in rows (`expertd --capacity`).
    pub spark_capacity_rows: u64,
    /// Explicit pool tokens; `None` sizes the pool from what is left.
    pub pool_tokens: Option<u64>,
    /// Upper bound for an automatically sized pool.
    pub target_pool_tokens: u64,
    /// External drafter resident bytes on the lead GPU (DFlash), if any.
    pub drafter_bytes: u64,
    /// Keep this much of every GPU free for runtime growth.
    pub headroom_bytes: u64,
}

impl Default for LayoutOptions {
    fn default() -> Self {
        Self {
            rtx_bytes: vec![95 * GIB + 512 * MIB],
            // 121.7 GiB GB10 minus the host OS and sparknestd measured idle (~13 GiB).
            spark_bytes: 108 * GIB,
            head_split: true,
            prefill_rows: 4096,
            spark_capacity_rows: 4096,
            pool_tokens: None,
            target_pool_tokens: cuteafd_core::serving_capacity::DEFAULT_GPU_KV_TOKENS,
            drafter_bytes: 0,
            headroom_bytes: 2 * GIB,
        }
    }
}

/// How a component's weights sit on two coordinator GPUs.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Share {
    /// Lead GPU only (embedding, head, router, speculators).
    Lead,
    /// A full copy on every GPU (norms, small operands).
    Replicated,
    /// Split by heads / intermediate rows; `replicated` of it is copied on both.
    Sharded { replicated: f64 },
}

/// Per-family costs calibrated against the allocation ledger (Phase 6 audit).
#[derive(Debug, Clone, Copy)]
pub struct FamilyCosts {
    /// CUDA context + loaded modules + cuBLAS per GPU (untracked by the ledger).
    pub runtime_bytes: u64,
    /// Lead GPU workspaces at `prefill_rows` rows (all lanes, decode, sampler, intake).
    pub lead_workspace_bytes: u64,
    /// Peer GPU workspaces under a head split.
    pub peer_workspace_bytes: u64,
    /// Retained decode/verify graph executables per GPU.
    pub graph_bytes: u64,
    /// Fraction of attention weights replicated on both GPUs under a head split.
    pub attention_replicated: f64,
    /// Resident bytes per source byte of coordinator weights (load-time conversion).
    pub resident_factor: f64,
    /// Spark worker workspace + host exchange at 4096 rows.
    pub spark_workspace_bytes: u64,
    /// RDMA rings per Spark rank (both coordinator endpoints).
    pub spark_ring_bytes: u64,
}

pub fn family_costs(family: &str) -> FamilyCosts {
    // Measured 2026-10-03 on the ledger build (see PLAN.md Phase 6 audit);
    // families without a measurement use the generic row.
    let generic = FamilyCosts {
        runtime_bytes: 1536 * MIB,
        lead_workspace_bytes: 6 * GIB,
        peer_workspace_bytes: 3 * GIB,
        graph_bytes: 640 * MIB,
        attention_replicated: 0.0,
        resident_factor: 1.0,
        spark_workspace_bytes: 160 * MIB,
        spark_ring_bytes: 1280 * MIB,
    };
    match family {
        "glm5" => FamilyCosts { attention_replicated: 0.10, ..generic },
        _ => generic,
    }
}

fn share_of(family: &str, component: Component) -> Share {
    match component {
        Component::Norm | Component::HyperConnection => Share::Replicated,
        Component::Indexer if family == "glm5" => Share::Replicated,
        Component::Attention | Component::DenseFfn | Component::SharedExpert | Component::Indexer
        | Component::Compressor => Share::Sharded { replicated: 0.0 },
        _ => Share::Lead,
    }
}

/// Lays out `report` (a `plan` of the checkpoint) on the inventory.
pub fn layout(report: &PlanReport, model: &dyn super::FamilyModel, checkpoint: &super::Checkpoint,
    options: &LayoutOptions) -> MemoryLayout {
    let family = report.family.as_deref().unwrap_or("unknown");
    let conversions = load_conversions(family, checkpoint);
    let costs = family_costs(family);
    let gpus = options.rtx_bytes.len().clamp(1, 2);
    let split = gpus == 2 && options.head_split;
    let mut devices: Vec<DeviceLayout> = options.rtx_bytes.iter().take(gpus).enumerate()
        .map(|(index, &bytes)| DeviceLayout { kind: DeviceKind::Rtx, index: index as u32,
            capacity_bytes: bytes.saturating_sub(options.headroom_bytes), items: Vec::new(), kv_tokens: 0 })
        .collect();
    let mut waste = Vec::new();
    let mut notes = Vec::new();

    // Coordinator weights: a family's exact resident layout where it has one,
    // else checkpoint bytes per component under the family's conversions.
    let exact = resident_layout(family, checkpoint, if split { 2 } else { 1 });
    if let Some(ranks) = &exact {
        for (device, rank) in devices.iter_mut().zip(ranks) {
            for (group, format, bytes) in rank {
                let category = if group == "embedding" { Category::Embedding } else { Category::Weights };
                device.items.push(Item::new(category, group.clone(), format.clone(), *bytes, Basis::Exact));
            }
        }
    }
    let covered = |c: Component| exact.is_some() && !matches!(c, Component::Speculator | Component::SpeculatorExpert
        | Component::Vision | Component::TableProjection);
    for component in report.components.iter()
        .filter(|c| c.owner == Owner::Rtx && c.status != Status::Unused && !covered(c.component)) {
        let (resident, format) = match conversions.iter().find(|c| c.component == component.component) {
            Some(c) => (component.bytes.saturating_sub(c.saved_bytes), c.format.to_string()),
            None => ((component.bytes as f64 * costs.resident_factor) as u64,
                component.formats.keys().cloned().collect::<Vec<_>>().join("+")),
        };
        let category = match component.component {
            Component::Embedding => Category::Embedding,
            Component::Speculator | Component::SpeculatorExpert => Category::Drafter,
            Component::TableProjection => Category::Tables,
            _ => Category::Weights,
        };
        let group = component.component.label();
        let share = if split { share_of(family, component.component) } else { Share::Lead };
        let attention_share = if component.component == Component::Attention { costs.attention_replicated } else { 0.0 };
        match share {
            Share::Lead => devices[0].items.push(Item::new(category, group, &format, resident, Basis::Exact)),
            Share::Replicated => {
                for device in devices.iter_mut() {
                    device.items.push(Item::new(category, group, &format, resident, Basis::Exact));
                }
                if gpus == 2 && resident >= 16 * MIB {
                    waste.push(Waste { device: "rtx1".into(), what: format!("{group} replicated on both GPUs"),
                        bytes: resident });
                }
            }
            Share::Sharded { replicated } => {
                let replicated = (resident as f64 * (replicated + attention_share)) as u64;
                let sharded = resident - replicated;
                for device in devices.iter_mut() {
                    device.items.push(Item::new(category, group, &format, sharded / gpus as u64 + replicated,
                        Basis::Exact));
                }
                if replicated >= 16 * MIB {
                    waste.push(Waste { device: "rtx1".into(), what: format!("{group}: operands replicated under \
                        the head split"), bytes: replicated });
                }
            }
        }
    }
    if options.drafter_bytes > 0 {
        let last = devices.len() - 1;
        devices[last].items.push(Item::new(Category::Drafter, "drafter", "", options.drafter_bytes, Basis::Exact));
    }

    // Fixed runtime costs.
    for (index, device) in devices.iter_mut().enumerate() {
        device.items.push(Item::new(Category::Runtime, "context+modules", "", costs.runtime_bytes, Basis::Calibrated));
        let workspace = if index == 0 { costs.lead_workspace_bytes } else { costs.peer_workspace_bytes };
        let workspace = workspace * options.prefill_rows.max(1) / 4096;
        device.items.push(Item::new(Category::Workspace, "steps", "", workspace, Basis::Calibrated));
        device.items.push(Item::new(Category::Runtime, "graphs", "", costs.graph_bytes, Basis::Calibrated));
    }

    // KV pool: per-device bytes per logical token from the family geometry.
    let geometry = model.cache_geometry(CacheOptions { coordinator_ranks: if split { 2 } else { 1 }, ..Default::default() });
    let mut pool_tokens = 0;
    match geometry {
        Ok(Some(geometry)) => {
            let unit = geometry.logical_unit_rows.max(1);
            let per_token: Vec<u64> = (0..devices.len()).map(|d| geometry.ranks.get(d)
                .map_or(0, |r| (r.persistent_unit_bytes + r.pool_metadata_unit_bytes).div_ceil(unit))).collect();
            let free: Vec<i64> = devices.iter().map(DeviceLayout::free_bytes).collect();
            pool_tokens = options.pool_tokens.unwrap_or_else(|| size_pool(&free, &per_token, unit, options.target_pool_tokens));
            for (device, (rank, &cost)) in devices.iter_mut().zip(geometry.ranks.iter().zip(&per_token)) {
                device.items.push(Item::new(Category::Kv, "records", "", cost * pool_tokens, Basis::Formula));
                device.items.push(Item::new(Category::Kv, "state", "", rank.fixed_state_bytes
                    + rank.active_state_per_sequence_bytes * 20, Basis::Formula));
                device.kv_tokens = pool_tokens;
            }
            if geometry.placement == KvPlacement::Replicated && devices.len() == 2 {
                waste.push(Waste { device: "rtx1".into(), what: "KV records replicated on both GPUs (MLA latent)".into(),
                    bytes: per_token.get(1).copied().unwrap_or(0) * pool_tokens });
            }
        }
        Ok(None) => notes.push(format!("{family}: no cache geometry in the planner yet (engine sizes its own pool)")),
        Err(error) => notes.push(format!("{family}: cache geometry: {error}")),
    }

    // Spark ranks.
    if let ExpertPlacement::Sparks { ranks } = report.placement {
        let routed: u64 = report.components.iter().filter(|c| c.owner == Owner::SparkSliced).map(|c| c.bytes).sum();
        let stored = (routed as f64 * report.spark_rank_share) as u64;
        let even = routed / ranks.max(1) as u64;
        for rank in 0..ranks {
            let mut device = DeviceLayout { kind: DeviceKind::Spark, index: rank as u32, capacity_bytes: options.spark_bytes,
                items: Vec::new(), kv_tokens: 0 };
            let format = report.components.iter().find(|c| c.owner == Owner::SparkSliced)
                .map(|c| c.formats.keys().cloned().collect::<Vec<_>>().join("+")).unwrap_or_default();
            device.items.push(Item::new(Category::Experts, "routed_expert", format, stored, Basis::Exact));
            let workspace = costs.spark_workspace_bytes * options.spark_capacity_rows / 4096;
            device.items.push(Item::new(Category::Workspace, "expert waves", "", workspace, Basis::Calibrated));
            device.items.push(Item::new(Category::Transport, "rdma rings", "", costs.spark_ring_bytes, Basis::Calibrated));
            device.items.push(Item::new(Category::Runtime, "context+modules", "", 512 * MIB, Basis::Calibrated));
            devices.push(device);
        }
        if report.spark_rank_share * ranks as f64 > 1.001 {
            waste.push(Waste { device: format!("spark x{ranks}"), what: format!("routed slices padded to the widest \
                128-row slice ({:.1}% of the even share) on every rank", 100.0 * (stored - even) as f64 / even as f64),
                bytes: (stored - even) * ranks as u64 });
        }
    }
    MemoryLayout { devices, pool_tokens, waste, notes }
}

/// A component the family loader converts at load: bytes saved against the
/// checkpoint's source storage, and the resident format.
struct Conversion {
    component: Component,
    saved_bytes: u64,
    format: &'static str,
}

fn load_conversions(family: &str, checkpoint: &super::Checkpoint) -> Vec<Conversion> {
    use crate::families::mimo_v2::projection::{MimoProjectionLayout, MimoProjectionRepresentation as R};
    use crate::families::mimo_v2::weight_policy::{default_policy, MimoDefaultPolicy};
    match family {
        // The qualified MiMo V2.6 Pro default stores the head and every target
        // o_proj as FP8 (one copy), not the checkpoint's BF16.
        "mimo_v2" => {
            let Ok(cfg) = crate::families::mimo_v2::MimoV2Config::from_hf(&checkpoint.config) else { return Vec::new() };
            if default_policy(checkpoint, &cfg) != MimoDefaultPolicy::QualifiedProFp8 {
                return Vec::new();
            }
            let bytes = |rows: usize, cols: usize, r: R, ranks: u64| MimoProjectionLayout::new(rows as u64, cols as u64, r, ranks)
                .ok().and_then(|l| l.resident_bytes().ok()).unwrap_or(0);
            let head = bytes(cfg.vocab_size, cfg.hidden, R::Bf16, 1).saturating_sub(bytes(cfg.vocab_size, cfg.hidden, R::Fp8, 1));
            let o = cfg.heads * cfg.v_head_dim;
            let output = (bytes(cfg.hidden, o, R::Bf16, 1).saturating_sub(bytes(cfg.hidden, o, R::Fp8, 1))) * cfg.layers as u64;
            vec![Conversion { component: Component::LmHead, saved_bytes: head, format: "fp8-block128" },
                Conversion { component: Component::Attention, saved_bytes: output, format: "fp8-block128" }]
        }
        _ => Vec::new(),
    }
}

/// Exact per-rank resident weights `(group, format, bytes)` for families
/// whose loader publishes its resident layout (MiMo: the codex capacity
/// contract's `MimoResidentLayout`, with the default weight policy).
fn resident_layout(family: &str, checkpoint: &super::Checkpoint, ranks: usize) -> Option<Vec<Vec<(String, String, u64)>>> {
    use crate::families::mimo_v2::projection::MimoProjectionRepresentation as R;
    use crate::families::mimo_v2::resident::{MimoResidentLayout, MimoResidentOptions};
    use crate::families::mimo_v2::weight_policy::{default_policy, MimoDefaultPolicy};
    if family != "mimo_v2" {
        return None;
    }
    let cfg = crate::families::mimo_v2::MimoV2Config::from_hf(&checkpoint.config).ok()?;
    let qualified = default_policy(checkpoint, &cfg) == MimoDefaultPolicy::QualifiedProFp8;
    let source = |name: &str| checkpoint.tensors.iter().find(|t| t.meta.name == name)
        .map(|t| if t.meta.dtype == cuteafd_core::DType::Bf16 && !qualified { R::Bf16 } else { R::Fp8 });
    let output_formats = (0..cfg.layers).filter_map(|layer| {
        let name = format!("model.layers.{layer}.self_attn.o_proj.weight");
        source(&name).map(|r| (name, r))
    }).collect();
    let options = MimoResidentOptions {
        layers: cfg.layers,
        coordinator_ranks: ranks,
        checkpoint_tp: crate::families::mimo_v2::qkv::checkpoint_tp(&checkpoint.snapshot).ok()?,
        native_mtp_layers: 0,
        gpu_embedding: true,
        head_format: source("lm_head.weight").unwrap_or(R::Bf16),
        output_formats,
    };
    let layout = MimoResidentLayout::new(checkpoint, &cfg, &options).ok()?;
    Some(layout.ranks.iter().map(|rank| {
        let mut groups: std::collections::BTreeMap<(String, String), u64> = std::collections::BTreeMap::new();
        for reservation in rank {
            let name = reservation.name.as_str();
            let group = if name.contains("embed") { "embedding" } else if name.starts_with("lm_head") { "lm_head" }
                else if name.contains("mlp.gate") { "router" } else if name.contains("norm") { "norm" }
                else if name.contains("gate_up") || name.contains(".down") || name.contains("mlp.") { "dense_ffn" }
                else { "attention" };
            let format = if name.ends_with(".fp8") { "fp8" } else if name.ends_with("scale") { "fp8-scale" }
                else if name.ends_with(".bf16") { "bf16" } else { "native" };
            *groups.entry((group.to_string(), format.to_string())).or_default() += reservation.bytes;
        }
        groups.into_iter().map(|((g, f), b)| (g, f, b)).collect()
    }).collect())
}
