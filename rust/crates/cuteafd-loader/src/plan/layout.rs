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

fn family_costs(family: &str) -> FamilyCosts {
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
pub fn layout(report: &PlanReport, model: &dyn super::FamilyModel, options: &LayoutOptions) -> MemoryLayout {
    let family = report.family.as_deref().unwrap_or("unknown");
    let costs = family_costs(family);
    let gpus = options.rtx_bytes.len().clamp(1, 2);
    let split = gpus == 2 && options.head_split;
    let mut devices: Vec<DeviceLayout> = options.rtx_bytes.iter().take(gpus).enumerate()
        .map(|(index, &bytes)| DeviceLayout { kind: DeviceKind::Rtx, index: index as u32,
            capacity_bytes: bytes.saturating_sub(options.headroom_bytes), items: Vec::new(), kv_tokens: 0 })
        .collect();
    let mut waste = Vec::new();
    let mut notes = Vec::new();

    // Coordinator weights.
    for component in report.components.iter().filter(|c| c.owner == Owner::Rtx && c.status != Status::Unused) {
        let resident = (component.bytes as f64 * costs.resident_factor) as u64;
        let format = component.formats.keys().cloned().collect::<Vec<_>>().join("+");
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
        if stored > even + 64 * MIB {
            waste.push(Waste { device: format!("spark x{ranks}"), what: format!("routed slices padded to the widest \
                128-row slice ({:.1}% of the even share) on every rank", 100.0 * (stored - even) as f64 / even as f64),
                bytes: (stored - even) * ranks as u64 });
        }
    }
    MemoryLayout { devices, pool_tokens, waste, notes }
}
