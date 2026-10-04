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
    /// Concurrent sequences (state slots = concurrency + 2).
    pub concurrency: u64,
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
            concurrency: 8,
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

/// Per-family costs calibrated against the allocation ledger (Phase 6 audit,
/// 2026-10-03: one launch per family at its min and max reference layouts,
/// prefill 4096 rows, after an 8K prefill, a C4 and a C1 request).
#[derive(Debug, Clone, Copy)]
pub struct FamilyCosts {
    /// CUDA context, modules and cuBLAS at ready (untracked): one GPU, lead and peer of a head split.
    pub runtime_bytes: [u64; 3],
    /// Allowance for decode/verify graph executables captured as traffic arrives
    /// (keyed by exact row counts and table widths; grows after ready).
    pub graph_bytes: [u64; 3],
    /// Step workspaces incl. sampler and Spark intake at 4096 prefill rows: one GPU, lead, peer.
    pub workspace_bytes: [u64; 3],
    /// Head-split peer exchange slots on each GPU.
    pub exchange_bytes: u64,
    /// The family's default drafter (weights and its buffers) on the lead GPU.
    pub drafter_bytes: u64,
    /// Prefix-cache mark slots resident on the device (per GPU).
    pub mark_slots: u64,
    /// Native MTP layers stay resident (false: the default drafter replaces them).
    pub mtp_resident: bool,
    /// Fraction of attention weights replicated on both GPUs under a head split.
    pub attention_replicated: f64,
    /// Resident bytes per source byte of coordinator weights (load-time conversion).
    pub resident_factor: f64,
    /// Spark worker scratch, workspace and host exchange at 4096 rows.
    pub spark_workspace_bytes: u64,
    /// RDMA rings per Spark rank (every coordinator endpoint).
    pub spark_ring_bytes: u64,
}


const fn gib(hundredths: u64) -> u64 {
    hundredths * GIB / 100
}

pub fn family_costs(family: &str) -> FamilyCosts {
    let generic = FamilyCosts {
        runtime_bytes: [gib(90), gib(90), gib(85)],
        graph_bytes: [gib(150), gib(150), gib(150)],
        workspace_bytes: [gib(500), gib(450), gib(300)],
        exchange_bytes: gib(50),
        drafter_bytes: 0,
        mark_slots: 0,
        mtp_resident: true,
        attention_replicated: 0.0,
        resident_factor: 1.0,
        spark_workspace_bytes: gib(100),
        spark_ring_bytes: gib(150),
    };
    match family {
        // GLM 5.3 EXL3 K4 + DFlash2 (BF16, 4.58 GiB checkpoint + 1.3 GiB context/buffers).
        "glm5" => FamilyCosts {
            // Untracked 0.89 / 0.82 GiB at ready, 2.62 / 2.14 after one decode+prefill
            // bench and still rising (graphs per layer x exact rows x table width).
            runtime_bytes: [gib(74), gib(89), gib(82)],
            graph_bytes: [gib(300), gib(300), gib(300)],
            workspace_bytes: [gib(651), gib(559), gib(422)],
            exchange_bytes: gib(56),
            drafter_bytes: gib(588),
            mtp_resident: false,
            attention_replicated: 0.10,
            spark_workspace_bytes: gib(190),
            spark_ring_bytes: gib(231),
            ..generic
        },
        // MiMo V2.6 Pro + embedded DFlash (qualified FP8 bundle); MTP unused.
        "mimo_v2" => FamilyCosts {
            runtime_bytes: [gib(100), gib(111), gib(100)],
            graph_bytes: [gib(50), gib(50), gib(50)],
            workspace_bytes: [gib(290), gib(249), gib(120)],
            exchange_bytes: gib(38),
            drafter_bytes: gib(321),
            mark_slots: 42,
            mtp_resident: false,
            spark_workspace_bytes: gib(51),
            spark_ring_bytes: gib(117),
            ..generic
        },
        // GLM 5.3 Flash EXL3 + DFlash2; one GPU (no head split).
        "glm5_flash" => FamilyCosts {
            runtime_bytes: [gib(78), gib(78), gib(78)],
            graph_bytes: [gib(150), gib(150), gib(150)],
            workspace_bytes: [gib(472), gib(472), gib(472)],
            drafter_bytes: gib(324),
            mark_slots: 18,
            mtp_resident: false,
            spark_workspace_bytes: gib(56),
            spark_ring_bytes: gib(78),
            ..generic
        },
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
    // Qwen executes on one GPU even when the inventory names two.
    let split = gpus == 2 && options.head_split && family != "qwen4";
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
    let covered = |c: Component| (exact.is_some() && !matches!(c, Component::Speculator | Component::SpeculatorExpert
        | Component::Vision | Component::TableProjection))
        || (!costs.mtp_resident && matches!(c, Component::Speculator | Component::SpeculatorExpert));
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
    if report.placement == ExpertPlacement::Local {
        let routed = report.components.iter().filter(|c| c.owner == Owner::SparkSliced && c.status != Status::Unused);
        for component in routed {
            devices[0].items.push(Item::new(Category::Experts, component.component.label(),
                component.formats.keys().cloned().collect::<Vec<_>>().join("+"), component.bytes, Basis::Exact));
        }
    }
    // The drafter lives on the lead GPU (taps and head are there under a head split).
    let drafter = if options.drafter_bytes > 0 { options.drafter_bytes } else { costs.drafter_bytes };
    if drafter > 0 {
        devices[0].items.push(Item::new(Category::Drafter, "drafter", "", drafter, Basis::Calibrated));
    }

    // Fixed runtime costs.
    let gpus_now = devices.len();
    for (index, device) in devices.iter_mut().enumerate() {
        let role = if gpus_now == 1 { 0 } else if index == 0 { 1 } else { 2 };
        device.items.push(Item::new(Category::Runtime, "context+modules", "", costs.runtime_bytes[role],
            Basis::Calibrated));
        device.items.push(Item::new(Category::Runtime, "graph allowance", "", costs.graph_bytes[role], Basis::Calibrated));
        let workspace = costs.workspace_bytes[role] * options.prefill_rows.max(1) / 4096;
        device.items.push(Item::new(Category::Workspace, "steps", "", workspace, Basis::Calibrated));
        if split {
            device.items.push(Item::new(Category::Transport, "peer exchange", "", costs.exchange_bytes, Basis::Calibrated));
        }
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
                    + rank.active_state_per_sequence_bytes * (options.concurrency + 2), Basis::Formula));
                if costs.mark_slots > 0 && rank.retained_mark_bytes > 0 {
                    device.items.push(Item::new(Category::Prefix, "marks", "", rank.retained_mark_bytes * costs.mark_slots,
                        Basis::Formula));
                }
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
        // EXL3 packages and FP8/MXFP4/NVFP4 packages with exact layouts store each
        // rank's own whole 128-row blocks; other packages (V4.1 native) pad every
        // rank to the widest slice.
        let package = report.experts.as_ref().map_or("", |e| e.package.as_str());
        let intermediate = model.spec().moe.as_ref().map_or(0, |m| m.intermediate);
        let exact = !package.starts_with("v41") && intermediate % 128 == 0 && intermediate / 128 >= ranks;
        let rank_bytes = |rank: usize| -> u64 {
            if !exact {
                return stored;
            }
            let blocks = intermediate / 128;
            let own = blocks / ranks + usize::from(rank < blocks % ranks);
            (routed as f64 * (own * 128) as f64 / intermediate as f64) as u64
        };
        for rank in 0..ranks {
            let stored = rank_bytes(rank);
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
        if !exact && report.spark_rank_share * ranks as f64 > 1.001 {
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
        // The measured MiMo default stores the head and every target o_proj
        // as FP8 (one copy), not the checkpoint's BF16.
        "mimo_v2" => {
            let Ok(cfg) = crate::families::mimo_v2::MimoV2Config::from_hf(&checkpoint.config) else { return Vec::new() };
            if default_policy(checkpoint, &cfg) != MimoDefaultPolicy::Fp8 {
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
        // GLM 5.3 Flash serves MLA, dense and shared-expert projections as FP8
        // from the official FP8 release (--fp8-snapshot); a BF16 checkpoint's
        // copies of them are not loaded. KDA in/out projections and the head
        // default to one per-row FP8 copy (--kda-fp8 row128, --fp8-head).
        "glm5_flash" => {
            let bf16 = |c: &crate::plan::checkpoint::CheckpointTensor| c.meta.dtype == cuteafd_core::DType::Bf16;
            let bytes = |filter: &dyn Fn(&str) -> bool| -> u64 {
                checkpoint.tensors.iter().filter(|t| bf16(t) && filter(&t.meta.name)).map(|t| t.meta.byte_length).sum()
            };
            let mla_layers: std::collections::BTreeSet<String> = checkpoint.tensors.iter()
                .filter(|t| t.meta.name.ends_with("self_attn.q_a_proj.weight"))
                .map(|t| t.meta.name.trim_end_matches("q_a_proj.weight").to_string()).collect();
            let mla = bytes(&|n: &str| mla_layers.iter().any(|p| n.starts_with(p.as_str()))
                && ["q_a_proj.weight", "kv_a_proj_with_mqa.weight", "q_b_proj.weight", "o_proj.weight"].iter().any(|s| n.ends_with(s)));
            let shared = bytes(&|n: &str| n.contains("shared_experts.") && n.ends_with("_proj.weight"));
            let dense = bytes(&|n: &str| n.contains(".mlp.") && !n.contains("experts") && n.ends_with("_proj.weight")
                && !n.contains(".gate."));
            let kda = bytes(&|n: &str| !mla_layers.iter().any(|p| n.starts_with(p.as_str())) && n.contains(".self_attn.")
                && ["q_proj.weight", "k_proj.weight", "v_proj.weight", "f_a_proj.weight", "g_a_proj.weight",
                    "b_proj.weight", "o_proj.weight"].iter().any(|s| n.ends_with(s)));
            let head = bytes(&|n: &str| n == "lm_head.weight");
            vec![Conversion { component: Component::Attention, saved_bytes: (mla + kda) / 2, format: "bf16+fp8" },
                Conversion { component: Component::LmHead, saved_bytes: head / 2, format: "fp8-row128" },
                Conversion { component: Component::SharedExpert, saved_bytes: shared / 2, format: "fp8" },
                Conversion { component: Component::DenseFfn, saved_bytes: dense / 2, format: "fp8" }]
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
    let qualified = default_policy(checkpoint, &cfg) == MimoDefaultPolicy::Fp8;
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
