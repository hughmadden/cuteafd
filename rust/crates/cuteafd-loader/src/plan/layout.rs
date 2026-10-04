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

mod deepseek_v4;
mod v41;

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
    /// Explicit pool tokens; `None` or `Some(0)` sizes the pool from what is left.
    pub pool_tokens: Option<u64>,
    /// Upper bound for an automatically sized pool.
    pub target_pool_tokens: u64,
    /// External drafter resident bytes on the lead GPU (DFlash), if any.
    pub drafter_bytes: u64,
    /// Keep this much of every GPU free for runtime growth.
    pub headroom_bytes: u64,
    /// Concurrent sequences (state slots = concurrency + 2).
    pub concurrency: u64,
    /// Exact engine slots when they differ from concurrency + 2.
    pub state_slots: Option<u64>,
    /// Prefix mark arena slots; absent selects the family policy.
    pub prefix_slots: Option<u64>,
    /// Compiled context extent (RoPE and index workspaces).
    pub context_tokens: u64,
    /// Resident native drafter stages (zero disables optional native MTP).
    pub native_mtp_layers: usize,
    /// Whole routed backbone layers resident on the coordinator.
    pub local_expert_layers: Option<usize>,
    /// Matching image's PROGRAMS.json for exact V4 workspace geometry.
    pub workspace_manifest: Option<std::path::PathBuf>,
}

impl Default for LayoutOptions {
    fn default() -> Self {
        Self {
            rtx_bytes: vec![95 * GIB + 512 * MIB],
            // 121.7 GiB GB10 minus the host OS and sparknestd measured idle (~13 GiB).
            spark_bytes: 108 * GIB,
            head_split: true,
            prefill_rows: 0,
            spark_capacity_rows: 4096,
            pool_tokens: None,
            target_pool_tokens: cuteafd_core::serving_capacity::DEFAULT_GPU_KV_TOKENS,
            drafter_bytes: 0,
            headroom_bytes: 2 * GIB,
            concurrency: 0,
            state_slots: None,
            prefix_slots: None,
            context_tokens: 0,
            native_mtp_layers: 3,
            local_expert_layers: None,
            workspace_manifest: None,
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
        "deepseek_v4" => FamilyCosts {
            runtime_bytes: [gib(65), gib(60), gib(60)],
            graph_bytes: [gib(35), gib(40), gib(25)],
            workspace_bytes: [gib(480), gib(414), gib(355)],
            exchange_bytes: gib(31),
            mark_slots: 18,
            spark_workspace_bytes: gib(35),
            spark_ring_bytes: gib(52),
            ..generic
        },
        "deepseek_v41" => FamilyCosts {
            runtime_bytes: [gib(133), gib(100), gib(100)],
            graph_bytes: [gib(150), gib(100), gib(90)],
            workspace_bytes: [gib(1101), gib(948), gib(570)],
            mark_slots: 42,
            spark_workspace_bytes: gib(21),
            spark_ring_bytes: gib(48),
            ..generic
        },
        "qwen4" => FamilyCosts {
            runtime_bytes: [gib(91), gib(91), 0],
            graph_bytes: [gib(50), gib(50), 0],
            workspace_bytes: [gib(120), gib(120), 0],
            mark_slots: 18,
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
    let discovered_native_layers = match model.spec().speculator.as_ref() {
        Some(super::spec::SpeculatorSpec::Dspark { stages, .. }) => *stages,
        Some(super::spec::SpeculatorSpec::NativeMtp { layers }) => *layers,
        None => 0,
    };
    let native_layers = if family == "qwen4" && report.placement != ExpertPlacement::Local {
        0 // Spark ranks do not serve Qwen's MTP experts.
    } else if family == "deepseek_v4" && options.native_mtp_layers > 0 {
        discovered_native_layers
    } else { options.native_mtp_layers.min(discovered_native_layers) };
    // V4 always loads all checkpoint stages and their caches; --dspark only
    // changes expert residency and whether the scheduler drafts with them.
    let cache_native_layers = if family == "deepseek_v4" { discovered_native_layers } else { native_layers };
    let workspace_manifest = options.workspace_manifest.as_deref().or_else(|| {
        let path = std::path::Path::new("/opt/cuteafd/share/PROGRAMS.json");
        path.is_file().then_some(path)
    }).and_then(|path| std::fs::read(path).ok()).and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok());
    let prefill_rows = if options.prefill_rows > 0 { options.prefill_rows }
        else if family == "deepseek_v41" { 2048 }
        else { workspace_manifest.as_ref().and_then(|m| m["capacities"]["prefill_rows"].as_u64()).unwrap_or(4096) };
    let decode_rows = workspace_manifest.as_ref().and_then(|m| m["capacities"]["decode_rows"].as_u64()).unwrap_or(64);
    let concurrency = if options.concurrency > 0 { options.concurrency } else if family == "deepseek_v41" { 16 } else { 8 };
    let context_tokens = if options.context_tokens > 0 { options.context_tokens }
        else if family == "deepseek_v4" { workspace_manifest.as_ref().and_then(|m| m["capacities"]["max_context"].as_u64()).unwrap_or(131072) }
        else { 0 };
    let conversions = load_conversions(family, checkpoint);
    let costs = family_costs(family);
    let gpus = options.rtx_bytes.len().clamp(1, 2);
    // Qwen currently executes entirely on the first coordinator GPU.
    let split = gpus == 2 && options.head_split && family != "qwen4";
    let active_gpus = if family == "qwen4" { 1 } else if split { 2 } else { 1 };
    let automatic = options.pool_tokens.unwrap_or(0) == 0;
    let mut devices: Vec<DeviceLayout> = options.rtx_bytes.iter().take(gpus).enumerate()
        .map(|(index, &bytes)| DeviceLayout { kind: DeviceKind::Rtx, index: index as u32,
            capacity_bytes: (if family == "deepseek_v41" && automatic { (bytes as u128 * 97 / 100) as u64 } else { bytes })
                .saturating_sub(if matches!(family, "deepseek_v4" | "deepseek_v41" | "qwen4") { options.headroom_bytes.max(3 * GIB) } else { options.headroom_bytes }), items: Vec::new(), kv_tokens: 0 })
        .collect();
    let mut waste = Vec::new();
    let mut notes = Vec::new();
    let qualified = matches!(family, "mimo_v2" | "glm5" | "glm5_flash");
    let allowance_basis = if qualified { Basis::Calibrated } else { Basis::Estimated };
    if !qualified {
        notes.push(format!("{family}: workspace/runtime/Spark allowances are unqualified; validate against the allocation ledger before using this layout for admission"));
    }
    if family == "qwen4" && gpus == 2 {
        notes.push("Qwen serves on rtx0; rtx1 is idle and contributes no KV capacity".into());
    }

    let mapped: u64 = report.components.iter().filter(|c| c.component == Component::MappedTable).map(|c| c.bytes).sum();
    if mapped > 0 { notes.push(format!("host-mapped table backing: {:.2} GiB (device budget includes staging only)", mapped as f64 / GIB as f64)); }

    // Coordinator weights: a family's exact resident layout where it has one,
    // else checkpoint bytes per component under the family's conversions.
    let v41_weights = if family == "deepseek_v41" { v41::resident_weights(checkpoint, active_gpus, native_layers > 0) } else { None };
    if let Some(ranks) = &v41_weights {
        for (device, items) in devices.iter_mut().zip(ranks) { device.items.extend(items.iter().cloned()); }
    }
    let exact = resident_layout(family, checkpoint, if split { 2 } else { 1 }, native_layers > 0);
    if let Some(ranks) = &exact {
        for (device, rank) in devices.iter_mut().zip(ranks) {
            for (group, format, bytes) in rank {
                let category = match group.as_str() {
                    "embedding" => Category::Embedding,
                    "speculator" | "speculator_expert" => Category::Drafter,
                    "table_projection" => Category::Tables,
                    _ => Category::Weights,
                };
                device.items.push(Item::new(category, group.clone(), format.clone(), *bytes, Basis::Exact));
            }
        }
    }
    let covered = |c: Component| (v41_weights.is_some() && !matches!(c, Component::RoutedExpert | Component::MappedTable))
        || (family.starts_with("deepseek_v4") && exact.is_some() && c == Component::Speculator)
        || (family == "qwen4" && exact.is_some() && matches!(c, Component::Speculator | Component::TableProjection))
        || (exact.is_some() && !matches!(c, Component::Speculator | Component::SpeculatorExpert
        | Component::Vision | Component::TableProjection | Component::RoutedExpert)
            && !(family.starts_with("deepseek_v4") && matches!(c, Component::Speculator | Component::SpeculatorExpert)))
        || (!costs.mtp_resident && matches!(c, Component::Speculator | Component::SpeculatorExpert));
    for component in report.components.iter()
        .filter(|c| (c.owner == Owner::Rtx
            || (c.owner == Owner::SparkSliced && report.placement == ExpertPlacement::Local))
            && c.status != Status::Unused && !covered(c.component)) {
        let (resident, format) = match conversions.iter().find(|c| c.component == component.component) {
            Some(c) => (component.bytes.saturating_sub(c.saved_bytes), c.format.to_string()),
            None => ((component.bytes as f64 * costs.resident_factor) as u64,
                component.formats.keys().cloned().collect::<Vec<_>>().join("+")),
        };
        let category = match component.component {
            Component::Embedding => Category::Embedding,
            Component::Speculator | Component::SpeculatorExpert => Category::Drafter,
            Component::TableProjection => Category::Tables,
            Component::RoutedExpert => Category::Experts,
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
    if family == "qwen4" && native_layers > 0 && report.placement == ExpertPlacement::Local {
        let native_experts: u64 = report.components.iter().filter(|c| c.component == Component::SpeculatorExpert && c.status == Status::Unused)
            .map(|c| c.bytes).sum();
        if native_experts > 0 {
            devices[0].items.push(Item::new(Category::Experts, "native MTP experts", "native", native_experts, Basis::Exact));
        }
    }
    let exl3_workspace = expert_workspace(report, model, checkpoint, options.workspace_manifest.as_deref(), prefill_rows);
    if exl3_workspace.is_none() && report.experts.as_ref().is_some_and(|e| e.package.contains("exl3"))
        && (family == "deepseek_v4" || (family == "qwen4" && report.placement == ExpertPlacement::Local)) {
        notes.push("Local EXL3 workspace allowance is estimated without matching rtx-tp1/m*/v41_exl3.json capacity manifests; images bundle them, or export the exl3 tree alongside PROGRAMS.json".into());
    }
    if family == "qwen4" && report.placement != ExpertPlacement::Local && options.native_mtp_layers > 0 {
        notes.push("Qwen Spark layouts omit native MTP: Spark ranks do not serve its draft expert layer".into());
    }
    if family == "qwen4" && report.placement == ExpertPlacement::Local
        && report.experts.as_ref().is_some_and(|e| e.package.contains("exl3")) {
        // The EXL3 window retains its shared capacity arenas in addition to
        // checkpoint trellis bytes (1.15 GiB in the reference allocation ledger).
        devices[0].items.push(Item::new(Category::Experts, "local EXL3 workspace", "",
            exl3_workspace.unwrap_or(gib(115) * prefill_rows / 4096),
            if exl3_workspace.is_some() { Basis::Formula } else { allowance_basis }));
    }
    // The drafter lives on the lead GPU (taps and head are there under a head split).
    let drafter = if options.drafter_bytes > 0 { options.drafter_bytes } else { costs.drafter_bytes };
    if drafter > 0 {
        devices[0].items.push(Item::new(Category::Drafter, "drafter", "", drafter, allowance_basis));
    }

    let v4_workspace = if family == "deepseek_v4" {
        (|| {
            let manifest = workspace_manifest.as_ref()?;
            let cfg = crate::families::deepseek_v4::DeepseekV4Config::read(&checkpoint.snapshot, cache_native_layers).ok()?;
            let id = if cfg.dim == 4096 { "dsv4f" } else { "dsv4p" };
            let scratch = crate::serving_capacity::deepseek_v4_workspace_scratch(manifest, id, prefill_rows, decode_rows).ok()?;
            crate::serving_capacity::deepseek_v4_workspace_geometry(&cfg, prefill_rows, decode_rows,
                context_tokens, active_gpus, scratch).ok()
        })()
    } else { None };
    if family == "deepseek_v4" && v4_workspace.is_none() {
        notes.push("V4 workspace allowance is estimated without a matching PROGRAMS.json; use --workspace-manifest for Flash/Pro allocation geometry".into());
    }

    // Fixed runtime costs.
    let gpus_now = active_gpus;
    for (index, device) in devices.iter_mut().take(active_gpus).enumerate() {
        let role = if gpus_now == 1 { 0 } else if index == 0 { 1 } else { 2 };
        device.items.push(Item::new(Category::Runtime, "context+modules", "", costs.runtime_bytes[role],
            allowance_basis));
        device.items.push(Item::new(Category::Runtime, "graph allowance", "", costs.graph_bytes[role], allowance_basis));
        let workspace = v4_workspace.as_ref().and_then(|ranks| ranks.get(index)).map_or_else(
            || costs.workspace_bytes[role] * prefill_rows.max(1) / if family == "deepseek_v41" { 2048 } else { 4096 },
            |rank| rank.fixed_device_bytes);
        // V4 keeps one 4096-row intake plane per Spark and prefill lane.
        // Decode reuses lane zero; every plane belongs to the lead GPU.
        let intake = if family == "deepseek_v4" && index == 0 {
            match report.placement {
                ExpertPlacement::Sparks { ranks } => 2 * ranks as u64 * 4096 * model.spec().hidden as u64 * 2,
                _ => 0,
            }
        } else { 0 };
        let workspace = workspace + intake;
        if family == "deepseek_v4" {
            // --reserve-gib 10 covers the future workspace and graph budget;
            // admission keeps the unused remainder, with a 3 GiB floor.
            let headroom = (10 * GIB).saturating_sub(workspace + costs.graph_bytes[role])
                .max(options.headroom_bytes).max(3 * GIB);
            device.capacity_bytes = options.rtx_bytes[index].saturating_sub(headroom);
        }
        let workspace_basis = if v4_workspace.is_some() { Basis::Formula } else { allowance_basis };
        device.items.push(Item::new(Category::Workspace, "steps", "", workspace, workspace_basis));
        if split {
            let exact_peer = if family == "deepseek_v4" {
                crate::serving_capacity::deepseek_v4_peer_exchange_bytes(model.spec().hidden as u64, prefill_rows, decode_rows).ok()
            } else { None };
            device.items.push(Item::new(Category::Transport, "peer exchange", "", exact_peer.unwrap_or(costs.exchange_bytes),
                if exact_peer.is_some() { Basis::Formula } else { allowance_basis }));
        }
    }

    // Resident V4.1 layers are taken out of Spark slices; V4 retains the complete
    // target on Sparks even when RTX also holds a layer. Experts are charged
    // before the KV pool. Explicit placement is reproducible on any inventory.
    let routed_bytes: u64 = report.components.iter().filter(|c| c.component == Component::RoutedExpert)
        .map(|c| c.bytes).sum();
    let layer_bytes = routed_bytes / model.spec().layers.len().max(1) as u64;
    let mut local_layers = if family.starts_with("deepseek_v4") {
        options.local_expert_layers.unwrap_or(0).min(model.spec().layers.len())
    } else { 0 };
    let draft_experts: u64 = if family == "deepseek_v4" && native_layers > 0 {
        report.components.iter().filter(|c| c.component == Component::SpeculatorExpert).map(|c| c.bytes).sum()
    } else { 0 };
    if draft_experts > 0 {
        devices[0].items.push(Item::new(Category::Experts, "dSpark stage experts", "native", draft_experts, Basis::Exact));
    }
    if family == "deepseek_v4" && options.local_expert_layers.is_none() && layer_bytes > 0 {
        if let Ok(Some(cache)) = model.cache_geometry(CacheOptions { coordinator_ranks: active_gpus,
            native_mtp_layers: cache_native_layers, prefill_rows: prefill_rows, ..Default::default() }) {
            let mark_bytes: u64 = cache.ranks.iter().map(|r| r.retained_mark_bytes).sum();
            let slots = options.prefix_slots.unwrap_or(42.min(2 * GIB / mark_bytes.max(1)).max(2 * concurrency + 2));
            let state = cache.ranks[0].active_state_per_sequence_bytes * options.state_slots.unwrap_or(concurrency)
                + cache.ranks[0].fixed_state_bytes + cache.ranks[0].context_table_bytes_per_token * context_tokens;
            let role = if active_gpus == 1 { 0 } else { 1 };
            let workspaces: u64 = devices[0].items.iter().filter(|i| i.group == "steps").map(|i| i.bytes).sum();
            let already = devices[0].used_bytes().saturating_sub(workspaces + costs.graph_bytes[role]);
            let legacy = cache.ranks[0].persistent_unit_bytes * 262144u64.div_ceil(cache.logical_unit_rows);
            let expert_workspace = exl3_workspace.unwrap_or(160 * MIB * prefill_rows / 4096);
            let reserve = state + legacy + slots * mark_bytes + 10 * GIB + expert_workspace;
            local_layers = (options.rtx_bytes[0].saturating_sub(already + reserve) / layer_bytes)
                .min(model.spec().layers.len() as u64) as usize;
        }
    }
    let mut local_bytes = layer_bytes * local_layers as u64;
    if local_bytes > 0 && !matches!(report.placement, ExpertPlacement::Local) {
        if family == "deepseek_v41" && gpus == 2 {
            for device in devices.iter_mut().take(2) {
                device.items.push(Item::new(Category::Experts, "resident routed layers", "native-tp2", local_bytes / 2, Basis::Formula));
            }
        } else {
            devices[0].items.push(Item::new(Category::Experts, "resident routed layers", "native", local_bytes, Basis::Exact));
        }
    }

    if family == "deepseek_v4" && local_bytes + draft_experts > 0 {
        devices[0].items.push(Item::new(Category::Experts, "local expert workspace", "",
            exl3_workspace.unwrap_or(160 * MIB * prefill_rows / 4096),
            if exl3_workspace.is_some() { Basis::Formula } else { Basis::Estimated }));
    }

    // KV pool: per-device bytes per logical token from the family geometry.
    let geometry = model.cache_geometry(CacheOptions { coordinator_ranks: active_gpus,
        native_mtp_layers: if family == "deepseek_v4" || family == "qwen4" { cache_native_layers } else { 0 },
        prefill_rows: prefill_rows, ..Default::default() });
    let mut pool_tokens = 0;
    match geometry {
        Ok(Some(mut geometry)) => {
            for rank in &mut geometry.ranks {
                rank.pool_metadata_unit_bytes += match family {
                    "deepseek_v4" => (2 * prefill_rows + decode_rows) * 4,
                    "qwen4" => (1 + 64) * 5 * 4,
                    _ => 0,
                };
            }
            let marks = options.prefix_slots.unwrap_or_else(|| {
                if matches!(family, "deepseek_v4" | "qwen4") {
                    let bytes: u64 = geometry.ranks.iter().map(|r| r.retained_mark_bytes).sum();
                    42.min((2 * GIB) / bytes.max(1)).max(2 * concurrency + 2)
                } else { costs.mark_slots }
            });
            let unit = geometry.logical_unit_rows.max(1);
            let per_token: Vec<u64> = (0..devices.len()).map(|d| geometry.ranks.get(d)
                .map_or(0, |r| (r.persistent_unit_bytes + r.pool_metadata_unit_bytes).div_ceil(unit))).collect();
            // Reserve fixed state and marks before sizing records. Otherwise
            // an automatic pool consumes the bytes those allocations need.
            for (device, rank) in devices.iter_mut().zip(&geometry.ranks) {
                let state_slots = options.state_slots.unwrap_or(if matches!(family, "qwen4" | "deepseek_v4" | "deepseek_v41") { concurrency } else { concurrency + 2 });
                device.items.push(Item::new(Category::Kv, "state", "", rank.fixed_state_bytes
                    + (rank.active_state_per_sequence_bytes + if family == "deepseek_v4" { rank.pool_metadata_unit_bytes } else { 0 }) * state_slots
                    + rank.speculative_replay_bytes + rank.context_table_bytes_per_token * context_tokens, Basis::Formula));
                if family != "deepseek_v41" && marks > 0 && rank.retained_mark_bytes > 0 {
                    device.items.push(Item::new(Category::Prefix, "marks", "", rank.retained_mark_bytes * marks,
                        Basis::Formula));
                }
            }
            if family == "deepseek_v41" {
                let prefixes = v41::prefix_arena_bytes(marks, native_layers > 0, active_gpus);
                let retained_turns = marks.saturating_sub(2) / 2;
                for (device, &bytes) in devices.iter_mut().zip(&prefixes) {
                    device.items.push(Item::new(Category::Prefix, "snapshot arenas", "", bytes, Basis::Formula));
                    device.items.push(Item::new(Category::Kv, "active and retained COW tails", "", geometry.ranks[device.index as usize].persistent_unit_bytes * (concurrency + 2 * retained_turns), Basis::Formula));
                }
                if native_layers > 0 {
                    devices[active_gpus - 1].items.push(Item::new(Category::Drafter, "dSpark window state", "", v41::dspark_cache_bytes(concurrency, prefill_rows), Basis::Formula));
                }
            }
            let free: Vec<i64> = devices.iter().map(DeviceLayout::free_bytes).collect();
            pool_tokens = options.pool_tokens.filter(|&tokens| tokens != 0)
                .unwrap_or_else(|| size_pool(&free, &per_token, unit, if family == "deepseek_v41" { v41::DEFAULT_POOL_TOKENS } else { options.target_pool_tokens }));
            for (device, &cost) in devices.iter_mut().zip(&per_token) {
                if cost > 0 {
                    let units = pool_tokens.div_ceil(unit);
                    let rank = &geometry.ranks[device.index as usize];
                    device.items.push(Item::new(Category::Kv, "records", "",
                        (rank.persistent_unit_bytes + rank.pool_metadata_unit_bytes) * units, Basis::Formula));
                    device.kv_tokens = pool_tokens;
                }
                if device.free_bytes() < 0 {
                    notes.push(format!("{}: requested KV pool exceeds the device budget", device.name()));
                }
            }
            if geometry.placement == KvPlacement::Replicated && devices.len() == 2 {
                waste.push(Waste { device: "rtx1".into(), what: "KV records replicated on both GPUs (MLA latent)".into(),
                    bytes: per_token.get(1).copied().unwrap_or(0) * pool_tokens });
            }
        }
        Ok(None) => notes.push(format!("{family}: no cache geometry in the planner yet (engine sizes its own pool)")),
        Err(error) => notes.push(format!("{family}: cache geometry: {error}")),
    }

    if family == "deepseek_v41" && options.local_expert_layers.is_none() && layer_bytes > 0 {
        let per_gpu = if active_gpus == 2 { layer_bytes / 2 } else { layer_bytes };
        let room = devices.iter().take(active_gpus).map(|d| d.free_bytes().max(0) as u64).min().unwrap_or(0);
        local_layers = (room.saturating_sub(512 * MIB) / per_gpu).min(model.spec().layers.len() as u64) as usize;
        local_bytes = layer_bytes * local_layers as u64;
        for device in devices.iter_mut().take(active_gpus) {
            device.items.push(Item::new(Category::Experts, "resident routed layers", "native", per_gpu * local_layers as u64 + 256 * MIB, Basis::Formula));
        }
    }
    if local_layers > 0 { notes.push(format!("{local_layers} routed backbone layers resident on RTX")); }

    // Spark ranks.
    if let ExpertPlacement::Sparks { ranks } = report.placement {
        let routed: u64 = report.components.iter().filter(|c| c.owner == Owner::SparkSliced && c.component != Component::SpeculatorExpert).map(|c| c.bytes).sum::<u64>().saturating_sub(if family == "deepseek_v41" { local_bytes } else { 0 });
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
            if matches!(family, "deepseek_v4" | "deepseek_v41" | "qwen4") {
                // A GB10 CUDA sample includes host OS and sparknestd. The
                // supplied Spark inventory is usable memory after this reserve.
                device.capacity_bytes += 13 * GIB;
                device.items.push(Item::new(Category::Reserved, "host OS+sparknestd", "", 13 * GIB, Basis::Calibrated));
            }
            let format = report.components.iter().find(|c| c.owner == Owner::SparkSliced)
                .map(|c| c.formats.keys().cloned().collect::<Vec<_>>().join("+")).unwrap_or_default();
            device.items.push(Item::new(Category::Experts, "routed_expert", format, stored, Basis::Exact));
            let workspace = costs.spark_workspace_bytes * options.spark_capacity_rows / 4096;
            device.items.push(Item::new(Category::Workspace, "expert waves", "", workspace, allowance_basis));
            device.items.push(Item::new(Category::Transport, "rdma rings", "", costs.spark_ring_bytes, allowance_basis));
            device.items.push(Item::new(Category::Runtime, "context+modules", "", 512 * MIB, allowance_basis));
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

/// Local EXL3 arenas from the same capacity manifests used by the loader.
/// An exported PROGRAMS.json can have its expert JSON tree alongside it;
/// in an image the standard tree lives in ../lib/exl3 instead.
fn expert_workspace(report: &PlanReport, model: &dyn super::FamilyModel, checkpoint: &super::Checkpoint,
    manifest: Option<&std::path::Path>, rows: u64) -> Option<u64> {
    if !report.experts.as_ref()?.package.contains("exl3") { return None; }
    // Display labels name the checkpoint tier; the decoder can include an
    // adjacent tier too (Pro K2 uses its k23 package). Read the catalog contract.
    let catalog = crate::read_expert_catalog(&checkpoint.snapshot).ok()?;
    let tiers = catalog.exl3()?.decoder_tiers().iter().map(usize::to_string).collect::<String>();
    let family = match report.family.as_deref()? {
        "qwen4" => "qwen4",
        "deepseek_v4" if model.spec().hidden == 4096 => "dsv4f",
        "deepseek_v4" => "dsv4p",
        _ => return None,
    };
    let parent = manifest.unwrap_or(std::path::Path::new("/opt/cuteafd/share/PROGRAMS.json")).parent()?;
    let stem = format!("exl3-{family}-k{tiers}");
    let root = [parent.join("exl3").join(&stem), parent.join("../lib/exl3").join(&stem)]
        .into_iter().find(|p| p.join("rtx-tp1/m4096/v41_exl3.json").is_file())?;
    let maximum = if family.starts_with("dsv4") { rows.max(64) } else { rows.max(1) };
    const CAPACITIES: [u64; 6] = [1, 16, 80, 256, 1024, 4096];
    if maximum > 4096 { return None; }
    let manifests = CAPACITIES.into_iter().filter(|&n| n <= maximum)
        .chain(CAPACITIES.into_iter().find(|&n| n >= maximum))
        .map(|capacity| serde_json::from_slice::<serde_json::Value>(
            &std::fs::read(root.join(format!("rtx-tp1/m{capacity}/v41_exl3.json"))).ok()?).ok())
        .collect::<Option<Vec<_>>>()?;
    let moe = model.spec().moe.as_ref()?;
    if manifests.iter().any(|m| m["hidden"].as_u64() != Some(model.spec().hidden as u64)
        || m["intermediate"].as_u64() != Some(moe.intermediate as u64)
        || m["experts"].as_u64() != Some(moe.experts as u64)) { return None; }
    crate::serving_capacity::exl3_workspace_bytes(&manifests, true).ok()?
        .checked_add(maximum * model.spec().hidden as u64 * 2)
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
        // Qwen 3.8: target and MTP share one per-row FP8 head (--mtp-fp8-head).
        "qwen4" => {
            let head: u64 = checkpoint.tensors.iter()
                .filter(|t| t.meta.name.ends_with("lm_head.weight") && t.meta.dtype == cuteafd_core::DType::Bf16)
                .map(|t| t.meta.byte_length).sum();
            if head == 0 { Vec::new() }
            else { vec![Conversion { component: Component::LmHead, saved_bytes: head / 2, format: "fp8-row128" }] }
        }
        _ => Vec::new(),
    }
}

/// Exact per-rank resident weights `(group, format, bytes)` for families
/// whose loader publishes its resident layout (MiMo: the codex capacity
/// contract's `MimoResidentLayout`, with the default weight policy).
fn resident_layout(family: &str, checkpoint: &super::Checkpoint, ranks: usize, mtp: bool) -> Option<Vec<Vec<(String, String, u64)>>> {
    use crate::families::mimo_v2::projection::MimoProjectionRepresentation as R;
    use crate::families::mimo_v2::resident::{MimoResidentLayout, MimoResidentOptions};
    use crate::families::mimo_v2::weight_policy::{default_policy, MimoDefaultPolicy};
    if family == "qwen4" {
        use crate::families::qwen4::{Qwen4Config, resident::{checkpoint_resident_bytes, Qwen4Representation}};
        let cfg = Qwen4Config::from_hf(&checkpoint.config).ok()?;
        let mtp = mtp && cfg.mtp_layers > 0;
        let bytes = checkpoint_resident_bytes(checkpoint, &cfg, cfg.layers, mtp,
            Qwen4Representation { fp8_projections: false, fp8_head: true }).ok()?;
        return Some(vec![vec![
            ("target incl PLE projections".into(), "bf16+f32".into(), bytes.target_bytes),
            ("embedding".into(), "bf16".into(), bytes.embedding_bytes),
            ("lm_head".into(), "fp8-row128".into(), bytes.head_bytes),
            ("speculator".into(), "bf16+f32".into(), bytes.mtp_bytes),
        ]]);
    }
    if family == "deepseek_v4" {
        return deepseek_v4::resident_weights(checkpoint, ranks);
    }
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
