//! Header-only lower bound for MiniMax M3 on one or two RTX coordinators and
//! 2, 3, 4 or 6 Spark ranks: exact header weights, the FP8 KV and MSA index
//! records per token, and the fewest bottom routed layers onboarded to the RTX
//! that keep every Spark rank under its budget. Workspaces, graphs and
//! transport are estimates until the engine's allocation ledger exists; this
//! is NOT serving admission.
use super::Model;
use crate::families::{minimax_m3::{cache::MiniMaxCacheLayout, config::MiniMaxM3Config}, qwen4::Qwen4KvCache};
use crate::placement::{ArchContext, AttentionPlacement};
use crate::plan::{Component, ExpertPlacement, Hint, MediaMode, PlanReport};
use crate::serving_capacity::{CacheGeometryError, CacheOptions, FamilyCacheGeometry};
use cuteafd_core::memory_layout::{Basis, Category, DeviceKind, DeviceLayout, Item, MemoryLayout};

const GIB: u64 = 1 << 30;
/// Logical KV allocation unit: one MSA block (two 64-row b12x pages).
pub const UNIT_ROWS: u64 = 128;
/// Per-rank routed-expert allocation ceiling without `--spark-budget-gib`
/// (the Kimi K3 precedent: a GB10 rank keeps ~113-116 GiB available idle).
/// Estimated until the first full worker load/steady census.
pub const SPARK_BUDGET_BYTES: u64 = 108 << 30;
/// Spark runtime residual plus an estimated NVFP4 package workspace.
const SPARK_RESERVE_BYTES: u64 = cuteafd_core::memory_layout::SPARK_RUNTIME_RESIDUAL_BYTES + (512 << 20);
/// Coordinator step workspaces, sampler, intake and graphs before the engine
/// measures them: an unqualified allowance per GPU.
const COORDINATOR_WORKSPACE_BYTES: u64 = 3 * GIB;
/// The smallest pool this plan calls a card: one full 256K agentic request.
const POOL_FLOOR_TOKENS: u64 = 262_144;

/// KV record bytes per token on each coordinator rank (the exact paged layout:
/// that rank's KV heads, plus the whole shared MSA index key).
fn per_token(cfg: &MiniMaxM3Config, ranks: usize, kv: Qwen4KvCache) -> u64 {
    MiniMaxCacheLayout::new(cfg, ranks, cfg.max_position as u64, kv).map_or(0, |layout| layout.token_bytes(0))
}

pub(super) fn cache_geometry(cfg: &MiniMaxM3Config, options: CacheOptions) -> Result<FamilyCacheGeometry, CacheGeometryError> {
    if options.attention_placement.is_some_and(|p| p != AttentionPlacement::Heads) {
        return Err(CacheGeometryError::Unsupported { family: "minimax_m3", what: "one GPU, or two splitting KV heads" });
    }
    MiniMaxCacheLayout::new(cfg, options.coordinator_ranks, cfg.max_position as u64, options.minimax_kv).map(|l| l.geometry())
}

pub(super) fn layout(model: &Model, report: &mut PlanReport, opts: &crate::plan::layout::LayoutOptions,
    requested: ExpertPlacement) -> MemoryLayout {
    let cfg = model.cfg();
    let ranks = requested.spark_ranks();
    let gpus = opts.rtx_bytes.len();
    let mut memory = MemoryLayout { devices: Vec::new(), pool_tokens: 0, waste: Vec::new(), notes: vec![
        "MiniMax M3 header lower bound: workspaces/graphs/transport are estimates; NOT serving admission".into(),
    ] };
    let refusal = |report: &mut PlanReport, what: String, how: &str| {
        report.fits = false;
        report.placement_supported = false;
        report.hints.push(Hint { what, how: how.into() });
    };
    report.bytes_by_owner.clear();
    report.attention_placement = AttentionPlacement::Heads;
    if !model.complete {
        refusal(report, "MiniMax M3 placement cannot infer fit from missing safetensors headers".into(),
            "Place every shard's header locally (nest replicate) or plan from a complete snapshot.");
        return memory;
    }
    if !(1..=2).contains(&gpus) || (gpus == 2 && !opts.head_split) {
        refusal(report, "MiniMax M3 runs one coordinator GPU, or two splitting its KV heads".into(), "Use --rtx 1 or --rtx 2.");
        return memory;
    }
    if ranks == 0 {
        refusal(report, "MiniMax M3 routed experts (216 GiB) need Spark ranks beside the coordinator".into(),
            "Use --spark-ranks 2, 3, 4 or 6.");
        return memory;
    }
    let edge = |c| *model.bytes.get(&c).unwrap_or(&0);
    let embedding = edge(Component::Embedding);
    let head = edge(Component::LmHead);
    let body: u64 = [Component::Attention, Component::Indexer, Component::Norm, Component::Router,
        Component::SharedExpert, Component::DenseFfn].into_iter().map(edge).sum();
    // Routers, norms and the one-head index key projection (128 of the
    // indexer's 640 rows) stay whole on each GPU of a head split.
    let replicated = edge(Component::Router) + edge(Component::Norm) + edge(Component::Indexer) / 5;
    let drafter = if opts.native_mtp_layers > 0 { edge(Component::Speculator) } else { 0 };
    let vision = edge(Component::Vision);
    let vision_rtx = matches!(report.vision, MediaMode::Rtx(_)) && vision > 0;
    let vision_spark = matches!(report.vision, MediaMode::Spark(_) | MediaMode::Auto) && vision > 0;
    let experts = edge(Component::RoutedExpert);
    report.bytes_by_owner.insert("coordinator text weights".into(), body + head + if opts.host_embedding { 0 } else { embedding });
    report.bytes_by_owner.insert("routed expert weights".into(), experts);
    if opts.host_embedding { report.bytes_by_owner.insert("host-mapped embedding".into(), embedding); }

    // Spark ranks own exact 1/world intermediate slices of every hosted layer
    // (3072 splits into whole 16-value NVFP4 blocks at 2, 3, 4 and 6 ranks).
    let layers: Vec<(usize, u64)> = model.expert_layer_bytes.iter().map(|(l, b)| (*l, *b)).collect();
    let world = ranks as u64;
    let budget = opts.spark_allocation_budget_bytes.unwrap_or(SPARK_BUDGET_BYTES);
    let rank_bytes = |onboard: usize| layers[onboard..].iter().map(|(_, b)| b.div_ceil(world)).sum::<u64>();
    let onboard = match opts.onboard {
        Some(crate::placement::Onboard::Layers(n)) => n.min(layers.len()),
        _ => (0..=layers.len()).find(|&n| rank_bytes(n) + SPARK_RESERVE_BYTES
            + if vision_spark { vision } else { 0 } <= budget).unwrap_or(layers.len()),
    };
    let local: u64 = layers[..onboard].iter().map(|(_, b)| b).sum();
    report.min_spark_ranks = crate::plan::experts::spark_worlds(super::PACKAGE, cfg.moe_intermediate).into_iter()
        .find(|&w| layers.iter().map(|(_, b)| b.div_ceil(w as u64)).sum::<u64>() + SPARK_RESERVE_BYTES <= budget);

    let gpu_items = |gpu: usize, local: u64| -> Vec<Item> {
        let share = |bytes: u64| if gpus == 1 { bytes } else { bytes / 2 };
        let capacity = opts.rtx_bytes[gpu];
        let mut items = vec![
            Item::new(Category::Runtime, "CUDA context", "sm_120", ArchContext::coordinator(capacity, None).context_bytes, Basis::Calibrated),
            Item::new(Category::Weights, if gpus == 1 { "attention, indexer, shared/dense MLP (MXFP8), routers, norms" }
                else { "attention/indexer heads, shared/dense MLP halves (MXFP8)" }, "mxfp8",
                share(body - replicated) + replicated, Basis::Exact),
            Item::new(Category::Workspace, "step workspaces, sampler, graphs (unmeasured)", "", COORDINATOR_WORKSPACE_BYTES, Basis::Estimated),
            Item::new(Category::Reserved, "coordinator headroom", "", opts.headroom_bytes, Basis::Formula),
        ];
        if gpu == 0 && !opts.host_embedding { items.push(Item::new(Category::Embedding, "token embedding", "bf16", embedding, Basis::Exact)); }
        if gpu + 1 == gpus { items.push(Item::new(Category::Weights, "lm head", "bf16", head, Basis::Exact)); }
        if gpu == 0 && drafter > 0 { items.push(Item::new(Category::Drafter, "bundled DSpark drafter (5 layers)", "mxfp8", drafter, Basis::Exact)); }
        if gpu == 0 && vision_rtx { items.push(Item::new(Category::Weights, "vision tower + projector", "bf16", vision, Basis::Exact)); }
        if local > 0 { items.push(Item::new(Category::Experts, "onboarded bottom routed layers (TP1)", "nvfp4 W4A4", share(local), Basis::Exact)); }
        items
    };
    // An explicit onboard that overflows the GPU is refused below; an automatic
    // one is what the Sparks need, so the GPU check reports the shortfall.
    let token = per_token(cfg, gpus, opts.minimax_kv);
    let free = |local: u64| (0..gpus).map(|gpu| {
        let used: u64 = gpu_items(gpu, local).iter().map(|item| item.bytes).sum();
        opts.rtx_bytes[gpu].saturating_sub(used)
    }).min().unwrap_or(0);
    // An automatic pool takes every free byte (the engine pages it per sequence);
    // an explicit one is clamped to what fits.
    let fitting = free(local) / token / UNIT_ROWS * UNIT_ROWS;
    let pool_tokens = opts.pool_tokens.filter(|n| *n > 0).map_or(fitting, |n| n.min(fitting));
    memory.pool_tokens = pool_tokens;
    for gpu in 0..gpus {
        let mut items = gpu_items(gpu, local);
        items.push(Item::new(Category::Kv, format!("{:?} KV ({} KV heads) + MSA index keys, {token} B/token",
            opts.minimax_kv, cfg.kv_heads / gpus), "paged", pool_tokens * token, Basis::Formula));
        memory.devices.push(DeviceLayout { kind: DeviceKind::Rtx, index: gpu as u32, capacity_bytes: opts.rtx_bytes[gpu],
            items, kv_tokens: pool_tokens });
    }
    let per_rank = rank_bytes(onboard);
    for rank in 0..ranks {
        let mut items = vec![
            Item::new(Category::Experts, format!("routed experts TP{ranks} (layers {}..{})",
                layers.get(onboard).map_or(0, |l| l.0), layers.last().map_or(0, |l| l.0 + 1)), "nvfp4 W4A4", per_rank, Basis::Exact),
            cuteafd_core::memory_layout::spark_runtime_residual(),
            Item::new(Category::Workspace, "NVFP4 package workspace (unmeasured)", "", 512 << 20, Basis::Estimated),
        ];
        if rank == 0 && vision_spark { items.push(Item::new(Category::Weights, "vision tower + projector", "bf16", vision, Basis::Exact)); }
        memory.devices.push(DeviceLayout { kind: DeviceKind::Spark, index: rank as u32, capacity_bytes: budget, items, kv_tokens: 0 });
    }
    let gib = |b: u64| b as f64 / GIB as f64;
    memory.notes.push(format!("{} routed layers on the RTX ({:.1} GiB), {} on {ranks} Sparks at {:.1} GiB/rank (budget {:.0} GiB)",
        onboard, gib(local), layers.len() - onboard, gib(per_rank), gib(budget)));
    memory.notes.push(format!("pool {pool_tokens} tokens ({:.2} full 1M-token contexts) at {token} B/token per GPU; vision {}; drafter {}",
        pool_tokens as f64 / cfg.max_position as f64,
        if vision == 0 { "absent" } else if vision_rtx { "on rtx0" } else if vision_spark { "on spark0" } else { "off" },
        if drafter > 0 { "on rtx0" } else { "off" }));
    if per_rank + SPARK_RESERVE_BYTES + if vision_spark { vision } else { 0 } > budget {
        refusal(report, format!("MiniMax M3 TP{ranks} Spark ranks need {:.1} GiB with {onboard} onboarded layers, over the {:.0} GiB budget",
            gib(per_rank + SPARK_RESERVE_BYTES), gib(budget)),
            "Onboard more bottom routed layers to the RTX (--rtx-expert-layers), add Spark ranks, or raise --spark-budget-gib after a load census.");
    }
    if pool_tokens < POOL_FLOOR_TOKENS {
        report.fits = false;
        report.hints.push(Hint {
            what: format!("MiniMax M3 coordinator pool {pool_tokens} tokens is below one 256K request ({POOL_FLOOR_TOKENS})"),
            how: "Move the token embedding to host (--embedding-placement host), run without the drafter or RTX vision, \
                or add Spark ranks so fewer routed layers onboard.".into(),
        });
    }
    memory
}
