//! Native serving allocation ledger when program metadata is available;
//! otherwise an explicitly unqualified checkpoint-header lower bound.
use super::{admission, Model};
use crate::families::{minimax_m3::{cache::MiniMaxCacheLayout, config::MiniMaxM3Config,
    dspark::{DraftMemory, DsparkContract}}, qwen4::Qwen4KvCache};
use crate::placement::{ArchContext, AttentionPlacement};
use crate::plan::{Component, ExpertPlacement, Hint, MediaMode, PlanReport};
use crate::plan::encoder::{encoder_placement, EncoderGpuBudget, EncoderHardware, EncoderKind, EncoderSparkBudget};
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

pub(super) fn layout(model: &Model, input: &crate::plan::PlanInput, report: &mut PlanReport,
    opts: &crate::plan::layout::LayoutOptions, requested: ExpertPlacement) -> anyhow::Result<MemoryLayout> {
    let path = opts.workspace_manifest.as_deref().unwrap_or_else(|| std::path::Path::new("/opt/cuteafd/share/PROGRAMS.json"));
    if opts.workspace_manifest.is_none() && !path.is_file() {
        anyhow::ensure!(!opts.minimax_cache.proposed() && !opts.minimax_5090,
            "MiniMax cache proposals and 5090 profile require --workspace-manifest");
        return Ok(header_layout(model, report, opts, requested));
    }
    let mut base = opts.clone();
    if opts.minimax_5090 {
        anyhow::ensure!(opts.native_mtp_layers > 0, "MiniMax 5090 profile requires the bundled DSpark");
        if base.prefill_rows == 0 { base.prefill_rows = 256; }
        if base.concurrency == 0 { base.concurrency = 1; }
    }
    if opts.minimax_cache.proposed() { base.pool_tokens = None; }
    let mut memory = exact_layout(model, input, report, &base, requested, path)?;
    if opts.minimax_5090 { memory.notes.push("Opt-in 5090 C1 sizing: prefill rows and sequence slots only; DSpark stays on; C1 performance unmeasured".into()); }
    if opts.minimax_cache.proposed() { super::capacity::apply(model.cfg(), &mut memory, report, opts)?; }
    Ok(memory)
}

fn header_layout(model: &Model, report: &mut PlanReport, opts: &crate::plan::layout::LayoutOptions,
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
    let vision_scratch = if vision > 0 { crate::plan::layout::minimax_vision_scratch_bytes() } else { 0 };
    // Every official tensor/vector extent is 256-byte aligned; BF16 vectors
    // expand to FP32, followed by thirteen FP32 rotary frequencies.
    let vector_bytes: u64 = model.contracts.values().filter(|c| c.role.component == Component::Vision)
        .map(|c| if c.shape.len() == 1 { c.shape[0] as u64 * 2 } else { 0 }
            + if c.bias { c.shape[0] as u64 * 2 } else { 0 }).sum();
    let vision_weights = if vision > 0 { vision + vector_bytes + 52 } else { 0 };
    let vision_spark = matches!(report.vision, MediaMode::Spark(_)) && vision > 0;
    let vision_reserve = vision_weights + vision_scratch + GIB;
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
            + if vision_spark { vision_reserve } else { 0 } <= budget).unwrap_or(layers.len()),
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
        if local > 0 { items.push(Item::new(Category::Experts, "onboarded bottom routed layers (TP1)", "nvfp4 W4A4", share(local), Basis::Exact)); }
        items
    };
    let token = per_token(cfg, gpus, opts.minimax_kv);
    let hardware = EncoderHardware {
        v41: false,
        gpus: (0..gpus).map(|gpu| EncoderGpuBudget {
            free_bytes: opts.rtx_bytes[gpu].saturating_sub(gpu_items(gpu, local).iter().map(|i| i.bytes).sum()),
            kv_target_bytes: token * opts.pool_tokens.unwrap_or(POOL_FLOOR_TOKENS),
        }).collect(),
        sparks: (0..ranks).map(|rank| EncoderSparkBudget {
            rank, host: String::new(), idle: false, expert_bytes: rank_bytes(onboard),
            free_bytes: budget.saturating_sub(rank_bytes(onboard) + SPARK_RESERVE_BYTES),
        }).collect(),
    };
    let encoder = encoder_placement(report.vision, &hardware, vision_weights, vision_scratch, opts.vision_replicas);
    if encoder.kind == EncoderKind::Off && matches!(report.vision, MediaMode::Rtx(_) | MediaMode::Spark(_)) {
        refusal(report, format!("MiniMax vision placement unavailable: {} (shortfall {} bytes)", encoder.reason, encoder.shortfall),
            "Select an encoder device with tower/scratch/KV headroom, or use --vision off.");
    }
    let add_vision = |items: &mut Vec<Item>| {
        items.push(Item::new(Category::Weights, "vision tower + projector", "BF16 + F32 conv/vectors", encoder.weights, Basis::Formula));
        items.push(Item::new(Category::Workspace, "vision scratch", "resident; 4096 merged tokens", encoder.scratch, Basis::Formula));
    };
    let gpu_items = |gpu, local| {
        let mut items = gpu_items(gpu, local);
        if encoder.kind == (EncoderKind::Rtx { gpu }) { add_vision(&mut items); }
        items
    };
    // An explicit onboard that overflows the GPU is refused below; an automatic
    // one is what the Sparks need, so the GPU check reports the shortfall.
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
        if encoder.spark_ranks().contains(&rank) { add_vision(&mut items); }
        memory.devices.push(DeviceLayout { kind: DeviceKind::Spark, index: rank as u32, capacity_bytes: budget, items, kv_tokens: 0 });
    }
    let gib = |b: u64| b as f64 / GIB as f64;
    memory.notes.push(format!("{} routed layers on the RTX ({:.1} GiB), {} on {ranks} Sparks at {:.1} GiB/rank (budget {:.0} GiB)",
        onboard, gib(local), layers.len() - onboard, gib(per_rank), gib(budget)));
    memory.notes.push(format!("pool {pool_tokens} tokens ({:.2} full 1M-token contexts) at {token} B/token per GPU; vision {}; drafter {}",
        pool_tokens as f64 / cfg.max_position as f64,
        format!("{:?} ({} bytes admitted)", encoder.kind, encoder.admitted_bytes()),
        if drafter > 0 { "on rtx0" } else { "off" }));
    if per_rank + SPARK_RESERVE_BYTES + if encoder.spark_ranks().is_empty() { 0 } else { encoder.admitted_bytes() } > budget {
        refusal(report, format!("MiniMax M3 TP{ranks} Spark ranks need {:.1} GiB with {onboard} onboarded layers, over the {:.0} GiB budget",
            gib(per_rank + SPARK_RESERVE_BYTES), gib(budget)),
            "Onboard more bottom routed layers to the RTX (--rtx-expert-layers), add Spark ranks, or raise --spark-budget-gib after a load census.");
    }
    if pool_tokens < POOL_FLOOR_TOKENS {
        report.fits = false;
        report.hints.push(Hint {
            what: format!("MiniMax M3 coordinator pool {pool_tokens} tokens is below one 256K request ({POOL_FLOOR_TOKENS})"),
            how: "Use --vision off or add Spark ranks to reduce coordinator residency; provide --workspace-manifest \
                for the native allocation ledger instead of this header lower bound.".into(),
        });
    }
    report.encoder = Some(encoder);
    memory
}

fn exact_layout(model: &Model, input: &crate::plan::PlanInput, report: &mut PlanReport,
    opts: &crate::plan::layout::LayoutOptions, requested: ExpertPlacement, path: &std::path::Path) -> anyhow::Result<MemoryLayout> {
    use anyhow::{ensure, Context};
    use crate::placement::{inventory, Onboard, ProgramSet, SparkAssignment, SparkResidentLayout, SparkWorkingSet};
    use admission::{AdmissionError, Reservations, RuntimeLedger, RuntimeSizing};
    ensure!(model.complete, "MiniMax native admission requires complete checkpoint headers");
    ensure!(opts.rtx_bytes.len() == 1, AdmissionError::Unsupported("two-GPU serving"));
    ensure!(opts.drafter_bytes == 0, AdmissionError::Unsupported("external drafter; use the bundled DSpark"));
    ensure!(report.requested_attention.is_none_or(|p| p == AttentionPlacement::Heads), AdmissionError::Unsupported("attention placement"));
    ensure!(report.audio == MediaMode::Off, AdmissionError::Unsupported("audio"));
    let cfg = model.cfg();
    let rows = if opts.prefill_rows == 0 { 4096 } else { usize::try_from(opts.prefill_rows)? };
    let sequences = if opts.concurrency == 0 { 64 } else { usize::try_from(opts.concurrency)? };
    ensure!(opts.state_slots.is_none_or(|n| n == sequences as u64), AdmissionError::Unsupported("sequence-slot mismatch"));
    let rings = opts.draft_context_slots.map_or(Ok(crate::families::minimax_m3::dspark::default_ring_slots(sequences)),
        usize::try_from)?;
    ensure!((1..=sequences).contains(&rings), AdmissionError::Unsupported("draft context slots outside 1..=max sequences"));
    let arch = ArchContext::coordinator(opts.rtx_bytes[0], None);
    let draft = if opts.native_mtp_layers > 0 {
        let contract = DsparkContract::open(&input.checkpoint().snapshot).context("MiniMax bundled DSpark")?;
        Some(DraftMemory::new(&contract, rings, rows, opts.physical_sms.unwrap_or(arch.sms) as usize)?)
    } else { None };
    let draft_rows = draft.as_ref().map_or(0, |d| d.sequences * 7);
    let sizing = RuntimeSizing { rows, decode_rows: rows.min(64), sequences, context: if opts.context_tokens == 0 {
        cfg.max_position as u64 } else { opts.context_tokens }, spark_ranks: requested.spark_ranks(),
        logit_rows: RuntimeSizing::head_rows(rows, opts.full_prefill_logits, draft_rows) };
    sizing.validate(cfg)?;
    let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(path).context("MiniMax PROGRAMS.json")?)?;
    let programs = admission::ProgramSizes::from_manifest(&manifest, sizing)?;
    let ledger = RuntimeLedger::new(cfg, sizing, &programs)?;
    let cache = MiniMaxCacheLayout::for_runtime(cfg, 1, sizing.context, opts.minimax_kv)?;
    let parent = path.parent().context("MiniMax PROGRAMS.json has no parent")?;
    let package_path = [parent.join("fp8/fp8-minimax-nvfp4a4/manifest.json"),
        parent.join("../lib/fp8/fp8-minimax-nvfp4a4/manifest.json"), parent.join("fp8-minimax-nvfp4a4/manifest.json")]
        .into_iter().find(|p| p.is_file());
    let package: Option<serde_json::Value> = package_path.map(|p| -> anyhow::Result<_> {
        Ok(serde_json::from_slice(&std::fs::read(p)?)?) }).transpose()?;
    // Remote packages are architecture-specific and need not be installed on
    // the coordinator. Live worker admission replaces this explicit estimate.
    let spark_manifest = package.as_ref().filter(|p| p["layouts"][format!("tp{}", sizing.spark_ranks)].is_object());
    let spark_scratch = spark_manifest.map(|p| admission::expert_workspace(p, sizing.spark_ranks, rows))
        .transpose()?.unwrap_or(512 << 20);
    let spark_scratch_basis = if spark_manifest.is_some() { Basis::Exact } else { Basis::Estimated };
    let weights = super::coordinator::RankWeightLayout::from_checkpoint(input.checkpoint(), 0, 1)?;
    let weights = if opts.host_embedding { weights.with_host_embedding(input.checkpoint())? } else { weights };
    let catalog = input.expert_catalog()?;
    let layers = cfg.moe.iter().enumerate().filter_map(|(l, &m)| m.then_some(l)).collect::<Vec<_>>();
    let resident = |selected: &[usize], world, rank| -> anyhow::Result<u64> {
        Ok(admission::expert_resident_bytes(catalog, selected, world, rank)?)
    };
    let edge = |component| *model.bytes.get(&component).unwrap_or(&0);
    let vision_source = edge(Component::Vision);
    let vector_bytes: u64 = model.contracts.values().filter(|c| c.role.component == Component::Vision)
        .map(|c| if c.shape.len() == 1 { c.shape[0] as u64 * 2 } else { 0 }
            + if c.bias { c.shape[0] as u64 * 2 } else { 0 }).sum();
    let vision_weights = if vision_source > 0 { vision_source + vector_bytes + 52 } else { 0 };
    let vision_scratch = if vision_source > 0 { crate::plan::layout::minimax_vision_scratch_bytes() } else { 0 };
    let spark_vision_reserve = if matches!(report.vision, MediaMode::Spark(_)) {
        vision_weights + vision_scratch + GIB } else { 0 };
    let endpoints = [cuteafd_core::SparkEndpoint { capacity: u32::try_from(rows)?, ingress: cuteafd_core::SparkIngress::Bf16 }];
    let rings = cuteafd_core::SparkRings { max_frame_bytes: 64 << 20, depth: 8, slot_bytes: 8 << 20, alignment: 4096 };
    let working = |onboard: usize| SparkWorkingSet { rtx_layers: layers[..onboard].to_vec(),
        spark_layers: layers[onboard..].to_vec(), spark_ranks: sizing.spark_ranks, prefill_lanes: 1 };
    let io = working(0).rank_endpoints_io(0, catalog.routed_experts().geometry()?, &endpoints, rings)?;
    let io_bytes = io.intake + io.host_exchange + io.row_indices + io.endpoint_staging + io.mapped_rings;
    let spark_overhead = io_bytes + spark_scratch + cuteafd_core::memory_layout::SPARK_RUNTIME_RESIDUAL_BYTES;
    let budget = opts.spark_allocation_budget_bytes.unwrap_or(SPARK_BUDGET_BYTES);
    let onboard_request = opts.onboard.or(opts.local_expert_layers.map(Onboard::Layers)).unwrap_or(Onboard::Auto);
    let requested_layers = match onboard_request {
        Onboard::Layers(n) => Some(n),
        Onboard::Auto => None,
        _ => return Err(AdmissionError::Unsupported("onboarding policy; use auto or a layer count").into()),
    };
    let onboarding = admission::onboard_layers(catalog, cfg, sizing, budget,
        spark_overhead + spark_vision_reserve, requested_layers)?;
    let onboard = onboarding.local_layers.len();
    let local = onboarding.local_bytes;
    let local_scratch = if onboard > 0 { admission::expert_workspace(package.as_ref()
        .context("MiniMax onboarded layers require fp8-minimax-nvfp4a4/manifest.json")?, 1, rows)? } else { 0 };
    let program_set = ProgramSet::from_manifest(&manifest, &["minimax"]);
    let code = inventory::loaded_code_allowance("minimax", "fp8-minimax-nvfp4a4", false, 0, &arch, Some(&program_set)).bytes;
    let mut reservations = Reservations { weights: weights.resident_bytes.checked_add(draft.as_ref().map_or(0, |d| d.bytes))
            .context("MiniMax drafter reservation overflow")?, local_experts: local,
        expert_workspace: local_scratch, pending_code: code, runtime_baseline: arch.context_bytes,
        headroom: opts.headroom_bytes, ..Default::default() };
    let fixed = reservations.fixed_bytes(&ledger)?;
    let hardware = EncoderHardware { v41: false, gpus: vec![EncoderGpuBudget {
        free_bytes: opts.rtx_bytes[0].saturating_sub(fixed),
        kv_target_bytes: cache.token_bytes(0) * opts.pool_tokens.filter(|n| *n > 0).unwrap_or(POOL_FLOOR_TOKENS),
    }], sparks: (0..sizing.spark_ranks).map(|rank| {
        let expert_bytes = resident(&layers[onboard..], sizing.spark_ranks, rank)?;
        Ok(EncoderSparkBudget { rank, host: String::new(), idle: false, expert_bytes,
            free_bytes: budget.saturating_sub(expert_bytes + spark_overhead) })
    }).collect::<anyhow::Result<Vec<_>>>()? };
    let encoder = encoder_placement(report.vision, &hardware, vision_weights, vision_scratch, opts.vision_replicas);
    ensure!(encoder.kind != EncoderKind::Off || !matches!(report.vision, MediaMode::Rtx(_) | MediaMode::Spark(_)),
        "MiniMax vision placement unavailable: {}", encoder.reason);
    reservations.media = if encoder.kind == (EncoderKind::Rtx { gpu: 0 }) { encoder.admitted_bytes() } else { 0 };
    // The cache is allocated last; source scales exist transiently beside the
    // native workspace and prepared destination during sequential weight load.
    reservations.startup = reservations.fixed_bytes(&ledger)?.checked_add(admission::scale_prep_staging_bytes(&weights))
        .context("MiniMax load peak overflow")?;
    let admitted = admission::admit(cfg, sizing, &ledger, reservations, opts.rtx_bytes[0], opts.pool_tokens, opts.minimax_kv)?;
    let mut items = vec![
        Item::new(Category::Weights, "native MXFP8 text, BF16 embedding/head, FP32 router", "checkpoint", weights.resident_bytes, Basis::Exact),
        Item::new(Category::Runtime, "CUDA context", "sm_120", arch.context_bytes, Basis::Calibrated),
        Item::new(Category::Runtime, "pending program code", "", code, Basis::Estimated),
        Item::new(Category::Reserved, "coordinator headroom", "", opts.headroom_bytes, Basis::Formula),
    ];
    for allocation in &ledger.allocations { items.push(Item::new(Category::Workspace,
        format!("MiniMax {}", allocation.name), "native allocation", allocation.bytes, Basis::Exact)); }
    if let Some(draft) = &draft {
        items.push(Item::new(Category::Drafter, "MiniMax DSpark weights", "MXFP8 + BF16", draft.weights_bytes, Basis::Exact));
        items.push(Item::new(Category::Drafter, "MiniMax DSpark context rings", "BF16 K/V", draft.ring_bytes, Basis::Exact));
        for (&name, &bytes) in &draft.buffers { items.push(Item::new(Category::Drafter,
            format!("MiniMax DSpark {name}"), "native allocation", bytes as u64, Basis::Exact)); }
    }
    if local > 0 {
        items.push(Item::new(Category::Experts, "onboarded bottom routed layers TP1", "NVFP4 W4A4", local, Basis::Exact));
        items.push(Item::new(Category::Workspace, "local expert package scratch", "exported capacity", local_scratch, Basis::Exact));
    }
    let add_vision = |items: &mut Vec<Item>| {
        items.push(Item::new(Category::Weights, "vision tower + projector", "BF16 + F32 conv/vectors", encoder.weights, Basis::Formula));
        items.push(Item::new(Category::Workspace, "vision scratch", "4096 merged tokens", encoder.scratch, Basis::Formula));
    };
    if reservations.media > 0 { add_vision(&mut items); }
    items.push(Item::new(Category::Kv, "separate FP8 K/V planes + packed MSA index", "64-token pages", admitted.cache_bytes, Basis::Exact));
    let mut memory = MemoryLayout { devices: vec![DeviceLayout { kind: DeviceKind::Rtx, index: 0,
        capacity_bytes: opts.rtx_bytes[0], items, kv_tokens: admitted.pool_tokens }], pool_tokens: admitted.pool_tokens,
        waste: vec![], notes: vec![format!("MiniMax coordinator native allocation ledger; causal paged MSA tiles; zero graphs; speculation {}",
            if draft.is_some() { "bundled DSpark" } else { "off" }),
            "Offline CUDA/code baselines remain estimates; serving re-admits measured free memory before allocation".into(),
            format!("{onboard} routed layers on the RTX ({:.1} GiB), {} on {} Sparks", local as f64 / GIB as f64, layers.len() - onboard, sizing.spark_ranks)] };
    report.bytes_by_owner.clear();
    report.bytes_by_owner.insert("coordinator text weights".into(), weights.resident_bytes);
    if let Some(draft) = &draft { report.bytes_by_owner.insert("bundled DSpark weights".into(), draft.weights_bytes); }
    report.bytes_by_owner.insert("local routed expert weights".into(), local);
    let recipe = crate::plan::experts::checkpoint_spark_recipe(input.checkpoint(), sizing.spark_ranks, &layers[onboard..])?;
    for rank in 0..sizing.spark_ranks {
        let assigned = SparkAssignment::from_catalog(catalog, &working(onboard), rank,
            SparkResidentLayout::Fp8 { world: sizing.spark_ranks, rank, slicing: recipe.slicing }, &endpoints, rings)?;
        let mut items = vec![Item::new(Category::Experts, "routed expert weights", "NVFP4 W4A4", assigned.resident_bytes, Basis::Exact),
            cuteafd_core::memory_layout::spark_runtime_residual(),
            Item::new(Category::Workspace, "expert package scratch", "exported capacity or remote estimate", spark_scratch, spark_scratch_basis)];
        for (name, bytes) in [("expert intake", assigned.io.intake), ("host expert exchange", assigned.io.host_exchange),
            ("expert row indices", assigned.io.row_indices), ("endpoint response prefixes", assigned.io.endpoint_staging),
            ("rdma rings", assigned.io.mapped_rings)] { items.push(Item::new(Category::Transport, name, "BF16", bytes, Basis::Formula)); }
        if encoder.spark_ranks().contains(&rank) { add_vision(&mut items); }
        let used: u64 = items.iter().map(|i| i.bytes).sum();
        ensure!(used <= budget, "MiniMax Spark rank {rank} needs {used} bytes, budget {budget}");
        *report.bytes_by_owner.entry("routed expert weights".into()).or_default() += assigned.resident_bytes;
        memory.devices.push(DeviceLayout { kind: DeviceKind::Spark, index: rank as u32, capacity_bytes: budget, items, kv_tokens: 0 });
    }
    memory.notes.push(format!("MiniMax shape: prefill {rows}, sequences {sequences}, decode tile {}, context {}; exported scratch is not scaled by runtime rows",
        programs.decode_capacity(), sizing.context));
    if opts.host_embedding {
        report.bytes_by_owner.insert("host-mapped embedding".into(), edge(Component::Embedding));
        memory.notes.push("embedding placement: host (mapped BF16); untied LM head remains on GPU".into());
    }
    report.prefill_rows = Some(rows as u64);
    report.attention_placement = AttentionPlacement::Heads;
    report.encoder = Some(encoder);
    if spark_manifest.is_none() { memory.notes.push("Remote expert scratch reserves 512 MiB per rank until measured worker admission; it is not an exact remote allocation ledger".into()); }
    Ok(memory)
}
