//! The native serving allocation ledger, from the image's own program metadata.
use super::{admission, Model};
use crate::families::minimax_m3::dspark::{DraftMemory, DsparkContract};
use crate::placement::{ArchContext, AttentionPlacement};
use crate::plan::{Component, ExpertPlacement, MediaMode, PlanReport};
use crate::plan::encoder::{encoder_placement, EncoderGpuBudget, EncoderHardware, EncoderKind, EncoderSparkBudget};
use cuteafd_core::memory_layout::{Basis, Category, DeviceKind, DeviceLayout, Item, MemoryLayout};

const GIB: u64 = 1 << 30;
/// Per-rank routed-expert allocation ceiling without `--spark-budget-gib`
/// (the Kimi K3 precedent: a GB10 rank keeps ~113-116 GiB available idle).
/// Estimated until the first full worker load/steady census.
pub const SPARK_BUDGET_BYTES: u64 = 108 << 30;
/// The KV an RTX-hosted tower must leave beside it when no pool is requested:
/// one full 256K agentic request.
const POOL_FLOOR_TOKENS: u64 = 262_144;

pub(super) fn layout(model: &Model, input: &crate::plan::PlanInput, report: &mut PlanReport,
    opts: &crate::plan::layout::LayoutOptions, requested: ExpertPlacement) -> anyhow::Result<MemoryLayout> {
    use anyhow::{ensure, Context};
    use crate::placement::{inventory, Onboard, ProgramSet, SparkAssignment, SparkResidentLayout};
    use admission::{AdmissionError, RuntimeSizing};
    ensure!(model.complete, "MiniMax native admission requires complete checkpoint headers");
    ensure!(opts.rtx_bytes.len() == 1, AdmissionError::Unsupported("two-GPU serving"));
    ensure!(opts.drafter_bytes == 0, AdmissionError::Unsupported("external drafter; use the bundled DSpark"));
    ensure!(report.requested_attention.is_none_or(|p| p == AttentionPlacement::Heads), AdmissionError::Unsupported("attention placement"));
    ensure!(report.audio == MediaMode::Off, AdmissionError::Unsupported("audio"));
    let cfg = model.cfg();
    let rows = if opts.prefill_rows == 0 { 4096 } else { usize::try_from(opts.prefill_rows)? };
    let lanes = if opts.prefill_lanes == 0 { crate::serving_capacity::prefill_lanes::DEFAULT_PREFILL_LANES }
        else { usize::try_from(opts.prefill_lanes)? };
    let sequences = if opts.concurrency == 0 { 64 } else { usize::try_from(opts.concurrency)? };
    ensure!(opts.state_slots.is_none_or(|n| n == sequences as u64), AdmissionError::Unsupported("sequence-slot mismatch"));
    let rings = opts.draft_context_slots.map_or(Ok(crate::families::minimax_m3::dspark::default_ring_slots(sequences)),
        usize::try_from)?;
    ensure!((1..=sequences).contains(&rings), AdmissionError::Unsupported("draft context slots outside 1..=max sequences"));
    let arch = ArchContext::coordinator(opts.rtx_bytes[0], None);
    let chunk = crate::serving_capacity::prefill_lanes::lane_capacity(lanes, rows, crate::families::minimax_m3::cache::PAGE_ROWS as usize);
    let draft = if opts.native_mtp_layers > 0 {
        let contract = DsparkContract::open(&input.checkpoint().snapshot).context("MiniMax bundled DSpark")?;
        Some(DraftMemory::new(&contract, rings, chunk, opts.physical_sms.unwrap_or(arch.sms) as usize)?)
    } else { None };
    let draft_rows = draft.as_ref().map_or(0, |d| d.sequences * 7);
    let sizing = RuntimeSizing { rows, decode_rows: rows.min(64), sequences, context: if opts.context_tokens == 0 {
        cfg.max_position as u64 } else { opts.context_tokens }, spark_ranks: requested.spark_ranks(),
        logit_rows: RuntimeSizing::head_rows(chunk, opts.full_prefill_logits, draft_rows), lanes,
        host_landing: !arch.gpu_landing };
    sizing.validate(cfg)?;
    let path = opts.workspace_manifest.as_deref().unwrap_or_else(|| std::path::Path::new("/opt/cuteafd/share/PROGRAMS.json"));
    let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(path)
        .context("MiniMax native admission requires its matching PROGRAMS.json")?)?;
    let parent = path.parent().context("MiniMax PROGRAMS.json has no parent")?;
    let package_path = [parent.join("fp8/fp8-minimax-nvfp4a4/manifest.json"),
        parent.join("../lib/fp8/fp8-minimax-nvfp4a4/manifest.json"), parent.join("fp8-minimax-nvfp4a4/manifest.json")]
        .into_iter().find(|p| p.is_file());
    let package: Option<serde_json::Value> = package_path.map(|p| -> anyhow::Result<_> {
        Ok(serde_json::from_slice(&std::fs::read(p)?)?) }).transpose()?;
    let catalog = input.expert_catalog()?;
    let edge = |component| *model.bytes.get(&component).unwrap_or(&0);
    let vision_source = edge(Component::Vision);
    let vector_bytes: u64 = model.contracts.values().filter(|c| c.role.component == Component::Vision)
        .map(|c| if c.shape.len() == 1 { c.shape[0] as u64 * 2 } else { 0 }
            + if c.bias { c.shape[0] as u64 * 2 } else { 0 }).sum();
    let vision_weights = if vision_source > 0 { vision_source + vector_bytes + 52 } else { 0 };
    let vision_scratch = if vision_source > 0 { crate::plan::layout::minimax_vision_scratch_bytes() } else { 0 };
    let budget = opts.spark_allocation_budget_bytes.unwrap_or(SPARK_BUDGET_BYTES);
    let local_layers = match opts.onboard.or(opts.local_expert_layers.map(Onboard::Layers)).unwrap_or(Onboard::Auto) {
        Onboard::Layers(n) => admission::LocalLayers::Bottom(n),
        Onboard::Auto => admission::LocalLayers::Fewest,
        _ => return Err(AdmissionError::Unsupported("onboarding policy; use auto or a layer count").into()),
    };
    // The serving recipe, with the rings the launch's Spark links register by default.
    let ring = crate::plan::layout::spark_ring_geometry(lanes);
    let mut recipe = admission::Recipe::new(cfg, input.checkpoint(), catalog, &manifest, package.as_ref(), admission::Request {
        sizing, kv_format: opts.minimax_kv, local_layers, spark_budget: budget,
        spark_vision_bytes: if matches!(report.vision, MediaMode::Spark(_)) { vision_weights + vision_scratch + GIB } else { 0 },
        headroom: opts.headroom_bytes, draft_bytes: draft.as_ref().map_or(0, |d| d.bytes),
        draft_rings: draft.as_ref().map_or(0, |d| d.slots), host_embedding: opts.host_embedding, startup_graphs: true,
        rings: cuteafd_core::SparkRings { max_frame_bytes: ring.max_frame_bytes, depth: ring.depth,
            slot_bytes: ring.slot_bytes, alignment: ring.alignment } })?;
    let local = recipe.onboarding.local_bytes;
    let onboarded = recipe.onboarding.local_layers.len();
    let program_set = ProgramSet::from_manifest(&manifest, &["minimax"]);
    let code = inventory::loaded_code_allowance("minimax", "fp8-minimax-nvfp4a4", false, 0, &arch, Some(&program_set)).bytes;
    // What the server's measured sample already holds: the CUDA context and the loaded code.
    (recipe.reservations.pending_code, recipe.reservations.runtime_baseline) = (code, arch.context_bytes);
    let fixed = recipe.reservations.fixed_bytes(&recipe.ledger)?;
    let hardware = EncoderHardware { v41: false, gpus: vec![EncoderGpuBudget {
        free_bytes: opts.rtx_bytes[0].saturating_sub(fixed),
        kv_target_bytes: recipe.cache.token_bytes() * opts.pool_tokens.filter(|n| *n > 0).unwrap_or(POOL_FLOOR_TOKENS),
    }], sparks: (0..sizing.spark_ranks).map(|rank| {
        let expert_bytes = admission::expert_resident_bytes(catalog, &recipe.onboarding.spark_layers, sizing.spark_ranks, rank)?;
        Ok(EncoderSparkBudget { rank, host: String::new(), idle: false, expert_bytes,
            free_bytes: budget.saturating_sub(expert_bytes + recipe.remote_reserved) })
    }).collect::<anyhow::Result<Vec<_>>>()? };
    let encoder = encoder_placement(report.vision, &hardware, vision_weights, vision_scratch, opts.vision_replicas);
    ensure!(encoder.kind != EncoderKind::Off || !matches!(report.vision, MediaMode::Rtx(_) | MediaMode::Spark(_)),
        "MiniMax vision placement unavailable: {}", encoder.reason);
    recipe.reservations.media = if encoder.kind == (EncoderKind::Rtx { gpu: 0 }) { encoder.admitted_bytes() } else { 0 };
    let (admitted, _) = recipe.admit(cfg, opts.rtx_bytes[0], opts.pool_tokens)?;
    let weights = &recipe.weights;
    let mut items = vec![
        Item::new(Category::Weights, "native MXFP8 text, BF16 embedding/head, FP32 router", "checkpoint", weights.resident_bytes, Basis::Exact),
        Item::new(Category::Runtime, "CUDA context", "sm_120", arch.context_bytes, Basis::Calibrated),
        Item::new(Category::Runtime, "pending program code", "", code, Basis::Estimated),
        Item::new(Category::Reserved, "coordinator headroom", "", opts.headroom_bytes, Basis::Formula),
    ];
    items.extend(recipe.graphs.inventory.items(0).into_iter().map(|item| Item { basis: Basis::Estimated, ..item }));
    for allocation in &recipe.ledger.allocations { items.push(Item::new(Category::Workspace,
        format!("MiniMax {}", allocation.name), "native allocation", allocation.bytes, Basis::Exact)); }
    if let Some(draft) = &draft {
        items.push(Item::new(Category::Drafter, "MiniMax DSpark weights", "MXFP8 + BF16", draft.weights_bytes, Basis::Exact));
        items.push(Item::new(Category::Drafter, "MiniMax DSpark context rings", "BF16 K/V", draft.ring_bytes, Basis::Exact));
        for (&name, &bytes) in &draft.buffers { items.push(Item::new(Category::Drafter,
            format!("MiniMax DSpark {name}"), "native allocation", bytes as u64, Basis::Exact)); }
    }
    if local > 0 {
        items.push(Item::new(Category::Experts, "onboarded bottom routed layers TP1", "NVFP4 W4A4", local, Basis::Exact));
        items.push(Item::new(Category::Workspace, "local expert package scratch", "exported capacity",
            recipe.reservations.expert_workspace, Basis::Exact));
    }
    let add_vision = |items: &mut Vec<Item>| {
        items.push(Item::new(Category::Weights, "vision tower + projector", "BF16 + F32 conv/vectors", encoder.weights, Basis::Formula));
        items.push(Item::new(Category::Workspace, "vision scratch", "4096 merged tokens", encoder.scratch, Basis::Formula));
    };
    if recipe.reservations.media > 0 { add_vision(&mut items); }
    items.push(Item::new(Category::Kv, "separate FP8 K/V planes + packed MSA index", "64-token pages", admitted.cache_bytes, Basis::Exact));
    let graphs = &recipe.graphs;
    let mut memory = MemoryLayout { devices: vec![DeviceLayout { kind: DeviceKind::Rtx, index: 0,
        capacity_bytes: opts.rtx_bytes[0], items, kv_tokens: admitted.pool_tokens }], pool_tokens: admitted.pool_tokens,
        waste: vec![], notes: vec![format!("MiniMax coordinator native allocation ledger; {} decode steps recorded at startup \
            (rows 1..={}, estimated {} MiB each); speculation {}", graphs.keys.len(), graphs.keys.len(),
            admission::DECODE_GRAPH_BYTES >> 20, if draft.is_some() { "bundled DSpark" } else { "off" }),
            "Offline CUDA/code baselines remain estimates; serving re-admits measured free memory before allocation".into(),
            format!("{onboarded} routed layers on the RTX ({:.1} GiB), {} on {} Sparks", local as f64 / GIB as f64,
                recipe.onboarding.spark_layers.len(), sizing.spark_ranks)] };
    report.bytes_by_owner.clear();
    report.bytes_by_owner.insert("coordinator text weights".into(), weights.resident_bytes);
    if let Some(draft) = &draft { report.bytes_by_owner.insert("bundled DSpark weights".into(), draft.weights_bytes); }
    report.bytes_by_owner.insert("local routed expert weights".into(), local);
    let slicing = crate::plan::experts::checkpoint_spark_recipe(input.checkpoint(), sizing.spark_ranks,
        &recipe.onboarding.spark_layers)?.slicing;
    let working = recipe.working_set();
    let (remote_scratch, remote_basis) = recipe.remote_scratch.map_or(
        (admission::REMOTE_SCRATCH_ESTIMATE_BYTES, Basis::Estimated), |bytes| (bytes, Basis::Exact));
    for rank in 0..sizing.spark_ranks {
        let assigned = SparkAssignment::from_catalog(catalog, &working, rank,
            SparkResidentLayout::Fp8 { world: sizing.spark_ranks, rank, slicing }, &recipe.endpoints, recipe.request.rings)?;
        let mut items = vec![Item::new(Category::Experts, "routed expert weights", "NVFP4 W4A4", assigned.resident_bytes, Basis::Exact),
            cuteafd_core::memory_layout::spark_runtime_residual(),
            Item::new(Category::Workspace, "expert package scratch", "exported capacity or remote estimate", remote_scratch, remote_basis)];
        for (name, bytes) in [("expert intake", assigned.io.intake), ("host expert exchange", assigned.io.host_exchange),
            ("expert row indices", assigned.io.row_indices), ("endpoint response prefixes", assigned.io.endpoint_staging),
            ("rdma rings", assigned.io.mapped_rings)] { items.push(Item::new(Category::Transport, name, "BF16", bytes, Basis::Formula)); }
        if encoder.spark_ranks().contains(&rank) { add_vision(&mut items); }
        let used: u64 = items.iter().map(|i| i.bytes).sum();
        ensure!(used <= budget, "MiniMax Spark rank {rank} needs {used} bytes, budget {budget}");
        *report.bytes_by_owner.entry("routed expert weights".into()).or_default() += assigned.resident_bytes;
        memory.devices.push(DeviceLayout { kind: DeviceKind::Spark, index: rank as u32, capacity_bytes: budget, items, kv_tokens: 0 });
    }
    memory.notes.push(format!("MiniMax shape: prefill {lanes} lanes of {rows} rows, sequences {sequences}, decode tile {}, context {}, Spark partials land in {}; exported scratch is not scaled by runtime rows",
        recipe.programs.decode_capacity(), sizing.context, if sizing.host_landing { "pinned host planes" } else { "GPU planes" }));
    if opts.host_embedding {
        report.bytes_by_owner.insert("host-mapped embedding".into(), edge(Component::Embedding));
        memory.notes.push("embedding placement: host (mapped BF16); untied LM head remains on GPU".into());
    }
    report.prefill_rows = Some(rows as u64);
    report.prefill_lanes = Some(lanes as u64);
    report.attention_placement = AttentionPlacement::Heads;
    report.encoder = Some(encoder);
    if recipe.remote_scratch.is_none() { memory.notes.push("Remote expert scratch reserves 512 MiB per rank until measured worker admission; it is not an exact remote allocation ledger".into()); }
    Ok(memory)
}
