//! The native serving allocation ledger, from the image's own program metadata.
use super::{admission, Model};
use crate::families::glm5::draft_representation::GlmDraftRepresentation;
use crate::families::minimax_m3::{dflash::DflashMemory, dspark::{DraftMemory, DsparkContract}};
use crate::placement::{ArchContext, AttentionPlacement};
use crate::plan::{Component, ExpertPlacement, MediaMode, PlanReport};
use crate::plan::encoder::{encoder_placement, EncoderGpuBudget, EncoderHardware, EncoderKind, EncoderSparkBudget};
use cuteafd_core::memory_layout::{Basis, Category, DeviceKind, DeviceLayout, Item, MemoryLayout};

const GIB: u64 = 1 << 30;

/// MiniMax on this engine serves an RTX PRO 6000 coordinator, whose memory the NIC lands Spark
/// replies in. The RTX 5090 is not supported yet: a coordinator budget of the planner's RTX 5090
/// class (`ArchContext::for_device`) is refused by name.
pub(super) fn refuse_rtx_5090(budgets: &[u64]) -> anyhow::Result<()> {
    use crate::placement::inventory::PRO_TOTAL_BYTES;
    let pro = ArchContext::for_device("sm_120", PRO_TOTAL_BYTES);
    for &budget in budgets {
        let class = ArchContext::for_device("sm_120", budget);
        anyhow::ensure!(class == pro, "MiniMax on an RTX 5090 is not supported on this engine yet: a {:.1} GiB \
            coordinator budget is the RTX 5090 class; serve MiniMax from an RTX PRO 6000", budget as f64 / GIB as f64);
    }
    Ok(())
}

pub(super) fn layout(model: &Model, input: &crate::plan::PlanInput, report: &mut PlanReport,
    opts: &mut crate::plan::layout::LayoutOptions, requested: ExpertPlacement) -> anyhow::Result<MemoryLayout> {
    use anyhow::{ensure, Context};
    use crate::placement::{inventory, Onboard, ProgramSet, SparkAssignment, SparkResidentLayout};
    use admission::{AdmissionError, RuntimeSizing};
    refuse_rtx_5090(&opts.rtx_bytes)?;
    ensure!(model.complete, "MiniMax native admission requires complete checkpoint headers");
    ensure!(opts.rtx_bytes.len() == 1, AdmissionError::Unsupported("two-GPU serving"));
    ensure!(opts.drafter_bytes == 0,
        AdmissionError::Unsupported("an external drafter by size; use the bundled DSpark or a DFlash2 snapshot"));
    ensure!(report.requested_attention.is_none_or(|p| p == AttentionPlacement::Heads), AdmissionError::Unsupported("attention placement"));
    ensure!(report.audio == MediaMode::Off, AdmissionError::Unsupported("audio"));
    let cfg = model.cfg();
    // The shared prefill contract (#762): admitted slots of the unit on one Spark transport, the depth
    // starting at the estimate's. `cuteafd plan` passes --prefill-waves as `prefill_lanes` as well,
    // which is how an explicit count (a pinned depth) shows.
    let prefill = admission::prefill_shape(opts.prefill_rows, opts.prefill_lanes, 0, requested.spark_ranks())
        .context("MiniMax prefill shape")?;
    (opts.prefill_unit_rows, opts.prefill_waves, opts.prefill_slots) = (Some(prefill.unit_rows), prefill.waves, prefill.slots);
    (report.prefill_span_rows, report.prefill_transports) = (Some(prefill.span_rows), Some(1));
    let (rows, lanes, slots, span) = (usize::try_from(prefill.unit_rows)?, usize::try_from(prefill.waves)?,
        usize::try_from(prefill.slots)?, usize::try_from(prefill.span_rows)?);
    let sequences = if opts.concurrency == 0 { 64 } else { usize::try_from(opts.concurrency)? };
    ensure!(opts.state_slots.is_none_or(|n| n == sequences as u64), AdmissionError::Unsupported("sequence-slot mismatch"));
    let rings = opts.draft_context_slots.map_or(Ok(admission::default_draft_rings(sequences,
        rows.min(admission::DECODE_ROWS))), usize::try_from)?;
    ensure!((1..=sequences).contains(&rings), AdmissionError::Unsupported("draft context slots outside 1..=max sequences"));
    let arch = ArchContext::coordinator(opts.rtx_bytes[0], None);
    let draft = if opts.native_mtp_layers > 0 {
        let contract = DsparkContract::open(&input.checkpoint().snapshot).context("MiniMax bundled DSpark")?;
        Some(DraftMemory::new(&contract, rings, span, opts.physical_sms.unwrap_or(arch.sms) as usize)?)
    } else { None };
    // A DFlash2 checkpoint on the shared drafter, sized by serve-minimax's own recipe.
    let dflash = opts.minimax_drafter_snapshot.as_ref().map(|snapshot| -> anyhow::Result<_> {
        ensure!(draft.is_none(), AdmissionError::Unsupported("the bundled DSpark beside a DFlash2 drafter"));
        let config = crate::plan::checkpoint::read_json(&snapshot.join("config.json")).context("MiniMax DFlash2 drafter")?;
        DflashMemory::new(&config, rings, opts.physical_sms.unwrap_or(arch.sms) as u64,
            if opts.minimax_draft_bf16 { GlmDraftRepresentation::Bf16Only } else { GlmDraftRepresentation::Fp8Only },
            opts.draft_kv_cache)
    }).transpose()?;
    // DFlash2 projects its drafts through its own logits; DSpark through the target's.
    let draft_rows = draft.as_ref().map_or(0, |d| d.sequences * 7);
    let sizing = RuntimeSizing { unit_rows: rows, decode_rows: rows.min(admission::DECODE_ROWS), sequences, context: if opts.context_tokens == 0 {
        cfg.max_position as u64 } else { opts.context_tokens }, spark_ranks: requested.spark_ranks(),
        logit_rows: RuntimeSizing::head_rows(span, opts.full_prefill_logits, draft_rows), waves: lanes, slots }
        .with_prefill(prefill)?;
    sizing.validate(cfg)?;
    let path = opts.workspace_manifest.as_deref().unwrap_or_else(|| std::path::Path::new("/opt/cuteafd/share/PROGRAMS.json"));
    let manifest = crate::plan::checkpoint::read_json(path).context("MiniMax native admission requires its matching PROGRAMS.json")?;
    let parent = path.parent().context("MiniMax PROGRAMS.json has no parent")?;
    let package_path = [parent.join("fp8/fp8-minimax-nvfp4a4/manifest.json"),
        parent.join("../lib/fp8/fp8-minimax-nvfp4a4/manifest.json"), parent.join("fp8-minimax-nvfp4a4/manifest.json")]
        .into_iter().find(|p| p.is_file());
    let package = package_path.map(|p| crate::plan::checkpoint::read_json(&p)).transpose()?;
    let catalog = input.expert_catalog()?;
    let edge = |component| *model.bytes.get(&component).unwrap_or(&0);
    let vision_source = edge(Component::Vision);
    let vector_bytes: u64 = model.contracts.values().filter(|c| c.role.component == Component::Vision)
        .map(|c| if c.shape.len() == 1 { c.shape[0] as u64 * 2 } else { 0 }
            + if c.bias { c.shape[0] as u64 * 2 } else { 0 }).sum();
    let vision_weights = if vision_source > 0 { vision_source + vector_bytes + 52 } else { 0 };
    let vision_scratch = if vision_source > 0 { crate::plan::layout::minimax_vision_scratch_bytes() } else { 0 };
    let budget = opts.spark_allocation_budget_bytes.unwrap_or(admission::SPARK_BUDGET_BYTES);
    let local_layers = match opts.onboard.or(opts.local_expert_layers.map(Onboard::Layers)).unwrap_or(Onboard::Auto) {
        Onboard::Layers(n) => admission::LocalLayers::Bottom(n),
        Onboard::Auto => admission::LocalLayers::Fewest,
        _ => return Err(AdmissionError::Unsupported("onboarding policy; use auto or a layer count").into()),
    };
    // The serving recipe, with the rings the launch's one Spark link registers by default.
    let ring = crate::plan::layout::spark_ring_geometry(1);
    let mut recipe = admission::Recipe::new(cfg, input.checkpoint(), catalog, &manifest, package.as_ref(), admission::Request {
        sizing, kv_format: opts.minimax_kv, local_layers, spark_budget: budget,
        spark_vision_bytes: if matches!(report.vision, MediaMode::Spark(_)) { vision_weights + vision_scratch + GIB } else { 0 },
        headroom: opts.headroom_bytes, draft_bytes: draft.as_ref().map(|d| d.bytes).or(dflash.map(|d| d.bytes)).unwrap_or(0),
        draft_rings: draft.as_ref().map(|d| d.slots).or(dflash.map(|d| d.slots)).unwrap_or(0),
        draft_staging: dflash.map_or(0, |d| d.staging), host_embedding: opts.host_embedding, startup_graphs: false,
        rings: cuteafd_core::SparkRings { max_frame_bytes: ring.max_frame_bytes, depth: ring.depth,
            slot_bytes: ring.slot_bytes, alignment: ring.alignment } })?;
    let local = recipe.onboarding.local_bytes;
    let onboarded = recipe.onboarding.local_layers.len();
    let program_set = ProgramSet::from_manifest(&manifest, &["minimax"]);
    let code = inventory::loaded_code_allowance("minimax", "fp8-minimax-nvfp4a4", false, 0, &arch, Some(&program_set)).bytes;
    // What the server's measured sample already holds: the CUDA context and the loaded code.
    (recipe.reservations.pending_code, recipe.reservations.runtime_baseline) = (code, arch.context_bytes);
    let fixed = recipe.reservations.fixed_bytes(&recipe.ledger)?;
    // As serve-minimax admits automatic vision: the tower stays on the coordinator GPU whenever it
    // leaves a KV pool, which takes what is left (a requested pool must fit beside the tower).
    let hardware = EncoderHardware { v41: false, gpus: vec![EncoderGpuBudget {
        free_bytes: opts.rtx_bytes[0].saturating_sub(fixed), kv_target_bytes: 0,
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
    if let Some(dflash) = &dflash {
        let format = if opts.minimax_draft_bf16 { "BF16" } else { "E4M3 GEMMs + BF16" };
        items.push(Item::new(Category::Drafter, "MiniMax DFlash2 weights", format, dflash.weights_bytes, Basis::Exact));
        items.push(Item::new(Category::Drafter, "MiniMax DFlash2 context rings", format!("{} K/V", opts.draft_kv_cache),
            dflash.ring_bytes, Basis::Exact));
        items.push(Item::new(Category::Drafter, "MiniMax DFlash2 taps, context and step buffers", "shared drafter",
            dflash.bytes - dflash.weights_bytes - dflash.ring_bytes, Basis::Formula));
    }
    if local > 0 {
        items.push(Item::new(Category::Experts, "onboarded bottom routed layers TP1", "NVFP4 W4A4", local, Basis::Exact));
        items.push(Item::new(Category::Workspace, "local expert package scratch", "exported capacity",
            recipe.reservations.expert_workspace, Basis::Exact));
    }
    let add_vision = |items: &mut Vec<Item>| {
        items.push(Item::new(Category::Weights, "vision tower + projector", "BF16 + F32 conv/vectors", encoder.weights, Basis::Formula));
        items.push(Item::new(Category::Workspace, "vision scratch", "one image, 576 merged tokens", encoder.scratch, Basis::Formula));
    };
    if recipe.reservations.media > 0 { add_vision(&mut items); }
    items.push(Item::new(Category::Kv, "separate FP8 K/V planes + packed MSA index", "64-token pages", admitted.cache_bytes, Basis::Exact));
    let mut memory = MemoryLayout { devices: vec![DeviceLayout { kind: DeviceKind::Rtx, index: 0,
        capacity_bytes: opts.rtx_bytes[0], items, kv_tokens: admitted.pool_tokens }], pool_tokens: admitted.pool_tokens,
        waste: vec![], notes: vec![format!("MiniMax coordinator native allocation ledger; decode runs eagerly, with no \
            startup decode graphs; speculation {}",
            if draft.is_some() { "bundled DSpark" } else if dflash.is_some() { "DFlash2 (shared drafter)" } else { "off" }),
            "Offline CUDA/code baselines remain estimates; serving re-admits measured free memory before allocation".into(),
            format!("{onboarded} routed layers on the RTX ({:.1} GiB), {} on {} Sparks", local as f64 / GIB as f64,
                recipe.onboarding.spark_layers.len(), sizing.spark_ranks)]
            .into_iter().chain(draft.as_ref().map(|d| (d.slots, d.ring_bytes)).or(dflash.map(|d| (d.slots, d.ring_bytes)))
                .map(|(slots, bytes)| format!("{slots} drafter context rings of {:.1} MiB each come off the pool \
                    (default min(sequences, decode rows / 2); --draft-context-slots sets them)",
                    bytes as f64 / slots as f64 / (1u64 << 20) as f64)))
            .collect() };
    report.bytes_by_owner.clear();
    report.bytes_by_owner.insert("coordinator text weights".into(), weights.resident_bytes);
    if let Some(draft) = &draft { report.bytes_by_owner.insert("bundled DSpark weights".into(), draft.weights_bytes); }
    if let Some(dflash) = &dflash { report.bytes_by_owner.insert("DFlash2 drafter weights".into(), dflash.weights_bytes); }
    report.bytes_by_owner.insert("local routed expert weights".into(), local);
    let slicing = crate::plan::experts::checkpoint_spark_recipe(input.checkpoint(), sizing.spark_ranks,
        &recipe.onboarding.spark_layers)?.slicing;
    let working = recipe.working_set();
    for rank in 0..sizing.spark_ranks {
        let assigned = SparkAssignment::from_catalog(catalog, &working, rank,
            SparkResidentLayout::Fp8 { world: sizing.spark_ranks, rank, slicing }, &recipe.endpoints, recipe.request.rings)?;
        let mut items = vec![Item::new(Category::Experts, "routed expert weights", "NVFP4 W4A4", assigned.resident_bytes, Basis::Exact),
            cuteafd_core::memory_layout::spark_runtime_residual()];
        for (name, bytes) in [("expert intake", assigned.io.intake), ("host expert exchange", assigned.io.host_exchange),
            ("expert row indices", assigned.io.row_indices), ("endpoint response prefixes", assigned.io.endpoint_staging),
            ("rdma rings", assigned.io.mapped_rings)] { items.push(Item::new(Category::Transport, name, "BF16", bytes, Basis::Formula)); }
        if encoder.spark_ranks().contains(&rank) { add_vision(&mut items); }
        let mut device = DeviceLayout { kind: DeviceKind::Spark, index: rank as u32, capacity_bytes: budget, items, kv_tokens: 0 };
        // The typed owners, through the owner sum the worker admits (one decode, one prefill owner).
        crate::plan::layout::spark_slots::charge_owners(&mut device, &recipe.executions,
            |extent| Ok(admission::spark_owner_bytes(package.as_ref(), sizing.spark_ranks, extent)?))?;
        let used: u64 = device.items.iter().map(|i| i.bytes).sum();
        ensure!(used <= budget, "MiniMax Spark rank {rank} needs {used} bytes, budget {budget}");
        *report.bytes_by_owner.entry("routed expert weights".into()).or_default() += assigned.resident_bytes;
        memory.devices.push(device);
    }
    memory.notes.push(format!("MiniMax shape: prefill {slots} admitted slots of up to {rows} rows (span {span}) on one \
        Spark link, starting at {lanes} waves{}; each rank's typed owners: one decode owner and one prefill owner buffering \
        {slots} units; sequences {sequences}, decode tile {}, context {}; Spark replies land in GPU planes; exported scratch \
        is not scaled by runtime rows", if sizing.adaptive() { " (the engine chooses each call's depth at runtime)" } else { "" },
        recipe.programs.decode_capacity(), sizing.context));
    if opts.host_embedding {
        report.bytes_by_owner.insert("host-mapped embedding".into(), edge(Component::Embedding));
        memory.notes.push("embedding placement: host (mapped BF16); untied LM head remains on GPU".into());
    }
    report.attention_placement = AttentionPlacement::Heads;
    report.encoder = Some(encoder);
    if !recipe.exported_scratch { memory.notes.push("Remote expert scratch is estimated (512 MiB at the m4096 export, scaled by \
        export) until the package's layout for the Spark world is installed beside PROGRAMS.json; it is not an exact remote \
        allocation ledger".into()); }
    Ok(memory)
}
