//! Muse Spark physical owners for the shared contract planner.
//!
//! A family supplies facts, never a private solver: this module names every
//! owner Muse charges (layer weights, KV, rings, marks, workspaces, transport,
//! expert arenas, bundled towers, drafter), where it lives and for which
//! lifetime. `cuteafd plan --layout` and serving startup build the same recipe
//! and run the same solve, so the two edges differ only in the coordinator
//! baseline (a planned forecast versus one measured sample) and in whether the
//! image's code is already loaded.
use super::{combine_tables_bytes, executor, intake, mark_bytes, ring_rows, rope_tables, spark_endpoint,
    spark_reserve_bytes, step_tables, tp2_exchange_bytes, tp2_partial_bytes, LocalPackage, MediaAdmission,
    MediaOptions, MuseDrafter, LANES, MUSE_SPARK, PAGE_ROWS};
use crate::families::muse_spark::{core::CoreInventory, weight_layout::{GlobalLayout, LayerLayout},
    MuseAttention, MuseSparkConfig};
use crate::placement::planner::*;
use crate::placement::{plan_hops, residual_homes, AttentionClass, AttentionPlacement, Demand, ExpertHome,
    ExpertRange, HopSpec, HopStorage, LayerAssignment, LayerMode, Onboard, Placement, PoolLimits, PoolPolicy,
    ResidentWeights, SparkBudgetInventory, SparkWorkingSet, TerminalHome, Tp2Range};
use crate::plan::encoder::EncoderPlacement;
use crate::plan::MediaMode;
use crate::serving_capacity::prefill_waves::PrefillPlan;
use cuteafd_core::memory_layout::{Basis, Category, Item};
use cuteafd_core::spark_reply::SparkReplyFormats;
use anyhow::Context as _;

/// A condition names a domain value, never a solver variable index: one
/// onboarding count or one joint bundled-tower option, or nothing at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum When {
    Onboard(usize),
    Encoder(usize),
}

#[derive(Debug, Clone)]
pub struct Case<T> {
    pub when: Vec<When>,
    pub value: T,
}

fn invalid(detail: &str) -> Refusal {
    Refusal::InvalidFact(format!("Muse Spark component contract: {detail}"))
}

fn anyhow_refusal(error: anyhow::Error) -> Refusal { invalid(&format!("{error:#}")) }

/// The largest coordinator KV pool this deployment admits, from the one shared
/// pool policy. Both edges call this before either solves.
pub fn maximum_pool_tokens(capacities: &[u64], context: u64, requested: Option<u64>) -> u64 {
    PoolPolicy::resolve(capacities, context, requested, PAGE_ROWS, false).maximum_tokens()
}

/// One coordinator GPU's planned budget: the CUDA context is the forecast
/// charge; everything else the family states as an owner.
pub fn planned_budget(gpu: u8, capacity_bytes: u64, headroom_bytes: u64, context_bytes: u64) -> Budget {
    Budget { id: format!("rtx{gpu}"), locations: vec![Location::coordinator(gpu)],
        physical_bytes: capacity_bytes, logical_bytes: capacity_bytes, headroom: headroom_bytes,
        baseline: Baseline::Forecast { occupied: context_bytes,
            source: "CUDA context and modules on the coordinator GPU".into(), assumed: false } }
}

/// A measured coordinator sample. `free` is clamped to the admitted capacity,
/// so an operator cap below the sampled free memory cannot invert the charge.
pub fn measured_budget(gpu: u8, capacity_bytes: u64, headroom_bytes: u64, free_bytes: u64) -> Budget {
    Budget { id: format!("rtx{gpu}"), locations: vec![Location::coordinator(gpu)],
        physical_bytes: capacity_bytes, logical_bytes: capacity_bytes, headroom: headroom_bytes,
        baseline: Baseline::Sampled { free: free_bytes.min(capacity_bytes), phase: Phase::Load,
            source: "coordinator free memory at admission".into() } }
}

/// One Spark worker's managed arena: the operator budget is both ceilings, and
/// its I/O and runtime reserve are owners inside it (as the runtime charges them).
pub fn spark_budget(rank: u8, bytes: u64) -> Budget {
    Budget { id: format!("spark{rank}"),
        locations: vec![Location { device: Device::Spark(rank), domain: Domain::Managed }],
        physical_bytes: bytes, logical_bytes: bytes, headroom: 0,
        baseline: Baseline::Forecast { occupied: 0,
            source: "Spark worker arena owned by the serving process".into(), assumed: false } }
}

/// The one request both edges submit; no edge re-derives a pin.
pub fn production_request(context: u64, pool: Option<u64>, onboard: Onboard, prefill: PrefillShape,
    operation: Operation, resident_weight_limit: Option<ResidentWeights>,
    spark_reply_formats: cuteafd_core::spark_reply::SparkReplyFormats) -> Request {
    Request { context,
        pool: PoolPolicy::normalize_request(pool).map_or(Pin::Auto, Pin::Exact),
        onboard: match onboard { Onboard::Auto => Pin::Auto, Onboard::Layers(n) => Pin::Exact(n as u64) },
        prefill: Pin::Exact(prefill), operation, resident_weight_limit, spark_reply_formats }
}

/// Everything Muse knows before a solve: geometry, budgets and header-exact
/// cost tables, identical on the planned and measured edges.
pub struct ProductionInput<'a> {
    pub cfg: &'a MuseSparkConfig,
    pub core: &'a CoreInventory,
    pub catalog: &'a crate::OfficialV41Catalog,
    /// One coordinator budget per RTX plus one managed budget per Spark rank.
    pub budgets: Vec<Budget>,
    pub prefill: PrefillPlan,
    pub full_prefill_logits: bool,
    /// Active sequence slots (each owns its sliding rings).
    pub slots: u64,
    /// Device mark slots from the shared prefix rule (the host tier holds retention).
    pub mark_slots: u64,
    /// The compiled context floor; `maximum_pool_tokens` is the extent.
    pub context: u64,
    pub maximum_pool_tokens: u64,
    pub partial_bf16_max_rows: usize,
    pub partial_reply_formats: cuteafd_core::spark_reply::SparkReplyFormats,
    /// The coordinator EXL3 package for onboarded layers.
    pub local: LocalPackage,
    /// The DFlash2 drafter, charged on the head GPU.
    pub drafter: Option<MuseDrafter>,
    /// Runtime-only code the image will load (the planner's full allowance,
    /// startup's not-yet-loaded remainder).
    pub fixed: Vec<Demand>,
    /// Bundled-tower headroom facts.
    pub media: MediaOptions,
}

/// The prepared recipe plus everything the caller reports from it.
pub struct PreparedMuse<'a> {
    pub recipe: MuseRecipe<'a>,
    budgets: Vec<Budget>,
    prefill_logits_bytes: usize,
    local: LocalPackage,
    drafter: Option<MuseDrafter>,
    layouts: Vec<Vec<LayerLayout>>,
    global: GlobalLayout,
    projections: Vec<crate::families::muse_spark::weight_layout::RankProjections>,
    spark_budget: SparkBudgetInventory,
}

/// The one admitted answer. Both edges read the same fields.
pub struct AdmittedMuse {
    pub placement: Placement,
    pub spark: SparkWorkingSet,
    pub solved: ResolvedPlan,
    pub media: MediaAdmission,
    pub local: LocalPackage,
    pub drafter: Option<MuseDrafter>,
    pub prefill_logits_bytes: usize,
    /// Per layer, per coordinator rank: the operands the device packs.
    pub layouts: Vec<Vec<LayerLayout>>,
    pub global: GlobalLayout,
    /// Per rank: the K6 programs it binds and their shared scratch.
    pub projections: Vec<crate::families::muse_spark::weight_layout::RankProjections>,
    /// Header-exact per-rank Spark arena slices for the selected onboarding.
    pub spark_budget: SparkBudgetInventory,
}

impl AdmittedMuse {
    /// Bytes of one rank's layer weights across all layers.
    pub fn layer_bytes(&self, gpu: usize) -> anyhow::Result<u64> {
        self.layouts.iter().try_fold(0u64, |sum, ranks| {
            let bytes = ranks.get(gpu).context("Muse layout rank")?.bytes()? as u64;
            sum.checked_add(bytes).context("Muse layer bytes overflow")
        })
    }
}

/// The family's immutable cost terms. Every component byte figure is fixed;
/// the pool charge is the paged full-layer KV slope the contract prices per page.
pub struct MuseRecipe<'a> {
    pub cfg: &'a MuseSparkConfig,
    pub context: u64,
    pub gpus: usize,
    pub maximum_pages: u64,
    pub prefill: PrefillShape,
    pub attention: Vec<LayerMode>,
    pub spark_ranks: usize,
    /// Joint bundled-tower options, in preference order (`Off` last for `Auto`).
    pub encoders: Vec<(EncoderPlacement, EncoderPlacement)>,
    pub modes: (MediaMode, MediaMode),
    pub cases: Vec<Case<Vec<Component>>>,
    /// Expert arena peak per onboarding count, per GPU.
    pub peaks: Vec<[u64; 2]>,
    pub local: LocalPackage,
}

/// One allocation, dropped when it charges nothing.
fn buffer(id: String, location: Location, bytes: u64, lifetime: Lifetime, category: Category, group: &str,
    maximum: u64) -> Option<Component> {
    (bytes > 0).then(|| Component::allocation(
        Allocation::buffer(id, location, bytes, lifetime, category, group),
        ByteCurve::fixed(bytes, maximum), false, maximum))
}

/// Retained weights: their allocation plus the residency fact, whether or not
/// the allocation is still in the ledger.
fn weights(id: String, location: Location, bytes: u64, resident: u64, category: Category, group: &str,
    maximum: u64) -> Result<Vec<Component>, Refusal> {
    if resident > bytes { return Err(invalid("resident weights exceed their backing allocation")); }
    let mut out = buffer(id.clone(), location, bytes, Lifetime::RESIDENT, category, group, maximum)
        .into_iter().collect::<Vec<_>>();
    if resident > 0 {
        out.push(Component::resident_weight(ResidentWeight { id, location, weights: ResidentWeights(resident) },
            ByteCurve::fixed(resident, maximum)));
    }
    Ok(out)
}

/// The joint preference order of the two bundled towers: every Spark replica
/// set (the requested lead first), then the RTX owner, then — for automatic
/// media only — disabled last. No free-memory screen: the one solve decides
/// capacity, so a plan and a measured startup build the same order.
fn tower_options(media: &MediaOptions, gpus: usize, sparks: usize)
    -> Result<Vec<(EncoderPlacement, EncoderPlacement)>, Refusal> {
    use crate::plan::encoder::{candidates, encoder_placement, EncoderGpuBudget, EncoderHardware,
        EncoderRequest, EncoderSparkBudget};
    let hardware = EncoderHardware { v41: false,
        gpus: (0..gpus).map(|_| EncoderGpuBudget { free_bytes: u64::MAX, kv_target_bytes: 0 }).collect(),
        sparks: (0..sparks).map(|rank| EncoderSparkBudget { rank, host: format!("spark{rank}"), idle: false,
            expert_bytes: 0, free_bytes: u64::MAX }).collect() };
    let tower = |name: &str, mode: MediaMode, bytes: (u64, u64), copies: usize| -> Result<Vec<EncoderPlacement>, Refusal> {
        let mut list = candidates(&EncoderRequest { mode, weights: bytes.0, scratch: bytes.1, copies }, &hardware);
        if mode == MediaMode::Auto && bytes.0 > 0 {
            // Auto always closes its domain with Off; an enabled tower the
            // solver rejects is then reported by the shared capacity
            // diagnosis, while a selected Off shows as disabled media bytes.
            list.push(encoder_placement(MediaMode::Off, &hardware, 0, 0, 1));
        }
        if list.is_empty() {
            return Err(invalid(&format!("requested {name} tower ({mode:?}) has no executable device")));
        }
        Ok(list)
    };
    let vision = tower("vision", media.vision, media.vision_bytes, media.replicas)?;
    let audio = tower("audio", media.audio, media.audio_bytes, 1)?;
    Ok(vision.iter().flat_map(|vision| audio.iter().map(move |audio| (vision.clone(), audio.clone())))
        .collect())
}

impl<'a> ProductionInput<'a> {
    pub fn prepare(self) -> Result<PreparedMuse<'a>, Refusal> {
        let ProductionInput { cfg, core, catalog, budgets, prefill, full_prefill_logits, slots, mark_slots,
            context, maximum_pool_tokens, partial_bf16_max_rows, partial_reply_formats, local, drafter, fixed,
            media } = self;
        let coordinator = budgets.iter().filter(|budget| budget.locations.iter()
            .any(|location| matches!(location.device, Device::Coordinator(_)))).count();
        let spark_ranks = budgets.iter().filter(|budget| budget.locations.iter()
            .any(|location| matches!(location.device, Device::Spark(_)))).count();
        if !(1..=2).contains(&coordinator) || !(1..=8).contains(&spark_ranks) {
            return Err(invalid("Muse Spark runs on one or two RTX with one to eight Spark ranks"));
        }
        if maximum_pool_tokens < context {
            return Err(Refusal::PoolBelowContext { tokens: maximum_pool_tokens, context });
        }
        if slots == 0 || context == 0 || context > cfg.max_position as u64 {
            return Err(invalid(&format!("slots {slots} / context {context} outside 1..={}", cfg.max_position)));
        }
        if !catalog.per_expert_output_norm() {
            return Err(invalid("Spark experts must carry their post-expert norm"));
        }
        let gpus = coordinator;
        let pair = gpus == 2;
        let (_, modes) = executor(gpus).map_err(anyhow_refusal)?;
        let attention = vec![modes[0]; cfg.layers];
        let layouts = (0..cfg.layers).map(|layer| LayerLayout::ranks(cfg, layer, gpus, core))
            .collect::<anyhow::Result<Vec<_>>>().map_err(anyhow_refusal)?;
        let global = GlobalLayout::new(cfg, core).map_err(anyhow_refusal)?;
        let maximum_pages = maximum_pool_tokens.div_ceil(PAGE_ROWS);
        let slots_usize = usize::try_from(slots).map_err(|_| Refusal::Overflow("Muse slots"))?;
        let unit_rows = usize::try_from(prefill.unit_rows).map_err(|_| Refusal::Overflow("Muse prefill rows"))?;
        let waves = usize::try_from(prefill.waves).map_err(|_| Refusal::Overflow("Muse prefill waves"))?;
        let record = cfg.kv_record_bytes() as u64;
        let rings = cfg.sliding_layers() as u64 * ring_rows(cfg).map_err(anyhow_refusal)? * record;
        let marks = mark_bytes(cfg, 1);
        let lane = crate::families::muse_spark::lane::LanePlanes::with_waves(cfg, gpus, unit_rows, waves)
            .and_then(|planes| planes.bytes()).map_err(anyhow_refusal)? as u64;
        let tables = step_tables(unit_rows, slots_usize, context, waves).map_err(anyhow_refusal)?.bytes()
            as u64;
        let projections = crate::families::muse_spark::weight_layout::projections(cfg, core, gpus, unit_rows)
            .map_err(anyhow_refusal)?;
        let exchange = if pair { Some(tp2_exchange_bytes(cfg, unit_rows).map_err(anyhow_refusal)? as u64) }
            else { None };
        let combine = combine_tables_bytes(unit_rows) as u64 * prefill.waves;
        let rope = rope_tables(cfg, context).map_err(anyhow_refusal)?.allocation_bytes;
        let (stored, trimmed) = crate::families::muse_spark::lane::head_planes(
            global.head.as_ref().map_or(cfg.vocab_size, |head| head.shape[0]), cfg.vocab_size)
            .map_err(anyhow_refusal)?;
        let prefill_logits_bytes = if full_prefill_logits {
            crate::families::muse_spark::lane::prefill_logits_bytes(
                usize::try_from(prefill.span_rows).map_err(|_| Refusal::Overflow("Muse span rows"))?,
                cfg.vocab_size).map_err(anyhow_refusal)?
        } else { 0 };
        // The transport cutoff the runtime seals: the reservation assumes one
        // verify row without a drafter, as the planner has always sized it.
        let cutoff = crate::families::muse_spark::replies::Cutoff::with_formats(partial_bf16_max_rows, 1,
            partial_reply_formats).map_err(anyhow_refusal)?;
        let (_, planes) = intake(cfg, prefill, cutoff, slots_usize, spark_ranks).map_err(anyhow_refusal)?;
        let head_gpu = u8::try_from(gpus - 1).map_err(|_| Refusal::Overflow("Muse GPUs"))?;

        // Fixed owners: the list the request builder charged, now with real
        // lifetimes and one uniform charge path.
        let mut components = Vec::new();
        for gpu in 0..gpus {
            let location = Location::coordinator(gpu as u8);
            let layer = layouts.iter().try_fold(0u64, |sum, ranks| {
                sum.checked_add(ranks[gpu].bytes()? as u64).context("Muse layer bytes overflow")
            }).map_err(anyhow_refusal)?;
            components.extend(weights(format!("muse.layer_weights.{gpu}"), location, layer, layer,
                Category::Weights,
                "layer weights (head-split attention; MoE projections, router, norms, residual gates whole)",
                maximum_pages)?);
            components.extend(weights(format!("muse.final_norm.{gpu}"), location, global.final_norm as u64,
                global.final_norm as u64, Category::Weights, "final norm", maximum_pages)?);
            components.extend(buffer(format!("muse.rings.{gpu}"), location, rings * slots / gpus as u64,
                Lifetime::RESIDENT, Category::Kv,
                &format!("sliding rings, {slots} sequences x {} layers", cfg.sliding_layers()), maximum_pages));
            if mark_slots > 0 {
                components.extend(buffer(format!("muse.marks.{gpu}"), location, marks * mark_slots / gpus as u64,
                    Lifetime::RESIDENT, Category::Prefix,
                    &format!("Muse exact sliding marks, {mark_slots} slots"), maximum_pages));
            }
            components.extend(buffer(format!("muse.lane_planes.{gpu}"), location, lane, Lifetime::RESIDENT,
                Category::Workspace, "Muse lane planes", maximum_pages));
            components.extend(buffer(format!("muse.step_tables.{gpu}"), location, tables, Lifetime::RESIDENT,
                Category::Workspace, "Muse step metadata", maximum_pages));
            if let Some(scratch) = &projections[gpu].scratch {
                let bytes = scratch.bytes().map_err(anyhow_refusal)? as u64;
                components.extend(buffer(format!("muse.projections.{gpu}"), location, bytes,
                    Lifetime::RESIDENT, Category::Workspace, "Muse K6 projection scratch", maximum_pages));
            }
            if let Some(exchange) = exchange {
                components.extend(buffer(format!("muse.tp2_exchange.{gpu}"), location, exchange,
                    Lifetime::RESIDENT, Category::Transport, "Muse TP2 exchange slots", maximum_pages));
            }
            components.extend(buffer(format!("muse.rope.{gpu}"), location, rope, Lifetime::RESIDENT,
                Category::Workspace, "Muse RoPE tables", maximum_pages));
            // Full layers page BF16 K+V records: the pool's only slope, split
            // across a head-split pair like every other per-layer owner.
            let slope = record.checked_mul(PAGE_ROWS).ok_or(Refusal::Overflow("Muse full-layer KV"))?
                / gpus as u64;
            for layer in 0..cfg.layers {
                if cfg.attention[layer] != MuseAttention::Full { continue; }
                components.push(Component::allocation(
                    Allocation::buffer(format!("muse.kv.{gpu}.{layer}"), location, slope, Lifetime::TARGET,
                        Category::Kv, "full-layer KV (paged K+V records)"),
                    ByteCurve::linear(slope, slope, maximum_pages), false, maximum_pages));
            }
        }
        components.extend(buffer("muse.combine".into(), Location::coordinator(0), combine, Lifetime::RESIDENT,
            Category::Workspace, "Muse Spark combine tables", maximum_pages));
        components.extend(weights("muse.embedding".into(), Location::coordinator(0),
            global.embedding as u64, global.embedding as u64, Category::Embedding, "token embedding",
            maximum_pages)?);
        components.extend(buffer("muse.head_logits".into(), Location::coordinator(head_gpu),
            (stored + trimmed) as u64, Lifetime::RESIDENT, Category::Workspace, "Muse head logits",
            maximum_pages));
        components.extend(buffer("muse.prefill_logits".into(), Location::coordinator(head_gpu),
            prefill_logits_bytes as u64, Lifetime::RESIDENT, Category::Workspace, "Muse prefill logits",
            maximum_pages));
        components.extend(weights("muse.head".into(), Location::coordinator(head_gpu),
            global.head_bytes() as u64, global.head_bytes() as u64, Category::Weights, "lm head (K6)",
            maximum_pages)?);
        components.extend(buffer("muse.partial_planes".into(), Location::coordinator(0), planes,
            Lifetime::RESIDENT, Category::Transport, "Muse Spark partial reply planes", maximum_pages));
        if let Some(drafter) = &drafter {
            let bytes = [drafter.owned, drafter.scratch, drafter.head_scratch, drafter.runtime].into_iter()
                .try_fold(0u64, |sum, bytes| sum.checked_add(bytes)).ok_or(Refusal::Overflow("Muse drafter"))?;
            components.extend(weights("muse.drafter".into(), Location::coordinator(head_gpu), bytes,
                drafter.weights.0, Category::Drafter, "Drafter", maximum_pages)?);
        }
        for (index, demand) in fixed.iter().enumerate() {
            if demand.bytes == 0 { continue; }
            components.extend(weights(format!("muse.fixed.{index}"), Location::coordinator(demand.gpu),
                demand.bytes, demand.resident_weights.0, demand.category, &demand.group, maximum_pages)?);
        }
        let mut cases = vec![Case { when: vec![], value: components }];

        // Expert arenas per onboarding count. A pair splits every routed layer
        // across both GPUs (TP2 halves plus the partial combine); one RTX holds
        // the whole layer. The executor workspace is charged exactly when a
        // layer is, and the Spark side keeps the intermediate slices of every
        // layer not onboarded.
        let manifest = catalog.exl3().ok_or_else(|| invalid("routed experts need EXL3 headers"))?;
        let tp2_partial = tp2_partial_bytes(cfg, unit_rows) as u64;
        let base = if pair { [tp2_partial + local.workspace(); 2] }
            else { [tp2_partial / 2 + local.workspace(), 0] };
        let arena = |layer: usize, world: usize, rank: usize| -> Result<u64, Refusal> {
            let residency = manifest.residency(crate::V41Exl3Layer::Backbone(layer), world, rank)
                .map_err(anyhow_refusal)?;
            let (_, bytes) = residency.device_arena_layout()
                .map_err(|error| invalid(&format!("Muse expert arena: {error}")))?;
            Ok(bytes as u64)
        };
        let working = SparkWorkingSet { rtx_layers: vec![], spark_layers: (0..cfg.layers).collect(),
            spark_ranks, prefill_lanes: LANES };
        let (endpoint, spark_rings) = spark_endpoint(prefill, cutoff, slots_usize).map_err(anyhow_refusal)?;
        let spark_bytes = budgets.iter().find(|budget| budget.locations.iter()
            .any(|location| matches!(location.device, Device::Spark(0))))
            .map_or(0, |budget| budget.physical_bytes);
        let spark_budget = SparkBudgetInventory::exl3(catalog, &working, &[endpoint], spark_rings, spark_bytes,
            spark_reserve_bytes(spark_ranks, prefill.waves, prefill.unit_rows).map_err(anyhow_refusal)?)
            .map_err(anyhow_refusal)?;
        let mut peaks = vec![[0u64; 2]; cfg.layers + 1];
        let mut expert_cases = Vec::with_capacity(cfg.layers + 1);
        // Count zero is a case too: with every routed layer on the Sparks, the
        // slice ledger is the whole Spark charge and must bind their budgets.
        for count in 0..=cfg.layers {
            let mut sum = [0u64; 2];
            for layer in 0..count {
                for gpu in 0..gpus {
                    let bytes = if pair { arena(layer, 2, gpu)? } else { arena(layer, 1, 0)? };
                    sum[gpu] = sum[gpu].checked_add(bytes).ok_or(Refusal::Overflow("Muse expert arena"))?;
                }
            }
            let mut components = Vec::new();
            for gpu in 0..gpus {
                // The executor workspace opens with the first onboarded layer:
                // a zero-layer case carries no coordinator expert demand.
                let bytes = if count == 0 { 0 }
                    else { base[gpu].checked_add(sum[gpu]).ok_or(Refusal::Overflow("Muse expert arena"))? };
                peaks[count][gpu] = bytes;
                let group = if pair { format!("TP2 expert layer halves 0..{count}") }
                    else { format!("resident routed layers 0..{count}") };
                components.extend(weights(format!("muse.experts.{gpu}.{count}"),
                    Location::coordinator(gpu as u8), bytes, sum[gpu], Category::Experts, &group,
                    maximum_pages)?);
            }
            for rank in 0..spark_ranks {
                let location = Location { device: Device::Spark(rank as u8), domain: Domain::Managed };
                if let Some(costs) = spark_budget.resident.get(rank).and_then(|costs| costs.get(count..)) {
                    components.extend(buffer(format!("muse.spark.experts.{rank}.{count}"), location,
                        costs.iter().sum::<u64>(), Lifetime::RESIDENT, Category::Experts,
                        "routed experts (intermediate slices)", maximum_pages));
                }
                if count < cfg.layers {
                    components.extend(buffer(format!("muse.spark.io.{rank}.{count}"), location,
                        spark_budget.overhead.get(rank).copied().unwrap_or(0), Lifetime::RESIDENT,
                        Category::Runtime, "Spark I/O slots and runtime reserve", maximum_pages));
                }
            }
            expert_cases.push(Case { when: vec![When::Onboard(count)], value: components });
        }

        // Bundled towers: one joint option per (vision, audio) candidate pair.
        let encoders = tower_options(&media, gpus, spark_ranks)?;
        let mut encoder_cases = Vec::with_capacity(encoders.len());
        for (option, (vision, audio)) in encoders.iter().enumerate() {
            let mut components = Vec::new();
            for (name, placement) in [("vision", vision), ("audio", audio)] {
                let locations: Vec<Location> = match placement.kind {
                    crate::plan::encoder::EncoderKind::Off => Vec::new(),
                    crate::plan::encoder::EncoderKind::Rtx { gpu } => vec![Location::coordinator(gpu as u8)],
                    _ => placement.spark_ranks().into_iter()
                        .map(|rank| Location { device: Device::Spark(rank as u8), domain: Domain::Managed })
                        .collect(),
                };
                for (copy, location) in locations.into_iter().enumerate() {
                    // Residency is a coordinator ceiling; Spark towers are
                    // charged only inside their worker's managed arena.
                    let resident = if matches!(location.device, Device::Coordinator(_)) {
                        placement.weights
                    } else { 0 };
                    components.extend(weights(format!("muse.{name}.weights.{option}.{copy}"), location,
                        placement.weights, resident, Category::Weights, &format!("{name} tower"),
                        maximum_pages)?);
                    components.extend(buffer(format!("muse.{name}.scratch.{option}.{copy}"), location,
                        placement.scratch, Lifetime::RESIDENT, Category::Workspace,
                        &format!("{name} scratch"), maximum_pages));
                }
            }
            encoder_cases.push(Case { when: vec![When::Encoder(option)], value: components });
        }
        cases.extend(expert_cases);
        cases.extend(encoder_cases);
        let recipe = MuseRecipe { cfg, context, gpus, maximum_pages,
            prefill: PrefillShape { rows: prefill.unit_rows, waves: prefill.waves }, attention,
            spark_ranks, encoders, modes: (media.vision, media.audio), cases, peaks, local: local.clone() };
        Ok(PreparedMuse { recipe, budgets, prefill_logits_bytes, local, drafter, layouts, global, projections,
            spark_budget })
    }
}

struct Domains {
    onboard: usize,
    encoders: usize,
}

impl Domains {
    /// The activation of one component condition: every option satisfying all
    /// of its conditions, as a cartesian product of single-choice clauses.
    fn activation(&self, when: &[When], onboard: usize, encoder: usize) -> Result<Activation, Refusal> {
        let mut domains = std::collections::BTreeMap::<usize, std::collections::BTreeSet<usize>>::new();
        for condition in when {
            let (variable, options): (usize, Vec<usize>) = match condition {
                When::Onboard(count) => {
                    if *count > self.onboard {
                        return Err(invalid("onboarding condition above the routed layer count"));
                    }
                    (onboard, vec![*count])
                }
                When::Encoder(option) => {
                    if *option >= self.encoders { return Err(invalid("encoder condition outside the options")); }
                    (encoder, vec![*option])
                }
            };
            match domains.get_mut(&variable) {
                Some(existing) => {
                    existing.retain(|option| options.contains(option));
                    if existing.is_empty() { return Err(invalid("contradictory component conditions")); }
                }
                None => { domains.insert(variable, options.into_iter().collect()); }
            }
        }
        let mut clauses = vec![vec![]];
        for (variable, options) in domains {
            clauses = clauses.into_iter().flat_map(|clause| {
                options.iter().map(move |&option| {
                    let mut clause = clause.clone();
                    clause.push(Choice { variable, option });
                    clause
                })
            }).collect();
        }
        Ok(Activation { clauses })
    }
}

impl MuseRecipe<'_> {
    /// Place the selected assignment: layer modes, expert homes, residual hops
    /// and the Spark working set. Plan and runtime read this verbatim.
    pub fn execution(&self, solved: &ResolvedPlan) -> Result<(Placement, SparkWorkingSet), Refusal> {
        if solved.decisions.attention != self.attention {
            return Err(invalid("resolved layer modes differ from the contract"));
        }
        let local: Vec<usize> = solved.decisions.onboard_layers.iter().flatten().copied()
            .collect::<std::collections::BTreeSet<_>>().into_iter().collect();
        let count = local.len();
        if count > self.cfg.layers || local.iter().copied().ne(0..count) {
            return Err(invalid("resolved onboarding is not the declared ordered prefix"));
        }
        let peak = *self.peaks.get(count).ok_or_else(|| invalid("resolved count outside the peaks"))?;
        let pair = self.gpus == 2;
        let assignments = (0..self.cfg.layers).map(|layer| LayerAssignment {
            mode: self.attention[layer],
            experts: if layer >= count { ExpertHome::Spark }
                else if pair { ExpertHome::RtxTp2 } else { ExpertHome::RtxWhole { gpu: 0 } },
        }).collect();
        let tp2 = (!local.is_empty() && pair)
            .then(|| Tp2Range { first: local[0], layers: count, peak_bytes: peak });
        let ranges = (0..self.gpus).map(|gpu| {
            if tp2.is_some() || local.is_empty() { ExpertRange { first: 0, layers: 0, peak_bytes: 0 } }
            else { ExpertRange { first: local[0], layers: count, peak_bytes: peak[gpu] } }
        }).collect();
        let hop_spec = HopSpec { storage: HopStorage::ReceiveSlots,
            row_bytes: (self.cfg.hidden as u64).checked_mul(2).ok_or(Refusal::Overflow("Muse hop rows"))?,
            rows: 0, lanes: 0, entry_gpu: 0, head_gpu: u8::try_from(self.gpus - 1).map_err(|_| Refusal::Overflow("Muse GPUs"))?,
            terminal: if pair { TerminalHome::Replicated } else { TerminalHome::AttentionOwner } };
        let residual = residual_homes(&self.attention, &hop_spec)
            .map_err(|error| invalid(&error.to_string()))?;
        let hops = plan_hops(&self.attention, &hop_spec).map_err(|error| invalid(&error.to_string()))?;
        let placement = Placement { pool_tokens: solved.pool_tokens, pool_limits: PoolLimits(vec![]),
            attention_placement: AttentionPlacement::Heads,
            attention_by_kind: vec![(AttentionClass::Gqa, AttentionPlacement::Heads),
                (AttentionClass::Swa, AttentionPlacement::Heads)],
            peer_row_bytes: 0, onboard_layers: count, max_pool_tokens_no_onboard: None,
            layers: assignments, movables: vec![], expert_ranges: ranges, residual, hops, tp2,
            items: (0..self.gpus).map(|gpu| solved.items(Location::coordinator(gpu as u8)))
                .collect::<Result<_, _>>()? };
        MUSE_SPARK.check(&placement).map_err(|error| invalid(&error.to_string()))?;
        self.local.check(&placement).map_err(anyhow_refusal)?;
        let working = SparkWorkingSet { rtx_layers: local,
            spark_layers: (count..self.cfg.layers).collect(),
            spark_ranks: if count == self.cfg.layers { 0 } else { self.spark_ranks },
            prefill_lanes: LANES };
        Ok((placement, working))
    }

    /// The selected towers. An automatic tower the shared solution disabled
    /// carries the solution's own diagnostic, in the shared wording.
    fn media(&self, solved: &ResolvedPlan) -> Result<MediaAdmission, Refusal> {
        let option = solved.decisions.encoder.ok_or_else(|| invalid("missing admitted encoder"))?;
        let (vision, audio) = self.encoders.get(option)
            .ok_or_else(|| invalid("admitted encoder outside the prepared options"))?.clone();
        let mut spark_items = vec![Vec::new(); self.spark_ranks];
        for (name, placement) in [("vision", &vision), ("audio", &audio)] {
            let items = [Item::new(Category::Weights, format!("{name} tower"), "BF16 native",
                placement.weights, Basis::Exact),
                Item::new(Category::Workspace, format!("{name} scratch"), "resident", placement.scratch,
                    Basis::Exact)];
            for rank in placement.spark_ranks() {
                let rank_items = spark_items.get_mut(rank)
                    .ok_or_else(|| invalid("tower rank outside the Spark worker set"))?;
                rank_items.extend(items.iter().cloned());
            }
        }
        let disabled = |placement: &mut EncoderPlacement, mode: MediaMode| {
            use crate::plan::encoder::EncoderKind;
            if mode != MediaMode::Auto || placement.kind != EncoderKind::Off || placement.weights > 0 {
                return;
            }
            if placement.reason.is_empty() || placement.reason == "disabled or no checkpoint tower" { return; }
            placement.reason = format!("auto native tower disabled by shared admission: {}",
                solved.encoder_diagnostics.iter().map(|(reason, _)| reason.as_str())
                    .collect::<Vec<_>>().join("; "));
            placement.shortfall = solved.encoder_diagnostics.iter().map(|(_, bytes)| *bytes)
                .filter(|bytes| *bytes > 0).min().unwrap_or(0);
        };
        let mut vision = vision;
        let mut audio = audio;
        disabled(&mut vision, self.modes.0);
        disabled(&mut audio, self.modes.1);
        Ok(MediaAdmission { vision, audio, spark_items })
    }
}

impl Recipe for MuseRecipe<'_> {
    fn prepare(&self, request: &Request) -> Result<Contract, Refusal> {
        if let (LocalPackage::Missing(package), Pin::Exact(n)) = (&self.local, request.onboard) {
            if n > 0 {
                return Err(invalid(&format!("{}; an explicit {n} RTX expert layers was requested",
                    LocalPackage::refusal(package))));
            }
        }
        if !(1..=2).contains(&self.gpus)
            || self.cfg.layers == 0
            || self.context == 0
            || request.context != self.context
            || self.maximum_pages == 0
            || self.maximum_pages == u64::MAX
            || self.prefill.rows == 0
            || self.prefill.waves == 0
            || self.attention.len() != self.cfg.layers
            || !(1..=8).contains(&self.spark_ranks)
            || self.encoders.is_empty()
            || self.peaks.len() != self.cfg.layers + 1
        {
            return Err(invalid("invalid extent or empty domain"));
        }
        if self.attention.iter().any(|mode| match mode {
            LayerMode::HeadSplit => self.gpus != 2,
            LayerMode::Whole { gpu, .. } => usize::from(*gpu) >= self.gpus,
            _ => true,
        }) {
            return Err(invalid("layer mode outside the Muse executor"));
        }
        let mut contract = Contract {
            resident_weight_limit: request.resident_weight_limit,
            phase_order: PhaseOrder::TARGET_THEN_DRAFT,
            spark_reply_formats: Some(SparkReplyFormats {
                prefill: Some(request.spark_reply_formats.prefill(
                    cuteafd_core::ExpertGeometry::MUSE_SPARK.partial_replies())),
                decode: Some(request.spark_reply_formats.decode()),
            }),
            page_rows: PAGE_ROWS,
            maximum_pages: self.maximum_pages,
            gpus: self.gpus,
            variables: self.attention.iter().enumerate()
                .map(|(layer, &mode)| Variable::Attention { layer, values: vec![mode] })
                .collect(),
            requires: vec![],
            components: vec![],
        };
        contract.variables.push(Variable::Prefill(vec![self.prefill]));
        contract.variables.push(Variable::Lanes(vec![LANES]));
        let onboard = contract.variables.len();
        contract.variables.push(Variable::Onboard {
            layers: (0..self.cfg.layers).collect(),
            destination: (0..self.gpus as u8).collect(),
            counts: (0..=self.cfg.layers).collect(),
        });
        let encoder = contract.variables.len();
        contract.variables.push(Variable::Encoder((0..self.encoders.len()).collect()));
        // A Spark tower keeps a retained expert worker: a full onboarding has
        // no Spark layer left to run it.
        for (option, (vision, audio)) in self.encoders.iter().enumerate() {
            if vision.spark_ranks().is_empty() && audio.spark_ranks().is_empty() { continue; }
            contract.requires.push(Requires { when: Choice { variable: encoder, option },
                any_of: (0..self.cfg.layers).map(|count| Choice { variable: onboard, option: count }).collect() });
        }
        let domains = Domains { onboard: self.cfg.layers, encoders: self.encoders.len() };
        for case in &self.cases {
            let activation = domains.activation(&case.when, onboard, encoder)?;
            for mut component in case.value.iter().cloned() {
                if component.activation != Activation::always() {
                    return Err(invalid("subrecipe supplied a private activation"));
                }
                component.activation = activation.clone();
                contract.components.push(component);
            }
        }
        Ok(contract)
    }
}

impl PreparedMuse<'_> {
    /// The single admission solve. Planner and serving call this one verb.
    pub fn admit(self, request: Request) -> Result<AdmittedMuse, Refusal> {
        let PreparedMuse { recipe, budgets, prefill_logits_bytes, local, drafter, layouts, global, projections,
            spark_budget } = self;
        let solved = solve(request, budgets, &recipe)?;
        let (placement, spark) = recipe.execution(&solved)?;
        let media = recipe.media(&solved)?;
        Ok(AdmittedMuse { placement, spark, solved, media, local, drafter, prefill_logits_bytes, layouts,
            global, projections, spark_budget })
    }
}
