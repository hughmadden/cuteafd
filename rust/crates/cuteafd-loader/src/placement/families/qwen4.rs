//! Qwen's shared planner facts. Checkpoint weights, the mapped PLE table and
//! the resident expert arenas are already in the admission baseline; every
//! owner after that is declared once here and admitted by one typed request.
//! The startup graph set is a single flat reserve priced at the widest compiled
//! pool, so admission needs one solve instead of a second pass over the pool.
use crate::families::qwen4::Qwen4Attention;
use crate::placement::planner::{
    solve, Activation, Allocation, Allowance, Baseline as PlanBaseline, Budget, ByteCurve, Component,
    ComponentKind, Contract, Lifetime as PlanLifetime, Location, Operation, Phase, PhaseOrder, Pin,
    PrefillShape as PlanPrefill, Recipe, Refusal, Request, ResidentWeight, ResolvedPlan, Variable,
};
use crate::placement::*;
use crate::serving_capacity::qwen_graphs::{
    qwen_admission, qwen_startup_graphs, qwen_startup_graphs_placed, QwenAdmission, QwenAdmissionInputs,
    QWEN_DECODE_ROWS, QWEN_UNIT_ROWS,
};
use crate::serving_capacity::Exl3Epilogue;
use cuteafd_core::memory_layout::{Basis, Category};
use cuteafd_core::spark_reply::{SparkReplyFormat, SparkReplyFormats};
use std::collections::BTreeMap;

pub mod encoder;
pub mod host;
pub use encoder::QwenEncoders;
pub use host::QwenHost;

pub const DEFAULT_MARK_SLOTS: u64 = 18;
pub const LAZY_GRAPH_BYTES: u64 = (1 << 30) / 2;
pub const SPARK_WORKSPACE_BYTES: u64 = 56 * (1 << 30) / 100;

/// A dual Qwen executor requires nonempty whole-width ownership on both ends.
/// Do not silently turn the shared solver's legal all-on-one-rank cut into a
/// private cut heuristic or a partly idle dual executor.
pub fn check_dual_layer_owners(owners: &[usize]) -> anyhow::Result<()> {
    anyhow::ensure!(owners.first() == Some(&0) && owners.last() == Some(&1)
        && owners.iter().all(|&owner| owner <= 1)
        && owners.windows(2).filter(|pair| pair[0] != pair[1]).count() == 1,
        "Qwen dual attention needs nonempty owners and one contiguous owner-zero to owner-one cutover");
    Ok(())
}

pub struct QwenInputs<'a> {
    pub resident_weight_limit: Option<ResidentWeights>,
    /// Checkpoint owners already charged to the physical baseline.
    pub loaded_weights: ResidentWeights,
    pub spark_reply_format: cuteafd_core::SparkReplyFormat,
    /// True only when the operator left context automatic.
    pub automatic_context: bool,
    pub admission: QwenAdmissionInputs<'a>,
    pub capacity_bytes: u64,
    pub baseline: Baseline,
    pub pending_code_bytes: u64,
    pub max_context: u64,
    pub requested_pool: Option<u64>,
    pub spark_ranks: usize,
    /// None means lazy diagnostic graphs, not the serving startup set.
    pub startup_graph_modes: Option<(usize, bool)>,
    /// Coordinator host owners and the host sample they are admitted against.
    pub host: QwenHost,
    /// Bundled vision tower candidates, chosen by the solve.
    pub encoders: QwenEncoders,
}

const WHOLE_OWNER_0: LayerMode = LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner };

/// The single-owner pool policy: CLI zero and an omitted pool select the same
/// automatic admission, and an automatic Spark-free layout keeps the agentic
/// context floor.
pub fn single_policy(inputs: &QwenInputs<'_>) -> PoolPolicy {
    PoolPolicy::resolve(&[inputs.capacity_bytes], inputs.max_context, inputs.requested_pool,
        QWEN_UNIT_ROWS as u64, inputs.spark_ranks == 0)
        .with_automatic_context(inputs.automatic_context)
}

/// Serving context and the finite compiled pool extent, both derived from the
/// same policy. The legacy ceiling was unbounded; a contract curve needs a
/// finite domain, so the extent is the larger of the automatic target and the
/// explicit request. Every pool the policy accepts is inside it.
fn single_extent(policy: &PoolPolicy) -> Result<(u64, u64), Refusal> {
    let page = QWEN_UNIT_ROWS as u64;
    let tokens = policy.maximum_tokens().max(policy.floor);
    let pages = tokens.div_ceil(page);
    if policy.floor == 0 || tokens == 0 || pages == 0 || pages == u64::MAX {
        return Err(Refusal::InvalidFact("invalid Qwen single-owner pool extent".into()));
    }
    Ok((policy.floor, pages))
}

/// One normalization for both routes: an omitted or zero pool is automatic and
/// an explicit pool rounds up to the allocation unit, exactly as the former
/// unit rounding did.
fn pool_pin(policy: &PoolPolicy, page: u64) -> Result<Pin<u64>, Refusal> {
    Ok(match policy.requested {
        None => Pin::Auto,
        Some(tokens) => Pin::Exact(tokens.div_ceil(page).checked_mul(page)
            .ok_or(Refusal::Overflow("Qwen pool request"))?),
    })
}

fn prefill_of(admission: &QwenAdmissionInputs<'_>) -> PlanPrefill {
    match admission.prefill_shape {
        Some(shape) => PlanPrefill { rows: shape.unit_rows.max(1), waves: shape.waves.max(1) },
        None => PlanPrefill { rows: admission.prefill_rows.max(64), waves: 1 },
    }
}

fn reply_formats(format: SparkReplyFormat, spark_ranks: usize) -> SparkReplyFormats {
    if spark_ranks == 0 { SparkReplyFormats::default() }
    else { SparkReplyFormats { prefill: Some(format), decode: Some(format) } }
}

/// The legacy budget is one physical capacity with a sampled or planned
/// baseline; the contract names the same witness, so both routes admit the same
/// bytes without subtracting the owners twice.
fn plan_baseline(baseline: &Baseline, physical_bytes: u64, source: &str) -> Result<PlanBaseline, Refusal> {
    Ok(match baseline {
        Baseline::Planned { context_bytes, loaded_bytes } => PlanBaseline::Forecast {
            occupied: context_bytes.checked_add(*loaded_bytes)
                .ok_or(Refusal::Overflow("Qwen planned baseline"))?,
            source: format!("{source}: plan-time owners already on the device; the recipe charges every later owner"),
            assumed: false },
        // The legacy sample is a free-byte witness and the contract names the
        // same witness, so both routes admit the same bytes without a second
        // subtraction.
        Baseline::Measured { free_bytes } => {
            if *free_bytes > physical_bytes {
                return Err(Refusal::InvalidFact(
                    format!("{source}: free sample exceeds physical capacity")));
            }
            PlanBaseline::Sampled {
                free: *free_bytes,
                phase: Phase::Load,
                source: format!("{source}: free memory sampled before admission; the recipe charges every later owner") }
        }
    })
}

/// One fixed byte owner. Zero-byte owners are dropped mechanically by the
/// contract, so callers may declare every optional demand.
fn owner(id: impl Into<String>, location: Location, bytes: u64, category: Category, group: &str,
    lifetime: PlanLifetime, maximum: u64) -> Component {
    Component::allocation(Allocation::buffer(id, location, bytes, lifetime, category, group),
        ByteCurve::fixed(bytes, maximum), false, maximum)
}

/// An owner whose extent is exactly `per_unit` bytes per 256-token pool unit.
fn pooled(id: impl Into<String>, location: Location, per_unit: u64, category: Category, group: &str,
    maximum: u64) -> Result<Component, Refusal> {
    let curve = ByteCurve::linear(per_unit, per_unit, maximum);
    let mut component = Component::allocation(Allocation::buffer(id, location, per_unit,
        PlanLifetime::TARGET, category, group), curve.clone(), false, maximum);
    component.byte_extent(curve)?;
    Ok(component)
}

/// Already-loaded weights: declared once per owner, limit-checked, never
/// capacity-charged because they are part of the admission baseline.
fn weights(id: impl Into<String>, location: Location, weights: ResidentWeights,
    maximum: u64) -> Component {
    Component::resident_weight(ResidentWeight { id: id.into(), location, weights },
        ByteCurve::fixed(weights.0, maximum))
}

/// A calibrated runtime allowance, not an allocation the family resizes.
fn allowance(source: &str, location: Location, bytes: u64, lifetime: PlanLifetime,
    maximum: u64) -> Component {
    Component { activation: Activation::always(), bytes: ByteCurve::fixed(bytes, maximum),
        kind: ComponentKind::Allowance(Allowance { location, bytes, lifetime,
            source: source.to_string(), assumed: false }) }
}

/// The single-owner route prepared once: admission bytes, the outer context and
/// the finite pool extent, and the graph reserve at the widest compiled pool.
pub struct QwenSingle<'a> {
    inputs: &'a QwenInputs<'a>,
    admission: QwenAdmission,
    context: u64,
    maximum_pages: u64,
    graphs: Option<GraphSet>,
}

impl<'a> QwenSingle<'a> {
    pub fn new(inputs: &'a QwenInputs<'a>) -> Result<Self, Refusal> {
        let admission = qwen_admission(&inputs.admission)
            .map_err(|error| Refusal::InvalidFact(format!("Qwen admission: {error}")))?;
        let policy = single_policy(inputs);
        let (context, maximum_pages) = single_extent(&policy)?;
        let page = QWEN_UNIT_ROWS as u64;
        let graphs = match inputs.startup_graph_modes {
            Some((sequences, speculation)) => {
                let tokens = maximum_pages.checked_mul(page)
                    .ok_or(Refusal::Overflow("Qwen pool extent"))?;
                Some(qwen_startup_graphs(context_of_extent(inputs.max_context)?,
                    tokens_of_extent(tokens)?, inputs.admission.cfg.dense_context(), sequences,
                    speculation, inputs.admission.layers)
                    .ok_or_else(|| Refusal::InvalidFact("Qwen graph count overflow".into()))?)
            }
            None => None,
        };
        Ok(Self { inputs, admission, context, maximum_pages, graphs })
    }

    pub fn request(&self) -> Result<Request, Refusal> {
        let inputs = self.inputs;
        let policy = single_policy(inputs);
        Ok(Request { context: self.context, pool: pool_pin(&policy, QWEN_UNIT_ROWS as u64)?,
            onboard: Pin::Auto, prefill: Pin::Exact(prefill_of(&inputs.admission)),
            operation: Operation::Serving, resident_weight_limit: inputs.resident_weight_limit,
            spark_reply_formats: reply_formats(inputs.spark_reply_format, inputs.spark_ranks) })
    }

    pub fn budget(&self) -> Result<Budget, Refusal> {
        Ok(Budget { id: "rtx0".into(), locations: vec![Location::coordinator(0)],
            physical_bytes: self.inputs.capacity_bytes, logical_bytes: self.inputs.capacity_bytes,
            baseline: plan_baseline(&self.inputs.baseline, self.inputs.capacity_bytes,
                "qwen4 single-owner")?,
            headroom: self.admission.headroom })
    }

    pub fn admit(&self) -> Result<ResolvedPlan, Refusal> {
        solve(self.request()?, vec![self.budget()?, self.inputs.host.budget()?], self)
    }

    /// Mechanical executor output: the admitted owners, not a second solve.
    pub fn execution(&self, solved: &ResolvedPlan) -> Result<Placement, Refusal> {
        use crate::placement::{AttentionClass, ExpertHome, ExpertRange, HopSpec, HopStorage,
            LayerAssignment, Placement, PoolLimits, TerminalHome, plan_hops, residual_homes};
        let inputs = self.inputs;
        let a = &inputs.admission;
        let modes = &solved.decisions.attention;
        if modes.len() != a.layers || modes.iter().any(|mode| *mode != WHOLE_OWNER_0) {
            return Err(Refusal::InvalidFact(
                "resolved Qwen modes differ from the prepared single owner".into()));
        }
        if solved.decisions.prefill != prefill_of(a) || solved.decisions.lanes != 1 {
            return Err(Refusal::InvalidFact(
                "resolved Qwen prefill differs from the prepared shape".into()));
        }
        let hop_spec = HopSpec { storage: HopStorage::ReceiveSlots,
            row_bytes: a.cfg.hc_count as u64 * a.cfg.hidden as u64 * 2,
            rows: a.prefill_rows.max(64), lanes: 1, entry_gpu: 0, head_gpu: 0,
            terminal: TerminalHome::AttentionOwner };
        let residual = residual_homes(modes, &hop_spec)
            .map_err(|error| Refusal::InvalidFact(error.to_string()))?;
        let hops = plan_hops(modes, &hop_spec)
            .map_err(|error| Refusal::InvalidFact(error.to_string()))?;
        let mut by_kind = BTreeMap::new();
        for &kind in &a.cfg.attention[..a.layers] {
            by_kind.insert(if kind == Qwen4Attention::Full { AttentionClass::Gqa }
                else { AttentionClass::Gdn }, AttentionPlacement::Heads);
        }
        Ok(Placement { pool_tokens: solved.pool_tokens, pool_limits: PoolLimits(vec![]),
            attention_placement: AttentionPlacement::Heads,
            attention_by_kind: by_kind.into_iter().collect(),
            peer_row_bytes: 0, onboard_layers: 0, max_pool_tokens_no_onboard: None,
            layers: modes.iter().map(|&mode| LayerAssignment { mode, experts: ExpertHome::Dense })
                .collect(),
            movables: Vec::new(),
            expert_ranges: vec![ExpertRange { first: 0, layers: 0, peak_bytes: 0 }],
            residual, hops, tp2: None,
            items: vec![solved.items(Location::coordinator(0))?] })
    }
}

fn context_of_extent(context: u64) -> Result<usize, Refusal> {
    usize::try_from(context).map_err(|_| Refusal::Overflow("Qwen context"))
}

fn tokens_of_extent(tokens: u64) -> Result<usize, Refusal> {
    usize::try_from(tokens).map_err(|_| Refusal::Overflow("Qwen pool extent"))
}

impl Recipe for QwenSingle<'_> {
    fn prepare(&self, request: &Request) -> Result<Contract, Refusal> {
        let inputs = self.inputs;
        let a = &inputs.admission;
        let cfg = a.cfg;
        let page = QWEN_UNIT_ROWS as u64;
        let gpu = Location::coordinator(0);
        if request.context != self.context || self.maximum_pages == 0
            || self.maximum_pages == u64::MAX || a.layers == 0 || a.layers > cfg.layers
        {
            return Err(Refusal::InvalidFact("invalid Qwen single-owner extent".into()));
        }
        let compute = |layers, mtp| crate::serving_capacity::qwen_cache_geometry(cfg, layers, mtp,
            a.kv_format).map_err(|error| Refusal::InvalidFact(format!("Qwen cache geometry: {error}")));
        let mut components = Vec::new();
        let mut previous = 0u64;
        for layer in 0..a.layers {
            let current = compute(layer + 1, false)?.ranks[0].persistent_unit_bytes;
            let unit = current.checked_sub(previous)
                .ok_or_else(|| Refusal::InvalidFact("Qwen per-layer records shrink".into()))?;
            previous = current;
            components.push(pooled(format!("qwen.records.{layer}"), gpu, unit, Category::Kv,
                "records", self.maximum_pages)?);
        }
        // Includes the optional MTP records, pool metadata, both page tables and
        // the per-token rounding of those table bytes.
        let unit_bytes = self.admission.per_token.checked_mul(page)
            .ok_or(Refusal::Overflow("Qwen pool unit"))?;
        let overhead = unit_bytes.checked_sub(previous)
            .ok_or_else(|| Refusal::InvalidFact("Qwen per-token pool total is below its records".into()))?;
        components.push(pooled("qwen.records.metadata", gpu, overhead, Category::Kv, "records",
            self.maximum_pages)?);
        for (index, &(category, group, bytes)) in self.admission.items.iter().enumerate() {
            components.push(owner(format!("qwen.admission.{index}"), gpu, bytes, category, group,
                PlanLifetime::RESIDENT, self.maximum_pages));
        }
        if inputs.pending_code_bytes > 0 {
            components.push(owner("qwen.loaded.code", gpu, inputs.pending_code_bytes,
                Category::Runtime, "loaded code", PlanLifetime::RESIDENT, self.maximum_pages));
        }
        if inputs.spark_ranks > 0 && matches!(inputs.baseline, Baseline::Planned { .. }) {
            // Runtime creates SparkLink before sampling free memory. Only the
            // header-only planner still reserves its return planes and codec.
            let bytes = crate::serving_capacity::prefill_return_bytes_with_format(
                a.prefill_shape.map_or(1, |shape| shape.waves),
                a.prefill_shape.map_or(a.prefill_rows.max(64), |shape| shape.unit_rows),
                inputs.spark_ranks as u64, cfg.hidden as u64, inputs.spark_reply_format)
                .map_err(|error| Refusal::InvalidFact(format!("Qwen Spark intake: {error:#}")))?;
            components.push(owner("qwen.spark.intake", gpu, bytes, Category::Transport,
                "Spark intake planes", PlanLifetime::RESIDENT, self.maximum_pages));
        }
        // One flat reserve at the widest compiled pool: the graph shape count is
        // monotone in the pool, so every admitted set is a subset of this one.
        let (graph_bytes, graph_source, graph_life) = match &self.graphs {
            Some(graphs) => (graphs.bytes(0), "startup graphs",
                PlanLifetime { first: Phase::Warmup, last: Phase::Growth }),
            None => (LAZY_GRAPH_BYTES, "graph growth",
                PlanLifetime { first: Phase::Ready, last: Phase::Growth }),
        };
        components.push(allowance(graph_source, gpu, graph_bytes, graph_life, self.maximum_pages));
        components.push(weights("qwen.loaded.owners", gpu, inputs.loaded_weights, self.maximum_pages));
        if inputs.host.uploads.len() != 1 {
            return Err(Refusal::InvalidFact("Qwen single-owner host owners must name one upload owner".into()));
        }
        components.extend(inputs.host.components(self.maximum_pages)?);
        let mut variables: Vec<_> = (0..a.layers)
            .map(|layer| Variable::Attention { layer, values: vec![WHOLE_OWNER_0] })
            .chain([Variable::Prefill(vec![prefill_of(a)]), Variable::Lanes(vec![1])])
            .collect();
        inputs.encoders.extend(&mut variables, &mut components, 1, self.maximum_pages)?;
        Ok(Contract { phase_order: PhaseOrder::TARGET_THEN_DRAFT,
            spark_reply_formats: (inputs.spark_ranks > 0)
                .then(|| reply_formats(inputs.spark_reply_format, inputs.spark_ranks)),
            resident_weight_limit: request.resident_weight_limit, page_rows: page,
            maximum_pages: self.maximum_pages, gpus: 1, variables, requires: vec![], components })
    }
}

/// Admit the single-owner route and publish the executor's placement.
pub fn placement(inputs: &QwenInputs<'_>) -> anyhow::Result<Placement> {
    Ok(admitted(inputs)?.0)
}

/// The placement with its solved plan (phase ledger and host owners).
pub fn admitted(inputs: &QwenInputs<'_>) -> anyhow::Result<(Placement, ResolvedPlan)> {
    if let Some((sequences, _)) = inputs.startup_graph_modes {
        cuteafd_core::concurrency::validate("qwen4", sequences as u64)?;
    }
    let recipe = QwenSingle::new(inputs)?;
    let solved = recipe.admit()?;
    Ok((recipe.execution(&solved)?, solved))
}

/// Checkpoint owners on the single target owner; the mapped PLE backing and
/// the target head borrowed by MTP are not separate device weight owners.
pub fn single_owner_weights(resident: &crate::families::qwen4::resident::Qwen4CheckpointResident,
    host_embedding: bool, ple: bool) -> Result<ResidentWeights, PlacementError> {
    [resident.target_bytes, resident.head_bytes, resident.mtp_bytes,
        if host_embedding { 0 } else { resident.embedding_bytes }, if ple { 256 } else { 0 }]
        .into_iter().try_fold(ResidentWeights::default(), |sum, bytes|
            sum.checked_add(ResidentWeights(bytes)))
}

/// Native TP1 arenas selected by the single-owner executor. Package scratch
/// and conversion staging are independent storage, not checkpoint weights.
pub fn single_expert_weights(catalog: &crate::OfficialV41Catalog, layers: usize,
    local: bool, mtp: bool) -> anyhow::Result<ResidentWeights> {
    use crate::formats::fp8_experts::{Fp8Projection, Slicing};
    use crate::V41Exl3Layer;
    let shape = catalog.routed_experts();
    let mut selections = (shape.first_layer..if local { layers } else { shape.first_layer })
        .map(V41Exl3Layer::Backbone)
        .chain((0..usize::from(mtp)).map(V41Exl3Layer::Dspark));
    selections.try_fold(ResidentWeights::default(), |sum, selection| {
        let bytes = if let Some(exl3) = catalog.exl3() {
            exl3.residency(selection, 1, 0)?.device_arena_layout()?.1 as u64
        } else {
            let layer = match selection {
                V41Exl3Layer::Backbone(layer) => layer,
                V41Exl3Layer::Dspark(stage) => shape.layers + stage,
            };
            let tensors = catalog.fp8().ok_or_else(|| anyhow::anyhow!("Qwen expert headers unavailable"))?
                .for_layer(layer)?.for_coordinator();
            tensors.validate_layer(layer)?;
            Fp8Projection::ALL.iter().try_fold(0u64, |sum, &projection| {
                let (weights, _) = tensors.slice_bytes_with(projection, 1, 0, Slicing::Padded)?;
                let scales = tensors.scale_region_bytes_with(projection, 1, 0, Slicing::Padded)?;
                (weights as u64).checked_mul(shape.experts as u64)
                    .and_then(|bytes| bytes.checked_add(scales as u64))
                    .and_then(|bytes| sum.checked_add(bytes))
                    .ok_or_else(|| anyhow::anyhow!("Qwen expert residency overflow"))
            })?
        };
        Ok(sum.checked_add(ResidentWeights(bytes))?)
    })
}

/// Header-only exact expert costs. Qwen TP2 owns complete H128 blocks: at
/// intermediate640 the halves are384/256. Callers must validate packages with
/// those widths before admitting this request; padded320/320 is not this ABI.
pub fn dual_expert_costs(catalog: &crate::OfficialV41Catalog, layers: usize)
    -> anyhow::Result<Vec<Option<ExpertCost>>> {
    let shape = catalog.routed_experts();
    anyhow::ensure!(shape.first_layer == 0 && layers <= shape.layers,
        "Qwen expert costs require a complete routed backbone prefix");
    (0..layers).map(|layer| {
        let bytes = |world, rank| -> anyhow::Result<Bytes2> {
            let resident = if let Some(exl3) = catalog.exl3() {
                exl3.residency(crate::V41Exl3Layer::Backbone(layer), world, rank)?.device_arena_layout()?.1 as u64
            } else {
                use crate::formats::fp8_experts::{Fp8Projection, Slicing};
                let tensors = catalog.fp8().ok_or_else(|| anyhow::anyhow!("Qwen TP2 needs EXL3/FP8/NVFP4 tensors"))?
                    .for_layer(layer)?.for_coordinator();
                tensors.validate_layer(layer)?;
                let slicing = if world == 2 { Slicing::Blocks(128) } else { Slicing::Padded };
                Fp8Projection::ALL.iter().try_fold(0u64, |total, &projection| {
                    let (weights, _) = tensors.slice_bytes_with(projection, world, rank, slicing)?;
                    let scales = tensors.scale_region_bytes_with(projection, world, rank, slicing)?;
                    (weights as u64).checked_mul(shape.experts as u64)
                        .and_then(|n| n.checked_add(scales as u64)).and_then(|n| n.checked_add(total))
                        .ok_or_else(|| anyhow::anyhow!("Qwen TP2 expert residency overflow"))
                })?
            };
            Ok(Bytes2 { resident, staging: 0 })
        };
        let supported = catalog.fp8().map(|tensors| tensors.layer_format(layer)
            .map(|format| format == tensors.format())).transpose()?.unwrap_or(true);
        Ok(Some(ExpertCost { whole: bytes(1, 0)?, half: [bytes(2, 0)?, bytes(2, 1)?],
            tp2: supported, whole_supported: supported, spark_ok: true }))
    }).collect()
}

/// Package selection is checkpoint-native and shared by plan and runtime.
pub fn dual_packages(catalog: &crate::OfficialV41Catalog, native_lib: &std::path::Path)
    -> anyhow::Result<[std::path::PathBuf; 2]> {
    let root = native_lib.parent().unwrap_or(std::path::Path::new("."));
    if let Some(exl3) = catalog.exl3() {
        let tiers = exl3.decoder_tiers().iter().map(usize::to_string).collect::<String>();
        let root = root.join("exl3").join(format!("exl3-qwen4-k{tiers}"));
        Ok([0, 1].map(|rank| root.join(format!("rtx-tp2-rank{rank}"))))
    } else {
        let tensors = catalog.fp8().ok_or_else(|| anyhow::anyhow!("Qwen TP2 needs native expert headers"))?.for_coordinator();
        let root = root.join("fp8").join(tensors.format().package_name("qwen4"));
        let rank = |rank| -> anyhow::Result<_> {
            Ok(root.join(format!("tp2-w{}", tensors.rank_width(2, rank,
                crate::formats::fp8_experts::Slicing::Blocks(128))?)))
        };
        Ok([rank(0)?, rank(1)?])
    }
}

/// The separate, full-width MTP expert arena stays on the head owner. Include
/// every retained TP1 capacity, not the target's half-width workspace.
pub fn dual_draft_demands(catalog: &crate::OfficialV41Catalog, native_lib: &std::path::Path,
    rows: u64) -> anyhow::Result<Vec<Demand>> {
    use crate::formats::fp8_experts::{ExpertFormat, Fp8Projection, Slicing};
    let shape = catalog.routed_experts();
    anyhow::ensure!(rows > 0, "Qwen MTP workspace needs positive rows");
    let root = native_lib.parent().unwrap_or(std::path::Path::new("."));
    let (resident, scratch) = if let Some(exl3) = catalog.exl3() {
        let resident = exl3.residency(crate::V41Exl3Layer::Dspark(0), 1, 0)?.device_arena_layout()?.1 as u64;
        let tiers = exl3.decoder_tiers().iter().map(usize::to_string).collect::<String>();
        let root = root.join("exl3").join(format!("exl3-qwen4-k{tiers}/rtx-tp1"));
        anyhow::ensure!(rows > 0 && rows <= 4096, "Qwen MTP EXL3 capacity exceeds 4096 rows");
        let manifests = crate::placement::inventory::exl3_capacities(rows).into_iter().map(|capacity| {
            let path = root.join(format!("m{capacity}/v41_exl3.json"));
            let m: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).map_err(|error| anyhow::anyhow!("Qwen expert package {}: {error}", path.display()))?)?;
            anyhow::ensure!(m["hidden"].as_u64() == Some(shape.hidden as u64)
                && m["intermediate"].as_u64() == Some(shape.intermediate as u64)
                && m["experts"].as_u64() == Some(shape.experts as u64)
                && m["top_k"].as_u64() == Some(shape.topk as u64)
                && m["capacity"].as_u64() == Some(capacity), "Qwen MTP package geometry: {}", path.display());
            Ok(m)
        }).collect::<anyhow::Result<Vec<_>>>()?;
        (resident, crate::serving_capacity::exl3_workspace_bytes(&manifests, true, Exl3Epilogue::Sum)? + rows * shape.hidden as u64 * 2)
    } else {
        let tensors = catalog.fp8().ok_or_else(|| anyhow::anyhow!("Qwen MTP format unsupported"))?.for_layer(shape.layers)?;
        tensors.validate_layer(shape.layers)?;
        let resident = Fp8Projection::ALL.iter().try_fold(0u64, |total, &projection| {
            let (weights, _) = tensors.slice_bytes_with(projection, 1, 0, Slicing::Padded)?;
            let scales = tensors.scale_region_bytes_with(projection, 1, 0, Slicing::Padded)?;
            Ok::<_, anyhow::Error>(total + weights as u64 * shape.experts as u64 + scales as u64)
        })?;
        let root = root.join("fp8").join(tensors.format().package_name("qwen4"));
        let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(root.join("manifest.json"))?)?;
        let info = &manifest["layouts"]["tp1"];
        let weights = match tensors.format() {
            ExpertFormat::Fp8Block128 => info["weights"].as_str() == Some("fp8"),
            ExpertFormat::Mxfp4 => info["weights"].as_str() == Some("mxfp4"),
            ExpertFormat::Bf16 => info["weights"].as_str() == Some("bf16"),
            ExpertFormat::Nvfp4 => matches!(info["weights"].as_str(), Some("nvfp4" | "nvfp4a4")),
        };
        anyhow::ensure!(info["tp"].as_u64() == Some(1)
            && info["hidden"].as_u64() == Some(shape.hidden as u64)
            && info["intermediate"].as_u64() == Some(shape.intermediate as u64)
            && info["slice"].as_u64() == Some(shape.intermediate as u64)
            && info["experts"].as_u64() == Some(shape.experts as u64)
            && info["top_k"].as_u64() == Some(shape.topk as u64)
            && info["input"].as_str() == Some("bf16") && weights,
            "Qwen MTP package must be native full-width BF16-input TP1: {}", root.display());
        let scratch = crate::placement::inventory::fp8moe_scratch_bytes(&manifest, "tp1", rows)
            .ok_or_else(|| anyhow::anyhow!("Qwen MTP package has no capacity for {rows} rows"))?.max(256);
        (resident, scratch)
    };
    Ok(vec![Demand::new(0, Category::Experts, "MTP expert weights", resident, Basis::Exact)
            .with_resident_weights(ResidentWeights(resident)),
        Demand::new(0, Category::Experts, "MTP expert workspace", scratch, Basis::Formula)])
}

/// Exact rank-package workspace before CUDA initialization. Packages are
/// explicitly selected by the caller; unequal ranks may never fall back to a
/// padded sibling. The runtime also checks its DSO ABI against these totals.
pub fn dual_expert_workspace(catalog: &crate::OfficialV41Catalog,
    packages: [&std::path::Path; 2], rows: u64) -> anyhow::Result<[u64; 2]> {
    use crate::formats::fp8_experts::{ExpertFormat, Slicing};
    let shape = catalog.routed_experts();
    anyhow::ensure!(rows > 0, "Qwen TP2 workspace needs positive rows");
    let mut bytes = [0; 2];
    for rank in 0..2 {
        let package = packages[rank];
        let width = if let Some(exl3) = catalog.exl3() {
            exl3.residency(crate::V41Exl3Layer::Backbone(0), 2, rank)?.intermediate as u64
        } else {
            catalog.fp8().ok_or_else(|| anyhow::anyhow!("Qwen TP2 expert format is unsupported"))?.for_coordinator()
                .rank_width(2, rank, Slicing::Blocks(128))? as u64
        };
        let scratch = if catalog.exl3().is_some() {
            anyhow::ensure!(rows <= 4096, "Qwen EXL3 TP2 capacity exceeds 4096 rows");
            let manifests = crate::placement::inventory::exl3_capacities(rows).into_iter().map(|capacity| {
                let path = package.join(format!("m{capacity}/v41_exl3.json"));
                let value: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).map_err(|error| anyhow::anyhow!("Qwen expert package {}: {error}", path.display()))?)?;
                anyhow::ensure!(value["hidden"].as_u64() == Some(shape.hidden as u64)
                    && value["intermediate"].as_u64() == Some(width)
                    && value["experts"].as_u64() == Some(shape.experts as u64)
                    && value["top_k"].as_u64() == Some(shape.topk as u64)
                    && value["capacity"].as_u64() == Some(capacity)
                    && value["output_dtype"].as_str() == Some("fp32"),
                    "Qwen EXL3 TP2 rank {rank} package geometry mismatch: {}", path.display());
                Ok(value)
            }).collect::<anyhow::Result<Vec<_>>>()?;
            crate::serving_capacity::exl3_workspace_bytes(&manifests, true, Exl3Epilogue::Sum)?
        } else {
            let tensors = catalog.fp8().unwrap();
            let parent = package.parent().ok_or_else(|| anyhow::anyhow!("Qwen TP2 package has no parent"))?;
            let name = package.file_name().and_then(|n| n.to_str())
                .ok_or_else(|| anyhow::anyhow!("Qwen TP2 package has no layout name"))?;
            let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(parent.join("manifest.json"))?)?;
            let info = &manifest["layouts"][name];
            let weights = match tensors.format() {
                ExpertFormat::Fp8Block128 => info["weights"].as_str() == Some("fp8"),
                ExpertFormat::Mxfp4 => info["weights"].as_str() == Some("mxfp4"),
            ExpertFormat::Bf16 => info["weights"].as_str() == Some("bf16"),
                ExpertFormat::Nvfp4 => matches!(info["weights"].as_str(), Some("nvfp4" | "nvfp4a4")),
            };
            anyhow::ensure!(info["tp"].as_u64() == Some(2)
                && info["hidden"].as_u64() == Some(shape.hidden as u64)
                && info["intermediate"].as_u64() == Some(shape.intermediate as u64)
                && info["slice"].as_u64() == Some(width)
                && info["experts"].as_u64() == Some(shape.experts as u64)
                && info["top_k"].as_u64() == Some(shape.topk as u64)
                && info["input"].as_str() == Some("bf16") && weights,
                "Qwen FP8/NVFP4 TP2 rank {rank} package geometry mismatch: {}", package.display());
            crate::placement::inventory::fp8moe_scratch_bytes(&manifest, name, rows)
                .ok_or_else(|| anyhow::anyhow!("Qwen TP2 rank {rank} has no capacity for {rows} rows"))?
        };
        let output_element = if catalog.exl3().is_some() { 4 } else { 2 };
        bytes[rank] = rows.checked_mul(shape.hidden as u64).and_then(|n| n.checked_mul(output_element))
            .and_then(|output| output.checked_add(scratch))
            .ok_or_else(|| anyhow::anyhow!("Qwen TP2 workspace overflow"))?;
    }
    Ok(bytes)
}

/// Exact TP2 row ABI shared by admission and the owner executor. EXL3 sends
/// E4M3/K32 wire rows and produces FP32 partials; native packages use BF16.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QwenTp2Rows {
    pub input: u64,
    pub ids: u64,
    pub weights: u64,
    pub bytes: u64,
    pub partial_bytes: u64,
    pub reduction_bytes: u64,
    pub exchange_slot_bytes: u64,
}

impl QwenTp2Rows {
    pub fn new(hidden: u64, topk: u64, rows: u64, wire: bool) -> anyhow::Result<Self> {
        anyhow::ensure!(hidden > 0 && topk > 0 && rows > 0 && (!wire || hidden % 32 == 0),
            "invalid Qwen TP2 row geometry");
        let overflow = || anyhow::anyhow!("Qwen TP2 row extent overflow");
        let mul = |a: u64, b: u64| a.checked_mul(b).ok_or_else(overflow);
        let align = |bytes: u64| bytes.checked_add(15).map(|n| n / 16 * 16).ok_or_else(overflow);
        let stride = if wire { hidden.checked_add(hidden / 32).ok_or_else(overflow)? }
            else { mul(hidden, 2)? };
        let input = mul(rows, stride)?;
        let ids = align(input)?;
        let route_bytes = align(mul(mul(rows, topk)?, 4)?)?;
        let weights = ids.checked_add(route_bytes).ok_or_else(overflow)?;
        let bytes = weights.checked_add(route_bytes).ok_or_else(overflow)?;
        let partial_bytes = mul(mul(rows, hidden)?, if wire { 4 } else { 2 })?;
        Ok(Self { input, ids, weights, bytes, partial_bytes,
            reduction_bytes: mul(mul(rows, hidden)?, 2)?,
            exchange_slot_bytes: align(bytes.max(partial_bytes))? })
    }

    /// Persistent buffers of the abortable two-slot expert exchange. Native
    /// publisher modules and streams belong to the context/module baseline.
    pub fn demands(self) -> anyhow::Result<Vec<Demand>> {
        let receive = self.exchange_slot_bytes.max(256).checked_mul(2)
            .ok_or_else(|| anyhow::anyhow!("Qwen TP2 receive extent overflow"))?;
        Ok((0..2).flat_map(|gpu| [
            Demand::new(gpu, Category::Transport, "expert send", self.bytes, Basis::Formula),
            Demand::new(gpu, Category::Transport, "expert receive", receive, Basis::Formula),
            Demand::new(gpu, Category::Transport, "expert reduction", self.reduction_bytes, Basis::Formula),
            Demand::new(gpu, Category::Transport, "expert peer control", 256, Basis::Formula),
            Demand::new(gpu, Category::Transport, "expert abort word", 256, Basis::Formula),
        ]).collect())
    }
}

/// Whole-width non-layer operands on the entry/head/MTP owner. Layer operands
/// are passed to `dual_request` separately; PLE backing follows its layer owner,
/// and draft expert arenas/workspaces must be supplied by their package plan.
pub fn dual_owner0_weights(resident: &crate::families::qwen4::resident::Qwen4CheckpointResident,
    host_embedding: bool) -> Vec<Demand> {
    [
        (Category::Weights, "owner0 head", resident.head_bytes),
        (Category::Weights, "owner0 mixer", resident.entry_bytes),
        (Category::Weights, "owner0 MTP weights", resident.mtp_bytes),
        (Category::Embedding, "owner0 embedding", if host_embedding { 0 } else { resident.embedding_bytes }),
    ].into_iter().filter(|(_, _, bytes)| *bytes > 0)
        .map(|(category, group, bytes)| Demand::new(0, category, group, bytes, Basis::Exact)
            .with_resident_weights(ResidentWeights(bytes))).collect()
}

/// Conservative load-only conversion peak on both nonempty owners, before the
/// cutover is chosen. Kept as a steady-state reserve until load lifetimes are
/// modeled; the paired group does not bias the owner-boundary objective.
pub fn dual_projection_load_staging(cfg: &crate::families::qwen4::Qwen4Config, layers: usize, mtp: bool,
    selected: crate::families::qwen4::resident::Qwen4Representation) -> Vec<Demand> {
    let bytes = crate::families::qwen4::resident::resident_bytes(cfg, layers, mtp, selected)
        .max_device_staging as u64;
    if bytes == 0 { return Vec::new(); }
    (0..2).map(|gpu| Demand::new(gpu, Category::Staging,
        "FP8 projection load staging", bytes, Basis::Formula)).collect()
}

/// Only the scale follows the owning layer; tables stay mapped and step
/// workspace accounting includes gathered row staging.
pub fn dual_ple_demands(cfg: &crate::families::qwen4::Qwen4Config, layers: usize,
    resident: &crate::families::qwen4::resident::Qwen4CheckpointResident) -> anyhow::Result<Vec<Vec<LayerExtraDemand>>> {
    anyhow::ensure!(layers <= cfg.layers && cfg.ple_layers.len() <= 1,
        "Qwen PLE demands need a backbone prefix and at most one table");
    let mut demands = vec![Vec::new(); layers];
    if let Some(&layer) = cfg.ple_layers.first().filter(|&&layer| layer < layers) {
        anyhow::ensure!(resident.ple_table_bytes > 0, "Qwen PLE layer {layer} has no table backing headers");
        demands[layer].push(LayerExtraDemand::new(Category::Tables, "PLE scale", 256,
            LayerResidency::Owner, Basis::Formula).with_resident_weights(ResidentWeights(256)));
    }
    Ok(demands)
}

/// Admission contract for the private whole-owner executor. Unlike the legacy
/// single-owner path, layer weights and routed halves are future allocations;
/// both baselines must be sampled before loading them. The public selector
/// stays single-owner until safety and numerical gates enable this caller.
/// Match each owner's loaded program recipe, including its attention reduction plan.
pub fn dual_program_manifests(manifest: &serde_json::Value, sms: [u32; 2],
    plan: cuteafd_core::coordinator_programs::AttentionReductionPlan) -> anyhow::Result<[serde_json::Value; 2]> {
    let selected = cuteafd_core::coordinator_programs::CoordinatorPrograms { family: "qwen4", split_family: None };
    let mut manifests = [manifest.clone(), manifest.clone()];
    for (owner, manifest) in manifests.iter_mut().enumerate() {
        cuteafd_core::coordinator_programs::select_card_plans(manifest, sms[owner])?;
        for entry in manifest["programs"].as_array_mut().into_iter().flatten() {
            if entry["name"].as_str().is_some_and(|name| selected.contains(name)) {
                if let Some(bytes) = cuteafd_core::coordinator_programs::attention_plan_scratch(&entry["geometry"], plan)? {
                    entry["scratch_bytes_at_capacity"]["scratch"] = bytes.into();
                }
            }
        }
    }
    Ok(manifests)
}

/// PeerExchange has two receive slots plus its control slot; progress is 48
/// bytes per slot, with the allocator's 256-byte minimum on each owner.
pub fn dual_peer_diagnostics(enabled: bool) -> Vec<Demand> {
    if !enabled { return Vec::new(); }
    [0, 1].map(|gpu| Demand::new(gpu, Category::Workspace,
        "expert peer diagnostics", 256, Basis::Exact)).to_vec()
}
pub struct QwenDualInputs<'a> {
    pub resident_weight_limit: Option<ResidentWeights>,
    /// True only when the operator left context automatic.
    pub automatic_context: bool,
    pub admission: QwenAdmissionInputs<'a>,
    pub owner_manifests: [Option<&'a serde_json::Value>; 2],
    pub gpus: [GpuBudget; 2],
    pub layer_weights: &'a [u64],
    pub layer_experts: &'a [Option<ExpertCost>],
    /// Owner-local side allocations, such as the PLE scale.
    pub layer_extra: &'a [Vec<LayerExtraDemand>],
    /// Head, MTP weights/experts, embedding and exact transport
    /// buffers. Head/MTP are owner-only groups, not paired rank groups.
    pub fixed: Vec<Demand>,
    pub tp2_workspace: [u64; 2],
    pub pending_code_bytes: [u64; 2],
    pub mark_slots: u64,
    pub max_context: u64,
    pub requested_pool: Option<u64>,
    pub spark_ranks: usize,
    pub onboard: Onboard,
    pub startup_graph_modes: Option<(usize, bool)>,
    /// Coordinator host owners and the host sample they are admitted against.
    pub host: QwenHost,
    /// Bundled vision tower candidates, chosen by the solve.
    pub encoders: QwenEncoders,
}

/// The executor's only legal owner chain today is one contiguous cutover, so
/// the family states the balanced cut once instead of searching per-layer cuts
/// the executor cannot run and the shipped dense route never reaches.
pub fn dual_cut(layers: usize) -> usize { layers / 2 }

fn dual_owners(layers: usize) -> Vec<usize> {
    let cut = dual_cut(layers);
    (0..layers).map(|layer| usize::from(layer >= cut)).collect()
}

/// The dual pool stops at the common target: the executor allocates both
/// ranks' page tables for the same admitted pool.
fn dual_policy(inputs: &QwenDualInputs<'_>) -> PoolPolicy {
    PoolPolicy::resolve(&inputs.gpus.map(|gpu| gpu.capacity_bytes), inputs.max_context,
        inputs.requested_pool, QWEN_UNIT_ROWS as u64, inputs.spark_ranks == 0)
        .with_automatic_context(inputs.automatic_context)
}

/// TP2 arenas follow the executor's load chronology: one workspace opened once,
/// both halves' resident bytes, and the largest staging peak beside them. A
/// layer without a routed expert package contributes nothing.
fn tp2_arena(inputs: &QwenDualInputs<'_>) -> Result<([u64; 2], [u64; 2]), Refusal> {
    let mut resident = inputs.tp2_workspace;
    let mut peak = inputs.tp2_workspace;
    for cost in inputs.layer_experts.iter().flatten() {
        for gpu in 0..2 {
            let loading = resident[gpu].checked_add(cost.half[gpu].resident)
                .and_then(|bytes| bytes.checked_add(cost.half[gpu].staging))
                .ok_or(Refusal::Overflow("Qwen TP2 arena"))?;
            peak[gpu] = peak[gpu].max(loading);
            resident[gpu] = resident[gpu].checked_add(cost.half[gpu].resident)
                .ok_or(Refusal::Overflow("Qwen TP2 resident halves"))?;
        }
    }
    let halves = [resident[0] - inputs.tp2_workspace[0], resident[1] - inputs.tp2_workspace[1]];
    Ok((peak, halves))
}

/// The dual route prepared once: the balanced cut, per-owner pool units, both
/// TP2 arenas and one flat graph reserve for each owner chain.
pub struct QwenDual<'a> {
    inputs: &'a QwenDualInputs<'a>,
    owners: Vec<usize>,
    context: u64,
    maximum_pages: u64,
    arena: [u64; 2],
    resident: [u64; 2],
    hop_spec: HopSpec,
    hop_bytes: [u64; 2],
    graphs: Option<GraphSet>,
}

impl<'a> QwenDual<'a> {
    pub fn new(inputs: &'a QwenDualInputs<'a>) -> Result<Self, Refusal> {
        let a = &inputs.admission;
        let page = QWEN_UNIT_ROWS as u64;
        if a.layers <= 1 || a.layers > a.cfg.layers
            || inputs.layer_weights.len() != a.layers
            || inputs.layer_experts.len() != a.layers
            || inputs.layer_extra.len() != a.layers
        {
            return Err(Refusal::InvalidFact("Qwen dual demands must name every backbone layer".into()));
        }
        if inputs.spark_ranks != 0 {
            return Err(Refusal::InvalidFact("Qwen dual is a dense RTX route and takes no Spark ranks".into()));
        }
        if a.future_expert_bytes != 0 {
            return Err(Refusal::InvalidFact("Qwen dual experts must be solver-owned TP2 halves".into()));
        }
        // A layer with no routed expert package has nothing to split; any
        // package that is present must be the TP2 half the executor loads.
        if inputs.layer_experts.iter().flatten().any(|cost| !cost.tp2) {
            return Err(Refusal::InvalidFact(
                "Qwen dual requires TP2 expert packages on every routed layer".into()));
        }
        match inputs.onboard {
            Onboard::Auto => {}
            Onboard::Layers(count) if count == a.layers => {}
            Onboard::Layers(count) => return Err(Refusal::InvalidFact(format!(
                "Qwen dual TP2 onboards all {} routed layers, not {count}", a.layers))),
        }
        let owners = dual_owners(a.layers);
        check_dual_layer_owners(&owners).map_err(|error| Refusal::InvalidFact(error.to_string()))?;
        let modes: Vec<_> = owners.iter().map(|&gpu|
            LayerMode::Whole { gpu: gpu as u8, ffn: FfnMode::Owner }).collect();
        let hop_spec = HopSpec { storage: HopStorage::ReceiveSlots,
            row_bytes: a.cfg.hc_width() as u64 * 2, rows: a.prefill_rows.max(64), lanes: 1,
            entry_gpu: 0, head_gpu: 0, terminal: TerminalHome::AttentionOwner };
        let hops = plan_hops(&modes, &hop_spec)
            .map_err(|error| Refusal::InvalidFact(error.to_string()))?;
        let hop_bytes: [u64; 2] = hop_buffer_bytes(&hops, &hop_spec, 2)
            .ok_or(Refusal::Overflow("Qwen residual hop buffers"))?
            .try_into().map_err(|_| Refusal::InvalidFact("Qwen hop owner count".into()))?;
        let (arena, resident) = tp2_arena(inputs)?;
        let policy = dual_policy(inputs);
        let maximum_pages = policy.maximum_tokens().max(policy.floor).div_ceil(page);
        if maximum_pages == 0 || maximum_pages == u64::MAX || policy.floor == 0 {
            return Err(Refusal::InvalidFact("invalid Qwen dual pool extent".into()));
        }
        let graphs = match inputs.startup_graph_modes {
            Some((sequences, speculation)) => {
                let tokens = maximum_pages.checked_mul(page)
                    .ok_or(Refusal::Overflow("Qwen pool extent"))?;
                Some(qwen_startup_graphs_placed(context_of_extent(inputs.max_context)?,
                    tokens_of_extent(tokens)?, a.cfg.dense_context(), sequences, speculation,
                    &owners)
                    .ok_or_else(|| Refusal::InvalidFact("Qwen owner graph count overflow".into()))?)
            }
            None => None,
        };
        Ok(Self { inputs, owners, context: policy.floor, maximum_pages, arena, resident,
            hop_spec, hop_bytes, graphs })
    }

    pub fn request(&self) -> Result<Request, Refusal> {
        let inputs = self.inputs;
        let policy = dual_policy(inputs);
        Ok(Request { context: self.context, pool: pool_pin(&policy, QWEN_UNIT_ROWS as u64)?,
            onboard: Pin::Auto, prefill: Pin::Exact(prefill_of(&inputs.admission)),
            operation: Operation::Serving, resident_weight_limit: inputs.resident_weight_limit,
            spark_reply_formats: reply_formats(SparkReplyFormat::Bf16, inputs.spark_ranks) })
    }

    pub fn budgets(&self) -> Result<Vec<Budget>, Refusal> {
        let mut budgets = self.inputs.gpus.iter().enumerate().map(|(gpu, budget)| Ok(Budget {
            id: format!("rtx{gpu}"), locations: vec![Location::coordinator(gpu as u8)],
            physical_bytes: budget.capacity_bytes, logical_bytes: budget.capacity_bytes,
            baseline: plan_baseline(&budget.baseline, budget.capacity_bytes,
                &format!("qwen4 dual owner{gpu}"))?,
            headroom: budget.headroom_bytes })).collect::<Result<Vec<_>, Refusal>>()?;
        budgets.push(self.inputs.host.budget()?);
        Ok(budgets)
    }

    pub fn admit(&self) -> Result<ResolvedPlan, Refusal> {
        solve(self.request()?, self.budgets()?, self)
    }

    /// Mechanical executor output: the prepared cut, TP2 arenas and hop
    /// buffers, never a re-solve.
    pub fn execution(&self, solved: &ResolvedPlan) -> Result<Placement, Refusal> {
        let a = &self.inputs.admission;
        let modes = &solved.decisions.attention;
        if modes.len() != a.layers || modes.iter().zip(&self.owners).any(|(mode, &owner)|
            *mode != LayerMode::Whole { gpu: owner as u8, ffn: FfnMode::Owner })
        {
            return Err(Refusal::InvalidFact(
                "resolved Qwen owner modes differ from the prepared cut".into()));
        }
        if solved.decisions.prefill != prefill_of(a) || solved.decisions.lanes != 1 {
            return Err(Refusal::InvalidFact(
                "resolved Qwen prefill differs from the prepared shape".into()));
        }
        if solved.decisions.onboard_layers.iter().any(|layers| layers.len() != a.layers) {
            return Err(Refusal::InvalidFact("Qwen dual requires full TP2 onboarding".into()));
        }
        if self.hop_spec.storage != HopStorage::ReceiveSlots {
            return Err(Refusal::InvalidFact("Qwen dual hops require receive slots".into()));
        }
        let residual = residual_homes(modes, &self.hop_spec)
            .map_err(|error| Refusal::InvalidFact(error.to_string()))?;
        let hops = plan_hops(modes, &self.hop_spec)
            .map_err(|error| Refusal::InvalidFact(error.to_string()))?;
        let mut by_kind = BTreeMap::new();
        for &kind in &a.cfg.attention[..a.layers] {
            by_kind.insert(if kind == Qwen4Attention::Full { AttentionClass::Gqa }
                else { AttentionClass::Gdn }, AttentionPlacement::Layers);
        }
        Ok(Placement { pool_tokens: solved.pool_tokens, pool_limits: PoolLimits(vec![]),
            attention_placement: AttentionPlacement::Layers,
            attention_by_kind: by_kind.into_iter().collect(),
            peer_row_bytes: 0, onboard_layers: a.layers, max_pool_tokens_no_onboard: None,
            layers: modes.iter().map(|&mode| LayerAssignment { mode, experts: ExpertHome::RtxTp2 })
                .collect(),
            movables: Vec::new(),
            expert_ranges: (0..2).map(|_| ExpertRange { first: 0, layers: 0, peak_bytes: 0 })
                .collect(),
            residual, hops,
            tp2: Some(Tp2Range { first: 0, layers: a.layers, peak_bytes: self.arena }),
            items: (0..2).map(|gpu| solved.items(Location::coordinator(gpu as u8)))
                .collect::<Result<Vec<_>, Refusal>>()? })
    }
}

impl Recipe for QwenDual<'_> {
    fn prepare(&self, request: &Request) -> Result<Contract, Refusal> {
        let inputs = self.inputs;
        let a = &inputs.admission;
        let cfg = a.cfg;
        let page = QWEN_UNIT_ROWS as u64;
        if request.context != self.context || self.maximum_pages == 0 {
            return Err(Refusal::InvalidFact("invalid Qwen dual extent".into()));
        }
        let compute = |layers, mtp| crate::serving_capacity::qwen_cache_geometry(cfg, layers, mtp,
            a.kv_format).map_err(|error| Refusal::InvalidFact(format!("Qwen cache geometry: {error}")));
        let mut components = Vec::new();
        let mut previous = crate::serving_capacity::RankCacheGeometry::default();
        for (layer, &gpu) in self.owners.iter().enumerate() {
            let current = compute(layer + 1, false)?.ranks[0];
            let state = current.active_state_per_sequence_bytes
                .checked_sub(previous.active_state_per_sequence_bytes)
                .and_then(|bytes| bytes.checked_mul(a.slots))
                .and_then(|bytes| bytes.checked_add(current.speculative_replay_bytes
                    .checked_sub(previous.speculative_replay_bytes)?))
                .and_then(|bytes| current.retained_mark_bytes
                    .checked_sub(previous.retained_mark_bytes)?
                    .checked_mul(inputs.mark_slots)
                    .and_then(|marks| bytes.checked_add(marks)))
                .ok_or(Refusal::Overflow("Qwen owner state"))?;
            let kv = current.persistent_unit_bytes
                .checked_sub(previous.persistent_unit_bytes)
                .ok_or_else(|| Refusal::InvalidFact("Qwen per-layer records shrink".into()))?;
            previous = current;
            let location = Location::coordinator(gpu as u8);
            if inputs.layer_weights[layer] > 0 {
                let id = format!("qwen.layer.{layer}.weights");
                components.push(owner(id.clone(), location, inputs.layer_weights[layer],
                    Category::Weights, "layers", PlanLifetime::RESIDENT, self.maximum_pages));
                components.push(weights(id, location, ResidentWeights(inputs.layer_weights[layer]),
                    self.maximum_pages));
            }
            components.push(owner(format!("qwen.layer.{layer}.state"), location, state,
                Category::Kv, "layer state and marks", PlanLifetime::RESIDENT, self.maximum_pages));
            components.push(pooled(format!("qwen.layer.{layer}.records"), location, kv,
                Category::Kv, "records", self.maximum_pages)?);
            for (index, demand) in inputs.layer_extra[layer].iter().enumerate() {
                let id = format!("qwen.layer.{layer}.extra.{index}");
                components.push(owner(id.clone(), location, demand.bytes, demand.category,
                    &demand.group, PlanLifetime::RESIDENT, self.maximum_pages));
                if demand.resident_weights.0 > 0 {
                    components.push(weights(id, location, demand.resident_weights, self.maximum_pages));
                }
            }
        }
        let complete = compute(a.layers, a.mtp)?.ranks[0];
        // Both ranks allocate full-width page tables; owner0 alone owns pool
        // metadata, MTP records/pending rows and the deferred MTP id buffer.
        let tables = (1 + QWEN_DECODE_ROWS as u64) * 5 * 4;
        let mut overhead = [tables, tables];
        let mtp_records = complete.persistent_unit_bytes
            .checked_sub(previous.persistent_unit_bytes)
            .ok_or_else(|| Refusal::InvalidFact("Qwen MTP records shrink".into()))?;
        overhead[0] = overhead[0].checked_add(mtp_records)
            .and_then(|bytes| bytes.checked_add(complete.pool_metadata_unit_bytes))
            .ok_or(Refusal::Overflow("Qwen pool metadata"))?;
        // Preserve the single-owner per-token rounding of table bytes.
        for bytes in &mut overhead {
            *bytes = bytes.div_ceil(page).checked_mul(page)
                .ok_or(Refusal::Overflow("Qwen pool rounding"))?;
        }
        for gpu in 0..2 {
            let location = Location::coordinator(gpu as u8);
            components.push(pooled(format!("qwen.owner{gpu}.records.metadata"), location,
                overhead[gpu], Category::Kv, "records", self.maximum_pages)?);
            components.push(owner(format!("qwen.owner{gpu}.commit"), location,
                3 * QWEN_DECODE_ROWS as u64 * 4, Category::Kv, "owner commit tables",
                PlanLifetime::RESIDENT, self.maximum_pages));
            components.push(owner(format!("qwen.owner{gpu}.code"), location,
                inputs.pending_code_bytes[gpu], Category::Runtime, "loaded code",
                PlanLifetime::RESIDENT, self.maximum_pages));
            components.push(owner(format!("qwen.owner{gpu}.hops"), location, self.hop_bytes[gpu],
                Category::Transport, "residual hops", PlanLifetime::RESIDENT, self.maximum_pages));
            let admitted = qwen_admission(&QwenAdmissionInputs { prefill_shape: None, cfg,
                layers: a.layers, mtp: a.mtp, kv_format: a.kv_format,
                manifest: inputs.owner_manifests[gpu].or(a.manifest), prefill_rows: a.prefill_rows,
                slots: a.slots, mark_bytes: a.mark_bytes, full_prefill_logits: a.full_prefill_logits,
                ple: a.ple, future_expert_bytes: 0, headroom: a.headroom })
                .map_err(|error| Refusal::InvalidFact(format!("Qwen owner admission: {error}")))?;
            for (index, &(category, group, bytes)) in admitted.items.iter().enumerate() {
                if !matches!(group, "decode step" | "prefill step") { continue; }
                components.push(owner(format!("qwen.owner{gpu}.step.{index}"), location, bytes,
                    category, group, PlanLifetime::RESIDENT, self.maximum_pages));
            }
            let group = format!("TP2 expert layer halves 0..{}", a.layers);
            components.push(owner(format!("qwen.tp2.{gpu}"), location, self.arena[gpu],
                Category::Experts, &group, PlanLifetime::RESIDENT, self.maximum_pages));
            components.push(weights(format!("qwen.tp2.{gpu}"), location,
                ResidentWeights(self.resident[gpu]), self.maximum_pages));
            components.push(match &self.graphs {
                Some(graphs) => allowance("startup graphs", location, graphs.bytes(gpu),
                    PlanLifetime { first: Phase::Warmup, last: Phase::Growth }, self.maximum_pages),
                None => allowance("graph growth", location, LAZY_GRAPH_BYTES,
                    PlanLifetime { first: Phase::Ready, last: Phase::Growth }, self.maximum_pages),
            });
        }
        for (index, demand) in inputs.fixed.iter().enumerate() {
            if usize::from(demand.gpu) >= 2 {
                return Err(Refusal::InvalidFact(format!(
                    "Qwen dual demand {} names owner {}", demand.group, demand.gpu)));
            }
            let location = Location::coordinator(demand.gpu);
            let id = format!("qwen.demand.{index}");
            components.push(owner(id.clone(), location, demand.bytes, demand.category,
                &demand.group, PlanLifetime::RESIDENT, self.maximum_pages));
            if demand.resident_weights.0 > 0 {
                components.push(weights(id, location, demand.resident_weights, self.maximum_pages));
            }
        }
        let pending = complete.active_state_per_sequence_bytes
            .checked_sub(previous.active_state_per_sequence_bytes)
            .and_then(|bytes| bytes.checked_mul(a.slots))
            .and_then(|bytes| bytes.checked_add(complete.fixed_state_bytes
                .checked_sub(3 * QWEN_DECODE_ROWS as u64 * 4)?))
            .ok_or(Refusal::Overflow("Qwen MTP state"))?;
        // The MTP layer reads the head owner's pre-mixer streams every step and
        // shares its head and embedding, so it stays on owner0; its MTP KV
        // record stays in that owner's pool unit.
        components.push(owner("qwen.owner0.mtp.pending", Location::coordinator(0), pending,
            Category::Kv, "owner0 MTP pending", PlanLifetime::RESIDENT, self.maximum_pages));
        if a.full_prefill_logits > 0 {
            components.push(owner("qwen.owner0.probe.logits", Location::coordinator(0),
                a.full_prefill_logits, Category::Workspace, "probe prefill logits",
                PlanLifetime::RESIDENT, self.maximum_pages));
        }
        if inputs.host.uploads.len() != 2 {
            return Err(Refusal::InvalidFact("Qwen dual host owners must name both upload owners".into()));
        }
        components.extend(inputs.host.components(self.maximum_pages)?);
        let mut variables = Vec::with_capacity(a.layers + 3);
        for (layer, &owner) in self.owners.iter().enumerate() {
            variables.push(Variable::Attention { layer,
                values: vec![LayerMode::Whole { gpu: owner as u8, ffn: FfnMode::Owner }] });
        }
        variables.push(Variable::Prefill(vec![prefill_of(a)]));
        variables.push(Variable::Lanes(vec![1]));
        variables.push(Variable::Onboard { layers: (0..a.layers).collect(),
            destination: vec![0, 1], counts: vec![a.layers] });
        inputs.encoders.extend(&mut variables, &mut components, 2, self.maximum_pages)?;
        Ok(Contract { phase_order: PhaseOrder::TARGET_THEN_DRAFT, spark_reply_formats: None,
            resident_weight_limit: request.resident_weight_limit, page_rows: page,
            maximum_pages: self.maximum_pages, gpus: 2, variables, requires: vec![], components })
    }
}

/// Admit the dual route and publish the executor's placement.
pub fn dual_placement(inputs: &QwenDualInputs<'_>) -> anyhow::Result<Placement> {
    Ok(dual_admitted(inputs)?.0)
}

/// The placement with its solved plan (phase ledger and host owners).
pub fn dual_admitted(inputs: &QwenDualInputs<'_>) -> anyhow::Result<(Placement, ResolvedPlan)> {
    if let Some((sequences, _)) = inputs.startup_graph_modes {
        cuteafd_core::concurrency::validate("qwen4", sequences as u64)?;
    }
    let recipe = QwenDual::new(inputs)?;
    let solved = recipe.admit()?;
    Ok((recipe.execution(&solved)?, solved))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::families::qwen4::{Qwen4Config, Qwen4KvCache};

    #[test]
    fn single_checkpoint_weights_are_independent_of_mapped_table_and_runtime_storage() {
        use crate::families::qwen4::resident::Qwen4CheckpointResident;
        let mut resident = Qwen4CheckpointResident { target_bytes: 700, layer_bytes: vec![200, 400],
            entry_bytes: 100, mtp_bytes: 300, head_bytes: 500, embedding_bytes: 600,
            ple_table_bytes: 900, ple_row_bytes: 160 };
        assert_eq!(single_owner_weights(&resident, false, false).unwrap(), ResidentWeights(2100));
        assert_eq!(single_owner_weights(&resident, true, false).unwrap(), ResidentWeights(1500));
        assert_eq!(single_owner_weights(&resident, true, true).unwrap(), ResidentWeights(1756));
        resident.ple_table_bytes = u64::MAX;
        assert_eq!(single_owner_weights(&resident, true, true).unwrap(), ResidentWeights(1756));
        let weights = single_owner_weights(&resident, false, true).unwrap();
        assert!(weights.check(0, weights).is_ok());
        assert_eq!(weights.check(0, ResidentWeights(weights.0 - 1)),
            Err(PlacementError::ResidentWeightLimit { gpu: 0, required: 2356, limit: 2355 }));
        resident.target_bytes = u64::MAX;
        assert!(single_owner_weights(&resident, false, true).is_err());
    }

    #[test]
    fn single_expert_weights_use_native_whole_arenas_not_tp2_halves_or_workspace() {
        use crate::plan::testing::{qwen4_config, qwen4_exl3, write_snapshot};
        let dir = tempfile::tempdir().unwrap();
        let config = qwen4_config(1);
        let (tensors, manifest) = qwen4_exl3(1, 4);
        write_snapshot(dir.path(), &config, &tensors, None);
        crate::plan::testing::write_quantize_config(dir.path(), &manifest);
        let catalog = crate::read_expert_catalog(dir.path()).unwrap();
        let cost = dual_expert_costs(&catalog, 1).unwrap()[0].unwrap();
        assert_eq!(single_expert_weights(&catalog, 1, true, false).unwrap(), ResidentWeights(cost.whole.resident));
        assert_eq!(single_expert_weights(&catalog, 1, false, false).unwrap(), ResidentWeights(0));
        assert_ne!(cost.whole.resident, cost.half[0].resident);
        assert_eq!(cost.whole.staging, 0);
    }

    #[test]
    fn qwen_dual_owner0_weights_exclude_layer_and_table_storage() {
        use crate::families::qwen4::resident::Qwen4CheckpointResident;
        let resident = Qwen4CheckpointResident { target_bytes: 700, layer_bytes: vec![200, 400], entry_bytes: 100,
            mtp_bytes: 300, head_bytes: 500, embedding_bytes: 600, ple_table_bytes: 900, ple_row_bytes: 160 };
        for (host, expected) in [(false, 1500), (true, 900)] {
            let demands = dual_owner0_weights(&resident, host);
            assert!(demands.iter().all(|d| d.gpu == 0 && d.basis == Basis::Exact));
            assert_eq!(demands.iter().map(|d| d.bytes).sum::<u64>(), expected);
            assert_eq!(demands.iter().filter(|d| d.category == Category::Embedding).count(), usize::from(!host));
            assert!(!demands.iter().any(|d| d.category == Category::Tables || d.category == Category::Experts));
        }
        assert!(dual_owner0_weights(&Qwen4CheckpointResident::default(), false).is_empty());
    }

    #[test]
    fn qwen_dual_ple_demands_charge_only_scale() {
        use crate::families::qwen4::resident::Qwen4CheckpointResident;
        let mut cfg = Qwen4Config::from_hf(&crate::plan::testing::qwen4_config(4)).unwrap();
        cfg.ple_layers = vec![1];
        let resident = Qwen4CheckpointResident { ple_table_bytes: 123_456, ..Default::default() };
        let demands = dual_ple_demands(&cfg, 4, &resident).unwrap();
        assert!(demands.iter().enumerate().all(|(layer, extra)| layer == 1 || extra.is_empty()));
        assert_eq!(demands[1].iter().map(|d| d.bytes).sum::<u64>(), 256);
        assert!(demands[1].iter().all(|d| d.category == Category::Tables && d.residency == LayerResidency::Owner));
        assert!(dual_ple_demands(&cfg, 1, &resident).unwrap()[0].is_empty());
        assert!(dual_ple_demands(&cfg, 4, &Qwen4CheckpointResident::default()).is_err());
        let resident = Qwen4CheckpointResident { ple_table_bytes: u64::MAX, ..Default::default() };
        assert_eq!(dual_ple_demands(&cfg, 4, &resident).unwrap()[1][0].bytes, 256);
    }
    /// A fixture TP2 cost with no resident halves: these tests price the
    /// planner's own owners, not checkpoint bytes.
    fn tp2_cost() -> ExpertCost {
        let zero = Bytes2 { resident: 0, staging: 0 };
        ExpertCost { whole: zero, half: [zero, zero], tp2: true, whole_supported: true, spark_ok: true }
    }

    #[test]
    fn qwen_dual_place_scale_follows_the_balanced_cut_and_hops_are_charged() {
        let mut cfg = Qwen4Config::from_hf(&crate::plan::testing::qwen4_config(4)).unwrap();
        cfg.ple_layers = vec![1];
        cfg.ple_dim = 2560;
        let resident = crate::families::qwen4::resident::Qwen4CheckpointResident {
            ple_table_bytes: 1 << 30, ..Default::default() };
        let extra = dual_ple_demands(&cfg, 4, &resident).unwrap();
        let weights = vec![1 << 20; 4];
        let experts = vec![Some(tp2_cost()); 4];
        let gib = 1u64 << 30;
        let inputs = QwenDualInputs {
            resident_weight_limit: None, automatic_context: false,
            admission: QwenAdmissionInputs { prefill_shape: None, cfg: &cfg, layers: 4, mtp: false,
                kv_format: Qwen4KvCache::Bf16, manifest: None, prefill_rows: 4096, slots: 16,
                mark_bytes: 0, full_prefill_logits: 0, ple: None, future_expert_bytes: 0, headroom: 0 },
            owner_manifests: [None; 2],
            gpus: [GpuBudget { capacity_bytes: 96 * gib, headroom_bytes: 0,
                baseline: Baseline::Measured { free_bytes: 96 * gib } }; 2],
            layer_weights: &weights, layer_experts: &experts, layer_extra: &extra,
            fixed: Vec::new(), tp2_workspace: [0; 2], pending_code_bytes: [0; 2], mark_slots: 18,
            max_context: 131072, requested_pool: Some(2 << 20), spark_ranks: 0, onboard: Onboard::Auto,
            startup_graph_modes: None,
            host: QwenHost::ample(&cfg, 2, 4096).unwrap(), encoders: Default::default(),
        };
        let placement = dual_placement(&inputs).unwrap();
        assert_eq!(placement.pool_tokens, 2 << 20);
        assert!(placement.movables.is_empty());
        assert_eq!(dual_cut(4), 2);
        // The PLE scale is an owner-local fact: it moves with the cutover the
        // family states, never a second search or a replicated copy.
        assert_eq!(placement.layers.iter().take_while(|layer|
            layer.mode == LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner }).count(), 2);
        for (gpu, scales) in [(0, vec![256u64]), (1, Vec::new())] {
            assert_eq!(placement.items[gpu].iter().filter(|item| item.category == Category::Tables)
                .map(|item| item.bytes).collect::<Vec<_>>(), scales);
        }
        // Both charged hops are receive slots the executor allocates.
        let row = cfg.hc_width() as u64 * 2;
        for gpu in 0..2 {
            assert_eq!(placement.items[gpu].iter().filter(|item| item.group == "residual hops")
                .map(|item| item.bytes).sum::<u64>(), 4096 * row);
        }
    }
    #[test]
    fn qwen_dual_projection_staging_preserves_bf16_and_host_head_defaults() {
        use crate::families::qwen4::resident::Qwen4Representation;
        let cfg = Qwen4Config::from_hf(&crate::plan::testing::qwen4_config(48)).unwrap();
        for mtp in [false, true] {
            for fp8_head in [false, true] {
                let selected = Qwen4Representation { fp8_projections: false, fp8_head };
                assert!(dual_projection_load_staging(&cfg, 48, mtp, selected).is_empty());
                let demands = dual_projection_load_staging(&cfg, 48, mtp,
                    Qwen4Representation { fp8_projections: true, fp8_head });
                assert_eq!(demands.len(), 2);
                for (gpu, demand) in demands.iter().enumerate() {
                    assert_eq!(demand.gpu, gpu as u8);
                    assert_eq!(demand.bytes, 84_377_600);
                    assert_eq!(demand.category, Category::Staging);
                    assert_eq!(demand.group, "FP8 projection load staging");
                }
            }
        }
    }

    #[test]
    fn qwen_tp2_transport_matches_owner_buffers_and_abi() -> anyhow::Result<()> {
        for wire in [false, true] {
            for rows in [1, 3, 64, 4096] {
                let p = QwenTp2Rows::new(2560, 10, rows, wire)?;
                let align = |bytes: u64| bytes.div_ceil(16) * 16;
                let payload = align(rows * if wire { 2640 } else { 5120 }) + 2 * align(rows * 40);
                assert_eq!(p.bytes, payload);
                assert_eq!(p.partial_bytes, rows * 2560 * if wire { 4 } else { 2 });
                assert_eq!(p.reduction_bytes, rows * 5120);
                assert_eq!(p.exchange_slot_bytes, align(payload.max(p.partial_bytes)));
                let demands = p.demands()?;
                assert_eq!(demands.len(), 10);
                for rank in 0..2 {
                    let rank_bytes = demands.iter().filter(|d| d.gpu == rank).map(|d| d.bytes).sum::<u64>();
                    assert_eq!(rank_bytes, payload + 2 * p.exchange_slot_bytes.max(256) + rows * 5120 + 512);
                }
            }
        }
        let tiny = QwenTp2Rows::new(32, 1, 1, true)?;
        assert_eq!(tiny.demands()?.iter().filter(|d| d.group == "expert receive")
            .map(|d| d.bytes).collect::<Vec<_>>(), vec![512, 512]);
        for (hidden, topk, rows, wire) in [(0, 10, 64, false), (2560, 0, 64, false),
            (2560, 10, 0, true), (2561, 10, 64, true), (u64::MAX, 10, 64, false)] {
            assert!(QwenTp2Rows::new(hidden, topk, rows, wire).is_err());
        }
        Ok(())
    }

    #[test]
    fn qwen_tp2_costs_use_exact_header_storage() {
        use crate::plan::testing::{qwen4_config, qwen4_exl3, nvfp4, write_snapshot};
        let dir = tempfile::tempdir().unwrap();
        let config = qwen4_config(1);
        let (tensors, manifest) = qwen4_exl3(1, 4);
        write_snapshot(dir.path(), &config, &tensors, None);
        crate::plan::testing::write_quantize_config(dir.path(), &manifest);
        let catalog = crate::read_expert_catalog(dir.path()).unwrap();
        let costs = dual_expert_costs(&catalog, 1).unwrap();
        let cost = costs[0].unwrap();
        assert!(cost.half[0].resident > cost.half[1].resident);
        for rank in 0..2 {
            let residency = catalog.exl3().unwrap().residency(crate::V41Exl3Layer::Backbone(0), 2, rank).unwrap();
            assert_eq!(cost.half[rank].resident, residency.device_arena_layout().unwrap().1 as u64);
            assert_eq!(cost.half[rank].staging, 0);
        }
        let packages = [dir.path().join("rtx-tp2-rank0"), dir.path().join("rtx-tp2-rank1")];
        for (rank, width) in [384, 256].into_iter().enumerate() {
            for capacity in [1, 16, 80] {
                let path = packages[rank].join(format!("m{capacity}"));
                std::fs::create_dir_all(&path).unwrap();
                let manifest = serde_json::json!({"hidden": 2560, "intermediate": width, "experts": 512,
                    "top_k": 10, "capacity": capacity, "output_dtype": "fp32", "tile": [64, 128, 64, 128], "trellis_lut": {"bytes": 32},
                    "buffers": {"scratch": {"allocation": "scratch", "bytes": capacity * width,
                        "dtype": "f16", "zero_on_create": false},
                        "state": {"allocation": "state", "bytes": 16, "zero_on_create": true}}});
                std::fs::write(path.join("v41_exl3.json"), serde_json::to_vec(&manifest).unwrap()).unwrap();
            }
        }
        assert_eq!(dual_expert_workspace(&catalog, packages.each_ref().map(|p| p.as_path()), 16).unwrap(),
            [80 * 384 + 3 * (32 + 16) + 97 * 2560 * 2 + 16 * 2560 * 4,
             80 * 256 + 3 * (32 + 16) + 97 * 2560 * 2 + 16 * 2560 * 4]);
        assert!(dual_expert_workspace(&catalog, [packages[1].as_path(), packages[0].as_path()], 16).is_err());
        assert_eq!(dual_expert_workspace(&catalog, packages.each_ref().map(|p| p.as_path()), 17).unwrap(),
            [80 * 384 + 3 * (32 + 16) + 97 * 2560 * 2 + 17 * 2560 * 4,
             80 * 256 + 3 * (32 + 16) + 97 * 2560 * 2 + 17 * 2560 * 4]);
        assert!(dual_expert_workspace(&catalog, packages.each_ref().map(|p| p.as_path()), 81).is_err());
        assert!(dual_expert_workspace(&catalog, packages.each_ref().map(|p| p.as_path()), 4097).is_err());
        std::fs::remove_file(packages[1].join("m80/v41_exl3.json")).unwrap();
        assert!(dual_expert_workspace(&catalog, packages.each_ref().map(|p| p.as_path()), 16).is_err());
        let mut config = config;
        config["quantization_config"] = serde_json::json!({"quant_method": "modelopt", "quant_algo": "NVFP4",
            "config_groups": {"group_0": {"weights": {"num_bits": 4, "type": "float", "group_size": 16}}}});
        let mut tensors = Vec::new();
        for expert in 0..512 {
            for (projection, n, k) in [("gate_proj", 640, 2560), ("up_proj", 640, 2560), ("down_proj", 2560, 640)] {
                tensors.extend(nvfp4(&format!("model.language_model.layers.0.mlp.experts.{expert}.{projection}"), n, k));
            }
        }
        // New directory avoids the EXL3 side-file cross-check from the first fixture.
        let dir = tempfile::tempdir().unwrap();
        write_snapshot(dir.path(), &config, &tensors, None);
        let catalog = crate::read_expert_catalog(dir.path()).unwrap();
        let cost = dual_expert_costs(&catalog, 1).unwrap()[0].unwrap();
        // Three projection value+scale grids, plus alpha and input scale per expert.
        for (rank, width) in [384_u64, 256].into_iter().enumerate() {
            assert_eq!(cost.half[rank], Bytes2 { resident: 512 * (3 * width * 2560 * 9 / 16 + 3 * 8), staging: 0 });
        }
        assert_eq!(cost.whole.resident, 512 * (3 * 640 * 2560 * 9 / 16 + 3 * 8));
        let package = |width, scratch| serde_json::json!({"tp": 2, "hidden": 2560, "slice": width,
            "intermediate": 640, "experts": 512, "top_k": 10, "input": "bf16", "weights": "nvfp4a4",
            "capacities": [{"capacity": 1, "scratch_bytes": 128}, {"capacity": 16, "scratch_bytes": scratch}]});
        let mut manifest = serde_json::json!({"layouts": {"tp2-w384": package(384, 1000),
            "tp2-w256": package(256, 2000)}});
        let save = |value: &serde_json::Value| std::fs::write(dir.path().join("manifest.json"),
            serde_json::to_vec(value).unwrap()).unwrap();
        save(&manifest);
        let packages = [dir.path().join("tp2-w384"), dir.path().join("tp2-w256")];
        let paths = packages.each_ref().map(|p| p.as_path());
        assert_eq!(dual_expert_workspace(&catalog, paths, 9).unwrap(), [1000 + 9 * 2560 * 2, 2000 + 9 * 2560 * 2]);
        assert_eq!(dual_expert_workspace(&catalog, paths, 1).unwrap(), [256 + 2560 * 2; 2]);
        assert!(dual_expert_workspace(&catalog, [paths[1], paths[0]], 9).is_err());
        assert!(dual_expert_workspace(&catalog, paths, 17).is_err());
        assert!(dual_expert_workspace(&catalog, paths, 0).is_err());
        manifest["layouts"]["tp2-w256"]["input"] = serde_json::json!("wire");
        save(&manifest);
        assert!(dual_expert_workspace(&catalog, paths, 9).is_err());

        config["text_config"]["mtp_num_hidden_layers"] = 1.into();
        for expert in 0..512 {
            for (projection, n, k) in [("gate_proj", 640, 2560), ("up_proj", 640, 2560), ("down_proj", 2560, 640)] {
                tensors.extend(nvfp4(&format!("mtp.layers.0.mlp.experts.{expert}.{projection}"), n, k));
            }
        }
        let draft_dir = tempfile::tempdir().unwrap();
        write_snapshot(draft_dir.path(), &config, &tensors, None);
        let catalog = crate::read_expert_catalog(draft_dir.path()).unwrap();
        let lib = draft_dir.path().join("lib/libcuteafd_native.so");
        let root = lib.parent().unwrap().join("fp8/fp8-qwen4-nvfp4a4");
        std::fs::create_dir_all(&root).unwrap();
        let path = root.join("manifest.json");
        let mut draft_manifest = serde_json::json!({"layouts": {"tp1": package(640, 1000)}});
        draft_manifest["layouts"]["tp1"]["tp"] = 1.into();
        let save_draft = |value: &serde_json::Value| std::fs::write(&path, serde_json::to_vec(value).unwrap()).unwrap();
        save_draft(&draft_manifest);
        let demand = dual_draft_demands(&catalog, &lib, 9).unwrap();
        assert_eq!(demand[0].bytes, cost.whole.resident);
        assert_eq!(demand[1].bytes, 1000);
        assert!(dual_draft_demands(&catalog, &lib, 0).is_err());
        assert!(dual_draft_demands(&catalog, &lib, 17).is_err());
        for (key, invalid) in [("tp", serde_json::json!(2)), ("slice", serde_json::json!(384)),
            ("input", serde_json::json!("wire")), ("weights", serde_json::json!("fp8"))] {
            let mut wrong = draft_manifest.clone();
            wrong["layouts"]["tp1"][key] = invalid;
            save_draft(&wrong);
            assert!(dual_draft_demands(&catalog, &lib, 9).is_err(), "invalid {key}");
        }
    }

    #[test]
    fn dual_card_scratch_selects_each_owner_and_diagnostics_are_opt_in() {
        let manifest = serde_json::json!({"programs": [{"name": "qwen4_sparse_gqa_kv_fp8_m64",
            "card_plans": [
                {"sm_count": 170, "scratch_bytes_at_capacity": {"scratch": 8192}, "geometry": {}},
                {"sm_count": 188, "scratch_bytes_at_capacity": {"scratch": 1024}, "geometry": {}}
            ]}]});
        let plan = cuteafd_core::coordinator_programs::AttentionReductionPlan::new(64, None, None).unwrap();
        for sms in [[170, 188], [188, 170]] {
            let owners = dual_program_manifests(&manifest, sms, plan).unwrap();
            for rank in 0..2 {
                assert_eq!(crate::serving_capacity::qwen_graphs::qwen_step_scratch(
                    &owners[rank], true, false, Qwen4KvCache::Fp8).0,
                    if sms[rank] == 170 { 8192 } else { 1024 });
            }
        }
        assert!(dual_peer_diagnostics(false).is_empty());
        let demands = dual_peer_diagnostics(true);
        assert_eq!(demands.len(), 2);
        for (rank, demand) in demands.iter().enumerate() {
            assert_eq!(demand.gpu as usize, rank);
            assert_eq!(demand.bytes, 256);
        }
    }

    fn single_inputs<'a>(cfg: &'a Qwen4Config, baseline: Baseline, spark_ranks: usize,
        startup_graph_modes: Option<(usize, bool)>, requested_pool: Option<u64>,
        loaded: ResidentWeights, limit: Option<ResidentWeights>) -> QwenInputs<'a> {
        let gib = 1u64 << 30;
        QwenInputs { resident_weight_limit: limit, loaded_weights: loaded,
            spark_reply_format: cuteafd_core::SparkReplyFormat::Bf16, automatic_context: false,
            admission: QwenAdmissionInputs { prefill_shape: None, cfg, layers: cfg.layers, mtp: false,
                kv_format: Qwen4KvCache::Bf16, manifest: None, prefill_rows: 4096, slots: 16,
                mark_bytes: 0, full_prefill_logits: 0, ple: None, future_expert_bytes: 0, headroom: 0 },
            capacity_bytes: 96 * gib, baseline, pending_code_bytes: 0, max_context: 131072,
            requested_pool, spark_ranks, startup_graph_modes,
            host: QwenHost::ample(cfg, 1, 4096).unwrap(), encoders: Default::default() }
    }

    /// Plan-time and serve-time baselines describe the same owners; the recipe
    /// declares every allocation after them, so both must admit one placement.
    #[test]
    fn planner_equals_runtime_qwen4() {
        let cfg = Qwen4Config::from_hf(&crate::plan::testing::qwen4_config(4)).unwrap();
        let planned = placement(&single_inputs(&cfg, Baseline::Planned { context_bytes: 0, loaded_bytes: 0 },
            0, Some((16, true)), None, ResidentWeights(1024), None)).unwrap();
        let runtime = placement(&single_inputs(&cfg, Baseline::Measured { free_bytes: 96 << 30 },
            0, Some((16, true)), None, ResidentWeights(1024), None)).unwrap();
        assert_eq!(planned, runtime);
        assert_eq!(planned.pool_tokens, 2 << 20);
        // Both baselines and CLI zero select the same automatic admission.
        let zero = placement(&single_inputs(&cfg, Baseline::Measured { free_bytes: 96 << 30 },
            0, Some((16, true)), Some(0), ResidentWeights(1024), None)).unwrap();
        assert_eq!(zero, runtime);
        let admission = qwen_admission(&single_inputs(&cfg, Baseline::Measured { free_bytes: 96 << 30 },
            0, Some((16, true)), None, ResidentWeights(1024), None).admission).unwrap();
        assert_eq!(planned.items[0].iter().filter(|item| item.group == "records")
            .map(|item| item.bytes).sum::<u64>(), admission.per_token * (2 << 20));
        let graphs = qwen_startup_graphs(131072, 2 << 20, cfg.dense_context(), 16, true, cfg.layers)
            .unwrap();
        assert_eq!(planned.items[0].iter().filter(|item| item.group == "startup graphs")
            .map(|item| item.bytes).collect::<Vec<_>>(), vec![graphs.bytes(0)]);
        // The first typed refusal is preserved, not clamped or re-solved.
        let limited = single_inputs(&cfg, Baseline::Measured { free_bytes: 96 << 30 }, 0,
            Some((16, true)), None, ResidentWeights(2048), Some(ResidentWeights(1)));
        let refused = QwenSingle::new(&limited).unwrap();
        assert_eq!(refused.admit().unwrap_err(),
            Refusal::ResidentWeightLimit { gpu: 0, required: 2048, limit: 1 });
    }

    /// The header-only planner reserves Spark return planes beside a planned
    /// baseline; the runtime already created SparkLink before sampling free
    /// memory, so the same owners are never charged twice.
    #[test]
    fn spark_intake_is_charged_once_across_planned_and_measured_baselines() {
        let cfg = Qwen4Config::from_hf(&crate::plan::testing::qwen4_config(4)).unwrap();
        let intake = |baseline| {
            let inputs = single_inputs(&cfg, baseline, 4, Some((16, true)), None,
                ResidentWeights(0), None);
            (placement(&inputs).unwrap(), inputs)
        };
        let (planned, inputs) = intake(Baseline::Planned { context_bytes: 0, loaded_bytes: 0 });
        let planes = planned.items[0].iter().filter(|item| item.group == "Spark intake planes")
            .map(|item| item.bytes).collect::<Vec<_>>();
        assert_eq!(planes.len(), 1);
        assert_eq!(planes[0], crate::serving_capacity::prefill_return_bytes_with_format(1, 4096,
            4, cfg.hidden as u64, cuteafd_core::SparkReplyFormat::Bf16).unwrap());
        assert_eq!(planned.pool_tokens, 2 << 20);
        drop(inputs);
        let (runtime, _) = intake(Baseline::Measured { free_bytes: 96 << 30 });
        assert!(!runtime.items[0].iter().any(|item| item.group == "Spark intake planes"));
        assert_eq!(runtime.pool_tokens, planned.pool_tokens);
    }

    /// A lazy diagnostic run keeps the former growth reserve; a serving run
    /// prices the exact owner graph set instead of a pool-dependent ladder.
    #[test]
    fn qwen_pool_demands_match_geometry_and_lazily_loaded_experts() {
        let cfg = Qwen4Config::from_hf(&crate::plan::testing::qwen4_config(4)).unwrap();
        let lazy = placement(&single_inputs(&cfg, Baseline::Measured { free_bytes: 96 << 30 },
            0, None, None, ResidentWeights(0), None)).unwrap();
        assert_eq!(lazy.items[0].iter().filter(|item| item.group == "graph growth")
            .map(|item| item.bytes).collect::<Vec<_>>(), vec![LAZY_GRAPH_BYTES]);
        // An explicit pool below the context is the first typed refusal.
        let short = single_inputs(&cfg, Baseline::Measured { free_bytes: 96 << 30 }, 0, None,
            Some(65536), ResidentWeights(0), None);
        let below = QwenSingle::new(&short).unwrap();
        assert_eq!(below.admit().unwrap_err(),
            Refusal::PoolBelowContext { tokens: 65536, context: 131072 });
    }

    /// The dual route states its cut, arenas and receive-slot hops once; the
    /// executor consumes them and never re-assigns an owner.
    #[test]
    fn dual_arenas_and_owner_graphs_come_from_the_prepared_facts() {
        let cfg = Qwen4Config::from_hf(&crate::plan::testing::qwen4_config(4)).unwrap();
        let weights = vec![1 << 20; 4];
        let extra = vec![Vec::new(); 4];
        let costs = [Bytes2 { resident: 64 << 20, staging: 8 << 20 }; 2];
        let experts: Vec<_> = (0..4).map(|_| Some(ExpertCost { whole: Bytes2 { resident: 0, staging: 0 },
            half: costs, tp2: true, whole_supported: true, spark_ok: true })).collect();
        let gib = 1u64 << 30;
        let build = |startup| QwenDualInputs {
            resident_weight_limit: None, automatic_context: false,
            admission: QwenAdmissionInputs { prefill_shape: None, cfg: &cfg, layers: 4, mtp: false,
                kv_format: Qwen4KvCache::Bf16, manifest: None, prefill_rows: 4096, slots: 16,
                mark_bytes: 0, full_prefill_logits: 0, ple: None, future_expert_bytes: 0, headroom: 0 },
            owner_manifests: [None; 2],
            gpus: [GpuBudget { capacity_bytes: 96 * gib, headroom_bytes: 0,
                baseline: Baseline::Measured { free_bytes: 96 * gib } }; 2],
            layer_weights: &weights, layer_experts: &experts,
            layer_extra: &extra, fixed: Vec::new(),
            tp2_workspace: [1 << 20; 2], pending_code_bytes: [0; 2], mark_slots: 18,
            max_context: 131072, requested_pool: None, spark_ranks: 0, onboard: Onboard::Auto,
            startup_graph_modes: startup,
            host: QwenHost::ample(&cfg, 2, 4096).unwrap(), encoders: Default::default(),
        };
        let lazy = build(None);
        let placement = dual_placement(&lazy).unwrap();
        // One workspace, four halves resident, and the largest staging peak.
        let peak = (1 << 20) + 4 * (64 << 20) + (8 << 20);
        assert_eq!(placement.tp2, Some(Tp2Range { first: 0, layers: 4, peak_bytes: [peak; 2] }));
        for gpu in 0..2 {
            assert_eq!(placement.items[gpu].iter().filter(|item|
                item.group == "TP2 expert layer halves 0..4").map(|item| item.bytes).collect::<Vec<_>>(),
                vec![peak]);
            assert_eq!(placement.items[gpu].iter().filter(|item| item.group == "graph growth")
                .map(|item| item.bytes).collect::<Vec<_>>(), vec![LAZY_GRAPH_BYTES]);
        }
        assert_eq!(placement.layers.iter().map(|layer| layer.experts).collect::<Vec<_>>(),
            vec![ExpertHome::RtxTp2; 4]);
        drop(lazy);
        let serving = build(Some((16, true)));
        let placement = dual_placement(&serving).unwrap();
        let graphs = qwen_startup_graphs_placed(131072, placement.pool_tokens as usize,
            cfg.dense_context(), 16, true, &dual_owners(4)).unwrap();
        assert_eq!(graphs.ranks.len(), 2);
        assert!(graphs.bytes(0) > graphs.bytes(1));
        for gpu in 0..2 {
            assert_eq!(placement.items[gpu].iter().filter(|item| item.group == "startup graphs")
                .map(|item| item.bytes).collect::<Vec<_>>(), vec![graphs.bytes(gpu)]);
        }
    }
}
