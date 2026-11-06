//! Step-5 physical owners for the shared planner: the executor's fixed CSA
//! boundary, exact per-rank native owners and the pool record unit, prepared
//! once. The family states facts; pool, onboarding count and fitting are the
//! shared solver's decisions, and execution consumes the owners it admitted.
use crate::placement::planner::*;
use crate::placement::{
    AttentionClass, AttentionPlacement, Bytes2, Demand, ExpertCost, ExpertHome, ExpertRange, FfnMode,
    HopSpec, HopStorage, Inventory, LayerAssignment, LayerExecutionPlan, LayerMode, Placement,
    PoolLimits, PoolPolicy, ResidentWeights, TerminalHome, Tp2Range,
};
use crate::plan::families::step5::{
    admission::{input_bytes, staging_bytes, MarkRule, OnboardInputs, Reservation, RuntimeInputs},
    native::RankAdmission,
    ownership::ExecutionOwnership,
    Config,
};
use crate::serving_capacity::step5::Step5CacheLayout;
use anyhow::{ensure, Context, Result};
use cuteafd_core::memory_layout::{Basis, Category};

pub const PAGE_ROWS: u64 = 64;
/// The agentic context floor; an explicit pool below it is refused.
const FLOOR: u64 = 262_144;
const WHOLE0: LayerMode = LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner };
const WHOLE1: LayerMode = LayerMode::Whole { gpu: 1, ffn: FfnMode::Owner };
const EXECUTOR: crate::placement::ExecutorModes = crate::placement::families::STEP5;
/// No onboarding profile: the per-layer expert owners are never active.
const NO_EXPERT: ExpertCost = ExpertCost {
    whole: Bytes2 { resident: 0, staging: 0 },
    half: [Bytes2 { resident: 0, staging: 0 }; 2],
    tp2: false,
    whole_supported: true,
    spark_ok: true,
};

/// The family's own figures at one fixed owner choice, before any pool.
#[derive(Clone)]
struct Figures {
    cache: Step5CacheLayout,
    marks: u64,
    /// Every non-pool, non-hop owner of each rank, in the plan's group names.
    fixed: Vec<Demand>,
    /// Residual receive slots, charged by the shared solver apart from `fixed`.
    hops: Vec<u64>,
}

impl Figures {
    fn total(&self, rank: u8, pages: u64) -> Result<u64> {
        let fixed = self
            .fixed
            .iter()
            .filter(|demand| demand.gpu == rank)
            .try_fold(0u64, |sum, demand| sum.checked_add(demand.bytes))
            .context("Step-5 rank owner total overflow")?;
        let unit = self
            .cache
            .geometry
            .ranks
            .get(usize::from(rank))
            .context("Step-5 pool geometry rank")?
            .persistent_unit_bytes;
        let pool = unit
            .checked_mul(pages)
            .context("Step-5 pool record total overflow")?;
        fixed.checked_add(pool).and_then(|bytes| bytes.checked_add(self.hops[usize::from(rank)]))
            .context("Step-5 rank total overflow")
    }
}

fn checked_sum(values: impl IntoIterator<Item = u64>) -> Result<u64> {
    values
        .into_iter()
        .try_fold(0u64, |n, v| n.checked_add(v))
        .context("Step-5 admission sum overflow")
}

fn exact(fixed: &mut Vec<Demand>, rank: usize, category: Category, group: &str, bytes: u64) {
    fixed.push(Demand::new(rank as u8, category, group, bytes, Basis::Exact));
}

fn scaled(bytes: u64, count: usize) -> Result<u64> {
    bytes
        .checked_mul(u64::try_from(count)?)
        .context("Step-5 admission product overflow")
}

/// One prepared candidate: the facts the shared planner prices.
pub struct Step5Recipe<'a> {
    pub cfg: &'a Config,
    pub inputs: &'a RuntimeInputs,
    pub depth: usize,
    pub gpus: usize,
    pub page_rows: u64,
    pub maximum_pages: u64,
    /// The executor's fixed attention ownership: head-split SWA and one
    /// contiguous CSA boundary per GPU (every layer on one GPU).
    pub attention: Vec<LayerMode>,
    pub prefill: PrefillShape,
    pub lanes: u64,
    /// Routed layers the RTX may onboard, in execution order.
    pub routes: Vec<usize>,
    /// Ordered onboarding counts, `0..=routes.len()`.
    pub counts: Vec<usize>,
    /// Destination GPUs of one onboarded routed layer.
    pub destination: Vec<u8>,
    /// Per-rank expert arena workspace, charged once a layer is onboarded.
    pub arena: Vec<u64>,
    pub expert: Vec<ExpertCost>,
    pub marks: u64,
    pub host_geometry: cuteafd_core::prefix::HostCacheGeometry,
    pub host_cache_bytes: u64,
    figures: Figures,
    /// Owners before onboarding: the executor's native binding.
    pub ownership: ExecutionOwnership,
}

impl<'a> Step5Recipe<'a> {
    /// Binds the native owners for every candidate CSA boundary, then keeps
    /// the boundary whose max per-rank total at the pool ceiling is smallest.
    /// The native bytes come from the same callback execution uses; this is
    /// one family fact, not a second admission.
    pub fn new(
        cfg: &'a Config,
        inputs: &'a RuntimeInputs,
        depth: usize,
        mut native: impl FnMut(&ExecutionOwnership) -> Result<Vec<RankAdmission>>,
    ) -> Result<Self> {
        inputs.validate(cfg)?;
        ensure!(depth <= cfg.mtp, "Step-5 forecast predictor depth exceeds checkpoint");
        let gpus = inputs.inventory.gpus.len();
        let capacities = inputs
            .inventory
            .gpus
            .iter()
            .map(|gpu| gpu.capacity_bytes)
            .collect::<Vec<_>>();
        let maximum_pages = PoolPolicy::resolve(
            &capacities,
            inputs.context,
            inputs.requested_pool,
            PAGE_ROWS,
            false,
        )
        .maximum_tokens()
        .div_ceil(PAGE_ROWS);
        ensure!(maximum_pages > 0, "Step-5 empty pool ceiling");
        let prefill = PrefillShape {
            rows: inputs.unit_rows() as u64,
            waves: inputs.waves as u64,
        };
        let lanes = inputs.waves as u64;
        let mut best: Option<(u64, Vec<LayerMode>, ExecutionOwnership, Figures)> = None;
        for boundary in Self::boundaries(cfg, gpus) {
            let attention = Self::modes(cfg, gpus, boundary);
            let ownership = ExecutionOwnership::for_modes(cfg, gpus, &attention, inputs.hops())?;
            let admitted = native(&ownership)?;
            let figures = Self::figures(cfg, inputs, depth, &ownership, &admitted)?;
            let score = (0..gpus as u8)
                .map(|rank| figures.total(rank, maximum_pages))
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .max()
                .unwrap_or(0);
            if best.as_ref().is_none_or(|(old, ..)| score < *old) {
                best = Some((score, attention, ownership, figures));
            }
        }
        let (_, attention, ownership, figures) =
            best.context("Step-5 has no legal attention owner candidate")?;
        let routes = (0..cfg.layers).filter(|&layer| cfg.moe[layer]).collect::<Vec<_>>();
        let counts = (0..=routes.len()).collect::<Vec<_>>();
        let destination = if gpus == 2 { vec![0, 1] } else { vec![0] };
        let arena = inputs.onboard.as_ref().map_or(vec![0; gpus], |onboard| {
            if gpus == 2 {
                onboard.tp2_workspace.to_vec()
            } else {
                vec![onboard.workspace]
            }
        });
        let expert = inputs
            .onboard
            .as_ref()
            .map_or_else(Vec::new, |onboard| onboard.costs.clone());
        let host_geometry = cuteafd_core::prefix::HostCacheGeometry {
            unit_rows: PAGE_ROWS,
            page_bytes: checked_sum(figures.cache.geometry.ranks.iter().map(|rank| rank.persistent_unit_bytes))?,
            mark_bytes: checked_sum(figures.cache.ranks.iter().map(|rank| rank.bytes))?,
            draft_bytes: 0,
        };
        let host_cache_bytes = if !inputs.marks.host || inputs.marks.entries == 0 { 0 } else {
            host_geometry.resolve(inputs.host.cache_bytes, inputs.context, inputs.sequences as u64,
                inputs.host.memory.0, host_geometry.chunk_bytes()).context("Step-5 host prefix geometry")?
        };
        Ok(Self {
            host_geometry, host_cache_bytes,
            cfg,
            inputs,
            depth,
            gpus,
            page_rows: PAGE_ROWS,
            maximum_pages,
            attention,
            prefill,
            lanes,
            routes,
            counts,
            destination,
            arena,
            expert,
            marks: figures.marks,
            figures,
            ownership,
        })
    }

    /// CSA boundaries are layer indexes: every layer past one belongs to GPU1.
    /// SWA layers are always head-split, so only a CSA layer can end a range.
    fn boundaries(cfg: &Config, gpus: usize) -> Vec<usize> {
        if gpus == 1 {
            return vec![cfg.layers];
        }
        let groups = cfg.full[..cfg.layers]
            .iter()
            .map(|&full| (full, None))
            .collect::<Vec<_>>();
        crate::placement::attention::owner_boundaries(&groups, false)
    }

    fn modes(cfg: &Config, gpus: usize, boundary: usize) -> Vec<LayerMode> {
        (0..cfg.layers)
            .map(|layer| {
                if gpus == 1 {
                    WHOLE0
                } else if cfg.full[layer] {
                    if layer < boundary {
                        WHOLE0
                    } else {
                        WHOLE1
                    }
                } else {
                    LayerMode::HeadSplit
                }
            })
            .collect()
    }

    /// Exact per-rank owners at one fixed attention choice. Nothing here is a
    /// free-memory total: each figure is a native/layout owner the executor
    /// allocates, and the pool enters as its own per-page record unit.
    fn figures(
        cfg: &Config,
        inputs: &RuntimeInputs,
        depth: usize,
        ownership: &ExecutionOwnership,
        native: &[RankAdmission],
    ) -> Result<Figures> {
        let ranks = ownership.ranks();
        ensure!(native.len() == ranks, "Step-5 native admission rank inventory");
        let cache = Step5CacheLayout::for_execution(
            cfg,
            ownership,
            inputs.context,
            depth,
            inputs.unit_rows() as u64,
            inputs.rows as u64,
            inputs.kv_format,
        )?;
        let mark_bytes = usize::try_from(checked_sum(cache.ranks.iter().map(|r| r.bytes))?)?;
        let marks = inputs.marks.slots(mark_bytes) as u64;
        let hops = crate::placement::hop_buffer_bytes(
            &crate::placement::plan_hops(ownership.execution().modes(), &inputs.hops())?,
            &inputs.hops(),
            ranks,
        )
        .context("Step-5 residual hop storage")?;
        let mut fixed = Vec::new();
        for rank in 0..ranks {
            let reservation =
                Reservation::new(&cache, rank, inputs.sequences as u64, marks, 1, usize::MAX)?;
            fixed.push(
                Demand::new(rank as u8, Category::Weights, "Step-5 native target and head",
                    native[rank].immutable, Basis::Exact)
                    .with_resident_weights(native[rank].resident_weights),
            );
            if rank == 0 {
                let bytes = scaled(4096 * 2, cfg.vocab)?;
                fixed.push(
                    Demand::new(0, Category::Embedding, "Step-5 GPU embedding", bytes,
                        Basis::Exact)
                        .with_resident_weights(ResidentWeights(bytes)),
                );
            }
            exact(&mut fixed, rank, Category::Workspace, "Step-5 native workspace", native[rank].workspace);
            exact(&mut fixed, rank, Category::Kv, "Step-5 active sequence state", reservation.active as u64);
            exact(&mut fixed, rank, Category::Prefix, "Step-5 exact retained marks", reservation.marks as u64);
            exact(&mut fixed, rank, Category::Workspace, "Step-5 private transaction staging",
                staging_bytes(&cache, rank, inputs.sequences)? as u64);
            exact(&mut fixed, rank, Category::Runtime, "Step-5 pending code", inputs.pending_code[rank]);
            if rank == 0 {
                exact(&mut fixed, rank, Category::Workspace, "Step-5 input", input_bytes(inputs.rows)?);
                let capacity = inputs.rows.min(64);
                exact(&mut fixed, rank, Category::Workspace, "Step-5 selector and device sampler",
                    checked_sum([
                        scaled(12, capacity)?,
                        cuteafd_core::target_sampler_device_bytes(capacity, cfg.vocab) as u64,
                    ])?);
            }
            if rank == usize::from(ownership.execution().spec().entry_gpu) {
                exact(&mut fixed, rank, Category::Transport, "Step-5 Spark intake planes",
                    crate::serving_capacity::prefill_return_bytes_with_format(1, inputs.rows as u64,
                        inputs.inventory.spark_ranks as u64, 4096, inputs.spark_reply_format)?
                        .max(256 * inputs.inventory.spark_ranks as u64));
                exact(&mut fixed, rank, Category::Workspace, "Step-5 routed reply", (inputs.rows as u64 * 4096 * 2).max(256));
            }
            if ranks == 2 {
                let progress = if inputs.peer_diagnostics { (9 * 48).max(256) as u64 } else { 0 };
                // Eight peer receive slots, peer control, abort word and one
                // private hop control plane; hop receives are charged apart.
                exact(&mut fixed, rank, Category::Transport, "Step-5 private peer and hop control",
                    checked_sum([
                        8 * (inputs.unit_rows() as u64 * 4096 * 4).max(256),
                        256,
                        256,
                        256,
                        progress,
                    ])?);
            }
        }
        Ok(Figures { cache, marks, fixed, hops })
    }

    /// The live/offline request: explicit CLI values stay strict, zero and
    /// omission both select automatic admission.
    pub fn request(&self) -> Request {
        Request {
            context: self.inputs.context,
            pool: PoolPolicy::normalize_request(self.inputs.requested_pool)
                .map_or(Pin::Auto, Pin::Exact),
            onboard: match self.inputs.onboard.as_ref().map(|onboard| onboard.mode) {
                None | Some(crate::placement::Onboard::Auto) => Pin::Auto,
                Some(crate::placement::Onboard::Layers(count)) => Pin::Exact(count as u64),
            },
            prefill: Pin::Exact(self.prefill),
            operation: Operation::Serving,
            resident_weight_limit: self.inputs.resident_weight_limit,
            spark_reply_formats: Default::default(),
        }
    }

    /// Charges the expert owners an ordered onboarding prefix holds: one
    /// cumulative owner set per count option, so option k prices layers[..k],
    /// plus one arena plane while any layer is onboarded.
    fn expert_components(
        &self,
        contract: &mut Contract,
        onboard: Option<usize>,
    ) -> Result<(), Refusal> {
        let Some(onboard) = onboard else { return Ok(()) };
        let mut sums = vec![0u64; self.gpus];
        for (position, &layer) in self.routes.iter().enumerate() {
            let cost = self.expert.get(position).copied().unwrap_or(NO_EXPERT);
            for gpu in 0..self.gpus {
                let part = if self.gpus == 2 {
                    cost.half[gpu].resident
                } else {
                    cost.whole.resident
                };
                sums[gpu] = sums[gpu]
                    .checked_add(part)
                    .ok_or(Refusal::Overflow("Step-5 onboarded expert sum"))?;
            }
            for gpu in 0..self.gpus {
                let bytes = sums[gpu];
                if bytes == 0 {
                    continue;
                }
                let owner = Allocation::buffer(
                    format!("step5.expert.{layer}.{gpu}"),
                    Location::coordinator(gpu as u8),
                    bytes,
                    Lifetime::RESIDENT,
                    Category::Experts,
                    "Step-5 onboarded routed expert",
                );
                for component in [
                    Component::resident_weight(
                        ResidentWeight::from_allocation(&owner, ResidentWeights(bytes)),
                        ByteCurve::fixed(bytes, self.maximum_pages),
                    ),
                    Component::allocation(
                        owner,
                        ByteCurve::fixed(bytes, self.maximum_pages),
                        false,
                        self.maximum_pages,
                    ),
                ] {
                    contract.components.push(Component {
                        activation: Activation::choice(onboard, position + 1),
                        ..component
                    });
                }
            }
        }
        // The arena plane is held by every count option that onboards a layer.
        let clauses = (1..=self.routes.len())
            .map(|option| vec![Choice { variable: onboard, option }])
            .collect::<Vec<_>>();
        for gpu in 0..self.gpus {
            let bytes = self.arena[gpu];
            if bytes == 0 {
                continue;
            }
            let owner = Allocation::buffer(
                format!("step5.expert.arena.{gpu}"),
                Location::coordinator(gpu as u8),
                bytes,
                Lifetime::RESIDENT,
                Category::Experts,
                "Step-5 RTX expert arena",
            );
            contract.components.push(Component {
                activation: Activation { clauses: clauses.clone() },
                ..Component::allocation(
                    owner,
                    ByteCurve::fixed(bytes, self.maximum_pages),
                    false,
                    self.maximum_pages,
                )
            });
        }
        Ok(())
    }
}

impl Recipe for Step5Recipe<'_> {
    fn prepare(&self, request: &Request) -> Result<Contract, Refusal> {
        if self.gpus != self.inputs.inventory.gpus.len()
            || !(1..=2).contains(&self.gpus)
            || self.attention.len() != self.cfg.layers
            || self.maximum_pages == 0
            || self.maximum_pages == u64::MAX
            || request.context != self.inputs.context
            || request.context < FLOOR
            || request.context > self.cfg.max_context
            || self.counts.len() != self.routes.len() + 1
            || self.destination.is_empty()
            || self.cfg.layers != 92
        {
            return Err(Refusal::InvalidFact("Step-5 invalid contract extent".into()));
        }
        let mut contract = Contract {
            phase_order: PhaseOrder::TARGET_THEN_DRAFT,
            spark_reply_formats: None,
            resident_weight_limit: request.resident_weight_limit,
            page_rows: self.page_rows,
            maximum_pages: self.maximum_pages,
            gpus: self.gpus,
            variables: vec![Variable::Prefill(vec![self.prefill])],
            requires: vec![],
            components: vec![],
        };
        contract.variables.push(Variable::Lanes(vec![self.lanes]));
        contract
            .variables
            .push(Variable::Embedding(vec![EmbeddingHome::Device(0)]));
        // Without an onboarding profile the domain is a single zero: nothing
        // may be onboarded, and the per-layer expert owners are never active.
        let onboard = self.inputs.onboard.as_ref().map(|_| {
            let index = contract.variables.len();
            contract.variables.push(Variable::Onboard {
                layers: self.routes.clone(),
                destination: self.destination.clone(),
                counts: self.counts.clone(),
            });
            index
        });
        contract.variables.push(Variable::ExpertHome {
            layers: self.routes.clone(),
            rank_sets: vec![(0..self.inputs.inventory.spark_ranks as u8).collect()],
        });
        for demand in &self.figures.fixed {
            let location = Location::coordinator(demand.gpu);
            let owner = Allocation::buffer(
                format!("step5.owner.{}.{}", demand.gpu, demand.group.replace(' ', ".")),
                location,
                demand.bytes,
                Lifetime::RESIDENT,
                demand.category,
                demand.group.clone(),
            );
            if demand.resident_weights.0 > 0 {
                contract.components.push(Component::resident_weight(
                    ResidentWeight::from_allocation(&owner, demand.resident_weights),
                    ByteCurve::fixed(demand.resident_weights.0, self.maximum_pages),
                ));
            }
            contract.components.push(Component::allocation(owner,
                ByteCurve::fixed(demand.bytes, self.maximum_pages), false, self.maximum_pages));
        }
        for rank in 0..self.gpus {
            let unit = self
                .figures
                .cache
                .geometry
                .ranks
                .get(rank)
                .ok_or_else(|| Refusal::InvalidFact("Step-5 pool geometry rank".into()))?
                .persistent_unit_bytes;
            let owner = Allocation::buffer(format!("step5.records.{rank}"),
                Location::coordinator(rank as u8), unit, Lifetime::RESIDENT, Category::Kv, "records");
            contract.components.push(Component::allocation(owner,
                ByteCurve::linear(unit, unit, self.maximum_pages), false, self.maximum_pages));
        }
        for (rank, &bytes) in self.figures.hops.iter().enumerate() {
            if bytes == 0 {
                continue;
            }
            let owner = Allocation::buffer(format!("step5.hops.{rank}"),
                Location::coordinator(rank as u8), bytes, Lifetime::RESIDENT, Category::Transport,
                "residual hops");
            contract.components.push(Component::allocation(owner,
                ByteCurve::fixed(bytes, self.maximum_pages), false, self.maximum_pages));
        }
        for (id, location, category, bytes, lifetime) in [
            ("prefix", Location::HOST, Category::Prefix, self.host_cache_bytes,
                Lifetime { first: Phase::Ready, last: Phase::Growth }),
            ("logits", Location::HOST, Category::Staging,
                cuteafd_core::memory_layout::host_logits_bytes(self.inputs.rows.min(64) as u64,
                    if self.inputs.full_prefill_logits { self.inputs.rows as u64 } else { self.inputs.rows.min(64) as u64 },
                    self.cfg.vocab as u64).ok_or(Refusal::Overflow("Step-5 host logits"))?,
                Lifetime { first: Phase::Ready, last: Phase::Growth }),
        ] {
            let owner = Allocation::buffer(format!("step5.host.{id}"), location, bytes, lifetime,
                category, format!("Step-5 host {id}"));
            contract.components.push(Component::allocation(owner,
                ByteCurve::fixed(bytes, self.maximum_pages), true, self.maximum_pages));
        }
        let transport = crate::serving_capacity::spark_host_transport(cuteafd_core::ExpertGeometry::STEP5,
            1, self.inputs.inventory.spark_ranks, false, false, false,
            self.inputs.unit_rows() as u64, self.inputs.unit_rows() as u64, self.inputs.waves as u64, None)
            .map_err(|error| Refusal::InvalidFact(error.to_string()))?;
        for (kind, bytes) in ["rings", "egress", "intake"].into_iter().zip(transport) {
            let owner = Allocation::buffer(format!("step5.host.transport.{kind}"), Location::PINNED,
                bytes, Lifetime::RESIDENT, Category::Transport, format!("Step-5 host transport {kind}"));
            contract.components.push(Component::allocation(owner,
                ByteCurve::fixed(bytes, self.maximum_pages), false, self.maximum_pages));
        }
        if let Some(push) = &self.inputs.host.push {
            for owner in push.allocations("step5.push")? {
                contract.components.push(Component::allocation(owner.clone(),
                    ByteCurve::fixed(owner.bytes, self.maximum_pages), false, self.maximum_pages));
            }
        }
        self.expert_components(&mut contract, onboard)?;
        Ok(contract)
    }
}

/// One physical budget per coordinator and Spark rank. Offline keeps its
/// predicted CUDA/code floor; live requires the measured pre-weight sample.
pub fn budgets(inputs: &RuntimeInputs) -> Result<Vec<Budget>, Refusal> {
    let mut budgets = Vec::new();
    for (rank, gpu) in inputs.inventory.gpus.iter().enumerate() {
        let baseline = match gpu.baseline {
            crate::placement::Baseline::Measured { free_bytes } => Baseline::Sampled {
                free: free_bytes,
                phase: Phase::Load,
                source: "pre-weight CUDA free memory".into(),
            },
            crate::placement::Baseline::Planned { context_bytes, loaded_bytes } => {
                Baseline::Forecast {
                    occupied: context_bytes
                        .checked_add(loaded_bytes)
                        .ok_or(Refusal::Overflow("Step-5 planned baseline"))?,
                    source: "offline CUDA context and loading forecast".into(),
                    assumed: true,
                }
            }
        };
        budgets.push(Budget {
            id: format!("rtx{rank}"),
            locations: vec![Location::coordinator(rank as u8)],
            physical_bytes: gpu.capacity_bytes,
            logical_bytes: gpu.capacity_bytes,
            baseline,
            headroom: gpu.headroom_bytes,
        });
    }
    let bytes = crate::plan::DEFAULT_SPARK_BUDGET_BYTES;
    for rank in 0..inputs.inventory.spark_ranks {
        budgets.push(Budget {
            id: format!("spark{rank}"),
            locations: vec![Location {
                device: Device::Spark(rank as u8),
                domain: Domain::Managed,
            }],
            physical_bytes: bytes,
            logical_bytes: bytes,
            baseline: Baseline::Forecast {
                occupied: 0,
                source: "Spark target weights outlive each request".into(),
                assumed: true,
            },
            headroom: 0,
        });
    }
    let (total, available) = inputs.host.memory;
    budgets.push(Budget { id: "host".into(), locations: vec![Location::HOST, Location::PINNED],
        physical_bytes: total, logical_bytes: total,
        baseline: Baseline::Forecast { occupied: total.saturating_sub(available),
            source: "Step-5 pre-admission host MemAvailable".into(), assumed: false },
        headroom: inputs.host.headroom.max(total / 10) });
    Ok(budgets)
}

/// The solver's admitted owners, bound to the executor's placement.
#[derive(Debug, Clone)]
pub struct Admission {
    pub placement: Placement,
    pub ownership: ExecutionOwnership,
    pub cache: Step5CacheLayout,
    pub marks: usize,
    pub reservations: Vec<Reservation>,
    /// Exact fixed demands without the pool records.
    pub fixed: Vec<Demand>,
    pub host_geometry: cuteafd_core::prefix::HostCacheGeometry,
    pub host_cache_bytes: u64,
    pub solved: ResolvedPlan,
}

impl Step5Recipe<'_> {
    pub fn execution(&self, solved: &ResolvedPlan) -> Result<Admission> {
        ensure!(
            solved.decisions.prefill == self.prefill
                && solved.decisions.lanes == self.lanes
                && solved.decisions.embedding == Some(EmbeddingHome::Device(0))
                && solved.decisions.attention.is_empty()
                && solved.pool_tokens % self.page_rows == 0
                && solved.pool_tokens >= self.inputs.context,
            "Step-5 resolved execution differs from prepared geometry"
        );
        let onboarded = solved.decisions.onboard_layers.first().cloned().unwrap_or_default();
        ensure!(
            solved.decisions.onboard_layers.iter().all(|layers| *layers == onboarded)
                && onboarded.iter().copied().eq(self.routes.iter().copied().take(onboarded.len()))
                && (self.inputs.onboard.is_some() || onboarded.is_empty()),
            "Step-5 resolved onboarding differs from the prepared profile"
        );
        let layers = self
            .attention
            .iter()
            .enumerate()
            .map(|(layer, &mode)| LayerAssignment {
                mode,
                experts: if !self.cfg.moe[layer] {
                    ExpertHome::Dense
                } else if onboarded.contains(&layer) {
                    if self.gpus == 2 {
                        ExpertHome::RtxTp2
                    } else {
                        ExpertHome::RtxWhole { gpu: 0 }
                    }
                } else {
                    ExpertHome::Spark
                },
            })
            .collect::<Vec<_>>();
        let mut arena = vec![0u64; self.gpus];
        if !onboarded.is_empty() {
            for gpu in 0..self.gpus {
                arena[gpu] = self.arena[gpu];
            }
            for &layer in &onboarded {
                let position = self
                    .routes
                    .iter()
                    .position(|&routed| routed == layer)
                    .context("Step-5 onboarded layer is not routed")?;
                let cost = self.expert.get(position).copied().unwrap_or(NO_EXPERT);
                for gpu in 0..self.gpus {
                    let part = if self.gpus == 2 {
                        cost.half[gpu].resident
                    } else {
                        cost.whole.resident
                    };
                    arena[gpu] = arena[gpu]
                        .checked_add(part)
                        .context("Step-5 expert arena overflow")?;
                }
            }
        }
        let first = self.routes.first().copied().unwrap_or(0);
        let mut expert_ranges = (0..self.gpus)
            .map(|gpu| ExpertRange { first, layers: 0, peak_bytes: arena[gpu] })
            .collect::<Vec<_>>();
        let tp2 = if self.gpus == 2 && !onboarded.is_empty() {
            Some(Tp2Range { first, layers: onboarded.len(), peak_bytes: [arena[0], arena[1]] })
        } else {
            expert_ranges[0].layers = onboarded.len();
            None
        };
        let spec = self.inputs.hops();
        let placement = Placement {
            pool_tokens: solved.pool_tokens,
            pool_limits: PoolLimits(vec![]),
            attention_placement: AttentionPlacement::Layers,
            attention_by_kind: if self.gpus == 2 {
                vec![
                    (AttentionClass::Csa, AttentionPlacement::Layers),
                    (AttentionClass::Swa, AttentionPlacement::Heads),
                ]
            } else {
                vec![]
            },
            peer_row_bytes: 0,
            onboard_layers: onboarded.len(),
            max_pool_tokens_no_onboard: None,
            layers,
            movables: vec![],
            expert_ranges,
            residual: crate::placement::residual_homes(&self.attention, &spec)?,
            hops: crate::placement::plan_hops(&self.attention, &spec)?,
            tp2,
            items: (0..self.gpus)
                .map(|rank| solved.items(Location::coordinator(rank as u8)))
                .collect::<Result<Vec<_>, _>>()?,
        };
        EXECUTOR.check(&placement)?;
        let ownership = ExecutionOwnership::new(
            self.cfg,
            self.gpus,
            LayerExecutionPlan::new(&placement, spec)?,
        )?
        .with_onboarded(onboarded)?;
        let pages = solved.pool_tokens / self.page_rows;
        ensure!(pages <= u32::MAX as u64, "Step-5 pool page identifiers overflow");
        let reservations = (0..self.gpus)
            .map(|rank| Reservation::new(&self.figures.cache, rank, self.inputs.sequences as u64,
                self.marks, pages, usize::MAX))
            .collect::<Result<Vec<_>>>()?;
        Ok(Admission {
            placement,
            ownership,
            cache: self.figures.cache.clone(),
            marks: usize::try_from(self.marks)?,
            reservations,
            fixed: self.figures.fixed.clone(),
            host_geometry: self.host_geometry,
            host_cache_bytes: self.host_cache_bytes,
            solved: solved.clone(),
        })
    }
}
