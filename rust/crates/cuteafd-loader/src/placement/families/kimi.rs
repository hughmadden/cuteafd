//! Kimi K3 physical owners for the shared planner. Callers resolve checkpoint,
//! executor and retention facts once; `planner::solve` is the only admission,
//! and both `cuteafd plan --layout` and live serve admission consume the same
//! facts. Nothing here samples free memory or runs a second fit loop.
use super::super::planner::*;
use super::super::{
    hop_buffer_bytes, plan_hops, residual_homes, AttentionClass, AttentionPlacement, Bytes2,
    Demand, ExpertHome, ExpertRange, HopSpec, LayerAssignment, LayerMode, ModeBytes, MovableId,
    Placement, PoolLimits, ResidentWeights, Tp2Range,
};
use crate::formats::dflash::MlaDraftRuntimeLayout;
use cuteafd_core::memory_layout::Category;
use cuteafd_core::spark_reply::SparkReplyFormats;

/// Native page quantum of the retained MLA records.
pub const PAGE_ROWS: u64 = 64;

fn invalid(detail: &str) -> Refusal {
    Refusal::InvalidFact(format!("Kimi component contract: {detail}"))
}

/// Attention weight bytes one rank retains for a layer's resolved mode.
fn attention_bytes(weights: &ModeBytes, mode: LayerMode, rank: usize) -> u64 {
    match mode {
        LayerMode::HeadSplit | LayerMode::ContextSplit => weights.split[rank],
        LayerMode::Whole { gpu, .. } if usize::from(gpu) == rank => weights.whole,
        LayerMode::Whole { .. } => 0,
    }
}

/// One rank-fixed owner. `weights` is the retained-weight part of the same
/// exact allocation, never a second charge.
#[derive(Debug, Clone)]
pub struct FixedOwner {
    pub rank: u8,
    pub category: Category,
    pub group: String,
    pub bytes: u64,
    pub weights: u64,
    pub lifetime: Lifetime,
}

impl FixedOwner {
    pub fn ready(demand: &Demand) -> Self {
        Self {
            rank: demand.gpu,
            category: demand.category,
            group: demand.group.clone(),
            bytes: demand.bytes,
            weights: demand.resident_weights.0,
            lifetime: Lifetime::RESIDENT,
        }
    }
}

/// One GPU's local expert arena at one admitted onboard count. The peak
/// includes the transient staging the loader needs above the resident halves.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Arena {
    pub count: usize,
    pub peak: Vec<u64>,
    pub weights: Vec<u64>,
}

/// One Spark rank's retained work at one admitted onboard count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SparkOwner {
    pub count: usize,
    pub rank: usize,
    pub experts: u64,
    pub overhead: u64,
}

/// One drafter owner whose extent grows with the retained pool.
#[derive(Debug, Clone)]
pub struct DraftOwner {
    pub category: Category,
    pub group: String,
    pub bytes_at_first: u64,
    pub bytes_per_page: u64,
    pub weights_at_first: u64,
    pub weights_per_page: u64,
}

/// Drafter owners plus the transient load peak that overlaps them.
#[derive(Debug, Clone)]
pub struct Draft {
    pub owners: Vec<DraftOwner>,
    pub peak_at_first: u64,
    pub peak_per_page: u64,
}

/// The declared local expert prefix domain. `destination` names the
/// coordinator GPUs that hold a layer; TP2 names both and counts it once.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertProfile {
    pub layers: Vec<usize>,
    pub destination: Vec<u8>,
    pub counts: Vec<usize>,
    pub rank_sets: Vec<Vec<u8>>,
}

/// Retained routed-expert halves per routed layer for TP2.
pub fn tp2_arenas(counts: &[usize], workspace: [u64; 2], halves: &[[Bytes2; 2]])
    -> Result<Vec<Arena>, Refusal> {
    counts.iter().map(|&count| {
        if count > halves.len() {
            return Err(invalid("arena count exceeds the routed inventory"));
        }
        let mut arena = Arena { count, peak: vec![0; 2], weights: vec![0; 2] };
        for gpu in 0..2 {
            let resident = halves[..count].iter().try_fold(workspace[gpu], |sum, pair|
                sum.checked_add(pair[gpu].resident).ok_or(Refusal::Overflow("Kimi TP2 expert arena")))?;
            let staging = halves[..count].iter().map(|pair| pair[gpu].staging).max().unwrap_or(0);
            arena.peak[gpu] = resident.checked_add(staging)
                .ok_or(Refusal::Overflow("Kimi TP2 expert staging"))?;
            arena.weights[gpu] = resident.checked_sub(workspace[gpu])
                .ok_or(Refusal::Overflow("Kimi TP2 resident weights"))?;
        }
        Ok(arena)
    }).collect()
}

/// Whole-layer arenas: parts load in order into the same workspace.
pub fn whole_arenas(counts: &[usize], workspace: u64, parts: &[Bytes2])
    -> Result<Vec<Arena>, Refusal> {
    counts.iter().map(|&count| {
        if count > parts.len() {
            return Err(invalid("arena count exceeds the routed inventory"));
        }
        let (mut resident, mut peak) = (workspace, workspace);
        for part in &parts[..count] {
            peak = peak.max(resident
                .checked_add(part.resident)
                .and_then(|n| n.checked_add(part.staging))
                .ok_or(Refusal::Overflow("Kimi expert arena"))?);
            resident = resident.checked_add(part.resident)
                .ok_or(Refusal::Overflow("Kimi expert arena"))?;
        }
        Ok(Arena { count, peak: vec![peak], weights: vec![resident - workspace] })
    }).collect()
}

/// Derive the drafter's pool-dependent owners from the admitted layout. The
/// native recipe is linear in whole pages; a non-linear recipe is refused
/// instead of sampled.
pub fn draft_facts(layout: &MlaDraftRuntimeLayout, page_rows: u64, maximum_pages: u64, gpu: u8)
    -> anyhow::Result<Draft> {
    use anyhow::Context;
    let at = |pages: u64| -> anyhow::Result<(Vec<(String, Category, u64, u64)>, u64)> {
        let tokens = page_rows.checked_mul(pages).context("Kimi draft pool overflow")?;
        let layout = layout.with_pool_tokens(usize::try_from(tokens).context("Kimi draft pool extent")?)?;
        let (demand, peak) = layout.demands(gpu)?;
        Ok((demand.owned.into_iter().map(|d| (d.group, d.category, d.bytes, d.resident_weights.0)).collect(),
            peak.bytes))
    };
    let (one, peak_one) = at(1)?;
    let (last, peak_last) = at(maximum_pages)?;
    let step = |first: u64, end: u64| -> anyhow::Result<u64> {
        if maximum_pages == 1 { return Ok(0); }
        let total = end.checked_sub(first).context("Kimi draft owner shrinks with the pool")?;
        anyhow::ensure!(total % (maximum_pages - 1) == 0, "Kimi draft owner is not linear in pool pages");
        Ok(total / (maximum_pages - 1))
    };
    if one.len() != last.len() || one.iter().zip(&last).any(|(a, b)| a.0 != b.0 || a.1 != b.1) {
        return Err(anyhow::anyhow!("Kimi draft owner set depends on the pool extent"));
    }
    let owners = one.into_iter().zip(last).map(|((group, category, bytes, weights), (_, _, end, end_weights))| {
        Ok(DraftOwner { category, group,
            bytes_at_first: bytes, bytes_per_page: step(bytes, end)?,
            weights_at_first: weights, weights_per_page: step(weights, end_weights)? })
    }).collect::<anyhow::Result<Vec<_>>>()?;
    Ok(Draft { owners, peak_at_first: peak_one, peak_per_page: step(peak_one, peak_last)? })
}

/// Kimi's physical owners and finite domains. Every field is a resolved fact:
/// attention modes, retention, expert inventory and the transient load peaks.
#[derive(Debug, Clone)]
pub struct KimiRecipe {
    pub context: u64,
    pub gpus: usize,
    /// Executor attention placement this recipe was prepared for.
    pub placement: AttentionPlacement,
    /// One resolved mode per layer, in order.
    pub attention: Vec<LayerMode>,
    pub prefill: PrefillShape,
    pub lanes: u64,
    pub spark_links: bool,
    pub hop_spec: HopSpec,
    /// Context exchange bytes per row, each way, under context placement.
    pub peer_row_bytes: u64,
    /// Biggest pool the build can index, in pages.
    pub maximum_pages: u64,
    /// Bytes of retained MLA records one page costs each rank.
    pub records_per_page: Vec<u64>,
    /// Attention weight bytes per rank for each layer.
    pub layer_attention: Vec<ModeBytes>,
    pub first_moe: usize,
    /// Attention classes the checkpoint runs, in layer order.
    pub kinds: Vec<AttentionClass>,
    pub experts: Option<ExpertProfile>,
    pub arenas: Vec<Arena>,
    pub spark: Vec<SparkOwner>,
    pub draft: Option<Draft>,
    pub fixed: Vec<FixedOwner>,
}

#[derive(Debug, Clone)]
pub struct AdmittedKimi {
    pub placement: Placement,
    pub solved: ResolvedPlan,
    pub onboard: usize,
}

impl KimiRecipe {
    /// Context placement stores half of every page pair on each rank.
    pub fn page_rows(&self) -> u64 {
        if self.attention.iter().any(|mode| *mode == LayerMode::ContextSplit) {
            PAGE_ROWS * 2
        } else {
            PAGE_ROWS
        }
    }

    pub fn admit(&self, request: Request, budgets: Vec<Budget>) -> Result<AdmittedKimi, Refusal> {
        let solved = solve(request, budgets, self)?;
        let placement = self.execution(&solved)?;
        let onboard = solved.decisions.onboard_layers.iter().map(Vec::len).max().unwrap_or(0);
        Ok(AdmittedKimi { placement, solved, onboard })
    }

    /// Mechanical executor output. Every byte an executor check names comes
    /// from the admitted owners; nothing is re-sampled here.
    pub fn execution(&self, solved: &ResolvedPlan) -> Result<Placement, Refusal> {
        let modes = &solved.decisions.attention;
        if modes != &self.attention
            || solved.decisions.prefill != self.prefill
            || solved.decisions.lanes != self.lanes
            || solved.decisions.onboard_layers.len() != self.gpus
        {
            return Err(invalid("resolved execution differs from the prepared facts"));
        }
        let count = solved.decisions.onboard_layers.first().map(Vec::len).unwrap_or(0);
        if solved.decisions.onboard_layers.iter().any(|layers| layers.len() != count) {
            return Err(invalid("resolved onboarding is not one count on every rank"));
        }
        let mut local_by_rank = vec![Vec::new(); self.gpus];
        let mut assignments = Vec::with_capacity(modes.len());
        for (layer, &mode) in modes.iter().enumerate() {
            let experts = match solved.decisions.expert_homes.get(&layer) {
                None if self.experts.is_none() || layer < self.first_moe => ExpertHome::Dense,
                None => return Err(invalid("resolved routed layer has no expert home")),
                Some(_) if self.experts.is_none() || layer < self.first_moe =>
                    return Err(invalid("resolved expert home names a dense layer")),
                Some(ExpertDestination::Coordinator { gpus }) => {
                    if self.experts.as_ref().is_none_or(|profile| *gpus != profile.destination) {
                        return Err(invalid("resolved local home differs from prepared destination"));
                    }
                    for &gpu in gpus {
                        local_by_rank[usize::from(gpu)].push(layer);
                    }
                    match gpus.as_slice() {
                        [0, 1] if self.gpus == 2 => ExpertHome::RtxTp2,
                        [gpu] if usize::from(*gpu) < self.gpus => ExpertHome::RtxWhole { gpu: *gpu },
                        _ => return Err(invalid("executor cannot bind the resolved local expert home")),
                    }
                }
                Some(ExpertDestination::Sparks { ranks }) => {
                    if ranks.is_empty() || ranks.len() > 8
                        || ranks.iter().enumerate().any(|(rank, &value)| usize::from(value) != rank)
                    {
                        return Err(invalid("executor requires a Spark rank prefix starting at logical rank 0"));
                    }
                    if self.experts.as_ref().is_none_or(|profile| !profile.rank_sets.contains(ranks)) {
                        return Err(invalid("resolved Spark layers are not a prepared rank set"));
                    }
                    ExpertHome::Spark
                }
            };
            assignments.push(LayerAssignment { mode, experts });
        }
        if solved.decisions.expert_homes.keys().any(|&layer| layer >= modes.len())
            || local_by_rank != solved.decisions.onboard_layers
            || self.experts.as_ref().is_some_and(|profile| !profile.counts.contains(&count))
            || self.arenas.iter().all(|arena| arena.count != count)
        {
            return Err(invalid("resolved onboarding is not the declared ordered prefix"));
        }
        let arena = self.arenas.iter().find(|arena| arena.count == count).expect("checked above");
        if arena.peak.len() != self.gpus || arena.weights.len() != self.gpus {
            return Err(invalid("arena rank geometry differs from the coordinator count"));
        }
        let whole = assignments.iter().any(|a| matches!(a.experts, ExpertHome::RtxWhole { .. }));
        let tp2 = (self.gpus == 2 && count > 0 && !whole).then(|| Tp2Range {
            first: self.first_moe,
            layers: count,
            peak_bytes: [arena.peak[0], arena.peak[1]],
        });
        let expert_ranges = (0..self.gpus).map(|rank| {
            if tp2.is_some() || count == 0 || self.gpus != 1 {
                ExpertRange { first: 0, layers: 0, peak_bytes: 0 }
            } else {
                ExpertRange { first: self.first_moe, layers: count, peak_bytes: arena.peak[rank] }
            }
        }).collect();
        let residual = residual_homes(modes, &self.hop_spec).map_err(|error| invalid(&error.to_string()))?;
        let hops = plan_hops(modes, &self.hop_spec).map_err(|error| invalid(&error.to_string()))?;
        let peer_row_bytes = if modes.iter().any(|mode| *mode == LayerMode::ContextSplit) {
            self.peer_row_bytes
        } else {
            0
        };
        let attention_by_kind = self.kind_order().into_iter()
            .map(|kind| (kind, self.placement)).collect();
        Ok(Placement {
            pool_tokens: solved.pool_tokens,
            pool_limits: PoolLimits(vec![]),
            attention_placement: self.placement,
            attention_by_kind,
            peer_row_bytes,
            onboard_layers: count,
            max_pool_tokens_no_onboard: None,
            layers: assignments,
            movables: self.draft.iter().map(|_| (MovableId::Drafter, (self.gpus - 1) as u8)).collect(),
            expert_ranges,
            residual,
            hops,
            tp2,
            items: (0..self.gpus)
                .map(|rank| solved.items(Location::coordinator(rank as u8)))
                .collect::<Result<_, _>>()?,
        })
    }

    fn kind_order(&self) -> Vec<AttentionClass> {
        self.kinds.clone()
    }
}

impl Recipe for KimiRecipe {
    fn prepare(&self, request: &Request) -> Result<Contract, Refusal> {
        if !(1..=2).contains(&self.gpus)
            || self.context == 0
            || request.context != self.context
            || self.attention.is_empty()
            || self.layer_attention.len() != self.attention.len()
            || self.records_per_page.len() != self.gpus
            || self.kinds.is_empty()
            || self.prefill.rows == 0
            || self.prefill.waves == 0
            || self.lanes == 0
            || self.maximum_pages == 0
            || self.maximum_pages == u64::MAX
        {
            return Err(invalid("invalid extent or empty domain"));
        }
        self.page_rows()
            .checked_mul(self.maximum_pages)
            .ok_or(Refusal::Overflow("Kimi pool extent"))?;
        if self.attention.iter().any(|mode| match mode {
            LayerMode::HeadSplit | LayerMode::ContextSplit => self.gpus != 2,
            LayerMode::Whole { gpu, .. } => usize::from(*gpu) >= self.gpus,
        }) || self.fixed.iter().any(|owner| usize::from(owner.rank) >= self.gpus)
        {
            return Err(invalid("attention ownership or fixed owner rank"));
        }
        let mut contract = Contract {
            resident_weight_limit: request.resident_weight_limit,
            phase_order: PhaseOrder::TARGET_THEN_DRAFT,
            spark_reply_formats: self.spark_links.then(|| SparkReplyFormats {
                prefill: Some(request.spark_reply_formats.prefill(
                    cuteafd_core::ExpertGeometry::KIMI_K3.partial_replies())),
                decode: Some(request.spark_reply_formats.decode()),
            }),
            page_rows: self.page_rows(),
            maximum_pages: self.maximum_pages,
            gpus: self.gpus,
            variables: self.attention.iter().enumerate()
                .map(|(layer, &mode)| Variable::Attention { layer, values: vec![mode] })
                .collect(),
            requires: vec![],
            components: vec![],
        };
        contract.variables.push(Variable::Prefill(vec![self.prefill]));
        contract.variables.push(Variable::Lanes(vec![self.lanes]));
        let onboard = self.experts.as_ref().map(|profile| -> Result<(usize, &ExpertProfile), Refusal> {
            if profile.layers.is_empty()
                || profile.layers != (self.first_moe..self.attention.len()).collect::<Vec<_>>()
                || profile.destination.is_empty()
                || profile.destination.iter().any(|&gpu| usize::from(gpu) >= self.gpus)
                || profile.destination.windows(2).any(|pair| pair[0] >= pair[1])
                || profile.counts.is_empty()
                || profile.counts.iter().any(|&count| count > profile.layers.len())
                || profile.counts.windows(2).any(|pair| pair[0] >= pair[1])
                || profile.rank_sets.iter().enumerate().any(|(index, ranks)| {
                    ranks.is_empty() || ranks.len() > 8
                        || ranks.iter().enumerate().any(|(rank, &value)| usize::from(value) != rank)
                        || profile.rank_sets[..index].contains(ranks)
                })
            {
                return Err(invalid("invalid expert profile"));
            }
            let variable = contract.variables.len();
            contract.variables.push(Variable::Onboard {
                layers: profile.layers.clone(),
                destination: profile.destination.clone(),
                counts: profile.counts.clone(),
            });
            Ok((variable, profile))
        }).transpose()?;
        if let Some((_, profile)) = onboard {
            if !profile.rank_sets.is_empty() {
                contract.variables.push(Variable::ExpertHome {
                    layers: profile.layers.clone(),
                    rank_sets: profile.rank_sets.clone(),
                });
            }
        }
        let option = |count: usize| -> Result<Choice, Refusal> {
            let (variable, profile) = onboard.ok_or_else(|| invalid("arena without an expert profile"))?;
            let option = profile.counts.iter().position(|&value| value == count)
                .ok_or_else(|| invalid("arena count is not in the declared profile"))?;
            Ok(Choice { variable, option })
        };
        for (rank, &bytes) in self.records_per_page.iter().enumerate() {
            if bytes == 0 { continue; }
            let owner = Allocation::buffer(format!("records.{rank}"),
                Location::coordinator(rank as u8), self.page_rows(), Lifetime::RESIDENT,
                Category::Kv, "records");
            contract.components.push(Component::allocation(owner,
                ByteCurve::linear(bytes, bytes, self.maximum_pages), false, self.maximum_pages));
        }
        for fixed in &self.fixed {
            if fixed.bytes == 0 { continue; }
            let owner = Allocation::buffer(format!("{}.{}", fixed.group, fixed.rank),
                Location::coordinator(fixed.rank),
                fixed.bytes, fixed.lifetime, fixed.category, fixed.group.clone());
            if fixed.weights > 0 {
                if fixed.weights > fixed.bytes {
                    return Err(invalid("retained weights exceed their allocation"));
                }
                contract.components.push(Component::resident_weight(
                    ResidentWeight::from_allocation(&owner, ResidentWeights(fixed.weights)),
                    ByteCurve::fixed(fixed.weights, self.maximum_pages)));
            }
            contract.components.push(Component::allocation(owner,
                ByteCurve::fixed(fixed.bytes, self.maximum_pages), false, self.maximum_pages));
        }
        for rank in 0..self.gpus {
            let weights = self.attention.iter().enumerate()
                .try_fold(0u64, |sum, (layer, &mode)| {
                    sum.checked_add(attention_bytes(&self.layer_attention[layer], mode, rank))
                        .ok_or(Refusal::Overflow("Kimi attention weights"))
                })?;
            if weights == 0 { continue; }
            let owner = Allocation::buffer(format!("layers.{rank}"),
                Location::coordinator(rank as u8),
                weights, Lifetime::RESIDENT, Category::Weights, "layers");
            contract.components.push(Component::resident_weight(
                ResidentWeight::from_allocation(&owner, ResidentWeights(weights)),
                ByteCurve::fixed(weights, self.maximum_pages)));
            contract.components.push(Component::allocation(owner,
                ByteCurve::fixed(weights, self.maximum_pages), false, self.maximum_pages));
        }
        if self.hop_spec.row_bytes > 0 {
            let hops = plan_hops(&self.attention, &self.hop_spec)
                .map_err(|error| invalid(&error.to_string()))?;
            let buffers = hop_buffer_bytes(&hops, &self.hop_spec, self.gpus)
                .ok_or(Refusal::Overflow("Kimi residual hop buffers"))?;
            for (rank, &bytes) in buffers.iter().enumerate() {
                if bytes == 0 { continue; }
                let owner = Allocation::buffer(format!("residual hops.{rank}"),
                    Location::coordinator(rank as u8), bytes, Lifetime::RESIDENT,
                    Category::Transport, "residual hops");
                contract.components.push(Component::allocation(owner,
                    ByteCurve::fixed(bytes, self.maximum_pages), false, self.maximum_pages));
            }
        }
        for arena in &self.arenas {
            let choice = option(arena.count)?;
            if arena.peak.len() != self.gpus || arena.weights.len() != self.gpus {
                return Err(invalid("arena rank geometry differs from the coordinator count"));
            }
            let group = match arena.count {
                0 => continue,
                count if self.gpus == 2 => format!("TP2 expert layer halves {first}..{end}",
                    first = self.first_moe, end = self.first_moe + count),
                count => format!("resident routed layers {:?}",
                    &self.experts.as_ref().map(|profile| profile.layers[..count].to_vec()).unwrap_or_default()),
            };
            for rank in 0..self.gpus {
                let peak = arena.peak[rank];
                if peak == 0 { continue; }
                let owner = Allocation::buffer(format!("experts.{}.{rank}", arena.count),
                    Location::coordinator(rank as u8), peak, Lifetime::RESIDENT,
                    Category::Experts, group.clone());
                if arena.weights[rank] > 0 {
                    if arena.weights[rank] > peak {
                        return Err(invalid("resident experts exceed their arena peak"));
                    }
                    let mut weights = Component::resident_weight(
                        ResidentWeight::from_allocation(&owner, ResidentWeights(arena.weights[rank])),
                        ByteCurve::fixed(arena.weights[rank], self.maximum_pages));
                    weights.activation = Activation::choice(choice.variable, choice.option);
                    contract.components.push(weights);
                }
                let mut component = Component::allocation(owner,
                    ByteCurve::fixed(peak, self.maximum_pages), false, self.maximum_pages);
                component.activation = Activation::choice(choice.variable, choice.option);
                contract.components.push(component);
            }
        }
        for spark in &self.spark {
            let choice = option(spark.count)?;
            for (owner_kind, group, bytes) in [
                ("experts", "routed experts (balanced H128, replicated 3584 rotations)", spark.experts),
                ("overhead", "Spark rank I/O and estimated runtime residual", spark.overhead),
            ] {
                if bytes == 0 { continue; }
                let owner = Allocation::buffer(format!("spark.{}.{owner_kind}.{}", spark.rank, spark.count),
                    Location { device: Device::Spark(spark.rank as u8), domain: Domain::Managed },
                    bytes, Lifetime::RESIDENT, Category::Experts, group);
                let mut component = Component::allocation(owner,
                    ByteCurve::fixed(bytes, self.maximum_pages), false, self.maximum_pages);
                component.activation = Activation::choice(choice.variable, choice.option);
                contract.components.push(component);
            }
        }
        if let Some(draft) = &self.draft {
            let rank = (self.gpus - 1) as u8;
            let lifetime = Lifetime { first: Phase::DraftLoad, last: Phase::Growth };
            for (index, owner) in draft.owners.iter().enumerate() {
                let allocation = Allocation::buffer(format!("draft.{index}"),
                    Location::coordinator(rank), owner.bytes_at_first, lifetime,
                    owner.category, owner.group.clone());
                if owner.weights_at_first > 0 || owner.weights_per_page > 0 {
                    contract.components.push(Component::resident_weight(
                        ResidentWeight::from_allocation(&allocation, ResidentWeights(owner.weights_at_first)),
                        ByteCurve::linear(owner.weights_at_first, owner.weights_per_page, self.maximum_pages)));
                }
                contract.components.push(Component::allocation(allocation,
                    ByteCurve::linear(owner.bytes_at_first, owner.bytes_per_page, self.maximum_pages),
                    false, self.maximum_pages));
            }
            // The packing peak overlaps the target weights and the drafter's
            // own resident owners; it is transient, never a resident owner.
            let peak = Allocation::buffer("draft.load_peak", Location::coordinator(rank),
                draft.peak_at_first, Lifetime { first: Phase::DraftLoad, last: Phase::DraftLoad },
                Category::Drafter, "MLA DFlash load peak including transient packing");
            contract.components.push(Component::allocation(peak,
                ByteCurve::linear(draft.peak_at_first, draft.peak_per_page, self.maximum_pages),
                false, self.maximum_pages));
        }
        Ok(contract)
    }
}

/// Physical capacities and operator ceilings are independent inputs. `occupied`
/// is what the process already holds at `Load` (CUDA context and modules), not
/// a second subtraction of owners the recipe charges.
pub fn budgets(cards: &[(u64, u64)], headroom: u64, occupied: &[u64],
    sparks: &[u64]) -> Result<Vec<Budget>, Refusal> {
    if !(1..=2).contains(&cards.len()) || occupied.len() != cards.len() {
        return Err(invalid("invalid physical capacities"));
    }
    let forecast = |bytes: u64| Baseline::Forecast {
        occupied: bytes,
        source: "planned context and loaded modules; every Kimi owner charged in the recipe".into(),
        assumed: true,
    };
    let mut budgets = cards.iter().enumerate().map(|(rank, &(physical, logical))| Budget {
        id: format!("rtx{rank}"),
        locations: vec![Location::coordinator(rank as u8)],
        physical_bytes: physical,
        logical_bytes: logical,
        baseline: forecast(occupied[rank]),
        headroom,
    }).collect::<Vec<_>>();
    budgets.extend(sparks.iter().enumerate().map(|(rank, &bytes)| Budget {
        id: format!("spark{rank}"),
        locations: vec![Location { device: Device::Spark(rank as u8), domain: Domain::Managed }],
        physical_bytes: bytes,
        logical_bytes: bytes,
        baseline: forecast(0),
        headroom: 0,
    }));
    Ok(budgets)
}

/// Zero and omission normalize identically; an explicit pin stays exact.
pub fn request(context: u64, pool: Option<u64>, onboard: crate::placement::Onboard,
    prefill: PrefillShape, operation: Operation, weight_limit: ResidentWeights,
    spark_reply_formats: SparkReplyFormats) -> Request {
    Request {
        context,
        pool: crate::placement::PoolPolicy::normalize_request(pool).map_or(Pin::Auto, Pin::Exact),
        onboard: match onboard {
            crate::placement::Onboard::Auto => Pin::Auto,
            crate::placement::Onboard::Layers(count) => Pin::Exact(count as u64),
        },
        prefill: Pin::Exact(prefill),
        operation,
        resident_weight_limit: Some(weight_limit),
        spark_reply_formats,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::placement::FfnMode;

    fn single() -> Request {
        request_with(Some(0))
    }

    fn request_with(pool: Option<u64>) -> Request {
        crate::placement::families::kimi::request(256, pool, crate::placement::Onboard::Auto,
            PrefillShape { rows: 64, waves: 1 },
            crate::placement::planner::Operation::Serving, ResidentWeights(1 << 30),
            SparkReplyFormats::default())
    }

    fn recipe() -> KimiRecipe {
        KimiRecipe {
            context: 256,
            gpus: 1,
            placement: AttentionPlacement::Heads,
            attention: vec![LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner }; 2],
            prefill: PrefillShape { rows: 64, waves: 1 },
            lanes: 1,
            spark_links: false,
            hop_spec: HopSpec { storage: crate::placement::HopStorage::ReceiveSlots, row_bytes: 0,
                rows: 0, lanes: 0, entry_gpu: 0, head_gpu: 0,
                terminal: crate::placement::TerminalHome::AttentionOwner },
            peer_row_bytes: 0,
            maximum_pages: 4,
            records_per_page: vec![64],
            layer_attention: vec![ModeBytes { whole: 100, split: [0; 2] }; 2],
            first_moe: 1,
            kinds: vec![AttentionClass::Kda],
            experts: Some(ExpertProfile { layers: vec![1], destination: vec![0],
                counts: vec![0, 1], rank_sets: vec![] }),
            arenas: vec![
                Arena { count: 0, peak: vec![0], weights: vec![0] },
                Arena { count: 1, peak: vec![1000], weights: vec![900] },
            ],
            spark: vec![],
            draft: None,
            fixed: vec![FixedOwner { rank: 0, category: Category::Prefix, group: "Kimi exact KDA marks".into(),
                bytes: 32, weights: 0, lifetime: Lifetime::RESIDENT }],
        }
    }

    #[test]
    fn zero_pool_pin_normalizes_like_omission() {
        assert_eq!(request_with(Some(0)).pool, request_with(None).pool);
        assert_ne!(request_with(Some(256)).pool, request_with(None).pool);
    }

    #[test]
    fn pool_domain_covers_every_page_and_only_the_selected_onboard_count_charges() {
        let recipe = recipe();
        let contract = recipe.prepare(&single()).unwrap();
        // Two attention layers, prefill, lanes, and the onboard count profile.
        assert_eq!(contract.variables.len(), 5);
        for pages in 1..=recipe.maximum_pages {
            let idle = contract.facts(&[0, 0, 0, 0, 0], pages).unwrap();
            let records = idle.allocations.iter().find(|owner| owner.group == "records").unwrap();
            assert_eq!(records.bytes, pages * 64);
            let weights: u64 = idle.resident_weights.iter().map(|owner| owner.weights.0).sum();
            assert_eq!(weights, 200);
            assert!(!idle.allocations.iter().any(|owner| owner.category == Category::Experts));
            let onboarded = contract.facts(&[0, 0, 0, 0, 1], pages).unwrap();
            let arena: Vec<_> = onboarded.allocations.iter()
                .filter(|owner| owner.category == Category::Experts).collect();
            assert_eq!(arena.len(), 1);
            assert_eq!(arena[0].bytes, 1000);
            assert_eq!(onboarded.resident_weights.iter().map(|owner| owner.weights.0).sum::<u64>(), 1100);
        }
    }

    #[test]
    fn admission_resolves_the_declared_prefix_and_refuses_a_pin_above_it() {
        let recipe = recipe();
        let budgets = budgets(&[(1 << 30, 1 << 30)], 0, &[0], &[]).unwrap();
        let admitted = recipe.admit(single(), budgets.clone()).unwrap();
        assert_eq!(admitted.solved.pool_tokens, 256);
        assert_eq!(admitted.placement.attention_placement, AttentionPlacement::Heads);
        assert_eq!(admitted.placement.layers.len(), 2);
        assert_eq!(admitted.onboard,
            admitted.solved.decisions.onboard_layers.iter().map(Vec::len).max().unwrap_or(0));
        let pinned = crate::placement::families::kimi::request(256, Some(0), crate::placement::Onboard::Auto,
            PrefillShape { rows: 64, waves: 1 }, crate::placement::planner::Operation::Serving,
            ResidentWeights(1 << 30), SparkReplyFormats::default());
        assert_eq!(recipe.admit(pinned, budgets).unwrap().solved.pool_tokens, 256);
    }

    #[test]
    fn a_missing_physical_budget_is_refused_before_any_fit() {
        let recipe = recipe();
        assert!(recipe.admit(single(), Vec::new()).is_err());
        assert!(recipe.prepare(&Request { context: 512, ..single() }).is_err());
    }
}
