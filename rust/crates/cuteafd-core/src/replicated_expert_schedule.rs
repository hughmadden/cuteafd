//! Deterministic CPU-only expert-group scheduling for the routed FFN.
//!
//! The official backbone routes each token to top-6 of 384 routed experts per
//! layer. A deployment holds up to [`MAX_REPLICATED_EXPERT_GROUPS`] expert
//! groups of two kinds:
//!
//! - *full* groups (the replicated Spark groups, `1`, `2` or `3` of them), each
//!   carrying an identical copy of every routed expert;
//! - *partial-residency* groups (the coordinator GPU expert tier), each holding
//!   a fixed subset of a layer's experts beside the full groups.
//!
//! This module decides, per layer/request, which group executes each *active*
//! expert. It is a pure CPU planner: it has no protocol, transport, topology or
//! GPU dependency, and it does not bind any wire format. The internal
//! tensor-parallel width of a group is irrelevant to the scheduling decision;
//! it only enters through the group's cost model.
//!
//! Policy: whole-expert, largest-processing-time-first (LPT) greedy.
//!
//! 1. Every expert with at least one routed row is *active*; an expert with zero
//!    routed rows is left at the [`ReplicatedExpertGroupId::INACTIVE`] sentinel.
//! 2. An active expert that a partial group holds is *pinned*: it may run only
//!    on a partial group that holds it. Residency, not load, decides, so where
//!    a resident expert runs is a function of the layer's static resident set
//!    alone, never of batch composition or tie seeds. Every other active expert
//!    may run on any full group.
//! 3. Each active expert receives an estimated integer cost from the
//!    [`ReplicatedExpertCostModel`] of each group allowed to run it
//!    (weight-residency cost plus activation-tile cost); the cheapest of these
//!    is its ordering key. Costs are model inputs, not measured coefficients.
//! 4. Active experts are ordered by descending key, with a deterministic tie
//!    break derived from a *local* tie seed, and each is assigned to the
//!    allowed group that would finish it first: the group's predicted load,
//!    plus its [`ReplicatedExpertGroupSpec::call_cost`] if it is still unused,
//!    plus the expert's cost on that group. For identical full groups with no
//!    call cost this is the group with the smallest current load, so their
//!    plans are bit-identical to the historical replicated planner.
//!
//! Whole-expert assignment preserves weight reuse: an expert's weights are
//! executed once, in one group, for all of its routed rows. The policy cannot
//! split a single dominant expert across groups, so a request whose routes
//! concentrate on one expert is bounded below by that expert's full cost even
//! when several groups are idle. Hot-expert splitting is deliberately not
//! implemented here; it would be a separate, opt-in change justified by
//! benchmarks.
//!
//! Determinism and allocation:
//!
//! - Output is a pure function of `(config, route_counts, tie_seed, resident
//!   sets, expert_capacity)`. No process-global or cross-lane state is read.
//! - The plan borrows the scheduler's preallocated buffers, and resident sets
//!   are borrowed bitsets. A successful plan whose histogram length is within
//!   the planned capacity performs no heap allocation. `plan` rejects a larger
//!   histogram instead of growing; callers grow explicitly with
//!   [`ReplicatedExpertScheduler::reserve_expert_capacity`].
//! - The caller copies or encodes the borrowed assignment into a request-owned
//!   buffer before the next plan reuses the scheduler.

use crate::CuteafdError;

/// Maximum number of expert groups this planner supports: up to three
/// replicated Spark groups (the native wire's limit) plus a partial-residency
/// coordinator tier.
pub const MAX_REPLICATED_EXPERT_GROUPS: u8 = 4;

/// Encoded form of [`ReplicatedExpertGroupId::INACTIVE`].
pub const INACTIVE_REPLICATED_EXPERT_GROUP: u8 = u8::MAX;

/// Deterministic integer cost model for one whole routed expert.
///
/// All values are abstract *cost units* (CU). They are not microseconds, bytes,
/// FLOPs or any other physical quantity unless a caller chooses so (see
/// [`ReplicatedExpertCostModel::streaming`], which uses nanoseconds), and this
/// module ships no calibrated production coefficients. Choose values that make
/// the relative ordering and the group loads meaningful for the deployment
/// being planned; the scheduler only relies on the ordering and on exact
/// integer arithmetic. Groups compared against each other must use the same
/// unit.
///
/// The estimated cost of an active expert with `routed_rows` routed rows is:
///
/// ```text
/// expert_weight_cost + ceil(routed_rows / tile_rows) * tile_cost
/// ```
///
/// The weight term is charged **once per active expert**, never once per routed
/// row, because whole-expert assignment executes each expert's weights once.
/// The activation term models the padded routed-row tile work on a group.
/// The production-shaped default is the pure-weight form of this expression:
/// a uniform per-expert weight cost plus `ceil(count / tile_rows) * tile_cost`
/// with no additional terms. Whether specific numbers are feasible for a
/// deployment must be established by measurement; this module intentionally
/// exposes the model instead of assuming coefficients.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReplicatedExpertCostModel {
    /// Cost of executing one active expert's resident weights, in cost units.
    pub expert_weight_cost: u64,
    /// Cost of one routed activation tile, in cost units.
    pub tile_cost: u64,
    /// Routed rows per activation tile. Must be non-zero.
    pub tile_rows: u32,
}

impl ReplicatedExpertCostModel {
    /// Creates a model. See [`ReplicatedExpertCostModel::validate`].
    pub const fn new(expert_weight_cost: u64, tile_cost: u64, tile_rows: u32) -> Self {
        Self {
            expert_weight_cost,
            tile_cost,
            tile_rows,
        }
    }

    /// Physical form in integer nanoseconds: each member of the group streams
    /// `expert_bytes` of weights per active expert at `bytes_per_second` (a
    /// Spark rank its TP slice, a GPU its TP2 half), plus `tile_ns` per
    /// `tile_rows` routed rows. The weight term rounds up, so a positive byte
    /// count never costs zero.
    pub fn streaming(
        expert_bytes: u64,
        bytes_per_second: u64,
        tile_ns: u64,
        tile_rows: u32,
    ) -> Result<Self, CuteafdError> {
        if bytes_per_second == 0 {
            return Err(rejected("streaming cost model needs a non-zero byte rate"));
        }
        let nanos = (u128::from(expert_bytes) * 1_000_000_000).div_ceil(u128::from(bytes_per_second));
        let weight = u64::try_from(nanos)
            .map_err(|_| rejected("streaming cost model weight cost overflow"))?;
        let model = Self::new(weight, tile_ns, tile_rows);
        model.validate()?;
        Ok(model)
    }

    /// Rejects a model that cannot produce a positive cost for an active expert
    /// or that has a zero tile height.
    pub fn validate(&self) -> Result<(), CuteafdError> {
        if self.tile_rows == 0 {
            return Err(rejected("cost model tile_rows must be non-zero"));
        }
        if self.expert_weight_cost == 0 && self.tile_cost == 0 {
            return Err(rejected(
                "cost model must give active experts a positive cost \
                 (expert_weight_cost and tile_cost are both zero)",
            ));
        }
        Ok(())
    }

    /// Estimated cost of an expert with `routed_rows` routed rows.
    ///
    /// A zero-row expert is inactive and costs nothing; this is the only case
    /// where the weight term is not charged.
    pub fn expert_cost(&self, routed_rows: u64) -> Result<u64, CuteafdError> {
        self.validate()?;
        if routed_rows == 0 {
            return Ok(0);
        }
        let tiles = routed_rows.div_ceil(self.tile_rows as u64);
        let activation_cost = tiles
            .checked_mul(self.tile_cost)
            .ok_or_else(|| rejected("cost model activation-tile cost overflow"))?;
        self.expert_weight_cost
            .checked_add(activation_cost)
            .ok_or_else(|| rejected("cost model expert cost overflow"))
    }
}

/// Which routed experts a group holds.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ReplicatedExpertResidency {
    /// Every routed expert: a replicated (Spark) group.
    Full,
    /// A fixed subset, passed to [`ReplicatedExpertScheduler::plan_resident`] as
    /// a bitset per plan (one per layer). Every active expert it holds runs on
    /// a partial group that holds it; full groups take only the rest.
    Partial,
}

/// One group's static description: what it holds and what it costs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReplicatedExpertGroupSpec {
    pub residency: ReplicatedExpertResidency,
    /// Per-expert cost on this group.
    pub cost_model: ReplicatedExpertCostModel,
    /// Fixed cost of using the group in one plan (a Spark round trip, or a
    /// kernel launch and its combine), in the cost model's units. It is charged
    /// once, when the group receives its first expert; an unused group costs
    /// nothing. Zero for the historical replicated planner.
    pub call_cost: u64,
}

impl ReplicatedExpertGroupSpec {
    /// A replicated group holding every expert, with no call cost.
    pub const fn full(cost_model: ReplicatedExpertCostModel) -> Self {
        Self {
            residency: ReplicatedExpertResidency::Full,
            cost_model,
            call_cost: 0,
        }
    }

    /// A partial-residency group, with no call cost.
    pub const fn partial(cost_model: ReplicatedExpertCostModel) -> Self {
        Self {
            residency: ReplicatedExpertResidency::Partial,
            cost_model,
            call_cost: 0,
        }
    }

    /// The same group with a fixed per-plan call cost.
    pub const fn with_call_cost(self, call_cost: u64) -> Self {
        Self { call_cost, ..self }
    }

    pub const fn is_partial(&self) -> bool {
        matches!(self.residency, ReplicatedExpertResidency::Partial)
    }
}

/// Placeholder for group slots past `group_count`; never validated or read.
const UNUSED_GROUP: ReplicatedExpertGroupSpec =
    ReplicatedExpertGroupSpec::full(ReplicatedExpertCostModel::new(0, 0, 1));

/// Static planner configuration. Per-request inputs (histogram, tie seed and
/// the layer's resident sets) are passed to
/// [`ReplicatedExpertScheduler::plan_resident`] so one scheduler can serve a
/// whole lane without rebuilding.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReplicatedExpertScheduleConfig {
    group_count: u8,
    groups: [ReplicatedExpertGroupSpec; MAX_REPLICATED_EXPERT_GROUPS as usize],
}

impl ReplicatedExpertScheduleConfig {
    /// `group_count` full replicas sharing one cost model, with no call cost:
    /// the historical replicated planner.
    pub const fn new(group_count: u8, cost_model: ReplicatedExpertCostModel) -> Self {
        let mut groups = [UNUSED_GROUP; MAX_REPLICATED_EXPERT_GROUPS as usize];
        let mut index = 0;
        while index < groups.len() && index < group_count as usize {
            groups[index] = ReplicatedExpertGroupSpec::full(cost_model);
            index += 1;
        }
        Self {
            group_count,
            groups,
        }
    }

    /// Groups in group-index order, each with its own residency and cost.
    pub fn with_groups(specs: &[ReplicatedExpertGroupSpec]) -> Result<Self, CuteafdError> {
        if specs.len() > MAX_REPLICATED_EXPERT_GROUPS as usize {
            return Err(rejected(format!(
                "group_count {} exceeds the supported maximum {MAX_REPLICATED_EXPERT_GROUPS}",
                specs.len()
            )));
        }
        let mut groups = [UNUSED_GROUP; MAX_REPLICATED_EXPERT_GROUPS as usize];
        groups[..specs.len()].copy_from_slice(specs);
        let config = Self {
            group_count: specs.len() as u8,
            groups,
        };
        config.validate()?;
        Ok(config)
    }

    /// Number of groups, the number of valid group indices.
    pub const fn group_count(&self) -> u8 {
        self.group_count
    }

    /// The groups, `group_count` entries in group order.
    pub fn groups(&self) -> &[ReplicatedExpertGroupSpec] {
        &self.groups[..(self.group_count as usize).min(self.groups.len())]
    }

    /// Whether any group holds only a subset of the experts.
    pub fn has_partial_groups(&self) -> bool {
        self.groups().iter().any(ReplicatedExpertGroupSpec::is_partial)
    }

    /// Rejects a zero or unsupported group count and an invalid cost model.
    pub fn validate(&self) -> Result<(), CuteafdError> {
        if self.group_count == 0 {
            return Err(rejected("group_count must be non-zero"));
        }
        if self.group_count > MAX_REPLICATED_EXPERT_GROUPS {
            return Err(rejected(format!(
                "group_count {} exceeds the supported maximum {MAX_REPLICATED_EXPERT_GROUPS}",
                self.group_count
            )));
        }
        self.groups()
            .iter()
            .try_for_each(|group| group.cost_model.validate())
    }

    /// Whether `group` holds `expert_id`; partial groups read their bitset.
    fn holds(&self, group: usize, expert_id: usize, resident: &[&[u64]]) -> bool {
        match self.groups[group].residency {
            ReplicatedExpertResidency::Full => true,
            ReplicatedExpertResidency::Partial => bit(resident[group], expert_id),
        }
    }

    /// Whether some partial group holds `expert_id`, which pins it to them.
    fn pinned(&self, expert_id: usize, resident: &[&[u64]]) -> bool {
        self.groups()
            .iter()
            .enumerate()
            .any(|(group, spec)| spec.is_partial() && bit(resident[group], expert_id))
    }

    /// Whether `group` may run `expert_id` (pinned: only partial holders).
    fn allowed(&self, group: usize, expert_id: usize, pinned: bool, resident: &[&[u64]]) -> bool {
        self.groups[group].is_partial() == pinned && self.holds(group, expert_id, resident)
    }

    /// One bitset per group when any group is partial (full groups' entries
    /// empty), sized exactly for the histogram with no bit past its end; or no
    /// sets at all when every group is full.
    fn validate_residency(&self, expert_count: usize, resident: &[&[u64]]) -> Result<(), CuteafdError> {
        if resident.is_empty() {
            if self.has_partial_groups() {
                return Err(rejected(
                    "partial-residency groups need their resident sets (plan_resident)",
                ));
            }
            return Ok(());
        }
        if resident.len() != self.group_count as usize {
            return Err(rejected(format!(
                "{} resident sets for {} groups",
                resident.len(),
                self.group_count
            )));
        }
        let words = expert_count.div_ceil(64);
        for (group, (spec, set)) in self.groups().iter().zip(resident).enumerate() {
            if !spec.is_partial() {
                if !set.is_empty() {
                    return Err(rejected(format!("full group {group} takes no resident set")));
                }
                continue;
            }
            if set.len() != words {
                return Err(rejected(format!(
                    "group {group} resident set has {} words for {expert_count} experts",
                    set.len()
                )));
            }
            let tail = expert_count % 64;
            if tail != 0 && set[words - 1] >> tail != 0 {
                return Err(rejected(format!(
                    "group {group} resident set names an expert at or past {expert_count}"
                )));
            }
        }
        Ok(())
    }
}

/// A fixed set of resident routed experts of one layer, as a bitset over expert
/// ids: what a partial-residency group holds. Built once (at startup, from the
/// placement); plans borrow its words.
#[derive(Clone, Debug, Default, Eq, PartialEq, Hash)]
pub struct ExpertResidencySet {
    experts: usize,
    words: Vec<u64>,
}

impl ExpertResidencySet {
    /// No resident expert among `experts` routed experts.
    pub fn empty(experts: usize) -> Self {
        Self {
            experts,
            words: vec![0; experts.div_ceil(64)],
        }
    }

    /// The set of `ids` among `experts` routed experts; out-of-range and
    /// repeated ids are rejected.
    pub fn from_ids(
        experts: usize,
        ids: impl IntoIterator<Item = usize>,
    ) -> Result<Self, CuteafdError> {
        let mut set = Self::empty(experts);
        for id in ids {
            if !set.insert(id)? {
                return Err(rejected(format!("expert {id} named twice in a resident set")));
            }
        }
        Ok(set)
    }

    /// Adds `id`; false when it was already resident.
    pub fn insert(&mut self, id: usize) -> Result<bool, CuteafdError> {
        if id >= self.experts {
            return Err(rejected(format!(
                "resident expert {id} is outside 0..{}",
                self.experts
            )));
        }
        let (word, mask) = (id / 64, 1u64 << (id % 64));
        let fresh = self.words[word] & mask == 0;
        self.words[word] |= mask;
        Ok(fresh)
    }

    pub fn contains(&self, id: usize) -> bool {
        id < self.experts && bit(&self.words, id)
    }

    /// Resident experts.
    pub fn len(&self) -> usize {
        self.words.iter().map(|word| word.count_ones() as usize).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.words.iter().all(|word| *word == 0)
    }

    /// Routed experts of the layer (the bitset's universe).
    pub fn experts(&self) -> usize {
        self.experts
    }

    /// The bitset [`ReplicatedExpertScheduler::plan_resident`] borrows.
    pub fn words(&self) -> &[u64] {
        &self.words
    }

    /// Resident expert ids, ascending.
    pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.experts).filter(|&id| bit(&self.words, id))
    }
}

/// Target group for one expert, or the inactive sentinel.
///
/// The encoded byte is `0..group_count` for an assigned expert and
/// [`INACTIVE_REPLICATED_EXPERT_GROUP`] for an expert with no routed rows, so a
/// caller can copy the assignment straight into a request-owned byte buffer.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ReplicatedExpertGroupId(u8);

impl ReplicatedExpertGroupId {
    /// Sentinel for an expert with no routed rows in this plan.
    pub const INACTIVE: Self = Self(INACTIVE_REPLICATED_EXPERT_GROUP);

    /// Builds an active group id for `index`, or `None` if it is out of range.
    pub const fn active(index: u8) -> Option<Self> {
        if index < MAX_REPLICATED_EXPERT_GROUPS {
            Some(Self(index))
        } else {
            None
        }
    }

    /// True when this id names a real group.
    pub const fn is_active(self) -> bool {
        self.0 != INACTIVE_REPLICATED_EXPERT_GROUP
    }

    /// Group index when active.
    pub const fn index(self) -> Option<u8> {
        if self.is_active() {
            Some(self.0)
        } else {
            None
        }
    }

    /// Raw byte form: group index, or [`INACTIVE_REPLICATED_EXPERT_GROUP`].
    pub const fn encoded(self) -> u8 {
        self.0
    }
}

/// Predicted per-group load for one plan, in the same cost units as the model.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ReplicatedExpertGroupLoad {
    /// Sum of estimated expert costs assigned to this group, plus its call cost
    /// when it received any expert (predicted makespan driver; the maximum over
    /// groups is the predicted makespan).
    pub cost: u64,
    /// Number of active experts assigned to this group.
    pub expert_count: u32,
    /// Sum of routed rows over the experts assigned to this group.
    pub routed_rows: u64,
}

/// Borrowed assignment and predicted loads for one histogram.
///
/// All slices borrow the producing [`ReplicatedExpertScheduler`]; the borrow
/// ends when this value is dropped, so the scheduler cannot be replanned while
/// a plan is live.
#[derive(Clone, Copy, Debug)]
pub struct ReplicatedExpertGroupPlan<'a> {
    assignment: &'a [ReplicatedExpertGroupId],
    expert_cost: &'a [u64],
    group_loads: &'a [ReplicatedExpertGroupLoad],
    total_cost: u64,
    group_count: u8,
    tie_seed: u64,
}

impl<'a> ReplicatedExpertGroupPlan<'a> {
    /// Expert-id-indexed target group, one entry per histogram entry. Experts
    /// with no routed rows hold [`ReplicatedExpertGroupId::INACTIVE`].
    pub fn assignment(&self) -> &'a [ReplicatedExpertGroupId] {
        self.assignment
    }

    /// Expert-id-indexed estimated cost on the group that runs the expert;
    /// zero exactly for inactive experts.
    pub fn expert_cost(&self) -> &'a [u64] {
        self.expert_cost
    }

    /// Per-group predicted cost/load, `group_count` entries in group order.
    pub fn group_loads(&self) -> &'a [ReplicatedExpertGroupLoad] {
        self.group_loads
    }

    /// Sum of all group loads: every active expert's cost plus the call cost of
    /// every used group.
    pub fn total_cost(&self) -> u64 {
        self.total_cost
    }

    /// Predicted completion of the slowest group, the groups running in
    /// parallel: the largest group load.
    pub fn makespan(&self) -> u64 {
        self.group_loads
            .iter()
            .map(|load| load.cost)
            .max()
            .unwrap_or(0)
    }

    /// Number of groups in this plan.
    pub fn group_count(&self) -> u8 {
        self.group_count
    }

    /// Tie seed used to order equal-cost experts and equal-load groups.
    pub fn tie_seed(&self) -> u64 {
        self.tie_seed
    }

    /// Assignment of one expert id, or `None` when the id is out of range.
    pub fn group_for_expert(&self, expert_id: usize) -> Option<ReplicatedExpertGroupId> {
        self.assignment.get(expert_id).copied()
    }
}

/// Lane-owned, reusable expert-group planner.
///
/// A scheduler owns its scratch and output buffers. It reads no global state,
/// so two lanes planning concurrently produce independent results, and a plan
/// is a pure function of its inputs.
pub struct ReplicatedExpertScheduler {
    config: ReplicatedExpertScheduleConfig,
    planned_expert_capacity: usize,
    assignment: Vec<ReplicatedExpertGroupId>,
    expert_cost: Vec<u64>,
    tie_rank: Vec<u64>,
    order: Vec<usize>,
    group_loads: Vec<ReplicatedExpertGroupLoad>,
}

impl ReplicatedExpertScheduler {
    /// Creates a scheduler with preallocated capacity for `expert_capacity`
    /// histogram entries. `expert_capacity` is the production expert count
    /// (384 for the official backbone), not the routed-row count.
    pub fn new(
        config: ReplicatedExpertScheduleConfig,
        expert_capacity: usize,
    ) -> Result<Self, CuteafdError> {
        config.validate()?;
        let group_load_slots = config.group_count as usize;
        Ok(Self {
            config,
            planned_expert_capacity: expert_capacity,
            assignment: Vec::with_capacity(expert_capacity),
            expert_cost: Vec::with_capacity(expert_capacity),
            tie_rank: Vec::with_capacity(expert_capacity),
            order: Vec::with_capacity(expert_capacity),
            group_loads: Vec::with_capacity(group_load_slots),
        })
    }

    /// Static configuration this scheduler was built with.
    pub fn config(&self) -> ReplicatedExpertScheduleConfig {
        self.config
    }

    /// Histogram length this scheduler can plan without allocating.
    pub fn expert_capacity(&self) -> usize {
        self.planned_expert_capacity
    }

    /// Grows the preallocated capacity so later plans of up to `capacity`
    /// experts are allocation-free. This is the only sizing operation that
    /// allocates, and it is explicit so no request path allocates implicitly.
    pub fn reserve_expert_capacity(&mut self, capacity: usize) -> Result<(), CuteafdError> {
        self.config.validate()?;
        if capacity > self.planned_expert_capacity {
            reserve_to(&mut self.assignment, capacity);
            reserve_to(&mut self.expert_cost, capacity);
            reserve_to(&mut self.tie_rank, capacity);
            reserve_to(&mut self.order, capacity);
            self.planned_expert_capacity = capacity;
        }
        Ok(())
    }

    /// Plans whole-expert assignment for `route_counts` across full groups.
    /// A configuration with partial-residency groups must use
    /// [`Self::plan_resident`].
    ///
    /// `route_counts[expert_id]` is the number of routed rows that selected
    /// that expert for this layer/request; the official backbone passes 384
    /// entries. `tie_seed` must be derived from local layer/request identity
    /// only, e.g. with [`replicated_expert_tie_seed`].
    ///
    /// On success every active expert appears exactly once in the returned
    /// assignment, and no expert contribution is duplicated. On error the
    /// scheduler is unchanged for planning purposes: a later valid plan behaves
    /// as if the failed call had not happened.
    pub fn plan(
        &mut self,
        route_counts: &[u32],
        tie_seed: u64,
    ) -> Result<ReplicatedExpertGroupPlan<'_>, CuteafdError> {
        self.plan_resident(route_counts, tie_seed, &[])
    }

    /// [`Self::plan`] with the layer's resident sets: `resident[group]` is the
    /// bitset ([`ExpertResidencySet::words`]) of a partial group and empty for
    /// a full group; with no partial group, `resident` may be empty. Each
    /// partial set has exactly `route_counts.len().div_ceil(64)` words.
    ///
    /// Every active expert some partial group holds runs on such a group; every
    /// other active expert runs on a full group. An active expert no group
    /// holds is rejected. Borrowing the sets keeps the plan allocation-free.
    pub fn plan_resident(
        &mut self,
        route_counts: &[u32],
        tie_seed: u64,
        resident: &[&[u64]],
    ) -> Result<ReplicatedExpertGroupPlan<'_>, CuteafdError> {
        self.config.validate()?;
        let expert_count = route_counts.len();
        if expert_count > self.planned_expert_capacity {
            return Err(rejected(format!(
                "histogram length {expert_count} exceeds planned expert capacity {}; \
                 call reserve_expert_capacity outside the request path",
                self.planned_expert_capacity
            )));
        }
        self.config.validate_residency(expert_count, resident)?;
        let group_count = self.config.group_count as usize;
        let config = self.config;

        self.assignment.clear();
        self.assignment
            .resize(expert_count, ReplicatedExpertGroupId::INACTIVE);
        self.expert_cost.clear();
        self.expert_cost.resize(expert_count, 0);
        self.tie_rank.clear();
        self.tie_rank.resize(expert_count, 0);
        self.order.clear();
        self.group_loads.clear();
        self.group_loads
            .resize(group_count, ReplicatedExpertGroupLoad::default());

        // Ordering key: the cheapest cost among the groups allowed to run the
        // expert (for identical full groups, the one model's cost).
        let mut key_total = 0_u64;
        for (expert_id, &routed_rows) in route_counts.iter().enumerate() {
            if routed_rows == 0 {
                continue;
            }
            let pinned = config.pinned(expert_id, resident);
            let mut key = None;
            for group in 0..group_count {
                if config.allowed(group, expert_id, pinned, resident) {
                    let cost = config.groups[group]
                        .cost_model
                        .expert_cost(routed_rows as u64)?;
                    key = Some(key.map_or(cost, |key: u64| key.min(cost)));
                }
            }
            let key = key.ok_or_else(|| {
                rejected(format!("active expert {expert_id} is held by no group"))
            })?;
            self.expert_cost[expert_id] = key;
            key_total = key_total
                .checked_add(key)
                .ok_or_else(|| rejected("plan total estimated cost overflow"))?;
            self.tie_rank[expert_id] = expert_tie_rank(tie_seed, expert_id);
            self.order.push(expert_id);
        }

        {
            let order = &mut self.order;
            let expert_cost = &self.expert_cost;
            let tie_rank = &self.tie_rank;
            order.sort_unstable_by(|left, right| {
                expert_cost[*right]
                    .cmp(&expert_cost[*left])
                    .then_with(|| tie_rank[*left].cmp(&tie_rank[*right]))
                    .then_with(|| left.cmp(right))
            });
        }

        let mut group_rank = [0_u64; MAX_REPLICATED_EXPERT_GROUPS as usize];
        for (group, rank) in group_rank.iter_mut().enumerate().take(group_count) {
            *rank = group_tie_rank(tie_seed, group);
        }

        for order_index in 0..self.order.len() {
            let expert_id = self.order[order_index];
            let routed_rows = route_counts[expert_id] as u64;
            let pinned = config.pinned(expert_id, resident);
            // The allowed group that would finish this expert first. A
            // saturated finish only ties groups whose accumulation below
            // overflows anyway.
            let mut best: Option<(usize, u64, u64)> = None;
            for group in 0..group_count {
                if !config.allowed(group, expert_id, pinned, resident) {
                    continue;
                }
                let spec = &config.groups[group];
                let cost = spec.cost_model.expert_cost(routed_rows)?;
                let load = &self.group_loads[group];
                let opening = if load.expert_count == 0 { spec.call_cost } else { 0 };
                let finish = load.cost.saturating_add(opening).saturating_add(cost);
                if best.is_none_or(|(current, current_finish, _)| {
                    group_load_better(
                        finish,
                        group_rank[group],
                        group,
                        current_finish,
                        group_rank[current],
                        current,
                    )
                }) {
                    best = Some((group, finish, cost));
                }
            }
            let (group, _, cost) = best.ok_or_else(|| {
                rejected(format!("active expert {expert_id} is held by no group"))
            })?;
            self.assignment[expert_id] = ReplicatedExpertGroupId::active(group as u8)
                .ok_or_else(|| rejected("plan selected a group outside the supported range"))?;
            self.expert_cost[expert_id] = cost;
            let load = &mut self.group_loads[group];
            if load.expert_count == 0 {
                load.cost = load
                    .cost
                    .checked_add(config.groups[group].call_cost)
                    .ok_or_else(|| rejected("plan group call cost overflow"))?;
            }
            load.cost = load
                .cost
                .checked_add(cost)
                .ok_or_else(|| rejected("plan group cost overflow"))?;
            load.expert_count = load
                .expert_count
                .checked_add(1)
                .ok_or_else(|| rejected("plan group expert count overflow"))?;
            load.routed_rows = load
                .routed_rows
                .checked_add(routed_rows)
                .ok_or_else(|| rejected("plan group routed-row count overflow"))?;
        }

        let total_cost = self.group_loads.iter().try_fold(0_u64, |total, load| {
            total
                .checked_add(load.cost)
                .ok_or_else(|| rejected("plan total estimated cost overflow"))
        })?;

        Ok(ReplicatedExpertGroupPlan {
            assignment: &self.assignment,
            expert_cost: &self.expert_cost,
            group_loads: &self.group_loads,
            total_cost,
            group_count: config.group_count,
            tie_seed,
        })
    }
}

/// Deterministic tie seed from purely local scheduling identity.
///
/// The seed depends only on the layer id and request id supplied by the caller.
/// It never reads process-global, GPU, transport or cross-lane state, so two
/// independent lanes cannot perturb each other's tie breaks.
pub const fn replicated_expert_tie_seed(layer_id: u32, request_id: u64) -> u64 {
    mix64(mix64(request_id) ^ (layer_id as u64))
}

/// Opt-in tie-seed derivation for the replicated whole-expert planner.
///
/// Both modes feed the same [`replicated_expert_tie_seed`] mix. `Dispatch`
/// (default, and the only production mode) uses the caller's per-dispatch
/// request id, reproducing historical behavior byte for byte. `Layer` pins the
/// request component to [`TIE_SEED_FIXED_REQUEST_ID`], so the seed depends on
/// the layer alone; this makes two dispatches of the same work at the same layer
/// produce the same ownership when their route histogram is unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReplicatedExpertTieSeedMode {
    /// Historical behavior: seed from `(layer_id, dispatch request_id)`.
    Dispatch,
    /// Directional control: seed from `(layer_id, TIE_SEED_FIXED_REQUEST_ID)`.
    Layer,
}

/// Request-id component used by [`ReplicatedExpertTieSeedMode::Layer`].
///
/// Zero is a fixed sentinel, not a real request id: it is hashed through the
/// existing `mix64(request_id)` leg, so `layer` mode is a documented special
/// case of the same function rather than a second seed algorithm.
pub const TIE_SEED_FIXED_REQUEST_ID: u64 = 0;

impl ReplicatedExpertTieSeedMode {
    /// Parse an opt-in environment value.
    ///
    /// `None` (unset) selects [`Self::Dispatch`]. Only the exact lowercase
    /// tokens `dispatch` and `layer` are accepted; every other value, including
    /// an empty string, is rejected so a typo fails closed at startup instead of
    /// silently changing ownership.
    pub fn from_env_value(value: Option<&str>) -> Result<Self, CuteafdError> {
        match value {
            None | Some("dispatch") => Ok(Self::Dispatch),
            Some("layer") => Ok(Self::Layer),
            Some(other) => Err(CuteafdError::UnknownReplicatedExpertTieSeed(other.to_owned())),
        }
    }

    /// Stable lowercase name for logs and tests.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Dispatch => "dispatch",
            Self::Layer => "layer",
        }
    }
}

/// Tie seed for `mode`. `Dispatch` is byte-identical to
/// [`replicated_expert_tie_seed`]. `Layer` reuses the same mix with the request
/// component pinned to [`TIE_SEED_FIXED_REQUEST_ID`].
///
/// Neither mode makes the resulting assignment independent of the route
/// histogram: a changed batching/route composition can still repartition the
/// groups, so this is not a universal bit-reproducibility guarantee.
pub const fn replicated_expert_tie_seed_for(
    mode: ReplicatedExpertTieSeedMode,
    layer_id: u32,
    request_id: u64,
) -> u64 {
    match mode {
        ReplicatedExpertTieSeedMode::Dispatch => replicated_expert_tie_seed(layer_id, request_id),
        ReplicatedExpertTieSeedMode::Layer => {
            replicated_expert_tie_seed(layer_id, TIE_SEED_FIXED_REQUEST_ID)
        }
    }
}

fn expert_tie_rank(seed: u64, expert_id: usize) -> u64 {
    mix64(seed ^ mix64(expert_id as u64))
}

fn group_tie_rank(seed: u64, group: usize) -> u64 {
    // Distinct tag so equal-cost expert ranks and equal-load group ranks do not
    // share a permutation.
    mix64(seed ^ mix64(0x4752_4F55_505F_5449 ^ group as u64))
}

fn group_load_better(
    candidate_cost: u64,
    candidate_rank: u64,
    candidate_group: usize,
    current_cost: u64,
    current_rank: u64,
    current_group: usize,
) -> bool {
    candidate_cost < current_cost
        || (candidate_cost == current_cost
            && (candidate_rank < current_rank
                || (candidate_rank == current_rank && candidate_group < current_group)))
}

fn reserve_to<T>(values: &mut Vec<T>, capacity: usize) {
    if values.capacity() < capacity {
        // `Vec::reserve` takes an additional count relative to `len`, not a new
        // absolute capacity. Reserving `capacity - capacity()` would under-grow
        // whenever `len` is non-zero, so reserve the shortfall against `len`.
        // The guard plus the `len <= capacity()` invariant makes this positive.
        values.reserve(capacity - values.len());
    }
}

/// Bit `id` of a bitset over expert ids; ids past its words are absent.
fn bit(words: &[u64], id: usize) -> bool {
    words.get(id / 64).is_some_and(|word| (word >> (id % 64)) & 1 == 1)
}

/// SplitMix64 finalizer: deterministic, allocation-free, no external state.
const fn mix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn rejected(reason: impl Into<String>) -> CuteafdError {
    CuteafdError::ExpertRoutePlanRejected {
        reason: format!("replicated expert schedule: {}", reason.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    type Snapshot = (Vec<u8>, Vec<(u64, u32, u64)>, u64);

    fn model(expert_weight_cost: u64, tile_cost: u64, tile_rows: u32) -> ReplicatedExpertCostModel {
        ReplicatedExpertCostModel::new(expert_weight_cost, tile_cost, tile_rows)
    }

    fn config(
        group_count: u8,
        cost_model: ReplicatedExpertCostModel,
    ) -> ReplicatedExpertScheduleConfig {
        ReplicatedExpertScheduleConfig::new(group_count, cost_model)
    }

    fn scheduler(
        group_count: u8,
        cost_model: ReplicatedExpertCostModel,
        expert_capacity: usize,
    ) -> ReplicatedExpertScheduler {
        ReplicatedExpertScheduler::new(config(group_count, cost_model), expert_capacity).unwrap()
    }

    fn snapshot(
        scheduler: &mut ReplicatedExpertScheduler,
        route_counts: &[u32],
        tie_seed: u64,
    ) -> Snapshot {
        let plan = scheduler.plan(route_counts, tie_seed).unwrap();
        (
            plan.assignment()
                .iter()
                .map(|group| group.encoded())
                .collect(),
            plan.group_loads()
                .iter()
                .map(|load| (load.cost, load.expert_count, load.routed_rows))
                .collect(),
            plan.total_cost(),
        )
    }

    /// Encoded per-expert group assignment for one plan.
    fn assignment_of(
        scheduler: &mut ReplicatedExpertScheduler,
        route_counts: &[u32],
        tie_seed: u64,
    ) -> Vec<u8> {
        scheduler
            .plan(route_counts, tie_seed)
            .unwrap()
            .assignment()
            .iter()
            .map(|group| group.encoded())
            .collect()
    }

    fn base4_counts(mut encoded: usize, expert_count: usize) -> Vec<u32> {
        (0..expert_count)
            .map(|_| {
                let count = (encoded % 4) as u32;
                encoded /= 4;
                count
            })
            .collect()
    }

    fn brute_force_optimal_max(costs: &[u64], group_count: usize) -> u64 {
        if costs.is_empty() {
            return 0;
        }
        let mut best = u64::MAX;
        let mut loads = vec![0_u64; group_count];
        for assignment in 0..group_count.pow(costs.len() as u32) {
            loads.fill(0);
            let mut code = assignment;
            for &cost in costs {
                let group = code % group_count;
                code /= group_count;
                loads[group] += cost;
            }
            best = best.min(loads.iter().copied().max().unwrap_or(0));
        }
        best
    }

    /// Deliberately naive, test-only baseline: assign active experts to groups
    /// round-robin in ascending expert order. It is not a production strategy.
    fn round_robin_baseline_max_load(
        route_counts: &[u32],
        group_count: usize,
        cost_model: ReplicatedExpertCostModel,
    ) -> u64 {
        let mut loads = vec![0_u64; group_count];
        let mut next_group = 0_usize;
        for &routed_rows in route_counts {
            if routed_rows == 0 {
                continue;
            }
            loads[next_group % group_count] += cost_model.expert_cost(routed_rows as u64).unwrap();
            next_group += 1;
        }
        loads.into_iter().max().unwrap_or(0)
    }

    #[test]
    fn six_single_row_experts_balance_across_replicated_groups() {
        let route_counts = [1_u32; 6];
        let cost_model = model(100, 10, 1);

        let mut two_groups = scheduler(2, cost_model, route_counts.len());
        let plan = two_groups.plan(&route_counts, 7).unwrap();
        assert_eq!(
            plan.group_loads()
                .iter()
                .map(|load| load.expert_count)
                .collect::<Vec<_>>(),
            vec![3, 3]
        );
        assert_eq!(
            plan.group_loads()
                .iter()
                .map(|load| load.cost)
                .collect::<Vec<_>>(),
            vec![3 * 110, 3 * 110]
        );
        assert_eq!(plan.total_cost(), 6 * 110);

        let mut three_groups = scheduler(3, cost_model, route_counts.len());
        let plan = three_groups.plan(&route_counts, 7).unwrap();
        assert_eq!(
            plan.group_loads()
                .iter()
                .map(|load| load.expert_count)
                .collect::<Vec<_>>(),
            vec![2, 2, 2]
        );
        assert_eq!(
            plan.group_loads()
                .iter()
                .map(|load| load.cost)
                .collect::<Vec<_>>(),
            vec![2 * 110, 2 * 110, 2 * 110]
        );
        assert!(plan.assignment().iter().all(|group| group.is_active()));
    }

    #[test]
    fn weight_cost_is_charged_once_per_active_expert_not_per_row() {
        let cost_model = model(500, 3, 2);
        let mut planner = scheduler(2, cost_model, 3);
        let plan = planner.plan(&[5, 0, 0], 1).unwrap();

        // ceil(5 / 2) = 3 activation tiles; the 500 weight cost appears once.
        assert_eq!(plan.expert_cost()[0], 500 + 3 * 3);
        assert_ne!(plan.expert_cost()[0], 5 * 500 + 3 * 3);
        assert_eq!(plan.total_cost(), 509);
        assert_eq!(
            plan.group_loads()[plan.group_for_expert(0).unwrap().index().unwrap() as usize].cost,
            509
        );

        // A pure-weight model charges the weight once for any row count.
        let mut pure_weight = scheduler(1, model(1000, 1, 8), 2);
        let plan = pure_weight.plan(&[8, 0], 1).unwrap();
        assert_eq!(plan.expert_cost()[0], 1001);
        assert_ne!(plan.expert_cost()[0], 8 * 1000);
    }

    #[test]
    fn tile_costs_use_ceiling_boundaries() {
        let cost_model = model(100, 7, 4);
        let mut planner = scheduler(2, cost_model, 9);
        let plan = planner.plan(&[0, 1, 2, 3, 4, 5, 7, 8, 9], 3).unwrap();

        assert_eq!(
            plan.expert_cost(),
            &[0, 107, 107, 107, 107, 114, 114, 114, 121]
        );
        assert!(!plan.group_for_expert(0).unwrap().is_active());
        assert!(plan.group_for_expert(1).unwrap().is_active());
        assert_eq!(cost_model.expert_cost(0).unwrap(), 0);
        assert_eq!(cost_model.expert_cost(4).unwrap(), 107);
        assert_eq!(cost_model.expert_cost(5).unwrap(), 114);
    }

    #[test]
    fn empty_and_all_zero_histograms_are_inactive() {
        let mut planner = scheduler(2, model(10, 1, 1), 6);

        {
            let plan = planner.plan(&[], 5).unwrap();
            assert!(plan.assignment().is_empty());
            assert_eq!(plan.total_cost(), 0);
            assert_eq!(
                plan.group_loads(),
                &[
                    ReplicatedExpertGroupLoad::default(),
                    ReplicatedExpertGroupLoad::default()
                ]
            );
            assert!(plan.group_for_expert(0).is_none());
        }

        let all_zero = [0_u32; 6];
        let (assignment, loads, total) = snapshot(&mut planner, &all_zero, 5);
        assert!(assignment
            .iter()
            .all(|raw| *raw == INACTIVE_REPLICATED_EXPERT_GROUP));
        assert_eq!(loads, vec![(0, 0, 0), (0, 0, 0)]);
        assert_eq!(total, 0);
    }

    #[test]
    fn single_group_receives_every_active_expert() {
        let mut planner = scheduler(1, model(4, 2, 2), 4);
        // Costs: count 2 -> 4 + 1*2 = 6, count 3 -> 4 + 2*2 = 8, count 1 -> 6.
        let (assignment, loads, total) = snapshot(&mut planner, &[2, 0, 3, 1], 9);

        assert_eq!(assignment[0], 0);
        assert_eq!(assignment[1], INACTIVE_REPLICATED_EXPERT_GROUP);
        assert_eq!(assignment[2], 0);
        assert_eq!(assignment[3], 0);
        assert_eq!(loads, vec![(20, 3, 6)]);
        assert_eq!(total, 20);
    }

    #[test]
    fn seeded_tie_breaks_are_deterministic_and_seed_sensitive() {
        let route_counts = [1_u32; 6];
        let mut planner = scheduler(2, model(100, 10, 1), 6);
        let first = snapshot(&mut planner, &route_counts, 111);
        let second = snapshot(&mut planner, &route_counts, 111);
        assert_eq!(first, second);
        assert_eq!(
            first.0,
            snapshot(&mut scheduler(2, model(100, 10, 1), 6), &route_counts, 111).0
        );

        let mut distinct = BTreeSet::new();
        for seed in 0..16_u64 {
            let mut fresh = scheduler(2, model(100, 10, 1), 6);
            distinct.insert(snapshot(&mut fresh, &route_counts, seed).0);
        }
        assert!(
            distinct.len() > 1,
            "tie seed must influence equal-cost assignment"
        );
    }

    #[test]
    fn tie_seed_is_a_pure_function_of_local_identity() {
        assert_eq!(
            replicated_expert_tie_seed(7, 42),
            replicated_expert_tie_seed(7, 42)
        );
        assert_ne!(
            replicated_expert_tie_seed(7, 42),
            replicated_expert_tie_seed(8, 42)
        );
        assert_ne!(
            replicated_expert_tie_seed(7, 42),
            replicated_expert_tie_seed(7, 43)
        );
    }

    #[test]
    fn tie_seed_mode_parser_defaults_to_dispatch_and_fails_closed() {
        assert_eq!(
            ReplicatedExpertTieSeedMode::from_env_value(None).unwrap(),
            ReplicatedExpertTieSeedMode::Dispatch
        );
        assert_eq!(
            ReplicatedExpertTieSeedMode::from_env_value(Some("dispatch")).unwrap(),
            ReplicatedExpertTieSeedMode::Dispatch
        );
        assert_eq!(
            ReplicatedExpertTieSeedMode::from_env_value(Some("layer")).unwrap(),
            ReplicatedExpertTieSeedMode::Layer
        );
        for invalid in [
            "", "Dispatch", "LAYER", "layer ", " layer", "fixed", "histogram", "logical", "0",
        ] {
            assert!(
                ReplicatedExpertTieSeedMode::from_env_value(Some(invalid)).is_err(),
                "invalid value {invalid:?} must fail closed"
            );
        }
        assert_eq!(ReplicatedExpertTieSeedMode::Dispatch.as_str(), "dispatch");
        assert_eq!(ReplicatedExpertTieSeedMode::Layer.as_str(), "layer");
    }

    #[test]
    fn dispatch_mode_is_byte_identical_to_the_legacy_seed() {
        for (layer, request) in [(0_u32, 1_u64), (7, 42), (39, u64::MAX)] {
            assert_eq!(
                replicated_expert_tie_seed_for(
                    ReplicatedExpertTieSeedMode::Dispatch,
                    layer,
                    request
                ),
                replicated_expert_tie_seed(layer, request)
            );
        }
    }

    #[test]
    fn layer_mode_is_stable_across_request_ids() {
        for layer in [0_u32, 7, 39] {
            let expected = replicated_expert_tie_seed(layer, TIE_SEED_FIXED_REQUEST_ID);
            for request in [0_u64, 1, 42, u64::MAX] {
                assert_eq!(
                    replicated_expert_tie_seed_for(
                        ReplicatedExpertTieSeedMode::Layer,
                        layer,
                        request
                    ),
                    expected
                );
            }
        }
    }

    #[test]
    fn layer_mode_assignment_is_stable_across_request_ids_for_fixed_histogram() {
        // Uniform costs make every ordering decision a tie, so only the seed can
        // change the assignment. Layer mode must ignore every request id.
        let cost_model = model(1, 0, 16);
        let histogram = [3_u32, 3, 3, 3];
        let seed = |request: u64| {
            replicated_expert_tie_seed_for(ReplicatedExpertTieSeedMode::Layer, 5, request)
        };
        let base = assignment_of(&mut scheduler(2, cost_model, 4), &histogram, seed(1));
        for request in [0_u64, 2, 99, u64::MAX] {
            assert_eq!(
                assignment_of(&mut scheduler(2, cost_model, 4), &histogram, seed(request)),
                base
            );
        }
    }

    #[test]
    fn dispatch_mode_assignment_varies_with_request_id_for_fixed_histogram() {
        // This is the sensitivity the `layer` control removes; show it exists on
        // the same tie-heavy histogram with a deterministic search.
        let cost_model = model(1, 0, 16);
        let histogram = [3_u32, 3, 3, 3];
        let assignment = |request: u64| {
            let seed = replicated_expert_tie_seed_for(
                ReplicatedExpertTieSeedMode::Dispatch,
                5,
                request,
            );
            assignment_of(&mut scheduler(2, cost_model, 4), &histogram, seed)
        };
        let base = assignment(1);
        assert!(
            (2..64).any(|request| assignment(request) != base),
            "dispatch-mode owners must depend on the request id"
        );
    }

    #[test]
    fn histogram_changes_repartition_even_in_layer_mode() {
        // Honest limitation guard: layer mode is not histogram-invariant.
        let cost_model = model(1, 0, 16);
        let seed = replicated_expert_tie_seed_for(ReplicatedExpertTieSeedMode::Layer, 5, 0);
        let assignment = |histogram: &[u32]| {
            let mut s = ReplicatedExpertScheduler::new(config(2, cost_model), histogram.len())
                .unwrap();
            let plan = s.plan(histogram, seed).unwrap();
            plan.assignment().iter().map(|g| g.encoded()).collect::<Vec<u8>>()
        };
        let base = assignment(&[3, 3, 3, 3]);
        let changed = [[1_u32, 0, 0, 0], [0, 2, 0, 2], [4, 1, 0, 0], [0, 0, 0, 9]]
            .iter()
            .any(|candidate| assignment(candidate) != base);
        assert!(changed, "histogram changes must be able to repartition owners");
    }

    #[test]
    fn single_group_ownership_is_seed_and_request_independent() {
        // EP1 negative: with one group the seed cannot affect ownership.
        let cost_model = model(1, 0, 16);
        let histogram = [3_u32, 3, 3, 3];
        for mode in [
            ReplicatedExpertTieSeedMode::Dispatch,
            ReplicatedExpertTieSeedMode::Layer,
        ] {
            for request in [1_u64, 42, u64::MAX] {
                let seed = replicated_expert_tie_seed_for(mode, 5, request);
                let assignment =
                    assignment_of(&mut scheduler(1, cost_model, 4), &histogram, seed);
                assert!(
                    assignment.iter().all(|&group| group == 0),
                    "single-group ownership must always be group 0"
                );
            }
        }
    }

    #[test]
    fn expert_permutation_preserves_group_load_multiset() {
        // Distinct route counts give distinct estimated costs, so LPT ordering
        // is permutation-invariant apart from tie ranks; compare the sorted
        // load distribution rather than per-group labels.
        let cost_model = model(10, 1, 1);
        let route_counts = [9_u32, 5, 3, 1];
        let permuted = [3_u32, 9, 1, 5];

        let (_, base_loads, base_total) =
            snapshot(&mut scheduler(2, cost_model, 4), &route_counts, 5);
        let (_, permuted_loads, permuted_total) =
            snapshot(&mut scheduler(2, cost_model, 4), &permuted, 5);

        let mut base_sorted = base_loads;
        let mut permuted_sorted = permuted_loads;
        base_sorted.sort_unstable();
        permuted_sorted.sort_unstable();
        assert_eq!(base_sorted, permuted_sorted);
        assert_eq!(base_total, permuted_total);
        assert_eq!(base_total, 4 * 10 + 18);
    }

    #[test]
    fn independent_schedulers_keep_independent_scratch() {
        let route_counts = [4_u32, 4, 4, 4];
        let mut lane_a = scheduler(2, model(3, 1, 1), 4);
        let mut lane_b = scheduler(3, model(5, 1, 1), 4);

        let a_first = snapshot(&mut lane_a, &route_counts, 21);
        let b_first = snapshot(&mut lane_b, &route_counts, 22);
        let a_second = snapshot(&mut lane_a, &route_counts, 21);
        let b_second = snapshot(&mut lane_b, &route_counts, 22);

        assert_eq!(a_first, a_second);
        assert_eq!(b_first, b_second);
        assert_ne!(a_first.1.len(), b_first.1.len());
        assert_eq!(
            a_first,
            snapshot(&mut scheduler(2, model(3, 1, 1), 4), &route_counts, 21)
        );
        assert_eq!(
            b_first,
            snapshot(&mut scheduler(3, model(5, 1, 1), 4), &route_counts, 22)
        );
    }

    #[test]
    fn plan_borrows_outputs_and_preserves_inputs() {
        let route_counts = [3_u32, 0, 2];
        let before = route_counts;
        let mut planner = scheduler(2, model(5, 2, 2), 3);

        // The plan holds an exclusive borrow of the scheduler for its whole
        // lifetime, so the scheduler cannot be replanned while it is live. The
        // scope below records the borrowed slice addresses as plain integers.
        let (assignment_ptr, expert_cost_ptr, group_loads_ptr) = {
            let plan = planner.plan(&route_counts, 11).unwrap();
            assert_eq!(plan.tie_seed(), 11);
            assert_eq!(plan.group_count(), 2);
            assert_eq!(
                plan.group_for_expert(1).unwrap().encoded(),
                INACTIVE_REPLICATED_EXPERT_GROUP
            );
            (
                plan.assignment().as_ptr() as usize,
                plan.expert_cost().as_ptr() as usize,
                plan.group_loads().as_ptr() as usize,
            )
        };

        assert_eq!(route_counts, before);
        assert_eq!(assignment_ptr, planner.assignment.as_ptr() as usize);
        assert_eq!(expert_cost_ptr, planner.expert_cost.as_ptr() as usize);
        assert_eq!(group_loads_ptr, planner.group_loads.as_ptr() as usize);
    }

    #[test]
    fn invalid_configurations_and_overflow_are_rejected_then_recovered() {
        // Group-count bounds and zero-cost models are rejected at construction.
        assert!(ReplicatedExpertScheduler::new(config(0, model(1, 1, 1)), 4).is_err());
        assert!(ReplicatedExpertScheduler::new(
            config(MAX_REPLICATED_EXPERT_GROUPS + 1, model(1, 1, 1)),
            4
        )
        .is_err());
        assert!(ReplicatedExpertScheduler::new(config(2, model(1, 1, 0)), 4).is_err());
        assert!(ReplicatedExpertScheduler::new(config(2, model(0, 0, 1)), 4).is_err());
        assert!(model(1, 1, 0).expert_cost(1).is_err());
        assert!(model(0, 0, 1).expert_cost(1).is_err());
        assert_eq!(model(1, 1, 1).expert_cost(0).unwrap(), 0);

        // Group id sentinel boundary.
        assert!(ReplicatedExpertGroupId::active(MAX_REPLICATED_EXPERT_GROUPS - 1).is_some());
        assert!(ReplicatedExpertGroupId::active(MAX_REPLICATED_EXPERT_GROUPS).is_none());
        assert!(!ReplicatedExpertGroupId::INACTIVE.is_active());
        assert_eq!(ReplicatedExpertGroupId::INACTIVE.index(), None);
        assert_eq!(
            ReplicatedExpertGroupId::INACTIVE.encoded(),
            INACTIVE_REPLICATED_EXPERT_GROUP
        );

        // Per-expert cost overflow, then recovery on the same scheduler.
        let mut overflow = scheduler(2, model(u64::MAX, 1, 1), 4);
        assert!(overflow.plan(&[1, 0, 0, 0], 3).is_err());
        let recovered = snapshot(&mut overflow, &[0, 0, 0, 0], 3);
        assert!(recovered
            .0
            .iter()
            .all(|raw| *raw == INACTIVE_REPLICATED_EXPERT_GROUP));
        assert_eq!(recovered.2, 0);

        // Total-cost overflow, then a valid plan on the same scheduler.
        let half = u64::MAX / 2;
        let mut total_overflow = scheduler(2, model(half, 0, 1), 4);
        assert!(total_overflow.plan(&[1, 1, 1, 0], 3).is_err());
        let recovered = snapshot(&mut total_overflow, &[1, 1, 0, 0], 3);
        assert_eq!(recovered.2, half * 2);

        // Capacity overflow is explicit and recoverable via reserve.
        let mut capacity = scheduler(2, model(1, 1, 1), 2);
        assert!(capacity.plan(&[1, 1, 1, 1], 3).is_err());
        assert_eq!(capacity.expert_capacity(), 2);
        capacity.reserve_expert_capacity(4).unwrap();
        assert_eq!(capacity.expert_capacity(), 4);
        let recovered = snapshot(&mut capacity, &[1, 1, 1, 1], 3);
        assert_eq!(recovered.2, 8);
    }

    #[test]
    fn reserve_expert_capacity_is_honored_before_and_after_planning() {
        // Before the first plan every buffer is empty. A reserve that used the
        // old capacity instead of len would under-grow here and then reallocate
        // during the first larger plan.
        let mut fresh = scheduler(2, model(3, 1, 1), 2);
        fresh.reserve_expert_capacity(6).unwrap();
        assert_eq!(fresh.expert_capacity(), 6);
        assert!(fresh.assignment.capacity() >= 6);
        assert!(fresh.expert_cost.capacity() >= 6);
        assert!(fresh.tie_rank.capacity() >= 6);
        assert!(fresh.order.capacity() >= 6);

        let counts = [1_u32; 6];
        let _ = fresh.plan(&counts, 2).unwrap();
        let fresh_capacities = (
            fresh.assignment.capacity(),
            fresh.expert_cost.capacity(),
            fresh.tie_rank.capacity(),
            fresh.order.capacity(),
        );
        let fresh_expected = snapshot(&mut fresh, &counts, 2);
        for _ in 0..64 {
            assert_eq!(snapshot(&mut fresh, &counts, 2), fresh_expected);
        }
        assert_eq!(
            fresh_capacities,
            (
                fresh.assignment.capacity(),
                fresh.expert_cost.capacity(),
                fresh.tie_rank.capacity(),
                fresh.order.capacity(),
            ),
            "a plan within the reserved capacity must not reallocate"
        );

        // After a short plan the buffers have non-zero len, which is the case a
        // reserve measured against the old capacity miscounts.
        let mut used = scheduler(2, model(3, 1, 1), 4);
        let _ = used.plan(&[1, 1], 1).unwrap();
        assert_eq!(used.expert_capacity(), 4);
        used.reserve_expert_capacity(8).unwrap();
        assert_eq!(used.expert_capacity(), 8);
        assert!(used.assignment.capacity() >= 8);
        assert!(used.expert_cost.capacity() >= 8);
        assert!(used.tie_rank.capacity() >= 8);
        assert!(used.order.capacity() >= 8);

        let counts = [1_u32; 8];
        let _ = used.plan(&counts, 3).unwrap();
        let used_capacities = (
            used.assignment.capacity(),
            used.expert_cost.capacity(),
            used.tie_rank.capacity(),
            used.order.capacity(),
            used.group_loads.capacity(),
        );
        let used_expected = snapshot(&mut used, &counts, 3);
        for _ in 0..64 {
            assert_eq!(snapshot(&mut used, &counts, 3), used_expected);
        }
        assert_eq!(
            used_capacities,
            (
                used.assignment.capacity(),
                used.expert_cost.capacity(),
                used.tie_rank.capacity(),
                used.order.capacity(),
                used.group_loads.capacity(),
            ),
            "a plan at the reserved capacity must not reallocate"
        );
    }

    #[test]
    fn whole_expert_policy_keeps_dominant_expert_unsplit() {
        // One expert owns most routed rows. Whole-expert assignment must place
        // its full cost in exactly one group; the predicted makespan cannot
        // drop below that cost, which is the documented policy limitation.
        let route_counts = [64_u32, 1, 1, 1];
        let mut planner = scheduler(2, model(0, 1, 1), 4);
        let plan = planner.plan(&route_counts, 9).unwrap();

        let dominant_group = plan.group_for_expert(0).unwrap();
        assert!(dominant_group.is_active());
        assert_eq!(plan.expert_cost()[0], 64);
        assert_eq!(
            plan.assignment()
                .iter()
                .filter(|group| **group == dominant_group)
                .count(),
            1
        );
        let dominant_load = plan.group_loads()[dominant_group.index().unwrap() as usize];
        assert!(dominant_load.cost >= 64);
        assert_eq!(plan.total_cost(), 67);
        assert_eq!(
            plan.group_loads().iter().map(|load| load.cost).sum::<u64>(),
            plan.total_cost()
        );
    }

    #[test]
    fn exhaustive_small_histograms_cover_each_active_expert_once() {
        for group_count in 1..=MAX_REPLICATED_EXPERT_GROUPS {
            for expert_count in 0..=4_usize {
                for encoded in 0..4_usize.pow(expert_count as u32) {
                    let route_counts = base4_counts(encoded, expert_count);
                    let cost_model = model(7, 3, 2);
                    let mut planner = scheduler(group_count, cost_model, expert_count);
                    let seed = 1234 + encoded as u64 + expert_count as u64;
                    let plan = planner.plan(&route_counts, seed).unwrap();

                    let mut active_costs = Vec::new();
                    let mut cost_by_group = vec![0_u64; group_count as usize];
                    let mut experts_by_group = vec![0_u32; group_count as usize];
                    let mut rows_by_group = vec![0_u64; group_count as usize];
                    for (expert_id, &routed_rows) in route_counts.iter().enumerate() {
                        let group = plan.group_for_expert(expert_id).unwrap();
                        if routed_rows == 0 {
                            assert!(!group.is_active());
                            assert_eq!(plan.expert_cost()[expert_id], 0);
                            continue;
                        }
                        assert!(group.is_active());
                        let group_index = group.index().unwrap() as usize;
                        assert!(group_index < group_count as usize);
                        let expected = 7 + (routed_rows as u64).div_ceil(2) * 3;
                        assert_eq!(plan.expert_cost()[expert_id], expected);
                        active_costs.push(expected);
                        cost_by_group[group_index] += expected;
                        experts_by_group[group_index] += 1;
                        rows_by_group[group_index] += routed_rows as u64;
                    }

                    // Every active expert is covered exactly once.
                    assert_eq!(
                        experts_by_group.iter().sum::<u32>(),
                        route_counts.iter().filter(|count| **count > 0).count() as u32
                    );
                    assert_eq!(
                        rows_by_group.iter().sum::<u64>(),
                        route_counts.iter().map(|count| *count as u64).sum::<u64>()
                    );
                    for (group, load) in plan.group_loads().iter().enumerate() {
                        assert_eq!(load.cost, cost_by_group[group]);
                        assert_eq!(load.expert_count, experts_by_group[group]);
                        assert_eq!(load.routed_rows, rows_by_group[group]);
                    }
                    assert_eq!(plan.total_cost(), cost_by_group.iter().sum::<u64>());
                    assert_eq!(plan.total_cost(), plan.expert_cost().iter().sum::<u64>());

                    // LPT is 4/3-competitive; assert the loose bound and never
                    // assert optimality.
                    let optimal = brute_force_optimal_max(&active_costs, group_count as usize);
                    let greedy = plan
                        .group_loads()
                        .iter()
                        .map(|load| load.cost)
                        .max()
                        .unwrap_or(0);
                    assert!(
                        greedy * 3 <= optimal * 4,
                        "greedy {greedy} exceeded 4/3 of optimal {optimal} \
                         for groups={group_count} counts={route_counts:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn greedy_lpt_matches_or_beats_test_only_round_robin_baseline() {
        let route_counts = [8_u32, 1, 1, 1, 1];
        let cost_model = model(0, 1, 1);
        let mut planner = scheduler(2, cost_model, route_counts.len());
        let (_, loads, _) = snapshot(&mut planner, &route_counts, 4);

        let greedy_max = loads.iter().map(|load| load.0).max().unwrap();
        let baseline_max = round_robin_baseline_max_load(&route_counts, 2, cost_model);
        assert_eq!(greedy_max, 8);
        assert_eq!(baseline_max, 10);
        assert!(greedy_max <= baseline_max);
    }

    #[test]
    fn production_shape_repeated_plans_are_allocation_stable() {
        // Cheap CPU regression that doubles as a microbenchmark shape check: it
        // makes no timing or GPU-speedup claim. 384 experts matches the official
        // backbone. Cost values are placeholders, not calibrated coefficients.
        let route_counts: Vec<u32> = (0..384)
            .map(|expert_id| ((expert_id * 37 + 11) % 9) as u32)
            .collect();
        let cost_model = model(2304, 1536, 64);
        let mut planner = scheduler(3, cost_model, 384);

        // Warm up so any first-use growth is done before capacity is recorded.
        let _ = planner.plan(&route_counts, 42).unwrap();
        let capacities = (
            planner.assignment.capacity(),
            planner.expert_cost.capacity(),
            planner.tie_rank.capacity(),
            planner.order.capacity(),
            planner.group_loads.capacity(),
        );
        let expected = snapshot(&mut planner, &route_counts, 42);

        for _ in 0..512 {
            assert_eq!(snapshot(&mut planner, &route_counts, 42), expected);
        }
        assert_eq!(
            capacities,
            (
                planner.assignment.capacity(),
                planner.expert_cost.capacity(),
                planner.tie_rank.capacity(),
                planner.order.capacity(),
                planner.group_loads.capacity(),
            ),
            "a successful plan within capacity must not reallocate"
        );
    }

    // ---- Partial residency, per-group costs and call costs ----

    /// The planner as it was before partial residency, kept verbatim as the
    /// bit-identity oracle for full replicas: (assignment, expert costs,
    /// (cost, experts, rows) per group, total).
    fn legacy_plan(
        group_count: usize,
        model: ReplicatedExpertCostModel,
        route_counts: &[u32],
        tie_seed: u64,
    ) -> Snapshot {
        let mut assignment = vec![INACTIVE_REPLICATED_EXPERT_GROUP; route_counts.len()];
        let mut expert_cost = vec![0_u64; route_counts.len()];
        let mut tie_rank = vec![0_u64; route_counts.len()];
        let mut order = Vec::new();
        let mut loads = vec![(0_u64, 0_u32, 0_u64); group_count];
        let mut total = 0_u64;
        for (expert_id, &rows) in route_counts.iter().enumerate() {
            if rows == 0 {
                continue;
            }
            let cost = model.expert_cost(rows as u64).unwrap();
            expert_cost[expert_id] = cost;
            total += cost;
            tie_rank[expert_id] = expert_tie_rank(tie_seed, expert_id);
            order.push(expert_id);
        }
        order.sort_unstable_by(|l, r| {
            expert_cost[*r]
                .cmp(&expert_cost[*l])
                .then_with(|| tie_rank[*l].cmp(&tie_rank[*r]))
                .then_with(|| l.cmp(r))
        });
        let rank: Vec<u64> = (0..group_count).map(|g| group_tie_rank(tie_seed, g)).collect();
        for expert_id in order {
            let mut best = 0;
            for group in 1..group_count {
                if group_load_better(loads[group].0, rank[group], group, loads[best].0, rank[best], best) {
                    best = group;
                }
            }
            assignment[expert_id] = best as u8;
            loads[best].0 += expert_cost[expert_id];
            loads[best].1 += 1;
            loads[best].2 += route_counts[expert_id] as u64;
        }
        (assignment, loads, total)
    }

    fn full_snapshot(plan: ReplicatedExpertGroupPlan<'_>) -> (Snapshot, Vec<u64>) {
        (
            (
                plan.assignment().iter().map(|g| g.encoded()).collect(),
                plan.group_loads()
                    .iter()
                    .map(|l| (l.cost, l.expert_count, l.routed_rows))
                    .collect(),
                plan.total_cost(),
            ),
            plan.expert_cost().to_vec(),
        )
    }

    /// SplitMix-driven histograms: sparse decode-like and dense prefill-like.
    fn histogram(seed: u64, experts: usize, routes: usize) -> Vec<u32> {
        let mut counts = vec![0_u32; experts];
        if experts == 0 {
            return counts;
        }
        let mut state = seed;
        for _ in 0..routes {
            state = mix64(state);
            counts[(state % experts as u64) as usize] += 1;
        }
        counts
    }

    fn mask(experts: usize, ids: impl IntoIterator<Item = usize>) -> ExpertResidencySet {
        ExpertResidencySet::from_ids(experts, ids).unwrap()
    }

    #[test]
    fn full_replica_plans_match_the_legacy_planner_bit_for_bit() {
        let models = [model(1, 0, 16), model(7, 3, 2), model(2304, 1536, 64), model(0, 1, 1)];
        for group_count in 1..=3_u8 {
            for (index, &cost_model) in models.iter().enumerate() {
                for (experts, routes) in [(0, 0), (4, 3), (17, 40), (384, 6), (384, 36), (384, 24_576)] {
                    for seed in [0_u64, 7, 0xdead_beef, u64::MAX] {
                        let counts = histogram(seed ^ index as u64 ^ routes as u64, experts, routes);
                        let expected = legacy_plan(group_count as usize, cost_model, &counts, seed);
                        let mut historical = scheduler(group_count, cost_model, experts);
                        let (got, costs) = full_snapshot(historical.plan(&counts, seed).unwrap());
                        assert_eq!(got, expected, "groups={group_count} model={index} experts={experts}");
                        let legacy_costs: Vec<u64> = counts
                            .iter()
                            .map(|&rows| cost_model.expert_cost(rows as u64).unwrap())
                            .collect();
                        assert_eq!(costs, legacy_costs);

                        // The same replicas spelled as explicit specs, with
                        // empty resident sets, plan identically.
                        let specs = vec![ReplicatedExpertGroupSpec::full(cost_model); group_count as usize];
                        let config = ReplicatedExpertScheduleConfig::with_groups(&specs).unwrap();
                        assert_eq!(config, ReplicatedExpertScheduleConfig::new(group_count, cost_model));
                        let mut explicit = ReplicatedExpertScheduler::new(config, experts).unwrap();
                        let empty: Vec<&[u64]> = vec![&[]; group_count as usize];
                        let (spelled, spelled_costs) =
                            full_snapshot(explicit.plan_resident(&counts, seed, &empty).unwrap());
                        assert_eq!((spelled, spelled_costs), (expected, costs.clone()));
                    }
                }
            }
        }
    }

    /// A GPU tier (partial, cheaper per expert) beside one Spark group (full).
    fn tier_config(gpu_ns: u64, spark_ns: u64, gpu_call: u64, spark_call: u64) -> ReplicatedExpertScheduleConfig {
        ReplicatedExpertScheduleConfig::with_groups(&[
            ReplicatedExpertGroupSpec::partial(model(gpu_ns, 0, 16)).with_call_cost(gpu_call),
            ReplicatedExpertGroupSpec::full(model(spark_ns, 0, 16)).with_call_cost(spark_call),
        ])
        .unwrap()
    }

    #[test]
    fn resident_active_experts_run_on_the_partial_group_whatever_its_load() {
        let resident = mask(384, 0..80);
        // Even a partial group far slower than the Sparks keeps every resident
        // expert: residency, not load, decides.
        for (gpu_ns, spark_ns) in [(11_000, 21_000), (1_000_000, 1)] {
            let mut planner = ReplicatedExpertScheduler::new(tier_config(gpu_ns, spark_ns, 0, 0), 384).unwrap();
            let counts = histogram(42, 384, 4096 * 6);
            let plan = planner.plan_resident(&counts, 9, &[resident.words(), &[]]).unwrap();
            for (expert, &rows) in counts.iter().enumerate() {
                let group = plan.group_for_expert(expert).unwrap();
                match (rows, resident.contains(expert)) {
                    (0, _) => assert!(!group.is_active()),
                    (_, true) => assert_eq!(group.index(), Some(0), "resident expert {expert}"),
                    (_, false) => assert_eq!(group.index(), Some(1), "non-resident expert {expert}"),
                }
            }
            let loads = plan.group_loads();
            assert_eq!(loads[0].expert_count as usize, (0..80).filter(|&e| counts[e] > 0).count());
            assert_eq!(loads[0].cost, gpu_ns * loads[0].expert_count as u64);
            assert_eq!(loads[1].cost, spark_ns * loads[1].expert_count as u64);
            assert_eq!(plan.makespan(), loads[0].cost.max(loads[1].cost));
        }
    }

    #[test]
    fn pinned_owners_ignore_tie_seeds_and_batch_composition() {
        // Byte-exact restores rely on this: whether an expert runs on the tier
        // is fixed by the layer's resident set, never by what else is batched.
        let resident = mask(384, (0..384).step_by(5));
        let config = tier_config(11_000, 21_000, 5_000, 34_000);
        let mut planner = ReplicatedExpertScheduler::new(config, 384).unwrap();
        for seed in 0..64_u64 {
            let counts = histogram(seed.wrapping_mul(31), 384, 6 * (1 + seed as usize % 9));
            let plan = planner.plan_resident(&counts, mix64(seed), &[resident.words(), &[]]).unwrap();
            for (expert, &rows) in counts.iter().enumerate() {
                if rows > 0 {
                    let expected = if resident.contains(expert) { 0 } else { 1 };
                    assert_eq!(plan.group_for_expert(expert).unwrap().index(), Some(expected));
                }
            }
        }
    }

    #[test]
    fn call_cost_is_charged_once_and_only_to_used_groups() {
        let resident = mask(8, [0, 1]);
        let mut planner = ReplicatedExpertScheduler::new(tier_config(10, 20, 5, 34), 8).unwrap();
        // Two resident and three non-resident active experts.
        let plan = planner
            .plan_resident(&[1, 2, 0, 1, 1, 3, 0, 0], 3, &[resident.words(), &[]])
            .unwrap();
        assert_eq!(plan.group_loads()[0].cost, 5 + 2 * 10);
        assert_eq!(plan.group_loads()[1].cost, 34 + 3 * 20);
        assert_eq!(plan.total_cost(), 5 + 20 + 34 + 60);
        assert_eq!(plan.makespan(), 94);
        assert_eq!(plan.expert_cost(), &[10, 10, 0, 20, 20, 20, 0, 0]);

        // Every active expert resident: the Spark group is unused, so its
        // round trip is not charged.
        let plan = planner
            .plan_resident(&[1, 2, 0, 0, 0, 0, 0, 0], 3, &[resident.words(), &[]])
            .unwrap();
        assert_eq!(plan.group_loads()[1], ReplicatedExpertGroupLoad::default());
        assert_eq!(plan.makespan(), 25);

        // Nothing resident active: the tier is unused.
        let plan = planner
            .plan_resident(&[0, 0, 4, 0, 0, 0, 0, 0], 3, &[resident.words(), &[]])
            .unwrap();
        assert_eq!(plan.group_loads()[0], ReplicatedExpertGroupLoad::default());
        assert_eq!(plan.makespan(), 54);
    }

    #[test]
    fn call_cost_keeps_a_group_closed_until_it_pays_off() {
        // Group 1 pays a 100-unit call (a round trip); group 0 none. Cheap
        // experts stay on group 0 until its load reaches the call cost.
        let config = ReplicatedExpertScheduleConfig::with_groups(&[
            ReplicatedExpertGroupSpec::full(model(10, 0, 16)),
            ReplicatedExpertGroupSpec::full(model(10, 0, 16)).with_call_cost(100),
        ])
        .unwrap();
        let mut planner = ReplicatedExpertScheduler::new(config, 20).unwrap();
        let plan = planner.plan(&[1; 5], 4).unwrap();
        assert_eq!(plan.group_loads()[1], ReplicatedExpertGroupLoad::default());
        assert_eq!(plan.makespan(), 50);
        assert_eq!(plan.total_cost(), 50);
        let plan = planner.plan(&[1; 20], 4).unwrap();
        let loads = plan.group_loads();
        assert!(loads[1].expert_count > 0);
        assert_eq!(loads[0].expert_count + loads[1].expert_count, 20);
        assert_eq!(plan.total_cost(), 200 + 100);
        assert!(plan.makespan() <= 160, "{loads:?}");
    }

    #[test]
    fn heterogeneous_full_groups_assign_by_earliest_finish() {
        let config = ReplicatedExpertScheduleConfig::with_groups(&[
            ReplicatedExpertGroupSpec::full(model(10, 0, 16)),
            ReplicatedExpertGroupSpec::full(model(30, 0, 16)),
        ])
        .unwrap();
        let mut planner = ReplicatedExpertScheduler::new(config, 4).unwrap();
        // Finish times: 10, 20 on group 0 then 30 on either; group 0 keeps
        // winning until its load passes the slow group's single cost.
        let plan = planner.plan(&[1, 1, 1, 1], 0).unwrap();
        assert_eq!(plan.group_loads()[0].expert_count, 3);
        assert_eq!(plan.group_loads()[1].expert_count, 1);
        assert_eq!(plan.makespan(), 30);
        assert_eq!(plan.total_cost(), 60);
    }

    #[test]
    fn replicated_partial_holders_balance_by_finish_time() {
        // Two partial groups (EP2 GPU halves) both holding experts 0..4 and a
        // full Spark group: the resident experts split across the holders.
        let config = ReplicatedExpertScheduleConfig::with_groups(&[
            ReplicatedExpertGroupSpec::partial(model(10, 0, 16)),
            ReplicatedExpertGroupSpec::partial(model(10, 0, 16)),
            ReplicatedExpertGroupSpec::full(model(20, 0, 16)),
        ])
        .unwrap();
        let both = mask(8, 0..4);
        let mut planner = ReplicatedExpertScheduler::new(config, 8).unwrap();
        let counts = [1, 1, 1, 1, 1, 1, 0, 0];
        let plan = planner
            .plan_resident(&counts, 5, &[both.words(), both.words(), &[]])
            .unwrap();
        assert_eq!(plan.group_loads()[0].expert_count, 2);
        assert_eq!(plan.group_loads()[1].expert_count, 2);
        assert_eq!(plan.group_loads()[2].expert_count, 2);
        // Expert 4 is pinned to the only partial group holding it.
        let mut union = both.clone();
        union.insert(4).unwrap();
        let plan = planner
            .plan_resident(&counts, 5, &[union.words(), both.words(), &[]])
            .unwrap();
        assert_eq!(plan.group_for_expert(4).unwrap().index(), Some(0));
        assert_eq!(plan.group_for_expert(5).unwrap().index(), Some(2));
    }

    #[test]
    fn residency_inputs_are_validated_then_recovered() {
        let mut planner = ReplicatedExpertScheduler::new(tier_config(10, 20, 0, 0), 128).unwrap();
        let counts = [1_u32; 100];
        let set = mask(100, [3, 99]);
        // Partial groups need their sets; plan() has none.
        assert!(planner.plan(&counts, 1).is_err());
        // Wrong set count, a set for the full group, wrong word count.
        assert!(planner.plan_resident(&counts, 1, &[set.words()]).is_err());
        assert!(planner.plan_resident(&counts, 1, &[set.words(), set.words()]).is_err());
        assert!(planner.plan_resident(&counts, 1, &[&set.words()[..1], &[]]).is_err());
        // A bit past the histogram's end.
        let mut past = set.words().to_vec();
        past[1] |= 1 << 40;
        assert!(planner.plan_resident(&counts, 1, &[&past, &[]]).is_err());
        // Recovery: a valid plan behaves as if the failures never happened.
        let recovered = full_snapshot(planner.plan_resident(&counts, 1, &[set.words(), &[]]).unwrap());
        let fresh = full_snapshot(
            ReplicatedExpertScheduler::new(tier_config(10, 20, 0, 0), 128)
                .unwrap()
                .plan_resident(&counts, 1, &[set.words(), &[]])
                .unwrap(),
        );
        assert_eq!(recovered, fresh);
        assert_eq!(recovered.0 .0[3], 0);
        assert_eq!(recovered.0 .0[4], 1);
    }

    #[test]
    fn an_active_expert_no_group_holds_is_rejected() {
        let config = ReplicatedExpertScheduleConfig::with_groups(&[ReplicatedExpertGroupSpec::partial(
            model(10, 0, 16),
        )])
        .unwrap();
        let mut planner = ReplicatedExpertScheduler::new(config, 4).unwrap();
        let set = mask(4, [0, 1]);
        assert!(planner.plan_resident(&[1, 1, 0, 0], 0, &[set.words()]).is_ok());
        assert!(planner.plan_resident(&[1, 1, 1, 0], 0, &[set.words()]).is_err());
        // Inactive experts need no holder.
        assert!(planner.plan_resident(&[0, 3, 0, 0], 0, &[set.words()]).is_ok());
    }

    #[test]
    fn group_specs_are_bounded_and_validated() {
        let full = ReplicatedExpertGroupSpec::full(model(1, 0, 1));
        assert!(ReplicatedExpertScheduleConfig::with_groups(&[]).is_err());
        assert!(ReplicatedExpertScheduleConfig::with_groups(&[full; MAX_REPLICATED_EXPERT_GROUPS as usize]).is_ok());
        assert!(ReplicatedExpertScheduleConfig::with_groups(&[full; MAX_REPLICATED_EXPERT_GROUPS as usize + 1]).is_err());
        assert!(ReplicatedExpertScheduleConfig::with_groups(&[full, ReplicatedExpertGroupSpec::partial(model(0, 0, 1))]).is_err());
        let config = tier_config(1, 2, 3, 4);
        assert_eq!(config.group_count(), 2);
        assert!(config.has_partial_groups());
        assert!(!ReplicatedExpertScheduleConfig::new(3, model(1, 0, 1)).has_partial_groups());
        assert_eq!(config.groups()[0].call_cost, 3);
        assert_eq!(config.groups()[1].residency, ReplicatedExpertResidency::Full);
    }

    #[test]
    fn streaming_cost_model_rounds_bytes_up_to_whole_nanoseconds() {
        // One V4.1 MXFP4 expert's TP4 slice at 278 GB/s, and its TP2 half at 1 TB/s.
        let spark = ReplicatedExpertCostModel::streaming(4_700_160, 278_000_000_000, 0, 16).unwrap();
        assert_eq!(spark.expert_weight_cost, 16_908); // 16_907.05.. ns, rounded up
        let gpu = ReplicatedExpertCostModel::streaming(9_400_320, 1_000_000_000_000, 0, 16).unwrap();
        assert_eq!(gpu.expert_weight_cost, 9_401);
        assert_eq!(gpu.expert_cost(5).unwrap(), 9_401);
        assert_eq!(ReplicatedExpertCostModel::streaming(1, 1, 0, 16).unwrap().expert_weight_cost, 1_000_000_000);
        assert!(ReplicatedExpertCostModel::streaming(1, 0, 0, 16).is_err());
        assert!(ReplicatedExpertCostModel::streaming(0, 1, 0, 16).is_err());
        assert!(ReplicatedExpertCostModel::streaming(u64::MAX, 1, 0, 16).is_err());
    }

    #[test]
    fn residency_sets_round_trip_ids() {
        let set = ExpertResidencySet::from_ids(130, [0, 63, 64, 129]).unwrap();
        assert_eq!(set.len(), 4);
        assert_eq!(set.experts(), 130);
        assert_eq!(set.words().len(), 3);
        assert_eq!(set.iter().collect::<Vec<_>>(), vec![0, 63, 64, 129]);
        assert!(set.contains(129) && !set.contains(128) && !set.contains(130));
        assert!(ExpertResidencySet::from_ids(130, [130]).is_err());
        assert!(ExpertResidencySet::from_ids(130, [5, 5]).is_err());
        let mut grown = ExpertResidencySet::empty(130);
        assert!(grown.is_empty());
        assert!(grown.insert(7).unwrap());
        assert!(!grown.insert(7).unwrap());
        assert!(!grown.is_empty());
    }

    #[test]
    fn production_shape_resident_plans_are_allocation_stable() {
        let resident = mask(384, (0..384).filter(|e| e % 5 == 1));
        let mut planner = ReplicatedExpertScheduler::new(tier_config(9_401, 21_250, 5_000, 34_000), 384).unwrap();
        let counts = histogram(11, 384, 36);
        let _ = planner.plan_resident(&counts, 42, &[resident.words(), &[]]).unwrap();
        let capacities = (
            planner.assignment.capacity(),
            planner.expert_cost.capacity(),
            planner.tie_rank.capacity(),
            planner.order.capacity(),
            planner.group_loads.capacity(),
        );
        let expected = full_snapshot(planner.plan_resident(&counts, 42, &[resident.words(), &[]]).unwrap());
        for _ in 0..512 {
            assert_eq!(
                full_snapshot(planner.plan_resident(&counts, 42, &[resident.words(), &[]]).unwrap()),
                expected
            );
        }
        assert_eq!(
            capacities,
            (
                planner.assignment.capacity(),
                planner.expert_cost.capacity(),
                planner.tie_rank.capacity(),
                planner.order.capacity(),
                planner.group_loads.capacity(),
            ),
            "a resident plan within capacity must not reallocate"
        );
    }
}
