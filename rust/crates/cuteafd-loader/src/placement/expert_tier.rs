//! GPU expert tier: per-layer residency of routed experts on the coordinator
//! GPUs beside the Sparks, which keep every expert.
//!
//! Whole-layer onboarding leaves one memory pool idle in every layer: the
//! Sparks during a local layer, the GPUs during a remote one. A tier
//! generalises it: each routed layer holds some share of its experts on the
//! GPUs (a whole layer is a share of 100%), so both pools stream weights in
//! parallel inside each layer. Per layer and step, an active expert runs on the
//! GPUs when it is resident there and on the Sparks otherwise
//! (`cuteafd_core::ReplicatedExpertScheduler::plan_resident`).
//!
//! - Which experts: [`optimized_order`] ranks every (layer, expert) slot by
//!   how much its residency cuts the predicted step time, the sum over layers
//!   of the slower pool ([`TierStepModel`]), from per-expert activations: a
//!   routing profile ([`ExpertRouteProfile`]) or uniform popularity. Every
//!   prefix of the order is then the greedy optimum for its budget. Uniform
//!   popularity spreads evenly over the layers, which saves ~1.65x what the
//!   same bytes save as whole layers. `--rtx-expert-layers N` pins N whole
//!   layers ahead of the ranked slots.
//! - How many: the solver admits the longest prefix that fits beside the
//!   pool target and still helps (`auto`, pool first), or exactly `N` per
//!   routed layer, in which case the KV pool is the output.
//! - Memory: exact per-expert residency from the layer's checkpoint-derived
//!   [`ExpertCost`] (TP2 halves on both GPUs, or whole experts on one GPU),
//!   in its own arena per GPU with the RTX expert workspace and the largest
//!   per-expert load staging.
//! - The set is fixed for a server's lifetime (re-ranked only at restart):
//!   where an expert runs is then a function of (layer, expert) alone, so a
//!   prefix prefilled now or restored from a snapshot ran every expert on the
//!   same device.
use super::ExpertCost;
use serde::{Deserialize, Serialize};

/// The `--gpu-expert-tier` spelling: off, sized from what the pool target
/// leaves (`auto`), or an exact number of resident experts per routed layer
/// (the KV pool is then the output).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "tier", content = "value")]
pub enum ExpertTierSetting {
    #[default]
    Off,
    Auto,
    PerLayer(usize),
}

impl ExpertTierSetting {
    pub fn enabled(self) -> bool {
        self != Self::Off
    }

    /// The solver's size for `routed` routed layers; `None` when off.
    pub fn size(self, routed: usize) -> Option<ExpertTierSize> {
        match self {
            Self::Off => None,
            Self::Auto => Some(ExpertTierSize::Auto),
            Self::PerLayer(n) => Some(ExpertTierSize::Experts(n.saturating_mul(routed))),
        }
    }
}

impl std::fmt::Display for ExpertTierSetting {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Off => f.write_str("off"),
            Self::Auto => f.write_str("auto"),
            Self::PerLayer(n) => write!(f, "{n}"),
        }
    }
}

impl std::str::FromStr for ExpertTierSetting {
    type Err = String;
    fn from_str(text: &str) -> Result<Self, String> {
        match text.trim() {
            "off" | "" => Ok(Self::Off),
            "auto" => Ok(Self::Auto),
            other => match other.parse::<usize>() {
                Ok(0) => Ok(Self::Off),
                Ok(n) => Ok(Self::PerLayer(n)),
                Err(_) => Err(format!("GPU expert tier must be off, auto or N experts per layer (got {other:?})")),
            },
        }
    }
}

/// How many experts the solver admits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "size", content = "experts")]
pub enum ExpertTierSize {
    /// Pool first: reserve the pool target, then the longest useful prefix
    /// that fits.
    Auto,
    /// Exactly this many (a prefix of the order); the pool is the output.
    Experts(usize),
}

/// Where the expert activations behind the order came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExpertTierSource {
    Uniform,
    Profile,
}

/// One resident expert: a backbone layer index (into the request's layers)
/// and an expert id.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TierSlot {
    pub layer: u16,
    pub expert: u16,
}

/// The tier a family asks the solver to admit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertTierRequest {
    pub size: ExpertTierSize,
    /// Routed experts per layer; a layer's per-expert bytes are its
    /// [`ExpertCost`] divided by this count, exactly.
    pub experts_per_layer: usize,
    /// Admission order, highest priority first; each slot at most once.
    pub order: Vec<TierSlot>,
    /// Leading slots that must be admitted: whole layers pinned by
    /// `--rtx-expert-layers`.
    pub pinned: usize,
    /// The longest prefix an automatic tier admits: past it the predicted
    /// step time stops improving.
    pub useful: usize,
    pub source: ExpertTierSource,
}

impl ExpertTierRequest {
    pub fn new(size: ExpertTierSize, experts_per_layer: usize, order: TierOrder, source: ExpertTierSource) -> Self {
        Self { size, experts_per_layer, pinned: order.pinned, useful: order.useful, order: order.order, source }
    }
}

/// How resident experts are split across the coordinator GPUs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "shard")]
pub enum ExpertTierShard {
    /// Half the intermediate width of every resident expert on each GPU; the
    /// halves meet in the layer's existing FFN combine.
    Tp2,
    /// Whole experts on one GPU.
    Whole { gpu: u8 },
}

/// What the solver admitted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpertTierPlacement {
    pub shard: ExpertTierShard,
    pub source: ExpertTierSource,
    /// Resident expert ids per backbone layer, ascending (empty for dense
    /// layers and layers the prefix did not reach).
    pub resident: Vec<Vec<u16>>,
    /// Resident experts over all layers.
    pub experts: usize,
    /// Routed layers whose every expert is resident (no Spark wave).
    pub whole_layers: usize,
    /// Routed experts over all routed layers.
    pub routed_experts: usize,
    /// Resident expert bytes per GPU (no workspace or staging).
    pub resident_bytes: Vec<u64>,
    /// Each GPU's tier arena peak: the RTX expert workspace, the resident
    /// experts and the largest per-expert load staging (zero when empty).
    pub peak_bytes: Vec<u64>,
    /// Resident share of all routed expert bytes, parts per million.
    pub byte_share_ppm: u32,
}

impl ExpertTierPlacement {
    /// Resident experts in each backbone layer, in layer order.
    pub fn per_layer(&self) -> impl Iterator<Item = usize> + '_ {
        self.resident.iter().map(Vec::len)
    }

    /// The `cuteafd plan --layout` summary.
    pub fn summary(&self) -> String {
        let counts = self.resident.iter().filter(|ids| !ids.is_empty()).map(Vec::len).collect::<Vec<_>>();
        let (lo, hi) = (counts.iter().min().copied().unwrap_or(0), counts.iter().max().copied().unwrap_or(0));
        let per_layer = if lo == hi { format!("{lo}") } else { format!("{lo}-{hi}") };
        let shard = match self.shard {
            ExpertTierShard::Tp2 => "TP2 halves on rtx0 and rtx1".to_string(),
            ExpertTierShard::Whole { gpu } => format!("whole experts on rtx{gpu}"),
        };
        format!("GPU expert tier ({:?}): {} of {} routed experts resident ({:.1}% of routed bytes), \
            {per_layer} per routed layer ({} whole), {shard}; resident bytes {:?}, arena peaks {:?}",
            self.source, self.experts, self.routed_experts, f64::from(self.byte_share_ppm) / 1e4,
            self.whole_layers, self.resident_bytes, self.peak_bytes)
    }
}

/// One step's expected expert activations, in units of `ACTIVATION_ONE`.
pub const ACTIVATION_ONE: u64 = 1 << 20;

/// Per-layer step costs of the two pools, in nanoseconds. A pool that runs no
/// expert of a layer costs nothing there; one that does costs its fixed part
/// plus a cost per expert. The pools run in parallel, so a layer costs the
/// slower one, and a step the sum over layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TierStepModel {
    /// One resident active expert on the coordinator (TP2: both GPUs stream
    /// their halves in parallel).
    pub gpu_expert_ns: u64,
    /// The coordinator's fixed part (launch, route reduce) per layer.
    pub gpu_fixed_ns: u64,
    /// One active expert on the Sparks (every TP rank streams its slice).
    pub spark_expert_ns: u64,
    /// The Spark round trip's fixed part (transport and host) per layer.
    pub spark_fixed_ns: u64,
    /// Distinct experts one step activates per layer.
    pub active_experts: u32,
    /// Rows of the step, for activations from a routing profile.
    pub step_rows: u32,
}

impl TierStepModel {
    /// The C1 verify step (5-6 rows, ~24 distinct experts per layer) as
    /// measured on a 48 GB coordinator pair and four Sparks at TP4: the SM89
    /// W4A8 expert kernel runs 24 TP2 halves in 235 us (~9.55 us each plus
    /// ~6 us, 959 GB/s per card); a 24-expert Spark wave takes 510 us of
    /// kernel (21.25 us each) plus 34 us of transport and host.
    pub const C1_VERIFY_48GB_PAIR_4_SPARKS: Self = Self {
        gpu_expert_ns: 9_550,
        gpu_fixed_ns: 6_000,
        spark_expert_ns: 21_250,
        spark_fixed_ns: 34_000,
        active_experts: 24,
        step_rows: 6,
    };

    /// One layer's time in `ns * ACTIVATION_ONE` for expected `gpu` and
    /// `spark` activations (in `ACTIVATION_ONE` units).
    fn layer_scaled(&self, gpu: u64, spark: u64) -> u128 {
        let side = |active: u64, fixed: u64, each: u64| if active == 0 { 0 } else {
            u128::from(fixed) * u128::from(ACTIVATION_ONE) + u128::from(each) * u128::from(active)
        };
        side(gpu, self.gpu_fixed_ns, self.gpu_expert_ns).max(side(spark, self.spark_fixed_ns, self.spark_expert_ns))
    }

    /// Predicted step time in nanoseconds over the routed layers with the
    /// given resident sets (one per routed layer, as `activation`'s rows).
    pub fn step_ns(&self, activation: &ExpertActivation, resident: &[&[u16]]) -> f64 {
        let none: &[u16] = &[];
        activation.layers.iter().enumerate().map(|(index, row)| {
            let ids = resident.get(index).copied().unwrap_or(none);
            let gpu = ids.iter().map(|&id| row[usize::from(id)]).sum::<u64>();
            let all = row.iter().sum::<u64>();
            self.layer_scaled(gpu, all - gpu) as f64 / ACTIVATION_ONE as f64
        }).sum()
    }

    /// Predicted step time with every routed layer whole on the GPUs, the
    /// whole-layer onboarding endpoint.
    pub fn whole_step_ns(&self, activation: &ExpertActivation) -> f64 {
        activation.layers.iter()
            .map(|row| self.layer_scaled(row.iter().sum(), 0) as f64 / ACTIVATION_ONE as f64).sum()
    }
}

/// Expected activations: for each routed layer and expert, the share of
/// steps that run it, in `ACTIVATION_ONE` units (a row sums to the experts a
/// step activates in that layer).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertActivation {
    pub layers: Vec<Vec<u64>>,
}

impl ExpertActivation {
    /// Uniform popularity: every expert active in `active / experts` of steps.
    pub fn uniform(routed_layers: usize, experts: usize, active: u32) -> Self {
        let each = u64::from(active) * ACTIVATION_ONE / experts.max(1) as u64;
        Self { layers: vec![vec![each; experts]; routed_layers] }
    }

    /// A routing profile at a step of `rows` rows of `topk` routes: an expert
    /// chosen by a share `p` of rows is read by a step with probability
    /// `1 - (1 - p)^rows`, scaled down per layer so a step activates at most
    /// `active` experts (serving steps share experts across adjacent rows,
    /// which independent draws over-count).
    pub fn from_profile(profile: &ExpertRouteProfile, routed_layers: usize, experts: usize, rows: u32, topk: u32,
        active: u32) -> Result<Self, ExpertTierError> {
        profile.validate(routed_layers, experts)?;
        let layers = profile.layers.iter().map(|counts| {
            let total = counts.iter().sum::<u64>() as f64;
            let read = counts.iter().map(|&count| if total == 0.0 { 0.0 } else {
                1.0 - (1.0 - (count as f64 * f64::from(topk) / total).min(1.0)).powi(rows as i32)
            }).collect::<Vec<_>>();
            let sum = read.iter().sum::<f64>();
            let scale = if sum > 0.0 { (f64::from(active) / sum).min(1.0) } else { 0.0 };
            read.into_iter().map(|p| ((p * scale).min(1.0) * ACTIVATION_ONE as f64).round() as u64).collect()
        }).collect();
        Ok(Self { layers })
    }

    /// Expected share of the expert bytes a step reads from `resident` (one
    /// set per routed layer): the byte hit rate.
    pub fn hit_rate(&self, resident: &[&[u16]]) -> f64 {
        let (mut hit, mut all) = (0u128, 0u128);
        for (row, ids) in self.layers.iter().zip(resident) {
            all += row.iter().map(|&a| u128::from(a)).sum::<u128>();
            hit += ids.iter().map(|&id| u128::from(row[usize::from(id)])).sum::<u128>();
        }
        if all == 0 { 0.0 } else { hit as f64 / all as f64 }
    }
}

/// [`optimized_order`]'s result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TierOrder {
    /// Every routed slot, best first.
    pub order: Vec<TierSlot>,
    /// Leading slots of pinned whole layers.
    pub pinned: usize,
    /// Leading slots whose residency still cut the predicted step time.
    pub useful: usize,
}

/// Ranks every (layer, expert) slot of `layers` (backbone indices, one per
/// activation row) by how much its residency cuts `model`'s predicted step
/// time, greedily: each pick takes the slot that cuts the most given those
/// before it (within a layer, the most active expert first). A layer's time
/// falls while its Spark side is the slower one and rises past the balance
/// point, so the picks' cuts never grow and every prefix is the greedy optimum
/// for its budget; `useful` ends where a pick stops helping. Ties go to the
/// layer holding fewer resident experts, then the lower layer and expert id,
/// so uniform popularity spreads evenly in ascending ids. The first
/// `pinned_layers` layers come first, whole.
pub fn optimized_order(layers: &[usize], activation: &ExpertActivation, model: &TierStepModel,
    pinned_layers: usize) -> TierOrder {
    use std::cmp::Reverse;
    let rows = &activation.layers[..activation.layers.len().min(layers.len())];
    let slot = |position: usize, expert: usize| TierSlot { layer: layers[position] as u16, expert: expert as u16 };
    let mut order = Vec::new();
    for (position, row) in rows.iter().enumerate().take(pinned_layers) {
        order.extend((0..row.len()).map(|expert| slot(position, expert)));
    }
    let pinned = order.len();
    struct Layer { ranked: Vec<usize>, taken: usize, gpu: u64, spark: u64 }
    let mut state = rows.iter().map(|row| {
        let mut ranked = (0..row.len()).collect::<Vec<_>>();
        ranked.sort_by_key(|&expert| (Reverse(row[expert]), expert));
        Layer { ranked, taken: 0, gpu: 0, spark: row.iter().sum() }
    }).collect::<Vec<_>>();
    // One candidate per layer: (cut, fewer resident first, lower layer, lower expert).
    type Candidate = (i128, Reverse<usize>, Reverse<usize>, Reverse<usize>);
    let candidate = |position: usize, layer: &Layer| -> Option<Candidate> {
        let expert = *layer.ranked.get(layer.taken)?;
        let active = rows[position][expert];
        let before = model.layer_scaled(layer.gpu, layer.spark);
        let after = model.layer_scaled(layer.gpu + active, layer.spark - active);
        Some((before as i128 - after as i128, Reverse(layer.taken), Reverse(position), Reverse(expert)))
    };
    let mut heap = std::collections::BinaryHeap::new();
    for (position, layer) in state.iter().enumerate().skip(pinned_layers) {
        heap.extend(candidate(position, layer));
    }
    let mut useful = None;
    while let Some((cut, _, Reverse(position), Reverse(expert))) = heap.pop() {
        if cut <= 0 && useful.is_none() { useful = Some(order.len()); }
        order.push(slot(position, expert));
        let layer = &mut state[position];
        let active = rows[position][expert];
        layer.taken += 1;
        layer.gpu += active;
        layer.spark -= active;
        heap.extend(candidate(position, layer));
    }
    let useful = useful.unwrap_or(order.len()).max(pinned);
    TierOrder { order, pinned, useful }
}

/// Per routed layer route counts from serving traffic (`--gpu-expert-tier-profile`):
/// how many routed rows chose each expert. JSON: `{"layers": [[count; experts]; routed layers]}`
/// in routed-layer order; other keys (model, prompts, tokens) are ignored.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpertRouteProfile {
    pub layers: Vec<Vec<u64>>,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ExpertTierError {
    #[error("expert route profile: {0}")]
    Profile(String),
}

impl ExpertRouteProfile {
    pub fn read(path: &std::path::Path) -> Result<Self, ExpertTierError> {
        let bytes = std::fs::read(path)
            .map_err(|error| ExpertTierError::Profile(format!("reading {}: {error}", path.display())))?;
        serde_json::from_slice(&bytes)
            .map_err(|error| ExpertTierError::Profile(format!("parsing {}: {error}", path.display())))
    }

    /// One row per routed layer, one count per expert, some route counted.
    pub fn validate(&self, routed_layers: usize, experts: usize) -> Result<(), ExpertTierError> {
        if self.layers.len() != routed_layers {
            return Err(ExpertTierError::Profile(format!(
                "{} layers of counts for {routed_layers} routed layers", self.layers.len())));
        }
        if let Some((layer, row)) = self.layers.iter().enumerate().find(|(_, row)| row.len() != experts) {
            return Err(ExpertTierError::Profile(format!(
                "routed layer {layer} has {} counts for {experts} experts", row.len())));
        }
        if self.layers.iter().flatten().all(|&count| count == 0) {
            return Err(ExpertTierError::Profile("no routes counted".into()));
        }
        Ok(())
    }

    /// Share of the profile's routes that land on `resident` (one set per
    /// routed layer): the byte hit rate of one-row steps.
    pub fn route_share(&self, resident: &[&[u16]]) -> f64 {
        let (mut hit, mut all) = (0u128, 0u128);
        for (counts, ids) in self.layers.iter().zip(resident) {
            all += counts.iter().map(|&c| u128::from(c)).sum::<u128>();
            hit += ids.iter().map(|&id| u128::from(counts[usize::from(id)])).sum::<u128>();
        }
        if all == 0 { 0.0 } else { hit as f64 / all as f64 }
    }
}

/// One resident expert's bytes on `gpu` for the layer's `cost`, exact: the
/// layer's residency must divide evenly over its experts.
pub(super) fn per_expert(cost: &ExpertCost, tp2: bool, gpu: usize, experts: usize)
    -> Option<super::Bytes2> {
    let part = if tp2 { cost.half[gpu] } else { cost.whole };
    (experts > 0 && part.resident % experts as u64 == 0)
        .then(|| super::Bytes2 { resident: part.resident / experts as u64, staging: part.staging })
}

#[cfg(test)]
mod tests {
    use super::*;

    const MODEL: TierStepModel = TierStepModel::C1_VERIFY_48GB_PAIR_4_SPARKS;

    #[test]
    fn setting_spellings_round_trip() {
        for (text, setting) in [("off", ExpertTierSetting::Off), ("auto", ExpertTierSetting::Auto),
            ("81", ExpertTierSetting::PerLayer(81)), ("0", ExpertTierSetting::Off)] {
            assert_eq!(text.parse::<ExpertTierSetting>().unwrap(), setting);
        }
        assert_eq!(ExpertTierSetting::PerLayer(81).to_string(), "81");
        assert!("half".parse::<ExpertTierSetting>().is_err());
        assert_eq!(ExpertTierSetting::PerLayer(81).size(40), Some(ExpertTierSize::Experts(3240)));
        assert_eq!(ExpertTierSetting::Auto.size(40), Some(ExpertTierSize::Auto));
        assert_eq!(ExpertTierSetting::Off.size(40), None);
        assert!(!ExpertTierSetting::default().enabled());
    }

    #[test]
    fn uniform_popularity_spreads_evenly_in_ascending_ids() {
        let layers = [1, 2, 5];
        let tier = optimized_order(&layers, &ExpertActivation::uniform(3, 384, 24), &MODEL, 0);
        assert_eq!(tier.order.len(), 3 * 384);
        assert_eq!(tier.pinned, 0);
        for prefix in 0..=tier.useful {
            let mut counts = [0; 6];
            for slot in &tier.order[..prefix] { counts[slot.layer as usize] += 1; }
            let used = layers.map(|l| counts[l]);
            assert!(used.iter().max().unwrap() - used.iter().min().unwrap() <= 1, "{prefix}: {used:?}");
        }
        assert_eq!(&tier.order[..4], &[TierSlot { layer: 1, expert: 0 }, TierSlot { layer: 2, expert: 0 },
            TierSlot { layer: 5, expert: 0 }, TierSlot { layer: 1, expert: 1 }]);
        // Residency helps until the GPUs become the slower side: 9.55 us x
        // 24 x share + 6 us = 34 us + 21.25 us x 24 x (1 - share) at ~72%.
        let per_layer = tier.useful / 3;
        assert!((270..=280).contains(&per_layer), "useful {per_layer} per layer");
    }

    #[test]
    fn a_profile_puts_the_most_active_experts_first_in_any_layer() {
        // One-row, top-1 steps: an expert's activation is its share of routes.
        let mut counts = vec![vec![1u64; 8]; 2];
        counts[1][5] = 100;
        counts[0][2] = 40;
        let profile = ExpertRouteProfile { layers: counts };
        let activation = ExpertActivation::from_profile(&profile, 2, 8, 1, 1, 1).unwrap();
        let tier = optimized_order(&[3, 7], &activation, &MODEL, 0);
        assert_eq!(&tier.order[..2], &[TierSlot { layer: 7, expert: 5 }, TierSlot { layer: 3, expert: 2 }]);
        assert!(activation.layers.iter().all(|row| row.iter().sum::<u64>().abs_diff(ACTIVATION_ONE) < 16));
        let resident: Vec<&[u16]> = vec![&[2], &[5]];
        assert!((activation.hit_rate(&resident) - (40.0 / 47.0 + 100.0 / 107.0) / 2.0).abs() < 1e-5);
        assert!((profile.route_share(&resident) - 140.0 / 154.0).abs() < 1e-12);
        // Independent rows over-count a step's experts: scaled down to `active`.
        let busy = ExpertActivation::from_profile(&profile, 2, 8, 6, 2, 1).unwrap();
        assert!(busy.layers.iter().all(|row| row.iter().sum::<u64>().abs_diff(ACTIVATION_ONE) < 16));
        assert!(ExpertActivation::from_profile(&profile, 2, 9, 6, 2, 3).is_err());
        assert!(ExpertActivation::from_profile(&profile, 3, 8, 6, 2, 3).is_err());
        let empty = ExpertRouteProfile { layers: vec![vec![0; 8]; 2] };
        assert!(empty.validate(2, 8).is_err());
        assert!(profile.validate(3, 8).is_err());
    }

    #[test]
    fn a_concentrated_profile_stops_before_the_gpus_become_the_bottleneck() {
        // 16 experts take ~98% of the routes: a step runs ~15 of them. Keeping
        // all 16 resident would leave the GPUs the slower side (149 us against
        // 51 us), so the order balances near 13 and only then spills over.
        let mut counts = vec![vec![1u64; 384]; 2];
        for row in &mut counts { row[300..316].fill(1000); }
        let profile = ExpertRouteProfile { layers: counts };
        let activation = ExpertActivation::from_profile(&profile, 2, 384, 6, 6, 24).unwrap();
        let tier = optimized_order(&[0, 1], &activation, &MODEL, 0);
        assert!(tier.order[..tier.useful].iter().all(|slot| (300..316).contains(&slot.expert)));
        assert!((24..=28).contains(&tier.useful), "useful {}", tier.useful);
        let (thirteen, all_hot) = ((300..313).collect::<Vec<u16>>(), (300..316).collect::<Vec<u16>>());
        let best = MODEL.step_ns(&activation, &[&thirteen, &thirteen]);
        assert!(best < MODEL.step_ns(&activation, &[&all_hot, &all_hot]));
    }

    #[test]
    fn pinned_whole_layers_lead_the_order() {
        let tier = optimized_order(&[0, 1, 2], &ExpertActivation::uniform(3, 16, 4), &MODEL, 1);
        assert_eq!(tier.pinned, 16);
        assert!(tier.order[..16].iter().all(|slot| slot.layer == 0));
        assert!(tier.order[16..].iter().all(|slot| slot.layer != 0));
        assert_eq!(tier.order.len(), 48);
        assert!(tier.useful >= tier.pinned);
    }

    #[test]
    fn step_model_matches_the_measured_round_trips() {
        let uniform = ExpertActivation::uniform(40, 384, 24);
        // All on the Sparks: 34 us + 24 x 21.25 us per layer.
        assert!((MODEL.step_ns(&uniform, &[]) - 40.0 * 544_000.0).abs() < 1.0);
        // Whole layers on the GPUs: 6 us + 24 x 9.55 us.
        assert!((MODEL.whole_step_ns(&uniform) - 40.0 * 235_200.0).abs() < 1.0);
    }

    /// The tier's physics on a V4.1 C1 verify step (~29 ms on the 5090 card
    /// with four Sparks; ~24 distinct experts per layer), from bytes and
    /// rates, through the scheduler itself: `plan_resident` per layer, and a
    /// layer costs its makespan, the slower of the GPU tier and the Spark
    /// round trip. Rates measured tonight: the SM89 W4A8 kernel streams TP2
    /// halves at 959 GB/s per 4090 D (9.55 us per expert plus ~6 us); a Spark
    /// TP4 wave takes 21.25 us per expert plus 34 us. The resident layouts
    /// (MXFP4, coded UE8M0 scales): a TP4 slice is 4,573,312 B, so the Spark
    /// wave streams 215 GB/s per Spark, 77% of its 278 GB/s roofline.
    #[test]
    fn c1_verify_step_model_reproduces_the_tier_physics() {
        use cuteafd_core::{ExpertResidencySet, ReplicatedExpertCostModel as Cost, ReplicatedExpertGroupSpec as Group,
            ReplicatedExpertScheduleConfig, ReplicatedExpertScheduler};
        const LAYERS: usize = 40;
        const EXPERTS: usize = 384;
        const ACTIVE: usize = 24;
        let slice = crate::formats::scale_codes::expert_bytes(1, 5120, 576).unwrap();
        assert_eq!(slice, 4_573_312);
        let spark_rate = slice * ACTIVE as u64 * 1_000_000_000 / 510_000;
        assert_eq!(spark_rate / 1_000_000_000, 215);
        // The checkpoint-layout TP2 half the measured kernel streams.
        let half = 9_400_320u64;
        let gpu_rate = half * 1_000_000_000 / MODEL.gpu_expert_ns;
        assert_eq!(gpu_rate / 1_000_000_000, 984);
        let spark = Group::full(Cost::streaming(slice, spark_rate, 0, 16).unwrap()).with_call_cost(MODEL.spark_fixed_ns);
        let gpu = Group::partial(Cost::streaming(half, gpu_rate, 0, 16).unwrap()).with_call_cost(MODEL.gpu_fixed_ns);
        let config = ReplicatedExpertScheduleConfig::with_groups(&[gpu, spark]).unwrap();
        let mut scheduler = ReplicatedExpertScheduler::new(config, EXPERTS).unwrap();

        // Deterministic active sets: `ACTIVE` distinct experts per layer, each
        // drawn from the resident set with probability `hit` (None: uniform).
        let mut state = 0x5eed_u64;
        let mut next = move || { state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state; z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB); z ^ (z >> 31) };
        let mut step_ns = |resident: &[ExpertResidencySet], hit: Option<f64>, steps: usize| -> f64 {
            let mut total = 0u64;
            for _ in 0..steps {
                for set in resident {
                    let (hot, cold): (Vec<usize>, Vec<usize>) = (0..EXPERTS).partition(|&e| set.contains(e));
                    let mut counts = [0u32; EXPERTS];
                    let mut chosen = 0;
                    while chosen < ACTIVE {
                        let pool = hit.map(|p| if (next() >> 11) as f64 / (1u64 << 53) as f64 <= p && !hot.is_empty()
                            || cold.is_empty() { &hot } else { &cold });
                        let expert = match pool { None => (next() % EXPERTS as u64) as usize,
                            Some(pool) => pool[(next() % pool.len() as u64) as usize] };
                        if counts[expert] == 0 { counts[expert] = 1 + (chosen % 2) as u32; chosen += 1; }
                    }
                    total += scheduler.plan_resident(&counts, 0, &[set.words(), &[]]).unwrap().makespan();
                }
            }
            total as f64 / steps as f64
        };
        let ms = |ns: f64| ns / 1e6;
        let nothing = vec![ExpertResidencySet::empty(EXPERTS); LAYERS];
        let all_spark = step_ns(&nothing, None, 1);
        assert!((all_spark / LAYERS as f64 - 544_000.0).abs() < 100.0, "{all_spark}");
        let whole_layer = |layers: usize| {
            let mut sets = nothing.clone();
            for set in &mut sets[..layers] { *set = ExpertResidencySet::from_ids(EXPERTS, 0..EXPERTS).unwrap(); }
            sets
        };
        // One whole layer: 24 experts on the GPUs (235 us), no Spark wave.
        let per_whole = all_spark - step_ns(&whole_layer(1), None, 1);
        assert!((per_whole - 308_800.0).abs() < 200.0, "{per_whole}");
        // Seven whole layers.
        let seven = ms(all_spark - step_ns(&whole_layer(7), None, 1));
        assert!((2.1..=2.25).contains(&seven), "7 whole layers save {seven:.2} ms");

        // ~21% residency (the ~60 GB first estimated for the pair): a uniform
        // tier of 81 experts per layer against the same bytes as whole layers.
        let uniform = vec![ExpertResidencySet::from_ids(EXPERTS, 0..81).unwrap(); LAYERS];
        let tier = ms(all_spark - step_ns(&uniform, None, 50));
        let whole = ms(per_whole) * 81.0 * LAYERS as f64 / EXPERTS as f64;
        assert!((4.1..=4.45).contains(&tier), "uniform tier saves {tier:.2} ms");
        assert!((2.5..=2.7).contains(&whole), "the same bytes as whole layers save {whole:.2} ms");
        assert!((1.55..=1.75).contains(&(tier / whole)), "spread/whole {:.2}", tier / whole);
        // The planner's analytic step model agrees with the scheduler.
        let activation = ExpertActivation::uniform(LAYERS, EXPERTS, ACTIVE as u32);
        let ids = (0..81u16).collect::<Vec<_>>();
        let resident = vec![ids.as_slice(); LAYERS];
        let predicted = ms(MODEL.step_ns(&activation, &[]) - MODEL.step_ns(&activation, &resident));
        assert!((predicted - tier).abs() < 0.1, "predicted {predicted:.2} vs scheduled {tier:.2} ms");

        // Skew: 40% of the active expert bytes resident.
        let skew = ms(all_spark - step_ns(&uniform, Some(0.4), 50));
        assert!((7.8..=8.4).contains(&skew), "40% hit rate saves {skew:.2} ms");

        // What today's planner admits on a 48 GB pair: 2,399 resident halves
        // (60 per layer, 15.6%) against 6 whole TP2 layers.
        let planned = vec![ExpertResidencySet::from_ids(EXPERTS, 0..60).unwrap(); LAYERS];
        let planned_tier = ms(all_spark - step_ns(&planned, None, 50));
        let planned_whole = ms(all_spark - step_ns(&whole_layer(6), None, 1));
        assert!((3.0..=3.4).contains(&planned_tier) && (1.8..=1.9).contains(&planned_whole),
            "planned tier {planned_tier:.2} ms, whole {planned_whole:.2} ms");

        // Prefill touches every expert: the GPUs take the tier's share of the
        // bytes in parallel, and the layer's expert phase shrinks by about it.
        let mut wide = [64u32; EXPERTS];
        wide[0] = 65;
        let spark_only = scheduler.plan_resident(&wide, 0, &[nothing[0].words(), &[]]).unwrap().makespan();
        let shared = scheduler.plan_resident(&wide, 0, &[uniform[0].words(), &[]]).unwrap();
        assert!(shared.group_loads()[0].cost < shared.group_loads()[1].cost);
        let shrink = 1.0 - shared.makespan() as f64 / spark_only as f64;
        assert!((0.20..=0.215).contains(&shrink), "prefill expert phase shrinks {shrink:.3}");
    }

    #[test]
    fn per_expert_bytes_divide_the_layer_exactly() {
        let cost = ExpertCost { whole: super::super::Bytes2 { resident: 384 * 18_800_640, staging: 7 },
            half: [super::super::Bytes2 { resident: 384 * 9_400_320, staging: 5 }; 2], tp2: true, spark_ok: true };
        assert_eq!(per_expert(&cost, true, 1, 384), Some(super::super::Bytes2 { resident: 9_400_320, staging: 5 }));
        assert_eq!(per_expert(&cost, false, 0, 384), Some(super::super::Bytes2 { resident: 18_800_640, staging: 7 }));
        assert_eq!(per_expert(&cost, true, 0, 385), None);
        assert_eq!(per_expert(&cost, true, 0, 0), None);
    }
}
