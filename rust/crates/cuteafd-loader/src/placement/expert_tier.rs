//! GPU expert tier: a fixed share of every routed layer's experts resident on
//! the coordinator GPUs beside the Sparks, which keep every expert.
//!
//! Whole-layer onboarding leaves one memory pool idle in every layer: the
//! Sparks during a local layer, the GPUs during a remote one. A tier holds some
//! experts of every layer instead, so both pools stream weights in parallel
//! inside each layer: per layer and step, an active expert runs on the GPUs when
//! it is resident there and on the Sparks otherwise
//! (`cuteafd_core::ReplicatedExpertScheduler::plan_resident`).
//!
//! - Which experts: an admission order ([`ExpertTierRequest::order`]). Uniform
//!   ([`uniform_order`]) takes expert ids in ascending order round-robin over
//!   the routed layers, so every prefix holds the same count per layer to
//!   within one; a routing profile ([`ExpertRouteProfile`], [`profile_order`])
//!   takes the most-routed (layer, expert) pairs first.
//! - How many: the solver admits the longest prefix that fits beside the
//!   pool target (`auto`, pool first), or exactly `N` per routed layer, in
//!   which case the KV pool is the output (as `--rtx-expert-layers N`).
//! - Memory: exact per-expert residency from the layer's checkpoint-derived
//!   [`ExpertCost`] (TP2 halves on both GPUs, or whole experts on one GPU),
//!   in its own arena per GPU with the RTX expert workspace and the largest
//!   per-expert load staging.
//! - The set is fixed for a server's lifetime: where an expert runs is then a
//!   function of (layer, expert) alone, so a prefix prefilled now or restored
//!   from a snapshot ran every expert on the same device.
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
    /// Pool first: reserve the pool target, then the longest prefix that fits.
    Auto,
    /// Exactly this many (a prefix of the order); the pool is the output.
    Experts(usize),
}

/// Where the admission order came from.
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
    pub source: ExpertTierSource,
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
    /// Resident experts in each routed layer, in layer order.
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
            {per_layer} per routed layer, {shard}; resident bytes {:?}, arena peaks {:?}",
            self.source, self.experts, self.routed_experts, f64::from(self.byte_share_ppm) / 1e4,
            self.resident_bytes, self.peak_bytes)
    }
}

/// Expert ids ascending, round-robin over `layers`: every prefix holds the
/// same number of experts per layer to within one (the earlier layers first).
/// Without a routing profile every choice of ids is equivalent in expectation.
pub fn uniform_order(layers: &[usize], experts: usize) -> Vec<TierSlot> {
    (0..experts)
        .flat_map(|expert| layers.iter().map(move |&layer| TierSlot { layer: layer as u16, expert: expert as u16 }))
        .collect()
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

    /// Share of the profile's routes that land on `resident` (per routed
    /// layer, ascending ids): the byte hit rate of one-row steps.
    pub fn route_share(&self, resident: &[&[u16]]) -> f64 {
        self.hit_rate(resident, None)
    }

    /// Expected share of the expert bytes a step of `rows` rows reads from
    /// resident experts, if rows route independently with the profile's
    /// per-layer frequencies (top-k without replacement approximated by
    /// independent draws): an expert is read once per step when any row
    /// chose it, `1 - (1 - p)^rows` with `p` its share of the layer's rows.
    pub fn step_hit_rate(&self, resident: &[&[u16]], rows: u32, topk: u32) -> f64 {
        self.hit_rate(resident, Some((rows, topk)))
    }

    fn hit_rate(&self, resident: &[&[u16]], step: Option<(u32, u32)>) -> f64 {
        let (mut hit, mut all) = (0.0, 0.0);
        for (counts, ids) in self.layers.iter().zip(resident) {
            let total = counts.iter().sum::<u64>() as f64;
            if total == 0.0 { continue; }
            let weight = |count: u64| match step {
                None => count as f64,
                Some((rows, topk)) => {
                    // Per-row probability of choosing the expert: topk draws per row.
                    let p = (count as f64 * f64::from(topk) / total).min(1.0);
                    1.0 - (1.0 - p).powi(rows as i32)
                }
            };
            all += counts.iter().map(|&c| weight(c)).sum::<f64>();
            hit += ids.iter().map(|&id| weight(counts[id as usize])).sum::<f64>();
        }
        if all == 0.0 { 0.0 } else { hit / all }
    }
}

/// The most-routed (layer, expert) pairs first over all routed layers; ties
/// by layer, then expert id. `layers` maps the profile's rows to backbone
/// layer indices.
pub fn profile_order(layers: &[usize], experts: usize, profile: &ExpertRouteProfile)
    -> Result<Vec<TierSlot>, ExpertTierError> {
    profile.validate(layers.len(), experts)?;
    let mut ranked = layers.iter().zip(&profile.layers).flat_map(|(&layer, counts)| {
        counts.iter().enumerate().map(move |(expert, &count)| (count, TierSlot { layer: layer as u16, expert: expert as u16 }))
    }).collect::<Vec<_>>();
    ranked.sort_by(|(left, a), (right, b)| right.cmp(left).then_with(|| a.cmp(b)));
    Ok(ranked.into_iter().map(|(_, slot)| slot).collect())
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
    fn uniform_prefixes_stay_within_one_expert_per_layer() {
        let layers = [1, 2, 5];
        let order = uniform_order(&layers, 4);
        assert_eq!(order.len(), 12);
        for prefix in 0..=order.len() {
            let mut counts = [0; 6];
            for slot in &order[..prefix] { counts[slot.layer as usize] += 1; }
            let used = layers.map(|l| counts[l]);
            assert!(used.iter().max().unwrap() - used.iter().min().unwrap() <= 1, "{prefix}: {used:?}");
        }
        assert_eq!(&order[..4], &[TierSlot { layer: 1, expert: 0 }, TierSlot { layer: 2, expert: 0 },
            TierSlot { layer: 5, expert: 0 }, TierSlot { layer: 1, expert: 1 }]);
    }

    #[test]
    fn profile_order_takes_the_most_routed_pairs_first() {
        let profile = ExpertRouteProfile { layers: vec![vec![5, 0, 9, 1], vec![9, 2, 0, 0]] };
        let order = profile_order(&[3, 7], 4, &profile).unwrap();
        let slots = |pairs: &[(u16, u16)]| pairs.iter().map(|&(layer, expert)| TierSlot { layer, expert }).collect::<Vec<_>>();
        // 9 (layer 3 expert 2) ties 9 (layer 7 expert 0): lower layer first.
        assert_eq!(order[..5], slots(&[(3, 2), (7, 0), (3, 0), (7, 1), (3, 3)])[..]);
        assert_eq!(order.len(), 8);
        assert!(profile_order(&[3], 4, &profile).is_err());
        assert!(profile_order(&[3, 7], 5, &profile).is_err());
        assert!(profile_order(&[3, 7], 4, &ExpertRouteProfile { layers: vec![vec![0; 4]; 2] }).is_err());
    }

    #[test]
    fn hit_rates_count_routes_and_step_reads() {
        let profile = ExpertRouteProfile { layers: vec![vec![60, 30, 10, 0]] };
        let hot: &[u16] = &[0];
        assert!((profile.route_share(&[hot]) - 0.6).abs() < 1e-12);
        // One row, top-1: the step reads exactly the routed expert.
        assert!((profile.step_hit_rate(&[hot], 1, 1) - 0.6).abs() < 1e-12);
        // Many rows read every expert with routes: the hot one is 1 of 3.
        assert!((profile.step_hit_rate(&[hot], 4096, 1) - 1.0 / 3.0).abs() < 1e-9);
        assert_eq!(profile.route_share(&[&[]]), 0.0);
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
