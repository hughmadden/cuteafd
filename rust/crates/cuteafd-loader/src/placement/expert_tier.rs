//! Static GPU expert tier: some routed experts of every layer stay resident on the coordinator
//! GPU while the Sparks keep every expert. Per layer and step the coordinator runs the active
//! resident experts beside the Sparks' wave and the outputs are summed; the set is fixed for the
//! server's lifetime (dynamic swapping and prefetch measured as losses on DeepSeek V4.1).
//!
//! The set is ranked from a deployment's own route profile: the executed-route census
//! (`cuteafd_core::expert_usage`) as JSON `{"geometry": {...}, "decode": [..], "prefill": [..]}`,
//! layer-major `[layer * experts + expert]` counts. A cell's score is its share of the profile's
//! decode routed rows plus [`PREFILL_WEIGHT`] times its share of the prefill rows: decode steps are
//! where the Sparks' wave bounds each layer, prefill gains in proportion to the rows anyway. Cells
//! no row ever chose are never resident.
use anyhow::{ensure, Context, Result};
use cuteafd_core::expert_usage::Geometry;
use std::path::{Path, PathBuf};

/// Weight of a cell's prefill share beside its decode share.
pub const PREFILL_WEIGHT: f64 = 0.25;

/// `--expert-tier`: off, or the route profile to rank the tier from.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum TierSetting {
    #[default]
    Off,
    Profile(PathBuf),
}

impl std::str::FromStr for TierSetting {
    type Err = String;
    fn from_str(text: &str) -> std::result::Result<Self, String> {
        match text.trim() {
            "" => Err("expert tier: give off or a route profile path".into()),
            "off" => Ok(Self::Off),
            path => Ok(Self::Profile(PathBuf::from(path))),
        }
    }
}

impl TierSetting {
    pub fn profile(&self) -> Option<&Path> {
        match self { Self::Off => None, Self::Profile(path) => Some(path) }
    }
}

/// `--expert-tier-bytes`: the coordinator bytes the tier may hold (its package workspace and
/// resident experts). `auto`: what `RTX_EXPERT_LAYERS=auto` would spend on whole RTX layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TierBytes {
    #[default]
    Auto,
    Bytes(u64),
}

impl std::str::FromStr for TierBytes {
    type Err = String;
    fn from_str(text: &str) -> std::result::Result<Self, String> {
        let text = text.trim();
        if text == "auto" { return Ok(Self::Auto); }
        let gib: f64 = text.parse().map_err(|_| format!("expert tier bytes: give auto or GiB, not {text:?}"))?;
        if !(gib.is_finite() && gib > 0.0 && gib <= 4096.0) {
            return Err(format!("expert tier bytes: {text} GiB is outside (0, 4096]"));
        }
        Ok(Self::Bytes((gib * (1u64 << 30) as f64) as u64))
    }
}

impl std::fmt::Display for TierBytes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Auto => f.write_str("auto"),
            Self::Bytes(bytes) => write!(f, "{}", *bytes as f64 / (1u64 << 30) as f64),
        }
    }
}

/// An executed-route census exported for ranking.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct RouteProfile {
    pub geometry: Geometry,
    pub decode: Vec<u64>,
    pub prefill: Vec<u64>,
    #[serde(default)]
    pub semantics: Option<String>,
}

impl RouteProfile {
    pub fn read(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path).with_context(|| format!("expert tier profile {}", path.display()))?;
        Self::parse(&bytes).with_context(|| format!("expert tier profile {}", path.display()))
    }

    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let profile: Self = serde_json::from_slice(bytes).context("route profile JSON")?;
        ensure!(profile.geometry.validate(), "route profile geometry is invalid");
        ensure!(profile.semantics.as_deref().is_none_or(|s| s == "executed_routed_rows"),
            "route profile semantics {:?} are not executed routed rows", profile.semantics);
        let cells = profile.geometry.cells().context("route profile cells")?;
        ensure!(profile.decode.len() == cells && profile.prefill.len() == cells,
            "route profile holds {} decode and {} prefill cells, its geometry {cells}",
            profile.decode.len(), profile.prefill.len());
        let routed = |layer: usize| profile.geometry.routed_layers.binary_search(&layer).is_ok();
        let experts = profile.geometry.experts;
        ensure!((0..cells).all(|cell| routed(cell / experts) || (profile.decode[cell] == 0 && profile.prefill[cell] == 0)),
            "route profile counts routes on a dense layer");
        ensure!(profile.decode.iter().chain(&profile.prefill).any(|&n| n > 0), "route profile counted no routes");
        Ok(profile)
    }

    /// Refuse a profile taken on another geometry (layers, experts, top-k, routed layers).
    pub fn require_geometry(&self, served: &Geometry) -> Result<()> {
        ensure!(&self.geometry == served,
            "expert tier profile geometry {:?} differs from the served model's {:?}", self.geometry, served);
        Ok(())
    }

    /// Per cell: decode share + [`PREFILL_WEIGHT`] x prefill share.
    pub fn scores(&self) -> Vec<f64> {
        let share = |counts: &[u64]| {
            let total = counts.iter().sum::<u64>().max(1) as f64;
            counts.iter().map(move |&n| n as f64 / total).collect::<Vec<_>>()
        };
        share(&self.decode).into_iter().zip(share(&self.prefill))
            .map(|(decode, prefill)| decode + PREFILL_WEIGHT * prefill).collect()
    }

    /// The shares of the profile's (decode, prefill) routed rows that `selection` holds.
    pub fn held(&self, selection: &TierSelection) -> (f64, f64) {
        let experts = self.geometry.experts;
        let share = |counts: &[u64]| {
            let total = counts.iter().sum::<u64>().max(1) as f64;
            selection.layers.iter().flat_map(|(layer, ids)| ids.iter().map(move |&id| counts[layer * experts + id]))
                .sum::<u64>() as f64 / total
        };
        (share(&self.decode), share(&self.prefill))
    }
}

/// The resident experts per routed layer, ascending, and their exact device bytes.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TierSelection {
    /// `(layer, strictly increasing expert ids)`, layers ascending, none empty.
    pub layers: Vec<(usize, Vec<usize>)>,
    /// Exact resident arena bytes of every layer's subset (the package workspace is separate).
    pub resident_bytes: u64,
}

impl TierSelection {
    pub fn experts(&self) -> usize { self.layers.iter().map(|(_, ids)| ids.len()).sum() }

    /// Whether `expert` of `layer` is resident.
    pub fn holds(&self, layer: usize, expert: usize) -> bool {
        self.layers.binary_search_by_key(&layer, |(l, _)| *l)
            .is_ok_and(|at| self.layers[at].1.binary_search(&expert).is_ok())
    }
}

/// Costs [`select`] charges: an upper bound per expert and per non-empty layer, and the exact
/// bytes of a layer's subset (the residency builder's arena). Bounds must never undercount.
pub trait TierCosts {
    fn expert_bound(&self, layer: usize, expert: usize) -> Result<u64>;
    fn layer_bound(&self, layer: usize) -> Result<u64>;
    fn exact(&self, layer: usize, experts: &[usize]) -> Result<u64>;
}

/// Greedy selection by score under `budget` resident bytes: candidates in descending score (ties
/// to the lower layer, then expert), each admitted while the running upper bound fits; then the
/// exact arenas are charged (never above the bound, so never above the budget).
pub fn select(geometry: &Geometry, scores: &[f64], budget: u64, costs: &impl TierCosts) -> Result<TierSelection> {
    let experts = geometry.experts;
    ensure!(scores.len() == geometry.cells().context("tier cells")?, "tier scores differ from the geometry");
    let mut order: Vec<usize> = (0..scores.len()).filter(|&cell| scores[cell] > 0.0
        && geometry.routed_layers.binary_search(&(cell / experts)).is_ok()).collect();
    order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
    let mut picked: std::collections::BTreeMap<usize, Vec<usize>> = Default::default();
    let mut bound = 0u64;
    for cell in order {
        let (layer, expert) = (cell / experts, cell % experts);
        let first = !picked.contains_key(&layer);
        let add = costs.expert_bound(layer, expert)?
            .checked_add(if first { costs.layer_bound(layer)? } else { 0 }).context("tier bound overflow")?;
        if bound.checked_add(add).is_some_and(|total| total <= budget) {
            bound += add;
            picked.entry(layer).or_default().push(expert);
        }
    }
    let mut selection = TierSelection::default();
    for (layer, mut ids) in picked {
        ids.sort_unstable();
        let bytes = costs.exact(layer, &ids)?;
        selection.resident_bytes = selection.resident_bytes.checked_add(bytes).context("tier bytes overflow")?;
        selection.layers.push((layer, ids));
    }
    ensure!(selection.resident_bytes <= bound && selection.resident_bytes <= budget,
        "expert tier exact bytes {} exceed their bound {bound} or the budget {budget}", selection.resident_bytes);
    Ok(selection)
}

/// EXL3 costs over a checkpoint's own headers (`V41Exl3Manifest::subset_residency`).
pub struct Exl3TierCosts<'m>(pub &'m crate::V41Exl3Manifest);

impl TierCosts for Exl3TierCosts<'_> {
    fn expert_bound(&self, layer: usize, expert: usize) -> Result<u64> { self.0.tier_expert_bytes(layer, expert) }
    fn layer_bound(&self, _layer: usize) -> Result<u64> { self.0.tier_layer_bytes() }
    fn exact(&self, layer: usize, experts: &[usize]) -> Result<u64> { self.0.subset_arena_bytes(layer, experts) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geometry() -> Geometry {
        Geometry { layers: 4, experts: 8, topk: 2, routed_layers: vec![1, 2, 3] }
    }

    fn profile(decode: Vec<u64>, prefill: Vec<u64>) -> String {
        serde_json::json!({"family": "test", "semantics": "executed_routed_rows", "geometry": geometry(),
            "decode": decode, "prefill": prefill}).to_string()
    }

    struct Flat { expert: u64, layer: u64, slack: u64 }
    impl TierCosts for Flat {
        fn expert_bound(&self, _: usize, expert: usize) -> Result<u64> { Ok(self.expert + expert as u64) }
        fn layer_bound(&self, _: usize) -> Result<u64> { Ok(self.layer) }
        fn exact(&self, _: usize, ids: &[usize]) -> Result<u64> {
            Ok(ids.iter().map(|&e| self.expert + e as u64).sum::<u64>() + self.layer - self.slack)
        }
    }

    #[test]
    fn settings_parse_off_paths_auto_and_gib_and_refuse_nonsense() {
        assert_eq!("off".parse::<TierSetting>().unwrap(), TierSetting::Off);
        assert_eq!("/p/x.json".parse::<TierSetting>().unwrap().profile(), Some(Path::new("/p/x.json")));
        assert!("".parse::<TierSetting>().is_err());
        assert_eq!("auto".parse::<TierBytes>().unwrap(), TierBytes::Auto);
        assert_eq!("4.5".parse::<TierBytes>().unwrap(), TierBytes::Bytes(9 << 29));
        assert_eq!(TierBytes::Bytes(9 << 29).to_string(), "4.5");
        for bad in ["0", "-1", "nan", "inf", "5000", "4.5GiB", ""] {
            assert!(bad.parse::<TierBytes>().is_err(), "{bad}");
        }
    }

    #[test]
    fn profiles_validate_geometry_cells_dense_layers_and_scores() {
        let cells = 32;
        let mut decode = vec![0; cells];
        let mut prefill = vec![0; cells];
        decode[8 + 3] = 30; decode[16] = 10; prefill[24 + 7] = 40;
        let parsed = RouteProfile::parse(profile(decode.clone(), prefill.clone()).as_bytes()).unwrap();
        parsed.require_geometry(&geometry()).unwrap();
        let mut other = geometry();
        other.topk = 4;
        assert!(parsed.require_geometry(&other).is_err());
        let scores = parsed.scores();
        assert!((scores[11] - 0.75).abs() < 1e-12 && (scores[16] - 0.25).abs() < 1e-12);
        assert!((scores[31] - PREFILL_WEIGHT).abs() < 1e-12 && scores[0] == 0.0);
        assert!(RouteProfile::parse(profile(decode[..31].to_vec(), prefill.clone()).as_bytes()).is_err());
        let mut dense = decode.clone();
        dense[2] = 1;
        assert!(RouteProfile::parse(profile(dense, prefill.clone()).as_bytes()).is_err(), "layer 0 is dense");
        assert!(RouteProfile::parse(profile(vec![0; cells], vec![0; cells]).as_bytes()).is_err());
        let wrong = profile(decode, prefill).replace("executed_routed_rows", "committed_tokens");
        assert!(RouteProfile::parse(wrong.as_bytes()).is_err());
    }

    #[test]
    fn greedy_selection_follows_scores_within_the_budget_and_charges_exact_bytes() {
        let geometry = geometry();
        let mut scores = vec![0.0; 32];
        // Layer 1: experts 3 (hot), 5 and 0 (cold); layer 2: expert 0; layer 3: expert 7.
        for (cell, score) in [(11, 0.5), (13, 0.2), (16, 0.3), (31, 0.1), (8, 0.05), (0, 0.9)] { scores[cell] = score; }
        let costs = Flat { expert: 100, layer: 50, slack: 7 };
        // Cell 0 is dense layer 0: never resident. Ranking: 11, 16, 13, 31, 8.
        let all = select(&geometry, &scores, u64::MAX, &costs).unwrap();
        assert_eq!(all.layers, vec![(1, vec![0, 3, 5]), (2, vec![0]), (3, vec![7])]);
        assert_eq!(all.experts(), 5);
        assert_eq!(all.resident_bytes, (100 + 103 + 105 + 50 - 7) + (100 + 50 - 7) + (107 + 50 - 7));
        // 153 fits the first; 150 more fits layer 2; layer 1's second (105) then fits; 157 does not.
        let some = select(&geometry, &scores, 153 + 150 + 105, &costs).unwrap();
        assert_eq!(some.layers, vec![(1, vec![3, 5]), (2, vec![0])]);
        assert!(some.holds(1, 5) && !some.holds(3, 7) && !some.holds(0, 0));
        // Candidates that do not fit (13, 31) are skipped and a cheaper later one (8) still enters.
        let skip = select(&geometry, &scores, 153 + 150 + 100, &costs).unwrap();
        assert_eq!(skip.layers, vec![(1, vec![0, 3]), (2, vec![0])]);
        assert_eq!(select(&geometry, &scores, 10, &costs).unwrap(), TierSelection::default());
        assert!(select(&geometry, &scores[..31], 1000, &costs).is_err());
        struct Under;
        impl TierCosts for Under {
            fn expert_bound(&self, _: usize, _: usize) -> Result<u64> { Ok(1) }
            fn layer_bound(&self, _: usize) -> Result<u64> { Ok(0) }
            fn exact(&self, _: usize, _: &[usize]) -> Result<u64> { Ok(100) }
        }
        assert!(select(&geometry, &scores, 1000, &Under).is_err(), "an undercounting bound is refused");
    }

    #[test]
    fn held_shares_count_the_profile_rows_the_selection_keeps() {
        let mut decode = vec![0; 32];
        decode[11] = 3; decode[12] = 1;
        let mut prefill = vec![0; 32];
        prefill[11] = 1; prefill[20] = 3;
        let parsed = RouteProfile::parse(profile(decode, prefill).as_bytes()).unwrap();
        let selection = TierSelection { layers: vec![(1, vec![3])], resident_bytes: 0 };
        assert_eq!(parsed.held(&selection), (0.75, 0.25));
    }
}
