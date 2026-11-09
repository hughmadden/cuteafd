//! MiMo prewarm inputs keyed by checkpoint geometry and resolved layout.
use crate::families::mimo_v2::decode_graph::{MIMO_DECODE_ROWS, MIMO_GRAPH_LEAD_OBSERVED, MIMO_GRAPH_PEER_OBSERVED};
use crate::families::mimo_v2::config::MimoV2Config;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MimoPrewarmEntry {
    pub rows: usize,
    pub pin: bool,
}

/// Measured prewarm arena per physical rank, plus one driver-pool granule.
/// Runtime capture headroom is added by the shared graph reserve policy.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MimoLayoutPrewarm {
    pub variant: String,
    pub ranks: usize,
    pub sparks: bool,
    pub reserve_base_bytes: Vec<u64>,
    pub reserve_extra_bytes: Vec<u64>,
    /// Per-rank measured row shares, in entry order; absent for bootstrap keys.
    #[serde(default)]
    pub entry_capture_bytes: Vec<Vec<u64>>,
    pub entries: Vec<MimoPrewarmEntry>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MimoPrewarmList {
    pub schema_version: u32,
    pub family: String,
    pub layouts: Vec<MimoLayoutPrewarm>,
}

/// Export geometry comes from the checkpoint's own config, never its HF name.
/// Layer count distinguishes checkpoints sharing the same arithmetic programs.
pub fn mimo_graph_variant(cfg: &MimoV2Config) -> String {
    format!("{}-{}", cfg.program_family().unwrap_or("unqualified"), cfg.layers)
}

pub fn mimo_prewarm_list() -> MimoPrewarmList {
    let list: MimoPrewarmList = serde_json::from_str(include_str!("mimo_prewarm.json"))
        .expect("checked-in MiMo prewarm list");
    assert_eq!(list.schema_version, 2);
    assert_eq!(list.family, "mimo_v2");
    let mut keys = std::collections::HashSet::new();
    for layout in &list.layouts {
        assert!(keys.insert((&layout.variant, layout.ranks, layout.sparks)), "duplicate MiMo prewarm key");
        assert!((1..=2).contains(&layout.ranks), "MiMo prewarm layout ranks");
        assert_eq!(layout.reserve_base_bytes.len(), layout.ranks);
        assert_eq!(layout.reserve_extra_bytes, vec![2 << 20; layout.ranks]);
        assert_eq!(layout.entry_capture_bytes.len(), layout.ranks);
        assert!(layout.entry_capture_bytes.iter().all(|shares| shares.len() == layout.entries.len()));
        let mut rows = std::collections::HashSet::new();
        for entry in &layout.entries {
            assert!((1..=MIMO_DECODE_ROWS).contains(&entry.rows) && rows.insert(entry.rows), "MiMo prewarm rows");
        }
    }
    list
}

/// Untraced keys retain the old startup rows and an explicitly provisional
/// byte estimate scaled by executable count. They never use another key's list.
pub fn mimo_prewarm_layout(cfg: &MimoV2Config, ranks: usize, sparks: bool) -> MimoLayoutPrewarm {
    let variant = mimo_graph_variant(cfg);
    if let Some(layout) = mimo_prewarm_list().layouts.into_iter()
        .find(|layout| layout.variant == variant && layout.ranks == ranks && layout.sparks == sparks) {
        return layout;
    }
    tracing::warn!(%variant, ranks, sparks, "MiMo graph prewarm key is untraced; using bootstrap startup set and byte estimate");
    let mut reserve_base_bytes = vec![bootstrap_bytes(cfg.layers + 2, MIMO_GRAPH_LEAD_OBSERVED)];
    if ranks == 2 { reserve_base_bytes.push(bootstrap_bytes(cfg.layers, MIMO_GRAPH_PEER_OBSERVED)); }
    MimoLayoutPrewarm {
        variant,
        ranks,
        sparks,
        reserve_base_bytes,
        reserve_extra_bytes: vec![2 << 20; ranks],
        entry_capture_bytes: Vec::new(),
        entries: (1..=MIMO_DECODE_ROWS).map(|rows| MimoPrewarmEntry { rows, pin: false }).collect(),
    }
}

fn bootstrap_bytes(segments: usize, observed: (u64, u64)) -> u64 {
    let bytes = observed.1.div_ceil(observed.0).saturating_mul(segments as u64).saturating_mul(MIMO_DECODE_ROWS as u64);
    bytes.div_ceil(2 << 20) * (2 << 20)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_in_layouts_roundtrip() {
        let list = mimo_prewarm_list();
        let _: MimoPrewarmList = serde_json::from_value(serde_json::to_value(&list).unwrap()).unwrap();
    }

    #[test]
    fn checkpoint_geometry_and_onboarding_are_independent_keys() {
        let flash = MimoV2Config::from_hf(&crate::plan::testing::mimo_flash_mopd_config()).unwrap();
        let pro = MimoV2Config::from_hf(&crate::plan::testing::mimo_pro_config()).unwrap();
        assert_eq!(mimo_graph_variant(&flash), format!("mimof-{}", flash.layers));
        assert_eq!(mimo_graph_variant(&pro), format!("mimop-{}", pro.layers));
        for cfg in [&flash, &pro] {
            for ranks in [1, 2] {
                for sparks in [false, true] {
                    let layout = mimo_prewarm_layout(cfg, ranks, sparks);
                    assert_eq!(layout.variant, mimo_graph_variant(cfg));
                    assert_eq!(layout.sparks, sparks);
                    assert_eq!(layout.reserve_base_bytes.len(), ranks);
                    assert_eq!(layout.reserve_extra_bytes, vec![2 << 20; ranks]);
                }
            }
        }
    }

    #[test]
    fn untraced_flash_tp2_layout_bootstraps_without_pins() {
        let mut cfg = MimoV2Config::from_hf(&crate::plan::testing::mimo_flash_mopd_config()).unwrap();
        cfg.layers = 48;
        let variant = mimo_graph_variant(&cfg);
        for sparks in [false, true] {
            assert!(!mimo_prewarm_list().layouts.iter().any(|layout|
                layout.variant == variant && layout.ranks == 2 && layout.sparks == sparks));
            let layout = mimo_prewarm_layout(&cfg, 2, sparks);
            assert_eq!(layout.variant, variant);
            assert_eq!(layout.ranks, 2);
            assert_eq!(layout.sparks, sparks);
            assert_eq!(layout.entries.iter().map(|entry| entry.rows).collect::<Vec<_>>(), (1..=64).collect::<Vec<_>>());
            assert!(layout.entries.iter().all(|entry| !entry.pin));
            let resolved = crate::families::mimo_v2::decode_graph::graph_layout(&cfg, 2, sparks, true).unwrap();
            assert!(resolved.pinned.is_empty());
            assert_eq!(resolved.rows, (1..=64).collect::<Vec<_>>());
        }
    }

    #[test]
    fn untraced_variant_bootstraps_instead_of_borrowing() {
        let mut cfg = MimoV2Config::from_hf(&crate::plan::testing::mimo_flash_mopd_config()).unwrap();
        cfg.layers = 47;
        let layout = mimo_prewarm_layout(&cfg, 2, true);
        assert_eq!(layout.variant, "mimof-47");
        assert_eq!(layout.entries.iter().map(|entry| entry.rows).collect::<Vec<_>>(), (1..=64).collect::<Vec<_>>());
        assert!(layout.entries.iter().all(|entry| !entry.pin));
        assert!(layout.reserve_base_bytes.iter().all(|&bytes| bytes > 0));
    }
}
