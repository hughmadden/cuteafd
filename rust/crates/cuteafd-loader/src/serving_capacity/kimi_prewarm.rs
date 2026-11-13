//! Kimi K3 prewarm shapes and per-layout graph reserve.
//!
//! Each core format and coordinator layout captures a different inventory: TP2
//! peer segments and RTX expert onboarding change which sites exist, so the
//! checked-in list carries one row per layout and admission reserves that
//! row's arena on every physical device.
//!
//! `reserve_base_bytes` holds each physical rank's measured prewarm arena;
//! `reserve_extra_bytes` is the 2 MiB allocation tail. Shared
//! GPU_GRAPH_RESERVE_EXTRA supplies runtime headroom; GPU_GRAPH_RESERVE
//! overrides the whole reserve. Bases and per-entry costs come from the seeded
//! adaptive min/max GRAPH_TRACE run; regenerate with
//! `scripts/dev/graph-prewarm-from-trace.py --family kimi_k3`.
//!
//! Listed rows preserve observed operation and verify variant within the
//! native 64-row replay band. Sequence layout stays in device metadata. Prefill
//! runs eagerly inside a row window: its keys never replay and are not prewarmed (#1646).

use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum KimiCoreFormat {
    Bf16, K6, Trellis,
    #[serde(rename = "original")] McOriginal,
    #[serde(rename = "k6-compact")] McK6,
    #[serde(rename = "k8-compact")] McK8,
    #[serde(rename = "fp8-compact")] McFp8,
}

impl KimiCoreFormat {
    /// Header-native trellis inventory, not a repository name.
    pub fn from_inventory(core: &crate::families::kimi_k3::core::CoreInventory) -> Self {
        if let Some(view) = core.selected_core_view {
            use crate::plan::checkpoint::CoreView;
            return match view { CoreView::Original => Self::McOriginal, CoreView::K6Compact => Self::McK6,
                CoreView::K8Compact => Self::McK8, CoreView::Fp8Compact => Self::McFp8 };
        }
        if core.projections.is_empty() { Self::Bf16 }
        else if core.projections.values().all(|p| p.bits == 6) { Self::K6 }
        else { Self::Trellis }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum KimiPrewarmOp { Ffn, Attention, Context }

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct KimiPrewarmEntry {
    pub rows: usize,
    /// Verify-shaped attention or context capture; false for flagless sites.
    pub speculative: bool,
    pub pin: bool,
    /// Absent only in the provisional startup bootstrap set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub op: Option<KimiPrewarmOp>,
    /// Sum of measured capture shares per logical rank. Zero is allocator reuse;
    /// an absent rank falls back to the provisional per-executable census.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub capture_bytes: Vec<Option<u64>>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct KimiPrewarmLayout {
    pub core: KimiCoreFormat,
    /// Coordinator ranks this row was measured on.
    pub ranks: usize,
    pub layout: String,
    pub reserve_base_bytes: Vec<u64>,
    pub reserve_extra_bytes: u64,
    pub entries: Vec<KimiPrewarmEntry>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct KimiPrewarmList {
    pub schema_version: u32,
    pub family: String,
    pub layouts: Vec<KimiPrewarmLayout>,
}

fn kimi_prewarm_list() -> &'static KimiPrewarmList {
    static LIST: OnceLock<KimiPrewarmList> = OnceLock::new();
    LIST.get_or_init(|| {
        let list: KimiPrewarmList = serde_json::from_str(include_str!("kimi_prewarm.json"))
            .expect("checked-in Kimi K3 prewarm list");
        assert_eq!(list.schema_version, 1);
        assert_eq!(list.family, "kimi_k3");
        let mut keys = std::collections::HashSet::new();
        for layout in &list.layouts {
            assert!(keys.insert((layout.core, layout.ranks)), "duplicate Kimi prewarm key");
            assert_eq!(layout.reserve_base_bytes.len(), layout.ranks, "Kimi per-rank graph bases");
        }
        list
    })
}

/// An untraced layout prewarms unpinned bootstrap geometry, never another
/// layout's geometries or pins.
pub fn kimi_layout(core: KimiCoreFormat, ranks: usize) -> &'static KimiPrewarmLayout {
    let list = kimi_prewarm_list();
    list.layouts.iter().find(|layout| layout.core == core && layout.ranks == ranks)
        .unwrap_or_else(|| {
            tracing::warn!(?core, ranks, "Kimi graph prewarm trace missing for core/layout; using bootstrap startup set");
            kimi_bootstrap_layout(core)
        })
}

fn kimi_bootstrap_layout(core: KimiCoreFormat) -> &'static KimiPrewarmLayout {
    static BF16: OnceLock<KimiPrewarmLayout> = OnceLock::new();
    static K6: OnceLock<KimiPrewarmLayout> = OnceLock::new();
    static TRELLIS: OnceLock<KimiPrewarmLayout> = OnceLock::new();
    static ORIGINAL: OnceLock<KimiPrewarmLayout> = OnceLock::new();
    static MC_K6: OnceLock<KimiPrewarmLayout> = OnceLock::new();
    static MC_K8: OnceLock<KimiPrewarmLayout> = OnceLock::new();
    static MC_FP8: OnceLock<KimiPrewarmLayout> = OnceLock::new();
    let bootstrap = match core { KimiCoreFormat::Bf16 => &BF16, KimiCoreFormat::K6 => &K6, KimiCoreFormat::Trellis => &TRELLIS,
        KimiCoreFormat::McOriginal => &ORIGINAL, KimiCoreFormat::McK6 => &MC_K6,
        KimiCoreFormat::McK8 => &MC_K8, KimiCoreFormat::McFp8 => &MC_FP8 };
    bootstrap.get_or_init(|| KimiPrewarmLayout {
        core,
        ranks: 0,
        layout: "bootstrap startup set (untraced)".into(),
        reserve_base_bytes: vec![],
        reserve_extra_bytes: 2 << 20,
        entries: [1, 4, 16, 64].into_iter().map(|rows|
            KimiPrewarmEntry { rows, speculative: false, pin: false, op: None, capture_bytes: vec![] })
            .chain((2..=64).map(|rows|
                KimiPrewarmEntry { rows, speculative: true, pin: false, op: None, capture_bytes: vec![] })).collect(),
    })
}

/// Per-device measured prewarm base and allocation tail.
pub fn kimi_reserve(core: KimiCoreFormat, ranks: usize, rank: usize) -> (u64, u64) {
    assert!(rank < ranks, "Kimi graph reserve rank");
    let layout = kimi_layout(core, ranks);
    let base = if layout.ranks == 0 {
        crate::families::kimi_k3::allocation::graph_census(ranks, rank).0
    } else {
        layout.reserve_base_bytes[rank]
    };
    (base, layout.reserve_extra_bytes)
}

/// Select an exact traced operation/attention geometry. Bootstrap entries have
/// no operation and retain the original startup inventory; flagless sites can
/// share a row across decode and verify but attention variants cannot.
pub fn kimi_selected(layout: &KimiPrewarmLayout, rows: usize, speculative: bool,
    op: KimiPrewarmOp, pinned: bool) -> bool {
    (1..=64).contains(&rows) && layout.entries.iter().any(|entry| {
        (!pinned || entry.pin) && entry.rows == rows && match entry.op {
            Some(kind) => kind == op && entry.speculative == speculative,
            None => op == KimiPrewarmOp::Ffn
                || entry.speculative == speculative,
        }
    })
}

/// Provisional per-capture estimate before the cache has a measurement: the
/// layout's startup census divided by its captured executables. That is a
/// byte basis (Kimi has no calibrated node rate for this route), so the cache
/// itself still selects by physical growth once a capture completes.
pub fn kimi_bootstrap(ranks: usize, rank: usize) -> u64 {
    let (bytes, executables) = crate::families::kimi_k3::allocation::graph_census(ranks, rank);
    bytes.div_ceil(executables.max(1)).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_layout_names_its_shapes_and_reserves_each_physical_rank() {
        let list = kimi_prewarm_list();
        assert!(list.layouts.len() >= 2);
        for layout in &list.layouts {
            assert!(layout.ranks >= 1 && !layout.layout.is_empty());
            assert_eq!(layout.reserve_base_bytes.len(), layout.ranks);
            assert_eq!(layout.reserve_extra_bytes, 2 << 20);
            assert!(!layout.entries.is_empty());
            assert!(layout.entries.iter().all(|entry| entry.rows > 0));
        }
        for layout in &list.layouts {
            for rank in 0..layout.ranks {
                assert_eq!(kimi_reserve(layout.core, layout.ranks, rank).0, layout.reserve_base_bytes[rank]);
            }
        }
        assert_eq!(kimi_layout(KimiCoreFormat::K6, 9).ranks, 0, "untraced layouts use bootstrap, not another row");
        assert_eq!(kimi_reserve(KimiCoreFormat::K6, 9, 0).0,
            crate::families::kimi_k3::allocation::graph_census(9, 0).0);
        assert_eq!(kimi_reserve(KimiCoreFormat::K6, 1, 0).1, 2 << 20);
        assert_eq!(kimi_reserve(KimiCoreFormat::K6, 2, 1).1, 2 << 20);
    }

    #[test]
    fn untraced_core_never_borrows_the_k6_rank_inventory_or_base() {
        let layout = kimi_layout(KimiCoreFormat::Bf16, 1);
        assert_eq!(layout.core, KimiCoreFormat::Bf16);
        assert_eq!(layout.ranks, 0);
        assert!(layout.entries.iter().all(|entry| !entry.pin && entry.capture_bytes.is_empty()));
        assert_eq!(kimi_reserve(KimiCoreFormat::Bf16, 1, 0).0,
            crate::families::kimi_k3::allocation::graph_census(1, 0).0);
        assert_ne!(kimi_reserve(KimiCoreFormat::Bf16, 1, 0), kimi_reserve(KimiCoreFormat::K6, 1, 0));
        assert_eq!(KimiCoreFormat::from_inventory(&Default::default()), KimiCoreFormat::Bf16);
    }

    #[test]
    fn each_mc_view_has_its_own_untraced_bootstrap() {
        use crate::plan::checkpoint::CoreView;
        for (view, format) in [(CoreView::Original, KimiCoreFormat::McOriginal),
            (CoreView::K6Compact, KimiCoreFormat::McK6), (CoreView::K8Compact, KimiCoreFormat::McK8),
            (CoreView::Fp8Compact, KimiCoreFormat::McFp8)] {
            let core = crate::families::kimi_k3::core::CoreInventory {
                selected_core_view: Some(view), ..Default::default()
            };
            assert_eq!(KimiCoreFormat::from_inventory(&core), format);
            assert_eq!(serde_json::to_value(format).unwrap(), view.name());
            for ranks in [1, 2] {
                let layout = kimi_layout(format, ranks);
                assert_eq!(layout.core, format);
                assert_eq!(layout.ranks, 0);
                assert!(layout.entries.iter().all(|e| !e.pin && e.capture_bytes.is_empty()));
            }
        }
    }

    #[test]
    fn bootstrap_only_contains_replayable_rows() {
        let layout = kimi_bootstrap_layout(KimiCoreFormat::Bf16);
        for entry in &layout.entries {
            assert!((1..=64).contains(&entry.rows));
        }
        for rows in [0, 65, 1024, 3072, 4096] {
            assert!(!kimi_selected(layout, rows, false, KimiPrewarmOp::Ffn, false));
        }
    }

    #[test]
    fn untraced_layout_prewarms_bootstrap_without_pinning() {
        let layout = kimi_layout(KimiCoreFormat::K6, 9);
        assert_eq!(layout.ranks, 0);
        assert!(!layout.entries.is_empty());
        assert!(layout.entries.iter().all(|entry| !entry.pin));
        for entry in &layout.entries {
            for op in [KimiPrewarmOp::Ffn,
                KimiPrewarmOp::Attention, KimiPrewarmOp::Context] {
                assert!(kimi_selected(layout, entry.rows, entry.speculative, op, false));
                assert!(!kimi_selected(layout, entry.rows, entry.speculative, op, true));
            }
        }
    }

    #[test]
    fn selection_stays_within_the_layout_operation_and_verify_variant() {
        let layout = KimiPrewarmLayout { core: KimiCoreFormat::K6, ranks: 1, layout: "test".into(),
            reserve_base_bytes: vec![1], reserve_extra_bytes: 2 << 20,
            entries: vec![KimiPrewarmEntry { rows: 20, speculative: true, pin: true,
                op: Some(KimiPrewarmOp::Attention), capture_bytes: vec![] }] };
        let select = |rows, spec, op| kimi_selected(&layout, rows, spec, op, false);
        assert!(select(20, true, KimiPrewarmOp::Attention));
        assert!(!select(20, false, KimiPrewarmOp::Attention));
        assert!(!select(20, true, KimiPrewarmOp::Context));
        assert!(!select(4, true, KimiPrewarmOp::Attention));
    }

    #[test]
    fn context_entry_selects_only_its_exact_context_variant() {
        let layout = KimiPrewarmLayout { core: KimiCoreFormat::K6, ranks: 2, layout: "context".into(),
            reserve_base_bytes: vec![1], reserve_extra_bytes: 2 << 20,
            entries: vec![KimiPrewarmEntry { rows: 16, speculative: false, pin: true,
                op: Some(KimiPrewarmOp::Context), capture_bytes: vec![] }] };
        assert!(kimi_selected(&layout, 16, false, KimiPrewarmOp::Context, false));
        assert!(!kimi_selected(&layout, 16, true, KimiPrewarmOp::Context, false));
        for op in [KimiPrewarmOp::Ffn, KimiPrewarmOp::Attention] {
            assert!(!kimi_selected(&layout, 16, false, op, false));
        }
    }

    #[test]
    fn ffn_entries_select_only_their_own_operation_class() {
        let layout = KimiPrewarmLayout { core: KimiCoreFormat::K6, ranks: 1, layout: "ffn".into(),
            reserve_base_bytes: vec![1], reserve_extra_bytes: 2 << 20,
            entries: vec![KimiPrewarmEntry { rows: 20, speculative: false, pin: false,
                op: Some(KimiPrewarmOp::Ffn), capture_bytes: vec![] }] };
        for op in [KimiPrewarmOp::Ffn, KimiPrewarmOp::Attention, KimiPrewarmOp::Context] {
            assert_eq!(kimi_selected(&layout, 20, false, op, false), op == KimiPrewarmOp::Ffn);
        }
        assert!(!kimi_selected(&layout, 20, false, KimiPrewarmOp::Ffn, true));
        assert!(!kimi_selected(&layout, 16, false, KimiPrewarmOp::Ffn, false));
    }

    #[test]
    fn bootstrap_is_the_census_share_of_one_capture() {
        let (bytes, executables) = crate::families::kimi_k3::allocation::graph_census(1, 0);
        assert_eq!(kimi_bootstrap(1, 0), bytes.div_ceil(executables));
        assert!(kimi_bootstrap(2, 1) > 0);
    }
}
