//! MiMo per-device graph cache inputs: one checked-in prewarm list per resolved
//! card layout (1x and 2x capture different sites), each with its own reserve.
use crate::families::mimo_v2::decode_graph::MIMO_DECODE_ROWS;

/// One prewarm row shape: every layer segment and both tail head variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MimoPrewarmEntry {
    pub rows: usize,
    pub pin: bool,
}

/// A single resolved layout's provisional reserve and startup work.
///
/// `reserve_base_bytes` is the measured whole-capture arena of the prewarm set
/// on that physical rank; `reserve_extra_bytes` is the room one further capture
/// needs (the largest single shape increment). Both are per physical rank, in
/// rank order, rounded up to the 2 MiB driver pool granule.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MimoLayoutPrewarm {
    pub ranks: usize,
    pub reserve_base_bytes: Vec<u64>,
    pub reserve_extra_bytes: Vec<u64>,
    pub entries: Vec<MimoPrewarmEntry>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MimoPrewarmList {
    pub schema_version: u32,
    pub family: String,
    pub layouts: Vec<MimoLayoutPrewarm>,
}

pub fn mimo_prewarm_list() -> MimoPrewarmList {
    let list: MimoPrewarmList = serde_json::from_str(include_str!("mimo_prewarm.json"))
        .expect("checked-in MiMo prewarm list");
    assert_eq!(list.schema_version, 1);
    assert_eq!(list.family, "mimo_v2");
    for layout in &list.layouts {
        assert!((1..=2).contains(&layout.ranks), "MiMo prewarm layout ranks");
        assert_eq!(layout.reserve_base_bytes.len(), layout.ranks);
        assert_eq!(layout.reserve_extra_bytes.len(), layout.ranks);
    }
    list
}

/// The prewarm set for `ranks` physical ranks. An untraced layout charges no
/// graph reserve and warms nothing: a capture it was never measured on would
/// otherwise be priced from another layout's arena.
pub fn mimo_prewarm_layout(ranks: usize) -> MimoLayoutPrewarm {
    let list = mimo_prewarm_list();
    if let Some(layout) = list.layouts.into_iter().find(|layout| layout.ranks == ranks) {
        return layout;
    }
    MimoLayoutPrewarm {
        ranks,
        reserve_base_bytes: vec![0; ranks],
        reserve_extra_bytes: vec![0; ranks],
        entries: Vec::new(),
    }
}

/// The row shapes a layout warms at startup, in ascending order.
pub fn mimo_prewarm_rows(ranks: usize) -> Vec<usize> {
    let mut rows: Vec<_> = mimo_prewarm_layout(ranks).entries.iter()
        .map(|entry| entry.rows).filter(|&rows| (1..=MIMO_DECODE_ROWS).contains(&rows)).collect();
    rows.sort_unstable();
    rows.dedup();
    rows
}

/// The shapes this layout pins: always-hot by the family's trace or by the
/// provisional capture it was checked in from.
pub fn mimo_prewarm_pinned(ranks: usize) -> Vec<usize> {
    mimo_prewarm_layout(ranks).entries.iter().filter(|entry| entry.pin).map(|entry| entry.rows).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_in_layouts_carry_a_reserve_for_every_rank() {
        let list = mimo_prewarm_list();
        assert!(!list.layouts.is_empty());
        for layout in &list.layouts {
            assert_eq!(layout.reserve_base_bytes.len(), layout.ranks);
            assert_eq!(layout.reserve_extra_bytes.len(), layout.ranks);
            assert!(layout.reserve_base_bytes.iter().all(|&bytes| bytes % (2 << 20) == 0));
            assert!(layout.reserve_extra_bytes.iter().all(|&bytes| bytes % (2 << 20) == 0));
        }
        let one = mimo_prewarm_layout(1);
        assert_eq!(one.reserve_base_bytes, vec![628 << 20]);
        assert_eq!(one.reserve_extra_bytes, vec![12 << 20]);
        let two = mimo_prewarm_layout(2);
        assert_eq!(two.reserve_base_bytes, vec![628 << 20, 566 << 20]);
        assert_eq!(two.reserve_extra_bytes, vec![12 << 20, 12 << 20]);
        let _: MimoPrewarmList = serde_json::from_value(serde_json::to_value(&list).unwrap()).unwrap();
    }

    #[test]
    fn every_layout_warms_every_decode_row_and_pins_the_single_row() {
        for ranks in [1, 2] {
            let rows = mimo_prewarm_rows(ranks);
            assert_eq!(rows, (1..=MIMO_DECODE_ROWS).collect::<Vec<_>>());
            assert_eq!(mimo_prewarm_pinned(ranks), vec![1]);
        }
    }
}
