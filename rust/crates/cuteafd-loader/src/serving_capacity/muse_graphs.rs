//! Muse decode prewarm: exact live rows, with a fixed admitted page-table pitch.
//! An untraced layout has no startup graphs or pins; explicit reserve permits
//! bounded lazy capture while the layout's own trace is being collected.
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Entry {
    pub rows: usize,
    pub pin: bool,
    /// Physical capture shares, including capture-only siblings, by rank.
    pub capture_bytes: Vec<u64>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Layout {
    pub ranks: usize,
    pub sparks: bool,
    pub provenance: String,
    pub reserve_base_bytes: Vec<u64>,
    pub reserve_extra_bytes: Vec<u64>,
    pub entries: Vec<Entry>,
}

#[derive(Debug, Deserialize)]
struct List {
    schema_version: u32,
    family: String,
    layouts: Vec<Layout>,
}

pub fn layout(ranks: usize, sparks: bool) -> Option<&'static Layout> {
    static LIST: OnceLock<List> = OnceLock::new();
    let list = LIST.get_or_init(|| {
        let list: List = serde_json::from_str(include_str!("muse_prewarm.json")).expect("Muse prewarm JSON");
        assert_eq!(list.schema_version, 1);
        assert_eq!(list.family, "muse_spark");
        for layout in &list.layouts {
            assert!(matches!(layout.ranks, 1 | 2));
            assert_eq!(layout.reserve_base_bytes.len(), layout.ranks);
            assert_eq!(layout.reserve_extra_bytes, vec![2 << 20; layout.ranks]);
            assert!(layout.entries.iter().all(|entry| (1..=64).contains(&entry.rows)
                && entry.capture_bytes.len() == layout.ranks));
        }
        list
    });
    list.layouts.iter().find(|layout| layout.ranks == ranks && layout.sparks == sparks)
}

pub fn policy(ranks: usize, sparks: bool, rank: usize) -> Result<crate::placement::graph_reserve::Resolved,
    crate::placement::graph_reserve::Error> {
    let (base, tail) = layout(ranks, sparks).map_or((0, 0), |layout|
        (layout.reserve_base_bytes[rank], layout.reserve_extra_bytes[rank]));
    crate::placement::graph_reserve::current(rank, false, base, tail)
}

pub fn selected<'a>(layout: &'a Layout, bases: &[u64]) -> Vec<&'a Entry> {
    let pins = layout.entries.iter().map(|entry| entry.pin).collect::<Vec<_>>();
    crate::placement::graph_reserve::prewarm_selection(&pins, bases, |indices, bases| {
        bases.iter().enumerate().all(|(rank, &base)| {
            let sum = indices.iter().try_fold(0u64, |sum, &i|
                sum.checked_add(layout.entries[i].capture_bytes[rank]));
            sum.is_some_and(|bytes| bytes.next_multiple_of(2 << 20) <= base)
        })
    }).into_iter().map(|i| &layout.entries[i]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn untraced_layout_borrows_no_inventory_and_pins_nothing() {
        // This family had no startup graphs before this implementation.
        assert!(layout(3, true).is_none());
        for ranks in [1, 2] {
            for sparks in [false, true] {
                if let Some(layout) = layout(ranks, sparks) {
                    assert_eq!(layout.reserve_base_bytes.len(), ranks);
                } else {
                    assert!(layout(ranks, sparks).is_none());
                }
            }
        }
    }

    #[test]
    fn smaller_base_fits_pins_as_a_set_then_a_heat_ordered_prefix() {
        let list = Layout { ranks: 1, sparks: true, provenance: "fixture".into(),
            reserve_base_bytes: vec![8 << 20], reserve_extra_bytes: vec![2 << 20],
            entries: vec![Entry { rows: 1, pin: true, capture_bytes: vec![2 << 20] },
                Entry { rows: 4, pin: false, capture_bytes: vec![4 << 20] },
                Entry { rows: 8, pin: false, capture_bytes: vec![1 << 20] }] };
        assert!(selected(&list, &[0]).is_empty());
        assert_eq!(selected(&list, &[4 << 20]).iter().map(|e| e.rows).collect::<Vec<_>>(), [1]);
        assert_eq!(selected(&list, &[6 << 20]).iter().map(|e| e.rows).collect::<Vec<_>>(), [1, 4]);
    }
}
