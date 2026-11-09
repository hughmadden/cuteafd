//! Trace-owned Step-5 row geometries shared by admission and startup.
use serde::Deserialize;
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum Step5GraphLayout {
    #[serde(rename = "fp8-2rtx-6spark")] Fp8Max,
    #[serde(rename = "nvfp4-1rtx-4spark")] Nvfp4Min,
}
impl Step5GraphLayout {
    pub fn resolve(catalog: &crate::OfficialV41Catalog, ranks: usize, sparks: usize) -> Option<Self> {
        use crate::formats::fp8_experts::ExpertFormat;
        match (catalog.fp8()?.format(), ranks, sparks) {
            (ExpertFormat::Fp8Block128, 2, 6) => Some(Self::Fp8Max),
            (ExpertFormat::Nvfp4, 1, 4) => Some(Self::Nvfp4Min),
            _ => None,
        }
    }
}
#[derive(Debug, Deserialize)]
pub struct Step5GraphSection {
    pub layer: usize,
    pub section: String,
    pub capture_bytes: u64,
    pub replay_hits: u64,
}
#[derive(Debug, Deserialize)]
pub struct Step5GraphRow {
    pub rows: usize,
    pub pin: bool,
    pub replay_hits: u64,
    pub capture_bytes: u64,
    pub sections: Vec<Step5GraphSection>,
}
#[derive(Debug, Deserialize)]
pub struct Step5GraphRank {
    pub rank: usize,
    pub reserve_base_bytes: u64,
    pub reserve_extra_bytes: u64,
    pub entries: Vec<Step5GraphRow>,
}
#[derive(Debug, Deserialize)]
pub struct Step5GraphTrace {
    pub schema_version: u32,
    pub family: String,
    pub layout: Step5GraphLayout,
    pub ranks: Vec<Step5GraphRank>,
}
#[derive(Deserialize)]
struct List { layouts: Vec<Step5GraphTrace> }
pub fn prewarm(layout: Option<Step5GraphLayout>) -> Option<&'static Step5GraphTrace> {
    static LIST: OnceLock<List> = OnceLock::new();
    let layout = layout?;
    LIST.get_or_init(|| {
        let list: List = serde_json::from_str(include_str!("step5_prewarm.json"))
            .expect("Step-5 trace inventory");
        for trace in &list.layouts {
            assert_eq!(trace.schema_version, 1);
            assert_eq!(trace.family, "step5");
            let ranks = if trace.layout == Step5GraphLayout::Fp8Max { 2 } else { 1 };
            assert_eq!(trace.ranks.len(), ranks);
            for (rank, data) in trace.ranks.iter().enumerate() {
                assert_eq!(data.rank, rank);
                assert_eq!(data.reserve_extra_bytes, 2 << 20);
                assert!(!data.entries.is_empty());
                for row in &data.entries {
                    assert!(row.rows > 0 && row.replay_hits > 0);
                    assert_eq!(row.capture_bytes, row.sections.iter().map(|s| s.capture_bytes).sum::<u64>());
                }
                assert_eq!(data.reserve_base_bytes,
                    data.entries.iter().map(|e| e.capture_bytes).sum::<u64>().div_ceil(2 << 20) * (2 << 20));
                assert_eq!(data.entries.iter().map(|e| (e.rows, e.pin)).collect::<Vec<_>>(),
                    trace.ranks[0].entries.iter().map(|e| (e.rows, e.pin)).collect::<Vec<_>>());
            }
        }
        list
    }).layouts.iter().find(|trace| trace.layout == layout)
}
pub fn reserve(layout: Option<Step5GraphLayout>, rank: usize) -> (u64, u64) {
    prewarm(layout).and_then(|trace| trace.ranks.get(rank))
        .map_or((0, 0), |data| (data.reserve_base_bytes, data.reserve_extra_bytes))
}
/// Select complete rows against all devices; never borrow an untraced layout.
pub fn selected_rows(layout: Option<Step5GraphLayout>, max_rows: usize) -> anyhow::Result<Vec<(usize, bool)>> {
    let Some(trace) = prewarm(layout) else { return Ok(Vec::new()) };
    let candidates = trace.ranks[0].entries.iter().enumerate()
        .filter_map(|(i, row)| (row.rows <= max_rows).then_some(i)).collect::<Vec<_>>();
    let bases = trace.ranks.iter().map(|rank| {
        crate::placement::graph_reserve::current(rank.rank, false,
            rank.reserve_base_bytes, rank.reserve_extra_bytes).map(|p| p.base_bytes)
    }).collect::<Result<Vec<_>, _>>()?;
    let pins = candidates.iter().map(|&i| trace.ranks[0].entries[i].pin).collect::<Vec<_>>();
    let selected = crate::placement::graph_reserve::prewarm_selection(&pins, &bases, |selected, bases| {
        trace.ranks.iter().zip(bases).all(|(rank, &base)| {
            let bytes = selected.iter().map(|&i| rank.entries[candidates[i]].capture_bytes).sum::<u64>();
            bytes.div_ceil(2 << 20) * (2 << 20) + rank.reserve_extra_bytes <= base
        })
    });
    Ok(selected.into_iter().map(|i| {
        let row = &trace.ranks[0].entries[candidates[i]];
        (row.rows, row.pin)
    }).collect())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn untraced_layout_has_no_pins_no_charge_and_no_borrowed_rows() {
        assert!(prewarm(None).is_none());
        assert_eq!(reserve(None, 0), (0, 0));
        assert!(selected_rows(None, 4096).unwrap().is_empty());
    }
}
