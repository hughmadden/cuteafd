//! V4.1 graph prewarm: one checked-in list drives the admission reserve and
//! the daemon's startup capture. Keys are decode rows; every bank on a device
//! shares them, so a row entry warms each site's geometry at that width.

/// One decode width the startup capture warms, and whether it is pinned.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct V41PrewarmEntry {
    pub rows: usize,
    pub pin: bool,
}

/// A draft key is requests by speculative width, not flattened target rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct V41DraftPrewarmEntry {
    pub requests: usize,
    pub width: usize,
    pub pin: bool,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct V41PrewarmList {
    pub schema_version: u32,
    pub family: String,
    /// Measured arena high-water for the whole live graph set. Bootstrap from
    /// the #1635 census10 lazy growth (1,046,478,848 bytes) until the trace card
    /// measures this family's warmed arena on the qualification config.
    pub reserve_base_bytes: u64,
    /// One largest capture, the peak a capture adds before its victim is freed.
    pub reserve_extra_bytes: u64,
    pub entries: Vec<V41PrewarmEntry>,
    pub draft_entries: Vec<V41DraftPrewarmEntry>,
    /// Qualified exact full-chunk and replay widths; never approximate tails.
    #[serde(default)]
    pub prefill_entries: Vec<V41PrewarmEntry>,
}

pub fn v41_prewarm_list() -> &'static V41PrewarmList {
    static LIST: std::sync::OnceLock<V41PrewarmList> = std::sync::OnceLock::new();
    LIST.get_or_init(|| {
        let list: V41PrewarmList = serde_json::from_str(include_str!("v41_prewarm.json"))
            .expect("checked-in V4.1 prewarm list");
        assert_eq!(list.schema_version, 1);
        assert_eq!(list.family, "deepseek_v41");
        list
    })
}

/// Decode preferences are directly reachable under the live decode-row cap.
pub fn v41_prewarm_rows(decode_rows: usize) -> Vec<V41PrewarmEntry> {
    v41_prewarm_list().entries.iter().copied().filter(|entry| entry.rows <= decode_rows).collect()
}

/// The chunk is a launch input. Substitute only the qualified full-chunk slot;
/// keep the existing replay shape and do not warm any other large geometry.
pub fn v41_prefill_rows(capacity: usize, chunk: Option<usize>) -> Vec<usize> {
    let replay = cuteafd_core::coordinator_programs::V41_DECODER_REPLAY_ROWS as usize;
    let mut rows: Vec<_> = v41_prewarm_list().prefill_entries.iter().filter_map(|entry| {
        let rows = if entry.rows == replay { Some(replay) } else { chunk }?;
        (rows <= capacity).then_some(rows)
    }).collect();
    rows.sort_unstable();
    rows.dedup();
    rows
}

/// Startup preferences remain a subset of the live draft domain; misses outside
/// this list still use ordinary bounded admission at runtime.
pub fn v41_prewarm_drafts(requests: usize, maximum_width: usize) -> Vec<V41DraftPrewarmEntry> {
    v41_prewarm_list().draft_entries.iter().copied()
        .filter(|entry| entry.requests <= requests && entry.width <= maximum_width).collect()
}

pub fn v41_pinned_draft(requests: usize, width: usize) -> bool {
    v41_prewarm_list().draft_entries.iter()
        .any(|entry| entry.requests == requests && entry.width == width && entry.pin)
}

/// The decode rows that must survive eviction: the C1 width the sweep shows is
/// always hot. The 128-row decoder replay and the prefill chunk are pinned by
/// their capture sites, which know the live chunk width.
pub fn v41_pinned_rows(decode_rows: usize) -> Vec<usize> {
    v41_prewarm_rows(decode_rows).into_iter().filter(|entry| entry.pin).map(|entry| entry.rows).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn checked_in_list_roundtrips_and_matches_the_startup_set() {
        let list = v41_prewarm_list();
        assert!(!list.entries.is_empty());
        let mut rows = HashSet::new();
        for entry in &list.entries {
            assert!(rows.insert(entry.rows), "duplicate prewarm width");
            assert!(entry.rows > 0 && entry.rows <= cuteafd_core::V41_VERIFY_ROWS);
        }
        assert_eq!(list.reserve_base_bytes % (2 << 20), 0);
        assert_eq!(list.reserve_extra_bytes % (2 << 20), 0);
        assert!(list.reserve_base_bytes > 0);
        assert_eq!(v41_prewarm_rows(cuteafd_core::V41_VERIFY_ROWS), list.entries);
        for cap in [1, 8, cuteafd_core::V41_VERIFY_ROWS] {
            let expected: Vec<_> = list.entries.iter().copied().filter(|e| e.rows <= cap).collect();
            assert_eq!(v41_prewarm_rows(cap), expected);
            assert_eq!(v41_pinned_rows(cap), expected.iter().filter(|e| e.pin).map(|e| e.rows).collect::<Vec<_>>());
        }
        let expected: Vec<_> = list.draft_entries.iter().copied()
            .filter(|e| e.requests <= 4 && e.width <= 5).collect();
        assert_eq!(v41_prewarm_drafts(4, 5), expected);
        let replay = cuteafd_core::coordinator_programs::V41_DECODER_REPLAY_ROWS as usize;
        let mut prefill = HashSet::new();
        for entry in &list.prefill_entries {
            assert!(entry.pin && (1..=4096).contains(&entry.rows));
            assert!(prefill.insert(entry.rows), "duplicate prefill geometry");
        }
        assert!(prefill.contains(&replay));
        assert_eq!(prefill.len(), 2, "one qualified full chunk plus replay");
        for chunk in [128, 2048, 2752] {
            let mut expected = vec![replay, chunk];
            expected.sort_unstable(); expected.dedup();
            assert_eq!(v41_prefill_rows(4096, Some(chunk)), expected);
        }
        assert_eq!(v41_prefill_rows(128, Some(2752)), vec![replay]);
        assert!(v41_prewarm_drafts(0, 7).is_empty());
        let mut keys = HashSet::new();
        for entry in &list.draft_entries {
            assert!(keys.insert((entry.requests, entry.width)), "duplicate draft geometry");
            assert!((1..=cuteafd_core::V41_MAX_REQUESTS).contains(&entry.requests));
            assert!(matches!(entry.width, 5 | 7));
        }
        let _: V41PrewarmList = serde_json::from_value(serde_json::to_value(&list).unwrap()).unwrap();
        let mut without_prefill = serde_json::to_value(&list).unwrap();
        without_prefill.as_object_mut().unwrap().remove("prefill_entries");
        let optional: V41PrewarmList = serde_json::from_value(without_prefill).unwrap();
        assert!(optional.prefill_entries.is_empty());
    }

}
