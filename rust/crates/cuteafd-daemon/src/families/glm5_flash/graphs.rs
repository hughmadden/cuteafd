//! Captured decode graphs under an optional byte budget (`--graph-budget-mib`): past it the least
//! recently launched executables leave the cache, and the engine destroys them once their stream
//! has drained. Unbounded by default.
//!
//! **Accounting.** No API reports one executable's device bytes, and the GPU's free memory moves
//! in whole allocator chunks. While no executable has been destroyed, every chunk the device
//! takes is a new executable's, so the bytes free memory lost over the captures, divided by
//! their count, is the mean size of one executable; the cache charges every executable that mean
//! and freezes it at the first eviction. After that a capture's own delta no longer measures it:
//! a new executable fills the room an evicted one left, and its free-memory delta reads 0. A
//! delta-per-executable budget therefore stops counting after the first eviction while the
//! executables keep accumulating (measured: 512 MiB "held" while 3,670 executables became 7,018,
//! about 1 GB on the device).
use std::collections::{HashMap, HashSet};
use std::hash::Hash;

/// What the cache did, since start-up.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GraphStats {
    /// Executables captured, and those captured again after an eviction.
    pub captures: u64,
    pub recaptures: u64,
    /// Executables evicted.
    pub evictions: u64,
}

struct Entry<E> {
    exec: E,
    /// The launch clock at its last launch (or capture).
    launched: u64,
}

pub(crate) struct GraphCache<K, E> {
    entries: HashMap<K, Entry<E>>,
    /// Every key ever captured (a capture of one of them again is a recapture).
    seen: HashSet<K>,
    budget: Option<u64>,
    clock: u64,
    stats: GraphStats,
    /// Free-memory bytes the captures took and their count, while nothing was destroyed.
    measured_bytes: u64,
    measured: u64,
    /// The size of one executable, fixed at the first eviction.
    frozen: Option<u64>,
}

impl<K: Hash + Eq + Clone, E> GraphCache<K, E> {
    /// A cache of at most `budget` bytes of executables (None: unbounded).
    pub fn new(budget: Option<u64>) -> Self {
        Self { entries: HashMap::new(), seen: HashSet::new(), budget, clock: 0, stats: GraphStats::default(),
            measured_bytes: 0, measured: 0, frozen: None }
    }

    pub fn set_budget(&mut self, budget: Option<u64>) {
        self.budget = budget;
    }

    pub fn budget(&self) -> Option<u64> {
        self.budget
    }

    /// Whether the next capture's free-memory delta still measures its executable (nothing has
    /// been evicted, so nothing has been freed).
    pub fn calibrating(&self) -> bool {
        self.frozen.is_none()
    }

    /// The bytes charged for one executable: the calibrated mean (rounded up).
    pub fn each(&self) -> u64 {
        self.frozen.unwrap_or_else(|| self.measured_bytes.div_ceil(self.measured.max(1)))
    }

    /// The executable of `key`, marked launched now.
    pub fn launch(&mut self, key: &K) -> Option<&E> {
        self.clock += 1;
        let clock = self.clock;
        self.entries.get_mut(key).map(|entry| {
            entry.launched = clock;
            &entry.exec
        })
    }

    /// Whether `key` was captured before (capturing it again is a recapture).
    pub fn seen(&self, key: &K) -> bool {
        self.seen.contains(key)
    }

    /// Adds `key`'s new executable (launched now), with the free-memory bytes its capture took
    /// while the cache is still calibrating, and returns the executables that leave the cache to
    /// bring it back under the budget, least recently launched first: never the new one. The
    /// first eviction freezes the per-executable size. The caller destroys them once their
    /// stream has drained.
    pub fn insert(&mut self, key: K, exec: E, measured: Option<u64>) -> Vec<E> {
        self.clock += 1;
        self.stats.captures += 1;
        if !self.seen.insert(key.clone()) {
            self.stats.recaptures += 1;
        }
        if let (None, Some(bytes)) = (self.frozen, measured) {
            self.measured_bytes += bytes;
            self.measured += 1;
        }
        let mut out = Vec::new();
        if let Some(old) = self.entries.insert(key.clone(), Entry { exec, launched: self.clock }) {
            out.push(old.exec);
        }
        let Some(budget) = self.budget else { return out };
        if self.bytes() <= budget {
            return out;
        }
        self.frozen.get_or_insert(self.measured_bytes.div_ceil(self.measured.max(1)));
        let mut order: Vec<(u64, K)> = self.entries.iter().filter(|(k, _)| **k != key)
            .map(|(k, entry)| (entry.launched, k.clone())).collect();
        order.sort_unstable_by_key(|(launched, _)| *launched);
        for (_, victim) in order {
            if self.bytes() <= budget {
                break;
            }
            if let Some(entry) = self.entries.remove(&victim) {
                self.stats.evictions += 1;
                out.push(entry.exec);
            }
        }
        out
    }

    /// Bytes of the executables held: their count times the size of one.
    pub fn bytes(&self) -> u64 {
        self.entries.len() as u64 * self.each()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether an executable of `key` is held.
    pub fn contains(&self, key: &K) -> bool {
        self.entries.contains_key(key)
    }

    /// Executables held whose key `matches` (a step shape is its segment 0's key).
    pub fn count(&self, matches: impl Fn(&K) -> bool) -> usize {
        self.entries.keys().filter(|key| matches(key)).count()
    }

    pub fn stats(&self) -> GraphStats {
        self.stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1 << 20;

    #[test]
    fn unbounded_caches_keep_every_executable() {
        let mut cache = GraphCache::new(None);
        for key in 0..100u32 {
            assert!(cache.insert(key, key, Some(if key % 14 == 0 { 2 * MIB } else { 0 })).is_empty());
        }
        assert_eq!(cache.len(), 100);
        assert_eq!(cache.each(), (8 * 2 * MIB).div_ceil(100));
        assert_eq!(cache.bytes(), 100 * cache.each());
        assert_eq!(cache.launch(&7), Some(&7));
        assert_eq!(cache.stats(), GraphStats { captures: 100, ..Default::default() });
        assert!(cache.calibrating());
    }

    #[test]
    fn executables_are_charged_the_mean_of_chunked_free_memory_deltas() {
        // Free memory moves in 2 MiB chunks: of 14 executables of ~146 KiB, one capture takes a
        // chunk and 13 take none. Charging each its own delta would let the 13 stay free of charge.
        let mut cache = GraphCache::new(None);
        for key in 0..1400u32 {
            cache.insert(key, key, Some(if key % 14 == 13 { 2 * MIB } else { 0 }));
        }
        assert_eq!(cache.each(), (2 * MIB * 100).div_ceil(1400));
        assert_eq!(cache.bytes(), 1400 * cache.each());
        assert!(cache.bytes() >= 200 * MIB && cache.bytes() < 200 * MIB + 1400);
    }

    #[test]
    fn evictions_bound_the_executables_even_when_new_captures_reuse_freed_memory() {
        // The measured failure: once evictions free room inside the allocator's chunks, new
        // captures fill it and their free-memory deltas read 0. The cache must keep counting them.
        let (budget, each) = (64 * MIB, 147_000u64);
        let mut cache = GraphCache::new(Some(budget));
        let mut key = 0u32;
        let mut evicted = 0usize;
        while cache.calibrating() {
            evicted += cache.insert(key, key, Some(each)).len();
            key += 1;
        }
        assert_eq!(cache.each(), each);
        let held = cache.len();
        assert_eq!(held as u64, budget / each);
        for _ in 0..20_000 {
            evicted += cache.insert(key, key, Some(0)).len();
            key += 1;
            assert!(cache.bytes() <= budget && cache.len() == held, "held {} executables", cache.len());
        }
        assert_eq!(evicted as u64, cache.stats().evictions);
        assert_eq!(cache.stats().captures, u64::from(key));
        assert_eq!(cache.each(), each, "the size stays frozen at the first eviction");
    }

    #[test]
    fn the_least_recently_launched_leave_first_and_never_the_new_one() {
        let mut cache = GraphCache::new(Some(3 * MIB));
        for key in 0..3u32 {
            assert!(cache.insert(key, key, Some(MIB)).is_empty());
        }
        // Launching 0 makes 1 the least recently launched.
        assert_eq!(cache.launch(&0), Some(&0));
        assert_eq!(cache.insert(3, 3, Some(MIB)), vec![1]);
        assert_eq!(cache.launch(&1), None);
        assert_eq!((cache.len(), cache.bytes(), cache.each()), (3, 3 * MIB, MIB));
        // Frozen: a capture that measures more is still charged one executable.
        assert_eq!(cache.insert(4, 4, Some(8 * MIB)), vec![2]);
        assert_eq!((cache.len(), cache.bytes()), (3, 3 * MIB));
        assert_eq!(cache.stats(), GraphStats { captures: 5, recaptures: 0, evictions: 2 });
    }

    #[test]
    fn a_key_captured_again_after_eviction_is_a_recapture() {
        let mut cache = GraphCache::new(Some(2 * MIB));
        cache.insert(1u32, 1, Some(MIB));
        cache.insert(2, 2, Some(MIB));
        assert_eq!(cache.insert(3, 3, Some(MIB)), vec![1]);
        assert!(cache.launch(&1).is_none());
        assert!(cache.seen(&1) && !cache.seen(&4));
        assert_eq!(cache.insert(1, 1, None), vec![2]);
        assert_eq!(cache.stats().recaptures, 1);
        assert_eq!(cache.count(|key| *key % 2 == 1), 2);
        cache.set_budget(None);
        assert!(cache.insert(10, 10, None).is_empty());
        assert_eq!(cache.budget(), None);
    }

    #[test]
    fn replacing_a_key_returns_the_old_executable() {
        let mut cache = GraphCache::new(None);
        cache.insert(1u32, 10, Some(4));
        assert_eq!(cache.insert(1, 11, Some(6)), vec![10]);
        assert_eq!((cache.len(), cache.launch(&1)), (1, Some(&11)));
    }
}
