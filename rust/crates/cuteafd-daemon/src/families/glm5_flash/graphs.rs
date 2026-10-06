//! Captured decode graphs under an optional byte budget (`--graph-budget-mib`). Each executable's
//! device bytes are measured at capture (the GPU's free memory before the capture and after its
//! instantiation); past the budget the least recently launched executables leave the cache, and
//! the engine destroys them once their stream has drained. Unbounded by default.
use std::collections::{HashMap, HashSet};
use std::hash::Hash;

/// What the cache did, since start-up.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GraphStats {
    /// Executables captured, and those captured again after an eviction.
    pub captures: u64,
    pub recaptures: u64,
    /// Executables evicted, and their measured bytes.
    pub evictions: u64,
    pub evicted_bytes: u64,
}

struct Entry<E> {
    exec: E,
    bytes: u64,
    /// The launch clock at its last launch (or capture).
    launched: u64,
}

pub(crate) struct GraphCache<K, E> {
    entries: HashMap<K, Entry<E>>,
    /// Every key ever captured (a capture of one of them again is a recapture).
    seen: HashSet<K>,
    budget: Option<u64>,
    bytes: u64,
    clock: u64,
    stats: GraphStats,
}

impl<K: Hash + Eq + Clone, E> GraphCache<K, E> {
    /// A cache of at most `budget` measured bytes (None: unbounded).
    pub fn new(budget: Option<u64>) -> Self {
        Self { entries: HashMap::new(), seen: HashSet::new(), budget, bytes: 0, clock: 0, stats: GraphStats::default() }
    }

    pub fn set_budget(&mut self, budget: Option<u64>) {
        self.budget = budget;
    }

    pub fn budget(&self) -> Option<u64> {
        self.budget
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

    /// Adds `key`'s new executable of `bytes` (launched now) and returns the executables that
    /// leave the cache to bring it back under the budget, least recently launched first: never
    /// the new one. The caller destroys them once their stream has drained.
    pub fn insert(&mut self, key: K, exec: E, bytes: u64) -> Vec<E> {
        self.clock += 1;
        self.stats.captures += 1;
        if !self.seen.insert(key.clone()) {
            self.stats.recaptures += 1;
        }
        let entry = Entry { exec, bytes, launched: self.clock };
        let mut out = Vec::new();
        if let Some(old) = self.entries.insert(key.clone(), entry) {
            self.bytes -= old.bytes;
            out.push(old.exec);
        }
        self.bytes += bytes;
        let Some(budget) = self.budget else { return out };
        if self.bytes <= budget {
            return out;
        }
        let mut order: Vec<(u64, K)> = self.entries.iter().filter(|(k, _)| **k != key)
            .map(|(k, entry)| (entry.launched, k.clone())).collect();
        order.sort_unstable_by_key(|(launched, _)| *launched);
        for (_, victim) in order {
            if self.bytes <= budget {
                break;
            }
            if let Some(entry) = self.entries.remove(&victim) {
                self.bytes -= entry.bytes;
                self.stats.evictions += 1;
                self.stats.evicted_bytes += entry.bytes;
                out.push(entry.exec);
            }
        }
        out
    }

    /// Whether `key` was captured before (capturing it again is a recapture).
    pub fn seen(&self, key: &K) -> bool {
        self.seen.contains(key)
    }

    /// Measured bytes of the executables held.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn stats(&self) -> GraphStats {
        self.stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unbounded_caches_keep_every_executable() {
        let mut cache = GraphCache::new(None);
        for key in 0..100u32 {
            assert!(cache.insert(key, key, 1 << 20).is_empty());
        }
        assert_eq!((cache.len(), cache.bytes()), (100, 100 << 20));
        assert_eq!(cache.launch(&7), Some(&7));
        assert_eq!(cache.stats(), GraphStats { captures: 100, ..Default::default() });
    }

    #[test]
    fn the_least_recently_launched_leave_first_and_never_the_new_one() {
        let mut cache = GraphCache::new(Some(3 << 20));
        for key in 0..3u32 {
            assert!(cache.insert(key, key, 1 << 20).is_empty());
        }
        // Launching 0 makes 1 the least recently launched.
        assert_eq!(cache.launch(&0), Some(&0));
        assert_eq!(cache.insert(3, 3, 1 << 20), vec![1]);
        assert_eq!(cache.launch(&1), None);
        assert_eq!((cache.len(), cache.bytes()), (3, 3 << 20));
        // A large executable evicts as many as it needs, oldest first.
        assert_eq!(cache.insert(4, 4, 2 << 20), vec![2, 0]);
        assert_eq!((cache.len(), cache.bytes()), (2, 3 << 20));
        // One larger than the whole budget stays alone.
        let evicted = cache.insert(5, 5, 8 << 20);
        assert_eq!(evicted.len(), 2);
        assert_eq!((cache.len(), cache.bytes()), (1, 8 << 20));
        assert_eq!(cache.stats(), GraphStats { captures: 6, recaptures: 0, evictions: 5, evicted_bytes: 6 << 20 });
    }

    #[test]
    fn a_key_captured_again_after_eviction_is_a_recapture() {
        let mut cache = GraphCache::new(Some(2 << 20));
        cache.insert(1u32, 1, 1 << 20);
        cache.insert(2, 2, 1 << 20);
        assert_eq!(cache.insert(3, 3, 1 << 20), vec![1]);
        assert!(cache.launch(&1).is_none());
        assert!(cache.seen(&1) && !cache.seen(&4));
        assert_eq!(cache.insert(1, 1, 1 << 20), vec![2]);
        assert_eq!(cache.stats().recaptures, 1);
        // Zero-byte executables (below the free-memory granularity) never force evictions.
        assert!(cache.insert(9, 9, 0).is_empty());
        cache.set_budget(None);
        assert!(cache.insert(10, 10, 64 << 20).is_empty());
        assert_eq!(cache.budget(), None);
    }

    #[test]
    fn replacing_a_key_returns_the_old_executable_and_its_bytes() {
        let mut cache = GraphCache::new(None);
        cache.insert(1u32, 10, 4);
        assert_eq!(cache.insert(1, 11, 6), vec![10]);
        assert_eq!((cache.len(), cache.bytes(), cache.launch(&1)), (1, 6, Some(&11)));
    }
}
