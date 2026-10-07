use super::{MediaKey, MediaError};
use std::{collections::HashMap, sync::Arc};

/// A request/encode pin. Cloning keeps the entry pinned; dropping needs no cache callback.
#[derive(Clone, Debug)]
pub struct EmbeddingLease {
    key: MediaKey,
    rows: Option<Arc<[u8]>>,
    _pin: Arc<()>,
}
impl EmbeddingLease {
    pub fn key(&self) -> MediaKey {
        self.key
    }
    pub fn features(&self) -> Option<&Arc<[u8]>> {
        self.rows.as_ref()
    }
}
struct Entry {
    rows: Option<Arc<[u8]>>,
    pin: Arc<()>,
    bytes: usize,
    last_use: u64,
}

/// Host-RAM LRU. Reservations count against the byte budget before the encoder allocates.
/// Pinned entries (including encodes in flight) never become eviction candidates.
pub struct EmbeddingCache {
    entries: HashMap<MediaKey, Entry>,
    capacity: usize,
    bytes: usize,
    clock: u64,
    hits: u64,
}
impl EmbeddingCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: HashMap::new(),
            capacity,
            bytes: 0,
            clock: 0,
            hits: 0,
        }
    }
    pub fn default_budget(host_ram: usize) -> usize {
        (8usize << 30).min(host_ram / 20)
    }
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    pub fn bytes(&self) -> usize {
        self.bytes
    }
    pub fn hits(&self) -> u64 {
        self.hits
    }
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    pub fn contains(&self, key: impl Into<MediaKey>) -> bool {
        self.entries.get(&key.into()).is_some_and(|e| e.rows.is_some())
    }

    /// Includes references hidden by a prefix hit: callers touch all history images.
    pub fn get(&mut self, key: impl Into<MediaKey>) -> Option<EmbeddingLease> {
        let key = key.into();
        let entry = self.entries.get_mut(&key)?;
        if entry.rows.is_none() {
            return None;
        }
        self.clock += 1;
        entry.last_use = self.clock;
        self.hits += 1;
        Some(EmbeddingLease {
            key,
            rows: entry.rows.clone(),
            _pin: entry.pin.clone(),
        })
    }
    pub fn reserve(&mut self, key: impl Into<MediaKey>, bytes: usize) -> Result<EmbeddingLease, MediaError> {
        let key = key.into();
        if bytes == 0 {
            return Err(MediaError::Features);
        }
        if bytes > self.capacity {
            return Err(MediaError::ImageTooLarge { needed: bytes, capacity: self.capacity });
        }
        if let Some(entry) = self.entries.get_mut(&key) {
            if entry.bytes != bytes {
                return Err(MediaError::Features);
            }
            self.clock += 1;
            entry.last_use = self.clock;
            return Ok(EmbeddingLease {
                key,
                rows: entry.rows.clone(),
                _pin: entry.pin.clone(),
            });
        }
        self.make_room(bytes)?;
        self.clock += 1;
        let pin = Arc::new(());
        self.entries.insert(
            key,
            Entry {
                rows: None,
                pin: pin.clone(),
                bytes,
                last_use: self.clock,
            },
        );
        self.bytes += bytes;
        Ok(EmbeddingLease {
            key,
            rows: None,
            _pin: pin,
        })
    }
    pub fn complete(
        &mut self,
        key: impl Into<MediaKey>,
        rows: Arc<[u8]>,
    ) -> Result<EmbeddingLease, MediaError> {
        let key = key.into();
        let entry = self
            .entries
            .get_mut(&key)
            .ok_or(MediaError::NotReady(key))?;
        if rows.len() != entry.bytes
            || entry
                .rows
                .as_ref()
                .is_some_and(|old| old.as_ref() != rows.as_ref())
        {
            return Err(MediaError::Features);
        }
        self.clock += 1;
        entry.last_use = self.clock;
        let rows = entry.rows.get_or_insert(rows).clone();
        Ok(EmbeddingLease {
            key,
            rows: Some(rows),
            _pin: entry.pin.clone(),
        })
    }
    /// Remove unused reservations after failed/cancelled jobs, without evicting warm embeddings.
    pub fn prune_reservations(&mut self) {
        let unused: Vec<_> = self
            .entries
            .iter()
            .filter(|(_, e)| e.rows.is_none() && Arc::strong_count(&e.pin) == 1)
            .map(|(&key, _)| key)
            .collect();
        for key in unused {
            self.remove(key);
        }
    }
    fn remove(&mut self, key: MediaKey) {
        if let Some(entry) = self.entries.remove(&key) {
            self.bytes -= entry.bytes;
        }
    }
    fn make_room(&mut self, needed: usize) -> Result<(), MediaError> {
        self.prune_reservations();
        let free = self.capacity - self.bytes;
        // Fail atomically when pins make this impossible (do not flush unrelated warm images).
        let evictable: usize = self
            .entries
            .values()
            .filter(|e| Arc::strong_count(&e.pin) == 1)
            .map(|e| e.bytes)
            .sum();
        if needed > free.saturating_add(evictable) {
            return Err(MediaError::CacheFull {
                needed,
                free,
                capacity: self.capacity,
            });
        }
        while needed > self.capacity - self.bytes {
            let victim = self
                .entries
                .iter()
                .filter(|(_, e)| Arc::strong_count(&e.pin) == 1)
                .min_by_key(|(key, e)| (e.last_use, **key))
                .map(|(&key, _)| key)
                .expect("checked evictable budget");
            self.remove(victim);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::ImageKey;
    fn key(i: u8) -> MediaKey {
        ImageKey([i; 32]).into()
    }
    fn put(cache: &mut EmbeddingCache, i: u8) -> EmbeddingLease {
        let pending = cache.reserve(key(i), 4).unwrap();
        let lease = cache.complete(key(i), Arc::from([i; 4])).unwrap();
        drop(pending);
        lease
    }
    #[test]
    fn pins_and_byte_lru_including_prefix_references() {
        let mut cache = EmbeddingCache::new(8);
        let a = put(&mut cache, 1);
        drop(put(&mut cache, 2));
        assert!(cache.reserve(key(3), 8).is_err());
        assert_eq!(cache.bytes(), 8);
        let a2 = a.clone();
        drop(a);
        drop(put(&mut cache, 3));
        assert!(cache.contains(key(1)) && !cache.contains(key(2)));
        drop(a2);
        drop(cache.get(key(1)).unwrap());
        drop(put(&mut cache, 4));
        assert!(cache.contains(key(1)) && !cache.contains(key(3)));
        assert!(cache.bytes() <= cache.capacity());
    }
    #[test]
    fn oversized_image_is_admission_not_pressure_and_preserves_cache() {
        let mut cache = EmbeddingCache::new(4);
        drop(put(&mut cache, 1));
        let error = cache.reserve(key(2), 5).unwrap_err();
        assert_eq!(error, MediaError::ImageTooLarge { needed: 5, capacity: 4 });
        assert_eq!(error.to_string(), "image needs 5 bytes > media cache capacity 4");
        assert!(cache.contains(key(1)));
        assert_eq!(cache.bytes(), 4);
        assert_eq!(cache.len(), 1);
        let pin = cache.get(key(1)).unwrap();
        assert!(matches!(cache.reserve(key(2), 4), Err(MediaError::CacheFull { .. })));
        drop(pin);
    }
    #[test]
    fn reservation_cleanup_and_no_unbudgeted_completion() {
        let mut cache = EmbeddingCache::new(4);
        let pin = cache.reserve(key(1), 4).unwrap();
        assert!(cache.reserve(key(2), 4).is_err());
        assert!(cache.complete(key(1), Arc::from([1; 3])).is_err());
        drop(pin);
        cache.prune_reservations();
        assert_eq!(cache.bytes(), 0);
        assert!(cache.complete(key(1), Arc::from([1; 4])).is_err());
    }
}
