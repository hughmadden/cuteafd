//! The generic prefix cache (PLAN.md "Prefix cache for every family"): one per generic engine,
//! owning the device page pool, the mark arena, the retained snapshots and the optional host
//! tier. Every call runs on the scheduler thread.
//!
//! Life of a request: [`PrefixCache::admit`] looks the prompt up (device first, then the host
//! tier, whose hit is promoted to the device), forks the snapshot's pages into the new placement
//! (full pages shared, the partial tail copied), restores its mark and returns where prefill
//! resumes, with the first token's logits when the whole prompt was retained. At prompt end the
//! scheduler captures a `Prompt` snapshot (unless the prompt was a total hit); at a cacheable
//! completion (EOS or max_tokens, client still there) a `Turn` snapshot; a prefill cancelled at
//! a chunk boundary is parked as a `Prompt` snapshot; a decode cancelled is not retained. Then
//! [`PrefixCache::release`] drops the placement's references.
//!
//! Selection is the most computation saved under the family's [`ReuseRule`](cuteafd_core::prefix::ReuseRule)
//! (`cuteafd_core::prefix::Retention`, prompts and turns in bounded banks). Eviction is least
//! recently used across both banks, prompt before turn at equal use ([`victim`]); `make_room`
//! runs before every admission and capture that needs pages. A restore that cannot complete is a
//! cache miss, never a request error.
use super::chain::{content_id, page_chain};
use super::entry::{victim, After, Entry, EntryId};
use super::family::{BoxError, FamilyLayout, PrefixFamily};
use super::marks::{MarkArena, MarkSlot};
use super::pages::{PoolExhausted, RefPagePool};
use cuteafd_core::prefix::{Retention, SnapshotKind};
use cuteafd_hostcache::cache::{
    DevicePage, DeviceSnapshot, EvictDecision, HostCache, RestoreOutcome, RestoreTarget, StoreOutcome,
};
use cuteafd_hostcache::copy::{CopyEngine, Stream};
use cuteafd_hostcache::snapshot::{DevicePageId, EvictionOrder, SnapshotMeta};
use serde::Serialize;
use std::collections::BTreeMap;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum PrefixError {
    #[error(transparent)]
    Pages(#[from] PoolExhausted),
    #[error("family {what} failed: {source}")]
    Family {
        what: &'static str,
        #[source]
        source: BoxError,
    },
    #[error("snapshot of {tokens} tokens, but the placement committed {committed} (reach {reach})")]
    Frontier { tokens: usize, committed: usize, reach: usize },
    #[error("host tier: {0}")]
    Host(String),
}

/// Knobs of the device tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct PrefixConfig {
    /// Entries per bank (prompts, turns); 0 disables retention (the pool still allocates).
    pub entries: usize,
    /// Device mark slots ([`MarkArena::slots_for`]).
    pub mark_slots: usize,
    /// Keep the last logit row with each snapshot, so sampled exact-length hits need no forward.
    pub keep_logits: bool,
    /// Shortest snapshot worth retaining.
    pub min_tokens: usize,
}

/// A reusable snapshot for a prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Hit {
    pub id: EntryId,
    pub kind: SnapshotKind,
    /// Rows restored; prefill resumes here.
    pub resume: usize,
    pub frontier: usize,
}

/// Where a restored request came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Source {
    pub kind: SnapshotKind,
    pub frontier: usize,
    /// Promoted from the host tier by this admission.
    pub host: bool,
    /// A partial match: the positional state restarts empty at `resume`, which lies the rule's
    /// replay window before the aligned common prefix (approximate; exact frontiers are exact).
    pub partial: bool,
}

/// An admitted request: its placement, the rows already in it, and the first token's source
/// when the whole prompt was retained.
pub struct Admitted<P> {
    pub placement: P,
    pub resume: usize,
    pub after: Option<After>,
    pub source: Option<Source>,
}

/// What travels with a host snapshot besides its device bytes.
pub struct HostPayload {
    pub after: After,
}

/// Counters and gauges for `/v1/stats`.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct PrefixStats {
    pub lookups: u64,
    pub hits: u64,
    /// Hits that restored the whole prompt (first token from the snapshot's logits).
    pub exact_hits: u64,
    /// Hits on a partial match (replay window, approximate positional state).
    pub partial_hits: u64,
    pub hit_tokens: u64,
    pub promotions: u64,
    pub restore_failures: u64,
    /// Hits dropped because the pool could not hold the fork.
    pub fork_no_room: u64,
    pub captures_prompt: u64,
    pub captures_turn: u64,
    /// Prompt snapshots parked by a prefill cancelled at a chunk boundary.
    pub parked: u64,
    pub capture_skips: u64,
    pub evictions: u64,
    pub cow_copies: u64,
    pub host_store_skips: u64,
    pub host_evict_waits: u64,
    pub host_evict_uncached: u64,
    pub entries_prompt: usize,
    pub entries_turn: usize,
    pub pages: usize,
    pub pages_free: usize,
    pub pages_shared: usize,
    /// Distinct pages held by retained snapshots; when no request runs, every used page is one.
    pub pages_retained: usize,
    pub mark_slots: usize,
    pub marks_in_use: usize,
    pub host: Option<cuteafd_hostcache::metrics::Snapshot>,
}

pub struct PrefixCache<E: CopyEngine> {
    layout: FamilyLayout,
    config: PrefixConfig,
    retained: Retention<EntryId>,
    entries: BTreeMap<EntryId, Entry>,
    pool: RefPagePool,
    arena: MarkArena,
    host: Option<HostCache<E, HostPayload>>,
    clock: u64,
    next_id: EntryId,
    /// Family copies were enqueued since the family last drained.
    dirty: bool,
    stats: PrefixStats,
}

impl<E: CopyEngine> PrefixCache<E> {
    /// `host`: the pinned host tier's knobs and copy engine (`None`, or a zero `bytes`, keeps it off).
    pub fn new(layout: FamilyLayout, config: PrefixConfig, host: Option<(cuteafd_hostcache::config::Config, E)>)
        -> Result<Self, PrefixError> {
        let host = match host {
            Some((host_config, engine)) if config.entries > 0 && host_config.enabled() => Some(
                HostCache::with_rule(host_config, layout.host_layout(), engine, layout.rule, EvictionOrder::LeastRecent)
                    .map_err(|e| PrefixError::Host(format!("{e:#}")))?,
            ),
            _ => None,
        };
        let slots = if config.entries > 0 && layout.mark_bytes > 0 { config.mark_slots } else { 0 };
        Ok(Self {
            retained: Retention::with_rule(config.entries, layout.rule),
            entries: BTreeMap::new(),
            pool: RefPagePool::new(layout.pages, layout.page_rows),
            arena: MarkArena::new(slots, layout.mark_bytes),
            host,
            clock: 0,
            next_id: 0,
            dirty: false,
            stats: PrefixStats::default(),
            layout,
            config,
        })
    }

    pub fn enabled(&self) -> bool {
        self.config.entries > 0
    }
    pub fn layout(&self) -> FamilyLayout {
        self.layout
    }
    pub fn pool(&self) -> &RefPagePool {
        &self.pool
    }
    pub fn arena(&self) -> &MarkArena {
        &self.arena
    }
    pub fn host_engine_mut(&mut self) -> Option<&mut E> {
        self.host.as_mut().map(HostCache::engine_mut)
    }

    /// Admit a request of `tokens` that may grow to `capacity` tokens. `build` turns a page list
    /// into the family's placement (called once per attempt; a failed restore retries cold).
    pub fn admit<F: PrefixFamily<Placement = P>, P>(&mut self, family: &F, tokens: &[u32], capacity: usize,
        sampled: bool, mut build: impl FnMut(Vec<u32>) -> P) -> Result<Admitted<P>, PrefixError> {
        let total = self.pool.pages_for(capacity.max(tokens.len()));
        if self.enabled() && !tokens.is_empty() {
            self.stats.lookups += 1;
            let mut promoted = false;
            let mut hit = self.lookup(tokens, sampled);
            if hit.is_none() {
                hit = self.promote(family, tokens, sampled)?;
                promoted = hit.is_some();
            }
            if let Some(hit) = hit {
                match self.restore_hit(family, &hit, tokens, total, promoted, &mut build) {
                    Ok(Some(admitted)) => return Ok(admitted),
                    Ok(None) => self.stats.fork_no_room += 1,
                    Err(PrefixError::Family { what, source }) => {
                        tracing::warn!(target: "cuteafd::prefix", what, error = %source, "prefix restore abandoned; prefilling");
                        self.stats.restore_failures += 1;
                    }
                    Err(error) => return Err(error),
                }
            }
        }
        self.make_room(family, total, None)?;
        let pages = self.pool.alloc(total)?;
        Ok(Admitted { placement: build(pages), resume: 0, after: None, source: None })
    }

    /// Device lookup under the family's rule. An exact-length match whose snapshot cannot
    /// produce the first token gives way to the longest shorter one.
    fn lookup(&mut self, tokens: &[u32], sampled: bool) -> Option<Hit> {
        let mut query = tokens;
        loop {
            let (common, frontier, &id) = self.retained.lookup_reusable(query)?;
            let resume = self.layout.rule.skipped(common, frontier);
            let entry = self.entries.get_mut(&id)?;
            if resume == tokens.len() && !entry.after.serves(sampled) {
                if query.len() < tokens.len() || tokens.len() < 2 {
                    return None;
                }
                query = &tokens[..tokens.len() - 1];
                continue;
            }
            self.clock += 1;
            entry.last_use = self.clock;
            return Some(Hit { id, kind: entry.kind, resume, frontier });
        }
    }

    fn restore_hit<F: PrefixFamily<Placement = P>, P>(&mut self, family: &F, hit: &Hit, tokens: &[u32], total: usize,
        promoted: bool, build: &mut impl FnMut(Vec<u32>) -> P) -> Result<Option<Admitted<P>>, PrefixError> {
        let need = self.pool.fork_cost(hit.resume, total);
        if !self.make_room(family, need, Some(hit.id))? {
            return Ok(None);
        }
        let entry = self.entries.get(&hit.id).expect("the hit is kept while making room");
        let fork = self.pool.fork(&entry.pages, hit.resume, total)?;
        let partial = hit.resume != entry.len();
        let mark = if partial { None } else { entry.mark };
        let after = (hit.resume == tokens.len()).then(|| entry.after.clone());
        let pages = fork.pages.clone();
        let mut placement = build(fork.pages);
        self.dirty = true;
        let restored = match fork.copy {
            Some(copy) => {
                self.stats.cow_copies += 1;
                family.copy_rows(copy)
            }
            None => Ok(()),
        }
        .and_then(|()| family.restore(mark, &mut placement, hit.resume));
        if let Err(source) = restored {
            self.release(family, &pages)?;
            return Err(PrefixError::Family { what: "restore", source });
        }
        self.stats.hits += 1;
        self.stats.hit_tokens += hit.resume as u64;
        self.stats.exact_hits += u64::from(after.is_some());
        self.stats.partial_hits += u64::from(partial);
        Ok(Some(Admitted {
            placement,
            resume: hit.resume,
            after,
            source: Some(Source { kind: hit.kind, frontier: hit.frontier, host: promoted, partial }),
        }))
    }

    /// Retain `placement`'s first `tokens.len()` rows as a `kind` snapshot; `after` is what
    /// follows them. The snapshot point may lie up to `family.capture_reach()` rows before the
    /// placement's commit point (an intermediate point). Ok(false) when it was not retained
    /// (disabled, too short, no room).
    pub fn capture<F: PrefixFamily>(&mut self, family: &F, kind: SnapshotKind, tokens: &[u32],
        placement: &F::Placement, mut after: After) -> Result<bool, PrefixError> {
        if !self.enabled() || tokens.len() < self.config.min_tokens.max(1) {
            return Ok(false);
        }
        let committed = family.commit_point(placement);
        let reach = family.capture_reach();
        if tokens.len() > committed || committed - tokens.len() > reach {
            return Err(PrefixError::Frontier { tokens: tokens.len(), committed, reach });
        }
        // Replace the same snapshot, or keep the bank within its bound, before taking storage.
        if let Some(old) = self.retained.bank_mut(kind).remove_exact(tokens) {
            self.evict(family, old)?;
        }
        while self.retained.bank(kind).entries() >= self.config.entries {
            let oldest = victim(self.entries.iter().filter(|(_, e)| e.kind == kind).map(|(&id, e)| (id, e.kind, e.last_use)));
            match oldest {
                Some(id) => self.evict(family, id)?,
                None => break,
            }
        }
        let slot = if self.layout.mark_bytes > 0 {
            loop {
                match self.arena.take() {
                    Ok(slot) => break Some(slot),
                    Err(_) => match self.victim(None) {
                        Some(id) => self.evict(family, id)?,
                        None => {
                            self.stats.capture_skips += 1;
                            return Ok(false);
                        }
                    },
                }
            }
        } else {
            None
        };
        let len = tokens.len();
        let total = self.pool.pages_for(len);
        if !self.make_room(family, self.pool.fork_cost(len, total), None)? {
            self.give_back(family, slot)?;
            self.stats.capture_skips += 1;
            return Ok(false);
        }
        let fork = self.pool.fork(family.pages(placement), len, total)?;
        self.dirty = true;
        let captured = match fork.copy {
            Some(copy) => family.copy_rows(copy),
            None => Ok(()),
        }
        .and_then(|()| slot.map_or(Ok(()), |slot| family.capture(slot, placement, len)));
        if let Err(source) = captured {
            self.release(family, &fork.pages)?;
            self.give_back(family, slot)?;
            return Err(PrefixError::Family { what: "capture", source });
        }
        if !self.config.keep_logits {
            after.logits = None;
        }
        self.clock += 1;
        let id = self.next_id;
        self.next_id += 1;
        self.entries.insert(id, Entry { tokens: tokens.to_vec(), kind, pages: fork.pages, mark: slot, after,
            last_use: self.clock, ticket: None });
        if let Some(evicted) = self.retained.bank_mut(kind).insert(tokens, id) {
            self.evict(family, evicted)?;
        }
        match kind {
            SnapshotKind::Prompt => self.stats.captures_prompt += 1,
            SnapshotKind::Turn => self.stats.captures_turn += 1,
        }
        self.host_store(family, id)?;
        Ok(true)
    }

    /// A prefill cancelled at a chunk boundary: keep what it computed as a prompt snapshot
    /// (the client usually retries the same prompt).
    pub fn park<F: PrefixFamily>(&mut self, family: &F, tokens: &[u32], placement: &F::Placement)
        -> Result<bool, PrefixError> {
        let parked = self.capture(family, SnapshotKind::Prompt, tokens, placement, After::default())?;
        self.stats.parked += u64::from(parked);
        Ok(parked)
    }

    /// Drop a placement's page references (after the family's queued work drained).
    pub fn release<F: PrefixFamily>(&mut self, family: &F, pages: &[u32]) -> Result<(), PrefixError> {
        self.drain(family)?;
        let freed = self.pool.release(pages);
        if let Some(host) = &mut self.host {
            for page in freed {
                host.device_page_freed(DevicePageId { compressor: 0, page: page.index, generation: page.generation });
            }
        }
        Ok(())
    }

    /// Evict least recently used snapshots (never `keep`) until `pages` pages are free; false when
    /// nothing is left to evict and the pool is still short.
    pub fn make_room<F: PrefixFamily>(&mut self, family: &F, pages: usize, keep: Option<EntryId>)
        -> Result<bool, PrefixError> {
        while self.pool.free() < pages {
            match self.victim(keep) {
                Some(id) => self.evict(family, id)?,
                None => return Ok(false),
            }
        }
        Ok(true)
    }

    /// Poll the host tier's store copies; once per scheduler step.
    pub fn tick(&mut self) {
        if let Some(host) = &mut self.host {
            host.tick();
        }
    }

    pub fn stats(&self) -> PrefixStats {
        let mut stats = self.stats.clone();
        stats.entries_prompt = self.retained.bank(SnapshotKind::Prompt).entries();
        stats.entries_turn = self.retained.bank(SnapshotKind::Turn).entries();
        stats.pages = self.pool.capacity();
        stats.pages_free = self.pool.free();
        stats.pages_shared = self.pool.shared();
        let mut retained: Vec<u32> = self.entries.values().flat_map(|e| e.pages.iter().copied()).collect();
        retained.sort_unstable();
        retained.dedup();
        stats.pages_retained = retained.len();
        stats.mark_slots = self.arena.slots();
        stats.marks_in_use = self.arena.in_use();
        stats.host = self.host.as_ref().map(HostCache::metrics);
        stats
    }

    /// Evict every snapshot (tests and shutdown).
    pub fn clear<F: PrefixFamily>(&mut self, family: &F) -> Result<(), PrefixError> {
        while let Some(id) = self.victim(None) {
            self.evict(family, id)?;
        }
        Ok(())
    }

    fn victim(&self, keep: Option<EntryId>) -> Option<EntryId> {
        victim(self.entries.iter().filter(|(&id, _)| Some(id) != keep).map(|(&id, e)| (id, e.kind, e.last_use)))
    }

    fn drain<F: PrefixFamily>(&mut self, family: &F) -> Result<(), PrefixError> {
        if self.dirty {
            family.drain().map_err(|source| PrefixError::Family { what: "drain", source })?;
            self.dirty = false;
        }
        Ok(())
    }

    fn give_back<F: PrefixFamily>(&mut self, family: &F, slot: Option<MarkSlot>) -> Result<(), PrefixError> {
        if let Some(slot) = slot {
            self.drain(family)?;
            self.arena.give_back(slot);
        }
        Ok(())
    }

    /// Drop a device snapshot: its host copy finishes within budget first, then its storage goes
    /// back once the family's queued copies drained.
    fn evict<F: PrefixFamily>(&mut self, family: &F, id: EntryId) -> Result<(), PrefixError> {
        let Some(entry) = self.entries.remove(&id) else { return Ok(()) };
        self.retained.bank_mut(entry.kind).remove_exact(&entry.tokens);
        if let Some(host) = &mut self.host {
            match host.before_device_evict(entry.ticket) {
                EvictDecision::Clean => {}
                EvictDecision::WaitedClean { .. } => self.stats.host_evict_waits += 1,
                EvictDecision::DroppedUncached => self.stats.host_evict_uncached += 1,
            }
        }
        self.stats.evictions += 1;
        self.release(family, &entry.pages)?;
        self.give_back(family, entry.mark)
    }

    /// Host identities of an entry's pages: full pages by content (hash chain over their tokens),
    /// the partial tail by device allocation.
    fn identities(&self, tokens: &[u32], pages: &[u32]) -> Vec<DevicePageId> {
        let chain = page_chain(tokens, self.layout.page_rows);
        pages
            .iter()
            .enumerate()
            .map(|(i, &page)| match chain.get(i) {
                Some(&id) => content_id(0, id),
                None => DevicePageId { compressor: 0, page, generation: self.pool.generation(page) },
            })
            .collect()
    }

    /// Issue the write-behind host copy of a new snapshot (after its device copies drained).
    fn host_store<F: PrefixFamily>(&mut self, family: &F, id: EntryId) -> Result<(), PrefixError> {
        if self.host.is_none() {
            return Ok(());
        }
        self.drain(family)?;
        let entry = self.entries.get(&id).expect("stored entry is retained");
        let ids = self.identities(&entry.tokens, &entry.pages);
        let mut pages: [Vec<DevicePage>; cuteafd_hostcache::COMPRESSORS] = Default::default();
        pages[0] = entry.pages.iter().zip(ids).map(|(&page, id)| DevicePage { id, segments: family.page_segments(page) }).collect();
        let snapshot = DeviceSnapshot {
            meta: SnapshotMeta { kind: entry.kind, tokens: entry.tokens.clone(), end: entry.len() as u32, has_draft: false },
            pages,
            tail: entry.mark.map_or_else(Vec::new, |slot| family.mark_segments(slot)),
            draft: None,
            scores: Vec::new(),
        };
        let payload = HostPayload { after: entry.after.clone() };
        let host = self.host.as_mut().expect("checked");
        let ticket = match host.store(&snapshot, payload) {
            StoreOutcome::Issued(ticket) | StoreOutcome::Deferred(ticket) => Some(ticket),
            StoreOutcome::Skipped(_) => {
                self.stats.host_store_skips += 1;
                None
            }
        };
        self.entries.get_mut(&id).expect("checked").ticket = ticket;
        Ok(())
    }

    /// On a device miss: rebuild the best host snapshot on the device so the device path finds
    /// it. Anything short of a completed restore is a miss.
    fn promote<F: PrefixFamily>(&mut self, family: &F, tokens: &[u32], sampled: bool) -> Result<Option<Hit>, PrefixError> {
        let Some(host) = self.host.as_mut() else { return Ok(None) };
        let Some(hit) = host.lookup(tokens) else { return Ok(None) };
        let (Some(snapshot_tokens), Some(payload)) = (host.snapshot_tokens(hit.key), host.payload(hit.key)) else {
            return Ok(None);
        };
        let snapshot_tokens = snapshot_tokens.to_vec();
        let after = payload.after.clone();
        let len = snapshot_tokens.len();
        if len == tokens.len() && !after.serves(sampled) {
            return Ok(None);
        }
        let need = self.pool.pages_for(len);
        if !self.make_room(family, need, None)? {
            return Ok(None);
        }
        let slot = if self.layout.mark_bytes > 0 {
            loop {
                match self.arena.take() {
                    Ok(slot) => break Some(slot),
                    Err(_) => match self.victim(None) {
                        Some(id) => self.evict(family, id)?,
                        None => return Ok(None),
                    },
                }
            }
        } else {
            None
        };
        if self.pool.free() < need {
            self.give_back(family, slot)?;
            return Ok(None);
        }
        let pages = self.pool.alloc(need)?;
        // The restore stream writes these pages and the slot: nothing queued may still use them.
        self.drain(family)?;
        let ids = self.identities(&snapshot_tokens, &pages);
        let mut target_pages: [Vec<DevicePage>; cuteafd_hostcache::COMPRESSORS] = Default::default();
        target_pages[0] = pages.iter().zip(ids).map(|(&page, id)| DevicePage { id, segments: family.page_segments(page) }).collect();
        let target = RestoreTarget {
            pages: target_pages,
            tail: slot.map_or_else(Vec::new, |slot| family.mark_segments(slot)),
            draft: None,
            scores: Vec::new(),
        };
        let host = self.host.as_mut().expect("checked");
        match host.restore(hit.key, &target) {
            RestoreOutcome::Done { .. } => {}
            outcome => {
                if outcome == RestoreOutcome::TimedOut {
                    // The copies may still land: wait them out before the storage goes back.
                    let engine = host.engine_mut();
                    if let Ok(event) = engine.record(Stream::Restore) {
                        let _ = engine.wait(event, u64::MAX);
                    }
                }
                tracing::warn!(target: "cuteafd::prefix", ?outcome, tokens = len, "host restore abandoned; prefilling");
                self.stats.restore_failures += 1;
                self.release(family, &pages)?;
                self.give_back(family, slot)?;
                return Ok(None);
            }
        }
        self.clock += 1;
        let id = self.next_id;
        self.next_id += 1;
        self.entries.insert(id, Entry { tokens: snapshot_tokens.clone(), kind: hit.kind, pages, mark: slot, after,
            last_use: self.clock, ticket: None });
        if let Some(evicted) = self.retained.bank_mut(hit.kind).insert(&snapshot_tokens, id) {
            self.evict(family, evicted)?;
        }
        self.stats.promotions += 1;
        Ok(self.lookup(tokens, sampled))
    }
}
