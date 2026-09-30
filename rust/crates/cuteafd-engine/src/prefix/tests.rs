//! A fake family whose device state lives in the host tier's stub memory: page row `r` holds a
//! hash of `tokens[..=r]`, the sliding-window ring holds the same for its last rows, and every
//! forward first checks that the whole context it reads is exactly what a straight prefill of
//! the same tokens would have written. Copies are queued until a drain or a forward (stream
//! order), so a snapshot published before its copies drained restores wrong bytes and fails.
use super::*;
use cuteafd_hostcache::config::Config as HostConfig;
use cuteafd_hostcache::copy::{CopyEngine, CopyModel, DeviceRange, Event, Stream, StubCopyEngine};
use cuteafd_hostcache::pool::{HostChunk, HostRange, PinnedMemory};
use std::cell::{Cell, RefCell};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::rc::Rc;

const ROWS: usize = 4; // page rows
const RING: usize = 16; // ring slots per sequence
const WINDOW: usize = 8; // rows a forward reads back from the ring
const ROW: usize = 8; // bytes per row

#[derive(Clone)]
struct Shared(Rc<RefCell<StubCopyEngine>>);

impl PinnedMemory for Shared {
    fn allocate_chunk(&mut self, bytes: usize) -> anyhow::Result<HostChunk> {
        self.0.borrow_mut().allocate_chunk(bytes)
    }
    fn release_chunk(&mut self, chunk: HostChunk) -> anyhow::Result<()> {
        self.0.borrow_mut().release_chunk(chunk)
    }
}

impl CopyEngine for Shared {
    fn d2h(&mut self, stream: Stream, src: DeviceRange, dst: HostRange) -> anyhow::Result<()> {
        self.0.borrow_mut().d2h(stream, src, dst)
    }
    fn h2d(&mut self, stream: Stream, src: HostRange, dst: DeviceRange) -> anyhow::Result<()> {
        self.0.borrow_mut().h2d(stream, src, dst)
    }
    fn d2h_many(&mut self, stream: Stream, copies: &[(DeviceRange, HostRange)]) -> anyhow::Result<()> {
        self.0.borrow_mut().d2h_many(stream, copies)
    }
    fn h2d_many(&mut self, stream: Stream, copies: &[(HostRange, DeviceRange)]) -> anyhow::Result<()> {
        self.0.borrow_mut().h2d_many(stream, copies)
    }
    fn record(&mut self, stream: Stream) -> anyhow::Result<Event> {
        self.0.borrow_mut().record(stream)
    }
    fn completed(&mut self, event: Event) -> anyhow::Result<bool> {
        self.0.borrow_mut().completed(event)
    }
    fn wait(&mut self, event: Event, budget_ns: u64) -> anyhow::Result<bool> {
        self.0.borrow_mut().wait(event, budget_ns)
    }
    fn now_ns(&self) -> u64 {
        self.0.borrow().now_ns()
    }
}

fn value(salt: u64, tokens: &[u32]) -> [u8; ROW] {
    let mut h = DefaultHasher::new();
    salt.hash(&mut h);
    tokens.hash(&mut h);
    h.finish().to_le_bytes()
}

#[derive(Debug, Clone)]
struct Placement {
    pages: Vec<u32>,
    ring: usize,
    len: usize,
}

enum Op {
    Copy(DeviceRange, DeviceRange),
}

struct Fake {
    mem: Rc<RefCell<StubCopyEngine>>,
    pages: usize,
    rings: usize,
    slots: usize,
    queue: RefCell<Vec<Op>>,
    fail_restore: Cell<bool>,
    drains: Cell<usize>,
}

impl Fake {
    fn new(pages: usize, rings: usize, slots: usize) -> Self {
        let bytes = (pages * ROWS + rings * RING + slots * WINDOW) * ROW;
        let mem = Rc::new(RefCell::new(StubCopyEngine::new(CopyModel::default(), bytes, 1 << 26)));
        Self { mem, pages, rings, slots, queue: RefCell::new(Vec::new()), fail_restore: Cell::new(false), drains: Cell::new(0) }
    }
    fn layout(&self) -> FamilyLayout {
        FamilyLayout { page_rows: ROWS, pages: self.pages, page_bytes: ROWS * ROW, mark_bytes: WINDOW * ROW,
            draft_bytes: 0, rule: ReuseRule::EXACT }
    }
    fn page_row(&self, page: u32, row: usize) -> DeviceRange {
        DeviceRange { addr: ((page as usize * ROWS + row) * ROW) as u64, bytes: ROW }
    }
    fn ring_row(&self, ring: usize, position: usize) -> DeviceRange {
        assert!(ring < self.rings);
        DeviceRange { addr: ((self.pages * ROWS + ring * RING + position % RING) * ROW) as u64, bytes: ROW }
    }
    fn slot_row(&self, slot: MarkSlot, i: usize) -> DeviceRange {
        assert!((slot.0 as usize) < self.slots);
        DeviceRange { addr: ((self.pages * ROWS + self.rings * RING + slot.0 as usize * WINDOW + i) * ROW) as u64, bytes: ROW }
    }
    fn flush(&self) {
        let mut mem = self.mem.borrow_mut();
        for Op::Copy(from, to) in self.queue.borrow_mut().drain(..) {
            let bytes = mem.read_device(from);
            mem.write_device(to, &bytes);
        }
    }
    /// Prefill `tokens[p.len..]` after checking the whole context; returns a logit row.
    fn forward(&self, p: &mut Placement, tokens: &[u32]) -> Result<Vec<f32>, String> {
        self.flush();
        let mut mem = self.mem.borrow_mut();
        for r in 0..p.len {
            let page = p.pages[r / ROWS];
            if mem.read_device(self.page_row(page, r % ROWS)) != value(1, &tokens[..=r]) {
                return Err(format!("page row {r} differs"));
            }
        }
        for r in p.len.saturating_sub(WINDOW)..p.len {
            if mem.read_device(self.ring_row(p.ring, r)) != value(2, &tokens[..=r]) {
                return Err(format!("ring row {r} differs"));
            }
        }
        for r in p.len..tokens.len() {
            let page = *p.pages.get(r / ROWS).ok_or("past the placement")?;
            mem.write_device(self.page_row(page, r % ROWS), &value(1, &tokens[..=r]));
            mem.write_device(self.ring_row(p.ring, r), &value(2, &tokens[..=r]));
        }
        p.len = tokens.len();
        let v = u64::from_le_bytes(value(3, tokens));
        Ok((0..8).map(|i| ((v >> (i * 8)) & 0xff) as f32).collect())
    }
}

impl PrefixFamily for Fake {
    type Placement = Placement;
    fn layout(&self) -> FamilyLayout {
        Fake::layout(self)
    }
    fn pages<'p>(&self, placement: &'p Placement) -> &'p [u32] {
        &placement.pages
    }
    fn commit_point(&self, placement: &Placement) -> usize {
        placement.len
    }
    fn capture(&self, slot: MarkSlot, p: &Placement, len: usize) -> Result<(), BoxError> {
        let first = len.saturating_sub(WINDOW);
        for (i, r) in (first..len).enumerate() {
            self.queue.borrow_mut().push(Op::Copy(self.ring_row(p.ring, r), self.slot_row(slot, i)));
        }
        Ok(())
    }
    fn restore(&self, mark: Option<MarkSlot>, p: &mut Placement, len: usize) -> Result<(), BoxError> {
        if self.fail_restore.get() {
            return Err("injected restore failure".into());
        }
        let slot = mark.ok_or("fake family always has a mark")?;
        let first = len.saturating_sub(WINDOW);
        for (i, r) in (first..len).enumerate() {
            self.queue.borrow_mut().push(Op::Copy(self.slot_row(slot, i), self.ring_row(p.ring, r)));
        }
        p.len = len;
        Ok(())
    }
    fn copy_rows(&self, copy: TailCopy) -> Result<(), BoxError> {
        for row in 0..copy.rows {
            self.queue.borrow_mut().push(Op::Copy(self.page_row(copy.from, row), self.page_row(copy.to, row)));
        }
        Ok(())
    }
    fn drain(&self) -> Result<(), BoxError> {
        self.drains.set(self.drains.get() + 1);
        self.flush();
        Ok(())
    }
    fn page_segments(&self, page: u32) -> Vec<DeviceRange> {
        vec![DeviceRange { addr: self.page_row(page, 0).addr, bytes: ROWS * ROW }]
    }
    fn mark_segments(&self, slot: MarkSlot) -> Vec<DeviceRange> {
        vec![DeviceRange { addr: self.slot_row(slot, 0).addr, bytes: WINDOW * ROW }]
    }
}

fn config(entries: usize, slots: usize) -> PrefixConfig {
    PrefixConfig { entries, mark_slots: slots, keep_logits: true, min_tokens: 1 }
}

fn cache(fake: &Fake, entries: usize, host_bytes: u64) -> PrefixCache<Shared> {
    let host = HostConfig { bytes: host_bytes, chunk_bytes: 1 << 16, min_tokens: 1, ..HostConfig::default() };
    PrefixCache::new(fake.layout(), config(entries, fake.slots), Some((host, Shared(fake.mem.clone())))).unwrap()
}

/// One request through the cache: admit, prefill the rest, capture its prompt, then optionally
/// "decode" `generated` tokens and capture a turn. Returns (resume, final tokens, placement).
fn serve(cache: &mut PrefixCache<Shared>, fake: &Fake, ring: usize, prompt: &[u32], generated: &[u32],
    capacity: usize) -> (usize, Vec<u32>, Placement) {
    let admitted = cache.admit(fake, prompt, capacity, false, |pages| Placement { pages, ring, len: 0 }).unwrap();
    let mut placement = admitted.placement;
    assert_eq!(placement.len, admitted.resume);
    let logits = if admitted.resume == prompt.len() {
        admitted.after.expect("exact-length hit brings its first token").logits.unwrap().to_vec()
    } else {
        let logits = fake.forward(&mut placement, prompt).unwrap();
        cache.capture(fake, SnapshotKind::Prompt, prompt, &placement, After::from_logits(&logits, true)).unwrap();
        logits
    };
    assert_eq!(logits.len(), 8);
    let mut tokens = prompt.to_vec();
    if !generated.is_empty() {
        tokens.extend_from_slice(generated);
        let logits = fake.forward(&mut placement, &tokens).unwrap();
        cache.capture(fake, SnapshotKind::Turn, &tokens, &placement, After::from_logits(&logits, true)).unwrap();
    }
    (admitted.resume, tokens, placement)
}

fn seq(base: u32, n: usize) -> Vec<u32> {
    (0..n as u32).map(|i| base + i).collect()
}

#[test]
fn prompt_repeat_is_an_exact_hit_with_its_logits_and_no_forward() {
    let fake = Fake::new(64, 4, 8);
    let mut cache = cache(&fake, 4, 0);
    let prompt = seq(100, 50);
    let (resume, _, a) = serve(&mut cache, &fake, 0, &prompt, &[], 64);
    assert_eq!(resume, 0);
    let (resume, _, mut b) = serve(&mut cache, &fake, 1, &prompt, &[], 64);
    assert_eq!(resume, 50);
    // B continues exactly as a straight prefill would (the forward checks every context row).
    let mut longer = prompt.clone();
    longer.extend(seq(900, 9));
    fake.forward(&mut b, &longer).unwrap();
    let stats = cache.stats();
    assert_eq!((stats.hits, stats.exact_hits, stats.hit_tokens, stats.cow_copies), (1, 1, 50, 1));
    cache.release(&fake, &a.pages).unwrap();
    cache.release(&fake, &b.pages).unwrap();
    cache.clear(&fake).unwrap();
    assert_eq!(cache.pool().free(), 64);
    assert_eq!(cache.arena().in_use(), 0);
}

#[test]
fn a_turn_snapshot_serves_the_next_turn_and_the_writer_keeps_its_own_tail() {
    let fake = Fake::new(64, 4, 8);
    let mut cache = cache(&fake, 4, 0);
    let prompt = seq(100, 30);
    let (_, turn, mut a) = serve(&mut cache, &fake, 0, &prompt, &seq(500, 13), 80);
    // The writer keeps decoding past its turn snapshot (43 rows: page 10 is partial and copied).
    let mut more = turn.clone();
    more.extend(seq(700, 7));
    fake.forward(&mut a, &more).unwrap();
    // The next turn re-sends the whole conversation plus a new user message.
    let mut next = turn.clone();
    next.extend(seq(800, 11));
    let (resume, _, mut b) = serve(&mut cache, &fake, 1, &next, &[], 80);
    assert_eq!(resume, turn.len());
    let mut after = next.clone();
    after.push(1);
    fake.forward(&mut b, &after).unwrap();
    // The writer's rows past the snapshot are untouched by B.
    fake.forward(&mut a, &more).unwrap();
    // A prefix of the snapshot is not an exact frontier: MiMo-style families reuse no partial match.
    let mut partial = turn[..35].to_vec();
    partial.push(4242);
    let (resume, _, c) = serve(&mut cache, &fake, 2, &partial, &[], 80);
    assert_eq!(resume, prompt.len(), "the prompt snapshot is the exact ancestor");
    for p in [a, b, c] {
        cache.release(&fake, &p.pages).unwrap();
    }
    cache.clear(&fake).unwrap();
    assert_eq!(cache.pool().free(), 64);
}

#[test]
fn a_parked_prefill_without_logits_gives_way_to_a_shorter_snapshot() {
    let fake = Fake::new(64, 4, 8);
    let mut cache = cache(&fake, 4, 0);
    let prompt = seq(100, 40);
    let (_, _, a) = serve(&mut cache, &fake, 0, &prompt[..20], &[], 64);
    // A prefill of the whole prompt is cancelled at a chunk boundary (32 rows): parked.
    let mut admitted = cache.admit(&fake, &prompt, 64, false, |pages| Placement { pages, ring: 1, len: 0 }).unwrap();
    assert_eq!(admitted.resume, 20);
    fake.forward(&mut admitted.placement, &prompt[..32]).unwrap();
    assert!(cache.park(&fake, &prompt[..32], &admitted.placement).unwrap());
    cache.release(&fake, &admitted.placement.pages).unwrap();
    // The retry resumes at the parked boundary.
    let (resume, _, b) = serve(&mut cache, &fake, 1, &prompt, &[], 64);
    assert_eq!(resume, 32);
    // A request of exactly the parked tokens cannot take its first token from the parked
    // snapshot (no logits), so it resumes from the 20-token prompt snapshot instead.
    let admitted = cache.admit(&fake, &prompt[..32], 64, false, |pages| Placement { pages, ring: 2, len: 0 }).unwrap();
    assert_eq!((admitted.resume, admitted.after.is_none()), (20, true));
    assert_eq!(cache.stats().parked, 1);
    for pages in [a.pages, b.pages, admitted.placement.pages] {
        cache.release(&fake, &pages).unwrap();
    }
}

#[test]
fn eviction_is_least_recent_prompt_first_and_admission_makes_room() {
    // 20 pages: a 30-token request takes 8, its snapshot shares 7 of them and copies 1.
    let fake = Fake::new(20, 4, 8);
    let mut cache = cache(&fake, 8, 0);
    let (_, _, a) = serve(&mut cache, &fake, 0, &seq(100, 30), &[], 30);
    cache.release(&fake, &a.pages).unwrap();
    let (_, _, b) = serve(&mut cache, &fake, 1, &seq(200, 30), &[], 30);
    cache.release(&fake, &b.pages).unwrap();
    assert_eq!(cache.stats().entries_prompt, 2);
    // Touch A's snapshot so B's is the least recently used.
    let (resume, _, a2) = serve(&mut cache, &fake, 0, &seq(100, 30), &[], 30);
    assert_eq!(resume, 30);
    cache.release(&fake, &a2.pages).unwrap();
    // A third conversation needs room: B's snapshot goes, A's stays.
    let (_, _, c) = serve(&mut cache, &fake, 2, &seq(300, 40), &[], 40);
    assert!(cache.stats().evictions >= 1);
    let (resume, _, a3) = serve(&mut cache, &fake, 0, &seq(100, 30), &[], 30);
    assert_eq!(resume, 30);
    cache.release(&fake, &a3.pages).unwrap();
    let (resume, _, b2) = serve(&mut cache, &fake, 1, &seq(200, 30), &[], 30);
    assert_eq!(resume, 0);
    for pages in [c.pages, b2.pages] {
        cache.release(&fake, &pages).unwrap();
    }
    cache.clear(&fake).unwrap();
    assert_eq!(cache.pool().free(), 20);
}

#[test]
fn full_arena_and_full_banks_evict_the_victim() {
    let fake = Fake::new(128, 4, 2);
    let mut cache = cache(&fake, 8, 0);
    for base in [100, 200, 300] {
        let (_, _, p) = serve(&mut cache, &fake, 0, &seq(base, 20), &[], 20);
        cache.release(&fake, &p.pages).unwrap();
    }
    assert_eq!((cache.stats().entries_prompt, cache.arena().in_use()), (2, 2));
    let (resume, _, p) = serve(&mut cache, &fake, 0, &seq(100, 20), &[], 20);
    assert_eq!(resume, 0, "the oldest snapshot was evicted for the third mark");
    cache.release(&fake, &p.pages).unwrap();
    let mut small = cache_with(&fake, 1);
    for base in [100, 200] {
        let (_, _, p) = serve(&mut small, &fake, 0, &seq(base, 20), &[], 20);
        small.release(&fake, &p.pages).unwrap();
    }
    assert_eq!(small.stats().entries_prompt, 1);
}

fn cache_with(fake: &Fake, entries: usize) -> PrefixCache<Shared> {
    cache(fake, entries, 0)
}

#[test]
fn a_failed_restore_is_a_miss_and_holds_nothing() {
    let fake = Fake::new(32, 4, 4);
    let mut cache = cache(&fake, 4, 0);
    let (_, _, a) = serve(&mut cache, &fake, 0, &seq(100, 20), &[], 20);
    cache.release(&fake, &a.pages).unwrap();
    fake.fail_restore.set(true);
    let admitted = cache.admit(&fake, &seq(100, 21), 24, false, |pages| Placement { pages, ring: 1, len: 0 }).unwrap();
    fake.fail_restore.set(false);
    assert_eq!((admitted.resume, cache.stats().restore_failures), (0, 1));
    cache.release(&fake, &admitted.placement.pages).unwrap();
    cache.clear(&fake).unwrap();
    assert_eq!(cache.pool().free(), 32);
}

#[test]
fn disabled_cache_is_a_plain_allocator() {
    let fake = Fake::new(8, 2, 0);
    let mut cache = cache(&fake, 0, 1 << 20);
    let (resume, _, a) = serve(&mut cache, &fake, 0, &seq(1, 20), &[], 20);
    let (resume2, _, b) = serve(&mut cache, &fake, 1, &seq(1, 11), &[], 12);
    assert_eq!((resume, resume2), (0, 0));
    assert!(cache.admit(&fake, &seq(1, 4), 4, false, |pages| Placement { pages, ring: 0, len: 0 }).is_err());
    cache.release(&fake, &a.pages).unwrap();
    cache.release(&fake, &b.pages).unwrap();
    assert_eq!((cache.pool().free(), cache.stats().captures_prompt), (8, 0));
}

#[test]
fn host_tier_restores_evicted_snapshots_exactly_and_shares_identical_prefixes() {
    let fake = Fake::new(24, 4, 8);
    let mut cache = cache(&fake, 8, 1 << 20);
    let system = seq(100, 16);
    let mut first = system.clone();
    first.extend(seq(1000, 10));
    let mut second = system.clone();
    second.extend(seq(2000, 10));
    for (ring, prompt) in [(0, &first), (1, &second)] {
        let (_, _, p) = serve(&mut cache, &fake, ring, prompt, &[], 26);
        // The store stream runs ahead of anything the family does next: a snapshot published
        // before its capture copies drained would be stored with stale bytes.
        fake.mem.borrow_mut().advance(10_000_000);
        cache.tick();
        cache.release(&fake, &p.pages).unwrap();
    }
    let host = cache.stats().host.unwrap();
    assert_eq!(host.stores_completed, 2);
    // The second prompt's first 16 tokens (4 full pages) were computed by a different request
    // into different device pages, yet share the first snapshot's host copies by content.
    assert_eq!((host.pages_copied, host.pages_shared), (7 + 3, 4));
    // Evict both from the device.
    cache.clear(&fake).unwrap();
    assert_eq!(cache.pool().free(), 24);
    // The first conversation comes back: promoted from the host tier, then continued exactly.
    let mut next = first.clone();
    next.extend(seq(3000, 5));
    let (resume, _, mut p) = serve(&mut cache, &fake, 2, &next, &[], 40);
    assert_eq!(resume, first.len());
    let stats = cache.stats();
    assert_eq!((stats.promotions, stats.host.unwrap().restores), (1, 1));
    let mut longer = next.clone();
    longer.push(7);
    fake.forward(&mut p, &longer).unwrap();
    cache.release(&fake, &p.pages).unwrap();
}

/// Randomized conversations over shared system prompts, with rings reused across requests and a
/// small pool, so eviction, forks, copies and host promotions interleave. Every forward checks its
/// whole context; at the end every page and slot is back.
#[test]
fn torture_interleaved_conversations_stay_exact_and_leak_nothing() {
    for host_bytes in [0u64, 1 << 20] {
        let fake = Fake::new(40, 4, 6);
        let mut cache = cache(&fake, 3, host_bytes);
        let mut rng = 0x2545_f491_4f6c_dd1du64;
        let mut next = |n: u64| {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng % n
        };
        let mut conversations: Vec<Vec<u32>> = (0..4).map(|c| seq(100 * (c % 2 + 1), 9)).collect();
        for step in 0..300 {
            let c = next(4) as usize;
            let mut prompt = conversations[c].clone();
            prompt.extend(seq(10_000 + step * 7, 1 + next(6) as usize));
            if prompt.len() > 60 {
                prompt = seq(100 * (c as u32 % 2 + 1), 9);
            }
            let generated = seq(50_000 + step * 3, next(5) as usize);
            let capacity = prompt.len() + generated.len() + next(4) as usize;
            let (_, tokens, p) = serve(&mut cache, &fake, c, &prompt, &generated, capacity);
            conversations[c] = tokens;
            fake.mem.borrow_mut().advance(1_000_000);
            cache.tick();
            cache.release(&fake, &p.pages).unwrap();
        }
        let stats = cache.stats();
        assert!(stats.hits > 50, "{stats:?}");
        cache.clear(&fake).unwrap();
        assert_eq!((cache.pool().free(), cache.arena().in_use()), (40, 0), "{stats:?}");
    }
}
