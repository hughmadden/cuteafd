// Failure interleavings use the same queued-copy family and accounting as normal prefix tests.
impl Fake {
    /// Mirror scheduler reclaim: a failed drain keeps the external request's owners intact.
    fn external_release(&self, cache: &mut PrefixCache<Shared>, ring: usize) -> Result<(), PrefixError> {
        let pages = self.bound_pages.borrow().as_ref().unwrap()[ring].clone();
        cache.release(self, &pages)?;
        self.discard(&mut Placement { pages, ring, len: 0, ring_from: 0 })
            .map_err(|source| PrefixError::Family { what: "external release", source })
    }
}

#[test]
fn partial_host_promotion_drains_before_taking_entry_mark() {
    for cleanup_fails in [false, true] {
        let mut fake = Fake::new(64, 2, 8);
        fake.rule = ReuseRule { align: 2, replay: Some(WINDOW) };
        let mut cache = cache(&fake, 8, 1 << 20);
        let saved = seq(100, 24);
        let (_, _, p) = serve(&mut cache, &fake, 0, &saved, &[], 24);
        cache.release(&fake, &p.pages).unwrap();
        fake.mem.borrow_mut().advance(10_000_000);
        cache.tick();
        cache.clear(&fake).unwrap(); // The divergent request must promote from host.
        assert_eq!((cache.pool().free(), cache.arena().in_use()), (64, 0));
        *fake.bound_pages.borrow_mut() = Some(vec![Vec::new(); 2]);
        let mut query = saved[..20].to_vec();
        query.extend(seq(500, 4));
        fake.drain_after_restore.set(cleanup_fails);
        let builds = Cell::new(0);
        let result = cache.admit(&fake, &query, 24, false, |pages| {
            builds.set(builds.get() + 1);
            Placement { pages, ring: 1, len: 0, ring_from: 0 }
        });
        assert_eq!(builds.get(), 1, "cleanup failure must abort without retrying");
        assert_eq!(cache.stats().promotions, 1);
        if cleanup_fails {
            assert!(matches!(result, Err(PrefixError::Family { what: "drain", .. })));
            assert_eq!(cache.arena().in_use(), 1, "the promoted entry still owns its mark");
            assert_eq!((cache.stats().hits, cache.stats().partial_hits, cache.stats().host_hits), (0, 0, 0));
        } else {
            let mut admitted = result.unwrap();
            assert_eq!(admitted.resume, 12, "aligned common 20 minus replay 8");
            let source = admitted.source.unwrap();
            assert!(source.host && source.partial);
            assert_eq!(admitted.placement.ring_from, 12);
            fake.forward(&mut admitted.placement, &query).unwrap();
            assert_eq!(cache.arena().in_use(), 0);
            assert_eq!((cache.stats().hits, cache.stats().partial_hits, cache.stats().host_hits), (1, 1, 1));
        }
        assert_eq!((cache.stats().quarantined, cache.stats().mark_restore_misses,
            cache.stats().restore_failures, cache.stats().fork_no_room), (0, 0, 0, 0));
        fake.external_release(&mut cache, 1).unwrap();
        cache.clear(&fake).unwrap();
        assert_eq!((cache.pool().free(), cache.arena().in_use()), (64, 0));
    }
}

#[test]
fn restore_drain_failure_leaves_bound_fork_to_caller_and_quarantines_only_temporary_mark() {
    for restore_fails in [false, true] {
        let fake = Fake::new(64, 2, 8);
        let mut cache = cache(&fake, 8, 1 << 20);
        for len in [8, 12] {
            let (_, _, p) = serve(&mut cache, &fake, 0, &seq(100, len), &[], len);
            cache.release(&fake, &p.pages).unwrap();
            fake.mem.borrow_mut().advance(10_000_000);
            cache.tick();
        }
        let before = cache.pool().free();
        let builds = Cell::new(0);
        *fake.bound_pages.borrow_mut() = Some(vec![Vec::new(); 2]);
        fake.fail_restore.set(restore_fails);
        fake.drain_after_restore.set(true);
        let query = seq(100, 15);
        let result = cache.admit(&fake, &query, 16, false, |pages| {
            builds.set(builds.get() + 1);
            Placement { pages, ring: 1, len: 0, ring_from: 0 }
        });
        assert!(result.is_err());
        assert_eq!(builds.get(), 1, "failed cleanup must abort admission");
        assert_eq!((cache.stats().quarantined, cache.arena().in_use()), (1, 1));
        assert_eq!(cache.pool().free(), before - 1, "one fresh fork page remains owned");
        assert_eq!(fake.bound_pages.borrow().as_ref().unwrap()[1].len(), 4);
        assert_eq!(cache.stats().fork_no_room, 0);
        // Repeated caller cleanup failures retain both the binding and staging mark.
        fake.fail_drain.set(true);
        for _ in 0..2 {
            assert!(fake.external_release(&mut cache, 1).is_err());
            assert_eq!(cache.pool().free(), before - 1);
            assert_eq!(cache.arena().in_use(), 1);
            assert_eq!(fake.bound_pages.borrow().as_ref().unwrap()[1].len(), 4);
        }
        fake.fail_drain.set(false);
        fake.external_release(&mut cache, 1).unwrap();
        assert_eq!(cache.pool().free(), before, "caller reclaims the fork exactly once");
        fake.fail_restore.set(false);
        let mut p = cache.admit(&fake, &query, 16, false,
            |pages| Placement { pages, ring: 1, len: 0, ring_from: 0 }).unwrap().placement;
        assert_eq!(p.len, 12);
        fake.forward(&mut p, &query).unwrap();
        fake.external_release(&mut cache, 1).unwrap();
        cache.clear(&fake).unwrap();
        assert_eq!((cache.pool().free(), cache.arena().in_use()), (64, 0));
    }
}

#[test]
fn failed_discard_keeps_bound_pages_with_caller() {
    let fake = Fake::new(64, 2, 8);
    let mut cache = cache(&fake, 8, 1 << 20);
    let (_, _, p) = serve(&mut cache, &fake, 0, &seq(100, 12), &[], 12);
    cache.release(&fake, &p.pages).unwrap();
    fake.mem.borrow_mut().advance(10_000_000);
    cache.tick();
    *fake.bound_pages.borrow_mut() = Some(vec![Vec::new(); 2]);
    fake.fail_restore_once.set(true);
    fake.fail_discard_once.set(true);
    let builds = Cell::new(0);
    assert!(cache.admit(&fake, &seq(100, 15), 16, false, |pages| {
        builds.set(builds.get() + 1);
        Placement { pages, ring: 1, len: 0, ring_from: 0 }
    }).is_err());
    assert_eq!(builds.get(), 1);
    assert_eq!(cache.pool().free(), 60);
    assert_eq!((cache.stats().quarantined, cache.arena().in_use()), (1, 1));
    fake.external_release(&mut cache, 1).unwrap();
    cache.clear(&fake).unwrap();
    assert_eq!((cache.pool().free(), cache.arena().in_use()), (64, 0));
}

#[test]
fn restore_drain_failure_leaves_live_draft_binding_to_caller() {
    let fake = Fake::new(80, 2, 8);
    let draft_tokens = 2048;
    let bytes = fake.draft_range(0).addr as usize + draft_tokens * ROW;
    *fake.mem.borrow_mut() = StubCopyEngine::new(CopyModel::default(), bytes, 1 << 26);
    *fake.draft.borrow_mut() = Some(DraftFixture {
        pool: DraftPagePool::new(draft_tokens, 2, 4096, 23).unwrap(), bindings: vec![None; 2],
    });
    let host = HostConfig { bytes: 1 << 20, chunk_bytes: 1 << 16, min_tokens: 1, ..HostConfig::default() };
    let mut cache = PrefixCache::new_with_draft_pages(fake.layout(), config(8, 8),
        Some((host, Shared(fake.mem.clone()))), fake.draft_page_bytes()).unwrap();
    let tokens = seq(100, 65);
    let mut p = cache.admit_cold(&fake, 65, 65,
        |pages| Placement { pages, ring: 0, len: 0, ring_from: 0 }).unwrap().placement;
    let logits = fake.forward(&mut p, &tokens).unwrap();
    {
        let mut draft = fake.draft.borrow_mut();
        let draft = draft.as_mut().unwrap();
        let reservation = draft.pool.reserve(draft.bindings[0].unwrap(), 0, 65).unwrap();
        draft.pool.commit(reservation, 65).unwrap();
    }
    cache.capture(&fake, SnapshotKind::Prompt, &tokens, &p, After::from_logits(&logits, true)).unwrap();
    fake.mem.borrow_mut().advance(10_000_000);
    cache.tick();
    fake.discard(&mut p).unwrap();
    cache.release(&fake, &p.pages).unwrap();
    let releases = fake.draft_releases.get();
    *fake.bound_pages.borrow_mut() = Some(vec![Vec::new(); 2]);
    fake.drain_after_restore.set(true);
    let builds = Cell::new(0);
    assert!(cache.admit(&fake, &tokens, 68, false, |pages| {
        builds.set(builds.get() + 1);
        Placement { pages, ring: 1, len: 0, ring_from: 0 }
    }).is_err());
    assert_eq!(builds.get(), 1);
    assert!(fake.draft.borrow().as_ref().unwrap().bindings[1].is_some());
    assert_eq!(fake.draft_releases.get(), releases);
    fake.external_release(&mut cache, 1).unwrap();
    assert_eq!(fake.draft_releases.get(), releases + 1);
    cache.clear(&fake).unwrap();
    assert_eq!(fake.draft_releases.get(), releases + 1, "clear releases snapshot attachments only");
    assert!(fake.draft.borrow().as_ref().unwrap().bindings.iter().all(Option::is_none));
    let draft = fake.draft.borrow();
    let pool = &draft.as_ref().unwrap().pool;
    assert_eq!(pool.free(), pool.capacity());
    assert_eq!((cache.pool().free(), cache.arena().in_use()), (80, 0));
}

#[test]
fn promotion_post_allocation_drain_failure_quarantines_pages_and_mark() {
    let family = Lagged::new(64, 2, 4);
    let mut cache = lag_cache(&family, 8, 1 << 20);
    let (_, p) = lag_serve(&mut cache, &family, 0, &seq(100, 32), 32);
    cache.release(&family, &p.pages).unwrap();
    family.fake.mem.borrow_mut().advance(10_000_000);
    cache.tick();
    cache.clear(&family).unwrap();
    let mut writer = cache.admit_cold(&family, 16, 16,
        |pages| Placement { pages, ring: 0, len: 0, ring_from: 0 }).unwrap().placement;
    family.fake.forward(&mut writer, &seq(900, 16)).unwrap();
    let Captured::Queued(ticket) = cache.queue_capture(&family, SnapshotKind::Prompt,
        &seq(900, 16), &[], &writer, After::default()).unwrap() else { panic!("queued capture") };
    let free = cache.pool().free();
    family.fake.fail_drain_once.set(true);
    let builds = Cell::new(0);
    assert!(cache.admit(&family, &seq(100, 33), 36, false, |pages| {
        builds.set(builds.get() + 1);
        Placement { pages, ring: 1, len: 0, ring_from: 0 }
    }).is_err());
    assert_eq!(builds.get(), 0, "promotion cleanup failure must not admit an ancestor or cold request");
    assert_eq!(cache.stats().quarantined, 1);
    assert_eq!(cache.pool().free(), free - 8);
    assert_eq!(cache.arena().in_use(), 2, "pending capture and quarantined promotion each own a slot");
    cache.abort_capture(&family, ticket).unwrap();
    cache.release(&family, &writer.pages).unwrap();
    let admitted = cache.admit(&family, &seq(100, 33), 36, false,
        |pages| Placement { pages, ring: 1, len: 0, ring_from: 0 }).unwrap();
    assert_eq!(admitted.resume, 32);
    cache.release(&family, &admitted.placement.pages).unwrap();
    cache.clear(&family).unwrap();
    assert_eq!((cache.pool().free(), cache.arena().in_use()), (64, 0));
}

#[test]
fn host_mark_segment_failure_retries_shallower_ancestor_after_clean_release() {
    let fake = Fake::pooled(64, 2);
    let mut cache = cache(&fake, 8, 1 << 20);
    for len in [8, 12] {
        let (_, _, p) = serve(&mut cache, &fake, 0, &seq(100, len), &[], len);
        cache.release(&fake, &p.pages).unwrap();
        fake.mem.borrow_mut().advance(10_000_000);
        cache.tick();
    }
    cache.clear(&fake).unwrap();
    fake.fail_mark_segments_once.set(true);
    let mut admitted = cache.admit(&fake, &seq(100, 15), 16, false,
        |pages| Placement { pages, ring: 1, len: 0, ring_from: 0 }).unwrap();
    assert_eq!(admitted.resume, 8);
    assert!(admitted.source.unwrap().host);
    assert_eq!((cache.stats().mark_restore_misses, cache.stats().restore_failures,
        cache.stats().fork_no_room, cache.stats().quarantined), (1, 1, 0, 0));
    fake.forward(&mut admitted.placement, &seq(100, 15)).unwrap();
    cache.release(&fake, &admitted.placement.pages).unwrap();
    cache.clear(&fake).unwrap();
    assert_eq!((cache.pool().free(), cache.arena().in_use()), (64, 0));
}

#[test]
fn missing_host_mark_preserves_partial_pages_but_exact_restore_falls_back() {
    let mut fake = Fake::new(64, 2, 8);
    fake.rule = ReuseRule { align: 2, replay: Some(WINDOW) };
    let host = HostConfig { bytes: 320, chunk_bytes: 64, min_tokens: 1, ..HostConfig::default() };
    let mut cache = PrefixCache::new(fake.layout(), config(8, 8), Some((host, Shared(fake.mem.clone())))).unwrap();
    let saved = seq(100, 24);
    // Three chunks of pages plus the tail chunk fill the quota. The unrelated
    // snapshot's next two page slabs must evict the only (device-referenced) key.
    for tokens in [&saved, &seq(900, 24)] {
        let mut p = cache.admit_cold(&fake, tokens.len(), tokens.len(),
            |pages| Placement { pages, ring: 0, len: 0, ring_from: 0 }).unwrap().placement;
        let logits = fake.forward(&mut p, tokens).unwrap();
        cache.capture(&fake, SnapshotKind::Prompt, tokens, &p, After::from_logits(&logits, true)).unwrap();
        fake.mem.borrow_mut().advance(10_000_000);
        cache.tick();
        cache.release(&fake, &p.pages).unwrap();
    }
    assert_eq!(cache.stats().host.unwrap().device_reference_evictions, 1);
    let mut divergent = saved[..20].to_vec();
    divergent.extend(seq(500, 4));
    assert_eq!(cache.peek(&divergent, false), 12);
    let admitted = cache.admit(&fake, &divergent, 24, false,
        |pages| Placement { pages, ring: 1, len: 0, ring_from: 0 }).unwrap();
    assert_eq!(admitted.resume, 12, "aligned common prefix minus replay");
    assert!(admitted.source.unwrap().partial);
    assert_eq!(admitted.placement.ring_from, 12, "mark=None restarts positional state");
    assert_eq!(cache.stats().mark_restore_misses, 0);
    assert_eq!(cache.stats().entries_prompt, 2, "missing mark must not retire valid pages");
    let mut p = admitted.placement;
    fake.forward(&mut p, &divergent).unwrap();
    cache.release(&fake, &p.pages).unwrap();
    assert_eq!(cache.peek(&saved, false), 0, "peek cannot promise a missing exact mark");
    let admitted = cache.admit(&fake, &saved, 24, false,
        |pages| Placement { pages, ring: 1, len: 0, ring_from: 0 }).unwrap();
    assert_eq!(admitted.resume, 0);
    assert_eq!((cache.stats().mark_restore_misses, cache.stats().fork_no_room), (1, 0));
    assert_eq!(cache.stats().entries_prompt, 2);
    cache.release(&fake, &admitted.placement.pages).unwrap();
    cache.clear(&fake).unwrap();
    assert_eq!((cache.pool().free(), cache.arena().in_use()), (64, 0));
}

#[test]
fn failed_host_promotion_compares_host_ancestor_after_pending_device_publication() {
    use cuteafd_hostcache::copy::CopyFault;
    let family = Lagged::new(96, 2, 1);
    let mut cache = lag_cache(&family, 8, 1 << 20);
    for len in [24, 32] {
        let (_, p) = lag_serve(&mut cache, &family, 0, &seq(100, len), len);
        cache.release(&family, &p.pages).unwrap();
        family.fake.mem.borrow_mut().advance(10_000_000);
        cache.tick();
    }
    cache.clear(&family).unwrap();
    let tokens = seq(100, 16);
    let mut writer = cache.admit_cold(&family, 16, 16,
        |pages| Placement { pages, ring: 0, len: 0, ring_from: 0 }).unwrap().placement;
    let logits = family.fake.forward(&mut writer, &tokens).unwrap();
    let Captured::Queued(ticket) = cache.queue_capture(&family, SnapshotKind::Prompt, &tokens, &[],
        &writer, After::from_logits(&logits, true)).unwrap() else { panic!("queued capture") };
    assert_eq!(cache.stats().entries_prompt, 0);
    assert_eq!(cache.arena().in_use(), 1);
    family.fake.mem.borrow_mut().inject(CopyFault::IssueFails(Stream::Restore));
    let admitted = cache.admit(&family, &seq(100, 33), 36, false,
        |pages| Placement { pages, ring: 1, len: 0, ring_from: 0 }).unwrap();
    assert_eq!(admitted.resume, 24, "host ancestor outranks newly published device frontier 16");
    assert!(admitted.source.unwrap().host);
    assert!(cache.poll_capture(&family, ticket).unwrap());
    assert_eq!((cache.stats().mark_restore_misses, cache.stats().fork_no_room), (1, 0));
    let mut p = admitted.placement;
    family.fake.forward(&mut p, &seq(100, 33)).unwrap();
    cache.release(&family, &p.pages).unwrap();
    cache.release(&family, &writer.pages).unwrap();
    cache.clear(&family).unwrap();
    assert_eq!((cache.pool().free(), cache.arena().in_use()), (96, 0));
}

#[test]
fn same_frontier_host_promotion_retires_device_owner_and_deferred_store() {
    use cuteafd_hostcache::config::StoreMode;
    for evict_other_first in [false, true] {
        let fake = Fake::new(64, 2, 8);
        let host = HostConfig { bytes: 1 << 20, chunk_bytes: 1 << 16, min_tokens: 1,
            store: StoreMode::OnEvict, ..HostConfig::default() };
        let mut cache = PrefixCache::new(fake.layout(), config(2, 8), Some((host, Shared(fake.mem.clone())))).unwrap();
        let saved = seq(100, 8);
        let other = seq(900, 8);
        let (_, _, p) = serve(&mut cache, &fake, 0, &saved, &[], 8);
        cache.release(&fake, &p.pages).unwrap();
        cache.clear(&fake).unwrap(); // older host snapshot at the same kind/frontier
        for tokens in [&saved, &other] {
            let mut p = cache.admit_cold(&fake, 8, 8,
                |pages| Placement { pages, ring: 0, len: 0, ring_from: 0 }).unwrap().placement;
            let logits = fake.forward(&mut p, tokens).unwrap();
            cache.capture(&fake, SnapshotKind::Prompt, tokens, &p, After::from_logits(&logits, true)).unwrap();
            cache.release(&fake, &p.pages).unwrap();
        }
        fake.fail_restore_once.set(true);
        let admitted = cache.admit(&fake, &saved, 8, false,
            |pages| Placement { pages, ring: 1, len: 0, ring_from: 0 }).unwrap();
        assert_eq!(admitted.resume, 8);
        assert!(admitted.source.unwrap().host);
        cache.release(&fake, &admitted.placement.pages).unwrap();
        let stats = cache.stats();
        assert_eq!((stats.promotions, stats.entries_prompt, stats.evictions), (1, 2, 2));
        assert_eq!(stats.host.unwrap().stores_completed, 2, "retirement completes and replaces the selected host key");
        assert_eq!(stats.pages_retained, 4, "exactly two owners, with no hidden old fork");
        assert_eq!(64 - cache.pool().free(), 4);
        // Select which owner becomes oldest, then insert a third key into the two-entry bank.
        if !evict_other_first {
            let admitted = cache.admit(&fake, &other, 8, false,
                |pages| Placement { pages, ring: 1, len: 0, ring_from: 0 }).unwrap();
            cache.release(&fake, &admitted.placement.pages).unwrap();
        }
        let (_, _, p) = serve(&mut cache, &fake, 0, &seq(500, 8), &[], 8);
        cache.release(&fake, &p.pages).unwrap();
        let survivor = if evict_other_first { &saved } else { &other };
        assert_eq!(cache.peek(survivor, false), 8);
        let admitted = cache.admit(&fake, survivor, 8, false,
            |pages| Placement { pages, ring: 1, len: 0, ring_from: 0 }).unwrap();
        assert_eq!(admitted.resume, 8);
        assert_eq!(cache.stats().promotions, 1, "the survivor remains device-indexed; no second full promotion");
        cache.release(&fake, &admitted.placement.pages).unwrap();
        assert_eq!(cache.stats().entries_prompt, 2);
        assert_eq!(64 - cache.pool().free(), 4);
        cache.clear(&fake).unwrap();
        assert_eq!((cache.pool().free(), cache.arena().in_use()), (64, 0));
    }
}

#[test]
fn host_replica_failure_retries_only_after_successful_cleanup() {
    for cleanup_fails in [false, true] {
        let fake = Fake::new(64, 2, 8);
        let mut cache = cache(&fake, 8, 1 << 20);
        for len in [8, 12] {
            let (_, _, p) = serve(&mut cache, &fake, 0, &seq(100, len), &[], len);
            cache.release(&fake, &p.pages).unwrap();
            fake.mem.borrow_mut().advance(10_000_000);
            cache.tick();
        }
        cache.clear(&fake).unwrap();
        fake.fail_replicas.set(Some(cleanup_fails));
        let builds = Cell::new(0);
        let admitted = cache.admit(&fake, &seq(100, 15), 16, false, |pages| {
            builds.set(builds.get() + 1);
            Placement { pages, ring: 1, len: 0, ring_from: 0 }
        });
        if cleanup_fails {
            assert!(admitted.is_err());
            assert_eq!(builds.get(), 0);
            assert_eq!((cache.stats().quarantined, cache.arena().in_use()), (1, 1));
            assert_eq!(cache.pool().free(), 61, "failed promotion's three pages remain owned");
            fake.fail_drain.set(true);
            for _ in 0..2 {
                assert!(cache.clear(&fake).is_err());
                assert_eq!((cache.pool().free(), cache.arena().in_use()), (61, 1));
            }
            fake.fail_drain.set(false);
        } else {
            let mut admitted = admitted.unwrap();
            assert_eq!(admitted.resume, 8);
            assert_eq!(builds.get(), 1);
            fake.forward(&mut admitted.placement, &seq(100, 15)).unwrap();
            cache.release(&fake, &admitted.placement.pages).unwrap();
            assert_eq!(cache.stats().quarantined, 0);
        }
        assert_eq!((cache.stats().mark_restore_misses, cache.stats().restore_failures,
            cache.stats().fork_no_room), (1, 1, 0));
        cache.clear(&fake).unwrap();
        assert_eq!((cache.pool().free(), cache.arena().in_use()), (64, 0));
    }
}
