// Failure interleavings use the same queued-copy family and accounting as normal prefix tests.
#[test]
fn restore_drain_failure_quarantines_fork_and_temporary_mark_without_retry() {
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
        assert_eq!(cache.stats().fork_no_room, 0);
        fake.fail_restore.set(false);
        let mut p = cache.admit(&fake, &query, 16, false,
            |pages| Placement { pages, ring: 1, len: 0, ring_from: 0 }).unwrap().placement;
        assert_eq!(p.len, 12);
        fake.forward(&mut p, &query).unwrap();
        cache.release(&fake, &p.pages).unwrap();
        cache.clear(&fake).unwrap();
        assert_eq!((cache.pool().free(), cache.arena().in_use()), (64, 0));
    }
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
