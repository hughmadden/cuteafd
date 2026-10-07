use crate::media::{ImageKey, MediaKeys, MediaSpan};

fn media_span(key: ImageKey) -> MediaSpan {
    MediaSpan {
        start: 2,
        len: 4,
        key: key.into(),
    }
}
fn media_keys(key: ImageKey, n: usize) -> MediaKeys {
    MediaKeys::new(&seq(10, n), 1000, &[media_span(key)]).unwrap()
}
fn media_place(pages: Vec<u32>, ring: usize) -> Placement {
    Placement {
        pages,
        ring,
        len: 0,
        ring_from: 0,
    }
}

// Force identical row hints by cancelling two words in the 64-bit fold. Verification must
// still reject these different SHA-256 identities; this does not depend on a birthday search.
fn colliding_keys() -> (ImageKey, ImageKey) {
    let a = ImageKey([0; 32]);
    let mut b = a;
    b.0[..8].copy_from_slice(&1u64.to_le_bytes());
    b.0[8..16].copy_from_slice(&1u64.rotate_right(13).to_le_bytes());
    (a, b)
}
fn media_capture(
    cache: &mut PrefixCache<Shared>,
    fake: &Fake,
    keys: &MediaKeys,
    state_tokens: &[u32],
    ring: usize,
) -> Placement {
    let mut p = cache
        .admit_media(
            fake,
            keys.tokens(),
            keys.spans(),
            state_tokens.len(),
            false,
            |pages| media_place(pages, ring),
        )
        .unwrap()
        .placement;
    let logits = fake.forward(&mut p, state_tokens).unwrap();
    assert!(cache
        .capture_media(
            fake,
            SnapshotKind::Prompt,
            keys.tokens(),
            keys.spans(),
            &p,
            After::from_logits(&logits, true)
        )
        .unwrap());
    p
}

#[test]
fn media_repeat_next_turn_and_changed_image_restore_exactly() {
    let fake = Fake::new(64, 4, 8);
    let mut cache = cache(&fake, 4, 0);
    let a = media_keys(ImageKey([1; 32]), 8);
    let p = media_capture(&mut cache, &fake, &a, a.tokens(), 0);
    cache.release(&fake, &p.pages).unwrap();
    let before = cache.stats();
    let drains = fake.drains.get();
    assert_eq!(cache.peek_media(a.tokens(), a.spans(), false), 8);
    assert_eq!(cache.stats(), before);
    assert_eq!(fake.drains.get(), drains);
    let r = cache
        .admit_media(&fake, a.tokens(), a.spans(), 16, false, |pages| {
            media_place(pages, 1)
        })
        .unwrap();
    assert_eq!(r.resume, 8);
    assert!(r.after.is_some());
    let mut p = r.placement;
    let mut native = seq(10, 12);
    // The second image is new work; the original image rows keep their keyed identity.
    let spans = [
        media_span(ImageKey([1; 32])),
        MediaSpan {
            start: 9,
            len: 2,
            key: ImageKey([2; 32]).into(),
        },
    ];
    native[..8].copy_from_slice(&seq(10, 8));
    let b = MediaKeys::new(&native, 1000, &spans).unwrap();
    fake.forward(&mut p, b.tokens()).unwrap();
    cache
        .capture_media(
            &fake,
            SnapshotKind::Turn,
            b.tokens(),
            b.spans(),
            &p,
            After::from_logits(&[1.0, 2.0], true),
        )
        .unwrap();
    cache.release(&fake, &p.pages).unwrap();
    assert_eq!(cache.peek_media(b.tokens(), b.spans(), false), 12);
    // The exact-family cache has no mark at the image start, so a changed image misses.
    let c = media_keys(ImageKey([3; 32]), 8);
    assert_eq!(cache.peek_media(c.tokens(), c.spans(), false), 0);
    assert_eq!(cache.stats().media_key_collisions, 0);
    let cold = cache
        .admit_cold(&fake, 8, 8, |pages| media_place(pages, 2))
        .unwrap();
    assert_eq!(cold.resume, 0);
    cache.release(&fake, &cold.placement.pages).unwrap();
    cache.clear(&fake).unwrap();
    assert_eq!(cache.pool().free(), 64);
}

#[test]
fn forced_row_key_collision_is_a_miss_on_device_and_host() {
    let fake = Fake::new(64, 3, 8);
    let mut cache = cache(&fake, 4, 1 << 20);
    let (a_key, b_key) = colliding_keys();
    let (a, b) = (media_keys(a_key, 8), media_keys(b_key, 8));
    assert_ne!(a_key, b_key);
    assert_eq!(a.tokens(), b.tokens());
    let p = media_capture(&mut cache, &fake, &a, a.tokens(), 0);
    cache.release(&fake, &p.pages).unwrap();
    assert_eq!(cache.peek_media(b.tokens(), b.spans(), false), 0);
    let r = cache
        .admit_media(&fake, b.tokens(), b.spans(), 8, false, |pages| {
            media_place(pages, 1)
        })
        .unwrap();
    assert_eq!(r.resume, 0);
    assert!(cache.stats().media_key_collisions > 0);
    cache.release(&fake, &r.placement.pages).unwrap();
    cache.clear(&fake).unwrap();
    let before = cache.stats().media_key_collisions;
    let stats_before_peek = cache.stats();
    assert_eq!(cache.peek_media(a.tokens(), a.spans(), false), 8);
    assert_eq!(cache.peek_media(b.tokens(), b.spans(), false), 0);
    assert_eq!(cache.stats(), stats_before_peek);
    let r = cache
        .admit_media(&fake, b.tokens(), b.spans(), 8, false, |pages| {
            media_place(pages, 1)
        })
        .unwrap();
    assert_eq!(r.resume, 0);
    assert!(cache.stats().media_key_collisions > before);
    cache.release(&fake, &r.placement.pages).unwrap();
    let r = cache
        .admit_media(&fake, a.tokens(), a.spans(), 8, false, |pages| {
            media_place(pages, 2)
        })
        .unwrap();
    assert_eq!(r.resume, 8);
    assert!(r.source.unwrap().host);
    let host_stats = cache.stats().host.unwrap();
    assert!(host_stats.lookups > 0);
    assert_eq!(host_stats.host_hits, 1);
    let mut p = r.placement;
    fake.forward(&mut p, a.tokens()).unwrap();
    cache.release(&fake, &p.pages).unwrap();
    cache.clear(&fake).unwrap();
}

#[test]
fn media_host_dedup_uses_full_keys_not_colliding_row_hints() {
    let fake = Fake::new(64, 4, 8);
    let mut cache = cache(&fake, 4, 1 << 20);
    let (a_key, b_key) = colliding_keys();
    let a = media_keys(a_key, 8);
    let mut native_b = seq(10, 9);
    native_b[8] = 77;
    let b = MediaKeys::new(&native_b, 1000, &[media_span(b_key)]).unwrap();
    assert_eq!(a.tokens(), &b.tokens()[..8]);
    // Different feature bytes affect every downstream state row, even though radix hints
    // collide. The fake family sees state tokens; the prefix tier sees the keyed copy.
    let mut state_b = b.tokens().to_vec();
    state_b[2] ^= 0x1234;
    let p = media_capture(&mut cache, &fake, &a, a.tokens(), 0);
    cache.release(&fake, &p.pages).unwrap();
    let p = media_capture(&mut cache, &fake, &b, &state_b, 1);
    cache.release(&fake, &p.pages).unwrap();
    cache.clear(&fake).unwrap();
    assert_eq!(cache.peek_media(b.tokens(), b.spans(), false), 9);
    let r = cache
        .admit_media(&fake, b.tokens(), b.spans(), 9, false, |pages| {
            media_place(pages, 2)
        })
        .unwrap();
    assert_eq!(r.resume, 9);
    assert!(r.source.unwrap().host);
    let mut p = r.placement;
    fake.forward(&mut p, &state_b)
        .expect("full-key page identity prevents corrupt dedup");
    cache.release(&fake, &p.pages).unwrap();
    cache.clear(&fake).unwrap();
}

#[test]
fn media_point_schedule_rounds_boundaries_and_periodic_points_before_capture() {
    let spans = [MediaSpan {
        start: 3,
        len: 10,
        key: ImageKey([1; 32]).into(),
    }];
    let policy = PointPolicy {
        gap: 4,
        boundaries: 4,
        per_request: 8,
    };
    for reach in [0, 2, 8] {
        let p = plan_media_points(0, 20, 4, &[7, 12, 15], reach, 1, policy, &spans);
        for &(chunk, point) in &p.points {
            assert_eq!(crate::media::round_frontier(point, &spans), point);
            assert!(p.chunks[chunk] - point <= reach);
        }
        assert!(p.points.iter().any(|&(_, point)| point == 3));
        let mut start = 0;
        for &end in &p.chunks {
            assert!(end - start <= 4);
            start = end;
        }
    }
    assert_eq!(
        plan_media_points(0, 20, 4, &[7], 2, 1, policy, &[]),
        plan_points(0, 20, 4, &[7], 2, 1, policy)
    );
}

#[test]
fn image_frontiers_round_down_for_capture_park_and_partial_resume() {
    let mut fake = Fake::new(64, 3, 8);
    fake.rule = ReuseRule {
        align: 1,
        replay: None,
    };
    let mut cache = cache(&fake, 8, 0);
    let a = media_keys(ImageKey([1; 32]), 8);
    let mut p = cache
        .admit_cold(&fake, 8, 8, |pages| media_place(pages, 0))
        .unwrap()
        .placement;
    fake.forward(&mut p, &a.tokens()[..4]).unwrap();
    // Parked/captured point is inside [2,6); snapshot rounds to 2 and drops After.
    assert!(cache
        .capture_media(
            &fake,
            SnapshotKind::Prompt,
            &a.tokens()[..4],
            a.spans(),
            &p,
            After::from_logits(&[1.0], true)
        )
        .unwrap());
    assert_eq!(cache.peek_media(a.tokens(), a.spans(), false), 2);
    assert!(cache
        .park_media(&fake, &a.tokens()[..4], a.spans(), &p)
        .unwrap());
    fake.forward(&mut p, a.tokens()).unwrap();
    cache
        .capture_media(
            &fake,
            SnapshotKind::Turn,
            a.tokens(),
            a.spans(),
            &p,
            After::from_logits(&[1.0], true),
        )
        .unwrap();
    let mut query = a.tokens().to_vec();
    query[5] ^= 1;
    assert_eq!(cache.peek_media(&query, a.spans(), false), 2);
    let changed_image = media_keys(ImageKey([8; 32]), 8);
    assert_eq!(
        cache.peek_media(changed_image.tokens(), changed_image.spans(), false),
        2
    );
    let r = cache
        .admit_media(&fake, &query, a.spans(), 8, false, |pages| {
            media_place(pages, 1)
        })
        .unwrap();
    assert_eq!(r.resume, 2);
    fake.forward(&mut r.placement.clone(), &query).unwrap();
    cache.release(&fake, &r.placement.pages).unwrap();
    cache.release(&fake, &p.pages).unwrap();
    cache.clear(&fake).unwrap();
    assert_eq!(cache.pool().free(), 64);
}

#[test]
fn embedding_eviction_reencode_and_prefix_restore_remain_byte_exact() {
    use crate::media::{
        EmbeddingCache, EncodeJob, FakeEncoder, MediaAdmission, MediaChunk, MediaPoll, MediaWaiter,
        RequestMedia,
    };
    use std::sync::Arc;
    let fake = Fake::new(64, 3, 8);
    let mut prefix = cache(&fake, 4, 0);
    let key = ImageKey([3; 32]);
    let keys = media_keys(key, 8);
    let mut queue = MediaAdmission::new(EmbeddingCache::new(8), FakeEncoder::default(), 4);
    let make_waiter = |resume| {
        MediaWaiter::new(
            1,
            RequestMedia::new(keys.spans().to_vec(), 1, 8).unwrap(),
            vec![EncodeJob::image(key, [1, 2, 2], Arc::from([3; 12]), 4, 1)],
            resume,
        )
        .unwrap()
    };
    let free = prefix.pool().free();
    queue.enqueue(make_waiter(0)).unwrap();
    let MediaPoll::Ready(first) = queue.poll(|_| false) else {
        panic!("fake ready");
    };
    assert_eq!(prefix.pool().free(), free); // encoding did not reserve any KV or mark
    let mut chunk = MediaChunk::default();
    first.media().write_chunk(0, 8, &mut chunk).unwrap();
    let original_bytes = chunk.features.clone();
    let mut state_tokens = keys.tokens().to_vec();
    for (row, bytes) in chunk.indices.iter().zip(chunk.features.chunks_exact(2)) {
        state_tokens[*row as usize] = u16::from_le_bytes(bytes.try_into().unwrap()) as u32;
    }
    let p = media_capture(&mut prefix, &fake, &keys, &state_tokens, 0);
    prefix.release(&fake, &p.pages).unwrap();
    drop(first);
    // Evict the embedding, independently of its retained prefix snapshot.
    let other = ImageKey([9; 32]);
    drop(queue.cache.reserve(other, 8).unwrap());
    drop(queue.cache.complete(other, Arc::from([9; 8])).unwrap());
    assert!(!queue.cache.contains(key));
    queue
        .enqueue(make_waiter(prefix.peek_media(
            keys.tokens(),
            keys.spans(),
            false,
        )))
        .unwrap();
    let MediaPoll::Ready(repeat) = queue.poll(|_| false) else {
        panic!("prefix skip ready");
    };
    assert_eq!(queue.encoder().submitted, 1);
    assert!(!repeat.media().has_features(key));
    let r = prefix
        .admit_media(&fake, keys.tokens(), keys.spans(), 8, false, |pages| {
            media_place(pages, 1)
        })
        .unwrap();
    assert_eq!(r.resume, 8);
    fake.forward(&mut r.placement.clone(), &state_tokens)
        .unwrap();
    prefix.release(&fake, &r.placement.pages).unwrap();
    // Lose the prefix between peek and admission. Release any attempted placement before
    // reconcile sends the host-only request back to media_pending.
    prefix.clear(&fake).unwrap();
    let retry = repeat.reconcile(0).unwrap_err();
    queue.enqueue(retry).unwrap();
    let MediaPoll::Ready(reencoded) = queue.poll(|_| false) else {
        panic!("reencode ready");
    };
    reencoded.media().write_chunk(0, 8, &mut chunk).unwrap();
    assert_eq!(chunk.features, original_bytes);
    assert_eq!(queue.encoder().submitted, 2);
    let p = media_capture(&mut prefix, &fake, &keys, &state_tokens, 2);
    prefix.release(&fake, &p.pages).unwrap();
    prefix.clear(&fake).unwrap();
    assert_eq!(prefix.pool().free(), 64);
}

#[test]
fn peek_is_read_only_on_text_and_does_not_refresh_device_lru() {
    let fake = Fake::new(64, 4, 8);
    let mut cache = cache(&fake, 2, 0);
    for (i, base) in [10, 20].into_iter().enumerate() {
        let (_, _, p) = serve(&mut cache, &fake, i, &seq(base, 8), &[], 8);
        cache.release(&fake, &p.pages).unwrap();
    }
    let stats = cache.stats();
    assert_eq!(cache.peek(&seq(10, 8), false), 8);
    assert_eq!(cache.stats(), stats);
    let (_, _, p) = serve(&mut cache, &fake, 2, &seq(30, 8), &[], 8);
    cache.release(&fake, &p.pages).unwrap();
    assert_eq!(cache.peek(&seq(10, 8), false), 0);
    assert_eq!(cache.peek(&seq(20, 8), false), 8);
    cache.clear(&fake).unwrap();
}
