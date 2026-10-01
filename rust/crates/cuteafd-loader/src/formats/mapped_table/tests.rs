use super::*;
use std::io::{Seek, SeekFrom, Write};

type TestResult = std::result::Result<(), Box<dyn std::error::Error>>;

fn u8_rows(width: usize) -> RowFormat {
    RowFormat { dtype: DType::U8, width, row_bytes: width }
}

#[test]
fn unaligned_payload_gather_preserves_order_and_rejects_bad_batches() -> TestResult {
    let mut shard = tempfile::NamedTempFile::new()?;
    shard.write_all(&[99; 13])?;
    shard.write_all(&[1, 2, 3, 4, 5, 6, 7, 8])?;
    let table = unsafe { MappedRows::open(shard.path(), 13, 4, 2)? };
    assert_eq!(table.prefetch(&[3, 1, 3], 1)?, 1);
    let mut out = [0; 6];
    table.gather_into(&[3, 1, 3], &mut out)?;
    assert_eq!(out, [7, 8, 3, 4, 7, 8]);
    assert!(table.gather_into(&[0, 4, 1], &mut out).is_err());
    assert_eq!(out, [7, 8, 3, 4, 7, 8]);
    assert!(table.prefetch(&[0], 0).is_err());
    assert_eq!(table.prefetch(&[], 0)?, 0);
    assert!(unsafe { MappedRows::open(shard.path(), 14, 4, 2) }.is_err());
    Ok(())
}

#[test]
fn sparse_checkpoint_rows_past_two_gib_use_wide_offsets() -> TestResult {
    let mut shard = tempfile::NamedTempFile::new()?;
    let row_bytes = 256;
    let high = (1_u64 << 31) / row_bytes as u64 + 7;
    shard.as_file().set_len((high + 1) * row_bytes as u64)?;
    shard.seek(SeekFrom::Start(high * row_bytes as u64))?;
    shard.write_all(&[0x5a; 256])?;
    let table = unsafe { MappedTable::single(shard.path(), 0, high + 1, u8_rows(row_bytes))? };
    assert_eq!(table.prefetch(&[high, high], 1)?, 1);
    let mut out = [0; 256];
    table.gather_into(&[high], &mut out)?;
    assert_eq!(out, [0x5a; 256]);
    assert!(table.prefetch(&[0, high], 1).is_err());
    Ok(())
}

/// Three parts (two shards, one with a header gap) of 4-byte rows; row r holds [r; 4].
fn parted(rows: [u64; 3]) -> std::result::Result<(tempfile::TempDir, Vec<TablePart>), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let mut parts = Vec::new();
    let mut next = 0u64;
    let mut a = std::fs::File::create(dir.path().join("a.safetensors"))?;
    let mut b = std::fs::File::create(dir.path().join("b.safetensors"))?;
    b.write_all(&[0xee; 5])?;
    let mut b_offset = 5u64;
    for (index, count) in rows.into_iter().enumerate() {
        let bytes: Vec<u8> = (next..next + count).flat_map(|r| [r as u8; 4]).collect();
        if index == 0 {
            a.write_all(&bytes)?;
            parts.push(TablePart { path: dir.path().join("a.safetensors"), offset: 0, rows: count });
        } else {
            b.write_all(&bytes)?;
            parts.push(TablePart { path: dir.path().join("b.safetensors"), offset: b_offset, rows: count });
            b_offset += bytes.len() as u64;
        }
        next += count;
    }
    Ok((dir, parts))
}

#[test]
fn parts_form_one_row_space_for_uniform_and_ragged_splits() -> TestResult {
    for split in [[3, 3, 2], [2, 5, 1]] {
        let (_dir, parts) = parted(split)?;
        let table = unsafe { MappedTable::open(&parts, u8_rows(4))? };
        assert_eq!(table.rows(), 8);
        assert_eq!(table.part_count(), 3);
        let rows = [7, 0, 3, 3, 5, 2];
        let mut out = vec![0; rows.len() * 4];
        table.gather_into(&rows, &mut out)?;
        let expected: Vec<u8> = rows.iter().flat_map(|&r| [r as u8; 4]).collect();
        assert_eq!(out, expected);
        assert!(matches!(table.gather_into(&[8], &mut [0; 4]), Err(MappedTableError::RowRange { row: 8, rows: 8 })));
        // Pages are per part: rows in three parts are at least three pages.
        assert!(table.prefetch(&[0, 3, 7], 16)? >= 2);
        assert!(table.prefetch(&[0, 7], 1).is_err());
        let pool = GatherPool::new("test-gather", 3)?;
        let mut pooled = vec![0; out.len()];
        let report = pool.gather(&table, &rows, &mut pooled, None)?;
        assert_eq!(pooled, expected);
        assert_eq!(report.rows, rows.len());
        assert!(pool.gather(&table, &[1, 9], &mut [0; 8], None).is_err());
        let stats = table.stats().snapshot();
        assert_eq!((stats.gathers, stats.rows, stats.bytes), (1, 6, 24));
    }
    Ok(())
}

#[test]
fn checkpoint_tensors_validate_dtype_width_payload_and_paths() -> TestResult {
    let (dir, parts) = parted([3, 3, 2])?;
    let meta = |name: &str, part: &TablePart, dtype: DType, width: usize| SafetensorsTensorMetadata {
        name: name.into(),
        dtype,
        shape: vec![part.rows as usize, width],
        byte_offset: part.offset,
        byte_length: part.rows * 4,
    };
    let shard = |part: &TablePart| part.path.file_name().unwrap().to_str().unwrap().to_owned();
    let metas: Vec<_> = parts.iter().enumerate().map(|(i, p)| meta(&format!("t.shard_{i}"), p, DType::U8, 4)).collect();
    let shards: Vec<_> = parts.iter().map(shard).collect();
    let tensors: Vec<_> = shards.iter().map(String::as_str).zip(&metas).collect();
    let table = unsafe { MappedTable::from_tensors(dir.path(), &tensors)? };
    assert_eq!((table.rows(), table.row_bytes(), table.format().width), (8, 4, 4));
    let mut out = [0; 4];
    table.gather_into(&[6], &mut out)?;
    assert_eq!(out, [6; 4]);
    // BF16 rows of two elements are also 4 bytes, but a part may not change dtype.
    let mixed = meta("t.shard_1", &parts[1], DType::Bf16, 2);
    let tensors_mixed = vec![(shards[0].as_str(), &metas[0]), (shards[1].as_str(), &mixed)];
    assert!(unsafe { MappedTable::from_tensors(dir.path(), &tensors_mixed) }.is_err());
    let mut short = metas[0].clone();
    short.byte_length -= 1;
    assert!(unsafe { MappedTable::from_tensors(dir.path(), &[(shards[0].as_str(), &short)]) }.is_err());
    assert!(unsafe { MappedTable::from_tensors(dir.path(), &[("../a.safetensors", &metas[0])]) }.is_err());
    assert!(unsafe { MappedTable::from_tensors(dir.path(), &[]) }.is_err());
    Ok(())
}

#[test]
fn hot_row_cache_serves_hits_and_evicts_least_recent() -> TestResult {
    let (_dir, parts) = parted([3, 3, 2])?;
    let table = unsafe { MappedTable::open(&parts, u8_rows(4))? };
    let cache = HotRowCache::new(4, 8).expect("two rows");
    assert!(HotRowCache::new(4, 3).is_none());
    let pool = GatherPool::new("test-cache", 2)?;
    let gather = |rows: &[u64]| -> std::result::Result<(Vec<u8>, GatherReport), MappedTableError> {
        let mut out = vec![0; rows.len() * 4];
        let report = pool.gather(&table, rows, &mut out, Some(&cache))?;
        Ok((out, report))
    };
    let (out, report) = gather(&[1, 2])?;
    assert_eq!((out, report.cache_hits), (vec![1, 1, 1, 1, 2, 2, 2, 2], 0));
    let (out, report) = gather(&[2, 1, 2])?;
    assert_eq!((out, report.cache_hits), (vec![2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2], 3));
    // Row 1 is least recent after [2, 1, 2]; inserting 5 evicts it.
    gather(&[5])?;
    assert_eq!(cache.len(), 2);
    let (out, report) = gather(&[1, 5])?;
    assert_eq!((out, report.cache_hits), (vec![1, 1, 1, 1, 5, 5, 5, 5], 1));
    let stats = table.stats().snapshot();
    assert_eq!((stats.cache_hits, stats.cache_misses), (4, 4));
    assert!(pool.gather(&table, &[0], &mut [0; 4], HotRowCache::new(8, 64).as_ref()).is_err());
    Ok(())
}

#[test]
fn prefetcher_bounds_rows_reports_pages_and_detaches() -> TestResult {
    let (_dir, parts) = parted([3, 3, 2])?;
    let table: Arc<MappedTable> = Arc::new(unsafe { MappedTable::open(&parts, u8_rows(4))? });
    let worker = TablePrefetcher::new("test-prefetch", 1, 4, 8)?;
    assert!(worker.try_submit(table.clone(), &[8]).is_err());
    assert!(worker.try_submit(table.clone(), &[0; 5]).is_err());
    let ticket = worker.try_submit(table.clone(), &[1, 0, 1])?.expect("queue has room");
    assert_eq!(ticket.wait()?, TablePrefetchOutcome::Advised(vec![1]));
    let cancelled = worker.try_submit(table.clone(), &[7])?.expect("queue has room");
    cancelled.cancel();
    assert!(matches!(cancelled.wait()?, TablePrefetchOutcome::Cancelled | TablePrefetchOutcome::Advised(_)));
    // Detached advice truncates to the row cap instead of failing.
    while !worker.submit_detached(table.clone(), &[0, 1, 2, 3, 4, 5])? {}
    drop(worker);
    let stats = table.stats().snapshot();
    assert!(stats.prefetch_jobs >= 2 && stats.prefetch_pages >= 2);
    Ok(())
}

#[test]
fn gather_worker_recycles_slots_and_cancels() -> TestResult {
    let (_dir, parts) = parted([3, 3, 2])?;
    let table = Arc::new(unsafe { MappedTable::open(&parts, u8_rows(4))? });
    let reader = Arc::clone(&table);
    let worker: GatherWorker<Vec<u8>, Vec<u64>, MappedTableError> =
        GatherWorker::new("test-worker", vec![vec![0; 8]], move |slot: &mut Vec<u8>, rows: &Vec<u64>| reader.gather_into(rows, slot))?;
    let wait = |ticket: &mut GatherTicket<Vec<u8>, Vec<u64>, MappedTableError>| loop {
        match ticket.poll() {
            Ok(GatherPoll::Pending) => std::thread::yield_now(),
            other => return other,
        }
    };
    let mut ticket = worker.try_submit(vec![7, 2], true)?.expect("one free slot");
    // The only slot is leased: backpressure, not blocking.
    assert!(worker.try_submit(vec![0, 0], false)?.is_none());
    let Ok(GatherPoll::Ready(lease)) = wait(&mut ticket) else { panic!("gather failed") };
    assert_eq!(lease.slot().unwrap(), &vec![7, 7, 7, 7, 2, 2, 2, 2]);
    assert_eq!(lease.job(), &vec![7, 2]);
    assert!(lease.timing().is_some());
    assert!(ticket.poll().is_err(), "a completion is consumed once");
    drop(lease);
    // The recycled slot is reused; a job error comes back as a job failure.
    let mut failing = loop {
        if let Some(ticket) = worker.try_submit(vec![9, 0], false)? {
            break ticket;
        }
    };
    assert!(matches!(wait(&mut failing), Err(GatherFailure::Job(MappedTableError::RowRange { .. }))));
    let mut cancelled = loop {
        if let Some(ticket) = worker.try_submit(vec![1, 1], false)? {
            break ticket;
        }
    };
    cancelled.cancel();
    assert!(matches!(cancelled.poll(), Ok(GatherPoll::Cancelled)));
    Ok(())
}

#[test]
fn spawned_gathers_complete_into_caller_storage() -> TestResult {
    let (_dir, parts) = parted([3, 3, 2])?;
    let table = Arc::new(unsafe { MappedTable::open(&parts, u8_rows(4))? });
    let pool = Arc::new(GatherPool::new("test-spawn", 2)?);
    let mut out = vec![0u8; 12];
    let pending = unsafe { pool.spawn_gather(table.clone(), vec![6, 0, 4], out.as_mut_ptr(), out.len(), None) };
    assert_eq!(pending.wait()?.rows, 3);
    assert_eq!(out, [6, 6, 6, 6, 0, 0, 0, 0, 4, 4, 4, 4]);
    // An invalid row fails without writing; dropping an unwaited gather waits for it.
    let failing = unsafe { pool.spawn_gather(table.clone(), vec![1, 8, 1], out.as_mut_ptr(), out.len(), None) };
    assert!(failing.wait().is_err());
    assert_eq!(out, [6, 6, 6, 6, 0, 0, 0, 0, 4, 4, 4, 4]);
    let cache = Arc::new(HotRowCache::new(4, 64).expect("rows"));
    drop(unsafe { pool.spawn_gather(table.clone(), vec![2, 2, 2], out.as_mut_ptr(), out.len(), Some(cache.clone())) });
    assert_eq!(out, [2; 12]);
    assert_eq!(cache.len(), 1);
    Ok(())
}

#[test]
fn warm_reads_every_part_and_stops_on_request() -> TestResult {
    let (_dir, parts) = parted([3, 3, 2])?;
    let table = unsafe { MappedTable::open(&parts, u8_rows(4))? };
    // Each part's mapping starts at its page; part 2 shares b's page with part 1.
    let total = table.warm(&AtomicBool::new(false), None)?;
    assert!(total >= 8 * 4);
    assert_eq!(table.warm(&AtomicBool::new(true), None)?, 0);
    // Pacing: the whole table at twice its size per second takes about half a second.
    let started = Instant::now();
    assert_eq!(table.warm(&AtomicBool::new(false), Some(2 * total))?, total);
    assert!(started.elapsed() >= Duration::from_millis(400));
    let mut out = [0; 4];
    table.gather_into(&[7], &mut out)?;
    assert_eq!(out, [7; 4]);
    Ok(())
}
