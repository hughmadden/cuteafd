//! Qwen 3.8 PLE n-gram rows through the mapped table (gather pool, with and
//! without the hot-row cache) against the bytes the host preload copies
//! (`pread` of each row from its shard): byte identity, plus timings.
//! Usage: SNAPSHOT [TOKENS].
use anyhow::{ensure, Context, Result};
use cuteafd_loader::families::qwen4::{NgramHasher, Qwen4Config};
use cuteafd_loader::plan::checkpoint::Checkpoint;
use cuteafd_loader::{GatherPool, HotRowCache, MappedTable};
use std::os::unix::fs::FileExt;
use std::path::Path;

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let snapshot = Path::new(args.get(1).context("missing snapshot")?);
    let count: usize = args.get(2).map_or(Ok(8192), |n| n.parse())?;
    let opened = std::time::Instant::now();
    let checkpoint = Checkpoint::open(snapshot)?;
    let cfg = Qwen4Config::read(snapshot)?;
    let layer = *cfg.ple_layers.first().context("no PLE layer")?;
    let prefix = format!("model.language_model.layers.{layer}.ple.ple_embedding.ngram_embedding.shard_");
    let mut shards: Vec<_> = checkpoint.tensors.iter().filter(|t| t.meta.name.starts_with(&prefix)
        && t.meta.name.ends_with(".weight")).collect();
    shards.sort_by_key(|t| t.meta.name[prefix.len()..].trim_end_matches(".weight").parse::<usize>().unwrap_or(usize::MAX));
    ensure!(!shards.is_empty(), "no PLE shards under {prefix}");
    let tensors: Vec<_> = shards.iter().map(|t| (t.shard.as_str(), &t.meta)).collect();
    // SAFETY: the checkpoint snapshot is immutable while this process runs.
    let table = unsafe { MappedTable::from_tensors(snapshot, &tensors)? };
    println!("mapped {} parts, {} rows x {} B ({:.1} GiB) in {:.3}s", table.part_count(), table.rows(),
        table.row_bytes(), table.bytes() as f64 / (1u64 << 30) as f64, opened.elapsed().as_secs_f64());
    let hasher = NgramHasher::from_config(cfg.vocab_size, 20_000_000, cfg.ngram_size, cfg.heads_per_ngram, 0, 1234,
        cfg.eos);
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let tokens: Vec<u32> = (0..count).map(|_| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state % cfg.vocab_size as u64) as u32
    }).collect();
    let mut ids = Vec::new();
    hasher.hash(&mut hasher.start(), &tokens, &mut ids)?;
    let rows: Vec<u64> = ids.iter().map(|&id| id as u64).collect();
    let row_bytes = table.row_bytes();
    // The preload's bytes: each row read from its shard at its offset.
    let mut starts = Vec::new();
    let mut at = 0u64;
    for shard in &shards {
        starts.push(at);
        at += shard.meta.shape[0] as u64;
    }
    let files: Vec<_> = shards.iter().map(|s| std::fs::File::open(snapshot.join(&s.shard))).collect::<Result<_, _>>()?;
    let mut expected = vec![0u8; rows.len() * row_bytes];
    for (i, &row) in rows.iter().enumerate() {
        let part = starts.partition_point(|&s| s <= row) - 1;
        files[part].read_exact_at(&mut expected[i * row_bytes..(i + 1) * row_bytes],
            shards[part].meta.byte_offset + (row - starts[part]) * row_bytes as u64)?;
    }
    let pool = GatherPool::new("ple-check", 16)?;
    let cache = HotRowCache::new(row_bytes, 64 << 20);
    for (label, cache) in [("pool", None), ("pool+cache", cache.as_ref()), ("pool+cache warm", cache.as_ref())] {
        let mut out = vec![0u8; expected.len()];
        let report = pool.gather(&table, &rows, &mut out, cache)?;
        ensure!(out == expected, "{label}: gathered rows differ from the shard bytes");
        println!("{label}: {} rows identical, {:.3} ms, {} major faults, {} cache hits", report.rows,
            report.elapsed.as_secs_f64() * 1e3, report.major_faults, report.cache_hits);
    }
    let mut out = vec![0u8; expected.len()];
    table.gather_into(&rows, &mut out)?;
    ensure!(out == expected, "gather_into rows differ from the shard bytes");
    println!("PASS {} PLE rows of {count} tokens byte-identical (pool, cache, gather_into)", rows.len());
    Ok(())
}
