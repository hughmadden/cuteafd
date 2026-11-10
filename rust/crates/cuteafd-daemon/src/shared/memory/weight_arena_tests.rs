use super::*;
use cuteafd_loader::push_load::{
    builder::{Builder, Destination},
    plan::{Entry, Pattern, RunWalk},
};
use std::collections::{BTreeMap, VecDeque};

struct FileWalk<'a> {
    walks: Vec<RunWalk<'a>>,
    heads: std::collections::BinaryHeap<std::cmp::Reverse<(u64, u64, u32, usize)>>,
}
impl<'a> FileWalk<'a> {
    fn new(entries: Vec<&'a Entry>, patterns: &'a [Pattern]) -> Self {
        let mut walks = entries
            .iter()
            .map(|entry| RunWalk::new(&patterns[entry.pattern as usize], entry, 0, u64::MAX))
            .collect::<Vec<_>>();
        let mut heads = std::collections::BinaryHeap::new();
        for (index, walk) in walks.iter_mut().enumerate() {
            if let Some((source, destination, bytes)) = walk.next() {
                heads.push(std::cmp::Reverse((source, destination, bytes, index)));
            }
        }
        Self { walks, heads }
    }
    // Merge all entry cursors by file offset, including replicated rotations.
    // Adjacent destination runs combine before touching the coverage tree.
    fn batch(&mut self, expected: &ExpectedCoverage, coverage: &mut Coverage) -> Result<bool> {
        let mut remaining = 4 * 1024 * 1024u64;
        let mut pending: Option<(u64, u64)> = None;
        while remaining > 0 {
            if let Some(std::cmp::Reverse((_, destination, bytes, index))) = self.heads.pop() {
                if let Some((source, next, bytes)) = self.walks[index].next() {
                    self.heads
                        .push(std::cmp::Reverse((source, next, bytes, index)));
                }
                let bytes = u64::from(bytes);
                match &mut pending {
                    Some((start, len)) if *start + *len == destination => *len += bytes,
                    _ => {
                        if let Some((start, len)) = pending.take() {
                            expected.record(coverage, start, len)?;
                        }
                        pending = Some((destination, bytes));
                    }
                }
                remaining = remaining.saturating_sub(bytes);
            } else {
                break;
            }
        }
        if let Some((start, len)) = pending {
            expected.record(coverage, start, len)?;
        }
        Ok(!self.heads.is_empty())
    }
}

#[test]
fn real_glm_full_plan_interleaved_sources_and_file_replay() -> Result<()> {
    full_exl3_plan("CUTEAFD_PUSH_EXL3")
}

fn full_exl3_plan(variable: &str) -> Result<()> {
    let Some(snapshot) = std::env::var_os(variable) else {
        return Ok(());
    };
    let snapshot = std::path::PathBuf::from(snapshot);
    let catalog = cuteafd_loader::read_expert_catalog(&snapshot)?;
    let shape = catalog.routed_experts();
    let started = std::time::Instant::now();
    let mut peak = 0;
    for rank in 0..6 {
        let residencies = (shape.first_layer..shape.layers)
            .map(|layer| {
                catalog.exl3().unwrap().balanced_residency(
                    cuteafd_loader::V41Exl3Layer::Backbone(layer),
                    6,
                    rank,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let layouts = residencies
            .iter()
            .map(|r| r.device_arena_layout())
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let arena = ArenaLayout::new(layouts.iter().map(|(_, n)| *n), usize::MAX)?;
        let mut sizes = vec![0; 6];
        sizes[rank] = arena.bytes() as u64;
        let mut builder = Builder::new(&snapshot, sizes);
        for (index, (residency, (offsets, bytes))) in residencies.iter().zip(&layouts).enumerate() {
            builder.exl3_residency(
                &catalog,
                residency,
                offsets,
                Destination {
                    rank: rank as u32,
                    offset: arena.region(index)?.start as u64,
                    bytes: *bytes as u64,
                },
            )?;
        }
        let plan = builder.finish()?;
        peak = peak.max(simulate(&plan, rank, variable)?);
    }
    eprintln!("{variable} full coverage simulation: peak {peak} intervals, conservative tree bound {} bytes, elapsed {:?}", peak*128, started.elapsed());
    Ok(())
}

/// Deliver a rank's plan from six then eight interleaved sources in file
/// order plus one whole-file replay; returns the peak coverage intervals.
pub(crate) fn simulate(
    plan: &cuteafd_loader::push_load::builder::CheckedPlan,
    rank: usize,
    label: &str,
) -> Result<usize> {
    let expected = ExpectedCoverage::from_plan(plan, rank as u32)?;
    let mut peak = 0;
    let mut files: BTreeMap<u32, Vec<&Entry>> = BTreeMap::new();
    for entry in &plan.plan.entries {
        files.entry(entry.file).or_default().push(entry);
    }
    for sources in [6, 8] {
        let mut queues = (0..sources).map(|_| VecDeque::new()).collect::<Vec<_>>();
        for (&file, entries) in &files {
            queues[file as usize % sources]
                .push_back(FileWalk::new(entries.clone(), &plan.plan.patterns));
        }
        let mut coverage = Coverage::default();
        while queues.iter().any(|q| !q.is_empty()) {
            for queue in &mut queues {
                if let Some(file) = queue.front_mut() {
                    if !file.batch(&expected, &mut coverage)? {
                        queue.pop_front();
                    }
                }
            }
        }
        assert!(expected.complete(&coverage));
        let before = coverage.bytes;
        let mut replay = FileWalk::new(
            files.first_key_value().unwrap().1.clone(),
            &plan.plan.patterns,
        );
        while replay.batch(&expected, &mut coverage)? {}
        assert_eq!(coverage.bytes, before);
        peak = peak.max(coverage.peak);
        eprintln!("{label} TP6 rank{rank}, {sources} sources + file replay: peak {} intervals, conservative tree bound {} bytes", coverage.peak, coverage.peak*128);
    }
    Ok(peak)
}

#[test]
fn real_v41_full_plan_interleaved_sources_and_file_replay() -> Result<()> {
    let Some(snapshot) = std::env::var_os("CUTEAFD_PUSH_V41") else {
        return Ok(());
    };
    let snapshot = std::path::PathBuf::from(snapshot);
    let catalog = cuteafd_loader::read_official_v41_catalog(&snapshot)?;
    let shape = catalog.routed_experts();
    for rank in 0..6 {
        let mut sizes = vec![0; 6];
        sizes[rank] = u64::MAX;
        let mut builder = Builder::new(&snapshot, sizes);
        let mut cursor = 0u64;
        for layer in shape.first_layer..shape.layers {
            for expert in 0..shape.experts {
                let staging =
                    catalog.expert_staging(cuteafd_loader::V41ExpertSelection::BackboneTp {
                        layer,
                        expert,
                        world: 6,
                        rank,
                    })?;
                builder.v41_regions(
                    &catalog,
                    staging.tensor_names(),
                    staging.tensor_ranges(),
                    6,
                    rank as u32,
                    Destination {
                        rank: rank as u32,
                        offset: cursor,
                        bytes: staging.staging_bytes() as u64,
                    },
                )?;
                cursor = cursor
                    .checked_add(staging.staging_bytes() as u64)
                    .context("V4.1 arena overflow")?;
            }
        }
        let mut plan = builder.finish()?;
        plan.arenas[rank] = cursor;
        plan.plan.check(&plan.file_sizes, &plan.arenas)?;
        simulate(&plan, rank, "V4.1 full-layer native")?;
    }
    Ok(())
}
