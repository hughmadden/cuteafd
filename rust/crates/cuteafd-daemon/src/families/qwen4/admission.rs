//! Qwen automatic KV admission after weights/PLE and exact expert ownership are established.
use super::EngineArgs;
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::NativeLibrary;
use cuteafd_loader::families::qwen4::Qwen4Config;
use cuteafd_loader::serving_capacity::qwen_cache_geometry;

// 2026-10-07 TP4 SM120: 1,855,979,520 physical bytes / 12,397 graphs,
// rounded up; the margin also covers the earlier 21,560-graph measurement.
const QWEN_GRAPH_BYTES_PER_GRAPH_2026_10_07: u64 = 149_712;
const GRAPH_RESERVE_MARGIN_BYTES: u64 = 256 << 20;

fn graph_reserve(count: usize) -> Result<u64> {
    let measured = u64::try_from(count)?.checked_mul(QWEN_GRAPH_BYTES_PER_GRAPH_2026_10_07)
        .context("Qwen graph reserve overflow")?;
    measured.checked_add(measured.div_ceil(10).max(GRAPH_RESERVE_MARGIN_BYTES))
        .context("Qwen graph margin overflow")
}

/// Descend from the no-graph upper bound until that pool's exact geometry fits.
/// Re-evaluation keeps the final graph reserve separate from fixed headroom;
/// a smaller geometry never causes an oscillating grow/shrink admission loop.
fn solve_graph_pool(available: u64, fixed: u64, per_token: u64, unit: u64,
    target: u64, requested: Option<u64>, mut count: impl FnMut(u64) -> Result<usize>) -> Result<(u64, usize, u64)> {
    ensure!(unit > 0 && per_token > 0, "Qwen KV admission needs positive units and token bytes");
    let rounded = requested.map(|tokens| tokens.div_ceil(unit).checked_mul(unit)
        .context("Qwen KV token overflow")).transpose()?;
    let mut tokens = rounded.unwrap_or_else(|| available.saturating_sub(fixed) / per_token / unit * unit)
        .min(if rounded.is_some() { u64::MAX } else { target / unit * unit });
    loop {
        ensure!(tokens >= unit, "no room for Qwen KV after graph and fixed reserves");
        let graphs = count(tokens)?;
        let reserve = graph_reserve(graphs)?;
        let total_reserve = fixed.checked_add(reserve).context("Qwen fixed reserve overflow")?;
        let capacity = available.saturating_sub(total_reserve) / per_token / unit * unit;
        if tokens <= capacity { return Ok((tokens, graphs, reserve)); }
        ensure!(rounded.is_none(), "fixed Qwen KV pool of {tokens} tokens does not fit with {graphs} startup graphs ({reserve} graph reserve bytes)");
        tokens = capacity;
    }
}

pub(super) fn pool_tokens(library: &NativeLibrary, args: &EngineArgs, cfg: &Qwen4Config,
    layers: usize, mtp: bool, future_expert_bytes: u64) -> Result<usize> {
    let geometry = qwen_cache_geometry(cfg, layers, mtp)?;
    let rank = &geometry.ranks[0];
    let costs = cuteafd_loader::plan::layout::family_costs("qwen4");
    let unit = geometry.logical_unit_rows;
    let marks = args.planner_prefix_bytes.unwrap_or(rank.retained_mark_bytes * costs.mark_slots);
    let fixed = future_expert_bytes + costs.workspace_bytes[0] * args.prefill_rows.max(1) as u64 / 4096
        // Persistent row and block-start T/H/W tables for both workspaces.
        + 24 * (args.prefill_rows.max(1) as u64 + super::engine::DECODE_ROWS as u64)
        + rank.active_state_per_sequence_bytes * args.slots as u64
        + rank.fixed_state_bytes + rank.speculative_replay_bytes + marks
        + cuteafd_loader::plan::layout::LayoutOptions::default().headroom_bytes.max(3 << 30);
    // Prefill owns one page table and decode owns 64, each with four record
    // page ids plus one pool page id per 256-token allocation unit.
    let context_tables_per_unit = (1 + super::engine::DECODE_ROWS as u64) * 5 * 4;
    let per_token = (rank.persistent_unit_bytes + rank.pool_metadata_unit_bytes
        + context_tables_per_unit).div_ceil(unit);
    let target = cuteafd_core::serving_capacity::DEFAULT_GPU_KV_TOKENS;
    let requested = (args.pool_tokens > 0).then_some(args.pool_tokens as u64);
    let enabled = super::engine::startup_graphs_enabled(
        std::env::var("CUTEAFD_QWEN4_GRAPHS").ok().as_deref(),
        std::env::var("CUTEAFD_QWEN4_STARTUP_GRAPHS").ok().as_deref());
    let (requested, graph_bytes) = if let Some((sequences, speculation)) = args.planner_graph_modes.filter(|_| enabled) {
        ensure!(sequences <= 16, "Qwen startup graphs support at most 16 concurrent sequences");
        let current = library.cuda_get_device()?;
        library.cuda_set_device(args.device)?;
        let sample = library.cuda_memory_info();
        library.cuda_set_device(current)?;
        let (available, _) = sample?;
        let (tokens, count, reserve) = solve_graph_pool(available as u64, fixed, per_token, unit, target, requested,
            |tokens| super::engine::serving_graph_count(args.max_context, usize::try_from(tokens)?,
                cfg.dense_context(), sequences, speculation, layers))?;
        tracing::info!(pool_tokens = tokens, graphs = count, graph_reserve_bytes = reserve,
            bytes_per_graph = QWEN_GRAPH_BYTES_PER_GRAPH_2026_10_07,
            headroom_bytes = cuteafd_loader::plan::layout::LayoutOptions::default().headroom_bytes.max(3 << 30),
            "Qwen graph-aware KV admission before allocation");
        (Some(tokens), reserve)
    } else { (requested, costs.graph_bytes[0]) };
    let tokens = crate::shared::memory_report::admitted_pool_tokens(library,
        &[crate::shared::memory_report::KvDevice { device: args.device,
            bytes_per_token: per_token, reserve_bytes: fixed.checked_add(graph_bytes)
                .context("Qwen admission reserve overflow")? }], unit, target, requested)?;
    Ok(usize::try_from(tokens)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_admission_subtracts_measured_graph_reserve_and_separate_headroom() {
        let available = 4_u64 << 30;
        let headroom = 1_u64 << 30;
        let per_token = 4096;
        let (tokens, count, reserve) = solve_graph_pool(available, headroom, per_token, 256,
            2_097_152, None, |_| Ok(12397)).unwrap();
        assert_eq!(reserve, 12397 * QWEN_GRAPH_BYTES_PER_GRAPH_2026_10_07 + GRAPH_RESERVE_MARGIN_BYTES);
        assert_eq!(count, 12397);
        assert_eq!(tokens, (available - headroom - reserve) / per_token / 256 * 256);
        assert!(tokens < (available - headroom) / per_token);
        assert!(tokens * per_token + headroom + reserve <= available);
    }

    #[test]
    fn admission_recounts_geometry_after_pool_shrinks_and_rejects_fixed_overcommit() {
        let count = |tokens| Ok(if tokens >= 32768 { 21560 } else { 7840 });
        let (tokens, graphs, reserve) = solve_graph_pool(4 << 30, 512 << 20, 65536,
            256, 65536, None, count).unwrap();
        assert!(tokens < 32768);
        assert_eq!(graphs, 7840);
        assert_eq!(reserve, graph_reserve(graphs).unwrap());
        assert!(tokens * 65536 + (512 << 20) + reserve <= 4 << 30);
        assert!(solve_graph_pool(4 << 30, 512 << 20, 65536,
            256, 65536, Some(32768), count).is_err());
    }

    #[test]
    fn margin_and_fixed_pool_rounding_do_not_spend_headroom() {
        let measured = 21560 * QWEN_GRAPH_BYTES_PER_GRAPH_2026_10_07;
        assert_eq!(graph_reserve(21560).unwrap(), measured + measured.div_ceil(10));
        let (tokens, _, reserve) = solve_graph_pool(8 << 30, 3 << 30, 28422,
            256, 2097152, Some(73727), |_| Ok(12397)).unwrap();
        assert_eq!(tokens, 73728);
        assert!(tokens * 28422 + reserve + (3 << 30) <= 8 << 30);
        assert!(solve_graph_pool(256 << 20, 0, 28422, 256, 2097152,
            None, |_| Ok(12397)).is_err());
    }

    #[test]
    fn graph_counts_match_actual_pool_context_layers_and_serving_modes() {
        use super::super::engine::serving_graph_count;
        assert_eq!(serving_graph_count(8192, 73728, 2051, 16, true, 48).unwrap(), 12397);
        assert_eq!(serving_graph_count(8192, 2097152, 2051, 16, true, 48).unwrap(), 12397);
        assert_eq!(serving_graph_count(8192, 73728, 2051, 16, false, 48).unwrap(), 4508);
        assert_eq!(serving_graph_count(32768, 32768, 2051, 16, true, 48).unwrap(), 21560);
        assert!(serving_graph_count(32768, 4096, 2051, 16, true, 48).unwrap() < 21560);
    }
}
