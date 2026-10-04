//! Qwen automatic KV admission after weights, PLE and experts are resident.
use super::EngineArgs;
use anyhow::Result;
use cuteafd_ffi::NativeLibrary;
use cuteafd_loader::families::qwen4::Qwen4Config;
use cuteafd_loader::serving_capacity::qwen_cache_geometry;

pub(super) fn pool_tokens(library: &NativeLibrary, args: &EngineArgs, cfg: &Qwen4Config,
    layers: usize, mtp: bool) -> Result<usize> {
    let geometry = qwen_cache_geometry(cfg, layers, mtp)?;
    let rank = &geometry.ranks[0];
    let costs = cuteafd_loader::plan::layout::family_costs("qwen4");
    let unit = geometry.logical_unit_rows;
    let marks = args.planner_prefix_bytes.unwrap_or(rank.retained_mark_bytes * costs.mark_slots);
    let reserve = costs.workspace_bytes[0] * args.prefill_rows.max(1) as u64 / 4096
        + costs.graph_bytes[0]
        + rank.active_state_per_sequence_bytes * args.slots as u64
        + rank.fixed_state_bytes + rank.speculative_replay_bytes + marks
        + cuteafd_loader::plan::layout::LayoutOptions::default().headroom_bytes;
    // Prefill owns one page table and decode owns 64, each with four record
    // page ids plus one pool page id per 256-token allocation unit.
    let context_tables_per_unit = (1 + super::engine::DECODE_ROWS as u64) * 5 * 4;
    let tokens = crate::shared::memory_report::auto_pool_tokens(library,
        &[crate::shared::memory_report::KvDevice { device: args.device,
            bytes_per_token: (rank.persistent_unit_bytes + rank.pool_metadata_unit_bytes
                + context_tables_per_unit).div_ceil(unit), reserve_bytes: reserve }], unit,
        cuteafd_core::serving_capacity::DEFAULT_GPU_KV_TOKENS)?;
    Ok(usize::try_from(tokens)?)
}
