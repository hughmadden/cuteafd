//! Pipelined Spark prefill lanes, shared by GLM 5.3 Flash and MiniMax M3: a prefill chunk of up
//! to `lanes` x `rows` rows (`--prefill-lanes`, `--prefill-rows`) runs as consecutive lanes on one
//! stream, each lane with its own Spark transport, so one lane's expert exchange is in flight
//! while another lane's GPU layers run. The lanes share one set of attention temporaries; each
//! holds its own rows of the step's per-row buffers.

/// Lanes a long Spark prefill chunk splits into by default, and at most (`--prefill-lanes`).
pub const DEFAULT_PREFILL_LANES: usize = 2;
pub const MAX_PREFILL_LANES: usize = 4;
/// The fewest rows per lane worth another exchange per layer.
pub const MIN_LANE_ROWS: usize = 256;

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum LaneError {
    #[error("prefill of {tokens} tokens exceeds {lanes} lanes of {rows} rows")]
    Capacity { tokens: usize, lanes: usize, rows: usize },
    #[error("prefill lane of {per_lane} tokens exceeds {rows} rows")]
    Lane { per_lane: usize, rows: usize },
}

/// The longest chunk `lanes` lanes of `rows` rows take. Multi-lane splits must land on `page`-row
/// boundaries (the family's cache page). A narrow workspace still accepts a single unpadded
/// tail, but cannot advertise unusable padding.
pub fn lane_capacity(lanes: usize, rows: usize, page: usize) -> usize {
    rows.max(lanes * (rows / page) * page)
}

/// How a chunk of `tokens` rows runs on up to `lanes` lanes of `rows` rows: (lanes, rows per
/// lane). One lane per [`MIN_LANE_ROWS`] rows, at most `lanes`, and at least as many as the rows
/// need once each lane's share is rounded up to whole pages: every lane but the last holds the
/// same number of pages, so each lane's pools and pages start where the previous lane's end.
/// Two lanes keep the rule they always had: two from twice [`MIN_LANE_ROWS`] rows (or when one
/// lane cannot hold the chunk), cut at the middle rounded up to a page.
pub fn lane_plan(tokens: usize, lanes: usize, rows: usize, page: usize) -> Result<(usize, usize), LaneError> {
    if tokens == 0 || tokens > lane_capacity(lanes, rows, page) {
        return Err(LaneError::Capacity { tokens, lanes, rows });
    }
    let mut wanted = (tokens / MIN_LANE_ROWS).clamp(1, lanes.max(1)).max(tokens.div_ceil(rows.max(1)));
    let per_lane = loop {
        let per_lane = if wanted == 1 { tokens } else { tokens.div_ceil(wanted).next_multiple_of(page) };
        if per_lane <= rows || wanted >= lanes {
            break per_lane;
        }
        wanted += 1;
    };
    if per_lane > rows {
        return Err(LaneError::Lane { per_lane, rows });
    }
    Ok((tokens.div_ceil(per_lane), per_lane))
}

/// The rows of each lane a pipelined prefill of `tokens` rows runs in, in order.
pub fn lane_cuts(tokens: usize, lanes: usize, rows: usize, page: usize) -> Result<Vec<usize>, LaneError> {
    let (_, per_lane) = lane_plan(tokens, lanes, rows, page)?;
    Ok((0..tokens).step_by(per_lane).map(|first| per_lane.min(tokens - first)).collect())
}

/// The longest chunk a lone prefill runs as one step: pipelined Spark lanes ([`lane_plan`])
/// cut a chunk into two lanes from twice [`MIN_LANE_ROWS`] rows; a serial prefill takes `rows`.
pub fn single_pass_rows(pipelined: bool, lanes: usize, rows: usize) -> usize {
    if pipelined && lanes > 1 { (2 * MIN_LANE_ROWS - 1).min(rows) } else { rows }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lanes_cut_whole_pages_of_any_family() {
        // GLM's MLA page and MiniMax's KV page are both 64 rows.
        assert_eq!(lane_capacity(2, 4096, 64), 8192);
        assert_eq!(lane_cuts(8192, 2, 4096, 64).unwrap(), [4096, 4096]);
        assert_eq!(lane_cuts(4412, 2, 4096, 64).unwrap(), [2240, 2172]);
        assert_eq!(lane_cuts(3712, 2, 4096, 64).unwrap(), [1856, 1856]);
        assert_eq!(lane_cuts(511, 2, 4096, 64).unwrap(), [511]);
        assert_eq!(lane_cuts(4096, 1, 4096, 64).unwrap(), [4096]);
        // A coarser page keeps every cut on it.
        assert_eq!(lane_cuts(4412, 2, 4096, 1024).unwrap(), [3072, 1340]);
        assert_eq!(lane_capacity(2, 3000, 1024), 4096);
        assert_eq!(lane_plan(8193, 2, 4096, 64), Err(LaneError::Capacity { tokens: 8193, lanes: 2, rows: 4096 }));
        assert!(lane_plan(0, 2, 4096, 64).is_err());
        assert_eq!(single_pass_rows(true, 2, 4096), 2 * MIN_LANE_ROWS - 1);
        assert_eq!(single_pass_rows(false, 2, 4096), 4096);
    }
}
