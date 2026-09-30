//! Intermediate snapshot points (off by default), so a request that shares only part of a retained prompt still
//! restores (byte-exactly) close to where it diverges: the deepest retained point whose tokens
//! are a prefix of it wins (`ReuseRule::EXACT` selection).
//!
//! Two sources, planned before a prompt's prefill:
//! - message boundaries of the rendered prompt (the template's message-start token): agent
//!   turns share everything up to the last message they did not re-render, and a chat
//!   template that drops old reasoning still shares everything before the assistant turn;
//! - periodic points every `gap` tokens at chunk boundaries of a long prefill
//!   (hughmadden/glm53f-afd's `branch_gap_tokens`).
//!
//! A point that the family can still capture from the state at a chunk end (within
//! `capture_reach` rows) costs nothing but its mark; a boundary further back splits the chunk.
//! At most `per_request` points per prompt, the deepest kept, so a 100K prompt does not flood
//! the banks; the prompt-end snapshot is separate.
use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct PointPolicy {
    /// Tokens between periodic points (0: none).
    pub gap: usize,
    /// Message boundaries to keep, counted back from the prompt end (0: none).
    pub boundaries: usize,
    /// Most points per prompt (the deepest kept).
    pub per_request: usize,
}

impl Default for PointPolicy {
    /// Off: agentic sessions hit prompt-end and turn-end snapshots exactly; intermediate points
    /// serve long shared prefixes with varying suffixes (e.g. gap 8192, 2 boundaries).
    fn default() -> Self {
        Self { gap: 0, boundaries: 0, per_request: 4 }
    }
}

/// A prefill schedule: chunk ends, and after each chunk the points to capture from it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PointPlan {
    /// Chunk ends in order; the last is the prompt length.
    pub chunks: Vec<usize>,
    /// (chunk index, point) in point order: capture `point` after chunk `index`.
    pub points: Vec<(usize, usize)>,
}

/// Positions of `marker` (message starts) in `tokens`: a snapshot of `tokens[..p]` ends right
/// before a message.
pub fn message_boundaries(tokens: &[u32], marker: u32) -> Vec<usize> {
    tokens.iter().enumerate().filter(|&(_, &t)| t == marker).map(|(i, _)| i).collect()
}

/// Plan the prefill of `tokens[resume..len]` in chunks of at most `chunk` rows with the points
/// `policy` asks for. `boundaries` are message-start positions; `reach` is how far behind a
/// chunk end the family can capture; points below `min_tokens` or at or before `resume` are
/// skipped (they are, or cannot beat, what was restored).
pub fn plan(resume: usize, len: usize, chunk: usize, boundaries: &[usize], reach: usize, min_tokens: usize,
    policy: PointPolicy) -> PointPlan {
    assert!(chunk > 0 && resume <= len);
    let floor = resume.max(min_tokens.saturating_sub(1));
    let mut wanted: Vec<usize> = boundaries
        .iter()
        .copied()
        .filter(|&b| b > floor && b < len)
        .collect();
    wanted.sort_unstable();
    wanted.dedup();
    let keep_boundaries = wanted.len().saturating_sub(policy.boundaries);
    wanted.drain(..keep_boundaries);
    // Regular chunk ends, then split chunks for boundaries a chunk end cannot reach.
    let regular = |from: usize| (1..).map(move |i| (from + i * chunk).min(len)).take_while(move |&e| e < len);
    let mut ends: Vec<usize> = regular(resume).collect();
    ends.push(len);
    for &b in &wanted {
        let end = *ends.iter().find(|&&e| e >= b).expect("the last end is len");
        if end - b > reach {
            ends.push(b);
            ends.sort_unstable();
        }
    }
    ends.dedup();
    // Periodic points at chunk ends (never the prompt end: that snapshot is separate).
    if policy.gap > 0 {
        let mut last = resume;
        for &e in &ends[..ends.len() - 1] {
            if e - last >= policy.gap {
                if e > floor {
                    wanted.push(e);
                }
                last = e;
            }
        }
    }
    wanted.sort_unstable();
    wanted.dedup();
    let excess = wanted.len().saturating_sub(policy.per_request);
    wanted.drain(..excess);
    let points = wanted
        .into_iter()
        .map(|p| (ends.iter().position(|&e| e >= p).expect("the last end is len"), p))
        .collect();
    PointPlan { chunks: ends, points }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boundaries_near_the_end_ride_on_the_last_chunk_and_far_ones_split_it() {
        let policy = PointPolicy { gap: 0, boundaries: 2, per_request: 4 };
        // A 10,000-token prompt from 0 in 4096-row chunks; messages start at 3000, 9000, 9995.
        let p = plan(0, 10_000, 4096, &[0, 3000, 9000, 9995], 124, 64, policy);
        // 9995 is 5 rows before the end: captured after the last chunk. 9000 is 1000 back: split.
        assert_eq!(p.chunks, vec![4096, 8192, 9000, 10_000]);
        assert_eq!(p.points, vec![(2, 9000), (3, 9995)]);
        // Points at or before the restored length are skipped.
        let p = plan(9500, 10_000, 4096, &[3000, 9000, 9995], 124, 64, policy);
        assert_eq!((p.chunks, p.points), (vec![10_000], vec![(0, 9995)]));
    }

    #[test]
    fn periodic_points_follow_the_gap_and_the_cap_keeps_the_deepest() {
        let policy = PointPolicy { gap: 8192, boundaries: 1, per_request: 4 };
        let p = plan(0, 100_000, 4096, &[99_990], 124, 64, policy);
        assert_eq!(p.chunks.len(), 25);
        // Periodic every 8192 up to 98304, the boundary at 99,990: the four deepest kept.
        let points: Vec<usize> = p.points.iter().map(|&(_, point)| point).collect();
        assert_eq!(points, vec![81_920, 90_112, 98_304, 99_990]);
        for &(chunk, point) in &p.points {
            assert!(p.chunks[chunk] >= point && p.chunks[chunk] - point <= 124);
        }
        // A resumed prefill counts the gap from where it resumed.
        let p = plan(40_000, 60_000, 4096, &[], 124, 64, PointPolicy { boundaries: 0, ..policy });
        assert_eq!(p.points.iter().map(|&(_, x)| x).collect::<Vec<_>>(), vec![48_192, 56_384]);
        // Nothing asked, nothing planned; a single short chunk.
        let p = plan(0, 300, 4096, &[], 124, 64, PointPolicy { gap: 0, boundaries: 0, per_request: 4 });
        assert_eq!((p.chunks, p.points), (vec![300], vec![]));
    }

    #[test]
    fn message_boundaries_are_marker_positions() {
        assert_eq!(message_boundaries(&[7, 1, 2, 7, 3, 7], 7), vec![0, 3, 5]);
    }
}
