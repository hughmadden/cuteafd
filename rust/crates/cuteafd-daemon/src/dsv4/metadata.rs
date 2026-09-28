//! Host-built step metadata for DeepSeek V4: RoPE tables and the slot, window
//! and compressor tables one sequence prefilled from position 0 needs.
use cuteafd_loader::deepseek_v4::DeepseekV4Config;

pub(crate) const WINDOW: usize = 128;
pub(crate) const SOURCE_PAGE_TOKENS: usize = 256;
pub(crate) const MAIN_PAGE_BYTES: usize = 149_760;
pub(crate) const INDEX_PAGE_BYTES: usize = 8_448;

pub(crate) fn compressed_page_rows(ratio: usize) -> usize {
    SOURCE_PAGE_TOKENS / ratio
}

/// Compressed-MLA page: rows x 584 bytes rounded up to a 576-byte multiple.
pub(crate) fn compressed_page_bytes(ratio: usize) -> usize {
    (compressed_page_rows(ratio) * 584).div_ceil(576) * 576
}

/// b12x RoPE table, FP32 [positions, 64] = [cos(32) | sin(32)] (model.py
/// precompute_freqs_cis). Window-only layers use rope_theta without YaRN;
/// compressed layers use compress_rope_theta with YaRN.
pub(crate) fn rope_table(cfg: &DeepseekV4Config, compressed: bool, positions: usize) -> Vec<f32> {
    let dim = cfg.rope_head_dim;
    let (original, base) = if compressed {
        (cfg.original_seq_len, cfg.compress_rope_theta)
    } else {
        (0, cfg.rope_theta)
    };
    let half = dim / 2;
    let mut freqs: Vec<f64> = (0..half)
        .map(|i| {
            let exponent = (2 * i) as f32 / dim as f32;
            f64::from(1.0f32 / (base as f32).powf(exponent))
        })
        .collect();
    if original > 0 {
        let correction = |rotations: f64| {
            dim as f64 * (original as f64 / (rotations * 2.0 * std::f64::consts::PI)).ln()
                / (2.0 * base.ln())
        };
        let low = correction(cfg.beta_fast).floor().max(0.0);
        let high = correction(cfg.beta_slow).ceil().min(dim as f64 - 1.0);
        let high = if low == high { high + 0.001 } else { high };
        for (i, freq) in freqs.iter_mut().enumerate() {
            let ramp = ((i as f64 - low) / (high - low)).clamp(0.0, 1.0);
            let smooth = 1.0 - ramp;
            *freq = *freq / cfg.rope_factor * (1.0 - smooth) + *freq * smooth;
        }
    }
    let mut table = vec![0f32; positions * dim];
    for t in 0..positions {
        for (i, freq) in freqs.iter().enumerate() {
            // torch.outer runs in FP32.
            let angle = f64::from(t as f32 * *freq as f32);
            table[t * dim + i] = angle.cos() as f32;
            table[t * dim + half + i] = angle.sin() as f32;
        }
    }
    table
}

/// Host tables for one step of one sequence: its rows start at `start`
/// (0 for the initial prefill) and the caches already hold `start` tokens.
/// Prefill steps must start at 0; decode steps are one row.
pub(crate) struct StepTables {
    pub decode: bool,
    pub rows: usize,
    pub positions: Vec<i64>,
    /// Physical window slots the producer writes, per row.
    pub main_slots: Vec<i64>,
    pub swa_indices: Vec<i32>,
    pub swa_lengths: Vec<i32>,
    /// Compressor metadata in program pointer order, per ratio.
    pub c4_tables: Vec<(&'static str, Vec<i32>)>,
    pub c128_tables: Vec<(&'static str, Vec<i32>)>,
    /// Completed groups after this step (grid bound for prefill programs).
    pub c4_groups: usize,
    pub c128_groups: usize,
    /// C4 index top-k: page table (shared row for prefill, [rows, stride] for decode).
    pub c4_page_table: Vec<i32>,
    pub c4_table_width: usize,
    pub c4_table_stride: usize,
    pub c4_visible: Vec<i32>,
    pub c4_indexed_lengths: Vec<i32>,
    pub c128_indices: Vec<i32>,
    pub c128_lengths: Vec<i32>,
}

/// Tables for a prefill chunk of one sequence that starts at position 0.
pub(crate) fn prefill_step(
    placement: &super::pool::Placement,
    shape: &super::pool::PoolShape,
    tokens: usize,
    index_topk: usize,
    c128_width: usize,
) -> anyhow::Result<StepTables> {
    let positions: Vec<i64> = (0..tokens as i64).collect();
    let slot = |p: usize| placement.window_slot(shape, p);
    let mut swa_indices = vec![-1i32; tokens * WINDOW];
    for t in 0..tokens {
        let start = t.saturating_sub(WINDOW - 1);
        for j in 0..WINDOW {
            if start + j <= t {
                swa_indices[t * WINDOW + j] = slot(start + j) as i32;
            }
        }
    }
    let prefill_tables = |ratio: usize| -> anyhow::Result<(Vec<(&'static str, Vec<i32>)>, usize)> {
        let groups = tokens / ratio;
        let starts: Vec<i32> = (0..groups).map(|j| (j * ratio) as i32).collect();
        let slots = (0..groups).map(|j| placement.group_slot(ratio, j)).collect::<anyhow::Result<Vec<_>>>()?;
        let nonempty = |v: Vec<i32>| if v.is_empty() { vec![0] } else { v };
        Ok((vec![
            ("active_groups", vec![groups as i32]),
            ("group_source_starts", nonempty(starts.clone())),
            ("group_rope_positions", nonempty(starts)),
            ("compressed_slots", nonempty(slots)),
            ("active_sequences", vec![1]),
            ("sequence_offsets", vec![0, tokens as i32]),
            ("state_sequence_ids", vec![placement.state as i32]),
        ], groups))
    };
    let (c4_tables, c4_groups) = prefill_tables(4)?;
    let (c128_tables, c128_groups) = prefill_tables(128)?;
    let visible4: Vec<i32> = (0..tokens).map(|t| ((t + 1) / 4) as i32).collect();
    let mut c128_indices = vec![-1i32; tokens * c128_width];
    for t in 0..tokens {
        for j in 0..((t + 1) / 128).min(c128_width) {
            c128_indices[t * c128_width + j] = placement.group_slot(128, j)?;
        }
    }
    Ok(StepTables {
        decode: false,
        rows: tokens,
        swa_indices,
        swa_lengths: (0..tokens).map(|t| (t + 1).min(WINDOW) as i32).collect(),
        main_slots: positions.iter().map(|&p| slot(p as usize)).collect(),
        positions,
        c4_tables,
        c128_tables,
        c4_groups,
        c128_groups,
        c4_page_table: placement.c4_pages.clone(),
        c4_table_width: c4_groups.div_ceil(compressed_page_rows(4)).max(1),
        c4_table_stride: 0,
        c4_indexed_lengths: visible4.iter().map(|&v| v.min(index_topk as i32)).collect(),
        c4_visible: visible4,
        c128_lengths: (0..tokens).map(|t| ((t + 1) / 128) as i32).collect(),
        c128_indices,
    })
}

/// Tables for one decode row per sequence: `(placement, position)`.
pub(crate) fn decode_step(
    rows: &[(&super::pool::Placement, usize)],
    shape: &super::pool::PoolShape,
    index_topk: usize,
    c128_width: usize,
) -> anyhow::Result<StepTables> {
    let n = rows.len();
    let mut tables = StepTables {
        decode: true,
        rows: n,
        positions: Vec::with_capacity(n),
        main_slots: Vec::with_capacity(n),
        swa_indices: vec![-1; n * WINDOW],
        swa_lengths: Vec::with_capacity(n),
        c4_tables: Vec::new(),
        c128_tables: Vec::new(),
        c4_groups: 0,
        c128_groups: 0,
        c4_page_table: Vec::new(),
        c4_table_width: 1,
        c4_table_stride: rows.iter().map(|(p, _)| p.c4_pages.len()).max().unwrap_or(1),
        c4_visible: Vec::with_capacity(n),
        c4_indexed_lengths: Vec::with_capacity(n),
        c128_indices: vec![-1; n * c128_width],
        c128_lengths: Vec::with_capacity(n),
    };
    let (mut positions4, mut states, mut slots4, mut slots128) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (row, (placement, position)) in rows.iter().enumerate() {
        let position = *position;
        tables.positions.push(position as i64);
        tables.main_slots.push(placement.window_slot(shape, position));
        let start = position.saturating_sub(WINDOW - 1);
        for j in 0..WINDOW {
            if start + j <= position {
                tables.swa_indices[row * WINDOW + j] = placement.window_slot(shape, start + j) as i32;
            }
        }
        tables.swa_lengths.push((position + 1).min(WINDOW) as i32);
        positions4.push(position as i32);
        states.push(placement.state as i32);
        slots4.push(placement.group_slot(4, position / 4)?);
        slots128.push(placement.group_slot(128, position / 128)?);
        let visible4 = (position + 1) / 4;
        let visible128 = (position + 1) / 128;
        tables.c4_groups = tables.c4_groups.max(visible4);
        tables.c128_groups = tables.c128_groups.max(visible128);
        tables.c4_visible.push(visible4 as i32);
        tables.c4_indexed_lengths.push(visible4.min(index_topk) as i32);
        tables.c4_table_width = tables.c4_table_width.max(visible4.div_ceil(compressed_page_rows(4)));
        let mut pages = placement.c4_pages.clone();
        pages.resize(tables.c4_table_stride, pages.last().copied().unwrap_or(0));
        tables.c4_page_table.extend(pages);
        for j in 0..visible128.min(c128_width) {
            tables.c128_indices[row * c128_width + j] = placement.group_slot(128, j)?;
        }
        tables.c128_lengths.push(visible128 as i32);
    }
    tables.c4_tables = vec![("positions", positions4.clone()), ("sequence_ids", states.clone()), ("compressed_slots", slots4)];
    tables.c128_tables = vec![("positions", positions4), ("sequence_ids", states), ("compressed_slots", slots128)];
    Ok(tables)
}

#[cfg(test)]
mod tests {
    use super::super::pool::{PoolAllocator, PoolShape};
    use super::*;

    #[test]
    fn prefill_and_decode_tables_use_physical_slots() -> anyhow::Result<()> {
        let shape = PoolShape::new(2, 131_072, 4096, 64, 64);
        let mut pool = PoolAllocator::new(shape);
        let _other = pool.admit(1000)?;
        let seq = pool.admit(1000)?;
        let base = (seq.state * shape.ring_pages * SOURCE_PAGE_TOKENS) as i32;
        let pre = prefill_step(&seq, &shape, 300, 512, 1024)?;
        assert_eq!(&pre.swa_indices[..3], &[base, -1, -1]);
        assert_eq!(pre.swa_indices[200 * WINDOW], base + 73);
        assert_eq!((pre.c4_groups, pre.c128_groups), (75, 2));
        assert_eq!(pre.c4_table_width, 2);
        assert_eq!(pre.c4_tables[6].1, vec![seq.state as i32]);
        assert_eq!(&pre.c128_indices[255 * 1024..255 * 1024 + 3],
            &[seq.group_slot(128, 0)?, seq.group_slot(128, 1)?, -1]);
        let dec = decode_step(&[(&seq, 300)], &shape, 512, 1024)?;
        assert_eq!(dec.main_slots, vec![i64::from(base) + 300]);
        assert_eq!(dec.c4_visible, vec![75]);
        assert_eq!(dec.c4_tables[1].1, vec![seq.state as i32]);
        assert_eq!(compressed_page_bytes(4), 37_440);
        assert_eq!(compressed_page_bytes(128), 1_728);
        Ok(())
    }
}
