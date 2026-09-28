//! Host-built step metadata for DeepSeek V4: RoPE tables and the slot, window
//! and compressor tables one sequence prefilled from position 0 needs.
use cuteafd_loader::deepseek_v4::DeepseekV4Config;

pub(crate) const WINDOW: usize = 128;
pub(crate) const SOURCE_PAGE_TOKENS: usize = 256;
pub(crate) const MAIN_PAGE_BYTES: usize = 149_760;
pub(crate) const INDEX_PAGE_ROWS: usize = 64;
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

/// One compressor ratio's prefill tables (single sequence from position 0).
pub(crate) struct CompressorPrefill {
    pub groups: usize,
    pub pages: usize,
    pub active_groups: Vec<i32>,
    pub group_source_starts: Vec<i32>,
    pub group_rope_positions: Vec<i32>,
    pub compressed_slots: Vec<i32>,
    pub active_sequences: Vec<i32>,
    pub sequence_offsets: Vec<i32>,
    pub state_sequence_ids: Vec<i32>,
    /// Completed groups visible to each row.
    pub visible: Vec<i32>,
    /// Lengths the sparse attention reads from the indexed (compressed) cache.
    pub indexed_lengths: Vec<i32>,
    /// C128: dense causal indices [T, width].
    pub indexed_indices: Vec<i32>,
}

pub(crate) struct PrefillMetadata {
    pub positions: Vec<i64>,
    pub swa_indices: Vec<i32>,
    pub swa_lengths: Vec<i32>,
    pub c4: CompressorPrefill,
    pub c128: CompressorPrefill,
}

pub(crate) fn prefill(tokens: usize, index_topk: usize, c128_width: usize) -> PrefillMetadata {
    let positions: Vec<i64> = (0..tokens as i64).collect();
    let mut swa_indices = vec![-1i32; tokens * WINDOW];
    for t in 0..tokens {
        let start = t.saturating_sub(WINDOW - 1);
        for j in 0..WINDOW {
            if start + j <= t {
                swa_indices[t * WINDOW + j] = (start + j) as i32;
            }
        }
    }
    let compressor = |ratio: usize| {
        let groups = tokens / ratio;
        let pages = groups.div_ceil(compressed_page_rows(ratio)).max(1);
        let visible: Vec<i32> = (0..tokens).map(|t| ((t + 1) / ratio) as i32).collect();
        let starts: Vec<i32> = (0..groups).map(|j| (j * ratio) as i32).collect();
        let (indexed_lengths, indexed_indices) = if ratio == 4 {
            (visible.iter().map(|&v| v.min(index_topk as i32)).collect(), Vec::new())
        } else {
            let mut indices = vec![-1i32; tokens * c128_width];
            for t in 0..tokens {
                for j in 0..(visible[t] as usize).min(c128_width) {
                    indices[t * c128_width + j] = j as i32;
                }
            }
            (visible.clone(), indices)
        };
        CompressorPrefill {
            groups,
            pages,
            active_groups: vec![groups as i32],
            group_source_starts: starts.clone(),
            group_rope_positions: starts,
            compressed_slots: (0..groups as i32).collect(),
            active_sequences: vec![1],
            sequence_offsets: vec![0, tokens as i32],
            state_sequence_ids: vec![0],
            visible,
            indexed_lengths,
            indexed_indices,
        }
    };
    PrefillMetadata {
        positions,
        swa_indices,
        swa_lengths: (0..tokens).map(|t| (t + 1).min(WINDOW) as i32).collect(),
        c4: compressor(4),
        c128: compressor(128),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefill_tables_match_the_prototype() {
        let meta = prefill(300, 512, 1024);
        assert_eq!(&meta.swa_indices[..3], &[0, -1, -1]);
        assert_eq!(meta.swa_indices[200 * WINDOW], 73);
        assert_eq!(meta.swa_lengths[299], 128);
        assert_eq!((meta.c4.groups, meta.c4.pages), (75, 2));
        assert_eq!(meta.c4.visible[7], 2);
        assert_eq!((meta.c128.groups, meta.c128.pages), (2, 1));
        assert_eq!(&meta.c128.indexed_indices[255 * 1024..255 * 1024 + 3], &[0, 1, -1]);
        assert_eq!(compressed_page_bytes(4), 37_440);
        assert_eq!(compressed_page_bytes(128), 1_728);
    }
}

/// Host tables for one step of one sequence: its rows start at `start`
/// (0 for the initial prefill) and the caches already hold `start` tokens.
/// Prefill steps must start at 0; decode steps are one row.
pub(crate) struct StepTables {
    pub decode: bool,
    pub rows: usize,
    pub positions: Vec<i64>,
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

/// Cache pages a sequence of `capacity` tokens needs, per cache.
pub(crate) fn cache_pages(capacity: usize) -> (usize, usize, usize) {
    (
        capacity.div_ceil(SOURCE_PAGE_TOKENS).max(1),
        (capacity / 4).div_ceil(compressed_page_rows(4)).max(1),
        (capacity / 128).div_ceil(compressed_page_rows(128)).max(1),
    )
}

pub(crate) fn prefill_step(tokens: usize, index_topk: usize, c128_width: usize, capacity: usize) -> StepTables {
    let meta = prefill(tokens, index_topk, c128_width);
    let c4 = &meta.c4;
    let c128 = &meta.c128;
    let nonempty = |v: &Vec<i32>| if v.is_empty() { vec![0] } else { v.clone() };
    let tables = |c: &CompressorPrefill| vec![
        ("active_groups", c.active_groups.clone()),
        ("group_source_starts", nonempty(&c.group_source_starts)),
        ("group_rope_positions", nonempty(&c.group_rope_positions)),
        ("compressed_slots", nonempty(&c.compressed_slots)),
        ("active_sequences", c.active_sequences.clone()),
        ("sequence_offsets", c.sequence_offsets.clone()),
        ("state_sequence_ids", c.state_sequence_ids.clone()),
    ];
    let (_, c4_pages, _) = cache_pages(capacity);
    StepTables {
        decode: false,
        rows: tokens,
        positions: meta.positions.clone(),
        swa_indices: meta.swa_indices.clone(),
        swa_lengths: meta.swa_lengths.clone(),
        c4_tables: tables(c4),
        c128_tables: tables(c128),
        c4_groups: c4.groups,
        c128_groups: c128.groups,
        c4_page_table: (0..c4_pages as i32).collect(),
        c4_table_width: c4.pages,
        c4_table_stride: 0,
        c4_visible: c4.visible.clone(),
        c4_indexed_lengths: c4.indexed_lengths.clone(),
        c128_indices: c128.indexed_indices.clone(),
        c128_lengths: c128.indexed_lengths.clone(),
    }
}

pub(crate) fn decode_step(position: usize, index_topk: usize, c128_width: usize, capacity: usize) -> StepTables {
    let start = position.saturating_sub(WINDOW - 1);
    let swa_indices = (0..WINDOW).map(|j| if start + j <= position { (start + j) as i32 } else { -1 }).collect();
    let visible4 = (position + 1) / 4;
    let visible128 = (position + 1) / 128;
    let decode_tables = |ratio: usize| vec![
        ("positions", vec![position as i32]),
        ("sequence_ids", vec![0]),
        ("compressed_slots", vec![(position / ratio) as i32]),
    ];
    let (_, c4_pages, _) = cache_pages(capacity);
    let used_pages = visible4.div_ceil(INDEX_PAGE_ROWS).max(1);
    StepTables {
        decode: true,
        rows: 1,
        positions: vec![position as i64],
        swa_indices,
        swa_lengths: vec![(position + 1).min(WINDOW) as i32],
        c4_tables: decode_tables(4),
        c128_tables: decode_tables(128),
        c4_groups: visible4,
        c128_groups: visible128,
        c4_page_table: (0..c4_pages as i32).collect(),
        c4_table_width: used_pages,
        c4_table_stride: c4_pages,
        c4_visible: vec![visible4 as i32],
        c4_indexed_lengths: vec![visible4.min(index_topk) as i32],
        c128_indices: (0..c128_width).map(|j| if j < visible128 { j as i32 } else { -1 }).collect(),
        c128_lengths: vec![visible128 as i32],
    }
}
