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
    /// C4: shared index page table. C128: dense causal indices [T, width].
    pub index_page_table: Vec<i32>,
    pub indexed_indices: Vec<i32>,
}

pub(crate) struct PrefillMetadata {
    pub tokens: usize,
    pub positions: Vec<i64>,
    pub positions_i32: Vec<i32>,
    pub main_pages: usize,
    pub main_slots: Vec<i64>,
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
        let (index_page_table, indexed_lengths, indexed_indices) = if ratio == 4 {
            (
                (0..pages as i32).collect(),
                visible.iter().map(|&v| v.min(index_topk as i32)).collect(),
                Vec::new(),
            )
        } else {
            let mut indices = vec![-1i32; tokens * c128_width];
            for t in 0..tokens {
                for j in 0..(visible[t] as usize).min(c128_width) {
                    indices[t * c128_width + j] = j as i32;
                }
            }
            (Vec::new(), visible.clone(), indices)
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
            index_page_table,
            indexed_indices,
        }
    };
    PrefillMetadata {
        tokens,
        positions_i32: positions.iter().map(|&p| p as i32).collect(),
        main_pages: tokens.div_ceil(SOURCE_PAGE_TOKENS),
        main_slots: positions.clone(),
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
