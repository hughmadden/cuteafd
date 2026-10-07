//! MiMo V2.6 fused `self_attn.qkv_proj`: FP8 E4M3 rows interleaved for
//! `metadata.tp_size` in the checkpoint index (Flash TP4, Pro TP8).
//! Every shard stores `[q | k | v]` with one 128x128 FP32 scale grid over
//! the whole shard. Flash SWA shards contain two contiguous key heads;
//! Pro has one key head per shard and pads it only in the program layout.
//! The reference modeling code splits the de-interleaved `[q; k; v]`.
use super::config::{MimoAttention, MimoV2Config};
use anyhow::{ensure, Context, Result};
use std::path::Path;

/// One run of rows that shares uniform 128-row scale blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QkvSegment {
    /// First row in the checkpoint tensor.
    pub source_row: usize,
    /// First grid row of this checkpoint shard, not this segment.
    pub scale_row: usize,
    /// Segment's row offset within its checkpoint shard.
    pub shard_offset: usize,
    /// First row in the de-interleaved `[q; k; v]`.
    pub dest_row: usize,
    pub rows: usize,
}

impl QkvSegment {
    /// Hugh Madden (issue #3): a block may straddle k/v; never restart at v.
    pub fn scale_row_of(&self, row: usize) -> usize {
        self.scale_row + (self.shard_offset + row) / 128
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FusedQkvLayout {
    pub shards: usize,
    /// Rows of q, k and v in one shard.
    pub q: usize,
    pub k: usize,
    pub v: usize,
}

impl FusedQkvLayout {
    pub fn new(cfg: &MimoV2Config, attention: MimoAttention, tp: usize) -> Result<Self> {
        let (heads, kv_heads) = (cfg.heads, cfg.kv_heads(attention));
        ensure!(tp > 0 && heads % tp == 0 && (kv_heads % tp == 0 || tp % kv_heads == 0),
            "{heads} query / {kv_heads} KV heads do not split over checkpoint TP {tp}");
        // SGLang replicates a KV head over the ranks sharing it (tp > kv): not stored that way here.
        ensure!(tp <= kv_heads, "fused qkv with replicated KV heads (checkpoint TP {tp} > {kv_heads}) is not supported");
        Ok(Self { shards: tp, q: heads / tp * cfg.head_dim, k: kv_heads / tp * cfg.head_dim,
            v: kv_heads / tp * cfg.v_head_dim })
    }

    /// Projection rows and copy runs for the selected program. Flash keeps
    /// multi-head checkpoint shards contiguous; only Pro pads one-head keys.
    pub fn program_segments(&self, cfg: &MimoV2Config) -> Result<(usize, Vec<QkvSegment>)> {
        let stride = cfg.qkv_key_stride();
        if stride == cfg.head_dim {
            return Ok((self.rows(), self.segments()));
        }
        ensure!(self.k == cfg.head_dim && stride % 128 == 0 && self.q % 128 == 0 && self.v % 128 == 0,
            "padded keys (stride {stride}) need one KV head per checkpoint shard and 128-row q/v shards");
        Ok((self.padded_rows(stride), self.segments_with_key_stride(stride)))
    }

    pub fn rows(&self) -> usize {
        self.shards * (self.q + self.k + self.v)
    }

    pub fn scale_rows(&self) -> usize {
        self.shards * (self.q + self.k + self.v).div_ceil(128)
    }

    /// Every (shard, part) run, in checkpoint order.
    pub fn segments(&self) -> Vec<QkvSegment> {
        self.segments_with_key_stride(self.k)
    }

    /// Rows of the de-interleaved `[q; k; v]` whose keys sit `key_stride` rows
    /// apart (`key_stride` >= a shard's key rows; the gap is zero padding).
    pub fn padded_rows(&self, key_stride: usize) -> usize {
        self.shards * (self.q + key_stride + self.v)
    }

    /// `segments` into the layout whose shard keys sit `key_stride` rows apart.
    /// Destination padding never participates in the checkpoint scale grid.
    pub fn segments_with_key_stride(&self, key_stride: usize) -> Vec<QkvSegment> {
        let (q_all, k_all) = (self.shards * self.q, self.shards * key_stride);
        let shard_rows = self.q + self.k + self.v;
        let mut out = Vec::with_capacity(3 * self.shards);
        for shard in 0..self.shards {
            let scale_row = shard * shard_rows.div_ceil(128);
            for (shard_offset, rows, dest_row) in [(0, self.q, shard * self.q),
                (self.q, self.k, q_all + shard * key_stride),
                (self.q + self.k, self.v, q_all + k_all + shard * self.v)] {
                out.push(QkvSegment { source_row: shard * shard_rows + shard_offset,
                    scale_row, shard_offset, dest_row, rows });
            }
        }
        out
    }
}

/// The index's positive integer `metadata.tp_size`; required for fused QKV.
pub fn checkpoint_tp(snapshot: &Path) -> Result<usize> {
    let index = crate::plan::checkpoint::read_json(&snapshot.join("model.safetensors.index.json"))
        .context("reading model.safetensors.index.json")?;
    let fused = index["weight_map"].as_object().is_some_and(|weights|
        weights.keys().any(|name| name.ends_with("self_attn.qkv_proj.weight")));
    let raw = &index["metadata"]["tp_size"];
    if raw.is_null() && !fused { return Ok(1); }
    let tp = raw.as_u64().filter(|&tp| tp > 0)
        .context("fused qkv_proj requires positive integer metadata.tp_size in the checkpoint index")?;
    usize::try_from(tp).context("metadata.tp_size does not fit usize")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shard_grid_row(layout: &FusedQkvLayout, source_row: usize) -> usize {
        let rows = layout.q + layout.k + layout.v;
        source_row / rows * rows.div_ceil(128) + source_row % rows / 128
    }

    #[test]
    fn poisoned_segment_pad_is_never_read() {
        // Hugh Madden's T2 (issue #3), adapted to 128-row blocks: the old
        // per-part ceilings read a third grid row that the shard never uses.
        let layout = FusedQkvLayout { shards: 2, q: 64, k: 64, v: 64 };
        assert_eq!(layout.scale_rows(), 4);
        let clean = [1.0f32, 2.0, 3.0, 4.0, 0.0, 0.0];
        let poisoned = [1.0f32, 2.0, 3.0, 4.0, 1e30, 1e30];
        let read = |grid: &[f32]| layout.segments().iter().flat_map(|segment|
            (0..segment.rows).map(|r| grid[segment.scale_row_of(r)]).collect::<Vec<_>>()).collect::<Vec<_>>();
        assert_eq!(read(&clean), read(&poisoned));
        let naive = |grid: &[f32]| (0..layout.shards * 3).flat_map(|part|
            std::iter::repeat_n(grid[part], 64)).collect::<Vec<_>>();
        assert_ne!(naive(&clean), naive(&poisoned));
        assert_ne!(naive(&poisoned), read(&poisoned));
    }

    #[test]
    fn flash_tp4_keeps_two_swa_key_heads_contiguous_and_scales_per_shard() -> Result<()> {
        let mut value = crate::plan::testing::mimo_flash_config();
        value["model_type"] = serde_json::json!("mimo_v2");
        value["rope_theta"] = serde_json::json!(1e7);
        value["layernorm_epsilon"] = serde_json::json!(1e-6);
        value["moe_router_dtype"] = serde_json::json!("bfloat16");
        let cfg = MimoV2Config::from_hf(&value)?;
        for (kind, rows, scales, key_rows) in [(MimoAttention::Full, 13568, 108, 192),
            (MimoAttention::Sliding, 14848, 116, 384)] {
            let layout = FusedQkvLayout::new(&cfg, kind, 4)?;
            let (width, segments) = layout.program_segments(&cfg)?;
            assert_eq!((width, layout.scale_rows(), layout.k), (rows, scales, key_rows));
            // Label each row by its shard grid; de-interleaving has no holes
            // or duplicate writes, and never restarts scales at a part boundary.
            let mut dest = vec![usize::MAX; width];
            for segment in &segments {
                for row in 0..segment.rows {
                    let at = segment.dest_row + row;
                    assert_eq!(dest[at], usize::MAX);
                    dest[at] = segment.scale_row_of(row);
                    assert_eq!(dest[at], shard_grid_row(&layout, segment.source_row + row));
                }
            }
            assert!(dest.iter().all(|&row| row < scales));
            let q_all = 64 * 192;
            for shard in 0..4 {
                let key = segments[shard * 3 + 1];
                assert_eq!(key.dest_row, q_all + shard * key_rows);
                assert_eq!(dest[key.dest_row + key_rows - 1], key.scale_row_of(key_rows - 1));
                let value = segments[shard * 3 + 2];
                if kind == MimoAttention::Full {
                    assert_eq!(dest[value.dest_row], dest[key.dest_row + 191]);
                    assert_eq!(dest[value.dest_row + 63], shard * 27 + 25);
                    assert_eq!(dest[value.dest_row + 64], shard * 27 + 26);
                }
            }
            let share = cfg.head_split(2)?;
            let half = FusedQkvLayout::new(&share, kind, 2)?;
            assert_eq!((half.program_segments(&share)?.0 * 2, half.scale_rows() * 2), (width, scales));
            for (whole, split) in segments[..6].iter().zip(half.segments()) {
                assert_eq!((whole.source_row, whole.scale_row, whole.rows), (split.source_row, split.scale_row, split.rows));
            }
        }
        Ok(())
    }

    #[test]
    fn fused_checkpoint_requires_positive_integer_tp_metadata() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("model.safetensors.index.json");
        for tp in [serde_json::Value::Null, serde_json::json!(0), serde_json::json!("4"),
            serde_json::json!(-1), serde_json::json!(4.5)] {
            let index = serde_json::json!({"metadata": {"tp_size": tp}, "weight_map": {
                "model.layers.0.self_attn.qkv_proj.weight": "model.safetensors"}});
            std::fs::write(&path, serde_json::to_vec(&index)?)?;
            assert!(checkpoint_tp(dir.path()).unwrap_err().to_string().contains("metadata.tp_size"));
        }
        std::fs::write(&path, serde_json::to_vec(&serde_json::json!({"metadata": {"tp_size": 4},
            "weight_map": {"model.layers.0.self_attn.qkv_proj.weight": "model.safetensors"}}))?)?;
        assert_eq!(checkpoint_tp(dir.path())?, 4);
        std::fs::write(&path, b"{\"weight_map\": {}}")?;
        assert_eq!(checkpoint_tp(dir.path())?, 1);
        Ok(())
    }

    #[test]
    fn pro_layout_matches_the_checkpoint_grid() -> Result<()> {
        let v = serde_json::json!({
            "model_type": "mimo_v2", "vocab_size": 152576, "hidden_size": 6144, "num_hidden_layers": 1,
            "num_attention_heads": 128, "num_key_value_heads": 8, "head_dim": 192, "v_head_dim": 128,
            "swa_num_key_value_heads": 8, "rope_theta": 1e7, "swa_rope_theta": 1e4, "sliding_window": 128,
            "hybrid_layer_pattern": [0], "moe_layer_freq": [0], "intermediate_size": 16384,
            "n_routed_experts": 384, "num_experts_per_tok": 8, "moe_intermediate_size": 2048,
            "attention_value_scale": 0.612,
        });
        let cfg = MimoV2Config::from_hf(&v)?;
        let layout = FusedQkvLayout::new(&cfg, MimoAttention::Full, 8)?;
        assert_eq!((layout.rows(), layout.scale_rows()), (27136, 216));
        let segments = layout.segments();
        assert_eq!(segments[1], QkvSegment { source_row: 3072, scale_row: 0, shard_offset: 3072,
            dest_row: 24576, rows: 192 });
        assert_eq!(segments[5], QkvSegment { source_row: 3392 + 3264, scale_row: 27, shard_offset: 3264,
            dest_row: 26112 + 128, rows: 128 });
        let (width, padded) = layout.program_segments(&cfg)?;
        assert_eq!(width, 8 * (3072 + 256 + 128));
        let mut rows = vec![None; width];
        for segment in &padded {
            for r in 0..segment.rows {
                assert!(rows[segment.dest_row + r].replace(segment.scale_row_of(r)).is_none());
                assert_eq!(segment.scale_row_of(r), shard_grid_row(&layout, segment.source_row + r));
            }
        }
        for shard in 0..8 {
            let key = padded[shard * 3 + 1];
            assert!(rows[key.dest_row + 192..key.dest_row + 256].iter().all(Option::is_none));
            let value = padded[shard * 3 + 2];
            assert_eq!(value.scale_row_of(0), shard * 27 + 25);
            assert_eq!(value.scale_row_of(63), key.scale_row_of(191));
            assert_eq!(value.scale_row_of(64), shard * 27 + 26);
        }
        // A two-GPU head split: each GPU takes four whole checkpoint shards (64 query heads,
        // 4 KV heads), the same per-shard layout at half the rows and grid rows.
        let share = cfg.head_split(2)?;
        assert_eq!((share.heads, share.full_kv_heads, share.dense_intermediate), (64, 4, 8192));
        assert_eq!((share.program_family()?, share.qkv_key_stride()), ("mimop2", 256));
        let half = FusedQkvLayout::new(&share, MimoAttention::Full, 4)?;
        assert_eq!((half.rows() * 2, half.scale_rows() * 2), (layout.rows(), layout.scale_rows()));
        // Rank 0's sources are the first four shards' (destinations follow the half layout).
        let source = |s: &[QkvSegment]| s.iter().map(|s| (s.source_row, s.scale_row, s.rows)).collect::<Vec<_>>();
        assert_eq!(source(&half.segments()), source(&segments[..12]));
        assert_eq!(half.segments()[1].dest_row, 64 * 192);
        assert!(cfg.head_split(3).is_err());
        Ok(())
    }

    /// The value of an E4M3 (fn) byte; 0x7E is the largest finite magnitude, 448.
    fn e4m3(code: u8) -> f32 {
        let (e, m) = (i32::from(code >> 3 & 0xF), f32::from(code & 7));
        let magnitude = if e == 0 { m / 8.0 * 2f32.powi(-6) } else { (1.0 + m / 8.0) * 2f32.powi(e - 7) };
        if code & 0x80 != 0 { -magnitude } else { magnitude }
    }

    /// The checkpoint writer's side of a fused `qkv_proj`: per source row, its part
    /// (0 q, 1 k, 2 v), its row in that whole projection and the grid row it was
    /// quantized with. Shard `s` stores `[q_s | k_s | v_s]` under one 128-row grid.
    fn written_rows(layout: &FusedQkvLayout) -> Vec<(usize, usize, usize)> {
        let (parts, per) = ([layout.q, layout.k, layout.v], layout.q + layout.k + layout.v);
        (0..layout.rows()).map(|row| {
            let (shard, offset) = (row / per, row % per);
            let part = usize::from(offset >= layout.q) + usize::from(offset >= layout.q + layout.k);
            let first: usize = parts[..part].iter().sum();
            (part, shard * parts[part] + offset - first, shard * per.div_ceil(128) + offset / 128)
        }).collect()
    }

    /// Every row `MimoLoader::fused_qkv` copies on the GPUs of a `ranks` head split,
    /// with its offsets: `[rank, destination row, source row, grid row]`.
    fn loaded_rows(cfg: &MimoV2Config, attention: MimoAttention, tp: usize, ranks: usize)
        -> Result<(FusedQkvLayout, usize, Vec<[usize; 4]>)> {
        let share = cfg.head_split(ranks)?;
        let layout = FusedQkvLayout::new(&share, attention, tp / ranks)?;
        let (width, segments) = layout.program_segments(&share)?;
        let mut rows = Vec::with_capacity(layout.rows() * ranks);
        for rank in 0..ranks {
            let (source0, scale0) = (rank * layout.rows(), rank * layout.scale_rows());
            for segment in &segments {
                rows.extend((0..segment.rows).map(|r|
                    [rank, segment.dest_row + r, source0 + segment.source_row + r, scale0 + segment.scale_row_of(r)]));
            }
        }
        Ok((layout, width, rows))
    }

    /// Flash TP4 and Pro TP8 (whose keys the program pads to 256 rows).
    fn checkpoint_geometries() -> Result<[(MimoV2Config, usize, Option<usize>); 2]> {
        Ok([(MimoV2Config::from_hf(&crate::plan::testing::mimo_flash_mopd_config())?, 4, None),
            (MimoV2Config::from_hf(&crate::plan::testing::mimo_pro_config())?, 8, Some(256))])
    }

    #[test]
    fn t1_tp_interleaved_rows_land_on_whole_projection_rows_with_their_shard_grid() -> Result<()> {
        // Issue #3's T1 on the real shard shapes and through the rows fused_qkv copies:
        // every checkpoint row is copied once, onto its row of the whole q/k/v
        // projection, with the grid row the writer quantized it with. The per-part
        // grids gave v rows 0..63 of every 3072/192/128 shard grid row 26, not 25.
        for (cfg, tp, padded_key) in checkpoint_geometries()? {
            for attention in [MimoAttention::Full, MimoAttention::Sliding] {
                let written = written_rows(&FusedQkvLayout::new(&cfg, attention, tp)?);
                for ranks in [1, 2] {
                    let (share, width, loaded) = loaded_rows(&cfg, attention, tp, ranks)?;
                    let key_stride = padded_key.unwrap_or(share.k);
                    let (sizes, strides) = ([share.q, share.k, share.v], [share.q, key_stride, share.v]);
                    let base = [0, share.shards * share.q, share.shards * (share.q + key_stride)];
                    assert_eq!(width, base[2] + share.shards * share.v);
                    let mut sources: Vec<usize> = loaded.iter().map(|row| row[2]).collect();
                    sources.sort_unstable();
                    assert!(sources.into_iter().eq(0..written.len()), "every checkpoint row is copied exactly once");
                    for &[rank, dest, source, grid] in &loaded {
                        let (part, whole, block) = written[source];
                        let shard = (whole / sizes[part]).checked_sub(rank * share.shards)
                            .filter(|&shard| shard < share.shards)
                            .expect("a GPU copies only its own checkpoint shards");
                        assert_eq!(dest, base[part] + shard * strides[part] + whole % sizes[part]);
                        assert_eq!(grid, block, "{attention:?} TP{tp} over {ranks} GPU(s): row {} of part {part} in a \
                            shard reads grid row {grid}; the writer quantized it under {block}", whole % sizes[part]);
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn t2_a_poisoned_grid_row_reaches_only_the_rows_quantized_with_it() -> Result<()> {
        // Issue #3's T2 on the real shard shapes. Per-part and whole-shard grids have
        // the same row count there (27, 29), so no pad row exists to poison: each grid
        // row is poisoned in turn, and only the rows the writer quantized with it may
        // change. Under the per-part grids, v rows 0..63 of every 3072/192/128 shard
        // read row 26, so poisoning row 25 missed them and poisoning row 26 reached
        // them (`poisoned_segment_pad_is_never_read` keeps the pad case).
        let span = |rows: &[usize]| rows.first().zip(rows.last())
            .map_or("no rows".to_string(), |(first, last)| format!("{} rows ({first}..={last})", rows.len()));
        for (cfg, tp, _) in checkpoint_geometries()? {
            for attention in [MimoAttention::Full, MimoAttention::Sliding] {
                let full = FusedQkvLayout::new(&cfg, attention, tp)?;
                let mut quantized = vec![Vec::new(); full.scale_rows()];
                for (source, &(_, _, block)) in written_rows(&full).iter().enumerate() {
                    quantized[block].push(source);
                }
                let clean: Vec<f32> = (0..full.scale_rows()).map(|g| 1.0 + g as f32 / 512.0).collect();
                for ranks in [1, 2] {
                    let (_, _, loaded) = loaded_rows(&cfg, attention, tp, ranks)?;
                    // Finite nonzero codes: a row's value moves with the scale it reads.
                    let codes: Vec<f32> = loaded.iter().map(|row| e4m3((row[2] % 0x7E + 1) as u8)).collect();
                    let mut grid = clean.clone();
                    for g in 0..grid.len() {
                        grid[g] = 1e30;
                        let mut changed: Vec<usize> = loaded.iter().zip(&codes)
                            .filter(|&(row, &code)| code * grid[row[3]] != code * clean[row[3]])
                            .map(|(row, _)| row[2]).collect();
                        changed.sort_unstable();
                        assert!(changed == quantized[g], "{attention:?} TP{tp} over {ranks} GPU(s): poisoning grid row \
                            {g} changed {}; the writer quantized {} with it", span(&changed), span(&quantized[g]));
                        grid[g] = clean[g];
                    }
                }
            }
        }
        Ok(())
    }

    /// Issue #3's probe on checkpoint bytes, through the rows `fused_qkv` copies.
    /// Amax/448 scaling leaves a +-448 code in every block, so the rows the loader
    /// reads under one grid row hold one in every (shard, 128-column) cell. Where a
    /// shard's k/v boundary falls inside a block, the first value head's two halves
    /// read different grid rows and still dequantize to the same RMS. Probes every
    /// full layer, the first and last sliding layers and every MTP stage, all
    /// shards, of each pinned snapshot in `CUTEAFD_CHECKPOINT_TEST_HUB` (the hub
    /// `test_mimo_qkv_scales.py` audits), else in the Hugging Face cache.
    #[test]
    #[ignore = "requires a MiMo V2.6 snapshot in CUTEAFD_CHECKPOINT_TEST_HUB or the Hugging Face cache"]
    fn checkpoint_qkv_blocks_match_the_loader_grid() -> Result<()> {
        use std::os::unix::fs::FileExt;
        let hub = std::env::var_os("CUTEAFD_CHECKPOINT_TEST_HUB").map(std::path::PathBuf::from)
            .unwrap_or_else(|| crate::default_hf_home().join("hub"));
        // (model, revision, a full layer whose key rows in the shared k/v block hold no +-448 code)
        let pinned = [("MiMo-V2.6-Flash-MOPD", "2479e2d0029eca9a34cc7e7f55a121925f81908e", Some(47)),
            ("MiMo-V2.6-Flash-RL", "5711b268169967567844e1e560e8a3966da959b1", Some(47)),
            ("MiMo-V2.6-Flash-RL", "3b38d063180c3e4aed9691fdc735f3d10b266ee4", Some(47)),
            ("MiMo-V2.6-Pro-MOPD", "adea8e2c5373181e5a973fa1ecb343cb31af214b", None)];
        let (mut probed, mut failures) = (0, Vec::new());
        for (model, revision, keyless) in pinned {
            let snapshot = hub.join(format!("models--XiaomiMiMo--{model}/snapshots/{revision}"));
            if !snapshot.join("model.safetensors.index.json").is_file() {
                eprintln!("{model}@{}: not in {}", &revision[..8], hub.display());
                continue;
            }
            probed += 1;
            let (cfg, tp) = (MimoV2Config::read(&snapshot)?, checkpoint_tp(&snapshot)?);
            let checkpoint = crate::plan::checkpoint::Checkpoint::open(&snapshot)?;
            let read = |name: &str| -> Result<(Vec<u8>, Vec<usize>)> {
                let at = checkpoint.tensors.binary_search_by(|t| t.meta.name.as_str().cmp(name))
                    .map_err(|_| anyhow::anyhow!("{model}: no tensor {name}"))?;
                let tensor = &checkpoint.tensors[at];
                let mut bytes = vec![0u8; tensor.meta.byte_length as usize];
                std::fs::File::open(snapshot.join(&tensor.shard))?.read_exact_at(&mut bytes, tensor.meta.byte_offset)?;
                Ok((bytes, tensor.meta.shape.clone()))
            };
            let sliding: Vec<usize> = (0..cfg.layers).filter(|&l| cfg.attention[l] == MimoAttention::Sliding).collect();
            let (first, last) = (sliding.first().copied(), sliding.last().copied());
            let tensors: Vec<(String, MimoAttention, Option<usize>)> = (0..cfg.layers)
                .filter(|&l| cfg.attention[l] == MimoAttention::Full || Some(l) == first || Some(l) == last)
                .map(|l| (format!("model.layers.{l}.self_attn.qkv_proj.weight"), cfg.attention[l], Some(l)))
                .chain(checkpoint.tensors.iter().map(|t| t.meta.name.clone())
                    .filter(|name| name.starts_with("model.mtp.") && name.ends_with("self_attn.qkv_proj.weight"))
                    .map(|name| (name, MimoAttention::Sliding, None)))
                .collect();
            let mut separating = 0;
            for (name, attention, layer) in &tensors {
                let ((codes, shape), (scales, scale_shape)) = (read(name)?, read(&format!("{name}_scale_inv"))?);
                let (layout, _, loaded) = loaded_rows(&cfg, *attention, tp, 1)?;
                let blocks = shape[1] / 128;
                ensure!(shape[0] == layout.rows() && scale_shape == [layout.scale_rows(), blocks],
                    "{name}: {shape:?} with {scale_shape:?} scales");
                let grid: Vec<f32> = scales.chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
                // saturated[row * blocks + b]: the row holds a +-448 code in column block b.
                let saturated: Vec<bool> = codes.chunks_exact(128)
                    .map(|cell| cell.iter().any(|&x| x & 0x7F == 0x7E)).collect();
                let (mut held, mut read_under) = (vec![false; grid.len()], vec![0; layout.rows()]);
                for &[_, _, source, row] in &loaded {
                    read_under[source] = row;
                    for b in 0..blocks {
                        held[row * blocks + b] |= saturated[source * blocks + b];
                    }
                }
                let empty = held.iter().filter(|&&h| !h).count();
                if empty > 0 {
                    failures.push(format!("{model} {name}: {empty} (grid row, 128-column) cells hold no +-448 code"));
                }
                let (per, boundary) = (layout.q + layout.k + layout.v, layout.q + layout.k);
                if boundary % 128 == 0 {
                    eprintln!("{model}@{} {name}: k/v boundary on a block edge; {empty} of {} cells without a \
                        +-448 code", &revision[..8], held.len());
                    continue;
                }
                // Shard rows: the key tail [tail, boundary) and value head [boundary, split) share a block.
                let (tail, split) = (boundary / 128 * 128, boundary.div_ceil(128) * 128);
                let head_end = boundary + cfg.v_head_dim;
                let cells: Vec<(usize, usize)> = (0..layout.shards)
                    .flat_map(|s| (0..blocks).map(move |b| (s, b))).collect();
                let occupancy = |rows: std::ops::Range<usize>| cells.iter().filter(|&&(s, b)|
                    rows.clone().any(|r| saturated[(s * per + r) * blocks + b])).count() as f64 / cells.len() as f64;
                let key_tail = occupancy(tail..boundary);
                separating += usize::from(key_tail < 1.0);
                if keyless.is_some() && *layer == keyless && key_tail > 0.0 {
                    failures.push(format!("{model} {name}: key rows {}..{} hold +-448 codes in {key_tail:.3} of cells",
                        tail - layout.q, layout.k));
                }
                let rms = |rows: std::ops::Range<usize>, s: usize, b: usize| {
                    let sum: f64 = rows.clone().map(|r| {
                        let row = s * per + r;
                        let scale = f64::from(grid[read_under[row] * blocks + b]);
                        let cell = &codes[row * shape[1] + b * 128..][..128];
                        cell.iter().map(|&x| (f64::from(e4m3(x)) * scale).powi(2)).sum::<f64>()
                    }).sum();
                    (sum / (rows.len() * 128) as f64).sqrt()
                };
                let mut ratios: Vec<f64> = cells.iter()
                    .map(|&(s, b)| rms(boundary..split, s, b) / rms(split..head_end, s, b)).collect();
                let mut steps: Vec<f64> = cells.iter().map(|&(s, b)| {
                    let row = s * per.div_ceil(128) + boundary / 128;
                    f64::from(grid[row * blocks + b]) / f64::from(grid[(row + 1) * blocks + b])
                }).collect();
                ratios.sort_by(f64::total_cmp);
                steps.sort_by(f64::total_cmp);
                let at = |v: &[f64], percent: usize| v[(v.len() - 1) * percent / 100];
                eprintln!("{model}@{} {name}: +-448 codes in key rows {}..{} {key_tail:.3}, value rows 0..{} {:.3} and \
                    {}..{} {:.3} of cells; value-head RMS ratio 5/50/95% {:.3}/{:.3}/{:.3}; shared/next grid row {:.3}",
                    &revision[..8], tail - layout.q, layout.k, split - boundary, occupancy(boundary..split),
                    split - boundary, cfg.v_head_dim, occupancy(split..head_end), at(&ratios, 5), at(&ratios, 50),
                    at(&ratios, 95), at(&steps, 50));
                if !(0.9..=1.1).contains(&at(&ratios, 50)) {
                    failures.push(format!("{model} {name}: value-head RMS ratio median {:.3}", at(&ratios, 50)));
                }
            }
            if separating == 0 {
                failures.push(format!("{model}: no sampled key tail separates whole-shard from per-part grids"));
            }
        }
        if probed == 0 {
            eprintln!("skipped: no pinned MiMo V2.6 snapshot under {}", hub.display());
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
        Ok(())
    }
}
