//! Routed experts kept as the checkpoint's FP8: E4M3 weights with one FP32
//! (or BF16, Qwen 3.8 Flash Next; widened at read) scale per 128x128 block
//! (`weight_scale_inv`), Hugging Face names
//! `model.layers.{l}.mlp.experts.{e}.{gate,up,down}_proj.weight[_scale_inv]`
//! or, for multimodal checkpoints, under `model.language_model.` (MiMo V2
//! Flash; GLM 5.x and GLM 5.3 Flash official FP8). The `fp8` expert family serves them without re-quantization.
//!
//! Tensor-parallel slices split the intermediate `I` over `tp` ranks in whole
//! 128-row blocks: rank `r` owns gate/up rows and down columns
//! `[r * I/tp, (r + 1) * I/tp)`, with the matching block-scale rows (gate/up
//! `[I/128, H/128]`) or columns (down `[H/128, I/128]`).
//!
//! MiMo V2.6 Pro stores its experts as MXFP4 instead (`ExpertFormat::Mxfp4`):
//! `weight` U8 `[N, K/2]` (two E2M1 codes per byte, the even element in the
//! low nibble) and `weight_scale` U8 `[N, K/32]` (UE8M0 exponents). The same
//! programs widen them exactly to BF16 (`fp8-mimop` packages). Their slices
//! split `I` in whole 32-element scale blocks, as evenly as the blocks allow
//! (TP6 of 2048: 352, 352, 352, 352, 320, 320), and every rank stores its
//! slice zero-padded to one 128-aligned width (384): zero gate/up rows give
//! SiLU(0) * 0 = 0 and zero down columns add nothing, so padding is exact.
use crate::catalog::read_safetensors_metadata;
use crate::v41_catalog::RoutedExpertShape;
use anyhow::{ensure, Context, Result};
use cuteafd_core::DType;
use std::collections::{BTreeMap, HashMap};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Fp8Projection {
    Gate,
    Up,
    Down,
}

impl Fp8Projection {
    pub const ALL: [Self; 3] = [Self::Gate, Self::Up, Self::Down];

    fn stem(self) -> &'static str {
        match self {
            Self::Gate => "gate_proj",
            Self::Up => "up_proj",
            Self::Down => "down_proj",
        }
    }
}

/// Storage of the routed expert weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExpertFormat {
    /// E4M3 with FP32 (or BF16) 128x128 block scales (`weight_scale_inv`).
    Fp8Block128,
    /// Packed E2M1 with UE8M0 scales per 32 values along K (`weight_scale`).
    Mxfp4,
}

#[derive(Debug, Clone)]
struct Located {
    shard: String,
    offset: u64,
    bytes: u64,
    dtype: DType,
    shape: Vec<usize>,
}

/// Where every routed FP8 expert tensor lives in the snapshot.
#[derive(Debug)]
pub struct Fp8ExpertTensors {
    snapshot: PathBuf,
    shape: RoutedExpertShape,
    /// `model.` or `model.language_model.`: where the decoder layers live.
    prefix: String,
    /// Whether draft (MTP) experts live under `mtp.layers.{s}.` (Qwen 3.8 Flash
    /// Next) rather than as decoder layers past the backbone: layer
    /// `shape.layers + s` names them.
    mtp_layers: bool,
    format: ExpertFormat,
    tensors: HashMap<String, Located>,
}

impl Fp8ExpertTensors {
    pub fn name(&self, layer: usize, expert: usize, projection: Fp8Projection) -> String {
        if self.mtp_layers && layer >= self.shape.layers {
            let stage = layer - self.shape.layers;
            return format!("mtp.layers.{stage}.mlp.experts.{expert}.{}.weight", projection.stem());
        }
        format!("{}layers.{layer}.mlp.experts.{expert}.{}.weight", self.prefix, projection.stem())
    }

    /// Reads the index and every shard header holding routed experts; checks
    /// that each expert tensor present is E4M3 with an FP32 128x128 grid, or
    /// MXFP4 (packed E2M1 U8 with UE8M0 per-32 scales).
    pub fn read(snapshot: &Path, shape: RoutedExpertShape) -> Result<Self> {
        let index = crate::v41_exl3::read_json(&snapshot.join("model.safetensors.index.json"), 64 * 1024 * 1024)?;
        let weight_map: BTreeMap<String, String> = serde_json::from_value(
            index.get("weight_map").cloned().context("index has no weight_map")?)?;
        let prefix = if weight_map.keys().any(|name| name.starts_with("model.language_model.layers.")) {
            "model.language_model."
        } else {
            "model."
        };
        let mtp_layers = weight_map.keys().any(|name| name.starts_with("mtp.layers.") && name.contains(".mlp.experts."));
        let routed = |name: &str| ((name.starts_with(prefix) && name[prefix.len()..].starts_with("layers."))
            || (mtp_layers && name.starts_with("mtp.layers."))) && name.contains(".mlp.experts.");
        let shards: std::collections::BTreeSet<&String> =
            weight_map.iter().filter(|(name, _)| routed(name)).map(|(_, shard)| shard).collect();
        let mut tensors = HashMap::new();
        for shard in shards {
            for meta in read_safetensors_metadata(&snapshot.join(shard))
                .with_context(|| format!("reading {shard} headers"))?
            {
                if routed(&meta.name) && weight_map.get(&meta.name) == Some(shard) {
                    tensors.insert(meta.name.clone(), Located {
                        shard: shard.clone(),
                        offset: meta.byte_offset,
                        bytes: meta.byte_length,
                        dtype: meta.dtype,
                        shape: meta.shape,
                    });
                }
            }
        }
        let mut catalog = Self { snapshot: snapshot.to_path_buf(), shape, prefix: prefix.to_string(), mtp_layers,
            format: ExpertFormat::Fp8Block128, tensors };
        let first = catalog.name(shape.first_layer, 0, Fp8Projection::Gate);
        if matches!(catalog.located(&first)?.dtype, DType::U8 | DType::I8) {
            catalog.format = ExpertFormat::Mxfp4;
        }
        // One expert of the first routed layer fixes the format contract.
        for projection in Fp8Projection::ALL {
            catalog.check(shape.first_layer, 0, projection)?;
        }
        Ok(catalog)
    }

    pub fn shape(&self) -> &RoutedExpertShape {
        &self.shape
    }

    pub fn format(&self) -> ExpertFormat {
        self.format
    }

    /// Whether `layer` has routed FP8 experts in this snapshot.
    pub fn has_layer(&self, layer: usize) -> bool {
        self.tensors.contains_key(&self.name(layer, 0, Fp8Projection::Gate))
    }

    fn dims(&self, projection: Fp8Projection) -> (usize, usize) {
        let (h, i) = (self.shape.hidden, self.shape.intermediate);
        if projection == Fp8Projection::Down { (h, i) } else { (i, h) }
    }

    fn located(&self, name: &str) -> Result<&Located> {
        self.tensors.get(name).with_context(|| format!("checkpoint has no routed expert tensor {name}"))
    }

    fn check(&self, layer: usize, expert: usize, projection: Fp8Projection) -> Result<()> {
        let name = self.name(layer, expert, projection);
        let (rows, cols) = self.dims(projection);
        let weight = self.located(&name)?;
        if self.format == ExpertFormat::Mxfp4 {
            ensure!(matches!(weight.dtype, DType::U8 | DType::I8) && weight.shape == [rows, cols / 2] && cols % 32 == 0,
                "{name}: expected packed E2M1 U8 [{rows}, {}], found {:?} {:?}", cols / 2, weight.dtype, weight.shape);
            let scale = self.located(&format!("{name}_scale"))?;
            ensure!(matches!(scale.dtype, DType::U8 | DType::F8E8M0) && scale.shape == [rows, cols / 32],
                "{name}_scale: expected UE8M0 [{rows}, {}], found {:?} {:?}", cols / 32, scale.dtype, scale.shape);
            return Ok(());
        }
        ensure!(weight.dtype == DType::F8E4M3 && weight.shape == [rows, cols],
            "{name}: expected E4M3 [{rows}, {cols}], found {:?} {:?}", weight.dtype, weight.shape);
        let scale = self.located(&format!("{name}_scale_inv"))?;
        ensure!(matches!(scale.dtype, DType::F32 | DType::Bf16) && scale.shape == [rows.div_ceil(128), cols.div_ceil(128)],
            "{name}_scale_inv: expected FP32 or BF16 [{}, {}] 128x128 block scales, found {:?} {:?}",
            rows.div_ceil(128), cols.div_ceil(128), scale.dtype, scale.shape);
        Ok(())
    }

    /// The stored intermediate slice width of every rank of `tp`: `I / tp` in
    /// whole 128-row blocks (FP8), or the widest MXFP4 rank range padded to 128.
    pub fn slice(&self, tp: usize) -> Result<usize> {
        let i = self.shape.intermediate;
        if self.format == ExpertFormat::Mxfp4 {
            ensure!(tp > 0 && i % 32 == 0 && i / 32 >= tp, "intermediate {i} does not split over {tp} ranks");
            return Ok((i / 32).div_ceil(tp) * 32).map(|widest| widest.div_ceil(128) * 128);
        }
        ensure!(tp > 0 && i % (128 * tp) == 0, "intermediate {i} does not split into {tp} whole 128-row blocks");
        Ok(i / tp)
    }

    /// The intermediate rows `[first, first + len)` rank `rank` of `tp` computes
    /// (the rest of its stored slice is zero padding).
    pub fn rank_range(&self, tp: usize, rank: usize) -> Result<(usize, usize)> {
        ensure!(rank < tp, "rank {rank} of TP{tp}");
        let slice = self.slice(tp)?;
        if self.format == ExpertFormat::Fp8Block128 {
            return Ok((rank * slice, slice));
        }
        let (blocks, tp_blocks) = (self.shape.intermediate / 32, tp);
        let (base, extra) = (blocks / tp_blocks, blocks % tp_blocks);
        let first = rank * base + rank.min(extra);
        Ok((first * 32, (base + usize::from(rank < extra)) * 32))
    }

    /// Bytes of one expert projection's slice: (weight, scales) as stored
    /// (E4M3 + FP32 128x128 grid, or packed E2M1 + UE8M0 per 32).
    pub fn slice_bytes(&self, projection: Fp8Projection, tp: usize) -> Result<(usize, usize)> {
        let (rows, cols) = self.dims(projection);
        let slice = self.slice(tp)?;
        let (rows, cols) = if projection == Fp8Projection::Down { (rows, slice) } else { (slice, cols) };
        if self.format == ExpertFormat::Mxfp4 {
            return Ok((rows * cols / 2, rows * cols / 32));
        }
        Ok((rows * cols, rows.div_ceil(128) * cols.div_ceil(128) * 4))
    }

    /// `read_slice` for MXFP4: rank rows `[first, first + len)` of gate/up
    /// (rows) or down (K columns), zero-padded to the stored slice width.
    #[allow(clippy::too_many_arguments)]
    fn read_mxfp4_slice(&self, name: &str, projection: Fp8Projection, tp: usize, rank: usize, weight: &mut [u8],
        scale: &mut [u8], staging: &mut Vec<u8>) -> Result<()> {
        let (rows, cols) = self.dims(projection);
        let slice = self.slice(tp)?;
        let (first, len) = self.rank_range(tp, rank)?;
        let w = self.located(name)?;
        let s = self.located(&format!("{name}_scale"))?;
        let open = |shard: &str| std::fs::File::open(self.snapshot.join(shard));
        let (w_file, s_file) = (open(&w.shard)?, open(&s.shard)?);
        weight.fill(0);
        scale.fill(0);
        if projection == Fp8Projection::Down {
            // K columns [first, first + len) of every row, into slice-wide rows.
            for (located, file, out, per) in [(w, &w_file, &mut *weight, 2usize), (s, &s_file, &mut *scale, 32)] {
                staging.resize(located.bytes as usize, 0);
                file.read_exact_at(staging, located.offset).with_context(|| format!("reading {name}"))?;
                let (row_in, row_out, take) = (cols / per, slice / per, len / per);
                for (row, out) in out.chunks_exact_mut(row_out).enumerate().take(rows) {
                    out[..take].copy_from_slice(&staging[row * row_in + first / per..][..take]);
                }
            }
        } else {
            let (w_row, s_row) = (cols / 2, cols / 32);
            w_file.read_exact_at(&mut weight[..len * w_row], w.offset + (first * w_row) as u64)
                .with_context(|| format!("reading {name}"))?;
            s_file.read_exact_at(&mut scale[..len * s_row], s.offset + (first * s_row) as u64)
                .with_context(|| format!("reading {name}_scale"))?;
        }
        Ok(())
    }

    /// Reads rank `rank`'s slice of one expert projection: the E4M3 weight
    /// (gate/up `[I/tp, H]`, down `[H, I/tp]`) and its FP32 block scales.
    /// `staging` is reused scratch for the down projection's column window.
    #[allow(clippy::too_many_arguments)]
    pub fn read_slice(&self, layer: usize, expert: usize, projection: Fp8Projection, tp: usize, rank: usize,
        weight: &mut [u8], scale: &mut [u8], staging: &mut Vec<u8>) -> Result<()> {
        ensure!(rank < tp, "rank {rank} of TP{tp}");
        self.check(layer, expert, projection)?;
        let name = self.name(layer, expert, projection);
        let (rows, cols) = self.dims(projection);
        let slice = self.slice(tp)?;
        let (weight_bytes, scale_bytes) = self.slice_bytes(projection, tp)?;
        ensure!(weight.len() == weight_bytes && scale.len() == scale_bytes, "{name}: slice buffers of the wrong size");
        if self.format == ExpertFormat::Mxfp4 {
            return self.read_mxfp4_slice(&name, projection, tp, rank, weight, scale, staging);
        }
        let w = self.located(&name)?;
        let s = self.located(&format!("{name}_scale_inv"))?;
        let open = |shard: &str| std::fs::File::open(self.snapshot.join(shard));
        let (w_file, s_file) = (open(&w.shard)?, open(&s.shard)?);
        let (scale_rows, scale_cols) = (rows.div_ceil(128), cols.div_ceil(128));
        // The whole block-scale grid (at most a few hundred entries), widened to
        // FP32: Qwen 3.8 Flash Next stores BF16 scales, which FP32 holds exactly.
        let mut raw = vec![0u8; s.bytes as usize];
        s_file.read_exact_at(&mut raw, s.offset)?;
        let grid: Vec<u8> = match s.dtype {
            DType::Bf16 => raw.chunks_exact(2)
                .flat_map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16).to_le_bytes())
                .collect(),
            _ => raw,
        };
        ensure!(grid.len() == scale_rows * scale_cols * 4, "{name}_scale_inv: unexpected grid size");
        if projection == Fp8Projection::Down {
            // Column window [rank*slice, +slice) of every row: read the whole
            // tensor once and gather (down is a third of an expert's bytes).
            staging.resize(w.bytes as usize, 0);
            w_file.read_exact_at(staging, w.offset).with_context(|| format!("reading {name}"))?;
            for (row, out) in weight.chunks_exact_mut(slice).enumerate() {
                out.copy_from_slice(&staging[row * cols + rank * slice..][..slice]);
            }
            let (window, first) = (slice / 128 * 4, rank * slice / 128 * 4);
            for (row, out) in scale.chunks_exact_mut(window).enumerate().take(scale_rows) {
                out.copy_from_slice(&grid[row * scale_cols * 4 + first..][..window]);
            }
        } else {
            w_file.read_exact_at(weight, w.offset + (rank * slice * cols) as u64)
                .with_context(|| format!("reading {name}"))?;
            let first = rank * slice / 128 * scale_cols * 4;
            scale.copy_from_slice(&grid[first..first + scale.len()]);
        }
        Ok(())
    }
}
