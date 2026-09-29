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
    /// that each expert tensor present is E4M3 with an FP32 128x128 grid.
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
        let catalog = Self { snapshot: snapshot.to_path_buf(), shape, prefix: prefix.to_string(), mtp_layers, tensors };
        // One expert of the first routed layer fixes the format contract.
        for projection in Fp8Projection::ALL {
            catalog.check(shape.first_layer, 0, projection)?;
        }
        Ok(catalog)
    }

    pub fn shape(&self) -> &RoutedExpertShape {
        &self.shape
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
        ensure!(weight.dtype == DType::F8E4M3 && weight.shape == [rows, cols],
            "{name}: expected E4M3 [{rows}, {cols}], found {:?} {:?}", weight.dtype, weight.shape);
        let scale = self.located(&format!("{name}_scale_inv"))?;
        ensure!(matches!(scale.dtype, DType::F32 | DType::Bf16) && scale.shape == [rows.div_ceil(128), cols.div_ceil(128)],
            "{name}_scale_inv: expected FP32 or BF16 [{}, {}] 128x128 block scales, found {:?} {:?}",
            rows.div_ceil(128), cols.div_ceil(128), scale.dtype, scale.shape);
        Ok(())
    }

    /// The intermediate slice of TP rank `rank` of `tp` (whole 128-row blocks).
    pub fn slice(&self, tp: usize) -> Result<usize> {
        let i = self.shape.intermediate;
        ensure!(tp > 0 && i % (128 * tp) == 0, "intermediate {i} does not split into {tp} whole 128-row blocks");
        Ok(i / tp)
    }

    /// Bytes of one expert projection's slice: (E4M3 weight, FP32 scales).
    pub fn slice_bytes(&self, projection: Fp8Projection, tp: usize) -> Result<(usize, usize)> {
        let (rows, cols) = self.dims(projection);
        let slice = self.slice(tp)?;
        let (rows, cols) = if projection == Fp8Projection::Down { (rows, slice) } else { (slice, cols) };
        Ok((rows * cols, rows.div_ceil(128) * cols.div_ceil(128) * 4))
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
