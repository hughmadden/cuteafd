//! Real-weight layer-3 attention branch over private staged pages, not a serving cache.
use super::{rope::TargetTables, weights::load_operand};
use crate::shared::memory::device::{Allocation, Device, Stream};
use anyhow::{ensure, Context, Result};
use cuteafd_core::DType;
use cuteafd_ffi::{
    programs::{Program, Programs, Scalar},
    CuteafdDeviceBuffer,
};
use cuteafd_loader::plan::{
    checkpoint::Checkpoint,
    families::step5::coordinator::{CoordinatorContract, RankWeightLayout},
};
use std::ffi::c_void;

const H: usize = 4096;
const Q: usize = 64 * 192;
const KV: usize = 4 * 192;
const TILE: usize = 64;
const CAPACITY: usize = 131072;
const SELECT: usize = 4 * 4104;
const X: usize = 0;
const NORM: usize = 1;
const QRAW: usize = 2;
const KRAW: usize = 3;
const VALUE: usize = 4;
const QUERY: usize = 5;
const KEY: usize = 6;
const LOGITS: usize = 7;
const ATTN: usize = 8;
const GATED: usize = 9;
const DELTA: usize = 10;
const POST: usize = 11;
const POSITIONS: usize = 12;
const REQUESTS: usize = 13;
const TABLE: usize = 14;
const IQRAW: usize = 15;
const IKRAW: usize = 16;
const IZ: usize = 17;
const IQ: usize = 18;
const IK: usize = 19;
const IW: usize = 20;
const IWSCR: usize = 21;
const STARTS: usize = 22;
const COUNTS: usize = 23;
const SUMMARY: usize = 24;
const UNROUNDED: usize = 25;
const SCORES: usize = 26;
const SELECTED: usize = 27;
const NORMALIZED: usize = 28;
const LENGTHS: usize = 29;
const VALUES: usize = 30;
const IDS: usize = 31;

struct Admission {
    rows: usize,
    pages: usize,
    planes: [usize; 32],
    bytes: u64,
}
impl Admission {
    fn new(
        rows: usize,
        weights: &[u64],
        rope: u64,
        available: u64,
        reservations: &[u64],
    ) -> Result<Self> {
        ensure!(
            (1..=8192).contains(&rows),
            "CSA staged row capacity must be 1..=8192"
        );
        let pages = rows.div_ceil(64);
        let regions = rows.div_ceil(8);
        // Scores and top-k scratch are bounded by query tiles, never prefill length.
        let planes = [
            rows * H * 4,
            rows * H * 2,
            rows * Q * 2,
            rows * KV * 2,
            pages * 64 * KV * 2,
            rows * Q * 2,
            pages * 64 * KV * 2,
            rows * 64 * 2,
            rows * Q * 2,
            rows * Q * 2,
            rows * H * 2,
            rows * H * 4,
            rows * 8,
            rows * 4,
            pages * 4,
            rows * H * 2,
            rows * 256 * 2,
            rows * 256 * 2,
            rows * H * 2,
            rows * 256 * 2,
            rows * 16 * 4,
            rows * 16 * 2,
            regions * 4,
            regions * 4,
            CAPACITY * 256,
            regions * 256 * 4,
            TILE * 4 * CAPACITY * 4,
            TILE * SELECT * 4,
            TILE * 4 * CAPACITY * 4,
            TILE * 4 * 4,
            TILE * 4 * 512 * 4,
            TILE * 4 * 512 * 4,
        ]
        .map(|n| n.max(256));
        let bytes = planes
            .iter()
            .map(|&n| n as u64)
            .chain(weights.iter().copied())
            .chain([rope])
            .try_fold(0u64, |a, b| a.checked_add(b))
            .context("CSA admission overflow")?;
        let total = reservations
            .iter()
            .try_fold(bytes, |a, b| a.checked_add(*b))
            .context("CSA reservation overflow")?;
        ensure!(
            total <= available,
            "CSA needs {total} bytes, available {available}"
        );
        Ok(Self {
            rows,
            pages,
            planes,
            bytes,
        })
    }
}

const NORM_POINTERS: &[&str] = &["x", "weight", "bias", "cos", "sin", "positions", "out"];
const DEFS: [(&str, &[&str], &[u8]); 17] = [
    ("step5_hidden_norm", NORM_POINTERS, b"if"),
    ("step5_attn_q", &["x", "w", "out"], b"i"),
    ("step5_attn_kv", &["x", "w", "out"], b"i"),
    ("step5_attn_gate", &["x", "w", "out"], b"i"),
    ("step5_csa_q_norm_rope", NORM_POINTERS, b"if"),
    ("step5_csa_k_norm_rope", NORM_POINTERS, b"if"),
    ("step5_index_q", &["x", "w", "out"], b"i"),
    ("step5_index_kz", &["x", "w", "out"], b"i"),
    ("step5_index_q_norm", NORM_POINTERS, b"if"),
    ("step5_index_k_norm", NORM_POINTERS, b"if"),
    (
        "step5_index_weight",
        &["x", "weight", "out", "scratch"],
        b"i",
    ),
    (
        "step5_region_compress",
        &["keys", "logits", "starts", "counts", "output", "unrounded"],
        b"il",
    ),
    (
        "step5_region_scores",
        &[
            "query",
            "weights",
            "summaries",
            "requests",
            "positions",
            "scores",
        ],
        b"il",
    ),
    (
        "step5_region_selection",
        &[
            "scores",
            "positions",
            "requests",
            "selected",
            "normalized",
            "lengths",
            "values",
            "ids",
        ],
        b"i",
    ),
    (
        "step5_csa_attention",
        &[
            "query",
            "key_cache",
            "value_cache",
            "k_descale",
            "v_descale",
            "block_table",
            "request_ids",
            "selected_positions",
            "query_positions",
            "partial_output",
            "partial_lse",
            "output",
        ],
        b"lllfii",
    ),
    ("step5_head_gate", &["x", "logits", "out"], b"i"),
    ("step5_attn_out", &["x", "w", "out"], b"i"),
];

pub(super) struct CsaLayer<'a> {
    // Drain the complete staged branch before any pointer owner retires.
    stream: Stream<'a>,
    rope: TargetTables<'a>,
    weights: Vec<Allocation<'a>>,
    planes: Vec<Allocation<'a>>,
    programs: Vec<Program<'a>>,
    residual: Program<'a>,
    admission: Admission,
    epsilon: f32,
}
impl<'a> CsaLayer<'a> {
    pub fn load(
        device: Device<'a>,
        programs: &'a Programs<'a>,
        checkpoint: &'a Checkpoint,
        rows: usize,
        available: u64,
        reservations: &[u64],
    ) -> Result<Self> {
        let contract = CoordinatorContract::from_checkpoint(checkpoint)?;
        let layer = &contract.layers[3];
        ensure!(
            layer.sliding.is_none()
                && layer.geometry.query_heads == 64
                && layer.geometry.kv_heads == 4
                && layer.geometry.qk_dim == 192
                && layer.rope.span == 64
                && layer.indexer_rope.is_some_and(|r| r.span == 32),
            "layer-3 CSA geometry"
        );
        let layout = RankWeightLayout::from_checkpoint(checkpoint, 0, 1)?;
        let suffixes = [
            "input_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "self_attn.g_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.sparse_indexer_q.weight",
            "self_attn.sparse_indexer_k.weight",
            "self_attn.sparse_indexer_z.weight",
            "self_attn.sparse_indexer_q_norm.weight",
            "self_attn.sparse_indexer_k_norm.weight",
            "self_attn.sparse_indexer_k_norm.bias",
            "self_attn.sparse_indexer_w.weight",
        ];
        let weights = suffixes
            .iter()
            .map(|s| {
                layout
                    .operands
                    .iter()
                    .find(|w| w.name == format!("model.layers.3.{s}"))
                    .with_context(|| format!("CSA operand {s}"))
            })
            .collect::<Result<Vec<_>>>()?;
        for (w, (dtype, shape)) in weights.iter().zip([
            (DType::F32, vec![H]),
            (DType::Bf16, vec![Q, H]),
            (DType::Bf16, vec![KV, H]),
            (DType::Bf16, vec![KV, H]),
            (DType::F32, vec![192]),
            (DType::F32, vec![192]),
            (DType::Bf16, vec![64, H]),
            (DType::Bf16, vec![H, Q]),
            (DType::Bf16, vec![H, H]),
            (DType::Bf16, vec![256, H]),
            (DType::Bf16, vec![256, H]),
            (DType::Bf16, vec![256]),
            (DType::Bf16, vec![256]),
            (DType::Bf16, vec![256]),
            (DType::Bf16, vec![16, H]),
        ]) {
            ensure!(
                w.dtype == dtype && w.shape == shape,
                "CSA weight shape/dtype: {}",
                w.name
            );
        }
        let admission = Admission::new(
            rows,
            &weights
                .iter()
                .map(|w| w.allocation_bytes)
                .collect::<Vec<_>>(),
            contract.rope_tables.allocation_bytes,
            available,
            reservations,
        )?;
        for (name, pointers, scalars) in DEFS {
            let s = programs.spec(name)?;
            ensure!(
                s.pointers == pointers
                    && s.scalar_kinds == scalars
                    && s.capacity_rows as usize >= rows.min(4096),
                "CSA ABI/capacity {name}"
            );
            let expected: Vec<(&str, usize)> = match name {
                "step5_index_weight" => vec![("scratch", 16 * 2)],
                "step5_region_selection" => vec![
                    ("normalized", 4 * CAPACITY * 4),
                    ("lengths", 4 * 4),
                    ("values", 4 * 512 * 4),
                    ("ids", 4 * 512 * 4),
                ],
                _ => vec![],
            };
            ensure!(
                s.scratch.len() == expected.len(),
                "CSA scratch names {name}"
            );
            for (key, bytes) in expected {
                ensure!(
                    s.scratch.get(key) == Some(&(bytes as u64 * s.capacity_rows as u64)),
                    "CSA scratch extent {name}/{key}"
                );
            }
        }
        let residual_spec = programs.spec("step5_residual_bf16")?;
        ensure!(
            residual_spec.pointers == ["residual", "delta", "out"]
                && residual_spec.scalar_kinds == b"i"
                && residual_spec.capacity_rows as usize >= TILE
                && residual_spec.scratch.is_empty(),
            "CSA residual ABI"
        );
        for (name, width, outputs) in [
            ("step5_attn_q", H, Q),
            ("step5_attn_kv", H, KV),
            ("step5_attn_gate", H, 64),
            ("step5_attn_out", Q, H),
            ("step5_index_q", H, H),
            ("step5_index_kz", H, 256),
        ] {
            let g = &programs.spec(name)?.geometry;
            ensure!(
                g["width"] == width
                    && g["outputs"] == outputs
                    && g["output_dtype"] == "torch.bfloat16"
                    && g["skinny_rows"] == 0,
                "CSA projection geometry {name}"
            );
        }
        for (name, width, heads, rotary, centered, mean, bias) in [
            ("step5_hidden_norm", H, 1, 0, true, false, false),
            ("step5_csa_q_norm_rope", 192, 64, 64, true, false, false),
            ("step5_csa_k_norm_rope", 192, 4, 64, true, false, false),
            ("step5_index_q_norm", 256, 16, 32, true, false, false),
            ("step5_index_k_norm", 256, 1, 32, false, true, true),
        ] {
            let g = &programs.spec(name)?.geometry;
            ensure!(
                g["width"] == width
                    && g["heads"] == heads
                    && g["rotary"] == rotary
                    && g["zero_centered"] == centered
                    && g["subtract_mean"] == mean
                    && g["bias"] == bias
                    && g["epsilon"] == "runtime_float32",
                "CSA norm geometry {name}"
            );
        }
        let g = &programs.spec("step5_index_weight")?.geometry;
        ensure!(
            g["width"] == H && g["outputs"] == 16 && g["prescale"] == 0.5,
            "CSA index weight geometry"
        );
        let g = &programs.spec("step5_region_compress")?.geometry;
        ensure!(
            g["region_tokens"] == 8 && g["proxy_dim"] == 256,
            "CSA compression geometry"
        );
        for name in ["step5_region_scores", "step5_region_selection"] {
            let g = &programs.spec(name)?.geometry;
            ensure!(
                g["groups"] == 4 && g["capacity"] == CAPACITY,
                "CSA region geometry {name}"
            );
        }
        let g = &programs.spec("step5_region_scores")?.geometry;
        ensure!(
            g["proxy_dim"] == 256 && g["heads_per_group"] == 4,
            "CSA score geometry"
        );
        let g = &programs.spec("step5_region_selection")?.geometry;
        ensure!(
            g["topk"] == 512 && g["region_tokens"] == 8,
            "CSA selection geometry"
        );
        let g = &programs.spec("step5_csa_attention")?.geometry;
        ensure!(
            g["query_heads"] == 64
                && g["kv_heads"] == 4
                && g["page_rows"] == 64
                && g["selection_width"] == 4104
                && g["selection_groups"] == 4
                && g["head_dim"] == 192
                && g["key_strides"] == serde_json::json!([49152, 768, 192])
                && g["value_strides"] == serde_json::json!([49152, 768, 192]),
            "CSA selected attention geometry"
        );
        let g = &programs.spec("step5_head_gate")?.geometry;
        ensure!(
            g["heads"] == 64 && g["dim"] == 192,
            "CSA head gate geometry"
        );
        ensure!(
            residual_spec.geometry["width"] == H,
            "CSA residual geometry"
        );
        let resolved = DEFS
            .iter()
            .map(|(name, pointers, _)| device.run(|| programs.program(name, pointers)))
            .collect::<Result<Vec<_>>>()?;
        let residual = device
            .run(|| programs.program("step5_residual_bf16", &["residual", "delta", "out"]))?;
        let rope = TargetTables::load(device, programs, &contract, "step5_", u64::MAX, &[])?;
        let weights = weights
            .into_iter()
            .map(|w| load_operand(device, checkpoint, &layout, w))
            .collect::<Result<Vec<_>>>()?;
        let planes = admission
            .planes
            .iter()
            .map(|&n| Allocation::new(device, n))
            .collect::<Result<Vec<_>>>()?;
        let owner = Self {
            stream: Stream::new(device)?,
            rope,
            weights,
            planes,
            programs: resolved,
            residual,
            admission,
            epsilon: contract.norms.input,
        };
        for (plane, data) in [
            (
                POSITIONS,
                (0..rows as i64)
                    .flat_map(i64::to_le_bytes)
                    .collect::<Vec<_>>(),
            ),
            (REQUESTS, vec![0; rows * 4]),
            (
                TABLE,
                (0..owner.admission.pages as i32)
                    .flat_map(i32::to_le_bytes)
                    .collect(),
            ),
            (
                STARTS,
                (0..rows.div_ceil(8) as i32)
                    .flat_map(|r| (r * 8).to_le_bytes())
                    .collect(),
            ),
            (
                COUNTS,
                (0..rows.div_ceil(8))
                    .flat_map(|_| 8i32.to_le_bytes())
                    .collect(),
            ),
        ] {
            device.run(|| device.library.copy_h2d(owner.planes[plane].buffer, &data))?;
        }
        Ok(owner)
    }
    fn ptr(&self, plane: usize, offset: usize) -> *mut c_void {
        self.planes[plane].buffer.ptr.wrapping_byte_add(offset)
    }
    fn launch(&self, stage: usize, pointers: &[*mut c_void], scalars: &[Scalar]) -> Result<()> {
        self.stream.device.run(|| {
            // SAFETY: callers use admitted planes with bounded row offsets on this stream.
            unsafe { self.programs[stage].launch(pointers, scalars, self.stream.raw) }
        })
    }
    fn norm(
        &self,
        stage: usize,
        input: usize,
        output: usize,
        weight: usize,
        bias: usize,
        start: usize,
        rows: usize,
        width: usize,
        indexer: bool,
    ) -> Result<()> {
        let table = if indexer {
            self.rope.layers[3]
                .indexer
                .as_ref()
                .context("CSA indexer tables")?
        } else {
            &self.rope.layers[3].attention
        };
        let (cos, sin) = table.buffers();
        self.launch(
            stage,
            &[
                self.ptr(input, start * width * 2),
                self.weights[weight].buffer.ptr,
                self.weights[bias].buffer.ptr,
                cos.ptr,
                sin.ptr,
                self.ptr(POSITIONS, start * 8),
                self.ptr(output, start * width * 2),
            ],
            &[Scalar::I32(rows as i32), Scalar::F32(self.epsilon)],
        )
    }
    pub fn enqueue(&self, rows: usize, query_start: usize) -> Result<()> {
        ensure!(
            rows > 0 && rows <= self.admission.rows && query_start < rows,
            "CSA live row range"
        );
        for start in (0..rows).step_by(4096) {
            let count = (rows - start).min(4096);
            let scalar = [Scalar::I32(count as i32)];
            let p = |i| self.ptr(i, start * H * 2);
            let w = |i: usize| self.weights[i].buffer.ptr;
            self.launch(
                0,
                &[
                    self.ptr(X, start * H * 4),
                    w(0),
                    w(0),
                    w(0),
                    w(0),
                    w(0),
                    p(NORM),
                ],
                &[scalar[0], Scalar::F32(self.epsilon)],
            )?;
            for (stage, weight, out, width) in [
                (1, 1, QRAW, Q),
                (2, 2, KRAW, KV),
                (2, 3, VALUE, KV),
                (3, 6, LOGITS, 64),
                (6, 8, IQRAW, H),
                (7, 9, IKRAW, 256),
                (7, 10, IZ, 256),
            ] {
                self.launch(
                    stage,
                    &[p(NORM), w(weight), self.ptr(out, start * width * 2)],
                    &scalar,
                )?;
            }
            self.norm(4, QRAW, QUERY, 4, 4, start, count, Q, false)?;
            self.norm(5, KRAW, KEY, 5, 5, start, count, KV, false)?;
            self.norm(8, IQRAW, IQ, 11, 11, start, count, H, true)?;
            self.norm(9, IKRAW, IK, 12, 13, start, count, 256, true)?;
            self.launch(
                10,
                &[
                    p(NORM),
                    w(14),
                    self.ptr(IW, start * 16 * 4),
                    self.ptr(IWSCR, start * 16 * 2),
                ],
                &scalar,
            )?;
        }
        self.compress(rows)?;
        for start in (query_start..rows).step_by(TILE) {
            let count = (rows - start).min(TILE);
            self.score_select(start, count)?;
            self.attend(start, count)?;
            self.finish(start, count)?;
        }
        Ok(())
    }
    fn compress(&self, rows: usize) -> Result<()> {
        let regions = rows / 8;
        if regions == 0 {
            return Ok(());
        }
        self.launch(
            11,
            &[
                self.ptr(IK, 0),
                self.ptr(IZ, 0),
                self.ptr(STARTS, 0),
                self.ptr(COUNTS, 0),
                self.ptr(SUMMARY, 0),
                self.ptr(UNROUNDED, 0),
            ],
            &[Scalar::I32(regions as i32), Scalar::I64(rows as i64)],
        )
    }
    fn score_select(&self, start: usize, rows: usize) -> Result<()> {
        ensure!(
            rows > 0 && rows <= TILE && start + rows <= self.admission.rows,
            "CSA query tile"
        );
        self.launch(
            12,
            &[
                self.ptr(IQ, start * H * 2),
                self.ptr(IW, start * 16 * 4),
                self.ptr(SUMMARY, 0),
                self.ptr(REQUESTS, start * 4),
                self.ptr(POSITIONS, start * 8),
                self.ptr(SCORES, 0),
            ],
            &[Scalar::I32(rows as i32), Scalar::I64(1)],
        )?;
        self.select(start, rows)
    }
    fn select(&self, start: usize, rows: usize) -> Result<()> {
        self.launch(
            13,
            &[
                self.ptr(SCORES, 0),
                self.ptr(POSITIONS, start * 8),
                self.ptr(REQUESTS, start * 4),
                self.ptr(SELECTED, 0),
                self.ptr(NORMALIZED, 0),
                self.ptr(LENGTHS, 0),
                self.ptr(VALUES, 0),
                self.ptr(IDS, 0),
            ],
            &[Scalar::I32(rows as i32)],
        )
    }
    fn attend(&self, start: usize, rows: usize) -> Result<()> {
        let q = self.ptr(QUERY, start * Q * 2);
        self.launch(
            14,
            &[
                q,
                self.ptr(KEY, 0),
                self.ptr(VALUE, 0),
                q,
                q,
                self.ptr(TABLE, 0),
                self.ptr(REQUESTS, start * 4),
                self.ptr(SELECTED, 0),
                self.ptr(POSITIONS, start * 8),
                q,
                q,
                self.ptr(ATTN, start * Q * 2),
            ],
            &[
                Scalar::I64(self.admission.pages as i64),
                Scalar::I64(1),
                Scalar::I64(self.admission.pages as i64),
                Scalar::F32(192f32.sqrt().recip()),
                Scalar::I32(rows as i32),
                Scalar::I32(1),
            ],
        )
    }
    fn finish(&self, start: usize, rows: usize) -> Result<()> {
        let scalar = [Scalar::I32(rows as i32)];
        self.launch(
            15,
            &[
                self.ptr(ATTN, start * Q * 2),
                self.ptr(LOGITS, start * 64 * 2),
                self.ptr(GATED, start * Q * 2),
            ],
            &scalar,
        )?;
        self.launch(
            16,
            &[
                self.ptr(GATED, start * Q * 2),
                self.weights[7].buffer.ptr,
                self.ptr(DELTA, start * H * 2),
            ],
            &scalar,
        )?;
        self.stream.device.run(|| {
            // SAFETY: this tile's residual and delta are retained by the stream-first owner.
            unsafe {
                self.residual.launch(
                    &[
                        self.ptr(X, start * H * 4),
                        self.ptr(DELTA, start * H * 2),
                        self.ptr(POST, start * H * 4),
                    ],
                    &scalar,
                    self.stream.raw,
                )
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture(root: &std::path::Path, manifest: &serde_json::Value, key: &str) -> Result<Vec<u8>> {
        use sha2::{Digest, Sha256};
        let bytes = std::fs::read(root.join(format!("{key}.bin")))?;
        ensure!(
            manifest["files"][key]["bytes"] == bytes.len()
                && manifest["files"][key]["sha256"].as_str()
                    == Some(format!("{:x}", Sha256::digest(&bytes)).as_str()),
            "CSA fixture hash/length mismatch"
        );
        Ok(bytes)
    }

    fn values(bytes: &[u8], size: usize) -> Result<Vec<f64>> {
        ensure!(bytes.len() % size == 0, "CSA value alignment");
        let result = bytes
            .chunks_exact(size)
            .map(|v| match size {
                2 => f32::from_bits(u32::from(u16::from_le_bytes(v.try_into().unwrap())) << 16)
                    as f64,
                4 => f32::from_le_bytes(v.try_into().unwrap()) as f64,
                8 => f64::from_le_bytes(v.try_into().unwrap()),
                _ => unreachable!(),
            })
            .collect::<Vec<_>>();
        ensure!(result.iter().all(|v| v.is_finite()), "nonfinite CSA output");
        Ok(result)
    }

    fn metrics(
        actual: &[u8],
        expected: &[u8],
        size: usize,
        reference_size: usize,
        width: usize,
    ) -> Result<f64> {
        let a = values(actual, size)?;
        let b = values(expected, reference_size)?;
        ensure!(a.len() == b.len(), "CSA metric length mismatch");
        let error = a.iter().zip(&b).map(|(a, b)| (a - b).powi(2)).sum::<f64>();
        let norm = b.iter().map(|v| v * v).sum::<f64>().max(f64::MIN_POSITIVE);
        let maximum = a
            .iter()
            .zip(&b)
            .map(|(a, b)| (a - b).abs())
            .fold(0f64, f64::max);
        let worst = a
            .chunks(width)
            .zip(b.chunks(width))
            .enumerate()
            .map(|(row, (a, b))| {
                let n = b.iter().map(|v| v * v).sum::<f64>().max(f64::MIN_POSITIVE);
                let e = a.iter().zip(b).map(|(a, b)| (a - b).powi(2)).sum::<f64>();
                (row, (e / n).sqrt())
            })
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap();
        eprintln!(
            "rel_l2={:.12e} max_abs={maximum:.12e} worst_position={} max_position_rel_l2={:.12e}",
            (error / norm).sqrt(),
            worst.0,
            worst.1
        );
        Ok((error / norm).sqrt())
    }

    fn check_rms(actual: &[u8], official: &[u8], fp64: &[u8], width: usize) -> Result<()> {
        let a = values(actual, 2)?;
        let b = values(official, 2)?;
        let r = values(fp64, 8)?;
        ensure!(
            a.len() == b.len() && a.len() == r.len(),
            "CSA RMS oracle extent"
        );
        let error = |a: &[f64], r: &[f64]| {
            a.iter()
                .zip(r)
                .map(|(a, r)| (a - r).powi(2))
                .sum::<f64>()
                .sqrt()
        };
        let native = error(&a, &r);
        let reference = error(&b, &r);
        let flips = actual
            .chunks_exact(2)
            .zip(official.chunks_exact(2))
            .filter_map(|(a, b)| {
                let a = u16::from_le_bytes([a[0], a[1]]);
                let b = u16::from_le_bytes([b[0], b[1]]);
                (a != b).then_some(a.abs_diff(b))
            })
            .collect::<Vec<_>>();
        eprintln!(
            "CSA RMS native/official FP64 error ratio={} flips={}/{} max_ulp={}",
            native / reference,
            flips.len(),
            a.len(),
            flips.iter().max().copied().unwrap_or(0)
        );
        ensure!(
            native <= reference * 1.01,
            "CSA global RMS FP64 accuracy bar"
        );
        let mut worst = 0f64;
        for (row, ((a, b), r)) in a
            .chunks_exact(width)
            .zip(b.chunks_exact(width))
            .zip(r.chunks_exact(width))
            .enumerate()
        {
            let maximum = r.iter().map(|v| v.abs()).fold(0f64, f64::max);
            let bits = (maximum as f32).to_bits() & 0x7f80_0000;
            let ulp = f32::from_bits(bits.max(0x0080_0000)) as f64 / 128.;
            let native = error(a, r);
            let reference = error(b, r);
            worst = worst.max(native / (reference * 1.01 + ulp));
            ensure!(
                native <= reference * 1.01 + ulp,
                "CSA RMS row {row} FP64 accuracy bar"
            );
        }
        eprintln!("CSA RMS worst row error/allowed={worst}");
        Ok(())
    }

    fn read(chain: &CsaLayer<'_>, plane: usize, offset: usize, bytes: usize) -> Result<Vec<u8>> {
        let source = chain.planes[plane].buffer;
        ensure!(offset + bytes <= source.bytes, "CSA read extent");
        let mut output = vec![0; bytes];
        let buffer = CuteafdDeviceBuffer {
            ptr: source.ptr.wrapping_byte_add(offset),
            bytes: source.bytes - offset,
            ..source
        };
        chain
            .stream
            .device
            .run(|| chain.stream.device.library.copy_d2h(&mut output, buffer))?;
        Ok(output)
    }
    fn upload(chain: &CsaLayer<'_>, plane: usize, offset: usize, bytes: &[u8]) -> Result<()> {
        let source = chain.planes[plane].buffer;
        ensure!(offset + bytes.len() <= source.bytes, "CSA upload extent");
        let buffer = CuteafdDeviceBuffer {
            ptr: source.ptr.wrapping_byte_add(offset),
            bytes: source.bytes - offset,
            ..source
        };
        chain
            .stream
            .device
            .run(|| chain.stream.device.library.copy_h2d(buffer, bytes))
    }
    fn score_bytes(chain: &CsaLayer<'_>, queries: usize, columns: usize) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        for row in 0..queries * 4 {
            out.extend(read(chain, SCORES, row * CAPACITY * 4, columns * 4)?);
        }
        Ok(out)
    }
    fn check_scores(actual: &[u8], expected: &[u8]) -> Result<()> {
        ensure!(actual.len() == expected.len(), "CSA score length");
        let mut a = Vec::new();
        let mut b = Vec::new();
        let mut maximum = 0f64;
        for (actual, expected) in actual.chunks_exact(4).zip(expected.chunks_exact(4)) {
            let x = f32::from_le_bytes(actual.try_into().unwrap());
            let y = f32::from_le_bytes(expected.try_into().unwrap());
            if y.is_infinite() {
                ensure!(x == y, "CSA score visibility mismatch");
            } else {
                ensure!(
                    x.is_finite() && y.is_finite(),
                    "CSA nonfinite visible score"
                );
                maximum = maximum.max((x - y).abs() as f64);
                a.extend(actual);
                b.extend(expected);
            }
        }
        if !a.is_empty() {
            let relative = metrics(&a, &b, 4, 4, a.len() / 4)?;
            ensure!(
                relative <= 1e-5 && maximum <= 1e-4,
                "CSA score error: relative={relative}, max={maximum}"
            );
        }
        Ok(())
    }
    #[test]
    fn rms_accuracy_checks_global_and_each_row() -> Result<()> {
        let bf16 = |v: &[f32]| {
            v.iter()
                .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
                .collect::<Vec<_>>()
        };
        let fp64 = |v: &[f64]| v.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
        let reference = fp64(&[1.003, 2.003]);
        assert!(check_rms(&bf16(&[1., 2.]), &bf16(&[1., 2.]), &reference, 1).is_ok());
        assert!(check_rms(&bf16(&[1.0625, 2.]), &bf16(&[1., 2.]), &reference, 1).is_err());
        // A large accurate row must not hide a bad small row in the global norm.
        assert!(check_rms(
            &bf16(&[1.03125, 1000.]),
            &bf16(&[1., 1004.]),
            &fp64(&[1., 1000.]),
            1
        )
        .is_err());
        Ok(())
    }

    #[test]
    fn csa_admission_bounds_query_workspace_independently_of_prefill() -> Result<()> {
        let short = Admission::new(513, &[1000], 3000, u64::MAX, &[])?;
        let long = Admission::new(4105, &[1000], 3000, u64::MAX, &[])?;
        for i in [SCORES, NORMALIZED, SELECTED, LENGTHS, VALUES, IDS] {
            assert_eq!(short.planes[i], long.planes[i]);
        }
        assert_eq!(short.planes[SCORES], 128 << 20);
        assert_eq!(short.planes[NORMALIZED], 128 << 20);
        assert_eq!(long.pages, 65);
        assert_eq!(long.planes[KEY], 65 * 64 * KV * 2);
        assert_eq!(
            short.bytes,
            short.planes.iter().sum::<usize>() as u64 + 4000
        );
        assert!(Admission::new(0, &[], 0, u64::MAX, &[]).is_err());
        assert!(Admission::new(8193, &[], 0, u64::MAX, &[]).is_err());
        assert!(Admission::new(513, &[1000], 3000, short.bytes - 1, &[]).is_err());
        assert!(Admission::new(1, &[], 0, u64::MAX, &[u64::MAX]).is_err());
        Ok(())
    }
    #[test]
    #[ignore = "requires pinned layer-3 CSA fixtures, sealed native library and CUDA"]
    fn native_csa_layer3_boundaries_and_graph_replay() -> Result<()> {
        let root = std::path::PathBuf::from(std::env::var("CUTEAFD_STEP5_CSA_FIXTURE_DIR")?);
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("manifest.json"))?)?;
        ensure!(
            manifest["schema"] == 1
                && manifest["seed"] == 274
                && manifest["layer"] == 3
                && manifest["checkpoint"] == "stepfun-ai/Step-5-Preview-BF16"
                && manifest["revision"] == "7734b7ffac118d70d8398bfa3eb0fab007eb45b1"
                && manifest["rows"] == serde_json::json!([1, 7, 8, 9, 128, 513, 4105])
                && manifest["query_tile"] == TILE,
            "CSA fixture identity"
        );
        // SAFETY: this explicit qualification supplies a sealed matching native library.
        let library =
            unsafe { cuteafd_ffi::NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        let device = crate::shared::memory::device::cache_test_device(&library)?;
        let programs = library
            .programs()?
            .with_manifest(std::path::Path::new(&std::env::var(
                "CUTEAFD_PROGRAM_MANIFEST",
            )?))?;
        let checkpoint =
            Checkpoint::open(std::path::Path::new(&std::env::var("CUTEAFD_SNAPSHOT")?))?;
        ensure!(
            checkpoint.snapshot.file_name().and_then(|s| s.to_str())
                == manifest["revision"].as_str(),
            "CSA snapshot revision"
        );
        let mut chain = CsaLayer::load(device, &programs, &checkpoint, 4105, u64::MAX, &[])?;
        ensure!(
            manifest["epsilon"].as_f64().map(|v| v as f32) == Some(chain.epsilon),
            "CSA epsilon"
        );
        assert!(CsaLayer::load(
            device,
            &programs,
            &checkpoint,
            4105,
            chain.admission.bytes - 1,
            &[]
        )
        .is_err());
        assert!(chain.enqueue(4106, 0).is_err());
        assert!(chain.score_select(0, TILE + 1).is_err());
        for rows in [1usize, 7, 8, 9, 128, 513, 4105] {
            let start = if rows == 4105 { 4104 } else { 0 };
            let count = rows - start;
            let load = |name: &str| fixture(&root, &manifest, &format!("r{rows}_{name}"));
            upload(&chain, X, 0, &load("residual")?)?;
            chain.enqueue(rows, start)?;
            chain.stream.drain()?;
            let mut eager = Vec::new();
            for (name, plane, width, size, offset, length) in [
                ("norm", NORM, H, 2, 0, rows),
                ("qraw", QRAW, Q, 2, 0, rows),
                ("kraw", KRAW, KV, 2, 0, rows),
                ("value", VALUE, KV, 2, 0, rows),
                ("query", QUERY, Q, 2, 0, rows),
                ("key", KEY, KV, 2, 0, rows),
                ("iqraw", IQRAW, H, 2, 0, rows),
                ("ikraw", IKRAW, 256, 2, 0, rows),
                ("iz", IZ, 256, 2, 0, rows),
                ("iq", IQ, H, 2, 0, rows),
                ("ik", IK, 256, 2, 0, rows),
                ("iw", IW, 16, 4, 0, rows),
                ("logits", LOGITS, 64, 2, start, count),
                ("attention", ATTN, Q, 2, start, count),
                ("gated", GATED, Q, 2, start, count),
                ("delta", DELTA, H, 2, start, count),
                ("post", POST, H, 4, start, count),
            ] {
                let actual = read(&chain, plane, offset * width * size, length * width * size)?;
                eprintln!("CSA chained rows={rows} boundary={name}");
                metrics(&actual, &load(name)?, size, size, width)?;
                if name == "norm" {
                    check_rms(&actual, &load(name)?, &load("norm_fp64")?, width)?;
                }
                eager.push((plane, offset * width * size, actual));
            }
            eager.push((SUMMARY, 0, read(&chain, SUMMARY, 0, rows / 8 * 256)?));
            let tile_count = count.min(TILE);
            let last_start = start + (count - 1) / TILE * TILE;
            let last_count = rows - last_start;
            eager.push((
                SELECTED,
                0,
                read(&chain, SELECTED, 0, last_count * SELECT * 4)?,
            ));
            let eager_scores = score_bytes(&chain, last_count, (rows / 8).max(1))?;
            device.run(|| {
                // SAFETY: the stream is idle and the complete owner is pinned below.
                unsafe { library.cuda_graph_begin_capture(chain.stream.raw) }
            })?;
            let enqueue = chain.enqueue(rows, start);
            let raw = device.run(|| {
                // SAFETY: end capture even when the enqueue returns an error.
                unsafe { library.cuda_graph_end_capture(chain.stream.raw) }
            })?;
            // SAFETY: the mutable borrow pins every stream, weight, table and workspace.
            let graph = unsafe {
                crate::shared::decode_graph::GraphOwner::new(&library, device.id, raw, &mut chain)?
            };
            enqueue?;
            let mut guard = graph.pins.stream.cancellation_guard("CSA graph replay");
            device.run(|| {
                // SAFETY: the graph's borrowed owner remains live through the stream drain.
                unsafe { library.cuda_graph_launch(graph.raw, graph.pins.stream.raw) }
            })?;
            graph.pins.stream.drain()?;
            guard.complete = true;
            drop(guard);
            for (plane, offset, expected) in eager {
                ensure!(
                    read(graph.pins, plane, offset, expected.len())? == expected,
                    "CSA eager/replay plane {plane}"
                );
            }
            ensure!(
                score_bytes(graph.pins, last_count, (rows / 8).max(1))? == eager_scores,
                "CSA score eager/replay"
            );
            drop(graph);
            // Every projection consumes the official same-BF16 producer for its FP64 gate.
            for (stage, weight, plane, name, width) in [
                (1, 1, QRAW, "qraw", Q),
                (2, 2, KRAW, "kraw", KV),
                (2, 3, VALUE, "value", KV),
                (3, 6, LOGITS, "logits", 64),
                (6, 8, IQRAW, "iqraw", H),
                (7, 9, IKRAW, "ikraw", 256),
                (7, 10, IZ, "iz", 256),
                (16, 7, DELTA, "delta", H),
            ] {
                let input = if stage == 16 { GATED } else { NORM };
                upload(
                    &chain,
                    input,
                    if stage == 16 { start * Q * 2 } else { 0 },
                    &load(if stage == 16 { "gated" } else { "norm" })?,
                )?;
                chain.launch(
                    stage,
                    &[
                        chain.ptr(
                            input,
                            if stage == 16 {
                                start * Q * 2
                            } else {
                                start * H * 2
                            },
                        ),
                        chain.weights[weight].buffer.ptr,
                        chain.ptr(plane, start * width * 2),
                    ],
                    &[Scalar::I32(count as i32)],
                )?;
                chain.stream.drain()?;
                let actual = read(&chain, plane, start * width * 2, count * width * 2)?;
                let expected = load(name)?;
                let expected = if expected.len() == rows * width * 2 {
                    &expected[start * width * 2..]
                } else {
                    &expected
                };
                let fp64 = load(&format!("{name}_fp64"))?;
                eprintln!("CSA isolated rows={rows} projection={name}");
                let native = metrics(&actual, &fp64, 2, 8, width)?;
                let official = metrics(expected, &fp64, 2, 8, width)?;
                ensure!(native <= official * 1.01, "CSA projection FP64 bar {name}");
            }
            for (stage, input, out, weight, bias, width, indexer, name, source) in [
                (4, QRAW, QUERY, 4, 4, Q, false, "query", "qraw"),
                (5, KRAW, KEY, 5, 5, KV, false, "key", "kraw"),
                (8, IQRAW, IQ, 11, 11, H, true, "iq", "iqraw"),
                (9, IKRAW, IK, 12, 13, 256, true, "ik", "ikraw"),
            ] {
                upload(&chain, input, 0, &load(source)?)?;
                for first in (0..rows).step_by(4096) {
                    chain.norm(
                        stage,
                        input,
                        out,
                        weight,
                        bias,
                        first,
                        (rows - first).min(4096),
                        width,
                        indexer,
                    )?;
                }
                chain.stream.drain()?;
                let actual = read(&chain, out, 0, rows * width * 2)?;
                let expected = load(name)?;
                if actual != expected {
                    eprintln!("CSA isolated norm/RoPE mismatch rows={rows} name={name}");
                    let error = metrics(&actual, &expected, 2, 2, width)?;
                    let mismatches = actual
                        .chunks_exact(2)
                        .zip(expected.chunks_exact(2))
                        .enumerate()
                        .filter(|(_, (a, b))| a != b)
                        .collect::<Vec<_>>();
                    eprintln!(
                        "mismatched_elements={} first={:?}",
                        mismatches.len(),
                        &mismatches[..mismatches.len().min(12)]
                    );
                    ensure!(name == "iq", "CSA isolated norm/RoPE {name}: {error}");
                }
                if name == "iq" {
                    check_rms(&actual, &expected, &load("iq_fp64")?, 256)?;
                }
            }
            upload(&chain, NORM, 0, &load("norm")?)?;
            chain.launch(
                10,
                &[
                    chain.ptr(NORM, start * H * 2),
                    chain.weights[14].buffer.ptr,
                    chain.ptr(IW, start * 16 * 4),
                    chain.ptr(IWSCR, start * 16 * 2),
                ],
                &[Scalar::I32(count as i32)],
            )?;
            chain.stream.drain()?;
            let expected = load("iw")?;
            let fp64 = load("iw_fp64")?;
            let native = metrics(
                &read(&chain, IW, start * 16 * 4, count * 16 * 4)?,
                &fp64,
                4,
                8,
                16,
            )?;
            let official = metrics(&expected[start * 16 * 4..], &fp64, 4, 8, 16)?;
            ensure!(
                native <= official * 1.01,
                "CSA index weight projection FP64 bar"
            );
            upload(&chain, IK, 0, &load("ik")?)?;
            upload(&chain, IZ, 0, &load("iz")?)?;
            // Sentinels prove the final incomplete region is not summarized.
            upload(&chain, SUMMARY, 0, &vec![0x38; (rows / 8 + 1) * 256])?;
            chain.compress(rows)?;
            chain.stream.drain()?;
            ensure!(
                read(&chain, SUMMARY, 0, rows / 8 * 256)? == load("summary")?,
                "CSA isolated E4M3 summary"
            );
            ensure!(
                read(&chain, SUMMARY, rows / 8 * 256, 256)? == vec![0x38; 256],
                "CSA partial region changed"
            );
            upload(&chain, IQ, 0, &load("iq")?)?;
            upload(&chain, IW, 0, &load("iw")?)?;
            let expected_scores = load("scores")?;
            let columns = (rows / 8).max(1);
            let expected_selected = load("selected")?;
            upload(&chain, QUERY, 0, &load("query")?)?;
            upload(&chain, KEY, 0, &load("key")?)?;
            upload(&chain, VALUE, 0, &load("value")?)?;
            for offset in (0..count).step_by(TILE) {
                let live = (count - offset).min(TILE);
                let position = start + offset;
                chain.score_select(position, live)?;
                chain.stream.drain()?;
                let scores = score_bytes(&chain, live, columns)?;
                check_scores(
                    &scores,
                    &expected_scores[offset * 4 * columns * 4..(offset + live) * 4 * columns * 4],
                )?;
                ensure!(
                    read(&chain, SELECTED, 0, live * SELECT * 4)?
                        == expected_selected[offset * SELECT * 4..(offset + live) * SELECT * 4],
                    "CSA selected token identity"
                );
                chain.attend(position, live)?;
            }
            chain.stream.drain()?;
            let actual = read(&chain, ATTN, start * Q * 2, count * Q * 2)?;
            let fp64 = load("attention_fp64")?;
            let rounded = load("attention_bf16_p")?;
            eprintln!("CSA isolated rows={rows} attention native/oracle/official");
            let native = metrics(&actual, &fp64, 2, 8, Q)?;
            let oracle = metrics(&rounded, &fp64, 2, 8, Q)?;
            let official = metrics(&load("attention")?, &fp64, 2, 8, Q)?;
            eprintln!(
                "CSA rows={rows} native_oracle_ratio={} native_official_ratio={}",
                native / oracle,
                native / official
            );
            ensure!(native <= oracle * 1.01, "CSA attention BF16-P FP64 bar");
            let a = values(&actual, 2)?;
            let b = values(&rounded, 2)?;
            let reference = values(&fp64, 8)?;
            for (position, ((a, b), r)) in a
                .chunks(Q)
                .zip(b.chunks(Q))
                .zip(reference.chunks(Q))
                .enumerate()
            {
                let error = |v: &[f64]| v.iter().zip(r).map(|(a, b)| (a - b).powi(2)).sum::<f64>();
                ensure!(
                    error(a) <= error(b) * 1.01f64.powi(2),
                    "CSA attention worst-position gate {position}"
                );
            }
            upload(&chain, ATTN, start * Q * 2, &load("attention")?)?;
            upload(&chain, LOGITS, start * 64 * 2, &load("logits")?)?;
            for offset in (0..count).step_by(TILE) {
                chain.finish(start + offset, (count - offset).min(TILE))?;
            }
            chain.stream.drain()?;
            ensure!(
                read(&chain, GATED, start * Q * 2, count * Q * 2)? == load("gated")?,
                "CSA isolated head gate"
            );
            upload(&chain, DELTA, start * H * 2, &load("delta")?)?;
            device.run(|| {
                // SAFETY: all residual operands belong to this admitted same-device owner.
                unsafe {
                    chain.residual.launch(
                        &[
                            chain.ptr(X, start * H * 4),
                            chain.ptr(DELTA, start * H * 2),
                            chain.ptr(POST, start * H * 4),
                        ],
                        &[Scalar::I32(count as i32)],
                        chain.stream.raw,
                    )
                }
            })?;
            chain.stream.drain()?;
            ensure!(
                read(&chain, POST, start * H * 4, count * H * 4)? == load("post")?,
                "CSA isolated residual"
            );
            eprintln!("CSA rows={rows} queries={count} complete_regions={} passed (query tile {tile_count})",rows/8);
        }
        Ok(())
    }
}
