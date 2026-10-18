//! Single-request layer-0 prefill composition. No committed ring is published.
use super::{
    ffn::{DenseFfn, DensePlan},
    rope::TargetTables,
    weights::load_operand,
};
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

const H: usize = 4096;
const Q: usize = 64 * 192;
const KV: usize = 4 * 192;
const X: usize = 0;
const NORM: usize = 1;
const QRAW: usize = 2;
const KRAW: usize = 3;
const VPAGES: usize = 4;
const QUERY: usize = 5;
const KPAGES: usize = 6;
const LOGITS: usize = 7;
const ATTN: usize = 8;
const GATED: usize = 9;
const DELTA: usize = 10;
const POSITIONS: usize = 11;
const REQUESTS: usize = 12;
const TABLE: usize = 13;
const SELECTED: usize = 14;

struct Admission {
    rows: usize,
    pages: usize,
    planes: [usize; 15],
    bytes: u64,
}
impl Admission {
    fn new(
        rows: usize,
        weights: &[u64],
        dense: u64,
        rope: u64,
        available: u64,
        reservations: &[u64],
    ) -> Result<Self> {
        ensure!(
            (1..=4096).contains(&rows),
            "SWA row capacity must be 1..=4096"
        );
        let pages = rows.div_ceil(64);
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
            rows * 8,
            rows * 4,
            pages * 4,
            rows * 512 * 4,
        ]
        .map(|n| n.max(256));
        let bytes = planes
            .iter()
            .map(|&n| n as u64)
            .chain(weights.iter().copied())
            .chain([dense, rope])
            .try_fold(0u64, |a, b| a.checked_add(b))
            .context("SWA admission overflow")?;
        let total = reservations
            .iter()
            .try_fold(bytes, |a, b| a.checked_add(*b))
            .context("SWA reservation overflow")?;
        ensure!(
            total <= available,
            "SWA layer needs {total} bytes, available {available}"
        );
        Ok(Self {
            rows,
            pages,
            planes,
            bytes,
        })
    }
}

const DEFS: [(&str, &[&str], &[u8]); 11] = [
    (
        "step5_hidden_norm",
        &["x", "weight", "bias", "cos", "sin", "positions", "out"],
        b"if",
    ),
    ("step5_attn_q", &["x", "w", "out"], b"i"),
    ("step5_attn_kv", &["x", "w", "out"], b"i"),
    ("step5_attn_gate", &["x", "w", "out"], b"i"),
    (
        "step5_swa_q_norm_rope",
        &["x", "weight", "bias", "cos", "sin", "positions", "out"],
        b"if",
    ),
    (
        "step5_swa_k_norm_rope",
        &["x", "weight", "bias", "cos", "sin", "positions", "out"],
        b"if",
    ),
    (
        "step5_sliding_positions",
        &["positions", "requests", "selected"],
        b"i",
    ),
    (
        "step5_swa_attention",
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
        b"lll fii",
    ),
    ("step5_head_gate", &["x", "logits", "out"], b"i"),
    ("step5_attn_out", &["x", "w", "out"], b"i"),
    ("step5_residual_bf16", &["residual", "delta", "out"], b"i"),
];

pub(super) struct SwaLayer<'a> {
    // This stream drains before every composed owner, including the FFN.
    stream: Stream<'a>,
    ffn: DenseFfn<'a>,
    rope: TargetTables<'a>,
    weights: Vec<Allocation<'a>>,
    planes: Vec<Allocation<'a>>,
    programs: Vec<Program<'a>>,
    admission: Admission,
    epsilon: f32,
}

impl<'a> SwaLayer<'a> {
    pub fn load(
        device: Device<'a>,
        programs: &'a Programs<'a>,
        checkpoint: &'a Checkpoint,
        rows: usize,
        available: u64,
        reservations: &[u64],
    ) -> Result<Self> {
        let contract = CoordinatorContract::from_checkpoint(checkpoint)?;
        let layer = &contract.layers[0];
        ensure!(
            layer.sliding.is_some_and(|w| w.window == 512)
                && layer.geometry.query_heads == 64
                && layer.geometry.kv_heads == 4
                && layer.geometry.qk_dim == 192
                && layer.rope.span == 192,
            "layer-0 SWA geometry mismatch"
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
        ];
        let weights = suffixes
            .iter()
            .map(|suffix| {
                layout
                    .operands
                    .iter()
                    .find(|w| w.name == format!("model.layers.0.{suffix}"))
                    .context("SWA operand missing")
            })
            .collect::<Result<Vec<_>>>()?;
        for (weight, (dtype, shape)) in weights.iter().zip([
            (DType::F32, vec![H]),
            (DType::Bf16, vec![Q, H]),
            (DType::Bf16, vec![KV, H]),
            (DType::Bf16, vec![KV, H]),
            (DType::F32, vec![192]),
            (DType::F32, vec![192]),
            (DType::Bf16, vec![64, H]),
            (DType::Bf16, vec![H, Q]),
        ]) {
            ensure!(
                weight.dtype == dtype && weight.shape == shape,
                "SWA weight shape/dtype"
            );
        }
        let dense = DensePlan::prepare(programs, checkpoint, 0, rows, u64::MAX, &[])?;
        let admission = Admission::new(
            rows,
            &weights
                .iter()
                .map(|w| w.allocation_bytes)
                .collect::<Vec<_>>(),
            dense.bytes(),
            contract.rope_tables.allocation_bytes,
            available,
            reservations,
        )?;
        for (name, pointers, scalars) in DEFS {
            let spec = programs.spec(name)?;
            let scalars = scalars
                .iter()
                .copied()
                .filter(|c| *c != b' ')
                .collect::<Vec<_>>();
            ensure!(
                spec.pointers == pointers
                    && spec.scalar_kinds == scalars
                    && spec.capacity_rows as usize >= rows
                    && spec.scratch.is_empty(),
                "{name}: SWA ABI/capacity mismatch"
            );
        }
        for (name, width, outputs) in [
            ("step5_attn_q", H, Q),
            ("step5_attn_kv", H, KV),
            ("step5_attn_gate", H, 64),
            ("step5_attn_out", Q, H),
        ] {
            let g = &programs.spec(name)?.geometry;
            ensure!(
                g["width"] == width
                    && g["outputs"] == outputs
                    && g["output_dtype"] == "torch.bfloat16"
                    && g["skinny_rows"] == 0,
                "{name}: SWA projection geometry mismatch"
            );
        }
        for (name, heads) in [("step5_swa_q_norm_rope", 64), ("step5_swa_k_norm_rope", 4)] {
            let g = &programs.spec(name)?.geometry;
            ensure!(
                g["width"] == 192
                    && g["heads"] == heads
                    && g["rotary"] == 192
                    && g["zero_centered"] == true
                    && g["bias"] == false
                    && g["subtract_mean"] == false
                    && g["epsilon"] == "runtime_float32",
                "SWA Q/K norm geometry mismatch"
            );
        }
        let g = &programs.spec("step5_swa_attention")?.geometry;
        ensure!(
            g["query_heads"] == 64
                && g["kv_heads"] == 4
                && g["page_rows"] == 64
                && g["selection_width"] == 512
                && g["selection_groups"] == 1
                && g["head_dim"] == 192
                && g["key_strides"] == serde_json::json!([49152, 768, 192])
                && g["value_strides"] == serde_json::json!([49152, 768, 192]),
            "SWA attention geometry"
        );
        ensure!(
            programs.spec("step5_sliding_positions")?.geometry["window"] == 512
                && programs.spec("step5_head_gate")?.geometry["heads"] == 64
                && programs.spec("step5_head_gate")?.geometry["dim"] == 192,
            "SWA pointwise geometry"
        );
        // All composed contracts and aggregate bytes are checked before the first allocation.
        let resolved = DEFS
            .iter()
            .map(|(name, pointers, _)| device.run(|| programs.program(name, pointers)))
            .collect::<Result<Vec<_>>>()?;
        let rope = TargetTables::load(device, programs, &contract, "step5_", u64::MAX, &[])?;
        let ffn = DenseFfn::load_plan(device, programs, checkpoint, dense)?;
        let weights = weights
            .into_iter()
            .map(|w| load_operand(device, checkpoint, &layout, w))
            .collect::<Result<Vec<_>>>()?;
        let planes = admission
            .planes
            .iter()
            .map(|&bytes| Allocation::new(device, bytes))
            .collect::<Result<Vec<_>>>()?;
        let stream = Stream::new(device)?;
        let owner = Self {
            stream,
            ffn,
            rope,
            weights,
            planes,
            programs: resolved,
            admission,
            epsilon: contract.norms.input,
        };
        // Immutable single-request metadata is initialized once, never on replay.
        for (plane, bytes) in [
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
        ] {
            device.run(|| device.library.copy_h2d(owner.planes[plane].buffer, &bytes))?;
        }
        Ok(owner)
    }

    fn launch(
        &self,
        stage: usize,
        pointers: &[*mut std::ffi::c_void],
        scalars: &[Scalar],
    ) -> Result<()> {
        self.stream.device.run(|| {
            // SAFETY: the private staged chain supplies admitted same-device planes;
            // its stream is the first field and drains before all referenced owners.
            unsafe { self.programs[stage].launch(pointers, scalars, self.stream.raw) }
        })
    }

    pub fn enqueue(&self, rows: usize) -> Result<()> {
        ensure!(
            rows > 0 && rows <= self.admission.rows,
            "SWA rows exceed capacity"
        );
        let p = |i: usize| self.planes[i].buffer.ptr;
        let w = |i: usize| self.weights[i].buffer.ptr;
        let scalars = [Scalar::I32(rows as i32)];
        let norm = [Scalar::I32(rows as i32), Scalar::F32(self.epsilon)];
        let (cos, sin) = self.rope.layers[0].attention.buffers();
        self.launch(0, &[p(X), w(0), w(0), w(0), w(0), w(0), p(NORM)], &norm)?;
        self.launch(1, &[p(NORM), w(1), p(QRAW)], &scalars)?;
        self.launch(2, &[p(NORM), w(2), p(KRAW)], &scalars)?;
        self.launch(2, &[p(NORM), w(3), p(VPAGES)], &scalars)?;
        self.launch(3, &[p(NORM), w(6), p(LOGITS)], &scalars)?;
        self.launch(
            4,
            &[
                p(QRAW),
                w(4),
                w(4),
                cos.ptr,
                sin.ptr,
                p(POSITIONS),
                p(QUERY),
            ],
            &norm,
        )?;
        self.launch(
            5,
            &[
                p(KRAW),
                w(5),
                w(5),
                cos.ptr,
                sin.ptr,
                p(POSITIONS),
                p(KPAGES),
            ],
            &norm,
        )?;
        self.launch(6, &[p(POSITIONS), p(REQUESTS), p(SELECTED)], &scalars)?;
        self.attend(rows)?;
        self.launch(8, &[p(ATTN), p(LOGITS), p(GATED)], &scalars)?;
        self.launch(9, &[p(GATED), w(7), p(DELTA)], &scalars)?;
        self.launch(10, &[p(X), p(DELTA), self.ffn.input().ptr], &scalars)?;
        self.ffn.enqueue_on(rows, &self.stream)
    }

    fn attend(&self, rows: usize) -> Result<()> {
        let p = |i: usize| self.planes[i].buffer.ptr;
        self.launch(
            7,
            &[
                p(QUERY),
                p(KPAGES),
                p(VPAGES),
                p(QUERY),
                p(QUERY),
                p(TABLE),
                p(REQUESTS),
                p(SELECTED),
                p(POSITIONS),
                p(QUERY),
                p(QUERY),
                p(ATTN),
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

    fn input(&self) -> CuteafdDeviceBuffer {
        self.planes[X].buffer
    }
    fn output(&self) -> CuteafdDeviceBuffer {
        self.ffn.output()
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
            "SWA fixture hash/length mismatch"
        );
        Ok(bytes)
    }

    fn values(bytes: &[u8], size: usize) -> Result<Vec<f64>> {
        ensure!(bytes.len() % size == 0, "SWA value alignment");
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
        ensure!(result.iter().all(|v| v.is_finite()), "nonfinite SWA output");
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
        ensure!(a.len() == b.len(), "SWA metric length mismatch");
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

    fn read(chain: &SwaLayer<'_>, buffer: CuteafdDeviceBuffer, bytes: usize) -> Result<Vec<u8>> {
        let mut output = vec![0; bytes];
        chain
            .stream
            .device
            .run(|| chain.stream.device.library.copy_d2h(&mut output, buffer))?;
        Ok(output)
    }

    #[test]
    #[ignore = "requires pinned layer-0 fixtures, sealed native library and CUDA"]
    fn native_swa_layer0_boundaries_and_graph_replay() -> Result<()> {
        let root = std::path::PathBuf::from(std::env::var("CUTEAFD_STEP5_SWA_FIXTURE_DIR")?);
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("manifest.json"))?)?;
        ensure!(
            manifest["schema"] == 1
                && manifest["seed"] == 274
                && manifest["layer"] == 0
                && manifest["checkpoint"] == "stepfun-ai/Step-5-Preview-BF16"
                && manifest["revision"] == "7734b7ffac118d70d8398bfa3eb0fab007eb45b1"
                && manifest["rows"] == serde_json::json!([1, 7, 128, 513]),
            "SWA fixture identity"
        );
        // SAFETY: the explicit GPU test supplies a sealed native library.
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
            "SWA snapshot revision mismatch"
        );
        let mut chain = SwaLayer::load(device, &programs, &checkpoint, 513, u64::MAX, &[])?;
        ensure!(
            manifest["epsilon"].as_f64().map(|v| v as f32) == Some(chain.epsilon),
            "SWA epsilon mismatch"
        );
        assert!(SwaLayer::load(
            device,
            &programs,
            &checkpoint,
            513,
            chain.admission.bytes - 1,
            &[]
        )
        .is_err());
        assert!(chain.enqueue(514).is_err());
        let boundaries = [
            ("norm", NORM, H, 2),
            ("qraw", QRAW, Q, 2),
            ("kraw", KRAW, KV, 2),
            ("value", VPAGES, KV, 2),
            ("query", QUERY, Q, 2),
            ("key", KPAGES, KV, 2),
            ("logits", LOGITS, 64, 2),
            ("attention", ATTN, Q, 2),
            ("gated", GATED, Q, 2),
            ("delta", DELTA, H, 2),
        ];
        for rows in [1, 7, 128, 513] {
            let load = |name: &str| fixture(&root, &manifest, &format!("r{rows}_{name}"));
            device.run(|| library.copy_h2d(chain.input(), &load("residual")?))?;
            chain.enqueue(rows)?;
            chain.stream.drain()?;
            let mut eager = Vec::new();
            for (name, plane, width, size) in boundaries {
                let actual = read(&chain, chain.planes[plane].buffer, rows * width * size)?;
                eprintln!("SWA chained rows={rows} boundary={name}");
                metrics(&actual, &load(name)?, size, size, width)?;
                if name == "norm" {
                    super::super::norm_accuracy::check(
                        &actual,
                        &load(name)?,
                        &load("norm_fp64")?,
                        width,
                    )?;
                }
                eager.push((chain.planes[plane].buffer, actual));
            }
            for (name, buffer) in [("post", chain.ffn.input()), ("output", chain.output())] {
                let actual = read(&chain, buffer, rows * H * 4)?;
                eprintln!("SWA chained rows={rows} boundary={name}");
                metrics(&actual, &load(name)?, 4, 4, H)?;
                eager.push((buffer, actual));
            }
            device.run(|| {
                // SAFETY: stream is idle; all graph operands are pinned below.
                unsafe { library.cuda_graph_begin_capture(chain.stream.raw) }
            })?;
            let enqueue = chain.enqueue(rows);
            let raw = device.run(|| {
                // SAFETY: end capture even if enqueue failed.
                unsafe { library.cuda_graph_end_capture(chain.stream.raw) }
            })?;
            // SAFETY: mutable chain borrow retains every stream, allocation and program.
            let graph = unsafe {
                crate::shared::decode_graph::GraphOwner::new(&library, device.id, raw, &mut chain)?
            };
            enqueue?;
            let mut guard = graph.pins.stream.cancellation_guard("SWA graph replay");
            device.run(|| {
                // SAFETY: retained graph operands remain live until the drain below.
                unsafe { library.cuda_graph_launch(graph.raw, graph.pins.stream.raw) }
            })?;
            graph.pins.stream.drain()?;
            guard.complete = true;
            drop(guard);
            for (buffer, expected) in eager {
                ensure!(
                    read(graph.pins, buffer, expected.len())? == expected,
                    "SWA eager/replay mismatch"
                );
            }
            drop(graph);
            // Isolate GEMMs with official producers and the accepted FP64 error bar.
            for (stage, input, weight, plane, name) in [
                (1, "norm", 1, QRAW, "qraw"),
                (2, "norm", 2, KRAW, "kraw"),
                (2, "norm", 3, VPAGES, "value"),
                (3, "norm", 6, LOGITS, "logits"),
                (9, "gated", 7, DELTA, "delta"),
            ] {
                let input_plane = if stage == 9 { GATED } else { NORM };
                device.run(|| library.copy_h2d(chain.planes[input_plane].buffer, &load(input)?))?;
                chain.launch(
                    stage,
                    &[
                        chain.planes[input_plane].buffer.ptr,
                        chain.weights[weight].buffer.ptr,
                        chain.planes[plane].buffer.ptr,
                    ],
                    &[Scalar::I32(rows as i32)],
                )?;
                chain.stream.drain()?;
                let expected = load(name)?;
                let actual = read(&chain, chain.planes[plane].buffer, expected.len())?;
                let fp64 = load(&format!("{name}_fp64"))?;
                eprintln!("SWA isolated rows={rows} op={name} native_vs_fp64");
                let native = metrics(&actual, &fp64, 2, 8, expected.len() / rows / 2)?;
                let official = metrics(&expected, &fp64, 2, 8, expected.len() / rows / 2)?;
                ensure!(
                    native <= official * 1.01,
                    "SWA GEMM exceeds 1.01 official FP64 error"
                );
            }
            // Q/K norm and RoPE consume the official projection outputs, not
            // the native GEMM's accumulation-order differences.
            for (stage, input, weight, output, name) in
                [(4, QRAW, 4, QUERY, "query"), (5, KRAW, 5, KPAGES, "key")]
            {
                let source = if stage == 4 { "qraw" } else { "kraw" };
                device.run(|| library.copy_h2d(chain.planes[input].buffer, &load(source)?))?;
                let (cos, sin) = chain.rope.layers[0].attention.buffers();
                chain.launch(
                    stage,
                    &[
                        chain.planes[input].buffer.ptr,
                        chain.weights[weight].buffer.ptr,
                        chain.weights[weight].buffer.ptr,
                        cos.ptr,
                        sin.ptr,
                        chain.planes[POSITIONS].buffer.ptr,
                        chain.planes[output].buffer.ptr,
                    ],
                    &[Scalar::I32(rows as i32), Scalar::F32(chain.epsilon)],
                )?;
                chain.stream.drain()?;
                let expected = load(name)?;
                let actual = read(&chain, chain.planes[output].buffer, expected.len())?;
                super::super::norm_accuracy::check(
                    &actual,
                    &expected,
                    &load(&format!("{name}_fp64"))?,
                    192,
                )?;
            }
            for (plane, name) in [(QUERY, "query"), (KPAGES, "key"), (VPAGES, "value")] {
                device.run(|| library.copy_h2d(chain.planes[plane].buffer, &load(name)?))?;
            }
            chain.attend(rows)?;
            chain.stream.drain()?;
            let expected = load("attention")?;
            let actual = read(&chain, chain.planes[ATTN].buffer, expected.len())?;
            let fp64 = load("attention_fp64")?;
            eprintln!("SWA isolated rows={rows} attention native_vs_fp64");
            let native = metrics(&actual, &fp64, 2, 8, Q)?;
            eprintln!("SWA isolated rows={rows} attention official_vs_fp64");
            let official = metrics(&expected, &fp64, 2, 8, Q)?;
            // Shared flash attention rounds unnormalized online P to BF16
            // before PV, unlike this official SDPA backend. Qualify that choice
            // against an FP64 online BF16-P oracle, not a Step-5-only kernel fork.
            // Measured native/oracle FP64-error ratios at M7/128/513:
            // 1.000000809 / 1.000010453 / 0.999995771; the explicit bar is 1.01.
            let bf16_p = load("attention_bf16_p")?;
            eprintln!("SWA isolated rows={rows} attention bf16_p_vs_fp64");
            let rounded = metrics(&bf16_p, &fp64, 2, 8, Q)?;
            eprintln!("SWA isolated rows={rows} attention native_vs_bf16_p");
            metrics(&actual, &bf16_p, 2, 2, Q)?;
            eprintln!(
                "SWA attention rows={rows} native_official_ratio={} native_bf16_p_ratio={}",
                native / official,
                native / rounded
            );
            ensure!(
                native <= rounded * 1.01,
                "SWA attention exceeds 1.01 BF16-P FP64 error"
            );
            // The SDPA ratio is diagnostic only: rounding P is a shared flash
            // precision choice, with context-dependent error (about 1.09x at M7).
            // Also gate each position, so a localized mask/window error cannot
            // hide behind the whole-batch norm.
            let actual_values = values(&actual, 2)?;
            let rounded_values = values(&bf16_p, 2)?;
            let reference_values = values(&fp64, 8)?;
            for (position, ((actual, rounded), reference)) in actual_values
                .chunks(Q)
                .zip(rounded_values.chunks(Q))
                .zip(reference_values.chunks(Q))
                .enumerate()
            {
                let squared_error = |a: &[f64]| {
                    a.iter()
                        .zip(reference)
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f64>()
                };
                ensure!(
                    squared_error(actual) <= squared_error(rounded) * 1.01f64.powi(2),
                    "SWA attention position {position} exceeds 1.01 BF16-P FP64 error"
                );
            }
            for (plane, name) in [(ATTN, "attention"), (LOGITS, "logits")] {
                device.run(|| library.copy_h2d(chain.planes[plane].buffer, &load(name)?))?;
            }
            chain.launch(
                8,
                &[
                    chain.planes[ATTN].buffer.ptr,
                    chain.planes[LOGITS].buffer.ptr,
                    chain.planes[GATED].buffer.ptr,
                ],
                &[Scalar::I32(rows as i32)],
            )?;
            chain.stream.drain()?;
            ensure!(
                read(&chain, chain.planes[GATED].buffer, load("gated")?.len())? == load("gated")?,
                "SWA isolated head gate is not byte-exact"
            );
            device.run(|| library.copy_h2d(chain.planes[DELTA].buffer, &load("delta")?))?;
            chain.launch(
                10,
                &[
                    chain.input().ptr,
                    chain.planes[DELTA].buffer.ptr,
                    chain.ffn.input().ptr,
                ],
                &[Scalar::I32(rows as i32)],
            )?;
            chain.stream.drain()?;
            ensure!(
                read(&chain, chain.ffn.input(), load("post")?.len())? == load("post")?,
                "SWA isolated residual is not byte-exact"
            );
        }
        Ok(())
    }

    #[test]
    fn swa_admission_charges_page_padding_and_composed_owners() -> Result<()> {
        let a = Admission::new(1, &[1000], 2000, 3000, u64::MAX, &[])?;
        assert_eq!(a.pages, 1);
        assert_eq!(a.planes[KPAGES], 64 * KV * 2);
        assert_eq!(a.bytes, a.planes.iter().sum::<usize>() as u64 + 6000);
        assert!(Admission::new(0, &[], 0, 0, u64::MAX, &[]).is_err());
        assert!(Admission::new(4097, &[], 0, 0, u64::MAX, &[]).is_err());
        assert!(Admission::new(1, &[1000], 2000, 3000, a.bytes - 1, &[]).is_err());
        assert!(Admission::new(1, &[], 0, 0, u64::MAX, &[u64::MAX]).is_err());
        Ok(())
    }
}
