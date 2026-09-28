//! DeepSeek V4 coordinator weights, built from raw checkpoint bytes exactly as
//! b12x.integration.cuteafd.weights.WEIGHT_SOURCES prescribes: every operand is
//! checkpoint bytes (row-concatenated where listed) except block-FP8 scales,
//! which the scale-prep program re-lays into MMA tile order.
use crate::v41_memory::DeviceAllocation;
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::dsv4::{Dsv4Programs, Dsv4Scalar};
use cuteafd_ffi::NativeLibrary;
use cuteafd_loader::deepseek_v4::DeepseekV4Config;
use cuteafd_loader::OfficialV41Catalog;
use std::collections::HashMap;
use std::ffi::c_void;
use std::os::unix::fs::FileExt;

enum Prep {
    Raw,
    /// Block-FP8 scales for a weight [groups*n, k].
    Scale { n: usize, k: usize, groups: usize },
}

struct Source {
    operand: &'static str,
    tensors: &'static [&'static str],
    prep: Prep,
}

pub(crate) struct LayerWeights<'a> {
    pub ratio: usize,
    pub hash: bool,
    operands: HashMap<&'static str, DeviceAllocation<'a>>,
    /// Host routing tables: score-correction bias (score layers) and the
    /// token -> expert table (hash layers, [vocab, topk]).
    pub gate_bias: Vec<f32>,
    pub tid2eid: Vec<i64>,
}

impl LayerWeights<'_> {
    pub fn ptr(&self, operand: &str) -> Result<*mut c_void> {
        Ok(self.operands.get(operand).with_context(|| format!("layer has no weight {operand}"))?.buffer.ptr)
    }
}

pub(crate) struct ModelWeights<'a> {
    pub layers: Vec<LayerWeights<'a>>,
    pub head: DeviceAllocation<'a>,
    pub head_fn: DeviceAllocation<'a>,
    pub head_scale: DeviceAllocation<'a>,
    pub head_base: DeviceAllocation<'a>,
    pub norm: DeviceAllocation<'a>,
}

fn layer_sources(cfg: &DeepseekV4Config, ratio: usize, hash: bool) -> Vec<Source> {
    let (h, q, heads, g, r) = (cfg.dim, cfg.q_lora_rank, cfg.n_heads, cfg.o_groups, cfg.o_lora_rank);
    let w = heads * 512 / g;
    let inter = cfg.moe_inter_dim;
    let mut sources = vec![
        Source { operand: "attn.fn", tensors: &["hc_attn_fn"], prep: Prep::Raw },
        Source { operand: "attn.scale", tensors: &["hc_attn_scale"], prep: Prep::Raw },
        Source { operand: "attn.base", tensors: &["hc_attn_base"], prep: Prep::Raw },
        Source { operand: "attn.norm", tensors: &["attn_norm.weight"], prep: Prep::Raw },
        Source { operand: "ffn.fn", tensors: &["hc_ffn_fn"], prep: Prep::Raw },
        Source { operand: "ffn.scale", tensors: &["hc_ffn_scale"], prep: Prep::Raw },
        Source { operand: "ffn.base", tensors: &["hc_ffn_base"], prep: Prep::Raw },
        Source { operand: "ffn.norm", tensors: &["ffn_norm.weight"], prep: Prep::Raw },
        Source { operand: "w_qkv", tensors: &["attn.wq_a.weight", "attn.wkv.weight"], prep: Prep::Raw },
        Source { operand: "w_qkv_scale", tensors: &["attn.wq_a.scale", "attn.wkv.scale"],
            prep: Prep::Scale { n: q + 512, k: h, groups: 1 } },
        Source { operand: "w_q", tensors: &["attn.wq_b.weight"], prep: Prep::Raw },
        Source { operand: "w_q_scale", tensors: &["attn.wq_b.scale"], prep: Prep::Scale { n: heads * 512, k: q, groups: 1 } },
        Source { operand: "q_norm", tensors: &["attn.q_norm.weight"], prep: Prep::Raw },
        Source { operand: "kv_norm", tensors: &["attn.kv_norm.weight"], prep: Prep::Raw },
        Source { operand: "attn_sink", tensors: &["attn.attn_sink"], prep: Prep::Raw },
        Source { operand: "wo_a", tensors: &["attn.wo_a.weight"], prep: Prep::Raw },
        Source { operand: "wo_a_scale", tensors: &["attn.wo_a.scale"], prep: Prep::Scale { n: r, k: w, groups: g } },
        Source { operand: "wo_b", tensors: &["attn.wo_b.weight"], prep: Prep::Raw },
        Source { operand: "wo_b_scale", tensors: &["attn.wo_b.scale"], prep: Prep::Scale { n: h, k: g * r, groups: 1 } },
        Source { operand: "w13", tensors: &["ffn.shared_experts.w1.weight", "ffn.shared_experts.w3.weight"], prep: Prep::Raw },
        Source { operand: "w13_scale", tensors: &["ffn.shared_experts.w1.scale", "ffn.shared_experts.w3.scale"],
            prep: Prep::Scale { n: 2 * inter, k: h, groups: 1 } },
        Source { operand: "w2", tensors: &["ffn.shared_experts.w2.weight"], prep: Prep::Raw },
        Source { operand: "w2_scale", tensors: &["ffn.shared_experts.w2.scale"], prep: Prep::Scale { n: h, k: inter, groups: 1 } },
        Source { operand: "gate", tensors: &["ffn.gate.weight"], prep: Prep::Raw },
    ];
    if ratio == 4 {
        sources.extend([
            Source { operand: "joint_projection", tensors: &["attn.compressor.wkv.weight", "attn.compressor.wgate.weight",
                "attn.indexer.compressor.wkv.weight", "attn.indexer.compressor.wgate.weight"], prep: Prep::Raw },
            Source { operand: "index_ape", tensors: &["attn.indexer.compressor.ape"], prep: Prep::Raw },
            Source { operand: "index_norm", tensors: &["attn.indexer.compressor.norm.weight"], prep: Prep::Raw },
            Source { operand: "index_w_q", tensors: &["attn.indexer.wq_b.weight"], prep: Prep::Raw },
            Source { operand: "index_w_q_scale", tensors: &["attn.indexer.wq_b.scale"],
                prep: Prep::Scale { n: cfg.index_n_heads * cfg.index_head_dim, k: q, groups: 1 } },
            Source { operand: "index_w_proj", tensors: &["attn.indexer.weights_proj.weight"], prep: Prep::Raw },
        ]);
    } else if ratio == 128 {
        sources.push(Source { operand: "joint_projection",
            tensors: &["attn.compressor.wkv.weight", "attn.compressor.wgate.weight"], prep: Prep::Raw });
    }
    if ratio != 0 {
        sources.extend([
            Source { operand: "main_ape", tensors: &["attn.compressor.ape"], prep: Prep::Raw },
            Source { operand: "main_norm", tensors: &["attn.compressor.norm.weight"], prep: Prep::Raw },
        ]);
    }
    sources
}

pub(crate) struct WeightLoader<'a, 'p> {
    pub library: &'a NativeLibrary,
    pub catalog: &'a OfficialV41Catalog,
    pub programs: &'p Dsv4Programs<'a>,
    pub family: &'static str,
    pub stream: *mut c_void,
}

impl<'a> WeightLoader<'a, '_> {
    fn read(&self, names: &[String]) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        for name in names {
            let tensor = self.catalog.tensor(name)?;
            let start = bytes.len();
            bytes.resize(start + tensor.metadata.byte_length as usize, 0);
            std::fs::File::open(self.catalog.snapshot().join(&tensor.shard))?
                .read_exact_at(&mut bytes[start..], tensor.metadata.byte_offset)
                .with_context(|| format!("reading {name}"))?;
        }
        Ok(bytes)
    }

    pub fn upload(&self, bytes: &[u8]) -> Result<DeviceAllocation<'a>> {
        let allocation = DeviceAllocation::new(self.library, bytes.len().max(16))?;
        self.library.copy_h2d(allocation.buffer, bytes)?;
        Ok(allocation)
    }

    pub fn tensor(&self, name: &str) -> Result<DeviceAllocation<'a>> {
        self.upload(&self.read(&[name.to_string()])?)
    }

    fn scale(&self, raw: &[u8], n: usize, k: usize, groups: usize) -> Result<DeviceAllocation<'a>> {
        ensure!(n % 128 == 0 && k % 128 == 0, "block-FP8 weights need 128-multiple extents");
        let (n_blocks, k_blocks) = (groups * n / 128, k / 128);
        ensure!(raw.len() == n_blocks * k_blocks, "scale bytes {} do not cover {n_blocks}x{k_blocks} blocks", raw.len());
        let source = self.upload(raw)?;
        let output = DeviceAllocation::new(self.library, n_blocks * k_blocks * 512)?;
        let program = self.programs.program(&format!("{}_block_fp8_scale_prep", self.family), &["scale", "scale_mma"])?;
        // SAFETY: both buffers are live device allocations sized for the grid;
        // the stream is synchronized before `source` drops.
        unsafe {
            program.launch(&[source.buffer.ptr, output.buffer.ptr],
                &[Dsv4Scalar::I32(n_blocks as i32), Dsv4Scalar::I32(k_blocks as i32)], self.stream)?;
            self.library.cuda_stream_synchronize(self.stream)?;
        }
        Ok(output)
    }

    pub fn layer(&self, cfg: &DeepseekV4Config, layer: usize) -> Result<LayerWeights<'a>> {
        let ratio = cfg.compress_ratios[layer];
        let hash = cfg.is_hash_layer(layer);
        let mut operands = HashMap::new();
        for source in layer_sources(cfg, ratio, hash) {
            let names: Vec<String> = source.tensors.iter().map(|t| format!("layers.{layer}.{t}")).collect();
            let raw = self.read(&names)?;
            let allocation = match source.prep {
                Prep::Raw => self.upload(&raw)?,
                Prep::Scale { n, k, groups } => self.scale(&raw, n, k, groups)?,
            };
            operands.insert(source.operand, allocation);
        }
        let (mut gate_bias, mut tid2eid) = (Vec::new(), Vec::new());
        if hash {
            let raw = self.read(&[format!("layers.{layer}.ffn.gate.tid2eid")])?;
            tid2eid = raw.chunks_exact(8).map(|b| i64::from_le_bytes(b.try_into().unwrap())).collect();
        } else {
            let raw = self.read(&[format!("layers.{layer}.ffn.gate.bias")])?;
            gate_bias = raw.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
        }
        Ok(LayerWeights { ratio, hash, operands, gate_bias, tid2eid })
    }

    pub fn model(&self, cfg: &DeepseekV4Config) -> Result<ModelWeights<'a>> {
        let layers = (0..cfg.n_layers).map(|layer| self.layer(cfg, layer)).collect::<Result<_>>()?;
        Ok(ModelWeights {
            layers,
            head: self.tensor("head.weight")?,
            head_fn: self.tensor("hc_head_fn")?,
            head_scale: self.tensor("hc_head_scale")?,
            head_base: self.tensor("hc_head_base")?,
            norm: self.tensor("norm.weight")?,
        })
    }

    /// Host copy of layer tensors (routing tables read on the CPU).
    pub fn host(&self, name: &str) -> Result<Vec<u8>> {
        self.read(&[name.to_string()])
    }
}
