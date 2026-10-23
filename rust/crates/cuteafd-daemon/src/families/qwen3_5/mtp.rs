//! Checkpoint-native dense MTP math; target embedding, head and scratch stay shared.
use super::*;
use super::super::weights::{DenseFormat, DenseLayer};
use cuteafd_engine::prefix::{DraftBinding, DraftPagePool};
use cuteafd_loader::families::qwen4::Qwen4KvCache;
use std::path::Path;

struct Mtp<'a> {
    library: &'a NativeLibrary,
    stream: *mut std::ffi::c_void,
    layer: DenseLayer<'a>,
    fc: Allocation<'a>,
    norm_hidden: Allocation<'a>,
    norm_embedding: Allocation<'a>,
    norm: Allocation<'a>,
    concat: Allocation<'a>,
    kv: Allocation<'a>,
    programs: DensePrograms,
    cfg: Qwen35Config,
    homes: LayerStateMap,
    pool: DraftPagePool,
    binding: DraftBinding,
}

fn format(reader: &WeightReader<'_>) -> Result<DenseFormat> {
    match reader.tensor("mtp.layers.0.mlp.gate_proj.weight")?.meta.dtype {
        cuteafd_core::DType::Bf16 => Ok(DenseFormat::Bf16),
        cuteafd_core::DType::F8E4M3 => Ok(DenseFormat::Fp8),
        ref dtype => anyhow::bail!("unsupported dense MTP projection format {dtype:?}"),
    }
}

fn fixed_bytes(reader: &WeightReader<'_>, cfg: &Qwen35Config, format: DenseFormat) -> Result<u64> {
    ensure!(cfg.mtp_layers == 1, "dense MTP requires one checkpoint-native layer");
    let mut bytes = reader.dense_layer_bytes("mtp.layers.0", HybridAttention::Full, format)?;
    for (name, shape) in [("fc", vec![cfg.hidden, 2 * cfg.hidden]), ("norm", vec![cfg.hidden]),
        ("pre_fc_norm_hidden", vec![cfg.hidden]), ("pre_fc_norm_embedding", vec![cfg.hidden])] {
        let tensor = &reader.tensor(&format!("mtp.{name}.weight"))?.meta;
        ensure!(tensor.dtype == cuteafd_core::DType::Bf16 && tensor.shape == shape,
            "mtp.{name}: expected native BF16 {shape:?}");
        bytes = bytes.checked_add(tensor.byte_length.max(256)).context("dense MTP inventory overflow")?;
    }
    Ok(bytes + (HEAD_ROWS * 2 * cfg.hidden * 2) as u64)
}

impl<'a> Mtp<'a> {
    fn load(engine: &DenseEngine<'a>, reader: &WeightReader<'a>, format: DenseFormat,
        programs: DensePrograms, expected: u64) -> Result<Self> {
        let device = engine.stream.device;
        let mut cfg = engine.cfg.clone();
        cfg.attention = vec![HybridAttention::Full];
        let homes = LayerStateMap::new(&cfg.attention, &[0], 1)?;
        let mut pool = DraftPagePool::new(engine.pool_tokens, 1, cfg.max_context, 0x514d545000000001)?;
        let binding = pool.bind(0, 0)?;
        let out = Self {
            library: device.library, stream: engine.stream.raw,
            layer: reader.dense_layer("mtp.layers.0", HybridAttention::Full, format)?,
            fc: reader.bf16("mtp.fc.weight")?, norm_hidden: reader.bf16("mtp.pre_fc_norm_hidden.weight")?,
            norm_embedding: reader.bf16("mtp.pre_fc_norm_embedding.weight")?, norm: reader.bf16("mtp.norm.weight")?,
            concat: Allocation::new(device, HEAD_ROWS * 2 * cfg.hidden * 2)?,
            kv: Allocation::new(device, engine.pool_tokens * Qwen4KvCache::Bf16.record_bytes(cfg.kv_heads, cfg.head_dim))?,
            programs, cfg, homes, pool, binding,
        };
        let actual = out.layer.bytes() + [&out.fc, &out.norm_hidden, &out.norm_embedding, &out.norm, &out.concat]
            .iter().map(|allocation| allocation.buffer.bytes).sum::<usize>();
        ensure!(actual as u64 == expected, "dense MTP admitted fixed bytes differ from allocations");
        Ok(out)
    }

    fn norm(&self, engine: &DenseEngine<'_>, source: CuteafdDeviceBuffer, weight: CuteafdDeviceBuffer,
        destination: CuteafdDeviceBuffer, rows: usize) -> Result<()> {
        ensure!(source.bytes >= rows * self.cfg.hidden * 2 && destination.bytes >= rows * self.cfg.hidden * 2,
            "dense MTP norm plane too small");
        let program = engine.programs.program(&format!("{}_norm", self.programs.prefix), &["x", "weight", "out"])?;
        // SAFETY: checked source/destination rows and owned BF16 weight survive the shared stream drain.
        unsafe { program.launch(&[source.ptr, weight.ptr, destination.ptr], &[Scalar::I32(i32::try_from(rows)?)], engine.stream.raw) }
    }

    /// Teacher-forced canonical pairs use the target position, never position+1.
    /// Hidden rows are already target-final-normalized, as specified by vLLM.
    fn step(&mut self, engine: &mut DenseEngine<'a>, hidden: CuteafdDeviceBuffer,
        tokens: &[u32], start: usize) -> Result<Vec<f32>> {
        let rows = tokens.len();
        ensure!((1..=HEAD_ROWS).contains(&rows), "dense MTP rows exceed shared target head");
        ensure!(self.pool.frontier(self.binding)?.1 == start, "dense MTP canonical frontier differs");
        engine.embedding.check(tokens)?;
        let reservation = self.pool.reserve(self.binding, start, rows)?;
        let result = (|| -> Result<Vec<f32>> {
            let kv = self.pool.destinations(reservation)?;
            let pages = self.pool.page_table(reservation)?;
            let positions = (start..start + rows).map(i64::try_from).collect::<std::result::Result<Vec<_>, _>>()?;
            let table = rows::Rows { tokens: vec![], physical: vec![], groups: vec![], slots: vec![0; rows],
                first: vec![0; rows], positions, kv, pages, stride: 0 };
            engine.row_tables(&table)?;
            let p = engine.planes;
            engine.embedding.embed(tokens, engine.ids.buffer, 1, p.residual, engine.stream.raw)?;
            self.norm(engine, p.residual, self.norm_embedding.buffer, p.norm, rows)?;
            self.norm(engine, hidden, self.norm_hidden.buffer, p.delta, rows)?;
            let pitch = self.cfg.hidden * 2;
            let second = CuteafdDeviceBuffer { ptr: self.concat.buffer.ptr.cast::<u8>().wrapping_add(pitch).cast(),
                bytes: self.concat.buffer.bytes - pitch, ..self.concat.buffer };
            // SAFETY: concat owns [64,2H] BF16; pitched, disjoint halves retain [embedding,hidden] order.
            unsafe {
                engine.stream.device.library.copy_d2d_2d_async(self.concat.buffer, 2 * pitch, p.norm, pitch,
                    pitch, rows, engine.stream.raw)?;
                engine.stream.device.library.copy_d2d_2d_async(second, 2 * pitch, p.delta, pitch,
                    pitch, rows, engine.stream.raw)?;
                engine.stream.device.library.linear_bf16(self.concat.buffer.ptr, self.fc.buffer.ptr,
                    p.residual.ptr, rows, 2 * self.cfg.hidden, self.cfg.hidden, engine.stream.raw)?;
            }
            let blocks = Blocks { programs: engine.programs, admitted: &self.programs, cfg: &self.cfg,
                homes: &self.homes, bank: &engine.bank, stream: engine.stream.raw };
            blocks.run(0, &self.layer, p, Some(self.kv.buffer), &Step {
                rows, capacity: HEAD_ROWS, table_stride: 0, splits: 1, speculative: false })?;
            self.norm(engine, p.residual, self.norm.buffer, p.norm, rows)?;
            let logits = engine.target_head(p.norm, engine.logits.buffer, rows)?;
            engine.stream.drain()?;
            logits.to_host(engine.stream.device.library)
        })();
        // No provisional destination is recycled before all producer reads/writes retire.
        engine.stream.drain()?;
        self.pool.commit(reservation, if result.is_ok() { rows } else { 0 })?;
        result
    }
}

impl Drop for Mtp<'_> {
    fn drop(&mut self) {
        // SAFETY: the target stream outlives this owner, including partially enqueued error paths.
        if let Err(error) = unsafe { self.library.cuda_stream_synchronize(self.stream) } {
            tracing::error!(%error, "retiring dense MTP allocations");
        }
    }
}

fn bf16(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(2).map(|v| f32::from_bits(u32::from(u16::from_le_bytes([v[0], v[1]])) << 16)).collect()
}

fn cosine(a: &[f32], b: &[f32]) -> Result<f64> {
    ensure!(a.len() == b.len() && !a.is_empty() && a.iter().chain(b).all(|v| v.is_finite()),
        "dense MTP nonfinite or mismatched comparison rows");
    let (dot, aa, bb) = a.iter().zip(b).fold((0., 0., 0.), |(dot, aa, bb), (&a, &b)| {
        let (a, b) = (f64::from(a), f64::from(b));
        (dot + a * b, aa + a * a, bb + b * b)
    });
    ensure!(aa > 0. && bb > 0., "dense MTP zero comparison norm");
    Ok(dot / (aa * bb).sqrt())
}

pub(crate) fn oracle<'a>(args: &super::super::GoldenArgs, library: &'a NativeLibrary,
    programs: &'a Programs<'a>, checkpoint: &'a Checkpoint, cfg: &Qwen35Config,
    target: &Path, reference: &Path, tokens: &[u32]) -> Result<()> {
    use sha2::{Digest, Sha256};
    let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(reference.join("meta.json"))?)?;
    let rows = tokens.len().checked_sub(1).context("dense MTP empty target")?;
    let width = cfg.hidden * 2;
    let source = std::fs::read(target.join(format!("layer{:02}.bin", cfg.attention.len() - 1)))?;
    let expected = std::fs::read(reference.join("mtp_streams.bin"))?;
    ensure!(rows > 1 && source.len() == tokens.len() * width && expected.len() == rows * width,
        "dense MTP reference hidden rows differ");
    ensure!(meta["family"] == "qwen3_5" && meta["rows"].as_u64() == Some(rows as u64)
        && meta["hidden"].as_u64() == Some(cfg.hidden as u64) && meta["vocab"].as_u64() == Some(cfg.vocab as u64)
        && meta["target_hidden"] == "post-final-norm" && meta["feedback"] == "post-mtp.norm"
        && meta["concat"] == serde_json::json!(["embedding", "hidden"]) && meta["position"] == "target"
        && meta["vllm_revision"] == "0510bd66795683d41f0dcdbcb374bfbd6e91595f"
        && meta["vllm_mtp_blob"] == "2acac6b5b7c5979fbca477dabcd0a0aa5bfb0ff7"
        && meta["source_pin"]["revision"] == "62d7ebd7de4938e072b7aaeb881593b79dc56835"
        && meta["source_pin"]["modeling_sha256"] == "7788020ca357c6c8fb6e750fd847dd80ac77625902213b47f4949c8085344a58"
        && meta["target_hidden_sha256"] == format!("{:x}", Sha256::digest(&source))
        && meta["tokens_sha256"] == format!("{:x}", Sha256::digest(std::fs::read(target.join("tokens.bin"))?)),
        "dense MTP independent reference contract/provenance differs");
    let selected = meta["logit_rows"].as_array().context("dense MTP logit rows missing")?.iter()
        .map(|row| usize::try_from(row.as_u64().context("dense MTP invalid logit row")?).map_err(Into::into))
        .collect::<Result<Vec<_>>>()?;
    ensure!(!selected.is_empty() && selected.iter().all(|&r| r < rows) && selected.windows(2).all(|r| r[0] < r[1]),
        "dense MTP logit selection invalid");
    let payload = std::fs::read(reference.join("mtp_logits.bin"))?;
    ensure!(payload.len() == selected.len() * cfg.vocab * 4, "dense MTP reference logits shape differs");
    let ref_logits = payload.chunks_exact(4).map(|v| f32::from_le_bytes(v.try_into().unwrap())).collect::<Vec<_>>();
    let payload = std::fs::read(reference.join("mtp_argmax.bin"))?;
    ensure!(payload.len() == rows * 4, "dense MTP reference argmax shape differs");
    let ref_best = payload.chunks_exact(4).map(|v| u32::from_le_bytes(v.try_into().unwrap())).collect::<Vec<_>>();
    let reader = WeightReader { library, checkpoint };
    let format = format(&reader)?;
    let admitted = DensePrograms::new(cfg, programs, Qwen4KvCache::Bf16, format)?;
    let fixed = fixed_bytes(&reader, cfg, format)?;
    // Target planes/scratch/head are reused; only concat, weights and one pooled GQA record are new.
    let demand = DrafterBytes { owned: fixed + (HEAD_ROWS * width) as u64, scratch: 0,
        pool_unit: (64 * Qwen4KvCache::Bf16.record_bytes(cfg.kv_heads, cfg.head_dim)) as u64 };
    let mut engine = DenseEngine::with_prefix(library, programs, checkpoint, cfg.clone(), args.pool_tokens,
        Some(demand), 1, None, args.kv_cache, false)?;
    ensure!(engine.planes.scratch.bytes >= admitted.scratch_bytes, "MTP scratch exceeds shared target admission");
    let input = Allocation::new(engine.stream.device, HEAD_ROWS * width)?;
    let mut mtp = Mtp::load(&engine, &reader, format, admitted, fixed)?;
    mtp.pool.admit(mtp.binding, rows, HEAD_ROWS)?;
    let (mut hidden_cos, mut logit_cos, mut agree, mut nll) = (Vec::new(), Vec::new(), 0usize, 0.);
    for first in (0..rows).step_by(HEAD_ROWS) {
        let end = (first + HEAD_ROWS).min(rows);
        library.copy_h2d(input.buffer, &source[first * width..end * width])?;
        mtp.norm(&engine, input.buffer, engine.weights.norm.buffer, input.buffer, end - first)?;
        let logits = mtp.step(&mut engine, input.buffer, &tokens[first + 1..end + 1], first)?;
        let byte_count = (end - first) * width;
        let mut bytes = vec![0; byte_count];
        library.copy_d2h(&mut bytes, CuteafdDeviceBuffer { bytes: byte_count, ..engine.planes.norm })?;
        let ours = bf16(&bytes);
        let theirs = bf16(&expected[first * width..end * width]);
        for (a, b) in ours.chunks_exact(cfg.hidden).zip(theirs.chunks_exact(cfg.hidden)) { hidden_cos.push(cosine(a, b)?); }
        for (offset, row) in logits.chunks_exact(cfg.vocab).enumerate() {
            let position = first + offset;
            ensure!(row.iter().all(|v| v.is_finite()), "dense MTP nonfinite logits");
            let best = row.iter().enumerate().max_by(|(i, a), (j, b)| a.total_cmp(b).then_with(|| j.cmp(i))).unwrap().0;
            agree += usize::from(best as u32 == ref_best[position]);
            if position + 2 < tokens.len() {
                let top = f64::from(row[best]);
                nll += top + row.iter().map(|&v| (f64::from(v) - top).exp()).sum::<f64>().ln()
                    - f64::from(row[tokens[position + 2] as usize]);
            }
            if let Ok(index) = selected.binary_search(&position) {
                logit_cos.push(cosine(row, &ref_logits[index * cfg.vocab..(index + 1) * cfg.vocab])?);
            }
        }
    }
    let hidden_min = hidden_cos.into_iter().fold(1., f64::min);
    let logit_min = logit_cos.into_iter().fold(1., f64::min);
    let nll = nll / (rows - 1) as f64;
    let reference_nll = meta["mean_nll"].as_f64().context("dense MTP reference NLL missing")?;
    println!("dense MTP oracle: {rows} rows, hidden cosine worst {hidden_min:.8}, logits cosine worst {logit_min:.8}, greedy agreement {agree}/{rows}, mean NLL {nll:.8}, reference {reference_nll:.8}");
    ensure!(hidden_min >= 0.99 && logit_min >= 0.99 && (nll - reference_nll).abs() <= args.nll_tolerance,
        "dense MTP independent oracle failed");
    Ok(())
}
