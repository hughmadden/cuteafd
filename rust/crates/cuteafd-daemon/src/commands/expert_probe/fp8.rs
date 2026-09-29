//! FP8 experts (the checkpoint's E4M3 weights with FP32 128x128 block scales):
//! the CPU oracle and the `--local` coordinator run.
//!
//! The oracle is the checkpoint math at full width: weights `bf16(w * s)`,
//! BF16 gate and up, `bf16(bf16(silu(g)) * u)`, BF16 down output, FP32 route
//! sum. Above 256 rows it checks a deterministic sample of 256 rows (every
//! output row depends only on its own input row and routes); experts run on
//! every core.
use crate::cli::ExpertProbeArgs;
use crate::v41_experts::fp8::{package_directory, Fp8Experts};
use crate::v41_memory::DeviceAllocation;
use anyhow::{Context, Result};
use cuteafd_ffi::NativeLibrary;
use cuteafd_loader::fp8_experts::{Fp8ExpertTensors, Fp8Projection};
use cuteafd_loader::OfficialV41Catalog;
use cuteafd_transport::ExpertProtocolV2RouteEntry;
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

const SAMPLE_ROWS: usize = 256;

/// Rows the oracle checks.
pub(super) fn sampled_rows(rows: usize) -> Vec<usize> {
    if rows <= SAMPLE_ROWS {
        (0..rows).collect()
    } else {
        (0..SAMPLE_ROWS).map(|i| i * rows / SAMPLE_ROWS).collect()
    }
}

/// Round to BF16 (nearest even), as torch's float -> bfloat16.
fn bf16(value: f32) -> f32 {
    let bits = u64::from(value.to_bits());
    f32::from_bits((((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) << 16) as u32)
}

fn e4m3(code: u8) -> f32 {
    super::e4m3(code)
}

/// Full-width `[rows, cols]` `bf16(w * s)` of one expert projection.
fn dequantize(tensors: &Fp8ExpertTensors, layer: usize, expert: usize, projection: Fp8Projection, _rows: usize,
    cols: usize) -> Result<Vec<f32>> {
    let (w_bytes, s_bytes) = tensors.slice_bytes(projection, 1)?;
    let (mut weight, mut scale, mut staging) = (vec![0u8; w_bytes], vec![0u8; s_bytes], Vec::new());
    tensors.read_slice(layer, expert, projection, 1, 0, &mut weight, &mut scale, &mut staging)?;
    let scale: Vec<f32> = scale.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
    let grid = cols.div_ceil(128);
    Ok(weight.iter().enumerate().map(|(i, &code)| {
        let (row, col) = (i / cols, i % cols);
        bf16(e4m3(code) * scale[(row / 128) * grid + col / 128])
    }).collect())
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Expected BF16-valued rows `sampled` (FP32 `[sampled.len(), H]`).
pub(super) fn oracle(catalog: &OfficialV41Catalog, layer: usize, input: &[f32], routes: &[ExpertProtocolV2RouteEntry],
    sampled: &[usize]) -> Result<Vec<f32>> {
    let tensors = catalog.fp8().context("the FP8 oracle needs the checkpoint's FP8 experts")?;
    let shape = *tensors.shape();
    let (hidden, inter) = (shape.hidden, shape.intermediate);
    let slot: BTreeMap<usize, usize> = sampled.iter().enumerate().map(|(i, &row)| (row, i)).collect();
    let mut by_expert: BTreeMap<u32, Vec<&ExpertProtocolV2RouteEntry>> = BTreeMap::new();
    for route in routes.iter().filter(|r| slot.contains_key(&(r.row_index as usize))) {
        by_expert.entry(route.expert_id).or_default().push(route);
    }
    let work: Vec<_> = by_expert.into_iter().collect();
    let threads = std::thread::available_parallelism().map_or(8, usize::from).min(work.len().max(1));
    let per = work.len().div_ceil(threads).max(1);
    let partials = std::thread::scope(|scope| -> Result<Vec<Vec<f32>>> {
        let (slot, input) = (&slot, input);
        let handles: Vec<_> = work.chunks(per).map(|chunk| scope.spawn(move || -> Result<Vec<f32>> {
            let mut out = vec![0f32; sampled.len() * hidden];
            for (expert, routes) in chunk {
                let e = *expert as usize;
                let w1 = dequantize(tensors, layer, e, Fp8Projection::Gate, inter, hidden)?;
                let w3 = dequantize(tensors, layer, e, Fp8Projection::Up, inter, hidden)?;
                let w2 = dequantize(tensors, layer, e, Fp8Projection::Down, hidden, inter)?;
                for route in routes {
                    let x = &input[route.row_index as usize * hidden..][..hidden];
                    let act: Vec<f32> = (0..inter).map(|n| {
                        let gate = bf16(dot(x, &w1[n * hidden..][..hidden]));
                        let up = bf16(dot(x, &w3[n * hidden..][..hidden]));
                        bf16(bf16(gate / (1.0 + (-gate).exp())) * up)
                    }).collect();
                    let row = &mut out[slot[&(route.row_index as usize)] * hidden..][..hidden];
                    for (h, value) in row.iter_mut().enumerate() {
                        *value += route.gate_weight * bf16(dot(&act, &w2[h * inter..][..inter]));
                    }
                }
            }
            Ok(out)
        })).collect();
        handles.into_iter().map(|h| h.join().map_err(|_| anyhow::anyhow!("oracle thread panicked"))?).collect()
    })?;
    let mut out = vec![0f32; sampled.len() * hidden];
    for partial in partials {
        for (o, p) in out.iter_mut().zip(partial) {
            *o += p;
        }
    }
    Ok(out)
}

/// `--local`: the probe's wire rows and routes through the full-width (TP1)
/// FP8 package on this GPU. Returns every row (FP32) and the checked launch time.
pub(super) fn run_local(args: &ExpertProbeArgs, catalog: &OfficialV41Catalog, wire: &[u8], input: &[f32],
    routes: &[ExpertProtocolV2RouteEntry]) -> Result<(Vec<f32>, Duration)> {
    let tensors = catalog.fp8().context("FP8 checkpoint")?;
    let shape = *catalog.routed_experts();
    let (hidden, rows) = (shape.hidden, args.rows as usize);
    let native_lib = args.native_lib.as_deref().context("--local needs --native-lib")?;
    // SAFETY: a trusted image library, loaded once for this process.
    let library = unsafe { NativeLibrary::load(native_lib) }?;
    library.cuda_set_device(0)?;
    let stream = library.cuda_stream_create()?;
    let (free, _) = library.cuda_memory_info()?;
    let directory = args.fp8_package.clone().unwrap_or_else(|| package_directory(native_lib, 1));
    let started = Instant::now();
    let experts = Fp8Experts::load(&library, tensors, &directory, args.layer..args.layer + 1, 1, 0, rows,
        free.saturating_sub(2 << 30))?;
    eprintln!("resident in {:.1} s", started.elapsed().as_secs_f64());
    let upload = |bytes: &[u8]| -> Result<DeviceAllocation<'_>> {
        let allocation = DeviceAllocation::new(&library, bytes.len().max(16))?;
        library.copy_h2d(allocation.buffer, bytes)?;
        Ok(allocation)
    };
    let ids: Vec<u8> = routes.iter().flat_map(|r| r.expert_id.to_le_bytes()).collect();
    let weights: Vec<u8> = routes.iter().flat_map(|r| r.gate_weight.to_le_bytes()).collect();
    // A BF16-input (coordinator) package takes the wire rows' exact BF16 values.
    let rows_in: Vec<u8> = if experts.wire_input() {
        wire.to_vec()
    } else {
        input.iter().flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes()).collect()
    };
    let (wire, ids, weights) = (upload(&rows_in)?, upload(&ids)?, upload(&weights)?);
    let output = DeviceAllocation::new(&library, rows * hidden * 2)?;
    // SAFETY: every buffer is a live device allocation of the documented extent;
    // the stream is drained before any is released.
    let launch = || unsafe {
        experts.run(0, rows, wire.buffer.ptr, ids.buffer.ptr, weights.buffer.ptr, output.buffer.ptr, stream)
    };
    let sync = || unsafe { library.cuda_stream_synchronize(stream) };
    launch()?;
    sync()?;
    let started = Instant::now();
    launch()?;
    sync()?;
    let elapsed = started.elapsed();
    if args.repeat > 0 {
        let started = Instant::now();
        for _ in 0..args.repeat {
            launch()?;
        }
        sync()?;
        println!("local fp8 layer {} rows {rows}: back-to-back {:.1} us over {} launches", args.layer,
            started.elapsed().as_secs_f64() * 1e6 / args.repeat as f64, args.repeat);
    }
    let mut bytes = vec![0u8; rows * hidden * 2];
    library.copy_d2h(&mut bytes, output.buffer)?;
    drop(experts);
    // SAFETY: nothing is queued on the drained stream.
    unsafe { library.cuda_stream_destroy(stream)? };
    Ok((bytes.chunks_exact(2).map(|p| f32::from_bits(u32::from(u16::from_le_bytes([p[0], p[1]])) << 16)).collect(),
        elapsed))
}
