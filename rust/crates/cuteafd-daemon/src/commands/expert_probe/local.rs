//! `expert-probe --local`: the probe's wire rows and routes through the
//! coordinator's resident-layer path (`dsv4::local::LocalExperts`) on this GPU.
use crate::cli::ExpertProbeArgs;
use crate::dsv4::local::{LocalExperts, LocalLayer};
use crate::v41_memory::DeviceAllocation;
use anyhow::{Context, Result};
use cuteafd_ffi::NativeLibrary;
use cuteafd_loader::OfficialV41Catalog;
use cuteafd_transport::ExpertProtocolV2RouteEntry;
use std::time::{Duration, Instant};

/// Loads only what the probe needs (dSpark stages `0..=stage`, or backbone
/// layers `0..=layer`), runs one checked launch and, with `--repeat`, times
/// back-to-back launches. Returns routed experts as FP32 rows and the time
/// of the checked launch.
pub(super) fn run(
    args: &ExpertProbeArgs,
    catalog: &OfficialV41Catalog,
    wire: &[u8],
    routes: &[ExpertProtocolV2RouteEntry],
) -> Result<(Vec<f32>, Duration)> {
    let shape = *catalog.routed_experts();
    let (hidden, topk, rows) = (shape.hidden, shape.topk, args.rows as usize);
    let native_lib = args.native_lib.as_deref().context("--local needs --native-lib")?;
    // SAFETY: a trusted image library, loaded once for this process.
    let library = unsafe { NativeLibrary::load(native_lib) }?;
    library.cuda_set_device(0)?;
    let stream = library.cuda_stream_create()?;
    let (free, _) = library.cuda_memory_info()?;
    let (stages, layers, layer) = match args.stage {
        Some(stage) => (stage + 1, 0, LocalLayer::Stage(stage)),
        None => (0, args.layer + 1, LocalLayer::Backbone(args.layer)),
    };
    let started = Instant::now();
    let mut local = LocalExperts::load(&library, native_lib, catalog, stages, layers, rows,
        free.saturating_sub(4 << 30), stream)?
        .context("no coordinator expert kernels or package for this checkpoint")?;
    anyhow::ensure!(local.stages() == stages && local.layers() == layers,
        "only {} stages and {} layers fit on this GPU", local.stages(), local.layers());
    eprintln!("resident in {:.1} s", started.elapsed().as_secs_f64());

    let upload = |bytes: &[u8]| -> Result<DeviceAllocation<'_>> {
        let allocation = DeviceAllocation::new(&library, bytes.len().max(16))?;
        library.copy_h2d(allocation.buffer, bytes)?;
        Ok(allocation)
    };
    let ids: Vec<u8> = routes.iter().flat_map(|r| r.expert_id.to_le_bytes()).collect();
    let weights: Vec<u8> = routes.iter().flat_map(|r| r.gate_weight.to_le_bytes()).collect();
    anyhow::ensure!(ids.len() == rows * topk * 4, "probe routes are not rows x top-k");
    let (wire, ids, weights) = (upload(wire)?, upload(&ids)?, upload(&weights)?);
    let shared = upload(&vec![0u8; rows * hidden * 2])?;
    let launch = |local: &mut LocalExperts<'_>| -> Result<()> {
        // SAFETY: every input is a live device allocation of the documented
        // extent, and the stream is drained before any of them is released.
        unsafe {
            local.run(layer, rows, wire.buffer.ptr, ids.buffer.ptr, weights.buffer.ptr, shared.buffer.ptr, stream)
        }
    };
    let sync = || unsafe { library.cuda_stream_synchronize(stream) };
    launch(&mut local)?;
    sync()?;
    let started = Instant::now();
    launch(&mut local)?;
    sync()?;
    let elapsed = started.elapsed();
    if args.repeat > 0 {
        let mut times = Vec::with_capacity(args.repeat);
        for _ in 0..args.repeat {
            let started = Instant::now();
            launch(&mut local)?;
            sync()?;
            times.push(started.elapsed().as_secs_f64() * 1e6);
        }
        times.sort_by(f64::total_cmp);
        let started = Instant::now();
        for _ in 0..args.repeat {
            launch(&mut local)?;
        }
        sync()?;
        let queued = started.elapsed().as_secs_f64() * 1e6 / args.repeat as f64;
        println!("local {layer:?} rows {rows} over {} repeats: synchronized median {:.0} us, min {:.0} us; \
            back-to-back {queued:.0} us", args.repeat, times[times.len() / 2], times[0]);
    }
    let mut bytes = vec![0u8; rows * hidden * 2];
    let mut output = local.output.buffer;
    output.bytes = bytes.len();
    library.copy_d2h(&mut bytes, output)?;
    let actual = bytes.chunks_exact(2)
        .map(|pair| f32::from_bits(u32::from(u16::from_le_bytes([pair[0], pair[1]])) << 16)).collect();
    drop(local);
    // SAFETY: nothing is queued on the drained stream.
    unsafe { library.cuda_stream_destroy(stream)? };
    Ok((actual, elapsed))
}
