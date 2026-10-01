//! `expert-probe --intake host,pinned,gpu`: one request's rank partials through
//! each coordinator intake (see `crate::shared::spark_intake`) against the same Spark
//! (or loopback) ranks. Checks that every mode yields bit-identical rank planes
//! and compact-reduced sums, and times waves from dispatch until the reduced
//! rows are complete on the GPU, interleaving the modes round by round.
use crate::cli::ExpertProbeArgs;
use crate::shared::spark_intake::{self, IntakeMode, SparkIntake, SparkLane};
use crate::shared::memory::DeviceAllocation;
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::NativeLibrary;
use cuteafd_transport::{expert::SparkExperts, ExpertProtocolV2Request, TcpTransportConfig};
use std::net::SocketAddr;
use std::time::Instant;

/// Intake modes, each optionally `+reduce` (the Sparks reduce-scatter the
/// wave; the coordinator gathers their reduced rows).
pub(super) fn parse_modes(list: &str) -> Result<Vec<(IntakeMode, bool)>> {
    list.split(',').map(|mode| {
        let mode = mode.trim();
        let (name, reduced) = match mode.strip_suffix("+reduce") {
            Some(name) => (name, true),
            None => (mode, false),
        };
        let mode = match name {
            "host" => IntakeMode::Host,
            "pinned" => IntakeMode::Pinned,
            "gpu" => IntakeMode::Gpu,
            other => anyhow::bail!("unknown intake mode {other:?} (host, pinned, gpu, each optionally +reduce)"),
        };
        Ok((mode, reduced))
    }).collect()
}

/// Differing values, largest absolute difference and cosine of two BF16 planes.
fn compare_bf16(a: &[u8], b: &[u8]) -> (usize, f64, f64) {
    let value = |pair: &[u8]| f64::from(f32::from_bits(u32::from(u16::from_le_bytes([pair[0], pair[1]])) << 16));
    let (mut differ, mut max_abs, mut dot, mut na, mut nb) = (0, 0f64, 0f64, 0f64, 0f64);
    for (x, y) in a.chunks_exact(2).zip(b.chunks_exact(2)) {
        let (x, y) = (value(x), value(y));
        if x.to_bits() != y.to_bits() {
            differ += 1;
        }
        max_abs = max_abs.max((x - y).abs());
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    (differ, max_abs, dot / (na.sqrt() * nb.sqrt()).max(f64::MIN_POSITIVE))
}

enum Path<'a> {
    // Drops before the intake whose planes it lands in.
    Inline { transport: SparkExperts, intake: SparkIntake<'a> },
    Lane(SparkLane<'a>),
}

struct Arm<'a> {
    mode: IntakeMode,
    reduced: bool,
    path: Path<'a>,
    times: Vec<f64>,
    landed: u8,
}

impl<'a> Arm<'a> {
    fn intake(&self) -> &SparkIntake<'a> {
        match &self.path {
            Path::Inline { intake, .. } => intake,
            Path::Lane(lane) => &lane.intake,
        }
    }
}

pub(super) async fn run(args: &ExpertProbeArgs, modes: &[(IntakeMode, bool)], peers: &[SocketAddr], executors: &[u64],
    request: &mut ExpertProtocolV2Request, hidden: usize) -> Result<()> {
    let native_lib = args.native_lib.as_deref().context("--intake needs --native-lib")?;
    // SAFETY: a trusted image library, loaded once for this process.
    let library = unsafe { NativeLibrary::load(native_lib) }?;
    library.cuda_set_device(0)?;
    let rows = request.header.row_count as usize;
    let ranks = peers.len();
    let row_bytes = hidden * 2;
    match spark_intake::probe_gpu_landing(&library) {
        Ok(p) => println!("gpu landing probe: dma_buf={} registered={} ordering={} gpu {:.1} GB/s host {:.1} GB/s ({}){}",
            p.dma_buf, p.registered, p.writes_ordering, p.gpu_gbps, p.host_gbps, p.status,
            p.error.as_deref().map(|e| format!(" error: {e}")).unwrap_or_default()),
        Err(error) => println!("gpu landing probe failed: {error:#}"),
    }
    let stream = library.cuda_stream_create()?;
    let shared = DeviceAllocation::new(&library, rows * row_bytes)?;
    library.cuda_zero_bytes(shared.buffer, shared.buffer.bytes)?;
    let output = DeviceAllocation::new(&library, rows * row_bytes)?;
    // A shared-expert stand-in (deterministic BF16 values of the partials'
    // scale) to measure the extra rounding a Spark-reduced wave takes before
    // the shared expert joins.
    let noise = DeviceAllocation::new(&library, rows * row_bytes)?;
    {
        let mut state = 0x9e37_79b9_7f4a_7c15u64 ^ args.seed;
        let values: Vec<u8> = (0..rows * hidden).flat_map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let value = ((state >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.25;
            ((value.to_bits() >> 16) as u16).to_le_bytes()
        }).collect();
        library.copy_h2d(noise.buffer, &values)?;
    }
    let mut reference_shared: Option<Vec<u8>> = None;
    let reducer = library.v41_compact_reducer()?;
    let config = TcpTransportConfig { timing: false, timeout: std::time::Duration::from_secs(60), max_frame_bytes: 64 << 20 };
    let mut arms = Vec::new();
    for &(mode, reduced) in modes {
        ensure!(!(reduced && args.intake_lane), "lane transports do not reduce on the Sparks");
        let intake = SparkIntake::new(&library, mode, ranks, args.capacity as usize, row_bytes)?;
        let path = if args.intake_lane {
            // SAFETY: the lane drops (joining its thread) before its intake,
            // and every submit below follows `before_dispatch`.
            let lane = unsafe { intake.spawn_lane(peers.to_vec(), executors.to_vec(), args.capacity, config.clone())? };
            Path::Lane(SparkLane { lane, intake })
        } else {
            let mut transport = SparkExperts::new_ranks(peers, executors, args.capacity, config.clone())?;
            // SAFETY: the arm drops its transport before its intake, and every
            // dispatch below follows `before_dispatch`.
            unsafe { intake.attach(&mut transport)? };
            Path::Inline { transport, intake }
        };
        arms.push(Arm { mode, reduced, path, times: Vec::new(), landed: 0 });
    }
    let mut reference: Option<(Vec<u8>, Vec<u8>)> = None;
    let rounds = args.repeat.max(1);
    for round in 0..=rounds {
        for arm in &mut arms {
            request.header.request_id += 1;
            let started = Instant::now();
            let landed = match &mut arm.path {
                Path::Inline { transport, intake } => {
                    intake.before_dispatch()?;
                    intake.set_row_sharded(arm.reduced);
                    let wave = if arm.reduced {
                        let mut flagged = request.clone();
                        flagged.header.flags |= cuteafd_transport::expert::SPARK_ROW_SHARD_FLAGS;
                        flagged.header.request_id = spark_intake::next_reduced_request_id();
                        transport.dispatch_wave(&flagged)?
                    } else {
                        transport.dispatch_wave(request)?
                    };
                    intake.receive(transport, wave, rows, stream).await?.landed
                }
                Path::Lane(lane) => {
                    let owned = request.clone();
                    lane.submit(rows, Box::new(move || Ok(owned)))?;
                    lane.wait(rows, std::time::Duration::from_secs(60), stream)?.landed
                }
            };
            // SAFETY: planes, the zero shared plane and the output hold `rows`
            // BF16 rows; the stream is ordered after the wave's uploads.
            unsafe {
                arm.intake().reduce_into(shared.buffer.ptr.cast(), output.buffer.ptr.cast(), rows, stream)?;
                library.cuda_stream_synchronize(stream)?;
            }
            arm.intake().consumed(stream)?;
            let elapsed = started.elapsed().as_secs_f64();
            arm.landed = landed;
            if round == 0 {
                // Warm-up wave: compare its planes and sums across modes.
                let mut planes = vec![0u8; ranks * rows * row_bytes];
                for (rank, plane) in arm.intake().planes().enumerate() {
                    library.copy_d2h(&mut planes[rank * rows * row_bytes..][..rows * row_bytes], plane)?;
                }
                let mut sums = vec![0u8; rows * row_bytes];
                library.copy_d2h(&mut sums, output.buffer)?;
                // The same wave plus a nonzero shared term.
                let mut with_shared = vec![0u8; rows * row_bytes];
                // SAFETY: as above; the planes still hold this wave.
                unsafe {
                    arm.intake().reduce_into(noise.buffer.ptr.cast(), output.buffer.ptr.cast(), rows, stream)?;
                    library.cuda_stream_synchronize(stream)?;
                }
                library.copy_d2h(&mut with_shared, output.buffer)?;
                if arm.reduced {
                    let (reference_sums, reference_with_shared) = match (&reference, &reference_shared) {
                        (Some((_, sums)), Some(shared)) => (sums, shared),
                        _ => anyhow::bail!("list a mode without +reduce first: it is the reference"),
                    };
                    let (differ, max_abs, cosine) = compare_bf16(reference_sums, &sums);
                    let (differ_shared, max_abs_shared, cosine_shared) =
                        compare_bf16(reference_with_shared, &with_shared);
                    println!("{}+reduce vs {}: reduced sums {differ} differing values (max |d| {max_abs:.3e},                         cosine {cosine:.9}); with a shared term {differ_shared} of {} differ (max |d|                         {max_abs_shared:.3e}, cosine {cosine_shared:.9})",
                        arm.mode.name(), modes[0].0.name(), rows * hidden);
                    ensure!(differ == 0, "Spark-reduced sums differ from the coordinator reduce");
                    ensure!(cosine_shared >= 0.99999, "Spark-reduced sums with a shared term drift");
                    continue;
                }
                if planes.iter().all(|&b| b == 0) {
                    // Workers answering without computing (CUTEAFD_EXPERTD_SKIP_COMPUTE).
                    println!("{}: all-zero planes, not compared", arm.mode.name());
                    continue;
                }
                match &reference {
                    None => {
                        reference = Some((planes, sums));
                        reference_shared = Some(with_shared);
                    }
                    Some((p, s)) => {
                        let plane_diff = p.iter().zip(&planes).filter(|(a, b)| a != b).count();
                        let sum_diff = s.chunks_exact(2).zip(sums.chunks_exact(2)).filter(|(a, b)| a != b).count();
                        println!("{} vs {}: planes {} differing bytes, reduced sums {} differing values ({})",
                            arm.mode.name(), modes[0].0.name(), plane_diff, sum_diff,
                            if plane_diff == 0 && sum_diff == 0 { "bit-identical" } else { "MISMATCH" });
                        ensure!(plane_diff == 0 && sum_diff == 0, "{} intake differs from {}", arm.mode.name(),
                            modes[0].0.name());
                    }
                }
            } else {
                arm.times.push(elapsed);
            }
        }
    }
    for arm in &mut arms {
        arm.times.sort_by(f64::total_cmp);
        let median = arm.times[arm.times.len() / 2];
        // Bytes the coordinator lands: one reduced plane, or one per rank.
        let bytes = (if arm.reduced { 1 } else { ranks } * rows * row_bytes) as f64;
        println!("intake {:6}{}{}: {} waves of {rows} rows x {ranks} ranks ({:.1} MB landed), landed ranks {:#04b}: \
            median {:.3} ms (min {:.3}), {:.1} GB/s landed",
            arm.mode.name(), if arm.reduced { "+reduce" } else { "" }, if args.intake_lane { " (lane)" } else { "" },
            arm.times.len(), bytes / 1e6, arm.landed, median * 1e3, arm.times[0] * 1e3, bytes / median / 1e9);
    }
    drop(arms);
    // SAFETY: the stream was created above and every use has been synchronized.
    unsafe { library.cuda_stream_destroy(stream)? };
    Ok(())
}
