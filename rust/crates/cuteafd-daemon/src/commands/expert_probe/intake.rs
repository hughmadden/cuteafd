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
use cuteafd_transport::{expert::V41Tp4Roce, ExpertProtocolV2Request, TcpTransportConfig};
use std::net::SocketAddr;
use std::time::Instant;

pub(super) fn parse_modes(list: &str) -> Result<Vec<IntakeMode>> {
    list.split(',').map(|mode| match mode.trim() {
        "host" => Ok(IntakeMode::Host),
        "pinned" => Ok(IntakeMode::Pinned),
        "gpu" => Ok(IntakeMode::Gpu),
        other => anyhow::bail!("unknown intake mode {other:?} (host, pinned, gpu)"),
    }).collect()
}

enum Path<'a> {
    // Drops before the intake whose planes it lands in.
    Inline { transport: V41Tp4Roce, intake: SparkIntake<'a> },
    Lane(SparkLane<'a>),
}

struct Arm<'a> {
    mode: IntakeMode,
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

pub(super) async fn run(args: &ExpertProbeArgs, modes: &[IntakeMode], peers: &[SocketAddr], executors: &[u64],
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
    let reducer = library.v41_compact_reducer()?;
    let config = TcpTransportConfig { timing: false, timeout: std::time::Duration::from_secs(60), max_frame_bytes: 64 << 20 };
    let mut arms = Vec::new();
    for &mode in modes {
        let intake = SparkIntake::new(&library, mode, ranks, args.capacity as usize, row_bytes)?;
        let path = if args.intake_lane {
            // SAFETY: the lane drops (joining its thread) before its intake,
            // and every submit below follows `before_dispatch`.
            let lane = unsafe { intake.spawn_lane(peers.to_vec(), executors.to_vec(), args.capacity, config.clone())? };
            Path::Lane(SparkLane { lane, intake })
        } else {
            let mut transport = V41Tp4Roce::new_ranks(peers, executors, args.capacity, config.clone())?;
            // SAFETY: the arm drops its transport before its intake, and every
            // dispatch below follows `before_dispatch`.
            unsafe { intake.attach(&mut transport)? };
            Path::Inline { transport, intake }
        };
        arms.push(Arm { mode, path, times: Vec::new(), landed: 0 });
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
                    let wave = transport.dispatch_wave(request)?;
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
                reducer.reduce_planes(arm.intake().pointers(), ranks as u32, shared.buffer.ptr.cast(),
                    output.buffer.ptr.cast(), rows as u32, stream)?;
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
                if planes.iter().all(|&b| b == 0) {
                    // Workers answering without computing (CUTEAFD_EXPERTD_SKIP_COMPUTE).
                    println!("{}: all-zero planes, not compared", arm.mode.name());
                    continue;
                }
                match &reference {
                    None => reference = Some((planes, sums)),
                    Some((p, s)) => {
                        let plane_diff = p.iter().zip(&planes).filter(|(a, b)| a != b).count();
                        let sum_diff = s.chunks_exact(2).zip(sums.chunks_exact(2)).filter(|(a, b)| a != b).count();
                        println!("{} vs {}: planes {} differing bytes, reduced sums {} differing values ({})",
                            arm.mode.name(), modes[0].name(), plane_diff, sum_diff,
                            if plane_diff == 0 && sum_diff == 0 { "bit-identical" } else { "MISMATCH" });
                        ensure!(plane_diff == 0 && sum_diff == 0, "{} intake differs from {}", arm.mode.name(),
                            modes[0].name());
                    }
                }
            } else {
                arm.times.push(elapsed);
            }
        }
    }
    let bytes = (ranks * rows * row_bytes) as f64;
    for arm in &mut arms {
        arm.times.sort_by(f64::total_cmp);
        let median = arm.times[arm.times.len() / 2];
        println!("intake {:6}{}: {} waves of {rows} rows x {ranks} ranks ({:.1} MB), landed ranks {:#04b}: \
            median {:.3} ms (min {:.3}), {:.1} GB/s partials",
            arm.mode.name(), if args.intake_lane { " (lane)" } else { "" }, arm.times.len(), bytes / 1e6, arm.landed, median * 1e3, arm.times[0] * 1e3,
            bytes / median / 1e9);
    }
    drop(arms);
    // SAFETY: the stream was created above and every use has been synchronized.
    unsafe { library.cuda_stream_destroy(stream)? };
    Ok(())
}
