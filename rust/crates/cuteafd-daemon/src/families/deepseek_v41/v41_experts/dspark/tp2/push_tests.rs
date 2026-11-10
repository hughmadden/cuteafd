//! B10 on two GPUs with synthetic weights (no checkpoint): the pushing draft
//! TP2 pair against the pulling one, eagerly and replayed from a graph, and
//! a timed three-stage round against the single-card drafter.
use super::{pair::{padded, Pair}, rank::Weights};
use crate::families::deepseek_v41::v41_experts::{ExpertLayer, ExpertWeights};
use crate::families::deepseek_v41::v41_peer_push;
use crate::shared::memory::device::{Allocation, Device, Stream};
use anyhow::{ensure, Result};
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary};
use std::ffi::c_void;

const CAPACITY: u32 = 16;
const ROWS: [u32; 3] = [1, 7, 16];

fn bf16(v: f32) -> u16 {
    let b = v.to_bits();
    ((b.wrapping_add(0x7fff + ((b >> 16) & 1))) >> 16) as u16
}

/// One round's inputs on the owner: BF16 hidden rows, three distinct expert
/// ids and FP32 weights a row, BF16 shared-expert rows.
struct Inputs<'a> {
    hidden: Allocation<'a>,
    ids: Allocation<'a>,
    weights: Allocation<'a>,
    shared: Allocation<'a>,
}
impl<'a> Inputs<'a> {
    fn new(owner: Device<'a>) -> Result<Self> {
        let rows = CAPACITY as usize;
        Ok(Self {
            hidden: Allocation::new(owner, rows * 10240)?,
            ids: Allocation::new(owner, padded(rows * 12))?,
            weights: Allocation::new(owner, padded(rows * 12))?,
            shared: Allocation::new(owner, rows * 10240)?,
        })
    }
    fn fill_own(&self, seed: usize) -> Result<()> {
        fill(self.hidden.device, [self.hidden.buffer, self.ids.buffer, self.weights.buffer], self.shared.buffer, seed)
    }
}

/// Writes a round's inputs (hidden, ids, weights) and shared rows on `owner`.
fn fill(owner: Device<'_>, [hidden, ids, weights]: [CuteafdDeviceBuffer; 3], shared: CuteafdDeviceBuffer,
    seed: usize) -> Result<()> {
        let library = owner.library;
        let rows = CAPACITY as usize;
        let values = |count: usize, scale: f32, salt: usize| -> Vec<u8> {
            (0..count).flat_map(|i| bf16((((i * 37 + seed * 11 + salt) % 257) as f32 - 128.0) * scale).to_ne_bytes()).collect()
        };
        let routes: Vec<u8> = (0..rows * 3)
            .flat_map(|i| (((i / 3 * 7 + i % 3 * 41 + seed) % 128) as i32).to_ne_bytes()).collect();
        let gates: Vec<u8> = (0..rows * 3)
            .flat_map(|i| (0.15 + ((i + seed) % 3) as f32 * 0.1).to_ne_bytes()).collect();
        owner.run(|| {
            library.copy_h2d(hidden, &values(rows * 5120, 0.003, 0))?;
            library.copy_h2d(ids, &routes)?;
            library.copy_h2d(weights, &gates)?;
            library.copy_h2d(shared, &values(rows * 5120, 0.0007, 5))
        })
}

/// A round on one pair per stage, each stage reading the previous output.
///
/// # Safety
/// Every buffer stays live through the stream's and the pairs' drains.
unsafe fn round(pairs: &mut [Pair<'_>], inputs: &Inputs<'_>, rows: u32, stream: *mut c_void) -> Result<()> {
    for stage in 0..pairs.len() {
        let hidden = if stage == 0 { inputs.hidden.buffer } else { pairs[stage - 1].output() };
        unsafe {
            pairs[stage].enqueue(stage, rows, [hidden, inputs.ids.buffer, inputs.weights.buffer],
                Some(inputs.shared.buffer), stream)?;
        }
    }
    Ok(())
}

fn drain(stream: &Stream<'_>, pairs: &[Pair<'_>]) -> Result<()> {
    let mut result = stream.drain();
    for pair in pairs { result = result.and(pair.drain_peer()); }
    result
}

fn output(pair: &Pair<'_>, owner: Device<'_>, rows: u32) -> Result<Vec<u8>> {
    let mut bytes = vec![0; rows as usize * 10240];
    owner.run(|| owner.library.copy_d2h(&mut bytes, pair.output()))?;
    Ok(bytes)
}

/// Captures `enqueue` on `stream` into a graph executable.
fn capture(library: &NativeLibrary, owner: Device<'_>, stream: *mut c_void,
    enqueue: impl FnOnce() -> Result<()>) -> Result<*mut c_void> {
    owner.run(|| unsafe {
        library.cuda_graph_begin_capture(stream)?;
        let queued = enqueue();
        let captured = library.cuda_graph_end_capture(stream);
        match (queued, captured) {
            (Ok(()), Ok(graph)) => Ok(graph),
            (Err(error), Ok(graph)) => { library.cuda_graph_exec_destroy(graph)?; Err(error) }
            (Err(error), Err(_)) | (Ok(()), Err(error)) => Err(error),
        }
    })
}

fn pairs<'a>(weights: &[std::rc::Rc<Weights<'a>>; 2], push: bool) -> Result<Vec<Pair<'a>>> {
    (0..super::LAYOUT.stages).map(|_| {
        let mut pair = Pair::new(weights.clone(), CAPACITY, super::LAYOUT)?;
        if push {
            pair.enable_push(v41_peer_push::lane_bytes_with(v41_peer_push::DECODE_ROWS, super::LAYOUT.route_bytes(), None)?)?;
        }
        Ok(pair)
    }).collect()
}

/// The pushing pair's round output equals the pulling pair's, byte for byte,
/// at 1, 7 and 16 rows: eagerly, then from one captured graph replayed with
/// new inputs (the positional flags pair each replay's waits with its pushes).
#[test]
#[ignore = "requires the native draft TP2 AOT and two CUDA GPUs with peer access"]
fn draft_pair_push_matches_pull_eager_and_replayed() -> Result<()> {
    let lib = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
    let devices = [Device { library: &lib, id: 0 }, Device { library: &lib, id: 1 }];
    let owner = devices[super::LAYOUT.owner];
    let weights = Weights::synthetic(devices, super::LAYOUT, 0x5eed)?;
    let (mut pulled, mut pushed) = (pairs(&weights, false)?, pairs(&weights, true)?);
    ensure!(pushed.iter().all(Pair::pushes) && !pulled.iter().any(Pair::pushes), "pair modes differ from the setup");
    let inputs = Inputs::new(owner)?;
    let streams = [Stream::new(owner)?, Stream::new(owner)?];
    let mut compared = 0;
    for rows in ROWS {
        for seed in [0usize, 7] {
            inputs.fill_own(seed)?;
            let mut outputs = Vec::new();
            for (pairs, stream) in [(&mut pulled, &streams[0]), (&mut pushed, &streams[1])] {
                let queued = unsafe { round(pairs, &inputs, rows, stream.raw) };
                queued.and(drain(stream, pairs))?;
                outputs.push(output(pairs.last().unwrap(), owner, rows)?);
            }
            ensure!(outputs[0].iter().any(|&b| b != 0), "draft round output is all zero at {rows} rows");
            ensure!(outputs[0] == outputs[1], "pushed draft round differs from the pull at {rows} rows, seed {seed}");
            compared += 1;
        }
        let graph = capture(&lib, owner, streams[1].raw, || unsafe { round(&mut pushed, &inputs, rows, streams[1].raw) })?;
        let checked = (|| -> Result<()> {
            for seed in [11usize, 23, 41] {
                inputs.fill_own(seed)?;
                let queued = unsafe { round(&mut pulled, &inputs, rows, streams[0].raw) };
                queued.and(drain(&streams[0], &pulled))?;
                let expected = output(pulled.last().unwrap(), owner, rows)?;
                owner.run(|| unsafe { lib.cuda_graph_launch(graph, streams[1].raw) })?;
                drain(&streams[1], &pushed)?;
                ensure!(output(pushed.last().unwrap(), owner, rows)? == expected,
                    "replayed pushed draft round differs from the pull at {rows} rows, seed {seed}");
                compared += 1;
            }
            Ok(())
        })();
        let drained = drain(&streams[1], &pushed);
        owner.run(|| unsafe { lib.cuda_graph_exec_destroy(graph) })?;
        drained?;
        checked?;
    }
    eprintln!("draft TP2 rounds: push == pull byte for byte in {compared} rounds (eager and replayed), rows {ROWS:?}");
    Ok(())
}

/// One timed variant of a round's routed experts.
#[derive(Clone, Copy)]
enum Variant { Single, Pull, Push }

/// Queues `variant`'s three stages on `stream`.
///
/// # Safety
/// Every buffer stays live through the stream's and the pairs' drains.
unsafe fn enqueue(variant: Variant, owner: Device<'_>, single: &mut [crate::shared::memory::device::DeviceOwner<'_,
    crate::families::deepseek_v41::v41_experts::ExpertExecution<'_, '_>>], pulled: &mut [Pair<'_>],
    pushed: &mut [Pair<'_>], inputs: &Inputs<'_>, rows: u32, stream: *mut c_void) -> Result<()> {
    match variant {
        Variant::Single => owner.run(|| {
            for execution in single.iter_mut() { unsafe { execution.launch_on(rows, true, stream)?; } }
            Ok(())
        }),
        Variant::Pull => unsafe { round(pulled, inputs, rows, stream) },
        Variant::Push => unsafe { round(pushed, inputs, rows, stream) },
    }
}

/// Times a three-stage round's routed experts (the part the drafters differ
/// in; router and shared FFN are the same) replayed from a graph, as the
/// draft chain replays it: the single-card drafter (`DraftExperts::Full`)
/// against the TP2 pair pulling and pushing, at 1, 7 and 16 rows. Reports a
/// replay's latency (start event to end event around one launch, so graph
/// submission included) and the per-round time of back-to-back replays.
/// Idle GPUs: a serving lane's contention on the peer GPU is not in these.
#[test]
#[ignore = "requires the native draft TP2 AOT and two CUDA GPUs with peer access; prints timings"]
fn draft_round_tp2_push_against_single_card() -> Result<()> {
    const WARM: usize = 20;
    const TIMED: usize = 200;
    const BACK_TO_BACK: usize = 100;
    let lib = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
    let devices = [Device { library: &lib, id: 0 }, Device { library: &lib, id: 1 }];
    let owner = devices[super::LAYOUT.owner];
    let weights = Weights::synthetic(devices, super::LAYOUT, 0x5eed)?;
    let full = (0..3).map(|stage| owner.own(|| ExpertWeights::synthetic(&lib, ExpertLayer::Dspark { stage }, 0xf011 + stage as u64)))
        .collect::<Result<Vec<_>>>()?;
    let budget = ExpertWeights::plan_execution(ExpertLayer::Dspark { stage: 0 }, &lib, CAPACITY, CAPACITY, false)?.total()?;
    let mut single = full.iter().map(|weights| owner.own(|| weights.execution(CAPACITY, CAPACITY, budget)))
        .collect::<Result<Vec<_>>>()?;
    for execution in &single {
        fill(owner, execution.inputs(), execution.shared().expect("draft execution owns its shared output"), 3)?;
    }
    let (mut pulled, mut pushed) = (pairs(&weights, false)?, pairs(&weights, true)?);
    let inputs = Inputs::new(owner)?;
    inputs.fill_own(3)?;
    let stream = Stream::new(owner)?;
    let events = [owner.run(|| lib.cuda_event_create())?, owner.run(|| lib.cuda_event_create())?];
    let mut table = Vec::new();
    let result = (|| -> Result<()> {
        for rows in ROWS {
            let mut row = Vec::new();
            for variant in [Variant::Single, Variant::Pull, Variant::Push] {
                // Eager first, so capture meets initialized modules only.
                let queued = unsafe { enqueue(variant, owner, &mut single, &mut pulled, &mut pushed, &inputs, rows, stream.raw) };
                queued.and(drain(&stream, &pulled)).and(drain(&stream, &pushed))?;
                let graph = capture(&lib, owner, stream.raw, || unsafe {
                    enqueue(variant, owner, &mut single, &mut pulled, &mut pushed, &inputs, rows, stream.raw)
                })?;
                let timed = (|| -> Result<(Vec<f32>, f32)> {
                    let mut samples = Vec::with_capacity(TIMED);
                    for i in 0..WARM + TIMED {
                        owner.run(|| unsafe {
                            lib.cuda_event_record(events[0], stream.raw)?;
                            lib.cuda_graph_launch(graph, stream.raw)?;
                            lib.cuda_event_record(events[1], stream.raw)
                        })?;
                        drain(&stream, &pulled).and(drain(&stream, &pushed))?;
                        if i >= WARM { samples.push(unsafe { lib.cuda_event_elapsed_ms(events[0], events[1])? } * 1000.0); }
                    }
                    owner.run(|| unsafe {
                        lib.cuda_event_record(events[0], stream.raw)?;
                        for _ in 0..BACK_TO_BACK { lib.cuda_graph_launch(graph, stream.raw)?; }
                        lib.cuda_event_record(events[1], stream.raw)
                    })?;
                    drain(&stream, &pulled).and(drain(&stream, &pushed))?;
                    let per_round = unsafe { lib.cuda_event_elapsed_ms(events[0], events[1])? } * 1000.0 / BACK_TO_BACK as f32;
                    Ok((samples, per_round))
                })();
                let drained = drain(&stream, &pulled).and(drain(&stream, &pushed));
                owner.run(|| unsafe { lib.cuda_graph_exec_destroy(graph) })?;
                drained?;
                let (mut samples, per_round) = timed?;
                samples.sort_by(f32::total_cmp);
                let at = |q: f64| samples[((samples.len() - 1) as f64 * q).round() as usize];
                row.push((at(0.5), at(0.1), at(0.9), per_round));
            }
            table.push((rows, row));
        }
        Ok(())
    })();
    for event in events { owner.run(|| unsafe { lib.cuda_event_destroy(event) })?; }
    drop(single);
    result?;
    eprintln!("three-stage draft round, routed experts, graph replay on idle GPUs, {TIMED} timed replays:");
    eprintln!("latency = median (p10..p90) of one launch, start to end event; b2b = per round over {BACK_TO_BACK} back-to-back replays");
    eprintln!("| rows | single card latency / b2b | TP2 pull latency / b2b | TP2 push latency / b2b | push / single (latency, b2b) |");
    eprintln!("|---:|---:|---:|---:|---:|");
    for (rows, row) in &table {
        let cell = |(median, p10, p90, b2b): (f32, f32, f32, f32)| format!("{median:.1} ({p10:.1}..{p90:.1}) / {b2b:.1}");
        eprintln!("| {rows} | {} | {} | {} | {:.2}, {:.2} |", cell(row[0]), cell(row[1]), cell(row[2]),
            row[2].0 / row[0].0, row[2].3 / row[0].3);
    }
    Ok(())
}
