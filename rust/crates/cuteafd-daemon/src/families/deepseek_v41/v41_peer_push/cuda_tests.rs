//! Two-GPU byte identity of every converted decode site: the same synthetic
//! inputs through the pull path and the push path give the same bytes, for
//! decode row counts 1, 6, 17, 48 and 64 (`CUTEAFD_V41_PEER_EXCHANGE`).
use super::lane_bytes_with;
use crate::shared::experts::rtx::combine::{Combine, OwnerReduce};
use crate::shared::experts::rtx::{PartialDtype, PartialLayout};
use crate::shared::memory::chain::StageChain;
use crate::shared::memory::device::{Allocation, Device, Event, PeerTransfer, Stream};
use crate::shared::peer_split::{hop::DirectInputHop, FlagRoute, PeerExchange};
use anyhow::{ensure, Result};
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary};
use std::time::Duration;

const ROWS: [usize; 5] = [1, 6, 17, 48, 64];
const LANE_ROWS: usize = 64;

fn library() -> Result<NativeLibrary> {
    // SAFETY: the explicitly selected native library outlives every owner below.
    let library = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
    library.cuda_set_device(0)?;
    for gpu in 0..2 {
        Device { library: &library, id: gpu }.run(|| library.cuda_enable_peer(1 - gpu))?;
    }
    Ok(library)
}

/// Finite FP32 or BF16-safe bytes: values in [-2, 2) on a 1/64 grid.
fn floats(count: usize, seed: usize) -> Vec<u8> {
    (0..count).flat_map(|i| ((((i * 37 + seed * 101) % 256) as f32 - 128.0) / 64.0).to_ne_bytes()).collect()
}

fn bf16s(count: usize, seed: usize) -> Vec<u8> {
    (0..count).flat_map(|i| {
        let value = (((i * 53 + seed * 7) % 256) as f32 - 128.0) / 64.0;
        ((value.to_bits() >> 16) as u16).to_ne_bytes()
    }).collect()
}

fn read(device: Device<'_>, buffer: CuteafdDeviceBuffer, bytes: usize) -> Result<Vec<u8>> {
    let mut out = vec![0; bytes];
    device.run(|| device.library.copy_d2h(&mut out, CuteafdDeviceBuffer { bytes, ..buffer }))?;
    Ok(out)
}

fn write(device: Device<'_>, buffer: CuteafdDeviceBuffer, bytes: &[u8]) -> Result<()> {
    device.run(|| device.library.copy_h2d(CuteafdDeviceBuffer { bytes: bytes.len(), ..buffer }, bytes))
}

fn ordered_chain(library: &NativeLibrary) -> Result<StageChain<'_>> {
    let chain = StageChain::on_devices(library, &[0, 1])?;
    chain.set_device_order(true);
    Ok(chain)
}

/// A1: the 19->20 hop, residual and `pre`, GPU0 -> GPU1 block inputs.
#[test]
#[ignore = "requires CUTEAFD_NATIVE_LIB and two CUDA GPUs with peer access"]
fn hop_push_matches_pull() -> Result<()> {
    let library = library()?;
    let devices = [Device { library: &library, id: 0 }, Device { library: &library, id: 1 }];
    let spec = cuteafd_loader::placement::families::deepseek_v41::execution_hops(2);
    let mut pulled = DirectInputHop::new(devices[0], devices[1], spec)?;
    let mut pushed = DirectInputHop::new(devices[0], devices[1], spec)?;
    pushed.enable_push(lane_bytes_with(LANE_ROWS, spec.row_bytes as usize, None)?)?;
    let residual = Allocation::new(devices[0], LANE_ROWS * 40960)?;
    let pre = Allocation::new(devices[0], LANE_ROWS * 16)?;
    let inputs = [Allocation::new(devices[1], LANE_ROWS * 40960)?, Allocation::new(devices[1], LANE_ROWS * 16)?];
    let chain = ordered_chain(&library)?;
    let runtime = tokio::runtime::Builder::new_current_thread().build()?;
    for rows in ROWS {
        write(devices[0], residual.buffer, &bf16s(LANE_ROWS * 20480, rows))?;
        write(devices[0], pre.buffer, &floats(LANE_ROWS * 4, rows + 1))?;
        let mut landed = Vec::new();
        for hop in [&mut pulled, &mut pushed] {
            for input in &inputs { write(devices[1], input.buffer, &vec![0xc3; input.buffer.bytes])?; }
            let sources = [CuteafdDeviceBuffer { bytes: rows * 40960, ..residual.buffer },
                CuteafdDeviceBuffer { bytes: rows * 16, ..pre.buffer }];
            let destinations = inputs.each_ref().map(|input| input.buffer);
            runtime.block_on(chain.handle().scope(async { unsafe { hop.copy(sources, destinations).await } }))?;
            chain.drain_bounded(Duration::from_secs(10))?;
            library.cuda_device_synchronize()?;
            landed.push([read(devices[1], inputs[0].buffer, inputs[0].buffer.bytes)?,
                read(devices[1], inputs[1].buffer, inputs[1].buffer.bytes)?]);
        }
        ensure!(landed[0] == landed[1], "A1 push differs from pull at {rows} rows");
        ensure!(landed[1][0][..rows * 40960] == read(devices[0], residual.buffer, rows * 40960)?[..]
            && landed[1][0][rows * 40960..].iter().all(|&b| b == 0xc3), "A1 push moved the wrong residual rows");
    }
    eprintln!("A1 hop: push == pull at rows {ROWS:?}");
    Ok(())
}

/// A3: the routed partial, remote rank -> owner, reduced by V4.1's reducer;
/// FP32 token sums and route planes, both owners.
#[test]
#[ignore = "requires CUTEAFD_NATIVE_LIB and two CUDA GPUs with peer access"]
fn routed_owner_reduce_push_matches_pull() -> Result<()> {
    let library = library()?;
    let devices = [Device { library: &library, id: 0 }, Device { library: &library, id: 1 }];
    let geometry = cuteafd_core::ExpertGeometry::DEEPSEEK_V41;
    let (hidden, topk) = (geometry.hidden as usize, geometry.topk as usize);
    let planes = [Allocation::new(devices[0], LANE_ROWS * hidden * topk * 4)?,
        Allocation::new(devices[1], LANE_ROWS * hidden * topk * 4)?];
    let producers = [Stream::new(devices[0])?, Stream::new(devices[1])?];
    let chain = ordered_chain(&library)?;
    let runtime = tokio::runtime::Builder::new_current_thread().build()?;
    let mut cases = 0;
    for owner in [0usize, 1] {
        let policy = Combine::OwnerReduce { owner: owner as u8 };
        let mut pulled = OwnerReduce::new(devices, policy, LANE_ROWS as u32, geometry)?;
        let mut pushed = OwnerReduce::new(devices, policy, LANE_ROWS as u32, geometry)?;
        pushed.enable_push(lane_bytes_with(LANE_ROWS, pushed.partial_row_bytes(), None)?)?;
        for layout in [PartialLayout::Tokens, PartialLayout::Routes { topk }] {
            for rows in ROWS {
                for (rank, plane) in planes.iter().enumerate() {
                    write(devices[rank], plane.buffer, &floats(LANE_ROWS * hidden * topk, rows * 3 + rank))?;
                }
                let mut outputs = Vec::new();
                for reduce in [&mut pulled, &mut pushed] {
                    let output = runtime.block_on(chain.handle().scope(async { unsafe {
                        reduce.reduce(&planes[owner], &planes[1 - owner], &producers[owner], &producers[1 - owner],
                            rows as u32, PartialDtype::F32, layout).await
                    } }))?;
                    chain.drain_bounded(Duration::from_secs(10))?;
                    library.cuda_device_synchronize()?;
                    outputs.push(read(devices[owner], output, rows * hidden * 2)?);
                }
                ensure!(outputs[0] == outputs[1], "A3 push differs from pull: owner {owner} {layout:?} rows {rows}");
                cases += 1;
            }
        }
    }
    eprintln!("A3 routed owner reduce: push == pull in {cases} cases");
    Ok(())
}

/// A4 (into GPU0) and A6 (into GPU1): the shared half's partial pushed into
/// the destination's staging and added to the local half there, as the
/// shared TP2 wave queues it.
#[test]
#[ignore = "requires CUTEAFD_NATIVE_LIB and two CUDA GPUs with peer access"]
fn shared_partial_push_matches_pull() -> Result<()> {
    let library = library()?;
    let devices = [Device { library: &library, id: 0 }, Device { library: &library, id: 1 }];
    let add = library.bf16_add()?;
    let chain = ordered_chain(&library)?;
    let runtime = tokio::runtime::Builder::new_current_thread().build()?;
    let mut cases = 0;
    for destination in [0usize, 1] {
        let remote = 1 - destination;
        let mut pulled = PeerTransfer::new(devices[remote], devices[destination])?;
        let mut pushed = PeerTransfer::new(devices[remote], devices[destination])?;
        pushed.enable_push(Box::new(FlagRoute::new(&library, [remote as i32, destination as i32],
            lane_bytes_with(LANE_ROWS, 10240, None)?)?));
        let partial = Allocation::new(devices[remote], LANE_ROWS * 10240)?;
        let local = Allocation::new(devices[destination], LANE_ROWS * 10240)?;
        let mut staging = Allocation::new(devices[destination], LANE_ROWS * 10240)?;
        let output = Allocation::new(devices[destination], LANE_ROWS * 10240)?;
        let producers = [Stream::new(devices[remote])?, Stream::new(devices[destination])?];
        let mut ready = Event::new(devices[destination])?;
        for rows in ROWS {
            write(devices[remote], partial.buffer, &bf16s(LANE_ROWS * 5120, rows))?;
            write(devices[destination], local.buffer, &bf16s(LANE_ROWS * 5120, rows + 9))?;
            let mut outputs = Vec::new();
            for transfer in [&mut pulled, &mut pushed] {
                ready.record(&producers[1])?;
                let ready_raw = ready.raw;
                let (local_buffer, output_buffer) = (local.buffer, output.buffer);
                runtime.block_on(chain.handle().scope(async { unsafe {
                    transfer.copy_then(&partial, &mut staging, &producers[0], rows * 10240, |peer, stream| {
                        library.cuda_stream_wait_event(stream, ready_raw)?;
                        add.launch(local_buffer, peer, output_buffer, rows * 5120, stream)
                    }).await
                } }))?;
                chain.drain_bounded(Duration::from_secs(10))?;
                library.cuda_device_synchronize()?;
                outputs.push(read(devices[destination], output.buffer, rows * 10240)?);
            }
            ensure!(outputs[0] == outputs[1], "shared partial push differs from pull: destination {destination} rows {rows}");
            cases += 1;
        }
    }
    eprintln!("A4/A6 shared partial: push == pull in {cases} cases");
    Ok(())
}

/// A8 and A9: the chained head's transfers with its geometry. Rank 1's
/// normalized rows land in rank 0's input; rank 0's winner block lands in
/// the merge's block and the real merge kernel reduces both.
#[test]
#[ignore = "requires CUTEAFD_NATIVE_LIB and two CUDA GPUs with peer access"]
fn head_push_matches_pull() -> Result<()> {
    let library = library()?;
    let devices = [Device { library: &library, id: 0 }, Device { library: &library, id: 1 }];
    let capacity = cuteafd_core::V41_VERIFY_ROWS;
    let split = 64640;
    let exchange = PeerExchange::new_flags(&library, [0, 1], 2, lane_bytes_with(capacity, 10240, None)?)?;
    let copy = devices[0].run(|| library.v41_peer_copy())?;
    let merge_copy = devices[1].run(|| library.v41_peer_copy())?;
    let streams = [Stream::new(devices[0])?, Stream::new(devices[1])?];
    let normalized = Allocation::new(devices[1], capacity * 10240)?;
    let input = Allocation::new(devices[0], capacity * 10240)?;
    let winners = [Allocation::new(devices[0], capacity * 8)?, Allocation::new(devices[1], capacity * 8)?];
    let remote = Allocation::new(devices[1], capacity * 8)?;
    let merged = Allocation::new(devices[1], capacity * 8)?;
    let slice = |buffer: CuteafdDeviceBuffer, offset: usize, bytes: usize| CuteafdDeviceBuffer {
        ptr: unsafe { buffer.ptr.cast::<u8>().add(offset).cast() }, bytes, ..buffer };
    for rows in ROWS.into_iter().filter(|&rows| rows <= capacity) {
        write(devices[1], normalized.buffer, &bf16s(capacity * 5120, rows))?;
        for (rank, block) in winners.iter().enumerate() {
            let ids: Vec<u8> = (0..capacity).flat_map(|i| (((i * 977 + rank * 31 + rows) % split) as u32).to_ne_bytes()).collect();
            let scores: Vec<u8> = (0..capacity).flat_map(|i| ((((i * 13 + rank * 5 + rows) % 17) as f32) - 8.0).to_ne_bytes()).collect();
            write(devices[rank], block.buffer, &[ids, scores].concat())?;
        }
        let mut results = Vec::new();
        for push in [false, true] {
            write(devices[0], input.buffer, &vec![0x3c; capacity * 10240])?;
            write(devices[1], remote.buffer, &vec![0x3c; capacity * 8])?;
            let ids = |block: CuteafdDeviceBuffer| slice(block, 0, rows * 4);
            let scores = |block: CuteafdDeviceBuffer| slice(block, capacity * 4, rows * 4);
            unsafe {
                if push {
                    exchange.push_to_on(1, streams[1].raw, 0, normalized.buffer.ptr, input.buffer.ptr, rows * 10240)?;
                    exchange.wait_on(0, streams[0].raw, 0)?;
                    exchange.push_to_on(0, streams[0].raw, 1, winners[0].buffer.ptr, remote.buffer.ptr, capacity * 8)?;
                    exchange.wait_on(1, streams[1].raw, 1)?;
                } else {
                    devices[0].run(|| copy.launch(input.buffer, normalized.buffer, rows * 10240, streams[0].raw))?;
                    devices[1].run(|| {
                        merge_copy.launch(ids(remote.buffer), ids(winners[0].buffer), rows * 4, streams[1].raw)?;
                        merge_copy.launch(scores(remote.buffer), scores(winners[0].buffer), rows * 4, streams[1].raw)
                    })?;
                }
                devices[1].run(|| library.v41_vocabulary_merge_greedy(
                    [(ids(remote.buffer), scores(remote.buffer)), (ids(winners[1].buffer), scores(winners[1].buffer))],
                    (ids(merged.buffer), scores(merged.buffer)), rows, split, streams[1].raw))?;
            }
            streams[0].drain()?;
            streams[1].drain()?;
            results.push((read(devices[0], input.buffer, rows * 10240)?,
                [read(devices[1], ids(merged.buffer), rows * 4)?, read(devices[1], scores(merged.buffer), rows * 4)?].concat()));
        }
        ensure!(results[0] == results[1], "A8/A9 push differs from pull at {rows} rows");
    }
    eprintln!("A8/A9 head: push == pull at rows {ROWS:?}");
    Ok(())
}
