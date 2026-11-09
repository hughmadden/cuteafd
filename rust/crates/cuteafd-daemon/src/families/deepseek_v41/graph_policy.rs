//! Captured shapes per backbone layer, sharing the containing lane's buffers.
use anyhow::{ensure, Result};

pub(super) const MAX_DECODE_ROWS: u32 = cuteafd_core::V41_VERIFY_ROWS as u32;

/// Rows of one full prefill chunk. Above the decode bank a live request captures
/// only this and the 128-row decoder replay, the two large shapes every long
/// prompt repeats. Unset (tests, tools), every shape keeps capturing.
static PREFILL_ROWS: std::sync::OnceLock<u32> = std::sync::OnceLock::new();

/// Startup: register the serving prefill chunk. A prompt's tail chunk has an
/// odd row count that no later request repeats, so it executes eagerly instead
/// of instantiating one single-use graph per layer and site (#410 A).
pub(super) fn set_prefill_rows(rows: u32) -> Result<()> {
    ensure!(rows > 0, "prefill chunk rows must be positive");
    match PREFILL_ROWS.get() {
        Some(&set) => ensure!(set == rows, "prefill chunk rows already set to {set}"),
        None => { let _ = PREFILL_ROWS.set(rows); }
    }
    Ok(())
}

pub(crate) fn pinned_shape(rows: u32) -> bool {
    static PINNED: std::sync::OnceLock<Vec<usize>> = std::sync::OnceLock::new();
    PINNED.get_or_init(|| cuteafd_loader::serving_capacity::v41_graphs::v41_pinned_rows(
        cuteafd_core::V41_VERIFY_ROWS)).contains(&(rows as usize))
        || rows == cuteafd_core::coordinator_programs::V41_DECODER_REPLAY_ROWS
        || PREFILL_ROWS.get().is_some_and(|&chunk| rows == chunk)
}

pub(super) fn captures_shape(rows: u32) -> bool {
    large_shape_captured(rows, PREFILL_ROWS.get().copied())
}

/// Only slot zero serves non-unit prompt, replay and continuation work.
pub(crate) const PREFILL_PREWARM_SLOTS: [usize; 1] = [0];
/// The terminal projects the selected last row, not the whole backbone chunk.
pub(crate) const PREFILL_HEAD_PREWARM_ROWS: [usize; 1] = [1];

/// Nonwindowed prompt/replay paths reach these exact row counts. Streamed
/// encoder units have offset bindings and deliberately remain eager (#1638).
pub(crate) fn prefill_prewarm_rows(capacity: usize) -> Vec<usize> {
    prefill_rows(capacity, PREFILL_ROWS.get().copied())
}

fn prefill_rows(capacity: usize, chunk: Option<u32>) -> Vec<usize> {
    let mut rows = vec![cuteafd_core::coordinator_programs::V41_DECODER_REPLAY_ROWS as usize];
    if let Some(chunk) = chunk { rows.push(chunk as usize); }
    rows.retain(|&row| row <= capacity);
    rows.sort_unstable();
    rows.dedup();
    rows
}

/// Records one eager run of a shape the policy would graph but this owner's
/// bank cannot capture. Every site that chooses eager for that reason reports
/// through here, so `/v1/stats["graph_missing_variants"]` holds one entry per
/// variant and the first execution logs one structured warning.
pub(super) fn note_miss(shape: String, context: impl FnOnce() -> String) {
    crate::shared::decode_graph::note_eager("deepseek_v41", &shape, context);
}

fn large_shape_captured(rows: u32, prefill: Option<u32>) -> bool {
    rows <= MAX_DECODE_ROWS || rows == cuteafd_core::coordinator_programs::V41_DECODER_REPLAY_ROWS
        || prefill.is_none_or(|chunk| rows == chunk)
}

fn site_name(name: &'static str) -> &'static str {
    match name {
        "window" => "deepseek_v41.window", "compressor" => "deepseek_v41.compressor",
        "index_selection" => "deepseek_v41.index_selection",
        "target_head" => "deepseek_v41.target_head", "target_norm" => "deepseek_v41.target_norm",
        "target_embedding" => "deepseek_v41.target_embedding", "tp2_ffn" => "deepseek_v41.tp2_ffn",
        "vocabulary_logits" => "deepseek_v41.vocabulary_logits", "vocabulary_greedy" => "deepseek_v41.vocabulary_greedy",
        "sparse_attention" => "deepseek_v41.sparse_attention", "dual_attention_tail" => "deepseek_v41.dual_attention_tail",
        "v41.attention.query" => "deepseek_v41.attention.query",
        "v41.attention.query.tp2_prefix" => "deepseek_v41.attention.query.tp2_prefix",
        "v41.attention.output" => "deepseek_v41.attention.output",
        "v41.backbone.shared" => "deepseek_v41.backbone.shared",
        "v41.router.remote" => "deepseek_v41.router.remote",
        "v41.router.local" => "deepseek_v41.router.local",
        "v41.index.query" => "deepseek_v41.index.query", "v41.engram.gate" => "deepseek_v41.engram.gate",
        "v41.projection.query.rank" => "deepseek_v41.projection.query.rank",
        "v41.projection.query.gather" => "deepseek_v41.projection.query.gather",
        "v41.projection.output.rank" => "deepseek_v41.projection.output.rank",
        "v41.projection.output.gather" => "deepseek_v41.projection.output.gather",
        "dspark_chain" => "deepseek_v41.dspark_chain",
        "dspark_cache" => "deepseek_v41.dspark_cache",
        "dspark_distributed_terminal" => "deepseek_v41.dspark_distributed_terminal",
        // Empty standalone banks are visited too; they have no serving capture sites.
        _ => name,
    }
}

pub(crate) fn attach(library: &cuteafd_ffi::NativeLibrary, ranks: usize,
    mut visit: impl FnMut(&mut crate::shared::decode_graph::device::GraphVisitor<'_>) -> Result<()>) -> Result<()> {
    use crate::shared::decode_graph::device::{Basis, Cycle, Preference, Site};
    let _ = library;
    tracing::warn!("V4.1 graph cache uses byte estimates: SM120 node rates are not calibrated");
    let (base, extra) = cuteafd_loader::placement::graph_reserve::v41_default();
    visit(&mut |device, name, spans, bank| {
        let reserve = cuteafd_loader::placement::graph_reserve::current(device as usize,
            false, base, extra).map_err(anyhow::Error::msg)?.bytes;
        bank.attach(device, reserve, Site { name: site_name(name), cycle: Cycle::Other,
            preference: Preference::Adaptive, bootstrap: extra, basis: Basis::Bytes,
            spans_devices: spans || ranks > 1 })
    })
}

pub(crate) fn cycle(cycle: crate::shared::decode_graph::device::Cycle,
    mut visit: impl FnMut(&mut crate::shared::decode_graph::device::GraphVisitor<'_>) -> Result<()>) -> Result<()> {
    visit(&mut |_, _, _, bank| { bank.cycle(cycle); Ok(()) })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::decode_graph::{LayerGraphs, RowGraphs};
    use cuteafd_ffi::NativeLibrary;
    use std::collections::BTreeMap;
    use crate::shared::memory::{DeviceAllocation, LoadStream};

    #[test]
    fn startup_prefill_keys_are_reachable_from_slot_zero_and_selected_head_row() {
        use crate::shared::memory::row_window::RowWindow;
        // Enumerate the geometry consumed by startup, including a chunk that
        // coincides with replay and a capacity that excludes the chunk.
        for (capacity, chunk, expected) in [(2048, 2048, vec![128, 2048]),
            (128, 128, vec![128]), (128, 2048, vec![128])] {
            let rows = prefill_rows(capacity, Some(chunk));
            assert_eq!(rows, expected);
            let keys: Vec<_> = PREFILL_PREWARM_SLOTS.into_iter().flat_map(|slot| {
                rows.iter().map(move |&rows| (slot, rows, RowWindow::NONE, false))
            }).collect();
            assert!(!keys.is_empty());
            for (slot, rows, window, attention_decode) in keys {
                assert_eq!(slot, 0);
                assert!(!window.is_windowed());
                assert!(!attention_decode);
                assert!(rows == 128 || rows == chunk as usize);
            }
            let head_keys: Vec<_> = PREFILL_PREWARM_SLOTS.into_iter()
                .flat_map(|slot| PREFILL_HEAD_PREWARM_ROWS.map(|rows| (slot, rows))).collect();
            assert_eq!(head_keys, vec![(0, 1)]);
        }
        // Tie the enumeration to both live startup implementations rather than
        // letting a correct policy test hide an unreachable caller inventory.
        for source in [include_str!("v41_native_serve.rs"), include_str!("v41_native_serve/distributed.rs")] {
            let at = source.find("for index in ").unwrap();
            let end = source[at..].find("if let Some(draft)").unwrap() + at;
            let startup = &source[at..end];
            assert!(startup.contains("graph_policy::PREFILL_PREWARM_SLOTS"));
            assert!(startup.contains("&mut prefill_slots.slots[index]"));
            assert_eq!(source.matches("slot.capture_startup_prefill(").count(), 1);
        }
        for source in [include_str!("v41_target_pass.rs"), include_str!("v41_target_pass/distributed.rs")] {
            let at = source.find("pub(crate) fn capture_startup_prefill").unwrap();
            let end = source[at..].find("pub(crate) fn capture_startup(").unwrap() + at;
            let startup = &source[at..end];
            assert!(startup.contains("select_window(crate::shared::memory::row_window::RowWindow::NONE)?"));
            assert!(startup.contains("graph_policy::PREFILL_HEAD_PREWARM_ROWS"));
            assert!(!startup.contains("head.capture_startup(&rows)"));
        }
        let lane = include_str!("v41_backbone_lane.rs");
        let at = lane.find("pub(crate) fn capture_startup_prefill").unwrap();
        let end = lane[at..].find("pub(crate) fn startup_prefill_rows").unwrap() + at;
        assert!(lane[at..end].contains("capture_startup_kind(&[], &[], false, false, local)"));
    }

    #[test]
    fn prefill_tail_chunks_execute_eagerly_while_decode_and_full_chunks_capture() {
        for rows in [1, 6, 38, MAX_DECODE_ROWS] {
            assert!(large_shape_captured(rows, Some(2048)));
        }
        assert!(large_shape_captured(2048, Some(2048)));
        assert!(large_shape_captured(128, Some(2048)));
        for rows in [MAX_DECODE_ROWS + 1, 1982, 2047] {
            assert!(!large_shape_captured(rows, Some(2048)));
        }
        assert!(large_shape_captured(1982, None));
    }

    #[test]
    fn cuda_row_bank_preserves_decode_handles_over_prefill() -> Result<()> {
        let Some(path) = std::env::var_os("CUTEAFD_LAYER_GRAPH_TEST_LIBRARY") else {
            eprintln!("skip CUDA row bank test: library unset");
            return Ok(());
        };
        // SAFETY: this fixture uses the task's native library and owns all buffers.
        let library = unsafe { NativeLibrary::load(path)? };
        let input = DeviceAllocation::new(&library, 128 * 4, crate::shared::memory::TEST)?;
        let output = DeviceAllocation::new(&library, 128 * 4, crate::shared::memory::TEST)?;
        let stream = LoadStream { library: &library, raw: library.cuda_stream_create()? };
        let mut bank = RowGraphs::new(&library, "fixture", MAX_DECODE_ROWS as usize, "deepseek_v41");
        let mut handles = BTreeMap::new();
        for (cycle, rows) in (1..=MAX_DECODE_ROWS as usize)
            .chain([80, 128, 2, 6, 96, 3, 48, 64, 56, 63, 64]).enumerate() {
            let expected = vec![(cycle + 1) as u8; 128 * 4];
            library.copy_h2d(input.buffer, &expected)?;
            if bank.get(rows).is_none() {
                // SAFETY: the previous iteration drained all work before capture.
                unsafe {
                    library.cuda_graph_begin_capture(stream.raw)?;
                    library.copy_d2d_async(output.buffer, input.buffer, rows * 4, stream.raw)?;
                    let graph = library.cuda_graph_end_capture(stream.raw)?;
                    bank.insert(rows, graph)?;
                }
            }
            let graph = bank.get(rows).unwrap();
            if rows <= MAX_DECODE_ROWS as usize {
                assert_eq!(*handles.entry(rows).or_insert(graph), graph);
            }
            // SAFETY: buffers remain owned and exclusive until this launch drains.
            unsafe {
                library.cuda_graph_launch(graph, stream.raw)?;
                library.cuda_stream_synchronize(stream.raw)?;
            }
            let mut actual = vec![0; rows * 4];
            let mut view = output.buffer; view.bytes = actual.len();
            library.copy_d2h(&mut actual, view)?;
            assert_eq!(actual, expected[..actual.len()]);
        }
        // Two large shapes stay (chunk and decoder replay); a third evicts the oldest.
        assert!(bank.get(80).is_none());
        assert!(bank.get(128).is_some() && bank.get(96).is_some());
        assert_eq!(bank.len(), MAX_DECODE_ROWS as usize + crate::shared::decode_graph::LARGE_SHAPES);
        // SAFETY: every fixture launch was synchronized before destruction.
        unsafe { bank.clear()?; bank.clear()?; }
        assert_eq!(bank.len(), 0);
        Ok(())
    }

    #[test]
    fn cuda_small_shape_bank_replays_changes_and_bounds_large_shapes() -> Result<()> {
        let Some(path) = std::env::var_os("CUTEAFD_LAYER_GRAPH_TEST_LIBRARY") else {
            eprintln!("skip CUDA shape bank test: library unset");
            return Ok(());
        };
        let library = unsafe { NativeLibrary::load(path)? };
        let weights = DeviceAllocation::new(&library, 128 * 4, crate::shared::memory::TEST)?;
        let other = DeviceAllocation::new(&library, 128 * 4, crate::shared::memory::TEST)?;
        let output = DeviceAllocation::new(&library, 128 * 4, crate::shared::memory::TEST)?;
        let stream = LoadStream { library: &library, raw: library.cuda_stream_create()? };
        let mut bank = LayerGraphs::new(&library, MAX_DECODE_ROWS, "deepseek_v41");
        bank.enable_small_shapes();
        let mut handles = BTreeMap::new();
        for (cycle, rows) in (1..=MAX_DECODE_ROWS).chain([80, 128, 2, 6, 96, 3, 2, 6, 48, 64, 56, 63, 64]).enumerate() {
            let expected = vec![(cycle + 1) as u8; 128 * 4];
            library.copy_h2d(weights.buffer, &expected)?;
            if bank.get_shape(0, &weights, rows).is_none() {
                unsafe {
                    library.cuda_graph_begin_capture(stream.raw)?;
                    library.copy_d2d_async(output.buffer, weights.buffer, rows as usize * 4, stream.raw)?;
                    let graph = library.cuda_graph_end_capture(stream.raw)?;
                    bank.insert(0, &weights, rows, graph)?;
                }
            }
            let (graph, _) = bank.get_shape(0, &weights, rows).unwrap();
            if rows <= MAX_DECODE_ROWS {
                assert_eq!(*handles.entry(rows).or_insert(graph), graph);
            }
            assert!(bank.get_shape(0, &other, rows).is_none());
            unsafe {
                library.cuda_graph_launch(graph, stream.raw)?;
                library.cuda_stream_synchronize(stream.raw)?;
            }
            let mut actual = vec![0; rows as usize * 4];
            let mut view = output.buffer;
            view.bytes = actual.len();
            library.copy_d2h(&mut actual, view)?;
            assert_eq!(actual, expected[..actual.len()]);
            assert!(bank.len() <= MAX_DECODE_ROWS as usize + crate::shared::decode_graph::LARGE_SHAPES);
        }
        assert!(bank.get_shape(0, &weights, 80).is_none());
        assert!(bank.get_shape(0, &weights, 128).is_some() && bank.get_shape(0, &weights, 96).is_some());
        unsafe {
            library.cuda_graph_begin_capture(stream.raw)?;
            library.copy_d2d_async(output.buffer, other.buffer, 8, stream.raw)?;
            let graph = library.cuda_graph_end_capture(stream.raw)?;
            bank.insert(0, &other, 2, graph)?;
        }
        assert!(bank.get_shape(0, &weights, 2).is_none());
        assert!(bank.get_shape(0, &weights, 6).is_none());
        assert!(bank.get_shape(0, &other, 2).is_some());
        unsafe { bank.clear()?; }
        assert_eq!(bank.len(), 0);
        assert_eq!(bank.len(), 0);
        Ok(())
    }

    #[test]
    fn cuda_sealed_prefill_tails_replay_and_cold_shapes_match_eager() -> Result<()> {
        let Some(path) = std::env::var_os("CUTEAFD_LAYER_GRAPH_TEST_LIBRARY") else {
            eprintln!("skip CUDA sealed prefill test: library unset");
            return Ok(());
        };
        // SAFETY: the fixture pins its library, buffers and stream until drained.
        let library = unsafe { NativeLibrary::load(path)? };
        let bytes = 2752 * 4;
        let weights = DeviceAllocation::new(&library, bytes, crate::shared::memory::TEST)?;
        let output = DeviceAllocation::new(&library, bytes, crate::shared::memory::TEST)?;
        let stream = LoadStream { library: &library, raw: library.cuda_stream_create()? };
        let mut bank = LayerGraphs::new(&library, MAX_DECODE_ROWS, "deepseek_v41");
        bank.declare_startup(&[(0, 52), (0, 54)])?;
        for rows in [52, 54] {
            // SAFETY: records only real pinned allocations, without execution.
            unsafe {
                library.cuda_graph_begin_capture(stream.raw)?;
                library.copy_d2d_async(output.buffer, weights.buffer, rows * 4, stream.raw)?;
                let graph = library.cuda_graph_end_capture(stream.raw)?;
                bank.insert(0, &weights, rows as u32, graph)?;
            }
        }
        bank.finish_startup()?;
        let handles = [bank.get_shape(0, &weights, 52), bank.get_shape(0, &weights, 54)];
        for (cycle, rows) in [54, 52, 54, 52, 128, 2752, 53, 52].into_iter().enumerate() {
            let input = vec![(cycle + 1) as u8; bytes];
            let mut expected = vec![0xee; bytes];
            expected[..rows * 4].copy_from_slice(&input[..rows * 4]);
            library.copy_h2d(weights.buffer, &input)?;
            library.copy_h2d(output.buffer, &vec![0xee; bytes])?;
            assert!(!bank.capture_allowed(0, rows as u32));
            // SAFETY: every prior iteration drained before reusing this storage.
            unsafe {
                if let Some((graph, _)) = bank.get_shape(0, &weights, rows as u32) {
                    library.cuda_graph_launch(graph, stream.raw)?;
                } else {
                    library.copy_d2d_async(output.buffer, weights.buffer, rows * 4, stream.raw)?;
                }
                library.cuda_stream_synchronize(stream.raw)?;
            }
            let mut actual = vec![0; bytes];
            library.copy_d2h(&mut actual, output.buffer)?;
            assert_eq!(actual, expected);
            assert_eq!([bank.get_shape(0, &weights, 52), bank.get_shape(0, &weights, 54)], handles);
            assert_eq!(bank.len(), 2);
        }
        // SAFETY: all fixture launches drained before bank destruction.
        unsafe { bank.clear()?; }
        assert!(!bank.capture_allowed(0, 52));
        Ok(())
    }

    #[test]
    fn cuda_layer_graphs_reuse_storage_and_retain_exact_weight_bindings() -> Result<()> {
        let Some(path) = std::env::var_os("CUTEAFD_LAYER_GRAPH_TEST_LIBRARY") else {
            eprintln!("skip CUDA layer graph test: CUTEAFD_LAYER_GRAPH_TEST_LIBRARY unset");
            return Ok(());
        };
        let library = unsafe { NativeLibrary::load(path)? };
        // Real CUDA copies isolate graph ownership from SM120 FP8 arithmetic.
        let weights = (0..40)
            .map(|_| DeviceAllocation::new(&library, 4096 * 4, crate::shared::memory::TEST))
            .collect::<Result<Vec<_>>>()?;
        let replacement = DeviceAllocation::new(&library, 4096 * 4, crate::shared::memory::TEST)?;
        let streams = (0..2)
            .map(|_| {
                Ok(LoadStream {
                    library: &library,
                    raw: library.cuda_stream_create()?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let outputs = (0..2)
            .map(|_| DeviceAllocation::new(&library, 4096 * 4, crate::shared::memory::TEST))
            .collect::<Result<Vec<_>>>()?;
        let mut banks = [LayerGraphs::new(&library, MAX_DECODE_ROWS, "deepseek_v41"), LayerGraphs::new(&library, MAX_DECODE_ROWS, "deepseek_v41")];
        let mut captures = 0;
        let mut replays = 0;
        for rows in [1u32, 80, 4096] {
            for cycle in 0..2u32 {
                for layer in 0..40 {
                    let expected: Vec<u8> = (0..rows)
                        .flat_map(|row| (row + 4096 * (layer as u32 + 40 * cycle)).to_ne_bytes())
                        .collect();
                    library.copy_h2d(weights[layer].buffer, &expected)?;
                    let mut handles = [std::ptr::null_mut(); 2];
                    for lane in 0..2 {
                        let bank = &mut banks[lane];
                        let current = bank.get(layer, &weights[layer]);
                        if current.is_none_or(|(_, count)| count != rows) {
                            unsafe {
                                bank.remove(layer)?;
                                library.cuda_graph_begin_capture(streams[lane].raw)?;
                                library.copy_d2d_async(
                                    outputs[lane].buffer,
                                    weights[layer].buffer,
                                    expected.len(),
                                    streams[lane].raw,
                                )?;
                                let graph = library.cuda_graph_end_capture(streams[lane].raw)?;
                                bank.insert(layer, &weights[layer], rows, graph)?;
                            }
                            captures += 1;
                        } else {
                            // A repeated same-shape pass must keep the graph handle.
                            assert_eq!(cycle, 1);
                        }
                        let (graph, count) = bank.get(layer, &weights[layer]).unwrap();
                        assert_eq!(count, rows);
                        assert!(bank.get(layer, &replacement).is_none());
                        handles[lane] = graph;
                        unsafe {
                            library.cuda_graph_launch(graph, streams[lane].raw)?;
                        }
                        replays += 1;
                    }
                    for lane in 0..2 {
                        unsafe {
                            library.cuda_stream_synchronize(streams[lane].raw)?;
                        }
                        let mut actual = vec![0; expected.len()];
                        library.copy_d2h(&mut actual, outputs[lane].buffer)?;
                        assert_eq!(actual, expected);
                        assert_eq!(
                            banks[lane].get(layer, &weights[layer]).unwrap().0,
                            handles[lane]
                        );
                    }
                }
                assert!(banks
                    .iter()
                    .all(|bank| bank.len() == 40));
            }
        }
        assert_eq!(captures, 240);
        assert_eq!(replays, 480);
        let bank = &mut banks[0];
        let original = bank.get(0, &weights[0]).unwrap();
        // Reject invalid metadata without touching the existing graph.
        unsafe {
            assert!(bank.insert(40, &replacement, 1, original.0).is_err());
            assert!(bank.insert(0, &replacement, 0, original.0).is_err());
            assert!(bank.insert(0, &replacement, 4097, original.0).is_err());
            assert!(bank
                .insert(0, &replacement, 1, std::ptr::null_mut())
                .is_err());
        }
        assert_eq!(bank.get(0, &weights[0]), Some(original));
        // Replacing an owner at the same layer evicts its old capture only.
        library.copy_h2d(replacement.buffer, &123456u32.to_ne_bytes())?;
        unsafe {
            library.cuda_graph_begin_capture(streams[0].raw)?;
            library.copy_d2d_async(outputs[0].buffer, replacement.buffer, 4, streams[0].raw)?;
            let graph = library.cuda_graph_end_capture(streams[0].raw)?;
            bank.insert(0, &replacement, 1, graph)?;
            assert!(bank.get(0, &weights[0]).is_none());
            assert!(bank.get(39, &weights[39]).is_some());
            library.cuda_graph_launch(bank.get(0, &replacement).unwrap().0, streams[0].raw)?;
            library.cuda_stream_synchronize(streams[0].raw)?;
        }
        let mut actual = [0u8; 4];
        library.copy_d2h(&mut actual, outputs[0].buffer)?;
        assert_eq!(u32::from_ne_bytes(actual), 123456);
        for bank in &mut banks {
            unsafe {
                bank.clear()?;
            }
            assert_eq!(bank.len(), 0);
            unsafe {
                bank.clear()?;
            }
        }
        eprintln!("PASS 240 captures, 480 replays, two independent lanes, 40 layers, rows 1/80/4096, changed payloads, owner replacement and invalid binding guards");
        Ok(())
    }
}
