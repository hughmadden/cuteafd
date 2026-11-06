use super::*;
use cuteafd_core::expert_usage::{Geometry, Reader};
use cuteafd_ffi::{expert_usage::{ExpertUsage, UsageDevice}, CuteafdDeviceBuffer};

#[test]
fn disabled_and_replica_routes_do_not_touch_cuda() {
    let captures = cuteafd_ffi::graph_capture_sites();
    for rank in [0, 1] {
        // SAFETY: the disabled path must not inspect any operand or query CUDA.
        unsafe { record_glm_routes(None, -1, rank, usize::MAX, std::ptr::null(),
            usize::MAX, None, std::ptr::null_mut()).unwrap(); }
    }
    assert_eq!(captures, cuteafd_ffi::graph_capture_sites());
}

#[test]
fn phase_lookup_covers_every_real_width_and_warmup_zero() {
    let lookup = usage_lookup();
    assert_eq!(lookup.len() * 4, cuteafd_loader::placement::families::glm5::USAGE_LOOKUP_BYTES);
    for decode in [false, true] {
        for rows in 0..=cuteafd_loader::placement::families::glm5::USAGE_MAX_ROWS {
            let first = usage_lookup_offset(decode, rows).unwrap() / 4;
            assert_eq!(&lookup[first..first + 2], if decode { [0, rows as i32] } else { [rows as i32, 0] });
        }
    }
    assert!(usage_lookup_offset(false, 4097).is_err());
}

struct Fixture<'a> {
    library: &'a NativeLibrary,
    stream: *mut c_void,
    graph: *mut c_void,
    buffers: Vec<CuteafdDeviceBuffer>,
    lookup: HostAllocation<'a>,
}
impl Fixture<'_> {
    fn upload(&mut self, words: &[u32]) -> Result<CuteafdDeviceBuffer> {
        let bytes = words.iter().flat_map(|n| n.to_ne_bytes()).collect::<Vec<_>>();
        let buffer = self.library.alloc_device_buffer(bytes.len())?;
        self.buffers.push(buffer);
        self.library.copy_h2d(buffer, &bytes)?;
        Ok(buffer)
    }
    fn phase(&self, destination: CuteafdDeviceBuffer, decode: bool, rows: usize) -> Result<()> {
        let offset = usage_lookup_offset(decode, rows)?;
        let source = cuteafd_ffi::CuteafdHostBuffer {
            ptr: self.lookup.buffer.ptr.cast::<u8>().wrapping_add(offset).cast(), bytes: 8, ..self.lookup.buffer };
        // SAFETY: immutable lookup and destination survive all stream consumers.
        unsafe { self.library.copy_host_buffer_h2d_async(destination, source, 8, self.stream) }
    }
}
impl Drop for Fixture<'_> {
    fn drop(&mut self) {
        // SAFETY: fixture retains its library and all operands until the stream drains.
        unsafe {
            if self.library.cuda_stream_synchronize(self.stream).is_err() { std::process::abort(); }
            if !self.graph.is_null() { let _ = self.library.cuda_graph_exec_destroy(self.graph); }
            for buffer in &mut self.buffers { let _ = self.library.free_device_buffer(buffer); }
            let _ = self.library.cuda_stream_destroy(self.stream);
        }
    }
}

#[test]
#[ignore = "requires two CUDA devices and matching CUTEAFD_NATIVE_LIB"]
fn glm_ml4_known_routes_packed_padding_warmup_replay_phase_and_devices() -> Result<()> {
    // SAFETY: matching native ABI library remains live through all fixture retirement.
    let library = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
    // GLM/ML4's only canonical producer is logical rank zero, even when its
    // physical CUDA ordinal is 1. Rank one's local router is a replica.
    for cuda in [0, 1] {
        for softmax in [false, true] {
            library.cuda_set_device(cuda)?;
            let reader = Reader::default();
            let usage = ExpertUsage::new(&library, Geometry { layers: 3, experts: 8,
                topk: 2, routed_layers: vec![1, 2] }, &[UsageDevice { logical: 0, cuda }], reader.clone())?;
            for suffix in ["counters", "snapshot", "pinned"] {
                let name = format!("expert-usage.gpu0.{suffix}");
                let records = cuteafd_ffi::memory_ledger::allocations().into_iter()
                    .filter(|record| record.key.owner == Some(name.as_str())).collect::<Vec<_>>();
                ensure!(records.len() == 1 && records[0].bytes == 3 * 8 * 16, "owner mismatch {name}");
                if suffix != "pinned" { ensure!(records[0].key.device == cuda, "physical owner mismatch"); }
            }
            let lookup = HostAllocation::new(&library, cuteafd_loader::placement::families::glm5::USAGE_LOOKUP_BYTES)?;
            // SAFETY: freshly allocated pinned storage holds the complete immutable lookup.
            unsafe { std::ptr::copy_nonoverlapping(usage_lookup().as_ptr(), lookup.buffer.ptr.cast(), lookup.buffer.bytes / 4); }
            // Declared after usage: errors drain fixture streams before counters retire.
            let mut fixture = Fixture { library: &library, stream: library.cuda_stream_create()?,
                graph: std::ptr::null_mut(), buffers: vec![], lookup };
            let logits = (0..4).flat_map(|row| (0..8).map(move |expert| {
                let selected = if row == 3 { expert >= 6 } else { expert == row || expert == row + 1 };
                if selected { 4.0f32.to_bits() } else { (-4.0f32).to_bits() }
            })).collect::<Vec<_>>();
            let logits = fixture.upload(&logits)?;
            let bias = fixture.upload(&[0; 8])?;
            let ids = fixture.upload(&[0; 8])?;
            let weights = fixture.upload(&[0; 8])?;
            let rows = fixture.upload(&[0, 0])?;
            let stream = fixture.stream;
            let route = || -> Result<()> {
                // SAFETY: four complete router rows and their outputs are fixture-owned.
                unsafe {
                    if softmax { library.router_select_softmax(logits.ptr, ids.ptr, weights.ptr, 4, 8, 2, 1.0, false, stream) }
                    else { library.router_select(logits.ptr, bias.ptr, std::ptr::null(), std::ptr::null(),
                        ids.ptr, weights.ptr, 4, 8, 2, 1.0, true, stream) }
                }
            };
            route()?;
            fixture.phase(rows, false, 2)?;
            // SAFETY: two packed members cover three real rows; no padded row is counted.
            unsafe { record_glm_routes(Some(&usage), cuda, 0, 1, ids.ptr, 4, Some(rows.ptr.cast()), stream)?; }
            fixture.phase(rows, false, 1)?;
            // SAFETY: the second packed member starts after the first two top-k rows.
            unsafe {
                record_glm_routes(Some(&usage), cuda, 0, 1, ids.ptr.cast::<u32>().add(4).cast(), 1, Some(rows.ptr.cast()), stream)?;
                record_glm_routes(Some(&usage), -1, 1, usize::MAX, std::ptr::null(), usize::MAX, None, stream)?;
            }
            fixture.phase(rows, true, 0)?;
            // SAFETY: capture uses stable operand addresses; metadata uploads remain outside it.
            unsafe { library.cuda_graph_begin_capture(stream)?; }
            route()?;
            // SAFETY: graph retains both phase tallies with replay-time real rows.
            unsafe {
                record_glm_routes(Some(&usage), cuda, 0, 2, ids.ptr, 4, Some(rows.ptr.cast()), stream)?;
                fixture.graph = library.cuda_graph_end_capture(stream)?;
                // warm_decode_graphs capture-then-replay must count zero, even though
                // its exchange_rows is one and its router executes four bucket rows.
                library.cuda_graph_launch(fixture.graph, stream)?;
            }
            let captures = cuteafd_ffi::graph_capture_sites();
            for (decode, real) in [(false, 3), (true, 2), (true, 3)] {
                fixture.phase(rows, decode, real)?;
                // SAFETY: no recapture: same graph, new phase and real width.
                unsafe { library.cuda_graph_launch(fixture.graph, stream)?; }
            }
            // SAFETY: all queued operands are still owned by fixture/usage.
            unsafe { library.cuda_stream_synchronize(stream)?; }
            ensure!(captures == cuteafd_ffi::graph_capture_sites(), "replay recaptured a graph");
            std::thread::sleep(std::time::Duration::from_millis(1500));
            let value = reader.snapshot();
            ensure!(value["error"].is_null(), "snapshot failed: {value}");
            let mut prefill = vec![0u64; 24];
            prefill[8..12].copy_from_slice(&[1, 2, 2, 1]);
            prefill[16..20].copy_from_slice(&[1, 2, 2, 1]);
            let mut decode = vec![0u64; 24];
            // All verification rows count, including rejected candidates.
            decode[16..20].copy_from_slice(&[2, 4, 3, 1]);
            ensure!(prefill.iter().sum::<u64>() == 6 * 2 && decode.iter().sum::<u64>() == 5 * 2, "rows x top-k");
            ensure!(value["windows"]["total"]["prefill_routed_rows"] == serde_json::json!(prefill), "prefill: {value}");
            ensure!(value["windows"]["total"]["decode_routed_rows"] == serde_json::json!(decode), "decode: {value}");
        }
    }
    Ok(())
}
