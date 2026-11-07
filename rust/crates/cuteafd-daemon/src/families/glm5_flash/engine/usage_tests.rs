use super::*;
use crate::shared::usage_replay::{record, ReplayLookup};
use cuteafd_core::expert_usage::{Geometry, Reader};
use cuteafd_ffi::{expert_usage::{ExpertUsage, UsageDevice}, CuteafdDeviceBuffer};

fn body<'s>(source: &'s str, first: &str, next: &str) -> &'s str {
    source.split_once(first).unwrap().1.split_once(next).unwrap().0
}

fn prefill_tables(rows: usize) -> StepTables {
    StepTables { positions: (0..rows as i64).collect(), kv_slots: (0..rows as i64).collect(), ..Default::default() }
}

fn padded_decode_tables(rows: usize, bucket: usize) -> StepTables {
    let mut tables = StepTables { decode: true, spec: true, real_rows: rows, page_stride: 8, pool_stride: 2,
        positions: vec![17; rows], kv_slots: vec![19; rows], kda_slots: vec![2; rows],
        seq_first: (0..rows as i32).collect(), pool_slots: vec![-1; rows], cache_lengths: vec![4; rows],
        page_table: vec![3; rows * 8], pool_table: vec![1; rows * 2], ..Default::default() };
    pad_decode_tables(&mut tables, bucket);
    tables
}

#[test]
fn usage_rows_are_real_decode_rows_and_every_prefill_row() {
    // Decode and verify steps count the rows before startup-graph padding, never the bucket.
    for (rows, bucket) in [(1, 1), (3, 4), (9, 16), (65, 127)] {
        let tables = padded_decode_tables(rows, bucket);
        assert_eq!((tables.kv_slots.len(), usage_rows(&tables)), (bucket, rows));
    }
    // Masked startup capture tables hold no real row.
    let geometry = GraphGeometry::keyed(4, 8, 2, false);
    for (rows, spec) in [(1, false), (16, true), (64, true)] {
        let tables = startup_tables(rows, spec, geometry);
        assert_eq!((tables.kv_slots.len(), usage_rows(&tables)), (rows, 0));
    }
    // Prefill never pads: a unit, and a packed step of several sequences, count all their rows.
    assert_eq!(usage_rows(&prefill_tables(2752)), 2752);
    let limits = packing::Limits { rows: 4096, chunk_rows: 511, table_pages: 2048, table_pool_pages: 512,
        taps: Some(2048), dense_context: 2051, max_context: 131_072 };
    let segments = packing::plan(&[(0, 40), (2100, 100), (0, 1)], &limits).unwrap();
    let packed = StepTables { segments, ..prefill_tables(141) };
    assert_eq!(usage_rows(&packed), 141);
    // Context placement's per-rank prefill tables keep every row of the step.
    let ranked = context::rank_tables(&prefill_tables(300), 0, 8).unwrap();
    assert_eq!(usage_rows(&ranked), 300);
}

#[test]
fn only_the_canonical_router_counts_and_every_step_uploads_its_rows_first() {
    let source = include_str!("../engine.rs");
    let source = &source[..source.find("\n#[cfg(test)]\n").unwrap()];
    // The canonical router (rank 0's moe_front) tallies right after its selection.
    let front = body(source, "    fn moe_front(", "    /// Per MoE layer, the weights");
    let select = front.find("self.library.router_select(").unwrap();
    let tally = front.find("self.record_routes(w, index, t, decode)").unwrap();
    assert!(select < tally && tally < front.find("self.probe_ring(index, w, t)").unwrap());
    assert_eq!(source.matches("self.record_routes(").count(), 1);
    // TP2 router replicas (rank 1) and startup priming never count.
    let replica = body(source, "    fn router_on(", "    fn tp2_rank(");
    assert!(!replica.contains("record_routes") && !replica.contains("usage_replay"));
    // Every rank-0 table upload writes the step's usage rows before any consumer.
    let tables = body(source, "    fn put_tables(", "    /// Upload rank 1's tables");
    assert!(tables.find("self.put_usage_rows(w, tables)?").unwrap() < tables.find("self.put(").unwrap());
    let rows = body(source, "    fn put_usage_rows(", "    /// Writes a step's tables");
    assert!(rows.contains("self.usage_active.get() && !self.warming_graphs.get()"));
    // Every workspace a canonical router uses owns a pair.
    for site in ["fn install_workspaces(", "fn decode_workspace_of(", "fn prefill_slots_of("] {
        assert!(body(source, site, "\n    }\n").contains("self.usage_slot("), "{site}");
    }
}

struct Fixture<'a> {
    library: &'a NativeLibrary,
    stream: *mut c_void,
    graph: *mut c_void,
    buffers: Vec<CuteafdDeviceBuffer>,
}
impl Fixture<'_> {
    fn upload(&mut self, words: &[u32]) -> Result<CuteafdDeviceBuffer> {
        let bytes = words.iter().flat_map(|n| n.to_ne_bytes()).collect::<Vec<_>>();
        let buffer = self.library.alloc_device_buffer(bytes.len())?;
        self.buffers.push(buffer);
        self.library.copy_h2d(buffer, &bytes)?;
        Ok(buffer)
    }
}
impl Drop for Fixture<'_> {
    fn drop(&mut self) {
        // SAFETY: the fixture retains its library and every operand until the stream drains.
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
fn flash_known_routes_padding_warmup_lazy_graph_replay_phases_and_devices() -> Result<()> {
    // SAFETY: the matching native ABI library stays live through all fixture retirement.
    let library = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
    // The canonical producer is logical rank zero (--device), whatever its CUDA ordinal;
    // a head split's rank one runs replica routers only.
    for cuda in [0, 1] {
        library.cuda_set_device(cuda)?;
        let reader = Reader::default();
        let usage = ExpertUsage::new(&library, Geometry { layers: 3, experts: 8, topk: 2,
            routed_layers: vec![1, 2] }, &[UsageDevice { logical: 0, cuda }], reader.clone())?;
        for suffix in ["counters", "snapshot", "pinned"] {
            let name = format!("expert-usage.gpu0.{suffix}");
            let records = cuteafd_ffi::memory_ledger::allocations().into_iter()
                .filter(|record| record.key.owner == Some(name.as_str())).collect::<Vec<_>>();
            ensure!(records.len() == 1 && records[0].bytes == 3 * 8 * 16, "owner mismatch {name}");
            if suffix != "pinned" { ensure!(records[0].key.device == cuda, "physical owner mismatch"); }
        }
        let lookup = ReplayLookup::new(&library, "glmf.usage.lookup")?;
        // Declared after usage: errors drain the fixture stream before counters retire.
        let mut fixture = Fixture { library: &library, stream: library.cuda_stream_create()?,
            graph: std::ptr::null_mut(), buffers: vec![] };
        // Four router rows: the step's real rows first, then padding that selects experts 6 and 7.
        let logits = (0..4).flat_map(|row| (0..8).map(move |expert| {
            let selected = if row == 3 { expert >= 6 } else { expert == row || expert == row + 1 };
            if selected { 4.0f32.to_bits() } else { (-4.0f32).to_bits() }
        })).collect::<Vec<_>>();
        let logits = fixture.upload(&logits)?;
        let bias = fixture.upload(&[0; 8])?;
        let ids = fixture.upload(&[0; 8])?;
        let weights = fixture.upload(&[0; 8])?;
        let pair = fixture.upload(&[0, 0])?;
        let stream = fixture.stream;
        // GLM Flash's selection: sigmoid scores with the gate bias, then top-k.
        let route = || -> Result<()> {
            // SAFETY: four complete router rows and their outputs are fixture-owned.
            unsafe { library.router_select(logits.ptr, bias.ptr, std::ptr::null(), std::ptr::null(),
                ids.ptr, weights.ptr, 4, 8, 2, 1.0, true, stream) }
        };
        let put = |tables: &StepTables| -> Result<()> {
            // SAFETY: the immutable table and the pair outlive the stream; outside capture.
            unsafe { lookup.put(pair, tables.decode, usage_rows(tables), stream) }
        };
        let tally = |layer: usize, decode: bool, rows: usize| -> Result<()> {
            // SAFETY: the router wrote all four rows of ids on this stream; the pair is live.
            unsafe { record(Some(&usage), cuda, 0, layer, decode, ids.ptr, rows,
                Some(pair.ptr.cast::<i32>().cast_const()), stream) }
        };
        route()?;
        // A prefill unit of three rows: the router ran over exactly its rows.
        put(&prefill_tables(3))?;
        tally(1, false, 3)?;
        // Startup work before serving (TP2 checks, padding checks) uploads zero rows.
        // SAFETY: immutable table and pair; outside capture.
        unsafe { lookup.put(pair, false, 0, stream)?; }
        tally(1, false, 4)?;
        // Replica routers enqueue nothing and read no operand.
        // SAFETY: the replica path returns before touching any argument.
        unsafe { record(Some(&usage), -1, 1, usize::MAX, true, std::ptr::null(), usize::MAX, None, stream)?; }
        // An eager padded verify (startup graphs: MoE between captured segments) of three real rows.
        put(&padded_decode_tables(3, 4))?;
        tally(2, true, 3)?;
        // Lazy graphs capture the router and its tally at exact rows.
        // SAFETY: capture uses stable operand addresses; every pair upload stays outside it.
        unsafe { library.cuda_graph_begin_capture(stream)?; }
        route()?;
        tally(2, true, 4)?;
        // SAFETY: ends the capture begun above on the same stream.
        fixture.graph = unsafe { library.cuda_graph_end_capture(stream)? };
        // The launch right after capture is masked warm-up here: zero rows count nothing.
        put(&startup_tables(4, true, GraphGeometry::keyed(4, 8, 2, false)))?;
        // SAFETY: the graph reads persistent fixture operands.
        unsafe { library.cuda_graph_launch(fixture.graph, stream)?; }
        let captures = cuteafd_ffi::graph_capture_sites();
        for tables in [padded_decode_tables(2, 4), padded_decode_tables(3, 4), prefill_tables(4)] {
            put(&tables)?;
            // SAFETY: no recapture: the same graph with a new real width; a prefill
            // pair under a decode graph reads the decode entry, which is zero.
            unsafe { library.cuda_graph_launch(fixture.graph, stream)?; }
        }
        // SAFETY: all queued operands are still owned by the fixture and usage.
        unsafe { library.cuda_stream_synchronize(stream)?; }
        ensure!(captures == cuteafd_ffi::graph_capture_sites(), "replay recaptured a graph");
        std::thread::sleep(std::time::Duration::from_millis(1500));
        let value = reader.snapshot();
        ensure!(value["error"].is_null(), "snapshot failed: {value}");
        let mut prefill = vec![0u64; 24];
        prefill[8..12].copy_from_slice(&[1, 2, 2, 1]);
        let mut decode = vec![0u64; 24];
        // Eager three rows, then replays of two and three; rejected verify rows count, padding never.
        decode[16..20].copy_from_slice(&[3, 6, 5, 2]);
        ensure!(prefill.iter().sum::<u64>() == 3 * 2 && decode.iter().sum::<u64>() == 8 * 2, "rows x top-k");
        ensure!(value["windows"]["total"]["prefill_routed_rows"] == serde_json::json!(prefill), "prefill: {value}");
        ensure!(value["windows"]["total"]["decode_routed_rows"] == serde_json::json!(decode), "decode: {value}");
    }
    Ok(())
}
