//! GPU checks of `cuteafd_steer_rows` against `steer_rows_ref`. Run with
//! `CUTEAFD_NATIVE_LIB=<libcuteafd_native.so> cargo test --release -p
//! cuteafd-ffi steer_rows -- --ignored --test-threads 1` on a GPU host.
use super::{SteerRows, SteerRowsShape};
use crate::{CuteafdDeviceBuffer, NativeLibrary};
use anyhow::{ensure, Result};
use cuteafd_core::steer_rows::{f32_to_bf16_rne, steer_rows_ref};
use std::ffi::c_void;

fn library() -> Result<NativeLibrary> {
    // SAFETY: the test names a native library built from this checkout.
    unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?) }
}

struct Rng(u64);

impl Rng {
    fn unit(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.unit();
        (self.0 >> 33) % n
    }
}

/// One case's host operands.
#[derive(Clone)]
struct Case {
    shape: SteerRowsShape,
    x: Vec<u16>,
    vidx: Vec<i32>,
    a: Vec<f32>,
    b: Vec<f32>,
    k: Vec<i32>,
}

impl Case {
    /// Three variants: `kk`, 0 (does not edit this site) and `(kk + 1) / 2`;
    /// rows draw base, each variant and an out-of-range negative index.
    fn new(seed: u64, rows: usize, h: usize, ld: usize, kk: usize) -> Self {
        let mut rng = Rng(seed);
        let shape = SteerRowsShape { rows, h, ld, k_max: kk, variants: 3 };
        let table = (shape.variants + 1) * kk * h;
        let b_scale = 2.0 / (h as f32).sqrt();
        let x = (0..rows * ld).map(|_| f32_to_bf16_rne(rng.unit() * 3.0)).collect();
        let vidx = (0..rows).map(|_| match rng.below(9) { 0 => -1, n => (n % 4) as i32 }).collect();
        let a = (0..table).map(|_| rng.unit() * 0.5).collect();
        let b = (0..table).map(|_| rng.unit() * b_scale).collect();
        Case { shape, x, vidx, a, b, k: vec![0, kk as i32, 0, kk.div_ceil(2) as i32] }
    }

    /// `steer_rows_ref` over row chunks on every host core.
    fn reference(&self) -> Vec<u16> {
        let mut x = self.x.clone();
        let SteerRowsShape { rows, h, ld, k_max, .. } = self.shape;
        let threads = std::thread::available_parallelism().map_or(8, |n| n.get());
        let chunk = rows.div_ceil(threads).max(1);
        std::thread::scope(|scope| {
            for (index, part) in x.chunks_mut(chunk * ld).enumerate() {
                let vidx = &self.vidx[index * chunk..(index * chunk + chunk).min(rows)];
                scope.spawn(move || steer_rows_ref(part, ld, vidx.len(), h, vidx, &self.a, &self.b, &self.k, k_max));
            }
        });
        x
    }
}

fn bytes<T: Copy, const N: usize>(values: &[T], encode: impl Fn(T) -> [u8; N]) -> Vec<u8> {
    values.iter().flat_map(|&v| encode(v)).collect()
}

/// Device copies of a case, freed on drop.
struct Device<'a> {
    library: &'a NativeLibrary,
    buffers: [CuteafdDeviceBuffer; 5],
}

impl<'a> Device<'a> {
    fn new(library: &'a NativeLibrary, case: &Case) -> Result<Self> {
        let mut device = Device { library, buffers: [CuteafdDeviceBuffer::default(); 5] };
        let sizes = [case.x.len() * 2, case.vidx.len().max(1) * 4, case.a.len() * 4, case.b.len() * 4, case.k.len() * 4];
        for (slot, bytes) in device.buffers.iter_mut().zip(sizes) {
            *slot = library.alloc_device_buffer(bytes)?;
        }
        device.upload(case)?;
        Ok(device)
    }

    fn upload(&self, case: &Case) -> Result<()> {
        let [x, vidx, a, b, k] = self.buffers;
        self.library.copy_h2d(x, &bytes(&case.x, u16::to_le_bytes))?;
        self.library.copy_h2d(vidx, &bytes(&case.vidx, i32::to_le_bytes))?;
        self.library.copy_h2d(a, &bytes(&case.a, f32::to_le_bytes))?;
        self.library.copy_h2d(b, &bytes(&case.b, f32::to_le_bytes))?;
        self.library.copy_h2d(k, &bytes(&case.k, i32::to_le_bytes))
    }

    fn x(&self) -> Result<Vec<u16>> {
        let mut raw = vec![0u8; self.buffers[0].bytes];
        self.library.copy_d2h(&mut raw, self.buffers[0])?;
        Ok(raw.chunks_exact(2).map(|p| u16::from_le_bytes([p[0], p[1]])).collect())
    }

    /// # Safety
    /// `stream` is a live stream on this device.
    unsafe fn launch(&self, steer: &SteerRows<'_>, shape: SteerRowsShape, stream: *mut c_void) -> Result<()> {
        let [x, vidx, a, b, k] = self.buffers;
        // SAFETY: the buffers outlive the launch (callers synchronize before drop).
        unsafe { steer.launch(x, vidx, a, b, k, shape, stream) }
    }
}

impl Drop for Device<'_> {
    fn drop(&mut self) {
        for buffer in &mut self.buffers {
            let _ = self.library.free_device_buffer(buffer);
        }
    }
}

fn run(library: &NativeLibrary, steer: &SteerRows<'_>, stream: *mut c_void, case: &Case) -> Result<Vec<u16>> {
    let device = Device::new(library, case)?;
    // SAFETY: stream is live on the current device; synchronized before the buffers drop.
    unsafe {
        device.launch(steer, case.shape, stream)?;
        library.cuda_stream_synchronize(stream)?;
    }
    device.x()
}

fn first_mismatch(got: &[u16], want: &[u16], ld: usize) -> Option<String> {
    got.iter().zip(want).position(|(g, w)| g != w)
        .map(|i| format!("row {} col {}: got {:#06x}, want {:#06x}", i / ld, i % ld, got[i], want[i]))
}

#[test]
#[ignore = "requires a CUDA GPU and CUTEAFD_NATIVE_LIB built with CUTEAFD_ENABLE_STEER_ROWS_AOT"]
fn steer_rows_matches_reference_bit_exactly() -> Result<()> {
    let library = library()?;
    let steer = library.steer_rows()?;
    let stream = library.cuda_stream_create()?;
    let mut cases = 0;
    for h in [4096, 5120, 6144] {
        for kk in [1, 2, 19, 32] {
            for rows in [1, 7, 64, 2048, 4096] {
                // A padded stride on odd row counts checks the columns past h stay put.
                let ld = if rows % 2 == 1 { h + 64 } else { h };
                let case = Case::new((h * 131 + kk * 7 + rows) as u64, rows, h, ld, kk);
                let want = case.reference();
                let got = run(&library, &steer, stream, &case)?;
                if let Some(error) = first_mismatch(&got, &want, ld) {
                    anyhow::bail!("h {h} k {kk} rows {rows}: {error}");
                }
                // Base, k == 0 and negative rows are byte-unchanged; steered rows changed.
                for (r, &v) in case.vidx.iter().enumerate() {
                    let row = r * ld..(r + 1) * ld;
                    if v <= 0 || v == 2 {
                        ensure!(got[row.clone()] == case.x[row], "h {h} k {kk}: row {r} (vidx {v}) was written");
                    }
                }
                ensure!(got != case.x || case.vidx.iter().all(|&v| v <= 0 || v == 2), "no row was steered");
                cases += 1;
            }
        }
    }
    // SAFETY: every launch above synchronized.
    unsafe { library.cuda_stream_destroy(stream)? };
    eprintln!("steer_rows: {cases} cases bit-exact");
    Ok(())
}

#[test]
#[ignore = "requires a CUDA GPU and CUTEAFD_NATIVE_LIB built with CUTEAFD_ENABLE_STEER_ROWS_AOT"]
fn steer_rows_is_batch_invariant() -> Result<()> {
    let library = library()?;
    let steer = library.steer_rows()?;
    let stream = library.cuda_stream_create()?;
    for (h, kk) in [(6144, 1), (4096, 19), (5120, 32)] {
        let big = Case::new(99 + kk as u64, 4096, h, h, kk);
        let probe = 4000;
        let mut lone = big.clone();
        lone.shape.rows = 1;
        lone.x = big.x[probe * h..(probe + 1) * h].to_vec();
        lone.vidx = vec![1];
        let mut mixed = Case::new(5 + kk as u64, 64, h, h, kk);
        mixed.a = big.a.clone();
        mixed.b = big.b.clone();
        mixed.x[37 * h..38 * h].copy_from_slice(&lone.x);
        mixed.vidx[37] = 1;
        let mut big = big;
        big.vidx[probe] = 1;
        let row_of = |x: Vec<u16>, r: usize| x[r * h..(r + 1) * h].to_vec();
        let want = row_of(run(&library, &steer, stream, &lone)?, 0);
        ensure!(want != lone.x, "probe row was not steered");
        ensure!(row_of(run(&library, &steer, stream, &mixed)?, 37) == want, "h {h} k {kk}: 64-row batch differs");
        ensure!(row_of(run(&library, &steer, stream, &big)?, probe) == want, "h {h} k {kk}: 4096-row batch differs");
    }
    // SAFETY: every launch above synchronized.
    unsafe { library.cuda_stream_destroy(stream)? };
    Ok(())
}

#[test]
#[ignore = "requires a CUDA GPU and CUTEAFD_NATIVE_LIB built with CUTEAFD_ENABLE_STEER_ROWS_AOT"]
fn steer_rows_graph_replays_new_rows_and_tables() -> Result<()> {
    let library = library()?;
    let steer = library.steer_rows()?;
    let stream = library.cuda_stream_create()?;
    for kk in [1, 2, 19] {
        let first = Case::new(1000 + kk as u64, 64, 5120, 5120, kk);
        let device = Device::new(&library, &first)?;
        // SAFETY: stream is live; the graph and buffers are destroyed after a final synchronize.
        unsafe {
            library.cuda_graph_begin_capture(stream)?;
            device.launch(&steer, first.shape, stream)?;
            let graph = library.cuda_graph_end_capture(stream)?;
            library.cuda_graph_launch(graph, stream)?;
            library.cuda_stream_synchronize(stream)?;
            if let Some(error) = first_mismatch(&device.x()?, &first.reference(), 5120) {
                anyhow::bail!("k {kk} first replay: {error}");
            }
            // New rows, assignments, tables and k at the same addresses.
            let mut second = Case::new(2000 + kk as u64, 64, 5120, 5120, kk);
            second.k = vec![0, 0, kk as i32, 1];
            device.upload(&second)?;
            library.cuda_graph_launch(graph, stream)?;
            library.cuda_stream_synchronize(stream)?;
            if let Some(error) = first_mismatch(&device.x()?, &second.reference(), 5120) {
                anyhow::bail!("k {kk} second replay: {error}");
            }
            library.cuda_graph_exec_destroy(graph)?;
        }
    }
    // SAFETY: every replay above synchronized.
    unsafe { library.cuda_stream_destroy(stream)? };
    Ok(())
}

/// Microseconds per call inside a 200-node graph (what a decode graph
/// pays per site), all rows steered or all rows base.
#[test]
#[ignore = "requires a CUDA GPU and CUTEAFD_NATIVE_LIB built with CUTEAFD_ENABLE_STEER_ROWS_AOT"]
fn steer_rows_timing() -> Result<()> {
    const NODES: usize = 200;
    const REPLAYS: usize = 50;
    let library = library()?;
    let steer = library.steer_rows()?;
    let stream = library.cuda_stream_create()?;
    let (start, stop) = (library.cuda_event_create()?, library.cuda_event_create()?);
    for (h, kk) in [(6144, 1), (4096, 19)] {
        for rows in [1, 16, 64] {
            for steered in [true, false] {
                let mut case = Case::new(7, rows, h, h, kk);
                case.vidx = vec![i32::from(steered); rows];
                let device = Device::new(&library, &case)?;
                // SAFETY: stream and events are live; synchronized before the buffers drop.
                let micros = unsafe {
                    library.cuda_graph_begin_capture(stream)?;
                    for _ in 0..NODES {
                        device.launch(&steer, case.shape, stream)?;
                    }
                    let graph = library.cuda_graph_end_capture(stream)?;
                    for _ in 0..3 {
                        library.cuda_graph_launch(graph, stream)?;
                    }
                    library.cuda_event_record(start, stream)?;
                    for _ in 0..REPLAYS {
                        library.cuda_graph_launch(graph, stream)?;
                    }
                    library.cuda_event_record(stop, stream)?;
                    library.cuda_event_synchronize(stop)?;
                    let ms = library.cuda_event_elapsed_ms(start, stop)?;
                    library.cuda_graph_exec_destroy(graph)?;
                    f64::from(ms) * 1000.0 / (NODES * REPLAYS) as f64
                };
                let label = if steered { "steered" } else { "base" };
                println!("steer_rows h={h} k={kk} rows={rows} {label}: {micros:.2} us/call");
            }
        }
    }
    // SAFETY: the last replay synchronized.
    unsafe {
        library.cuda_event_destroy(start)?;
        library.cuda_event_destroy(stop)?;
        library.cuda_stream_destroy(stream)?;
    }
    Ok(())
}

/// One `add_rows` case: variants 1 and 3 carry a delta, 2 does not.
struct AddCase {
    shape: super::AddRowsShape,
    y: Vec<u16>,
    d: Vec<f32>,
    vidx: Vec<i32>,
    on: Vec<i32>,
}

impl AddCase {
    fn new(seed: u64, rows: usize, h: usize) -> Self {
        let mut rng = Rng(seed);
        let (ld_y, ld_d) = if rows % 2 == 1 { (h + 8, h + 4) } else { (h, h) };
        let shape = super::AddRowsShape { rows, h, ld_y, ld_d, variants: 3 };
        AddCase {
            shape,
            y: (0..rows * ld_y).map(|_| f32_to_bf16_rne(rng.unit() * 3.0)).collect(),
            d: (0..rows * ld_d).map(|_| rng.unit() * 0.05).collect(),
            vidx: (0..rows).map(|_| match rng.below(9) { 0 => -1, n => (n % 4) as i32 }).collect(),
            on: vec![0, 1, 0, 1],
        }
    }

    fn reference(&self) -> Vec<u16> {
        let mut y = self.y.clone();
        let s = self.shape;
        cuteafd_core::steer_rows::add_rows_ref(&mut y, s.ld_y, &self.d, s.ld_d, s.rows, s.h, &self.vidx, &self.on);
        y
    }
}

struct AddDevice<'a> {
    library: &'a NativeLibrary,
    buffers: [CuteafdDeviceBuffer; 4],
}

impl<'a> AddDevice<'a> {
    fn new(library: &'a NativeLibrary, case: &AddCase) -> Result<Self> {
        let mut device = AddDevice { library, buffers: [CuteafdDeviceBuffer::default(); 4] };
        let sizes = [case.y.len() * 2, case.d.len() * 4, case.vidx.len().max(1) * 4, case.on.len() * 4];
        for (slot, bytes) in device.buffers.iter_mut().zip(sizes) {
            *slot = library.alloc_device_buffer(bytes)?;
        }
        device.upload(case)?;
        Ok(device)
    }

    fn upload(&self, case: &AddCase) -> Result<()> {
        let [y, d, vidx, on] = self.buffers;
        self.library.copy_h2d(y, &bytes(&case.y, u16::to_le_bytes))?;
        self.library.copy_h2d(d, &bytes(&case.d, f32::to_le_bytes))?;
        self.library.copy_h2d(vidx, &bytes(&case.vidx, i32::to_le_bytes))?;
        self.library.copy_h2d(on, &bytes(&case.on, i32::to_le_bytes))
    }

    fn y(&self) -> Result<Vec<u16>> {
        let mut raw = vec![0u8; self.buffers[0].bytes];
        self.library.copy_d2h(&mut raw, self.buffers[0])?;
        Ok(raw.chunks_exact(2).map(|p| u16::from_le_bytes([p[0], p[1]])).collect())
    }

    /// # Safety
    /// `stream` is a live stream on this device.
    unsafe fn launch(&self, add: &super::AddRows<'_>, shape: super::AddRowsShape, stream: *mut c_void) -> Result<()> {
        let [y, d, vidx, on] = self.buffers;
        // SAFETY: the buffers outlive the launch (callers synchronize before drop).
        unsafe { add.launch(y, d, vidx, on, shape, stream) }
    }
}

impl Drop for AddDevice<'_> {
    fn drop(&mut self) {
        for buffer in &mut self.buffers {
            let _ = self.library.free_device_buffer(buffer);
        }
    }
}

#[test]
#[ignore = "requires a CUDA GPU and CUTEAFD_NATIVE_LIB built with CUTEAFD_ENABLE_STEER_ROWS_AOT"]
fn add_rows_matches_reference_and_replays() -> Result<()> {
    let library = library()?;
    let add = library.add_rows()?;
    let stream = library.cuda_stream_create()?;
    let mut cases = 0;
    for h in [4096, 5120, 6144] {
        for rows in [1, 7, 64, 2048, 4096] {
            let case = AddCase::new((h + rows) as u64, rows, h);
            let device = AddDevice::new(&library, &case)?;
            // SAFETY: stream is live; synchronized before the buffers drop.
            unsafe {
                device.launch(&add, case.shape, stream)?;
                library.cuda_stream_synchronize(stream)?;
            }
            let got = device.y()?;
            if let Some(error) = first_mismatch(&got, &case.reference(), case.shape.ld_y) {
                anyhow::bail!("add_rows h {h} rows {rows}: {error}");
            }
            for (r, &v) in case.vidx.iter().enumerate() {
                let row = r * case.shape.ld_y..(r + 1) * case.shape.ld_y;
                if v <= 0 || v == 2 {
                    ensure!(got[row.clone()] == case.y[row], "add_rows h {h}: row {r} (vidx {v}) was written");
                }
            }
            cases += 1;
        }
    }
    // Graph replay with new rows, deltas, assignments and switches.
    let first = AddCase::new(77, 64, 4096);
    let device = AddDevice::new(&library, &first)?;
    // SAFETY: stream is live; the graph is destroyed after a final synchronize.
    unsafe {
        library.cuda_graph_begin_capture(stream)?;
        device.launch(&add, first.shape, stream)?;
        let graph = library.cuda_graph_end_capture(stream)?;
        library.cuda_graph_launch(graph, stream)?;
        library.cuda_stream_synchronize(stream)?;
        ensure!(device.y()? == first.reference(), "add_rows first replay differs");
        let mut second = AddCase::new(78, 64, 4096);
        second.on = vec![0, 0, 1, 1];
        device.upload(&second)?;
        library.cuda_graph_launch(graph, stream)?;
        library.cuda_stream_synchronize(stream)?;
        ensure!(device.y()? == second.reference(), "add_rows second replay differs");
        library.cuda_graph_exec_destroy(graph)?;
        library.cuda_stream_destroy(stream)?;
    }
    eprintln!("add_rows: {cases} cases bit-exact, graph replay ok");
    Ok(())
}

#[test]
#[ignore = "requires a CUDA GPU and CUTEAFD_NATIVE_LIB built with CUTEAFD_ENABLE_STEER_ROWS_AOT"]
fn add_rows_timing() -> Result<()> {
    const NODES: usize = 200;
    const REPLAYS: usize = 50;
    let library = library()?;
    let add = library.add_rows()?;
    let stream = library.cuda_stream_create()?;
    let (start, stop) = (library.cuda_event_create()?, library.cuda_event_create()?);
    for rows in [1, 16, 64] {
        for enabled in [true, false] {
            let mut case = AddCase::new(9, rows, 4096);
            case.shape.ld_y = 4096;
            case.shape.ld_d = 4096;
            case.vidx = vec![i32::from(enabled); rows];
            let device = AddDevice::new(&library, &case)?;
            // SAFETY: stream and events are live; synchronized before the buffers drop.
            let micros = unsafe {
                library.cuda_graph_begin_capture(stream)?;
                for _ in 0..NODES {
                    device.launch(&add, case.shape, stream)?;
                }
                let graph = library.cuda_graph_end_capture(stream)?;
                for _ in 0..3 {
                    library.cuda_graph_launch(graph, stream)?;
                }
                library.cuda_event_record(start, stream)?;
                for _ in 0..REPLAYS {
                    library.cuda_graph_launch(graph, stream)?;
                }
                library.cuda_event_record(stop, stream)?;
                library.cuda_event_synchronize(stop)?;
                let ms = library.cuda_event_elapsed_ms(start, stop)?;
                library.cuda_graph_exec_destroy(graph)?;
                f64::from(ms) * 1000.0 / (NODES * REPLAYS) as f64
            };
            let label = if enabled { "delta" } else { "base" };
            println!("add_rows h=4096 rows={rows} {label}: {micros:.2} us/call");
        }
    }
    // SAFETY: the last replay synchronized.
    unsafe {
        library.cuda_event_destroy(start)?;
        library.cuda_event_destroy(stop)?;
        library.cuda_stream_destroy(stream)?;
    }
    Ok(())
}
