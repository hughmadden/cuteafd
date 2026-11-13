//! Route census kernels (`native/shared/cuda/expert_router.cu`): per routed
//! layer, how many live rows chose each expert, counted on the device after
//! the router into one of two planes the stream selects.
use crate::{CuteafdDeviceBuffer, NativeLibrary};
use anyhow::{ensure, Result};
use std::ffi::c_void;

type Count = unsafe extern "C" fn(*const u32, *const u8, i32, i32, *const u8, *mut u64, i32, i32, i32,
    *mut c_void) -> i32;
type Select = unsafe extern "C" fn(*mut u8, i32, *mut c_void) -> i32;

/// The counts' shape: [`Self::PLANES`] planes of `[layers][experts]` U64,
/// filled from routers that pick `topk` experts per row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RouteCensusShape {
    pub layers: usize,
    pub experts: usize,
    pub topk: usize,
}

impl RouteCensusShape {
    pub const PLANES: usize = 2;

    pub fn validate(self) -> Result<()> {
        ensure!((1..=1024).contains(&self.layers) && (1..=1024).contains(&self.experts)
            && (1..=16).contains(&self.topk), "invalid route census shape {self:?}");
        Ok(())
    }
    /// Counters in one plane.
    pub fn plane_words(self) -> usize { self.layers * self.experts }
    /// Bytes of every plane's counters.
    pub fn bytes(self) -> usize { Self::PLANES * self.plane_words() * 8 }
}

/// The routes one router launch produced: `rows` rows of `ids` (U32
/// `[rows][topk]`); row `r` is live when `live` is absent or `live[r] != 0`.
#[derive(Clone, Copy, Debug)]
pub struct RouteCensusInput {
    pub ids: CuteafdDeviceBuffer,
    pub live: Option<CuteafdDeviceBuffer>,
    pub rows: usize,
}

/// The census entry points of one native library.
#[derive(Clone, Copy)]
pub struct RouteCensusKernels<'a> {
    _library: &'a NativeLibrary,
    count: Count,
    select: Select,
}

impl NativeLibrary {
    /// The route census entry points (fails on a library built without them).
    pub fn route_census(&self) -> Result<RouteCensusKernels<'_>> {
        // SAFETY: the symbols have the declared C ABI (`cuteafd_router.h`), and
        // the returned handle borrows the library that owns them.
        Ok(RouteCensusKernels {
            _library: self,
            count: unsafe { *self.lib.get::<Count>(b"cuteafd_route_census_count")? },
            select: unsafe { *self.lib.get::<Select>(b"cuteafd_route_census_select")? },
        })
    }
}

impl RouteCensusKernels<'_> {
    /// Adds the live routes of `input` to layer `layer` of the plane that
    /// `phase` selects when the kernel runs. Capturable: fixed pointers, no
    /// host synchronisation; writes nothing but `counts`.
    ///
    /// # Safety
    /// Every buffer lives on the stream's device through completion, the
    /// input's ids and live mask are written by earlier work on `stream`, and
    /// `counts` and `phase` are written only by census launches.
    pub unsafe fn count(&self, input: RouteCensusInput, phase: CuteafdDeviceBuffer, counts: CuteafdDeviceBuffer,
        shape: RouteCensusShape, layer: usize, stream: *mut c_void) -> Result<()> {
        let RouteCensusInput { ids, live, rows } = input;
        shape.validate()?;
        ensure!((1..=65536).contains(&rows) && layer < shape.layers,
            "invalid route census launch: {rows} rows, layer {layer}");
        ensure!(!ids.ptr.is_null() && ids.bytes >= rows * shape.topk * 4
            && live.is_none_or(|live| !live.ptr.is_null() && live.bytes >= rows)
            && !phase.ptr.is_null() && phase.bytes >= 1 && !counts.ptr.is_null() && counts.bytes >= shape.bytes(),
            "invalid route census buffers");
        // SAFETY: extents checked above (and again natively); the caller keeps
        // the buffers live and ordered on `stream`.
        let status = unsafe {
            (self.count)(ids.ptr.cast(), live.map_or(std::ptr::null(), |live| live.ptr.cast_const().cast()),
                rows as i32, shape.topk as i32, phase.ptr.cast_const().cast(), counts.ptr.cast(), layer as i32,
                shape.layers as i32, shape.experts as i32, stream)
        };
        ensure!(status == 0, "route census count CUDA status {status}");
        Ok(())
    }

    /// Selects the plane (0 or 1) later counts on `stream` add into: a
    /// one-byte write into `phase` in stream order. Never inside a capture.
    ///
    /// # Safety
    /// `phase` lives on the stream's device through completion.
    pub unsafe fn select(&self, phase: CuteafdDeviceBuffer, plane: usize, stream: *mut c_void) -> Result<()> {
        ensure!(!phase.ptr.is_null() && phase.bytes >= 1 && plane < RouteCensusShape::PLANES,
            "invalid route census plane selection");
        // SAFETY: a one-byte device write the caller orders on `stream`.
        let status = unsafe { (self.select)(phase.ptr.cast(), plane as i32, stream) };
        ensure!(status == 0, "route census select CUDA status {status}");
        Ok(())
    }
}

/// The count kernel's contract on the CPU: per expert, how many routes of
/// the first `rows` rows of `ids` (`[rows][topk]`) chose it, skipping rows
/// whose `live` byte is zero and IDs of `experts` or above.
pub fn route_histogram(ids: &[u32], rows: usize, topk: usize, live: Option<&[u8]>, experts: usize) -> Vec<u64> {
    let mut counts = vec![0u64; experts];
    for (row, routes) in ids.chunks_exact(topk).take(rows).enumerate() {
        if live.is_some_and(|live| live[row] == 0) { continue; }
        for &expert in routes {
            if let Some(count) = counts.get_mut(expert as usize) { *count += 1; }
        }
    }
    counts
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A seeded xorshift: the same ids on every run.
    fn ids(seed: u64, count: usize, bound: u32) -> Vec<u32> {
        let mut state = seed | 1;
        (0..count).map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % u64::from(bound)) as u32
        }).collect()
    }

    #[test]
    fn padded_rows_and_foreign_ids_are_not_counted() {
        // Two live rows, a dead third row, then a padded tail past `rows`.
        let ids = [1, 2, 3, 1, 1, 9, 2, 2, 2, 0, 0, 0];
        let counts = route_histogram(&ids, 3, 3, Some(&[1, 1, 0]), 4);
        assert_eq!(counts, [0, 3, 1, 1]);
        // Without a mask every row up to `rows` counts; the tail never does,
        // and the out-of-range id 9 never does.
        assert_eq!(route_histogram(&ids, 3, 3, None, 4), [0, 3, 4, 1]);
        assert_eq!(route_histogram(&ids, 4, 3, None, 4), [3, 3, 4, 1]);
        let total = |counts: Vec<u64>| counts.iter().sum::<u64>();
        assert_eq!(total(route_histogram(&ids, 2, 3, None, 4)), 5, "one route of the second row is foreign");
    }

    #[test]
    fn shape_bounds_the_device_layout() {
        let shape = RouteCensusShape { layers: 40, experts: 384, topk: 6 };
        shape.validate().unwrap();
        assert_eq!((shape.plane_words(), shape.bytes()), (15_360, 245_760));
        for bad in [RouteCensusShape { layers: 0, ..shape }, RouteCensusShape { experts: 1025, ..shape },
            RouteCensusShape { topk: 17, ..shape }] {
            assert!(bad.validate().is_err());
        }
    }

    /// The kernel against [`route_histogram`] on seeded random ids with
    /// padded tails, live masks and foreign ids, eager and from one captured
    /// graph whose replays follow the stream's selected plane.
    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIB and CUDA; route census kernel oracle"]
    fn census_counts_match_the_cpu_histogram_in_the_selected_plane() -> Result<()> {
        let library = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        library.cuda_set_device(0)?;
        let kernels = library.route_census()?;
        let shape = RouteCensusShape { layers: 40, experts: 384, topk: 6 };
        let capacity = 4096usize;
        let mut ids_buffer = library.alloc_device_buffer(capacity * shape.topk * 4, crate::memory_ledger::Tag::workspace("test/route-census"))?;
        let mut live_buffer = library.alloc_device_buffer(capacity, crate::memory_ledger::Tag::workspace("test/route-census"))?;
        let mut counts = library.alloc_device_buffer(shape.bytes(), crate::memory_ledger::Tag::workspace("test/route-census"))?;
        let mut phase = library.alloc_device_buffer(1, crate::memory_ledger::Tag::workspace("test/route-census"))?;
        let stream = library.cuda_stream_create()?;
        let read = |counts: CuteafdDeviceBuffer| -> Result<Vec<u64>> {
            let mut bytes = vec![0u8; shape.bytes()];
            library.copy_d2h(&mut bytes, counts)?;
            Ok(bytes.as_chunks::<8>().0.iter().map(|word| u64::from_ne_bytes(*word)).collect())
        };
        let result = (|| -> Result<()> {
            library.cuda_zero_bytes(counts, shape.bytes())?;
            library.cuda_zero_bytes(phase, 1)?;
            let mut expected = vec![0u64; shape.bytes() / 8];
            let add = |expected: &mut [u64], plane: usize, layer: usize, histogram: &[u64]| {
                let row = &mut expected[(plane * shape.layers + layer) * shape.experts..][..shape.experts];
                for (total, count) in row.iter_mut().zip(histogram) { *total += count; }
            };
            // Eager launches: decode widths, prefill units, a masked unit;
            // ids past 383 (foreign) and a poisoned tail past `rows`.
            let cases = [(1usize, 0usize, 0usize, false), (6, 1, 0, false), (64, 2, 1, false),
                (2048, 3, 1, true), (2752, 39, 0, true), (4096, 20, 1, false), (17, 39, 1, true)];
            for (case, &(rows, layer, plane, masked)) in cases.iter().enumerate() {
                let mut ids = ids(case as u64 + 1, capacity * shape.topk, 400);
                for id in &mut ids[rows * shape.topk..] { *id = 7; }
                let live: Vec<u8> = (0..capacity).map(|row| u8::from(row % 3 != 1)).collect();
                library.copy_h2d(ids_buffer, &ids.iter().flat_map(|id| id.to_ne_bytes()).collect::<Vec<_>>())?;
                library.copy_h2d(live_buffer, &live)?;
                let live = masked.then_some(live);
                unsafe {
                    kernels.select(phase, plane, stream)?;
                    let input = RouteCensusInput { ids: ids_buffer, live: masked.then_some(live_buffer), rows };
                    kernels.count(input, phase, counts, shape, layer, stream)?;
                    library.cuda_stream_synchronize(stream)?;
                }
                add(&mut expected, plane, layer, &route_histogram(&ids, rows, shape.topk, live.as_deref(), shape.experts));
                assert_eq!(read(counts)?, expected, "case {case}: {rows} rows, layer {layer}, plane {plane}");
                // The census only reads the router's output.
                let mut after = vec![0u8; ids.len() * 4];
                library.copy_d2h(&mut after, ids_buffer)?;
                assert!(after.as_chunks::<4>().0.iter().map(|id| u32::from_ne_bytes(*id)).eq(ids.iter().copied()));
            }
            // One graph captured once; its replays count into whichever plane
            // the stream selected before them.
            let rows = 6;
            let ids = ids(99, capacity * shape.topk, 384);
            library.copy_h2d(ids_buffer, &ids.iter().flat_map(|id| id.to_ne_bytes()).collect::<Vec<_>>())?;
            let histogram = route_histogram(&ids, rows, shape.topk, None, shape.experts);
            unsafe {
                library.cuda_graph_begin_capture(stream)?;
                let launched = kernels.count(RouteCensusInput { ids: ids_buffer, live: None, rows }, phase, counts, shape, 5,
                    stream);
                let graph = library.cuda_graph_end_capture(stream)?;
                launched?;
                for plane in [0, 0, 1, 0, 1, 1] {
                    kernels.select(phase, plane, stream)?;
                    library.cuda_graph_launch(graph, stream)?;
                    add(&mut expected, plane, 5, &histogram);
                }
                library.cuda_stream_synchronize(stream)?;
                library.cuda_graph_exec_destroy(graph)?;
            }
            assert_eq!(read(counts)?, expected, "captured replays");
            assert!(expected[..shape.plane_words()].iter().sum::<u64>() > 0
                && expected[shape.plane_words()..].iter().sum::<u64>() > 0, "both planes counted");
            // What a router graph pays per layer: 100 counts in one graph,
            // timed by events around a warm replay.
            let [start, end] = [library.cuda_event_create()?, library.cuda_event_create()?];
            for rows in [1usize, 6, 64, 2752, 4096] {
                unsafe {
                    library.cuda_graph_begin_capture(stream)?;
                    let input = RouteCensusInput { ids: ids_buffer, live: None, rows };
                    let launched = (0..100).try_for_each(|layer| kernels.count(input, phase, counts, shape,
                        layer % shape.layers, stream));
                    let graph = library.cuda_graph_end_capture(stream)?;
                    launched?;
                    library.cuda_graph_launch(graph, stream)?;
                    library.cuda_event_record(start, stream)?;
                    library.cuda_graph_launch(graph, stream)?;
                    library.cuda_event_record(end, stream)?;
                    library.cuda_stream_synchronize(stream)?;
                    library.cuda_graph_exec_destroy(graph)?;
                    eprintln!("route census count, {rows} rows of {}: {:.2} us per launch in a graph", shape.topk,
                        library.cuda_event_elapsed_ms(start, end)? * 10.0);
                }
            }
            unsafe { for event in [start, end] { library.cuda_event_destroy(event)?; } }
            Ok(())
        })();
        unsafe {
            library.cuda_stream_synchronize(stream)?;
            library.cuda_stream_destroy(stream)?;
        }
        for buffer in [&mut ids_buffer, &mut live_buffer, &mut counts, &mut phase] { library.free_device_buffer(buffer)?; }
        result
    }
}
