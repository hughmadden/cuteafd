//! The GPU expert tier's route slots (`native/shared/cuda/expert_router.cu`):
//! each route's compact slot in its layer's resident set, from the router's
//! expert ids and a per-layer `slot_of` table, on the device (no host round
//! trip). The tier's expert kernel then runs over the compact resident
//! planes with the slots as its route ids; a non-resident route (-1) reads no
//! weights and contributes a zero plane.
use crate::{CuteafdDeviceBuffer, NativeLibrary};
use anyhow::{ensure, Result};
use std::ffi::c_void;

type Slots = unsafe extern "C" fn(*const u32, *const i32, *mut i32, i32, i32, i32, i32, *mut c_void) -> i32;

/// The tables' shape: `layers` rows of `experts` I32 slots.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TierSlotShape {
    pub layers: usize,
    pub experts: usize,
}

impl TierSlotShape {
    pub fn validate(self) -> Result<()> {
        ensure!((1..=1024).contains(&self.layers) && (1..=1024).contains(&self.experts),
            "invalid GPU expert tier table shape {self:?}");
        Ok(())
    }
    /// Bytes of every layer's table.
    pub fn bytes(self) -> usize { self.layers * self.experts * 4 }
}

/// The slot entry point of one native library.
#[derive(Clone, Copy)]
pub struct TierSlotKernel<'a> {
    _library: &'a NativeLibrary,
    slots: Slots,
}

impl NativeLibrary {
    /// The GPU expert tier's slot kernel (fails on a library built without it).
    pub fn v41_tier_slots(&self) -> Result<TierSlotKernel<'_>> {
        // SAFETY: the symbol has the declared C ABI (`cuteafd_router.h`), and the
        // returned handle borrows the library that owns it.
        let slots = unsafe { self.lib.get::<Slots>(b"cuteafd_v41_tier_slots") }
            .map_err(|error| anyhow::anyhow!("native library lacks the GPU expert tier's slot kernel \
                (cuteafd_v41_tier_slots); rebuild it: {error}"))?;
        Ok(TierSlotKernel { _library: self, slots: *slots })
    }
}

impl TierSlotKernel<'_> {
    /// Writes `slots[r] = tables[layer][ids[r]]` for the first `routes` U32
    /// `ids` (-1 for an id of `experts` or more) as I32. Capturable: fixed
    /// pointers, no host synchronisation; writes nothing but `slots`.
    ///
    /// # Safety
    /// Every buffer lives on the stream's device through completion, `ids` is
    /// written by earlier work on `stream`, and `tables` is not written while
    /// launches may read it.
    pub unsafe fn launch(&self, ids: CuteafdDeviceBuffer, tables: CuteafdDeviceBuffer, slots: CuteafdDeviceBuffer,
        shape: TierSlotShape, layer: usize, routes: usize, stream: *mut c_void) -> Result<()> {
        shape.validate()?;
        ensure!((1..=65536 * 16).contains(&routes) && layer < shape.layers,
            "invalid GPU expert tier slot launch: {routes} routes, layer {layer}");
        ensure!(!ids.ptr.is_null() && ids.bytes >= routes * 4 && !slots.ptr.is_null() && slots.bytes >= routes * 4
            && !tables.ptr.is_null() && tables.bytes >= shape.bytes(), "invalid GPU expert tier slot buffers");
        // SAFETY: extents checked above (and again natively, with aliasing);
        // the caller keeps the buffers live and ordered on `stream`.
        let status = unsafe {
            (self.slots)(ids.ptr.cast_const().cast(), tables.ptr.cast_const().cast(), slots.ptr.cast(),
                routes as i32, layer as i32, shape.layers as i32, shape.experts as i32, stream)
        };
        ensure!(status == 0, "GPU expert tier slot kernel CUDA status {status}");
        Ok(())
    }
}

/// The slot kernel's contract on the CPU for one layer's `table`.
pub fn tier_slots(ids: &[u32], table: &[i32]) -> Vec<i32> {
    ids.iter().map(|&id| table.get(id as usize).copied().unwrap_or(-1)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn xorshift(seed: u64, count: usize, bound: u32) -> Vec<u32> {
        let mut state = seed | 1;
        (0..count).map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % u64::from(bound)) as u32
        }).collect()
    }

    /// A layer table holding `count` experts chosen from `seed`, ascending
    /// slots in expert order, -1 elsewhere.
    fn table(seed: u64, experts: usize, count: usize) -> Vec<i32> {
        let mut resident: Vec<usize> = xorshift(seed, count * 4, experts as u32).into_iter().map(|id| id as usize).collect();
        resident.sort_unstable();
        resident.dedup();
        resident.truncate(count);
        let mut table = vec![-1; experts];
        for (slot, &id) in resident.iter().enumerate() { table[id] = slot as i32; }
        table
    }

    #[test]
    fn cpu_contract_maps_foreign_ids_to_minus_one() {
        let table = vec![-1, 0, -1, 1];
        assert_eq!(tier_slots(&[0, 1, 2, 3, 4, 384, u32::MAX], &table), [-1, 0, -1, 1, -1, -1, -1]);
        let shape = TierSlotShape { layers: 40, experts: 384 };
        shape.validate().unwrap();
        assert_eq!(shape.bytes(), 61_440);
        assert!(TierSlotShape { layers: 0, experts: 384 }.validate().is_err());
        assert!(TierSlotShape { layers: 40, experts: 1025 }.validate().is_err());
    }

    /// The kernel against [`tier_slots`]: decode, verify and prefill widths,
    /// every layer of a 40-layer table set, ids past the experts, a poisoned
    /// tail past `routes` left untouched; eager and from a captured graph.
    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIB and CUDA; GPU expert tier slot kernel oracle"]
    fn tier_slots_match_the_cpu_table_lookup() -> Result<()> {
        // SAFETY: the opt-in path names the native library built for this GPU.
        let library = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        library.cuda_set_device(0)?;
        let kernel = library.v41_tier_slots()?;
        let shape = TierSlotShape { layers: 40, experts: 384 };
        let capacity = 4096 * 6;
        let mut ids_buffer = library.alloc_device_buffer(capacity * 4, crate::memory_ledger::Tag::workspace("test/tier-slots"))?;
        let mut slots_buffer = library.alloc_device_buffer(capacity * 4, crate::memory_ledger::Tag::workspace("test/tier-slots"))?;
        let mut tables_buffer = library.alloc_device_buffer(shape.bytes(), crate::memory_ledger::Tag::workspace("test/tier-slots"))?;
        let stream = library.cuda_stream_create()?;
        let bytes = |values: &[i32]| values.iter().flat_map(|value| value.to_ne_bytes()).collect::<Vec<_>>();
        let read = |buffer: CuteafdDeviceBuffer, count: usize| -> Result<Vec<i32>> {
            let mut raw = vec![0u8; count * 4];
            library.copy_d2h(&mut raw, CuteafdDeviceBuffer { bytes: count * 4, ..buffer })?;
            Ok(raw.as_chunks::<4>().0.iter().map(|word| i32::from_ne_bytes(*word)).collect())
        };
        let result = (|| -> Result<()> {
            let tables: Vec<Vec<i32>> = (0..shape.layers).map(|layer| table(layer as u64 + 7, shape.experts,
                [0, 1, 57, 58, 280, 384][layer % 6])).collect();
            library.copy_h2d(tables_buffer, &bytes(&tables.concat()))?;
            for (case, &(rows, layer)) in [(1usize, 0usize), (6, 1), (16, 2), (80, 3), (2048, 4), (4096, 39), (17, 5)]
                .iter().enumerate() {
                let routes = rows * 6;
                let ids = xorshift(case as u64 + 100, capacity, 400);
                library.copy_h2d(ids_buffer, &ids.iter().flat_map(|id| id.to_ne_bytes()).collect::<Vec<_>>())?;
                library.copy_h2d(slots_buffer, &bytes(&vec![0x5a5a_5a5a; capacity]))?;
                unsafe {
                    kernel.launch(ids_buffer, tables_buffer, slots_buffer, shape, layer, routes, stream)?;
                    library.cuda_stream_synchronize(stream)?;
                }
                let slots = read(slots_buffer, capacity)?;
                assert_eq!(&slots[..routes], tier_slots(&ids[..routes], &tables[layer]).as_slice(),
                    "case {case}: {rows} rows, layer {layer}");
                assert!(slots[routes..].iter().all(|&slot| slot == 0x5a5a_5a5a), "case {case}: wrote past its routes");
            }
            // One captured launch replays the same lookup.
            let ids = xorshift(5, capacity, 384);
            library.copy_h2d(ids_buffer, &ids.iter().flat_map(|id| id.to_ne_bytes()).collect::<Vec<_>>())?;
            unsafe {
                library.cuda_graph_begin_capture(stream)?;
                let launched = kernel.launch(ids_buffer, tables_buffer, slots_buffer, shape, 2, 36, stream);
                let graph = library.cuda_graph_end_capture(stream)?;
                launched?;
                library.copy_h2d(slots_buffer, &bytes(&vec![0; capacity]))?;
                library.cuda_graph_launch(graph, stream)?;
                library.cuda_stream_synchronize(stream)?;
                library.cuda_graph_exec_destroy(graph)?;
            }
            assert_eq!(read(slots_buffer, 36)?, tier_slots(&ids[..36], &tables[2]), "captured replay");
            // What a layer pays: 100 lookups in one graph, timed around a warm replay.
            let [start, end] = [library.cuda_event_create()?, library.cuda_event_create()?];
            for rows in [1usize, 6, 16, 96, 4096] {
                unsafe {
                    library.cuda_graph_begin_capture(stream)?;
                    let launched = (0..100).try_for_each(|layer| kernel.launch(ids_buffer, tables_buffer, slots_buffer,
                        shape, layer % shape.layers, rows * 6, stream));
                    let graph = library.cuda_graph_end_capture(stream)?;
                    launched?;
                    library.cuda_graph_launch(graph, stream)?;
                    library.cuda_event_record(start, stream)?;
                    library.cuda_graph_launch(graph, stream)?;
                    library.cuda_event_record(end, stream)?;
                    library.cuda_stream_synchronize(stream)?;
                    library.cuda_graph_exec_destroy(graph)?;
                    eprintln!("tier slots, {rows} rows of 6: {:.2} us per launch in a graph",
                        library.cuda_event_elapsed_ms(start, end)? * 10.0);
                }
            }
            // Aliasing the ids or the tables is refused natively.
            unsafe {
                assert!(kernel.launch(ids_buffer, tables_buffer, ids_buffer, shape, 0, 6, stream).is_err());
                assert!(kernel.launch(ids_buffer, tables_buffer, tables_buffer, shape, 0, 6, stream).is_err());
            }
            Ok(())
        })();
        unsafe { library.cuda_stream_destroy(stream)?; }
        library.free_device_buffer(&mut ids_buffer)?;
        library.free_device_buffer(&mut slots_buffer)?;
        library.free_device_buffer(&mut tables_buffer)?;
        result
    }
}
