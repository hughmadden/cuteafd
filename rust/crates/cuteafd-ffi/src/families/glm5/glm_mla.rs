//! Sparse MLA prefill on F16 tensor cores (`native/families/glm5/cuda/glm_mla_prefill.cu`):
//! GLM 5.x 656-byte and GLM 5.3 Flash 528-byte FP8 latent records.
use crate::NativeLibrary;
use anyhow::{ensure, Result};
use std::ffi::c_void;

impl NativeLibrary {
    /// `out` [rows, heads, 512] BF16 = softmax(scale * q . k) v over each row's
    /// first `lengths[r]` entries of `indices` [rows, topk] (record slots;
    /// negative entries masked) in `kv` (slot * `record_bytes`: 656 with RoPE,
    /// 528 without); `q` [rows, heads, 576 or 512] BF16. `scale_log2` is the
    /// softmax scale times log2(e).
    ///
    /// # Safety
    /// Every pointer is live device memory of those shapes on the stream's device.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn glm_mla_prefill(&self, q: *const c_void, kv: *const c_void, indices: *const c_void,
        lengths: *const c_void, out: *mut c_void, rows: usize, heads: usize, topk: usize, record_bytes: usize,
        scale_log2: f32, stream: *mut c_void) -> Result<()> {
        type F = unsafe extern "C" fn(*const c_void, *const c_void, *const c_void, *const c_void, *mut c_void, i32, i32,
            i32, i32, f32, *mut c_void) -> i32;
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_glm_mla_prefill") }?;
        let status = unsafe {
            f(q, kv, indices, lengths, out, i32::try_from(rows)?, i32::try_from(heads)?, i32::try_from(topk)?,
                i32::try_from(record_bytes)?, scale_log2, stream)
        };
        ensure!(status == 0, "sparse MLA prefill failed with {status}");
        Ok(())
    }
}

impl NativeLibrary {
    /// Sorts each of `rows` rows of `slots` [rows, width] (DSA top-k physical slots, 64 per page;
    /// negative entries last) by logical position `logical[slot / 64] * 64 + slot % 64`, so the
    /// sparse attention accumulates a selection in one canonical order
    /// (`native/shared/cuda/dsa_sorted_slots.cu`). `width` is at most 2048.
    ///
    /// # Safety
    /// `slots` is live device memory of `rows * width` i32 and `logical` maps every page a slot
    /// names; both on the stream's device.
    pub unsafe fn dsa_sort_slots(&self, slots: *mut c_void, logical: *const c_void, rows: usize, width: usize,
        stream: *mut c_void) -> Result<()> {
        type F = unsafe extern "C" fn(*mut c_void, *const c_void, i32, i32, *mut c_void) -> i32;
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_dsa_sort_slots") }?;
        // SAFETY: the caller's contract.
        let status = unsafe { f(slots, logical, i32::try_from(rows)?, i32::try_from(width)?, stream) };
        ensure!(status == 0, "DSA slot sort failed with {status}");
        Ok(())
    }
}
