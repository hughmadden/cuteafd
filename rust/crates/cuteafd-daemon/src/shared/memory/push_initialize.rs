//! Semantic initialization and bounded, out-of-place post-Done transforms.
//! Nothing here opens a checkpoint or exposes execution views.
use super::{device::{Allocation, Device}, weight_arena::{DeliveredArena, ReadyArena},
    weight_validation::ValidationLayout};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::CuteafdDeviceBuffer;
use cuteafd_loader::{push_load::native::NativePlanes, V41Exl3Residency};
use std::{ffi::c_void, ops::Range};

fn sub(buffer: CuteafdDeviceBuffer, offset: usize, bytes: usize) -> Result<CuteafdDeviceBuffer> {
    ensure!(offset.checked_add(bytes).is_some_and(|end| end <= buffer.bytes),
        "post-Done view exceeds plane");
    // SAFETY: the byte range is contained in the live unpublished owner.
    Ok(CuteafdDeviceBuffer { ptr: unsafe { buffer.ptr.cast::<u8>().add(offset).cast() }, bytes, ..buffer })
}

pub(crate) enum Layer<'a> {
    Native { region: usize, layout: &'a NativePlanes, scalars: Range<usize> },
    Exl3 { region: usize, residency: &'a V41Exl3Residency },
    Exlr8 { region: usize, residency: &'a cuteafd_loader::plan::exlr8::residency::Residency },
    Fp8 { widen: Vec<Range<usize>>, inputs: Option<(Range<usize>, Range<usize>, usize)> },
}

/// One completion barrier for the whole rank: validate, generate omitted zeros,
/// transform every layer, upload descriptors, then publish stable views.
pub(crate) fn initialize<'a>(arena: DeliveredArena<'a>, layers: &[Layer<'_>],
    validation: Option<&ValidationLayout>, admitted_scratch: usize) -> Result<ReadyArena<'a>> {
    if let Some(validation) = validation { validation.validate(arena.validation())?; }
    let required = layers.iter().filter_map(|layer| match layer {
        Layer::Native { layout, .. } => Some(layout.scratch_bytes()), _ => None,
    }).max().unwrap_or(0);
    ensure!(required <= admitted_scratch, "rank transform scratch exceeds admission");
    let transforms = if required > 0 { Some(Transforms::new(arena.device(), required, admitted_scratch)?) }
        else { None };
    arena.initialize(|arena| {
        arena.zero_padding()?;
        for layer in layers {
            match layer {
                Layer::Native { region, layout, scalars } => {
                    let image = arena.validation().get(scalars.clone()).context("native scalar tail absent")?;
                    transforms.as_ref().context("native transform scratch absent")?.native(arena, *region, layout, image)?;
                }
                Layer::Exl3 { region, residency } => exl3(arena, *region, residency)?,
                Layer::Exlr8 { region, residency } => exlr8(arena, *region, residency)?,
                Layer::Fp8 { widen, inputs } => {
                    for range in widen { widen_scales(arena, range.clone(), 1024 * 1024)?; }
                    if let Some((gate, up, layer)) = inputs {
                        let mut a = vec![0; gate.len()];
                        let mut b = vec![0; up.len()];
                        arena.device().run(|| {
                            arena.device().library.copy_d2h(&mut a, arena.buffer(gate.clone())?)?;
                            arena.device().library.copy_d2h(&mut b, arena.buffer(up.clone())?)?;
                            cuteafd_loader::formats::fp8_experts::Fp8ExpertTensors::check_input_scales(*layer, &a, &b)
                        })?;
                    }
                }
            }
        }
        Ok(())
    })
}

/// Scratch is admitted before allocation and reused for every layer/tensor.
/// This owner drains before its scratch can be destroyed, even on errors.
pub(crate) struct Transforms<'a> {
    device: Device<'a>,
    scratch: Allocation<'a>,
    stream: *mut c_void,
}
impl<'a> Transforms<'a> {
    pub fn new(device: Device<'a>, required: usize, admitted: usize) -> Result<Self> {
        ensure!(required <= admitted && required > 0, "push transform scratch exceeds admission");
        let scratch = Allocation::new(device, required)?;
        let stream = device.run(|| device.library.cuda_stream_create())?;
        Ok(Self { device, scratch, stream })
    }
    fn synchronize(&self) -> Result<()> {
        self.device.run(|| {
            // SAFETY: this owner retains its stream and every queued buffer.
            let result = unsafe { self.device.library.cuda_stream_synchronize(self.stream) };
            if result.is_err() { self.device.library.quarantine_module_after_failed_drain(); }
            result
        })
    }
    pub fn native(&self, arena: &DeliveredArena<'_>, layer: usize, layout: &NativePlanes,
        scalar_image: &[u8]) -> Result<()> {
        ensure!(arena.device().id == self.device.id
            && std::ptr::eq(arena.device().library, self.device.library), "push transform device mismatch");
        ensure!(layout.scratch_bytes() <= self.scratch.buffer.bytes, "native scratch is under-admitted");
        let owner = arena.region(layer)?;
        ensure!(owner.bytes >= layout.bytes, "native layout exceeds resident layer");
        // Validate the entire scalar image before any transform or vector write.
        let vectors = if layout.nvfp4 { Some(nvfp4_vectors(scalar_image, layout.experts)?) }
            else { ensure!(scalar_image.is_empty(), "MXFP4 has no scalar image"); None };
        let queued = self.device.run(|| {
            for expert in 0..layout.experts {
                for plane in 0..4 {
                    if layout.nvfp4 && plane % 2 == 0 { continue; }
                    let (offset, bytes) = layout.expert_plane(plane, expert)?;
                    let destination = sub(owner, offset, bytes)?;
                    // SAFETY: scratch is disjoint from the arena. All work is
                    // ordered on this stream; neither owner is released early.
                    unsafe {
                        self.device.library.copy_d2d_async(self.scratch.buffer, destination, bytes, self.stream)?;
                        if layout.nvfp4 {
                            let (rows, cols) = if plane == 1 {
                                (2 * layout.kernel_intermediate, layout.hidden / 16)
                            } else { (layout.hidden, layout.kernel_intermediate / 16) };
                            self.device.library.cuda_nvfp4_swizzle_scale_async(
                                self.scratch.buffer, destination, rows, cols, self.stream)?;
                        } else {
                            self.device.library.v41_pack_plane_async(self.scratch.buffer,
                                destination, layout.intermediate as u32, plane as u32, self.stream)?;
                        }
                    }
                }
            }
            Ok(())
        });
        // The arena is borrowed, so every return (including a partial enqueue
        // failure) must drain before its caller can discard that arena.
        let drained = self.synchronize();
        drained?;
        queued?;
        if let Some(vectors) = vectors {
            self.device.run(|| {
                for ((offset, bytes), values) in layout.vectors.iter().zip(vectors) {
                    let image = values.into_iter().flat_map(f32::to_le_bytes).collect::<Vec<_>>();
                    ensure!(image.len() == *bytes, "NVFP4 generated vector extent mismatch");
                    self.device.library.copy_h2d(sub(owner, *offset, *bytes)?, &image)?;
                }
                Ok(())
            })?;
        }
        Ok(())
    }
}
impl Drop for Transforms<'_> {
    fn drop(&mut self) {
        if self.device.library.is_quarantined_after_failed_drain() { return; }
        if let Err(error) = self.synchronize() {
            tracing::error!(%error, "quarantining post-Done transform storage");
            return;
        }
        if let Err(error) = self.device.run(|| {
            // SAFETY: synchronize proved there is no queued use of the stream.
            unsafe { self.device.library.cuda_stream_destroy(self.stream) }
        }) { tracing::error!(%error, "destroying post-Done transform stream"); }
    }
}

/// Same f32 operations and bit-equality checks as the native mount loader.
fn nvfp4_vectors(image: &[u8], experts: usize) -> Result<[Vec<f32>; 4]> {
    ensure!(image.len() == experts.checked_mul(24).context("NVFP4 scalar image overflow")?,
        "NVFP4 scalar image extent mismatch");
    let mut result: [Vec<f32>; 4] = std::array::from_fn(|_| Vec::with_capacity(experts));
    for expert in image.chunks_exact(24) {
        let v = expert.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect::<Vec<_>>();
        ensure!(v.iter().all(|v| v.is_finite() && *v > 0.), "invalid NVFP4 launch scalar");
        ensure!(v[0].to_bits() == v[1].to_bits() && v[3].to_bits() == v[4].to_bits(),
            "NVFP4 gate/up calibrated scales differ");
        let values = [v[0] * v[3], v[2] * v[5], 1. / v[3], 1. / v[5]];
        ensure!(values.iter().all(|v| v.is_finite() && *v > 0.), "NVFP4 launch scalar overflow");
        for (output, value) in result.iter_mut().zip(values) { output.push(value); }
    }
    Ok(result)
}

/// BF16 scales land packed at half-width offsets within their final FP32 grid.
/// Read a bounded chunk before widening it, working backward to avoid clobber.
pub(crate) fn widen_scales(arena: &DeliveredArena<'_>, region: Range<usize>,
    chunk_bytes: usize) -> Result<()> {
    ensure!(region.len() % 4 == 0 && chunk_bytes >= 4, "invalid scale widening extent");
    let device = arena.device();
    device.run(|| {
        let elements = region.len() / 4;
        let per = chunk_bytes / 4;
        let mut end = elements;
        while end > 0 {
            let start = end.saturating_sub(per);
            let mut raw = vec![0; (end - start) * 2];
            device.library.copy_d2h(&mut raw,
                arena.buffer(region.start + start * 2..region.start + end * 2)?)?;
            let widened = raw.chunks_exact(2).flat_map(|v| {
                (u32::from(u16::from_le_bytes([v[0], v[1]])) << 16).to_le_bytes()
            }).collect::<Vec<_>>();
            device.library.copy_h2d(arena.buffer(region.start + start * 4..region.start + end * 4)?, &widened)?;
            end = start;
        }
        Ok(())
    })
}

pub(crate) fn exl3(arena: &DeliveredArena<'_>, layer: usize,
    residency: &V41Exl3Residency) -> Result<()> {
    let (offsets, bytes) = residency.device_arena_layout()?;
    let owner = arena.region(layer)?;
    ensure!(bytes <= owner.bytes, "EXL3 semantic layout exceeds layer");
    let device = arena.device();
    device.run(|| {
        for load in &residency.loads {
            if load.tensor.ends_with(".trellis") { continue; }
            ensure!(load.bytes % 2 == 0, "EXL3 rotation contains a partial half");
            ensure!(!load.destinations.is_empty(), "EXL3 rotation has no destination");
            for &(buffer, offset) in &load.destinations {
                let base = offsets.get(buffer).context("EXL3 rotation plane absent")?
                    .checked_add(offset).context("EXL3 rotation offset overflow")?;
                for start in (0..load.bytes).step_by(1024 * 1024) {
                    let count = (load.bytes - start).min(1024 * 1024);
                    let mut image = vec![0; count];
                    device.library.copy_d2h(&mut image, sub(owner, base + start, count)?)?;
                    ensure!(image.chunks_exact(2).all(|v| u16::from_le_bytes([v[0], v[1]]) & 0x7c00 != 0x7c00),
                        "non-finite EXL3 rotation in {}", load.tensor);
                }
            }
        }
        for (spec, &offset) in residency.buffers.iter().zip(&offsets) {
            if spec.initial_words.is_empty() { continue; }
            let image = spec.initial_words.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
            ensure!(image.len() <= spec.bytes, "EXL3 generated descriptor exceeds plane");
            device.library.copy_h2d(sub(owner, offset, image.len())?, &image)?;
        }
        Ok(())
    })
}

pub(crate) fn exlr8(arena: &DeliveredArena<'_>, layer: usize,
    residency: &cuteafd_loader::plan::exlr8::residency::Residency) -> Result<()> {
    let owner = arena.region(layer)?;
    let (pointers, bits) = residency.descriptors(owner.ptr as u64)?;
    let (counts, units, _) = residency.compact_descriptors()?;
    arena.device().run(|| {
        for (offset, image) in [
            (residency.pointers_offset, pointers.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>()),
            (residency.bits_offset, bits.iter().flat_map(|v| v.to_le_bytes()).collect()),
            (residency.compact_offsets[0], counts.iter().flat_map(|v| v.to_le_bytes()).collect()),
            (residency.compact_offsets[1], units.iter().flat_map(|v| v.to_le_bytes()).collect()),
        ] {
            arena.device().library.copy_h2d(sub(owner, offset, image.len())?, &image)?;
        }
        Ok(())
    })
}

#[cfg(test)]
#[path = "push_initialize_tests.rs"]
mod tests;
