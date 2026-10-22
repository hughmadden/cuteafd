//! Pre-resolved operations used by asynchronous dual-device chains.
use crate::{CuteafdDeviceBuffer, NativeLibrary};
use anyhow::{ensure, Context, Result};
use std::ffi::c_void;

type AddFn = unsafe extern "C" fn(*const u16, *const u16, *mut u16, usize, *mut c_void) -> i32;
pub struct Bf16Add<'a> { _library: &'a NativeLibrary, launch: AddFn }
impl NativeLibrary {
    pub fn bf16_add(&self) -> Result<Bf16Add<'_>> {
        Ok(Bf16Add { _library: self,
            launch: unsafe { *self.lib.get(b"cuteafd_add_tp2_shared_async")? } })
    }
}
impl Bf16Add<'_> {
    /// # Safety
    /// Buffers must be live on the current stream device, with producers ordered
    /// before this operation. No conflicting aliases may access them until done.
    pub unsafe fn launch(&self, a: CuteafdDeviceBuffer, b: CuteafdDeviceBuffer,
        output: CuteafdDeviceBuffer, count: usize, stream: *mut c_void) -> Result<()> {
        validate_add_buffers(a, b, output, count)?;
        let status = unsafe {
            (self.launch)(a.ptr.cast(), b.ptr.cast(), output.ptr.cast(), count, stream)
        };
        ensure!(status == 0, "shared TP2 addition failed with CUDA status {status}");
        Ok(())
    }
}

type F32AddFn = unsafe extern "C" fn(*const f32, *const f32, *mut f32, usize, *mut c_void) -> i32;
pub struct F32Add<'a> { _library: &'a NativeLibrary, launch: F32AddFn }
impl NativeLibrary {
    pub fn f32_add(&self) -> Result<F32Add<'_>> {
        // SAFETY: the symbol has the native header's F32 residual-add signature.
        Ok(F32Add { _library: self,
            launch: unsafe { *self.lib.get(b"cuteafd_cuda_residual_add_f32_async")? } })
    }
}
impl F32Add<'_> {
    /// # Safety
    /// All operands are live on the current stream device and ordered after
    /// their producers. The caller retains storage through completion/replay.
    pub unsafe fn launch(&self, a: CuteafdDeviceBuffer, b: CuteafdDeviceBuffer,
        output: CuteafdDeviceBuffer, count: usize, stream: *mut c_void) -> Result<()> {
        validate_f32_add_buffers(a,b,output,count)?;
        // SAFETY: validated F32 extents/devices; caller owns stream/lifetimes.
        let status = unsafe { (self.launch)(a.ptr.cast(),b.ptr.cast(),output.ptr.cast(),count,stream) };
        ensure!(status == 0, "shared F32 addition failed with CUDA status {status}");
        Ok(())
    }
}
fn validate_f32_add_buffers(a: CuteafdDeviceBuffer, b: CuteafdDeviceBuffer,
    output: CuteafdDeviceBuffer, count: usize) -> Result<()> {
    let bytes=count.checked_mul(4).filter(|&n|n>0).ok_or_else(||anyhow::anyhow!("invalid F32 addition extent"))?;
    ensure!(count.div_ceil(256) <= i32::MAX as usize, "F32 addition grid exceeds native int32");
    ensure!(output.device_id>=0 && a.device_id==output.device_id && b.device_id==output.device_id,
        "F32 addition device mismatch");
    for buffer in [a,b,output] {
        ensure!(!buffer.ptr.is_null() && buffer.ptr as usize%4==0 && buffer.bytes>=bytes
            && buffer.flags==0 && (buffer.ptr as usize).checked_add(bytes).is_some(), "invalid F32 addition buffer");
    }
    let out=output.ptr as usize;
    for source in [a,b] {
        let start=source.ptr as usize;
        ensure!(out.abs_diff(start)>=bytes,"F32 addition output overlaps input");
    }
    Ok(())
}

fn validate_add_buffers(a: CuteafdDeviceBuffer, b: CuteafdDeviceBuffer,
    output: CuteafdDeviceBuffer, count: usize) -> Result<()> {
    let bytes = count.checked_mul(2).filter(|n| *n > 0)
        .ok_or_else(|| anyhow::anyhow!("invalid BF16 addition extent"))?;
    ensure!(output.device_id >= 0 && a.device_id == output.device_id && b.device_id == output.device_id,
        "BF16 addition device mismatch");
    for buffer in [a, b, output] {
        ensure!(!buffer.ptr.is_null() && buffer.ptr as usize % 2 == 0 && buffer.bytes >= bytes
            && (buffer.ptr as usize).checked_add(bytes).is_some(), "invalid BF16 addition buffer");
    }
    let out = output.ptr as usize;
    for source in [a, b] {
        let start = source.ptr as usize;
        let distance = if start <= out { out - start } else { start - out };
        ensure!(distance >= bytes, "BF16 addition output overlaps input");
    }
    Ok(())
}

#[cfg(test)]
mod add_tests {
    use super::*;

    fn buffers(count: usize) -> [CuteafdDeviceBuffer; 3] {
        std::array::from_fn(|i| CuteafdDeviceBuffer { ptr: (0x1000 + i * count * 2) as *mut c_void,
            bytes: count * 2, device_id: 0, flags: 0 })
    }

    #[test]
    fn shared_add_accepts_real_buffer_extents_without_a_family_width_limit() -> Result<()> {
        for hidden in [5120, 7168] {
            for rows in [1, 16, 4096] {
                let count = rows * hidden;
                let [a, b, output] = buffers(count);
                validate_add_buffers(a, b, output, count)?;
                validate_add_buffers(a, a, output, count)?;
                assert!(validate_add_buffers(a, b, output, count + 1).is_err());
                for rank in 0..3 {
                    let mut short = [a, b, output];
                    short[rank].bytes -= 1;
                    assert!(validate_add_buffers(short[0], short[1], short[2], count).is_err());
                }
            }
        }
        Ok(())
    }

    #[test]
    fn f32_tp_add_preserves_full_width_extents_and_rejects_aliases() {
        let [mut a,mut b,mut out]=buffers(7168);
        a.bytes*=2; b.bytes*=2; out.bytes*=2;
        a.ptr=0x10000usize as *mut c_void;
        b.ptr=0x20000usize as *mut c_void;
        out.ptr=0x30000usize as *mut c_void;
        validate_f32_add_buffers(a,b,out,7168).unwrap();
        assert!(validate_f32_add_buffers(a,b,out,7169).is_err());
        assert!(validate_f32_add_buffers(a,b,a,7168).is_err());
        assert!(validate_f32_add_buffers(a,b,out,usize::MAX).is_err());
        assert!(validate_f32_add_buffers(a,CuteafdDeviceBuffer{device_id:1,..b},out,7168).is_err());
    }

    #[test]
    fn shared_add_rejects_overflow_aliases_and_invalid_devices() {
        let [a, b, output] = buffers(16);
        for count in [0, usize::MAX / 2 + 1, usize::MAX] {
            assert!(validate_add_buffers(a, b, output, count).is_err());
        }
        let wrapped = CuteafdDeviceBuffer { ptr: (usize::MAX - 1) as *mut c_void, ..a };
        assert!(validate_add_buffers(wrapped, b, output, 16).is_err());
        assert!(validate_add_buffers(a, b, a, 16).is_err());
        let overlap = CuteafdDeviceBuffer { ptr: (a.ptr as usize + 2) as *mut c_void, ..output };
        assert!(validate_add_buffers(a, b, overlap, 16).is_err());
        for bad in [CuteafdDeviceBuffer { device_id: -1, ..a },
            CuteafdDeviceBuffer { device_id: 1, ..a },
            CuteafdDeviceBuffer { ptr: std::ptr::null_mut(), ..a },
            CuteafdDeviceBuffer { ptr: 1usize as *mut c_void, ..a }] {
            assert!(validate_add_buffers(bad, b, output, 16).is_err());
        }
    }
}

type PeerCopyFn = unsafe extern "C" fn(*mut c_void,*const c_void,u64,*mut c_void)->i32;
/// SM-issued peer copy, avoiding DMA queue coupling across lane-local event waits.
type PeerRowsFn = unsafe extern "C" fn(*mut c_void,*const c_void,u64,u64,u64,u64,*mut c_void)->i32;
pub struct V41PeerCopy<'a> { _library: &'a NativeLibrary, launch: PeerCopyFn, rows: PeerRowsFn }
impl NativeLibrary {
    pub fn v41_peer_copy(&self) -> Result<V41PeerCopy<'_>> {
        let initialize=unsafe { *self.lib.get::<unsafe extern "C" fn()->i32>(b"cuteafd_peer_copy_initialize")? };
        ensure!(unsafe { initialize() }==0,"peer copy initialization failed");
        Ok(V41PeerCopy { _library:self,launch:unsafe { *self.lib.get(b"cuteafd_peer_copy_async")? },
            rows:unsafe { *self.lib.get(b"cuteafd_peer_copy_rows_async")? } })
    }
}
impl V41PeerCopy<'_> {
    /// Interleave independent BF16 gate/up planes on one serialized stream.
    /// The caller retains allocation owners until the stream drains.
    pub fn interleave_bf16(&self, destination:CuteafdDeviceBuffer,gate:CuteafdDeviceBuffer,
        up:CuteafdDeviceBuffer,rows:usize,width:usize,stream:*mut c_void)->Result<()> {
        ensure!((1..=4096).contains(&rows) && width>0,"invalid gate/up interleave shape");
        let row_bytes=width.checked_mul(2).context("gate/up row overflow")?;
        let plane_bytes=rows.checked_mul(row_bytes).context("gate/up plane overflow")?;
        let total=plane_bytes.checked_mul(2).context("gate/up destination overflow")?;
        let pitch=row_bytes.checked_mul(2).context("gate/up pitch overflow")?;
        let mut ranges=[(0,0);3];
        for (i,(buffer,bytes)) in [(destination,total),(gate,plane_bytes),(up,plane_bytes)].into_iter().enumerate() {
            ensure!(buffer.device_id>=0 && buffer.device_id==destination.device_id && buffer.flags==0
                && !buffer.ptr.is_null() && (buffer.ptr as usize)%2==0 && buffer.bytes>=bytes,
                "gate/up interleave device/extent/alignment");
            let start=buffer.ptr as usize;
            let end=start.checked_add(bytes).context("gate/up interleave pointer overflow")?;
            ranges[i]=(start,end);
        }
        ensure!(ranges.iter().enumerate().all(|(i,&(start,end))|ranges[..i].iter()
            .all(|&(other_start,other_end)|end<=other_start || other_end<=start)),"gate/up planes overlap");
        let second=CuteafdDeviceBuffer {ptr:destination.ptr.wrapping_byte_add(row_bytes),
            bytes:destination.bytes-row_bytes,..destination};
        // SAFETY: disjoint local extents, device, pitches and pointer arithmetic
        // are checked above; the serialized caller retains all stream owners.
        unsafe {
            self.launch_rows(destination,gate,row_bytes,rows,pitch,row_bytes,stream)?;
            self.launch_rows(second,up,row_bytes,rows,pitch,row_bytes,stream)?;
        }
        Ok(())
    }
    /// # Safety
    /// Destination is on the current stream device; source peer access is enabled.
    /// Producers precede this stream, buffers are disjoint and live through
    /// completion, and no conflicting access occurs. Initialize before capture.
    pub unsafe fn launch(&self,destination: CuteafdDeviceBuffer,source: CuteafdDeviceBuffer,
        bytes: usize,stream:*mut c_void)->Result<()> {
        ensure!(destination.device_id>=0 && source.device_id>=0 && destination.device_id!=source.device_id,
            "peer copy needs distinct device owners");
        ensure!(bytes>0 && bytes<=destination.bytes && bytes<=source.bytes
            && destination.flags==0 && source.flags==0,"invalid peer copy extent or buffer flags");
        let status=unsafe { (self.launch)(destination.ptr,source.ptr,bytes as u64,stream) };
        ensure!(status==0,"SM peer copy failed with CUDA status {status}");
        Ok(())
    }
    /// # Safety
    /// Same publication/lifetime rules as launch; local copies are also allowed.
    /// Each row has `width` bytes and begins at its buffer's respective pitch.
    pub unsafe fn launch_rows(&self,destination:CuteafdDeviceBuffer,source:CuteafdDeviceBuffer,
        width:usize,rows:usize,destination_pitch:usize,source_pitch:usize,stream:*mut c_void)->Result<()> {
        ensure!(destination.device_id>=0 && source.device_id>=0 && destination.flags==0 && source.flags==0,
            "invalid pitched copy device owner or flags");
        ensure!((1..=4096).contains(&rows) && width>0,"invalid pitched copy shape");
        for (buffer,pitch) in [(destination,destination_pitch),(source,source_pitch)] {
            let bytes=(rows-1).checked_mul(pitch).and_then(|n|n.checked_add(width));
            ensure!(pitch>=width && bytes.is_some_and(|n|n<=buffer.bytes),"pitched copy exceeds buffer");
        }
        let status=unsafe { (self.rows)(destination.ptr,source.ptr,width as u64,rows as u64,
            destination_pitch as u64,source_pitch as u64,stream) };
        ensure!(status==0,"SM pitched copy failed with CUDA status {status}");Ok(())
    }

}
