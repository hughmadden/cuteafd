//! Resident ViT owner. Construct, use and drop on its dedicated CUDA thread.
use libloading::Library;
use std::{ffi::c_void, fmt, marker::PhantomData, path::Path, rc::Rc};

pub const ENCODER_NUMERICS: u32 = 1;
pub const NO_VISION_OFFSET: u64 = u64::MAX;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct VisionBlock {
    pub qkv: u64,
    pub qkv_bias: u64,
    pub proj: u64,
    pub proj_bias: u64,
    pub gate_up: u64,
    pub gate_up_bias: u64,
    pub down: u64,
    pub down_bias: u64,
    pub norm1: u64,
    pub norm2: u64,
    pub key0_bias: u64,
    pub window: i32,
    pub column_order: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct VisionSpec {
    pub abi_version: u32,
    pub max_tokens: u32,
    pub output_width: u32,
    pub reserved: u32,
    pub weight_bytes: u64,
    pub patch: u64,
    pub merger_norm: u64,
    pub merger_fc1: u64,
    pub merger_fc2: u64,
    pub inv_freq: u64,
    pub blocks: [VisionBlock; 28],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VisionLedger {
    pub weights: u64,
    pub scratch: u64,
    pub blas_workspace: u64,
    pub device_allocations: u64,
    pub encodes: u64,
}
impl VisionLedger {
    pub fn total_bytes(&self) -> u64 {
        self.weights + self.scratch + self.blas_workspace
    }
}

#[derive(Debug)]
pub enum VisionError {
    Native(i32),
    InvalidInput(&'static str),
    Library(libloading::Error),
}
impl fmt::Display for VisionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Native(status) => write!(f, "vision native status {status}"),
            Self::InvalidInput(reason) => f.write_str(reason),
            Self::Library(e) => write!(f, "vision library: {e}"),
        }
    }
}
impl std::error::Error for VisionError {}
impl From<libloading::Error> for VisionError {
    fn from(e: libloading::Error) -> Self {
        Self::Library(e)
    }
}
fn check(status: i32) -> Result<(), VisionError> {
    if status == 0 {
        Ok(())
    } else {
        Err(VisionError::Native(status))
    }
}

type Required = unsafe extern "C" fn(*const VisionSpec, *mut VisionLedger) -> i32;
type Create = unsafe extern "C" fn(*const VisionSpec, i32, u64, *mut *mut c_void) -> i32;
type Upload = unsafe extern "C" fn(*mut c_void, u64, *const u8, u64) -> i32;
type Observer = unsafe extern "C" fn(*mut c_void, i32, *const c_void, i32, i32, i32) -> i32;
type Encode = unsafe extern "C" fn(
    *mut c_void,
    *const u8,
    u64,
    *const f32,
    i32,
    i32,
    *mut u16,
    u64,
    Option<Observer>,
    *mut c_void,
) -> i32;
type Ledger = unsafe extern "C" fn(*mut c_void, *mut VisionLedger) -> i32;
type Destroy = unsafe extern "C" fn(*mut c_void) -> i32;

pub struct NativeVision {
    _library: Library,
    owner: *mut c_void,
    encode: Encode,
    ledger: Ledger,
    destroy: Destroy,
    max_tokens: usize,
    output_width: usize,
    // CUDA and the BLAS handle never migrate away from their owner thread.
    _thread: PhantomData<Rc<()>>,
}
impl NativeVision {
    /// Pure header/arena query, before loading tensor data or initializing CUDA.
    pub fn required(path: &Path, spec: &VisionSpec) -> Result<VisionLedger, VisionError> {
        // SAFETY: trusted library and versioned header ABI; query initializes no CUDA state.
        unsafe {
            let library = Library::new(path)?;
            let required = library.get::<Required>(b"cuteafd_vision_required")?;
            let mut ledger = VisionLedger::default();
            check(required(spec, &mut ledger))?;
            Ok(ledger)
        }
    }
    /// Check admission before native weights, scratch or cuBLAS allocations.
    /// `weights` must be the complete initialized arena described by `spec`.
    pub fn load(
        path: &Path,
        spec: &VisionSpec,
        device: i32,
        admitted_bytes: u64,
        weights: &[u8],
    ) -> Result<Self, VisionError> {
        if weights.len() as u64 != spec.weight_bytes {
            return Err(VisionError::InvalidInput(
                "vision weight arena length mismatch",
            ));
        }
        // SAFETY: caller selects a trusted native library. Every symbol has the
        // versioned header's exact ABI; the library outlives the native owner.
        unsafe {
            let library = Library::new(path)?;
            let required = *library.get::<Required>(b"cuteafd_vision_required")?;
            let create = *library.get::<Create>(b"cuteafd_vision_create")?;
            let upload = *library.get::<Upload>(b"cuteafd_vision_upload")?;
            let mut ledger = VisionLedger::default();
            check(required(spec, &mut ledger))?;
            if ledger.total_bytes() > admitted_bytes {
                return Err(VisionError::InvalidInput(
                    "vision weight/scratch admission shortfall",
                ));
            }
            let mut result = Self {
                owner: std::ptr::null_mut(),
                encode: *library.get::<Encode>(b"cuteafd_vision_encode")?,
                ledger: *library.get::<Ledger>(b"cuteafd_vision_get_ledger")?,
                destroy: *library.get::<Destroy>(b"cuteafd_vision_destroy")?,
                _library: library,
                max_tokens: spec.max_tokens as usize,
                output_width: spec.output_width as usize,
                _thread: PhantomData,
            };
            check(create(spec, device, admitted_bytes, &mut result.owner))?;
            if result.owner.is_null() {
                return Err(VisionError::InvalidInput("null vision owner"));
            }
            check(upload(
                result.owner,
                0,
                weights.as_ptr(),
                weights.len() as u64,
            ))?;
            Ok(result)
        }
    }
    /// Encode into caller-owned output. No Rust or CUDA allocation in this call.
    /// The LUT is normalized FP32 `[channel][rgb8]`, with reference operation order.
    pub fn encode_into(
        &mut self,
        rgb: &[u8],
        grid: [usize; 2],
        lut: &[f32; 768],
        output: &mut [u16],
    ) -> Result<(), VisionError> {
        let [h, w] = grid;
        let patches = h
            .checked_mul(w)
            .ok_or(VisionError::InvalidInput("vision grid overflow"))?;
        if h < 2
            || w < 2
            || h % 2 != 0
            || w % 2 != 0
            || patches > self.max_tokens * 4
            || rgb.len() != patches * 768
            || output.len() != patches / 4 * self.output_width
        {
            return Err(VisionError::InvalidInput(
                "vision RGB/grid/output extent mismatch",
            ));
        }
        // SAFETY: validated host extents stay live through synchronous encode.
        // Native owner serializes its stream and drains on success and errors.
        check(unsafe {
            (self.encode)(
                self.owner,
                rgb.as_ptr(),
                rgb.len() as u64,
                lut.as_ptr(),
                h as i32,
                w as i32,
                output.as_mut_ptr(),
                std::mem::size_of_val(output) as u64,
                None,
                std::ptr::null_mut(),
            )
        })
    }
    pub fn ledger(&self) -> Result<VisionLedger, VisionError> {
        let mut ledger = VisionLedger::default();
        // SAFETY: exclusive owner is live on this thread; native writes one ledger.
        check(unsafe { (self.ledger)(self.owner, &mut ledger) })?;
        Ok(ledger)
    }
}
impl Drop for NativeVision {
    fn drop(&mut self) {
        if !self.owner.is_null() {
            // SAFETY: destroy drains before freeing; library remains loaded.
            let status = unsafe { (self.destroy)(self.owner) };
            if status != 0 {
                tracing::error!(status, "vision teardown failed");
            }
            self.owner = std::ptr::null_mut();
        }
    }
}

type Inject =
    unsafe extern "C" fn(*const u16, *const u32, *mut u16, i32, i32, i32, i32, *mut c_void) -> i32;
/// Pre-resolved graph-compatible scatter after the normal token embedding gather.
pub struct EmbeddingInjection<'a> {
    _library: &'a crate::NativeLibrary,
    launch: Inject,
}
impl crate::NativeLibrary {
    pub fn embedding_injection(&self) -> Result<EmbeddingInjection<'_>, VisionError> {
        // SAFETY: library stays loaded for the lifetime of the resolved exact-ABI symbol.
        let launch = unsafe { *self.lib.get::<Inject>(b"cuteafd_embed_inject")? };
        Ok(EmbeddingInjection {
            _library: self,
            launch,
        })
    }
}
impl EmbeddingInjection<'_> {
    /// # Safety
    /// Buffers reside on the current stream device and remain live until it drains.
    /// Producers precede this launch; indices are unique and less than rows. Input
    /// and output storage are disjoint and no conflicting accesses are pending.
    pub unsafe fn launch(
        &self,
        features: crate::CuteafdDeviceBuffer,
        indices: crate::CuteafdDeviceBuffer,
        output: crate::CuteafdDeviceBuffer,
        feature_rows: usize,
        rows: usize,
        width: usize,
        copies: usize,
        stream: *mut c_void,
    ) -> Result<(), VisionError> {
        validate_injection(features, indices, output, feature_rows, rows, width, copies)?;
        // SAFETY: caller owns CUDA publication/lifetimes; extents/devices validated here.
        check(unsafe {
            (self.launch)(
                features.ptr.cast(),
                indices.ptr.cast(),
                output.ptr.cast(),
                feature_rows as i32,
                rows as i32,
                width as i32,
                copies as i32,
                stream,
            )
        })
    }
}
fn validate_injection(
    features: crate::CuteafdDeviceBuffer,
    indices: crate::CuteafdDeviceBuffer,
    output: crate::CuteafdDeviceBuffer,
    feature_rows: usize,
    rows: usize,
    width: usize,
    copies: usize,
) -> Result<(), VisionError> {
    if feature_rows == 0
        || feature_rows > rows
        || rows > 16384
        || width == 0
        || width > 16384
        || !(1..=4).contains(&copies)
    {
        return Err(VisionError::InvalidInput("embedding injection shape"));
    }
    for (buffer, bytes, align) in [
        (features, feature_rows * width * 2, 2),
        (indices, feature_rows * 4, 4),
        (output, rows * copies * width * 2, 2),
    ] {
        if buffer.ptr.is_null()
            || buffer.ptr as usize % align != 0
            || buffer.bytes < bytes
            || buffer.device_id < 0
            || buffer.device_id != output.device_id
            || buffer.flags != 0
        {
            return Err(VisionError::InvalidInput(
                "embedding injection buffer/device extent",
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn injection_rejects_bad_extents_and_devices_before_launch() {
        let buffer = crate::CuteafdDeviceBuffer {
            ptr: 0x1000usize as *mut c_void,
            bytes: 128,
            device_id: 0,
            flags: 0,
        };
        assert!(validate_injection(buffer, buffer, buffer, 2, 4, 4, 4).is_ok());
        assert!(validate_injection(buffer, buffer, buffer, 2, 4, 4, 5).is_err());
        assert!(validate_injection(buffer, buffer, buffer, 5, 4, 4, 1).is_err());
        assert!(validate_injection(buffer, buffer, buffer, 2, 4, 64, 4).is_err());
        assert!(validate_injection(
            buffer,
            crate::CuteafdDeviceBuffer {
                device_id: 1,
                ..buffer
            },
            buffer,
            2,
            4,
            4,
            1
        )
        .is_err());
    }
    #[test]
    fn c_header_layout_is_stable() {
        assert_eq!(std::mem::size_of::<VisionBlock>(), 96);
        assert_eq!(std::mem::size_of::<VisionSpec>(), 2752);
        assert_eq!(std::mem::size_of::<VisionLedger>(), 40);
        assert_eq!(std::mem::offset_of!(VisionSpec, blocks), 64);
    }
}
