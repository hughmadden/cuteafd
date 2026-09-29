//! Exact FP8 routed-expert package library (`native/include/cuteafd_fp8_moe.h`):
//! one dlopen'd `libcuteafd_fp8moe.so` per layout, one program per capacity.
//! Drain every stream that used the module before dropping it, on the thread
//! (and with the device) that loaded it.
use anyhow::{ensure, Context, Result};
use libloading::Library;
use std::{ffi::c_void, marker::PhantomData, path::Path, ptr::NonNull, rc::Rc};

pub const FP8_MOE_POINTERS: usize = 11;
pub const FP8_MOE_LIBRARY: &str = "libcuteafd_fp8moe.so";

#[derive(Debug, Clone, PartialEq)]
pub struct Fp8MoeInfo {
    pub hidden: usize,
    pub slice: usize,
    pub experts: usize,
    pub topk: usize,
    pub intermediate: usize,
    pub tp: usize,
    /// Input rows are FP8 K32 wire rows (Spark) rather than BF16 (coordinator).
    pub wire_input: bool,
    pub swiglu_limit: f32,
    pub capacities: Vec<usize>,
    /// MXFP4 weights (packed E2M1 + UE8M0 per 32, ABI 2) rather than E4M3 +
    /// FP32 128x128 scales (ABI 1). MXFP4 slices are zero-padded to 128.
    pub mxfp4: bool,
}

impl Fp8MoeInfo {
    fn from_words(words: [u32; 16]) -> Result<Self> {
        let count = words[9] as usize;
        ensure!(matches!(words[0], 1 | 2) && matches!(words[7], 1 | 7) && (1..=6).contains(&count),
            "unsupported FP8 expert package ABI {words:?}");
        let info = Self {
            hidden: words[1] as usize,
            slice: words[2] as usize,
            experts: words[3] as usize,
            topk: words[4] as usize,
            intermediate: words[5] as usize,
            tp: words[6] as usize,
            wire_input: words[7] == 7,
            swiglu_limit: f32::from_bits(words[8]),
            capacities: words[10..10 + count].iter().map(|&c| c as usize).collect(),
            mxfp4: words[0] == 2,
        };
        let sliced = if info.mxfp4 { info.slice * info.tp >= info.intermediate } else { info.slice * info.tp == info.intermediate };
        ensure!(info.tp > 0 && sliced && info.capacities.windows(2).all(|w| w[0] < w[1]),
            "inconsistent FP8 expert package info {info:?}");
        Ok(info)
    }

    /// Wire row bytes: hidden E4M3 values then hidden/32 UE8M0 scales.
    pub fn wire_row_bytes(&self) -> usize {
        self.hidden + self.hidden / 32
    }

    /// Smallest compiled capacity that holds `rows`.
    pub fn capacity_for(&self, rows: usize) -> Option<usize> {
        self.capacities.iter().copied().find(|&c| c >= rows)
    }
}

type Launch = unsafe extern "C" fn(*mut c_void, u32, *const *mut c_void, i32, *mut c_void) -> i32;
type Destroy = unsafe extern "C" fn(*mut c_void);
type Scratch = unsafe extern "C" fn(u32, *mut u64) -> i32;

pub struct Fp8MoeModule {
    // Function addresses and the context are valid only while the library is loaded.
    _library: Library,
    context: NonNull<c_void>,
    info: Fp8MoeInfo,
    scratch: Vec<usize>,
    launch: Launch,
    destroy: Destroy,
    _owner_thread: PhantomData<Rc<()>>,
}

impl Fp8MoeModule {
    /// Loads `directory/libcuteafd_fp8moe.so` and every program on the current device.
    ///
    /// # Safety
    /// Load only trusted generated code with a current CUDA device; keep that
    /// device current for launches and drop, and drain every stream using
    /// the module before dropping it.
    pub unsafe fn load(directory: &Path) -> Result<Self> {
        let path = directory.join(FP8_MOE_LIBRARY);
        let library = Library::new(&path).with_context(|| format!("loading {}", path.display()))?;
        let query = *library.get::<unsafe extern "C" fn(*mut u32, u32) -> i32>(b"cuteafd_fp8moe_info")?;
        let scratch_bytes = *library.get::<Scratch>(b"cuteafd_fp8moe_scratch_bytes")?;
        let create = *library.get::<unsafe extern "C" fn(*mut *mut c_void) -> i32>(b"cuteafd_fp8moe_create")?;
        let launch = *library.get::<Launch>(b"cuteafd_fp8moe_launch")?;
        let destroy = *library.get::<Destroy>(b"cuteafd_fp8moe_destroy")?;
        let mut words = [0u32; 16];
        ensure!(query(words.as_mut_ptr(), 16) == 0, "FP8 expert package info query failed");
        let info = Fp8MoeInfo::from_words(words)?;
        let scratch = info.capacities.iter().map(|&capacity| {
            let mut bytes = 0u64;
            ensure!(scratch_bytes(capacity as u32, &mut bytes) == 0, "FP8 expert scratch query failed");
            Ok(bytes as usize)
        }).collect::<Result<Vec<_>>>()?;
        let mut context = std::ptr::null_mut();
        let status = create(&mut context);
        ensure!(status == 0, "FP8 expert package initialization failed with CUDA status {status}");
        let context = NonNull::new(context).context("FP8 expert package returned a null context")?;
        Ok(Self { _library: library, context, info, scratch, launch, destroy, _owner_thread: PhantomData })
    }

    pub fn info(&self) -> &Fp8MoeInfo {
        &self.info
    }

    /// Scratch bytes of the program for `capacity`.
    pub fn scratch_bytes(&self, capacity: usize) -> Result<usize> {
        let index = self.info.capacities.iter().position(|&c| c == capacity)
            .with_context(|| format!("FP8 expert package has no capacity {capacity}"))?;
        Ok(self.scratch[index])
    }

    /// Runs the smallest program holding `rows`.
    ///
    /// # Safety
    /// Every pointer names live device memory of the documented extent for
    /// `rows` (`scratch` sized for the chosen capacity); `stream` belongs to
    /// the loading device and is drained before any of it is released.
    pub unsafe fn launch(&self, pointers: &[*mut c_void; FP8_MOE_POINTERS], rows: usize, stream: *mut c_void)
        -> Result<()> {
        let capacity = self.info.capacity_for(rows)
            .with_context(|| format!("{rows} rows exceed the FP8 expert package capacities"))?;
        let status = (self.launch)(self.context.as_ptr(), capacity as u32, pointers.as_ptr(), i32::try_from(rows)?,
            stream);
        ensure!(status == 0, "FP8 expert launch ({rows} rows, capacity {capacity}) failed with {status}");
        Ok(())
    }
}

impl Drop for Fp8MoeModule {
    fn drop(&mut self) {
        // SAFETY: the context came from this library's create and is destroyed once.
        unsafe { (self.destroy)(self.context.as_ptr()) }
    }
}
