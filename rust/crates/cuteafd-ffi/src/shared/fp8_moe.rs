//! Exact FP8 routed-expert package library (`native/shared/include/cuteafd_fp8_moe.h`):
//! one dlopen'd `libcuteafd_fp8moe.so` per layout, one program per capacity.
//! Drain every stream that used the module before dropping it, on the thread
//! (and with the device) that loaded it.
use anyhow::{ensure, Context, Result};
use libloading::Library;
use std::{ffi::c_void, marker::PhantomData, path::Path, ptr::NonNull, rc::Rc};

pub const FP8_MOE_POINTERS: usize = 11;
pub const FP8_MOE_LIBRARY: &str = "libcuteafd_fp8moe.so";

/// The expert weight format a package reads (ABI word 0).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fp8MoeWeights {
    /// ABI 1: E4M3 with FP32 128x128 block scales.
    Fp8,
    /// ABI 2: packed E2M1 with UE8M0 scales per 32.
    Mxfp4,
    /// ABI 3: ModelOpt NVFP4: packed E2M1, E4M3 scales per 16, each scale
    /// region followed by the experts' FP32 `weight_scale_2` (W4A16).
    Nvfp4,
}

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
    /// The weight format; packed FP4 slices are zero-padded to 128.
    pub weights: Fp8MoeWeights,
}

impl Fp8MoeInfo {
    fn from_words(words: [u32; 16]) -> Result<Self> {
        let count = words[9] as usize;
        ensure!(matches!(words[0], 1..=3) && matches!(words[7], 1 | 7) && (1..=6).contains(&count),
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
            weights: match words[0] {
                1 => Fp8MoeWeights::Fp8,
                2 => Fp8MoeWeights::Mxfp4,
                _ => Fp8MoeWeights::Nvfp4,
            },
        };
        // Slices are the widest rank range, zero-padded to 128 (MXFP4 32-blocks
        // or FP8 128-blocks split unevenly, TP6 of 2048: 384).
        let sliced = info.slice % 128 == 0 && info.slice * info.tp >= info.intermediate
            && (info.slice - 128) * info.tp < info.intermediate;
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
type SetOptions = unsafe extern "C" fn(*mut c_void, u32) -> i32;

/// `cuteafd_fp8moe_set_options` bit 0: run the W8A16 prefill fallback programs.
const OPTION_PREFILL_A16: u32 = 1;

/// How FP8 expert packages run their large (prefill) row counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Fp8MoePrefill {
    /// Block-scaled E4M3 x E4M3 gate/up over the FP8 K32 wire rows (BF16
    /// rows quantized to wire rows first at coordinator TP1): the default.
    #[default]
    W8a8,
    /// The former W8A16 programs (weights widened to `bf16(w * s)`; BF16
    /// input rows exact on the coordinator).
    W8a16,
}

impl std::str::FromStr for Fp8MoePrefill {
    type Err = anyhow::Error;
    fn from_str(value: &str) -> Result<Self> {
        match value {
            "w8a8" => Ok(Self::W8a8),
            "w8a16" => Ok(Self::W8a16),
            other => anyhow::bail!("FP8 expert prefill is w8a8 or w8a16, not {other:?}"),
        }
    }
}

pub struct Fp8MoeModule {
    // Function addresses and the context are valid only while the library is loaded.
    _library: Library,
    context: NonNull<c_void>,
    info: Fp8MoeInfo,
    scratch: Vec<usize>,
    launch: Launch,
    destroy: Destroy,
    set_options: Option<SetOptions>,
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
        // Packages built before the W8A16 fallback forms have no options.
        let set_options = library.get::<SetOptions>(b"cuteafd_fp8moe_set_options").ok().map(|f| *f);
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
        Ok(Self { _library: library, context, info, scratch, launch, destroy, set_options,
            _owner_thread: PhantomData })
    }

    /// Selects the programs large row counts run (W8A8 unless asked otherwise).
    /// A package without the option runs its single form: accepted for W8A8.
    pub fn set_prefill(&mut self, prefill: Fp8MoePrefill) -> Result<()> {
        let options = if prefill == Fp8MoePrefill::W8a16 { OPTION_PREFILL_A16 } else { 0 };
        let Some(set) = self.set_options else {
            ensure!(options == 0, "this FP8 expert package predates the W8A16 prefill fallback; rebuild it");
            return Ok(());
        };
        // SAFETY: the context came from this library's create and is still live.
        let status = unsafe { set(self.context.as_ptr(), options) };
        ensure!(status == 0, "FP8 expert package rejected options {options:#x} ({status})");
        Ok(())
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
