//! Exact FP8 routed experts (the `fp8` family): the checkpoint's E4M3 expert
//! weights with FP32 128x128 block scales (or MXFP4, MiMo V2.6 Pro:
//! `fp8-mimop` packages; or ModelOpt NVFP4 run W4A16, `fp8-<family>-nvfp4`
//! packages), resident per TP slice and run by
//! an `fp8-<family>` package (`python/tools/aot/package_fp8_moe_aot.py`,
//! `native/shared/include/cuteafd_fp8_moe.h`). Nothing is re-quantized. Output is
//! the BF16 `[rows, H]` route sum of the slice: the Spark rank partial of the
//! compact BF16 response, or the whole layer at TP1 on the coordinator.
pub(crate) mod worker;

use crate::shared::memory::DeviceAllocation;
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::fp8_moe::{Fp8MoeModule, Fp8MoePrefill, Fp8MoeWeights, FP8_MOE_POINTERS};
use cuteafd_ffi::NativeLibrary;
use cuteafd_loader::formats::fp8_experts::{ExpertFormat, Fp8ExpertTensors, Fp8Projection};
use std::ffi::c_void;
use std::path::{Path, PathBuf};

/// Parallel readers per layer load.
const READERS: usize = 16;

/// Environment switch for how FP8 expert packages run prefill row counts:
/// `w8a8` (default: block-scaled E4M3 x E4M3 gate/up) or `w8a16` (the former
/// programs, weights widened to `bf16(w * s)`; exact BF16 rows at coordinator TP1).
pub(crate) const PREFILL_ENV: &str = "CUTEAFD_FP8_EXPERT_PREFILL";

/// The prefill form `PREFILL_ENV` asks for.
pub(crate) fn prefill_mode() -> Result<Fp8MoePrefill> {
    match std::env::var(PREFILL_ENV) {
        Ok(value) if !value.is_empty() => value.parse().with_context(|| format!("{PREFILL_ENV}={value}")),
        _ => Ok(Fp8MoePrefill::default()),
    }
}

/// Loads an FP8 expert package and selects its prefill form.
///
/// # Safety
/// As `Fp8MoeModule::load`.
unsafe fn load_module(directory: &Path) -> Result<Fp8MoeModule> {
    let mut module = Fp8MoeModule::load(directory)?;
    let prefill = prefill_mode()?;
    module.set_prefill(prefill).with_context(|| format!("{} ({PREFILL_ENV})", directory.display()))?;
    if prefill != Fp8MoePrefill::default() {
        tracing::info!(package = %directory.display(), ?prefill, "FP8 expert prefill form");
    }
    Ok(module)
}

/// `<libdir>/fp8/fp8-<family>[-nvfp4]/tp<world>`: the package layout serving
/// TP degree `tp` of the process expert geometry in `format` (NVFP4 releases
/// share their geometry with the FP8 ones and get packages of their own).
pub(crate) fn package_directory(native_lib: &Path, tp: usize, format: ExpertFormat) -> PathBuf {
    let family = cuteafd_core::expert_geometry().family().unwrap_or("unknown");
    native_lib.parent().unwrap_or(Path::new(".")).join("fp8")
        .join(format!("fp8-{family}{}", format.package_suffix())).join(format!("tp{tp}"))
}

/// The BF16-input sibling of an FP8 package directory:
/// `.../fp8-<family>/tp<n>` -> `.../fp8-<family>-bf16/tp<n>`.
pub(crate) fn bf16_sibling(directory: &Path) -> Option<PathBuf> {
    let layout = directory.file_name()?;
    let package = directory.parent()?;
    let name = package.file_name()?.to_str()?;
    Some(package.with_file_name(format!("{name}-bf16")).join(layout))
}

/// One layer's resident slice: per projection the weights of every expert
/// (`[E, rows, cols]`) and their scale grids (NVFP4: then the experts'
/// FP32 alphas).
pub(crate) struct Fp8Layer<'a> {
    pub layer: usize,
    regions: Vec<DeviceAllocation<'a>>,
}

impl<'a> Fp8Layer<'a> {
    /// Device bytes of one layer's slice.
    pub fn bytes(tensors: &Fp8ExpertTensors, tp: usize) -> Result<usize> {
        let experts = tensors.shape().experts;
        Fp8Projection::ALL.iter().try_fold(0usize, |total, &p| {
            let (w, _) = tensors.slice_bytes(p, tp)?;
            Ok(total + experts * w + tensors.scale_region_bytes(p, tp)?)
        })
    }

    pub fn load(library: &'a NativeLibrary, tensors: &Fp8ExpertTensors, layer: usize, tp: usize, rank: usize)
        -> Result<Self> {
        ensure!(tensors.has_layer(layer), "layer {layer} has no routed FP8 experts");
        let experts = tensors.shape().experts;
        let mut regions = Vec::with_capacity(6);
        for projection in Fp8Projection::ALL {
            let (w_bytes, s_bytes) = tensors.slice_bytes(projection, tp)?;
            let mut weights = vec![0u8; experts * w_bytes];
            let mut scales = vec![0u8; tensors.scale_region_bytes(projection, tp)?];
            // NVFP4: the experts' FP32 alphas follow the scale grids.
            let (grids, alphas) = scales.split_at_mut(experts * s_bytes);
            let alpha_bytes = if alphas.is_empty() { 0 } else { 4 };
            let mut alpha_slots: Vec<&mut [u8]> = alphas.chunks_exact_mut(4).collect();
            alpha_slots.resize_with(experts, Default::default);
            let mut jobs: Vec<(usize, &mut [u8], &mut [u8], &mut [u8])> = weights.chunks_exact_mut(w_bytes)
                .zip(grids.chunks_exact_mut(s_bytes)).zip(alpha_slots).enumerate()
                .map(|(e, ((w, s), a))| (e, w, s, a)).collect();
            let per = jobs.len().div_ceil(READERS);
            std::thread::scope(|scope| -> Result<()> {
                let handles: Vec<_> = jobs.chunks_mut(per).map(|chunk| scope.spawn(move || -> Result<()> {
                    let mut staging = Vec::new();
                    for (expert, w, s, a) in chunk.iter_mut() {
                        tensors.read_slice(layer, *expert, projection, tp, rank, w, s, &mut staging)?;
                        if alpha_bytes > 0 {
                            a.copy_from_slice(&tensors.read_alpha(layer, *expert, projection)?.to_le_bytes());
                        }
                    }
                    Ok(())
                })).collect();
                for handle in handles {
                    handle.join().map_err(|_| anyhow::anyhow!("FP8 expert reader panicked"))??;
                }
                Ok(())
            })?;
            for bytes in [&weights, &scales] {
                let region = DeviceAllocation::new(library, bytes.len())?;
                library.copy_h2d(region.buffer, bytes)?;
                regions.push(region);
            }
        }
        // Regions in [w1, s1, w3, s3, w2, s2] order (gate, up, down).
        Ok(Self { layer, regions })
    }

    fn pointers(&self) -> [*mut c_void; 6] {
        std::array::from_fn(|i| self.regions[i].buffer.ptr)
    }
}

/// Resident FP8 layers of one TP slice and the package that runs them.
pub(crate) struct Fp8Experts<'a> {
    // Drop order: the module goes last; callers drain their streams first.
    pub layers: Vec<Fp8Layer<'a>>,
    scratch: DeviceAllocation<'a>,
    /// A second package over the same weights taking BF16 rows (a Spark
    /// rank whose coordinator sends unquantized expert input); it shares
    /// `scratch`, sized for both.
    pub bf16_module: Option<Fp8MoeModule>,
    pub module: Fp8MoeModule,
    pub tp: usize,
    pub rank: usize,
}

impl<'a> Fp8Experts<'a> {
    /// Loads the package at `directory` and layers `layers` of the slice
    /// `rank` of `tp`, with scratch for `capacity` rows.
    pub fn load(library: &'a NativeLibrary, tensors: &Fp8ExpertTensors, directory: &Path,
        layers: std::ops::Range<usize>, tp: usize, rank: usize, capacity: usize, budget: usize) -> Result<Self> {
        // SAFETY: a trusted package for the current device; the owner drains
        // its streams before dropping.
        let module = unsafe { load_module(directory) }
            .with_context(|| format!("FP8 expert package {} (build it with package_fp8_moe_aot.py)",
                directory.display()))?;
        let info = module.info().clone();
        let shape = tensors.shape();
        let weights = match tensors.format() {
            ExpertFormat::Fp8Block128 => Fp8MoeWeights::Fp8,
            ExpertFormat::Mxfp4 => Fp8MoeWeights::Mxfp4,
            ExpertFormat::Nvfp4 => Fp8MoeWeights::Nvfp4,
        };
        ensure!(info.hidden == shape.hidden && info.experts == shape.experts && info.topk == shape.topk
            && info.intermediate == shape.intermediate && info.tp == tp && info.weights == weights
            && info.slice == tensors.slice(tp)?,
            "FP8 package {} ({info:?}) does not serve this checkpoint at TP{tp}", directory.display());
        let top = info.capacity_for(capacity)
            .with_context(|| format!("FP8 package has no capacity for {capacity} rows"))?;
        let scratch_bytes = module.scratch_bytes(top)?;
        let resident = Fp8Layer::bytes(tensors, tp)? * layers.len();
        ensure!(resident + scratch_bytes <= budget,
            "FP8 experts need {} GiB resident + {} MiB scratch; budget {} GiB", resident >> 30, scratch_bytes >> 20,
            budget >> 30);
        let layers = layers.map(|layer| {
            let started = std::time::Instant::now();
            let loaded = Fp8Layer::load(library, tensors, layer, tp, rank)?;
            tracing::info!(layer, tp, rank, elapsed_ms = started.elapsed().as_millis() as u64,
                "FP8 expert layer resident");
            Ok(loaded)
        }).collect::<Result<Vec<_>>>()?;
        let scratch = DeviceAllocation::new(library, scratch_bytes.max(256))?;
        Ok(Self { layers, scratch, bf16_module: None, module, tp, rank })
    }

    /// Adds the BF16-input package at `directory` (same geometry and TP
    /// slice), growing the shared scratch when it needs more.
    pub fn add_bf16_module(&mut self, library: &'a NativeLibrary, directory: &Path, capacity: usize) -> Result<()> {
        // SAFETY: a trusted package for the current device; dropped with this object.
        let module = unsafe { load_module(directory) }
            .with_context(|| format!("BF16-input FP8 expert package {}", directory.display()))?;
        let (info, main) = (module.info().clone(), self.module.info());
        ensure!(!info.wire_input && info.weights == main.weights && info.hidden == main.hidden && info.experts == main.experts
            && info.topk == main.topk && info.intermediate == main.intermediate && info.tp == main.tp,
            "{} ({info:?}) is not the BF16-input form of this FP8 package", directory.display());
        let top = info.capacity_for(capacity)
            .with_context(|| format!("BF16-input FP8 package has no capacity for {capacity} rows"))?;
        let bytes = module.scratch_bytes(top)?;
        if bytes > self.scratch.buffer.bytes {
            self.scratch = DeviceAllocation::new(library, bytes)?;
        }
        self.bf16_module = Some(module);
        Ok(())
    }

    pub fn index_of(&self, layer: usize) -> Result<usize> {
        self.layers.iter().position(|l| l.layer == layer)
            .with_context(|| format!("FP8 expert layer {layer} is not resident"))
    }

    /// Whether the package takes FP8 K32 wire rows (else BF16 rows).
    pub fn wire_input(&self) -> bool {
        self.module.info().wire_input
    }

    /// Routed experts of resident layer `index` for `rows` input rows (wire
    /// or BF16, per `wire_input`) into `out` (BF16 `[rows, H]`).
    ///
    /// # Safety
    /// `wire`, `ids` (I32 `[rows, k]`), `weights` (F32 `[rows, k]`) and `out`
    /// are live device buffers of those extents; the stream is drained before
    /// any of them, or this object, is released.
    pub unsafe fn run(&self, index: usize, rows: usize, wire: *mut c_void, ids: *mut c_void, weights: *mut c_void,
        out: *mut c_void, stream: *mut c_void) -> Result<()> {
        self.run_with(&self.module, index, rows, wire, ids, weights, out, stream)
    }

    /// `run` over BF16 input rows through the BF16-input package.
    ///
    /// # Safety
    /// As `run`, with `rows` BF16 `[rows, H]` input rows.
    pub unsafe fn run_bf16(&self, index: usize, rows: usize, input: *mut c_void, ids: *mut c_void,
        weights: *mut c_void, out: *mut c_void, stream: *mut c_void) -> Result<()> {
        let module = self.bf16_module.as_ref().context("no BF16-input FP8 expert package is loaded")?;
        self.run_with(module, index, rows, input, ids, weights, out, stream)
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn run_with(&self, module: &Fp8MoeModule, index: usize, rows: usize, input: *mut c_void,
        ids: *mut c_void, weights: *mut c_void, out: *mut c_void, stream: *mut c_void) -> Result<()> {
        let layer = self.layers.get(index).context("FP8 expert layer index out of range")?;
        let [w1, s1, w3, s3, w2, s2] = layer.pointers();
        let pointers: [*mut c_void; FP8_MOE_POINTERS] =
            [input, ids, weights, w1, s1, w3, s3, w2, s2, out, self.scratch.buffer.ptr];
        module.launch(&pointers, rows, stream)
    }
}
