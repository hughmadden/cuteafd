//! Official coordinator tensors retain their native checkpoint representations.
use crate::shared::memory::{DeviceAllocation, HostAllocation};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary};
use cuteafd_loader::OfficialV41Catalog;
use std::collections::{BTreeMap, BTreeSet};
#[path = "v41_tensors/vocabulary_shard.rs"]
mod vocabulary_shard;
pub(crate) use vocabulary_shard::VocabularyShard;

pub(crate) struct NativeRtxTensors<'a> {
    tensors: BTreeMap<String, DeviceAllocation<'a>>,
    resident_bytes: usize,
}
impl<'a> NativeRtxTensors<'a> {
    pub fn plan(catalog: &OfficialV41Catalog, names: &[String]) -> Result<usize> {
        ensure!(!names.is_empty(), "RTX tensor set is empty");
        let mut seen = BTreeSet::new();
        names.iter().try_fold(0usize, |total, name| {
            ensure!(seen.insert(name), "duplicate RTX tensor {name}");
            ensure!(
                !name.contains(".ffn.experts."),
                "routed expert weights require native expert packing"
            );
            let bytes = usize::try_from(catalog.device_tensor_bytes(name, None)?)?;
            total
                .checked_add(bytes)
                .context("RTX resident tensor budget overflow")
        })
    }
    /// Admission precedes allocation and payload reads; bounded pinned staging is
    /// released after synchronous uploads. Engram tables cannot enter this path.
    pub fn load(
        library: &'a NativeLibrary,
        catalog: &OfficialV41Catalog,
        names: &[String],
        device_budget: usize,
        staging_bytes: usize,
    ) -> Result<Self> {
        let resident_bytes = Self::plan(catalog, names)?;
        ensure!(
            resident_bytes <= device_budget,
            "RTX tensor set exceeds device budget"
        );
        ensure!(
            staging_bytes > 0 && staging_bytes <= 64 * 1024 * 1024,
            "RTX pinned staging must be 1 byte through 64 MiB"
        );
        let mut staging = HostAllocation::new(library, staging_bytes)?;
        let mut tensors = BTreeMap::new();
        for name in names {
            let reader = catalog.coordinator_tensor_reader(name)?;
            let bytes = usize::try_from(reader.bytes())?;
            let allocation = DeviceAllocation::new(library, bytes)?;
            let mut offset = 0;
            while offset < bytes {
                let count = staging_bytes.min(bytes - offset);
                let source = &mut staging.bytes_mut()[..count];
                reader.read_into(offset as u64, source)?;
                let mut destination = allocation.buffer;
                destination.ptr = unsafe { destination.ptr.cast::<u8>().add(offset).cast() };
                destination.bytes = count;
                library.copy_h2d(destination, source)?;
                offset += count;
            }
            tensors.insert(name.clone(), allocation);
        }
        Ok(Self {
            tensors,
            resident_bytes,
        })
    }
    /// Borrowed native representation; never free or retain after the owner drops.
    pub fn get(&self, name: &str) -> Result<CuteafdDeviceBuffer> {
        Ok(self
            .tensors
            .get(name)
            .with_context(|| format!("RTX tensor is not resident: {name}"))?
            .buffer)
    }
    pub fn resident_bytes(&self) -> usize {
        self.resident_bytes
    }
}

/// One coordinator copy of the official BF16 vocabulary weight, shared by the
/// backbone and dSpark; each execution owns its own handle and workspace.
pub(crate) struct VocabularyHead<'library> {
    tensors: NativeRtxTensors<'library>,
    fp8: Option<crate::shared::fp8_linear::Fp8Weight<'library>>,
}

/// E4M3 vocabulary heads (`CUTEAFD_V41_FP8_HEAD`): `draft` projects only the
/// dSpark draft head through an FP8 copy (outputs unchanged, acceptance may
/// move), `all` the target head too; off by default. Copies use per-row x
/// 128-K FP32 scales (`fp8_linear`, the better of amax and power-of-two per
/// block) and the W8A16 tensor-core GEMV; the BF16 head stays resident.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Fp8Head { Off, Draft, All }
pub(crate) fn fp8_head() -> Fp8Head {
    static MODE: std::sync::OnceLock<Fp8Head> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| {
        let mode = match std::env::var("CUTEAFD_V41_FP8_HEAD").as_deref() {
            Ok("draft") => Fp8Head::Draft,
            Ok("all" | "1" | "on") => Fp8Head::All,
            _ => Fp8Head::Off,
        };
        if mode != Fp8Head::Off { tracing::info!(?mode, "V4.1 FP8 vocabulary heads"); }
        mode
    })
}

/// Packs an FP8 copy of a resident `[rows, 5120]` BF16 head slice on the
/// current device (drained before returning).
fn pack_fp8<'a>(library: &'a NativeLibrary, weight: CuteafdDeviceBuffer, rows: usize)
    -> Result<crate::shared::fp8_linear::Fp8Weight<'a>> {
    ensure!(weight.bytes == rows * 10240, "FP8 head copy of {rows} rows differs from the BF16 head");
    let stream = library.cuda_stream_create()?;
    let packed = crate::shared::fp8_linear::Fp8Weight::pack(library, weight.ptr, rows, 5120,
        crate::shared::fp8_linear::Fp8Scales::Best, stream);
    // SAFETY: the stream was created above and holds only the pack.
    let drained = unsafe { library.cuda_stream_synchronize(stream) };
    unsafe { library.cuda_stream_destroy(stream)?; }
    let packed = packed?;
    drained?;
    Ok(packed)
}

/// A vocabulary projection through the FP8 copy when the caller holds scratch
/// for it (see [`fp8_head`]), else the BF16 head.
///
/// # Safety
/// As [`cuteafd_ffi::V41VocabularyProjection::launch`]; `scratch` was sized by
/// [`fp8_scratch`] for at least `rows` rows of this head.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn project_vocabulary(library: &NativeLibrary, projection: &cuteafd_ffi::V41VocabularyProjection<'_>,
    bf16: CuteafdDeviceBuffer, fp8: Option<(&crate::shared::fp8_linear::Fp8Weight<'_>, &DeviceAllocation<'_>)>,
    input: CuteafdDeviceBuffer, logits: CuteafdDeviceBuffer, rows: usize, stream: *mut std::ffi::c_void) -> Result<()> {
    match fp8 {
        Some((weight, scratch)) => {
            ensure!(input.bytes >= rows * 10240 && logits.bytes >= rows * weight.n * 4
                && input.device_id == logits.device_id, "FP8 vocabulary projection buffers differ");
            unsafe { weight.apply(library, input.ptr, logits.ptr, true, rows, 0, weight.n, scratch, stream) }
        }
        None => unsafe { projection.launch(input, bf16, logits, rows, stream) },
    }
}

/// GEMV scratch for `rows` rows of an FP8 head copy, when `site` uses one.
pub(crate) fn fp8_scratch<'a>(library: &'a NativeLibrary, fp8: Option<&crate::shared::fp8_linear::Fp8Weight<'_>>,
    rows: usize, site: Fp8Head) -> Result<Option<DeviceAllocation<'a>>> {
    let enabled = match fp8_head() { Fp8Head::Off => false, Fp8Head::Draft => site == Fp8Head::Draft, Fp8Head::All => true };
    match fp8.filter(|_| enabled) {
        Some(weight) => Ok(Some(crate::shared::fp8_linear::scratch(library, rows, &[(weight.k, weight.n)])?)),
        None => Ok(None),
    }
}
impl<'library> VocabularyHead<'library> {
    pub fn plan(catalog: &OfficialV41Catalog) -> Result<usize> {
        NativeRtxTensors::plan(catalog, &["head.weight".into()])
    }
    pub fn load(
        library: &'library NativeLibrary,
        catalog: &OfficialV41Catalog,
        budget: usize,
        staging_bytes: usize,
    ) -> Result<Self> {
        let tensors = NativeRtxTensors::load(
            library,
            catalog,
            &["head.weight".into()],
            budget,
            staging_bytes,
        )?;
        let fp8 = (fp8_head() != Fp8Head::Off)
            .then(|| pack_fp8(library, tensors.get("head.weight")?, 129280)).transpose()?;
        Ok(Self { tensors, fp8 })
    }
    pub fn weight(&self) -> Result<CuteafdDeviceBuffer> {
        self.tensors.get("head.weight")
    }
    /// The FP8 copy ([`fp8_head`]), when one was packed.
    pub fn fp8(&self) -> Option<&crate::shared::fp8_linear::Fp8Weight<'library>> {
        self.fp8.as_ref()
    }
}
