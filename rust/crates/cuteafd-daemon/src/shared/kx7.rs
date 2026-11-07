//! KX7 exact copies of BF16 matrices born FP8 (`native/shared/cuda/kx7.cu`; format and packer in
//! `cuteafd_core::kx7`): packed and verified bit for bit on the host at load, from checkpoint
//! BF16 that never reaches the GPU, in format v1.1 ([`V11_LAYOUT`] with [`V11_CODE`]: lane16,
//! K split 4, 8-row ragged interleave, the two-class code; byte-identical to the reference
//! tool's `dump_pack`, the file set the KX7 GEMV reads), then held as four device buffers per
//! matrix. v0's linear decodes into one BF16 workspace with the reference device decoder and
//! runs the BF16 cuBLAS GEMM over it (bit for bit that GEMM over the source weights).
//!
//! Everything format specific sits behind three seams: the host encode ([`pack`]), the device
//! buffers ([`Kx7Image`], [`Kx7Weights`] and their [`Kx7DevicePack`] header) and the native
//! decode. Callers see only [`Kx7Weights::linear`] and byte counts.
use crate::shared::memory::DeviceAllocation;
use anyhow::{ensure, Context, Result};
use cuteafd_core::kx7::{CodeKind, Kx7DeviceBytes, Kx7Layout, Kx7Pack, READ_AHEAD_BYTES, V11_CODE, V11_LAYOUT};
use cuteafd_ffi::{CuteafdDeviceBuffer, Kx7DevicePack, NativeLibrary};
use std::cell::Cell;
use std::ffi::c_void;
use std::time::Instant;

/// The layout of every resident pack (v1.1).
pub(crate) const LAYOUT: Kx7Layout = V11_LAYOUT;
/// The code of every resident pack (v1.1).
pub(crate) const CODE: CodeKind = V11_CODE;
/// The smallest allocation the loader makes (its uploads take at least 256 bytes).
pub(crate) const FLOOR: u64 = 256;

/// One pack's device buffers as host bytes, in [`Kx7DevicePack`] order (the decode tables, the
/// stream offsets and lengths, the streams followed by their read-ahead zeros), and the header
/// scalars of its code.
pub(crate) struct Kx7Image {
    pub name: String,
    pub n: usize,
    pub k: usize,
    esc_len: u32,
    table_words: usize,
    buffers: [Vec<u8>; 4],
}

impl Kx7Image {
    /// The image of `pack` (of the matrix `name`), refused above the admission bound
    /// ([`Kx7DeviceBytes::bound`]): a matrix that is not FP8-born can pass 8 bits a weight.
    pub fn new(name: &str, pack: &Kx7Pack) -> Result<Self> {
        let header = pack.header();
        ensure!(header.layout == LAYOUT && pack.code().kind() == CODE,
            "{name}: KX7 pack in {:?} / {:?}, resident packs are v1.1", header.layout, pack.code().kind());
        let (n, k) = (header.n, header.k);
        let bytes = Kx7DeviceBytes::of(pack)?;
        let bound = Kx7DeviceBytes::bound(n, k, LAYOUT, CODE)?;
        ensure!(bytes.within(&bound), "{name}: its KX7 pack takes {:.3} bits a weight, above the admitted 8 \
            (its BF16 values are not FP8-born); serve it with --kda-exact off",
            pack.streams().len() as f64 * 8.0 / (n * k) as f64);
        let tables = pack.device_tables()?;
        let mut streams = Vec::with_capacity(pack.streams().len() + READ_AHEAD_BYTES);
        streams.extend_from_slice(pack.streams());
        streams.resize(pack.streams().len() + READ_AHEAD_BYTES, 0);
        let image = Self { name: name.into(), n, k, esc_len: pack.code().esc_len(), table_words: tables.len(),
            buffers: [tables.iter().flat_map(|w| w.to_le_bytes()).collect(), pack.offsets_bin(), pack.lengths_bin(),
                streams] };
        debug_assert_eq!(image.device_bytes(), bytes);
        Ok(image)
    }

    pub fn device_bytes(&self) -> Kx7DeviceBytes {
        let [tables, offsets, lengths, streams] = self.buffers.each_ref().map(|b| b.len() as u64);
        Kx7DeviceBytes { tables, offsets, lengths, streams }
    }

    /// The buffers in device order: tables, offsets, lengths, streams.
    pub fn buffers(&self) -> [&[u8]; 4] {
        self.buffers.each_ref().map(Vec::as_slice)
    }
}

/// Packs BF16 matrices `(name, little-endian bytes, n, k)` in parallel (rayon, across and within
/// matrices), verifies each against its source and checks it against its bound. Counted as the
/// load's quantization time and in this thread's [`PackSplit`].
pub(crate) fn pack(matrices: &[(String, Vec<u8>, usize, usize)]) -> Result<Vec<Kx7Image>> {
    let started = Instant::now();
    let bf16: usize = matrices.iter().map(|(_, bytes, _, _)| bytes.len()).sum();
    let images = crate::shared::memory::load_timing::quantize(bf16, || -> Result<Vec<Kx7Image>> {
        let mut values = Vec::with_capacity(matrices.len());
        for (name, bytes, n, k) in matrices {
            ensure!(bytes.len() == n * k * 2, "{name}: {} bytes, a BF16 [{n}, {k}] takes {}", bytes.len(), n * k * 2);
            values.push(bytes.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect::<Vec<u16>>());
        }
        let sources: Vec<_> = values.iter().zip(matrices).map(|(v, (_, _, n, k))| (v.as_slice(), *n, *k)).collect();
        let packs = cuteafd_core::kx7::encode_verified_all(&sources, LAYOUT, CODE)
            .context("KX7 pack of the KDA projections (serve them with --kda-exact off)")?;
        packs.iter().zip(matrices).map(|(pack, (name, _, _, _))| Kx7Image::new(name, pack)).collect()
    })?;
    let seconds = started.elapsed().as_secs_f64();
    let mut stats = STATS.with(Cell::get);
    for (image, (_, _, n, k)) in images.iter().zip(matrices) {
        stats.matrices += 1;
        stats.bf16_bytes += (n * k * 2) as u64;
        stats.device_bytes += image.device_bytes().total(FLOOR);
        stats.bound_bytes += Kx7DeviceBytes::bound(*n, *k, LAYOUT, CODE)?.total(FLOOR);
    }
    stats.seconds += seconds;
    STATS.with(|cell| cell.set(stats));
    Ok(images)
}

/// Packs made on this thread: matrices, their BF16 bytes, the device bytes they take, the
/// admission bound of those and the packing seconds (host conversion, encode and verify).
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub(crate) struct PackStats {
    pub matrices: u64,
    pub bf16_bytes: u64,
    pub device_bytes: u64,
    pub bound_bytes: u64,
    pub seconds: f64,
}

thread_local! {
    static STATS: Cell<PackStats> = const { Cell::new(PackStats { matrices: 0, bf16_bytes: 0, device_bytes: 0,
        bound_bytes: 0, seconds: 0.0 }) };
}

/// Brackets one load on this thread and reports the packs made inside it.
pub(crate) struct PackSplit(PackStats);
impl PackSplit {
    pub fn start() -> Self {
        Self(STATS.with(Cell::get))
    }

    pub fn finish(self) -> PackStats {
        let now = STATS.with(Cell::get);
        PackStats { matrices: now.matrices - self.0.matrices, bf16_bytes: now.bf16_bytes - self.0.bf16_bytes,
            device_bytes: now.device_bytes - self.0.device_bytes, bound_bytes: now.bound_bytes - self.0.bound_bytes,
            seconds: now.seconds - self.0.seconds }
    }

    /// Logs the bracket's packs, if any.
    pub fn log(self, label: &str) -> PackStats {
        let stats = self.finish();
        if stats.matrices > 0 {
            tracing::info!(label, matrices = stats.matrices, bf16_bytes = stats.bf16_bytes,
                packed_bytes = stats.device_bytes, admitted_bytes = stats.bound_bytes,
                slack_bytes = stats.bound_bytes.saturating_sub(stats.device_bytes),
                bits_per_weight = stats.device_bytes as f64 * 16.0 / stats.bf16_bytes.max(1) as f64,
                pack_seconds = stats.seconds,
                "KX7 exact packs made and verified on the host (BF16 never resident)");
        }
        stats
    }
}

/// One resident KX7 matrix [n, k]: its four buffers on one GPU and its code's header scalars.
pub(crate) struct Kx7Weights<'a> {
    buffers: [DeviceAllocation<'a>; 4],
    esc_len: u32,
    table_words: usize,
    pub n: usize,
    pub k: usize,
}

impl<'a> Kx7Weights<'a> {
    /// Uploads `image` to the current device (each buffer at least [`FLOOR`] bytes; device
    /// allocations are 256-byte aligned, as the streams' 16-byte quad loads need).
    pub fn upload(library: &'a NativeLibrary, image: &Kx7Image) -> Result<Self> {
        let _memory_format = cuteafd_ffi::memory_ledger::format("kx7");
        let [tables, offsets, lengths, streams] = image.buffers().map(|bytes| -> Result<DeviceAllocation<'a>> {
            let allocation = DeviceAllocation::new(library, (bytes.len() as u64).max(FLOOR) as usize)?;
            crate::shared::memory::load_timing::upload(bytes.len(), || library.copy_h2d(allocation.buffer, bytes))?;
            Ok(allocation)
        });
        Ok(Self { buffers: [tables?, offsets?, lengths?, streams?], esc_len: image.esc_len,
            table_words: image.table_words, n: image.n, k: image.k })
    }

    /// Device bytes of the four allocations.
    pub fn bytes(&self) -> usize {
        self.buffers.iter().map(|b| b.buffer.bytes).sum()
    }

    /// The allocations in the order a linear reads them: tables, offsets, lengths, streams.
    pub fn ranges(&self) -> [crate::shared::l2_prefetch::Range; 4] {
        self.buffers.each_ref().map(|b| (b.buffer.ptr.cast_const(), b.buffer.bytes))
    }

    /// The native header naming this matrix's buffers.
    pub fn device_pack(&self) -> Result<Kx7DevicePack> {
        let CodeKind::TwoClass { rare_bits, .. } = CODE else { anyhow::bail!("resident KX7 packs are two-class") };
        Kx7DevicePack::v11(self.n, self.k, LAYOUT.slices, self.esc_len, rare_bits, self.table_words,
            self.buffers.each_ref().map(|b| b.buffer.ptr.cast_const()))
    }

    /// `out` [rows, n] BF16 = `x` [rows, k] BF16 @ W^T, decoding through `workspace` (at least
    /// `kx7_workspace(rows, n, k)` bytes, free on `stream` for the call).
    ///
    /// # Safety
    /// `x` and `out` are live device buffers of those shapes on this matrix's device; `workspace`
    /// stays live and unused by any other stream until the call's work retires.
    pub unsafe fn linear(&self, library: &NativeLibrary, x: *const c_void, out: *mut c_void, rows: usize,
        workspace: CuteafdDeviceBuffer, stream: *mut c_void) -> Result<()> {
        let pack = self.device_pack()?;
        // SAFETY: the caller's contract; the pack names this matrix's live allocations.
        unsafe { library.kx7_linear(x, &pack, out, rows, workspace.ptr, workspace.bytes, stream) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_core::kx7::{encode_verified, Kx7Labels, Kx7Source};

    /// SplitMix64, as `cuteafd_core::kx7`'s tests.
    struct Mix(u64);
    impl Mix {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }
    }

    /// 85% an E4M3 value times a power of two, 15% ordinary BF16, over `exponents` exponents
    /// (integers only); 4 give about 7.5 bits a weight in v1.1 at these sizes, as GLM 5.3 Flash's KDA tensors.
    fn fp8_born(n: usize, k: usize, seed: u64, exponents: u64) -> Vec<u16> {
        let mut r = Mix(seed);
        (0..n * k).map(|_| {
            let sign = ((r.next() % 2) as u16) << 15;
            let exp = (124 - exponents as u16 + (r.next() % exponents) as u16) << 7;
            let fp8 = (r.next() >> 11) < (1u64 << 53) / 100 * 85;
            let mant = if fp8 { ((r.next() % 8) as u16) << 4 } else { (r.next() % 128) as u16 };
            sign | exp | mant
        }).collect()
    }

    fn bytes(w: &[u16]) -> Vec<u8> {
        w.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    #[test]
    fn kx7_images_are_the_packs_device_buffers() {
        // Large enough that the rare patterns' ESC + rank bits amortize (about 7.5 bits a weight).
        let w = fp8_born(64, 4096, 1, 4);
        let pack = encode_verified(&w, 64, 4096, LAYOUT, CODE).unwrap();
        let image = Kx7Image::new("w", &pack).unwrap();
        let [tables, offsets, lengths, streams] = image.buffers();
        assert_eq!(tables, pack.device_tables().unwrap().iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
        assert_eq!((offsets, lengths), (pack.offsets_bin().as_slice(), pack.lengths_bin().as_slice()));
        assert_eq!(&streams[..pack.streams().len()], pack.streams());
        assert!(streams[pack.streams().len()..].iter().all(|&b| b == 0));
        assert_eq!(streams.len(), pack.streams().len() + READ_AHEAD_BYTES);
        assert_eq!(image.device_bytes(), Kx7DeviceBytes::of(&pack).unwrap());
        assert_eq!((image.esc_len, image.table_words), (pack.code().esc_len(), 4096));
        assert_eq!(offsets.len(), 4 * (64 * 16 + 1));
        assert!(image.device_bytes().streams < 64 * 4096, "FP8-born over 4 exponents fits 8 bits a weight");
    }

    #[test]
    fn kx7_pack_verifies_counts_and_refuses_what_the_bound_cannot_hold() {
        let split = PackSplit::start();
        let (a, b) = (fp8_born(64, 4096, 3, 4), fp8_born(32, 4096, 4, 4));
        let images = pack(&[("a".into(), bytes(&a), 64, 4096), ("b".into(), bytes(&b), 32, 4096)]).unwrap();
        assert_eq!(images.iter().map(|i| (i.name.as_str(), i.n, i.k)).collect::<Vec<_>>(),
            [("a", 64, 4096), ("b", 32, 4096)]);
        let stats = split.finish();
        assert_eq!((stats.matrices, stats.bf16_bytes), (2, 2 * 96 * 4096));
        assert!(stats.device_bytes < stats.bound_bytes && stats.device_bytes < stats.bf16_bytes);
        // 16 exponents spread the ordinary values over 4,096 patterns: past 8 bits a weight.
        let wide = fp8_born(64, 1024, 9, 16);
        let error = pack(&[("wide".into(), bytes(&wide), 64, 1024)]).err().unwrap().to_string();
        assert!(error.contains("wide") && error.contains("--kda-exact off"), "{error}");
        // Random 16-bit patterns take no two-class code at all.
        let mut r = Mix(9);
        let noise: Vec<u16> = (0..64 * 1024).map(|_| r.next() as u16).collect();
        assert!(format!("{:#}", pack(&[("noise".into(), bytes(&noise), 64, 1024)]).err().unwrap())
            .contains("--kda-exact off"));
        // A source of the wrong length or shape is refused before packing.
        assert!(pack(&[("short".into(), vec![0; 10], 32, 1024)]).is_err());
        assert!(pack(&[("narrow".into(), bytes(&a[..32 * 128]), 32, 128)]).is_err());
    }

    /// The fixtures of `native/tests/kx7_selftest.cu` (`native/tests/fixtures/kx7`): each a v1.1
    /// file set (`Kx7Pack::dump`) with its source and the device tables (`device_tables.bin`, the
    /// words this module uploads). Regenerate with
    /// `CUTEAFD_KX7_FIXTURES=<dir> cargo test -p cuteafd-daemon kx7_dump_selftest_fixtures -- --ignored`;
    /// `CUTEAFD_KX7_FIXTURES_FULL=1` adds the GLM 5.3 Flash q/k/v and o shapes (not committed).
    #[test]
    #[ignore = "writes the native selftest's pack fixtures"]
    fn kx7_dump_selftest_fixtures() {
        let root = std::path::PathBuf::from(std::env::var_os("CUTEAFD_KX7_FIXTURES").expect("CUTEAFD_KX7_FIXTURES"));
        let mut fixtures = vec![("fp8_born_32x1024", fp8_born(32, 1024, 2026, 4), 32, 1024),
            ("rare_heavy_16x1024", fp8_born(16, 1024, 7, 16), 16, 1024)];
        if std::env::var_os("CUTEAFD_KX7_FIXTURES_FULL").is_some() {
            fixtures.push(("q_8192x4096", fp8_born(8192, 4096, 1, 4), 8192, 4096));
            fixtures.push(("o_4096x8192", fp8_born(4096, 8192, 2, 4), 4096, 8192));
        }
        for (name, w, n, k) in fixtures {
            let pack = encode_verified(&w, n, k, LAYOUT, CODE).unwrap();
            let dir = root.join(name);
            let labels = Kx7Labels { label: name.into(), components: vec![name.into()] };
            pack.dump(&dir, &labels, Some(&Kx7Source::matrix(&w, n, k).unwrap())).unwrap();
            let words: Vec<u8> = pack.device_tables().unwrap().iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(dir.join("device_tables.bin"), words).unwrap();
        }
    }

    /// The committed selftest fixtures are this packer's packs of their sources.
    #[test]
    fn kx7_selftest_fixtures_are_current_packs() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../native/tests/fixtures/kx7");
        for name in ["fp8_born_32x1024", "rare_heavy_16x1024"] {
            let dir = root.join(name);
            let pack = Kx7Pack::load(&dir).unwrap();
            let source: Vec<u16> = std::fs::read(dir.join("source.bf16.bin")).unwrap().chunks_exact(2)
                .map(|b| u16::from_le_bytes([b[0], b[1]])).collect();
            let (n, k) = (pack.header().n, pack.header().k);
            assert_eq!(pack, encode_verified(&source, n, k, LAYOUT, CODE).unwrap(), "{name}");
            let words: Vec<u8> = pack.device_tables().unwrap().iter().flat_map(|v| v.to_le_bytes()).collect();
            assert_eq!(std::fs::read(dir.join("device_tables.bin")).unwrap(), words, "{name}");
            assert!(pack.code().n_rare() > 0, "{name} exercises the ESC path");
        }
    }
}
