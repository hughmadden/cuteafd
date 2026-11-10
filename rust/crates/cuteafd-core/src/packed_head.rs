//! The exact 12-bit BF16 vocabulary head of `native/shared/cuda/packed_head.cu`:
//! the resident allocation the planner admits, the pack's status words, and a CPU
//! reference of the format (window choice, the two planes in MMA A-fragment
//! order, the exception table) that pins the layout the kernels read.
//!
//! Row r keeps a base exponent e0: of the windows [e0, e0 + 15] of BF16 biased
//! exponents (e0 <= 240), the one holding the most of its values, ties to the
//! lowest e0. A value whose exponent e is in the window is a coarse byte
//! (sign << 7 | (e - e0) << 3 | mantissa bits 6..4) and a fine nibble (mantissa
//! bits 3..0). A value outside is an exception: its planes hold it clamped into
//! the window with its sign (below: exponent e0, mantissa 0; above: e0 + 15,
//! mantissa 127) and a CSR table holds (column, exact BF16), in column order.
//! The exact GEMV adds (exact - coded) x per exception after its K reduction;
//! the coarse GEMV (drafters) reads the coarse plane alone, with mantissa bits
//! 3..0 at their midpoint 0b1000.
use thiserror::Error;

/// Largest base exponent: the window [e0, e0 + 15] stays inside the 8-bit field.
pub const MAX_BASE: u8 = 240;
/// Exponents per row window.
pub const WINDOW: u32 = 16;
/// The exception table holds one entry per this many weights. DeepSeek V4.1's
/// head needs about one per 13,600 (48,615 of 662M, at most 6 in a row).
pub const VALUES_PER_EXCEPTION: u64 = 1024;
/// Status words (u64) the native pack writes.
pub const STATUS_WORDS: usize = 4;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum PackedHeadError {
    #[error("packed head [{n}, {k}]: needs n a positive multiple of 16, k a positive multiple of 128 and n * k < 2^32")]
    Shape { n: usize, k: usize },
    #[error("packed head: {exceptions} values lie outside their rows' {WINDOW}-exponent windows, but the table \
             holds {capacity} (one per {VALUES_PER_EXCEPTION} weights)")]
    Capacity { exceptions: u64, capacity: u64 },
    #[error("packed head verification failed in {rows} rows (first: row {first}): the planes and table do not \
             decode to the BF16 source")]
    Verification { rows: u64, first: u64 },
}

/// The regions of one packed head, in allocation order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Region {
    /// `n * k` bytes: sign, 4-bit code, mantissa bits 6..4.
    Coarse,
    /// `n * k / 2` bytes: mantissa bits 3..0.
    Fine,
    /// `n` bytes: each row's base exponent.
    Base,
    /// `n + 1` u32: each row's first table entry.
    RowPtr,
    /// `capacity` u32: exception columns.
    Columns,
    /// `capacity` BF16: exact exception values.
    Values,
    /// [`STATUS_WORDS`] u64 the pack writes.
    Status,
}

impl Region {
    pub const ALL: [Region; 7] =
        [Region::Coarse, Region::Fine, Region::Base, Region::RowPtr, Region::Columns, Region::Values, Region::Status];
}

const fn align256(bytes: u64) -> u64 { bytes.div_ceil(256) * 256 }

/// One packed head's single allocation: its regions in [`Region::ALL`] order,
/// each 256-byte aligned. The native planes array is the first six regions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackedHeadLayout {
    pub n: usize,
    pub k: usize,
    /// Exception table entries.
    pub capacity: usize,
    offsets: [usize; 7],
    total: usize,
}

impl PackedHeadLayout {
    /// The layout of an `[n, k]` head (n % 16 == 0, k % 128 == 0, n * k < 2^32).
    pub fn new(n: usize, k: usize) -> Result<Self, PackedHeadError> {
        let elements = (n as u64).checked_mul(k as u64);
        if n == 0 || n % 16 != 0 || k == 0 || k % 128 != 0 || !elements.is_some_and(|e| e < 1 << 32) {
            return Err(PackedHeadError::Shape { n, k });
        }
        let sizes = Self::region_bytes(n as u64, k as u64);
        let mut offsets = [0usize; 7];
        let mut at = 0u64;
        for (offset, bytes) in offsets.iter_mut().zip(sizes) {
            *offset = at as usize;
            at += align256(bytes);
        }
        Ok(Self { n, k, capacity: Self::capacity(n as u64, k as u64) as usize, offsets, total: at as usize })
    }

    /// Exception table entries for `n * k` weights.
    pub const fn capacity(n: u64, k: u64) -> u64 { (n * k).div_ceil(VALUES_PER_EXCEPTION) }

    fn region_bytes(n: u64, k: u64) -> [u64; 7] {
        let capacity = Self::capacity(n, k);
        [n * k, n * k / 2, n, (n + 1) * 4, capacity * 4, capacity * 2, STATUS_WORDS as u64 * 8]
    }

    /// Device bytes of an `[n, k]` head, for admission (unvalidated shapes give
    /// the same formula [`Self::new`] allocates for valid ones).
    pub fn bytes(n: u64, k: u64) -> u64 { Self::region_bytes(n, k).into_iter().map(align256).sum() }

    pub fn total_bytes(&self) -> usize { self.total }

    pub fn offset(&self, region: Region) -> usize { self.offsets[region as usize] }
}

/// The native pack's status words.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackStatus {
    /// Values outside their rows' windows.
    pub exceptions: u64,
    /// Rows failing verification (every row when the table overflowed).
    pub bad_rows: u64,
    /// The first failing row (`u64::MAX` if none).
    pub first_bad_row: u64,
    /// The most exceptions in any row.
    pub max_row_exceptions: u64,
}

impl PackStatus {
    /// From the status region's bytes (little-endian u64 words).
    pub fn from_bytes(bytes: &[u8; STATUS_WORDS * 8]) -> Self {
        let word = |i: usize| u64::from_le_bytes(bytes[i * 8..i * 8 + 8].try_into().expect("eight bytes"));
        Self { exceptions: word(0), bad_rows: word(1), first_bad_row: word(2), max_row_exceptions: word(3) }
    }

    /// Whether the pack fitted its table and decodes to its source, bit for bit.
    pub fn check(&self, capacity: usize) -> Result<(), PackedHeadError> {
        if self.exceptions > capacity as u64 {
            return Err(PackedHeadError::Capacity { exceptions: self.exceptions, capacity: capacity as u64 });
        }
        if self.bad_rows != 0 {
            return Err(PackedHeadError::Verification { rows: self.bad_rows, first: self.first_bad_row });
        }
        Ok(())
    }
}

// --- CPU reference ----------------------------------------------------------

/// A BF16 value's biased exponent.
pub const fn exponent(bits: u16) -> u32 { (bits as u32 >> 7) & 0xff }

/// A row's base exponent (the fullest window, ties to the lowest base) and its exceptions.
pub fn choose_base(row: &[u16]) -> (u8, usize) {
    let mut histogram = [0usize; 256];
    for &bits in row { histogram[exponent(bits) as usize] += 1; }
    let mut best = (0usize, 0u8);
    for e0 in 0..=MAX_BASE {
        let count = histogram[e0 as usize..e0 as usize + WINDOW as usize].iter().sum::<usize>();
        if count > best.0 { best = (count, e0); }
    }
    (best.1, row.len() - best.0)
}

/// Whether `bits` lies in base `e0`'s window.
pub const fn in_window(bits: u16, e0: u8) -> bool {
    let e = exponent(bits);
    e >= e0 as u32 && e < e0 as u32 + WINDOW
}

/// The value base `e0` codes for `bits`: itself inside the window, else clamped
/// into it with its sign (the window's smallest or largest magnitude).
pub const fn clamp_to_window(bits: u16, e0: u8) -> u16 {
    let (e, e0) = (exponent(bits), e0 as u32);
    let sign = bits & 0x8000;
    if e < e0 { sign | (e0 << 7) as u16 } else if e >= e0 + WINDOW { sign | ((e0 + WINDOW - 1) << 7) as u16 | 0x7f }
    else { bits }
}

/// The coarse byte and fine nibble of `bits` under base `e0` (clamped first).
pub const fn encode(bits: u16, e0: u8) -> (u8, u8) {
    let coded = clamp_to_window(bits, e0);
    let code = exponent(coded) - e0 as u32;
    ((((coded >> 8) & 0x80) as u32 | code << 3 | ((coded >> 4) & 7) as u32) as u8, (coded & 0xf) as u8)
}

/// The exact BF16 of a coarse byte and fine nibble at base `e0`.
pub const fn decode_exact(coarse: u8, fine: u8, e0: u8) -> u16 {
    let (byte, e0) = (coarse as u32, e0 as u32);
    (((byte & 0x80) << 8) | (((byte & 0x7f) << 4) + (e0 << 7)) | (fine as u32 & 0xf)) as u16
}

/// The drafters' BF16 of a coarse byte at base `e0`: mantissa bits 3..0 at 0b1000.
pub const fn decode_coarse(coarse: u8, e0: u8) -> u16 { decode_exact(coarse, 0b1000, e0) }

/// Where row `row`, column `col` of a `k`-wide head lives: its coarse byte, and
/// the fine plane's u32 word and nibble shift. Tiles of 16 rows x 32 columns,
/// 32 lanes each; lane l = 4 g + q of a tile holds per k16 step s the A registers
/// j = 0..3 of mma.m16n8k16: (row g, k 2q 2q+1), (row g+8, same), (row g, k 2q+8
/// 2q+9), (row g+8, same); value h of register j is coarse byte 8 s + 2 j + h
/// of the lane's 16 and nibble 4 j + 16 h of its fine word s.
pub const fn fragment_position(row: usize, col: usize, k: usize) -> (usize, usize, u32) {
    let (g, upper, kk) = (row & 7, (row >> 3) & 1, col & 31);
    let (step, half, within) = (kk >> 4, (kk >> 3) & 1, kk & 7);
    let (lane, h, j) = (g * 4 + (within >> 1), within & 1, half * 2 + upper);
    let index = ((row >> 4) * (k / 32) + (col >> 5)) * 32 + lane;
    (index * 16 + step * 8 + j * 2 + h, index * 2 + step, (4 * j + 16 * h) as u32)
}

/// A head packed on the CPU, byte for byte what the native pack writes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackedReference {
    pub n: usize,
    pub k: usize,
    pub coarse: Vec<u8>,
    pub fine: Vec<u32>,
    pub base: Vec<u8>,
    pub row_ptr: Vec<u32>,
    pub columns: Vec<u32>,
    pub values: Vec<u16>,
}

impl PackedReference {
    /// Packs the row-major BF16 head `w` [n, k] (any number of exceptions).
    pub fn pack(w: &[u16], n: usize, k: usize) -> Result<Self, PackedHeadError> {
        PackedHeadLayout::new(n, k)?;
        if w.len() != n * k { return Err(PackedHeadError::Shape { n, k }); }
        let mut packed = Self { n, k, coarse: vec![0; n * k], fine: vec![0; n * k / 8], base: vec![0; n],
            row_ptr: vec![0; n + 1], columns: Vec::new(), values: Vec::new() };
        for r in 0..n {
            let row = &w[r * k..(r + 1) * k];
            let (e0, _) = choose_base(row);
            packed.base[r] = e0;
            for (col, &bits) in row.iter().enumerate() {
                let (coarse, fine) = encode(bits, e0);
                let (byte, word, shift) = fragment_position(r, col, k);
                packed.coarse[byte] = coarse;
                packed.fine[word] |= u32::from(fine) << shift;
                if !in_window(bits, e0) {
                    packed.columns.push(col as u32);
                    packed.values.push(bits);
                }
            }
            packed.row_ptr[r + 1] = packed.columns.len() as u32;
        }
        Ok(packed)
    }

    /// The exact BF16 of row `row`, column `col`: the planes, or the table's entry.
    pub fn decode(&self, row: usize, col: usize) -> u16 {
        let entries = self.row_ptr[row] as usize..self.row_ptr[row + 1] as usize;
        if let Some(at) = self.columns[entries.clone()].iter().position(|&c| c as usize == col) {
            return self.values[entries.start + at];
        }
        let (byte, word, shift) = fragment_position(row, col, self.k);
        decode_exact(self.coarse[byte], ((self.fine[word] >> shift) & 0xf) as u8, self.base[row])
    }

    /// The drafters' BF16 of row `row`, column `col` (the coarse plane alone).
    pub fn decode_coarse(&self, row: usize, col: usize) -> u16 {
        decode_coarse(self.coarse[fragment_position(row, col, self.k).0], self.base[row])
    }

    /// FNV-1a over every array, little-endian: one number to compare packers by.
    pub fn digest(&self) -> u64 {
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        let mut eat = |bytes: &[u8]| for &b in bytes { hash = (hash ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3); };
        eat(&self.coarse);
        self.fine.iter().for_each(|w| eat(&w.to_le_bytes()));
        eat(&self.base);
        self.row_ptr.iter().for_each(|w| eat(&w.to_le_bytes()));
        self.columns.iter().for_each(|w| eat(&w.to_le_bytes()));
        self.values.iter().for_each(|w| eat(&w.to_le_bytes()));
        hash
    }
}

/// The [32, 256] golden input `native/tests/packed_head_selftest.cu` packs on the
/// GPU: random signs and mantissas, exponents 0..22 below a per-row top (uniform,
/// so windows tie), every fourth row inside 12 exponents, plus a zero, a
/// subnormal, an infinity and a NaN.
pub fn golden_input() -> Vec<u16> {
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut next = || {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        (state >> 33) as u32
    };
    let mut w = vec![0u16; 32 * 256];
    for r in 0..32 {
        for c in 0..256 {
            let v = next();
            let mut d = (v >> 8) % 23;
            if r % 4 == 0 { d %= 12; }
            let e = 120 + (r as u32 % 3) - d;
            w[r * 256 + c] = ((v & 1) << 15 | e << 7 | (v >> 1) & 0x7f) as u16;
        }
    }
    w[5 * 256 + 3] = 0x0000;
    w[6 * 256 + 7] = 0x0001;
    w[7 * 256 + 9] = 0x7f80;
    w[9 * 256 + 200] = 0xffc1;
    w
}

/// [`PackedReference::digest`] of [`golden_input`], which the native self-test
/// checks its GPU pack against.
pub const GOLDEN_DIGEST: u64 = 0xbcf7_e770_276c_4482;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_aligns_every_region_and_admits_v41_heads_exactly() {
        let full = PackedHeadLayout::new(129_280, 5120).unwrap();
        assert_eq!(full.capacity, 646_400);
        assert_eq!(full.total_bytes(), 997_395_712);
        assert_eq!(PackedHeadLayout::bytes(129_280, 5120), 997_395_712);
        let shard = PackedHeadLayout::new(64_640, 5120).unwrap();
        assert_eq!(shard.total_bytes(), 498_698_240);
        for layout in [full, shard] {
            let mut previous = None;
            for region in Region::ALL {
                let offset = layout.offset(region);
                assert_eq!(offset % 256, 0, "{region:?}");
                if let Some(p) = previous { assert!(offset > p, "{region:?}"); }
                previous = Some(offset);
            }
            assert_eq!(layout.offset(Region::Fine), layout.n * layout.k);
            assert_eq!(layout.offset(Region::Base), layout.n * layout.k * 3 / 2);
            assert_eq!(layout.total_bytes() - layout.offset(Region::Status), 256);
        }
        // 12 bits per weight plus under half a percent for bases, row pointers and the table.
        assert!(full.total_bytes() as f64 / (129_280.0 * 5120.0) < 1.5 * 1.005);
        for (n, k) in [(0, 5120), (8, 5120), (16, 64), (16, 0), (65_536, 65_536)] {
            assert_eq!(PackedHeadLayout::new(n, k), Err(PackedHeadError::Shape { n, k }));
        }
    }

    #[test]
    fn status_refuses_overflowing_tables_and_failed_verification() {
        let mut bytes = [0u8; 32];
        bytes[0] = 7;
        bytes[16..24].copy_from_slice(&u64::MAX.to_le_bytes());
        bytes[24] = 3;
        let status = PackStatus::from_bytes(&bytes);
        assert_eq!(status, PackStatus { exceptions: 7, bad_rows: 0, first_bad_row: u64::MAX, max_row_exceptions: 3 });
        assert_eq!(status.check(7), Ok(()));
        assert_eq!(status.check(6), Err(PackedHeadError::Capacity { exceptions: 7, capacity: 6 }));
        let failed = PackStatus { bad_rows: 2, first_bad_row: 11, ..status };
        assert_eq!(failed.check(7), Err(PackedHeadError::Verification { rows: 2, first: 11 }));
        assert!(failed.check(7).unwrap_err().to_string().contains("first: row 11"));
    }

    #[test]
    fn window_holds_the_most_values_and_ties_to_the_lowest_base() {
        let at = |e: u32| (e << 7) as u16;
        // 100..=115 hold four values, 116 one: [100, 115] beats [101, 116].
        let mut row: Vec<u16> = (100..=115).flat_map(|e| [at(e); 4]).collect();
        row.push(at(116));
        assert_eq!(choose_base(&row), (100, 1));
        // Every window from 85 to 100 holds all of 100..=100: the lowest wins.
        assert_eq!(choose_base(&[at(100); 8]), (85, 0));
        // Zeros and infinities: the base never passes 240.
        assert_eq!(choose_base(&[0x7f80; 4]), (MAX_BASE, 0));
        assert_eq!(choose_base(&[0, 0, at(30)]), (0, 1));
        // An outlier on each side of a 16-exponent bulk.
        let mut row: Vec<u16> = (110..=125).flat_map(|e| [at(e); 3]).collect();
        row.extend([at(60), at(126) | 0x8000]);
        assert_eq!(choose_base(&row), (110, 2));
    }

    #[test]
    fn every_bf16_codes_exactly_in_window_and_clamped_outside() {
        for e0 in [0u8, 1, 97, 110, 239, 240] {
            for bits in 0..=u16::MAX {
                let (coarse, fine) = encode(bits, e0);
                assert!(fine < 16);
                let coded = decode_exact(coarse, fine, e0);
                assert_eq!(coded, clamp_to_window(bits, e0), "{bits:#06x} at {e0}");
                if in_window(bits, e0) {
                    assert_eq!(coded, bits, "{bits:#06x} at {e0}");
                } else {
                    let e = exponent(coded);
                    assert!(e == u32::from(e0) || e == u32::from(e0) + 15);
                    assert_eq!(coded & 0x8000, bits & 0x8000);
                    let magnitude = |b: u16| f32::from_bits(u32::from(b & 0x7fff) << 16);
                    // Clamping moves toward the window: never past the value it replaces.
                    if exponent(bits) < u32::from(e0) { assert!(magnitude(coded) > magnitude(bits)); }
                    else { assert!(magnitude(coded) < magnitude(bits) || magnitude(bits).is_nan()); }
                }
                assert_eq!(decode_coarse(coarse, e0), (coded & !0xf) | 0b1000);
            }
        }
    }

    /// The GEMV's register decode (packed_head.cu `decode2`): a byte permute that
    /// replicates each coarse byte's sign above it, a 4-bit shift, the mask
    /// 0x87f087f0 and one add of base and nibbles, emulated here for every pair.
    #[test]
    fn register_decode_matches_the_scalar_decode() {
        // PTX prmt.b32 d, a, 0, selector: selector nibble i picks byte i of d from
        // {0, a} (index 0..7, a's bytes first); its top bit replicates that byte's sign.
        fn prmt(a: u32, selector: u32) -> u32 {
            (0..4).fold(0, |d, i| {
                let s = selector >> (4 * i) & 0xf;
                let byte = if s & 7 < 4 { a >> (8 * (s & 3)) & 0xff } else { 0 };
                let value = if s & 8 != 0 { if byte & 0x80 != 0 { 0xff } else { 0 } } else { byte };
                d | value << (8 * i)
            })
        }
        let decode2 = |word: u32, selector: u32, nibbles: u32, bias: u32|
            ((prmt(word, selector) << 4) & 0x87f0_87f0).wrapping_add(bias).wrapping_add(nibbles);
        for e0 in [0u8, 7, 113, 240] {
            let bias = (u32::from(e0) << 7) * 0x0001_0001;
            for b0 in 0..=255u32 {
                for b1 in (0..=255u32).step_by(7) {
                    let (n0, n1) = (b0 % 16, (b1 * 5 + 3) % 16);
                    let nibbles = n0 | n1 << 16;
                    let low = decode2(b0 | b1 << 8 | 0xa5 << 16 | 0x5a << 24, 0x9180, nibbles, bias);
                    let high = decode2(0x3c | 0xc3 << 8 | b0 << 16 | b1 << 24, 0xb3a2, nibbles, bias);
                    for pair in [low, high] {
                        assert_eq!(pair as u16, decode_exact(b0 as u8, n0 as u8, e0), "{b0} {e0}");
                        assert_eq!((pair >> 16) as u16, decode_exact(b1 as u8, n1 as u8, e0), "{b1} {e0}");
                    }
                    let coarse = decode2(b0 | b1 << 8, 0x9180, 0, bias + 0x0008_0008);
                    assert_eq!(coarse as u16, decode_coarse(b0 as u8, e0));
                    assert_eq!((coarse >> 16) as u16, decode_coarse(b1 as u8, e0));
                }
            }
        }
    }

    #[test]
    fn fragment_positions_cover_both_planes_once() {
        let (n, k) = (48, 384);
        let mut bytes = vec![false; n * k];
        let mut nibbles = vec![false; n * k];
        for row in 0..n {
            for col in 0..k {
                let (byte, word, shift) = fragment_position(row, col, k);
                assert!(!std::mem::replace(&mut bytes[byte], true), "{row} {col}");
                assert!(shift % 4 == 0 && shift < 32 && word < n * k / 8);
                assert!(!std::mem::replace(&mut nibbles[word * 8 + shift as usize / 4], true), "{row} {col}");
            }
        }
        assert!(bytes.iter().all(|&b| b) && nibbles.iter().all(|&b| b));
        // Lane 0 of the first tile: rows 0 and 8, k 0 1 then 8 9 of each step.
        let order: Vec<_> = [(0, 0), (0, 1), (8, 0), (8, 1), (0, 8), (0, 9), (8, 8), (8, 9), (0, 16), (8, 25)]
            .iter().map(|&(r, c)| fragment_position(r, c, k).0).collect();
        assert_eq!(order, [0, 1, 2, 3, 4, 5, 6, 7, 8, 15]);
        // Lane 5 (g 1, q 1): row 1, k 2.
        assert_eq!(fragment_position(1, 2, k), (5 * 16, 5 * 2, 0));
        assert_eq!(fragment_position(9, 11, k), (5 * 16 + 7, 5 * 2, 4 * 3 + 16));
        // The next 32 columns are the next tile; the next 16 rows skip a whole tile row.
        assert_eq!(fragment_position(0, 32, k).0, 32 * 16);
        assert_eq!(fragment_position(16, 0, k).0, k / 32 * 32 * 16);
    }

    #[test]
    fn reference_pack_decodes_its_source_bit_for_bit() {
        let w = golden_input();
        let packed = PackedReference::pack(&w, 32, 256).unwrap();
        for r in 0..32 {
            let row = &w[r * 256..(r + 1) * 256];
            assert_eq!(packed.base[r], choose_base(row).0);
            let entries = packed.row_ptr[r] as usize..packed.row_ptr[r + 1] as usize;
            assert_eq!(entries.len(), choose_base(row).1);
            assert!(packed.columns[entries.clone()].windows(2).all(|p| p[0] < p[1]));
            for col in 0..256 {
                assert_eq!(packed.decode(r, col), row[col], "row {r} col {col}");
                let coarse = decode_coarse(encode(row[col], packed.base[r]).0, packed.base[r]);
                assert_eq!(packed.decode_coarse(r, col), coarse);
            }
        }
        // The golden input's exceptions: in-window rows have none; the zero, subnormal,
        // infinity and NaN are all outside their windows.
        assert!((0..32).step_by(4).all(|r| packed.row_ptr[r] == packed.row_ptr[r + 1]));
        for (r, c) in [(5, 3), (6, 7), (7, 9), (9, 200)] {
            let entries = packed.row_ptr[r] as usize..packed.row_ptr[r + 1] as usize;
            assert!(packed.columns[entries].contains(&c), "{r} {c}");
        }
        assert_eq!(packed.digest(), GOLDEN_DIGEST, "{:#018x}", packed.digest());
    }

    #[test]
    fn reference_pack_rejects_bad_shapes() {
        assert!(PackedReference::pack(&[0; 16 * 128], 16, 128).is_ok());
        assert_eq!(PackedReference::pack(&[0; 16 * 127], 16, 127), Err(PackedHeadError::Shape { n: 16, k: 127 }));
        assert_eq!(PackedReference::pack(&[0; 10], 16, 128), Err(PackedHeadError::Shape { n: 16, k: 128 }));
    }
}
