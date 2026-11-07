use super::*;
use sha2::{Digest, Sha256};
use std::collections::HashSet;

/// SplitMix64: fixtures must be identical on every machine, so no floating-point library
/// calls (only IEEE-exact +, *, /, sqrt).
struct Mix(u64);

impl Mix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn shuffle(&mut self, w: &mut [u16]) {
        for i in (1..w.len()).rev() {
            w.swap(i, self.below(i as u64 + 1) as usize);
        }
    }
}

/// 85% exactly an E4M3 value times a power of two (3 mantissa bits), 15% ordinary BF16 at
/// the same exponents (the shape of the tool's self-test fixture).
fn fp8_born(n: usize, k: usize, seed: u64) -> Vec<u16> {
    fp8_born_over(n, k, seed, 16)
}

fn fp8_born_over(n: usize, k: usize, seed: u64, exponents: u64) -> Vec<u16> {
    let mut r = Mix(seed);
    (0..n * k)
        .map(|_| {
            let sign = (r.below(2) as u16) << 15;
            let exp = (112 + r.below(exponents) as u16) << 7;
            let mant = if r.unit() < 0.85 { (r.below(8) as u16) << 4 } else { r.below(128) as u16 };
            sign | exp | mant
        })
        .collect()
}

/// Distinct scattered patterns for ranks (an odd multiplier is a bijection on u16).
fn pattern(rank: usize) -> u16 {
    ((rank as u32 + 1).wrapping_mul(40503) & 0xffff) as u16
}

/// Zipf(1.5) over `ranks` ranks: steep enough that the unlimited Huffman code exceeds 16
/// bits and package-merge takes over.
fn zipf(n: usize, k: usize, seed: u64, ranks: usize) -> Vec<u16> {
    let mut cdf = Vec::with_capacity(ranks);
    let mut total = 0.0f64;
    for r in 1..=ranks {
        let x = r as f64;
        total += 1.0 / (x * x.sqrt());
        cdf.push(total);
    }
    let mut g = Mix(seed);
    (0..n * k)
        .map(|_| {
            let u = g.unit() * total;
            pattern(cdf.partition_point(|&c| c < u).min(ranks - 1))
        })
        .collect()
}

/// Every pattern of 16 exponents x 128 mantissas x 2 signs equally often, shuffled.
fn uniform4096(n: usize, k: usize, seed: u64) -> Vec<u16> {
    let mut w: Vec<u16> = (0..n * k)
        .map(|i| {
            let p = i % 4096;
            ((p >> 11) as u16) << 15 | (112 + ((p >> 7) & 15) as u16) << 7 | (p & 127) as u16
        })
        .collect();
    Mix(seed).shuffle(&mut w);
    w
}

/// 1000 frequent patterns and 4100 rare ones (counts 20..26): the two-class code must
/// promote the 4 most frequent rare patterns (ties to the lowest value) to keep 4096.
fn promote(n: usize, k: usize, seed: u64) -> Vec<u16> {
    let mut w = Vec::with_capacity(n * k);
    for i in 0..4100 {
        w.extend(std::iter::repeat(pattern(1000 + i)).take(20 + i % 7));
    }
    let rest = n * k - w.len();
    w.extend((0..rest).map(|j| pattern(j % 1000)));
    Mix(seed).shuffle(&mut w);
    w
}

fn sha(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

const CODES: [CodeKind; 2] = [CodeKind::Huffman, V11_CODE];
/// A 12-bit primary: 12 + 12 raw bits needs a rare mass near 1/16 (a 4-bit ESC), so only
/// some fixtures take it.
const WIDE: CodeKind = CodeKind::TwoClass { pbits: 12, rare_bits: 12 };

fn layouts() -> [Kx7Layout; 5] {
    [
        V11_LAYOUT,
        Kx7Layout::lane16(4).interleaved(false),
        Kx7Layout::lane16(4),
        Kx7Layout::lane16(2).with_align(16),
        Kx7Layout::frag32(2),
    ]
}

fn round_trip(w: &[u16], n: usize, k: usize, layout: Kx7Layout, kind: CodeKind) -> Kx7Pack {
    let pack = encode_matrix(w, n, k, layout, kind).unwrap_or_else(|e| panic!("{layout:?} {kind:?}: {e}"));
    verify_matrix(&pack, w).unwrap();
    assert_eq!(decode(&pack).unwrap(), w);
    let again = Kx7Pack::from_parts(
        pack.header().clone(),
        pack.code().clone(),
        pack.offsets().to_vec(),
        pack.lengths().to_vec(),
        pack.streams().to_vec(),
    )
    .unwrap();
    assert_eq!(again, pack);
    pack
}

#[test]
fn kx7_default_slices() {
    assert_eq!(default_slices(8192, 4096), 2);
    assert_eq!(default_slices(4096, 8192), 4);
    assert_eq!(default_slices(24576, 4096), 1);
    assert_eq!(default_slices(16, 16), 1);
    assert_eq!(default_slices(64, 4096), 8);
    assert_eq!(default_slices(64, 32), 2);
}

#[test]
fn kx7_layouts_are_permutations_with_exact_bands() {
    let cases = [
        (Kx7Layout::frag32(1), 32, 128),
        (Kx7Layout::frag32(2), 48, 128),
        (Kx7Layout::frag32(4), 16, 256),
        (Kx7Layout::lane16(1), 3, 256),
        (Kx7Layout::lane16(2), 5, 256),
        (Kx7Layout::lane16(4), 2, 512),
    ];
    for (layout, n, k) in cases {
        layout.check(n, k).unwrap();
        let mut seen = vec![false; n * k];
        let mut pos = Vec::new();
        for s in 0..layout.streams(n, k) {
            layout.positions(n, k, s, &mut pos);
            assert_eq!(pos.len(), layout.symbols_per_stream(k));
            for &(r, c) in &pos {
                assert!(!std::mem::replace(&mut seen[r * k + c], true), "{layout:?} covers ({r}, {c}) twice");
            }
        }
        assert!(seen.iter().all(|&x| x), "{layout:?} misses positions");
        let mut covered = HashSet::new();
        let mut band = Vec::new();
        for b in 0..n / layout.band_rows() {
            layout.band_streams(n, b, &mut band);
            for &s in &band {
                assert!(covered.insert(s));
                layout.positions(n, k, s, &mut pos);
                assert!(pos.iter().all(|&(r, _)| r / layout.band_rows() == b));
            }
        }
        assert_eq!(covered.len(), layout.streams(n, k));
    }
}

#[test]
fn kx7_frag32_streams_hold_mma_a_fragments() {
    let (n, k) = (32, 64);
    let layout = Kx7Layout::frag32(2);
    // Stream (s = 1, t = 1, l = 5): g = 1, c = 2, rows 17 and 25, slice 1 = steps 2..4.
    let mut pos = Vec::new();
    layout.positions(n, k, (2 + 1) * 32 + 5, &mut pos);
    let mut want = Vec::new();
    for j in 2..4 {
        let k0 = 16 * j + 2;
        want.extend([(17, k0), (17, k0 + 1), (25, k0), (25, k0 + 1)]);
        want.extend([(17, k0 + 8), (17, k0 + 9), (25, k0 + 8), (25, k0 + 9)]);
    }
    assert_eq!(pos, want);
}

#[test]
fn kx7_lane16_follows_kdaexact_ksplit_permutation() {
    // The tool's lane16_permutation(K, ksplit): row-major streams of K / (4 S) inputs,
    // stream s * 4 + t of a row covering s * K / S + 64 c + 16 t + i.
    for (k, s) in [(256, 1), (1024, 4), (4096, 4), (512, 2)] {
        let ks = k / s;
        let mut perm = Vec::new();
        for sl in 0..s {
            for t in 0..4 {
                for c in 0..ks / 64 {
                    perm.extend(sl * ks + 64 * c + 16 * t..sl * ks + 64 * c + 16 * t + 16);
                }
            }
        }
        let layout = Kx7Layout::lane16(s);
        let (n, spr) = (3, 4 * s);
        let mut pos = Vec::new();
        for stream in 0..layout.streams(n, k) {
            layout.positions(n, k, stream, &mut pos);
            let c = layout.symbols_per_stream(k);
            let w = stream % spr;
            let want: Vec<_> = perm[w * c..(w + 1) * c].iter().map(|&col| (stream / spr, col)).collect();
            assert_eq!(pos, want);
        }
    }
}

#[test]
fn kx7_interleave_tiles_every_byte_once() {
    // 16 rows, S = 2: 4 groups of 32 lanes; random lengths 1..200 bytes.
    let (n, k) = (16, 256);
    let mut g = Mix(5);
    for ragged in [true, false] {
        let layout = Kx7Layout::lane16(2).interleaved(ragged);
        let lengths: Vec<u16> = (0..layout.streams(n, k)).map(|_| 1 + g.below(200) as u16).collect();
        let offsets = place(&layout, n, k, &lengths).unwrap();
        let total = *offsets.last().unwrap() as usize;
        let mut owner = vec![usize::MAX; total];
        for s in 0..lengths.len() {
            let e = extent(&layout, &offsets, &lengths, s);
            assert_eq!(e.first, offsets[s] as usize);
            for q in 0..e.span / QUAD {
                for b in e.quad(q)..e.quad(q) + QUAD {
                    assert_eq!(std::mem::replace(&mut owner[b], s), usize::MAX, "byte {b} twice (ragged {ragged})");
                }
            }
        }
        assert!(owner.iter().all(|&o| o != usize::MAX), "gaps (ragged {ragged})");
        // Group 0 (block 0, slice 0): lane l = 4 (row % 8) + t at l * 16, stream row * 8 + t.
        for l in 0..32 {
            assert_eq!(offsets[(l / 4) * 8 + l % 4], (l * QUAD) as u32);
        }
        let nq0: Vec<usize> = (0..32).map(|l| quads(lengths[(l / 4) * 8 + l % 4])).collect();
        let (lo, hi) = (*nq0.iter().min().unwrap(), *nq0.iter().max().unwrap());
        let group1 = if ragged {
            lo * QUAD_STRIDE + nq0.iter().map(|q| (q - lo) * QUAD).sum::<usize>()
        } else {
            hi * QUAD_STRIDE
        };
        assert_eq!(offsets[4], group1 as u32, "group 1 (block 0, slice 1) base");
    }
}

#[test]
fn kx7_round_trips_every_code_and_placement() {
    let fixtures: Vec<(&str, usize, usize, Vec<u16>)> = vec![
        ("fp8_born", 64, 4096, fp8_born(64, 4096, 100)),
        ("zipf", 64, 4096, zipf(64, 4096, 101, 8192)),
        ("uniform4096", 32, 4096, uniform4096(32, 4096, 102)),
        ("one", 32, 1024, vec![0x3c00; 32 * 1024]),
    ];
    for (name, n, k, w) in &fixtures {
        for kind in CODES {
            for layout in layouts() {
                let pack = round_trip(w, *n, *k, layout, kind);
                match (*name, kind) {
                    ("uniform4096", CodeKind::TwoClass { pbits: 11, .. }) => {
                        // Every pattern is rare: the code is ESC alone, one bit.
                        let c = pack.code();
                        assert_eq!((c.n_frequent(), c.n_rare(), c.esc_len(), c.esc_index()), (0, 4096, 1, Some(0)));
                    }
                    ("one", CodeKind::TwoClass { .. }) => {
                        assert_eq!((pack.code().n_rare(), pack.code().maxlen()), (0, 12))
                    }
                    ("zipf", CodeKind::Huffman) => assert_eq!(pack.code().maxlen(), 16),
                    _ => {}
                }
            }
        }
    }
    // Promotion: 4100 rare candidates, the 4 most frequent promoted.
    let w = promote(256, 4096, 103);
    round_trip(&w, 256, 4096, V11_LAYOUT, WIDE);
    round_trip(&uniform4096(32, 4096, 104), 32, 4096, V11_LAYOUT, WIDE);
    let pack = round_trip(&w, 256, 4096, V11_LAYOUT, V11_CODE);
    let c = pack.code();
    assert_eq!((c.n_rare(), c.n_frequent()), (4096, 1004));
    let promoted: Vec<u16> = (0..4100).filter(|i| i % 7 == 6).map(|i| pattern(1000 + i)).collect();
    let mut by_value = promoted.clone();
    by_value.sort_unstable();
    for p in &by_value[..4] {
        assert!(c.rare().binary_search(p).is_err(), "{p:#06x} should be promoted");
    }
    assert!(c.rare().binary_search(&by_value[4]).is_ok());
    // Every pattern, 4 times: every Huffman code is 16 bits, the u16 count wraps to 0, and the
    // 11-bit device table cannot hold the sub-tables (2048 + 2048 x 32 words), so encoding
    // refuses, as the kernels' table build does.
    let w: Vec<u16> = (0..256 * 1024).map(|i| (i as u32).wrapping_mul(40503) as u16).collect();
    let code = Kx7Code::build(&histogram(&Kx7Source::matrix(&w, 256, 1024).unwrap()), CodeKind::Huffman).unwrap();
    assert_eq!((code.n_sym(), code.table()[15]), (65536, 0));
    assert_eq!(Kx7Code::from_tables(CodeKind::Huffman, &code.table(), &[], None).unwrap(), code);
    assert!(matches!(
        encode_matrix(&w, 256, 1024, Kx7Layout::lane16(4), CodeKind::Huffman),
        Err(Kx7Error::Table { .. })
    ));
}

#[test]
fn kx7_fused_source_round_trips_under_one_code() {
    let k = 4096;
    let (q, kk, v) = (fp8_born(16, k, 21), fp8_born(32, k, 22), fp8_born_over(16, k, 23, 8));
    let src = Kx7Source::fused(&[(&q, 16), (&kk, 32), (&v, 16)], k).unwrap();
    let whole: Vec<u16> = [q.as_slice(), &kk, &v].concat();
    for (layout, kind) in [(V11_LAYOUT, V11_CODE), (Kx7Layout::frag32(2), CodeKind::Huffman)] {
        let pack = encode(&src, layout, kind).unwrap();
        verify(&pack, &src).unwrap();
        assert_eq!(decode(&pack).unwrap(), whole);
        assert_eq!(pack.header().part_ends, vec![16, 48, 64]);
        // The same rows as one matrix give the same files; only the part record differs.
        let single = encode_matrix(&whole, 64, k, layout, kind).unwrap();
        assert_eq!(
            (single.streams(), single.offsets(), single.lengths()),
            (pack.streams(), pack.offsets(), pack.lengths())
        );
        assert_eq!(single.table_bin(), pack.table_bin());
        assert!(matches!(verify(&single, &src), Err(Kx7Error::Source { .. })));
    }
}

#[test]
fn kx7_device_bytes_fit_their_bound() {
    // FP8-born over 4 exponents: about 7.2 bits a weight, as GLM 5.3 Flash's KDA tensors.
    let w = fp8_born_over(64, 4096, 31, 4);
    let pack = encode_verified(&w, 64, 4096, V11_LAYOUT, V11_CODE).unwrap();
    let bytes = Kx7DeviceBytes::of(&pack).unwrap();
    let bound = Kx7DeviceBytes::bound(64, 4096, V11_LAYOUT, V11_CODE).unwrap();
    assert_eq!(bytes.tables, 4 * pack.device_tables().unwrap().len() as u64);
    assert_eq!(bytes.tables, bound.tables);
    assert_eq!((bytes.offsets, bytes.lengths), (4 * (64 * 16 + 1), 2 * 64 * 16));
    assert_eq!((bound.offsets, bound.lengths), (bytes.offsets, bytes.lengths));
    assert_eq!(bytes.streams, pack.streams().len() as u64 + READ_AHEAD_BYTES as u64);
    assert!(bytes.within(&bound) && bytes.total(256) <= bound.total(256));
    // The GLM 5.3 Flash KDA shapes: q/k/v [8192, 4096], o [4096, 8192].
    let q = Kx7DeviceBytes::bound(8192, 4096, V11_LAYOUT, V11_CODE).unwrap();
    let o = Kx7DeviceBytes::bound(4096, 8192, V11_LAYOUT, V11_CODE).unwrap();
    assert_eq!((q.streams, q.offsets, q.lengths, q.tables), (33_554_448, 524_292, 262_144, 16_384));
    assert_eq!((o.streams, o.offsets, o.lengths, o.tables), (33_554_448, 262_148, 131_072, 16_384));
    assert_eq!(device_table_words(V11_CODE), 4096);
    assert_eq!(decoded_bytes(8192, 4096), 64 << 20);
    // 16 exponents spread the 15% ordinary values over 4,096 patterns: above 8 bits a weight,
    // past the bound; random 16-bit patterns do not even take a two-class code.
    let wide = encode_verified(&fp8_born(64, 1024, 9), 64, 1024, V11_LAYOUT, V11_CODE).unwrap();
    assert!(!Kx7DeviceBytes::of(&wide).unwrap().within(&Kx7DeviceBytes::bound(64, 1024, V11_LAYOUT, V11_CODE).unwrap()));
    let mut r = Mix(9);
    let noise: Vec<u16> = (0..64 * 1024).map(|_| r.next() as u16).collect();
    assert!(encode_verified(&noise, 64, 1024, V11_LAYOUT, V11_CODE).is_err());
    assert!(Kx7DeviceBytes::bound(4, 1024, V11_LAYOUT, V11_CODE).is_err());
    // In parallel, in order, each verified.
    let v = fp8_born_over(16, 1024, 5, 4);
    let packs = encode_verified_all(&[(&w, 64, 4096), (&v, 16, 1024)], V11_LAYOUT, V11_CODE).unwrap();
    assert_eq!(packs.iter().map(|p| (p.header().n, p.header().k)).collect::<Vec<_>>(), [(64, 4096), (16, 1024)]);
    assert_eq!(packs[0], pack);
    let mut other = v.clone();
    other[37] ^= 1;
    let stale = encode_verified(&v, 16, 1024, V11_LAYOUT, V11_CODE).unwrap();
    assert!(matches!(verify_matrix(&stale, &other), Err(Kx7Error::Mismatch { row: 0, col: 37, .. })));
}

#[test]
fn kx7_encoding_is_deterministic() {
    let w = fp8_born(64, 4096, 5);
    let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    for (layout, kind) in [(V11_LAYOUT, V11_CODE), (Kx7Layout::frag32(2), CodeKind::Huffman)] {
        let a = encode_matrix(&w, 64, 4096, layout, kind).unwrap();
        let b = encode_matrix(&w, 64, 4096, layout, kind).unwrap();
        let c = pool.install(|| encode_matrix(&w, 64, 4096, layout, kind).unwrap());
        assert_eq!(a, b);
        assert_eq!(a, c);
        assert_eq!(a.device_tables().unwrap(), c.device_tables().unwrap());
    }
}

#[test]
fn kx7_device_tables_decode_every_codeword() {
    for w in [fp8_born(64, 4096, 9), zipf(64, 4096, 10, 8192), uniform4096(16, 4096, 2), promote(256, 4096, 3)] {
        let hist = histogram(&Kx7Source::matrix(&w, w.len() / 4096, 4096).unwrap());
        let wide = w.len() == 256 * 4096 || w.len() == 16 * 4096;
        for kind in CODES.into_iter().chain(wide.then_some(WIDE)) {
            let code = Kx7Code::build(&hist, kind).unwrap();
            let pbits = match kind {
                CodeKind::TwoClass { pbits, .. } => pbits,
                CodeKind::Huffman => DEVICE_PBITS,
            };
            let dec = DeviceDecoder::new(&code, pbits).unwrap();
            let words = dec.words();
            match kind {
                CodeKind::TwoClass { pbits, rare_bits } => {
                    assert_eq!(words.len(), (1 << pbits) + (1 << (rare_bits - 1)));
                    if let Some(e) = code.esc_index() {
                        assert_eq!(code.table()[TABLE_COUNTS + e], ESC_SLOT);
                    }
                }
                CodeKind::Huffman => {
                    assert!(words[..1 << pbits].iter().all(|&e| e & 0xff <= MAXLEN || e & 0xff == ENTRY_NODE))
                }
            }
            let mut g = Mix(1);
            for p in 0..=u16::MAX {
                let cw = code.codewords()[p as usize];
                if cw == 0 {
                    continue;
                }
                let (c, l) = (cw >> 8, cw & 0xff);
                // The codeword left-aligned in a 32-bit word, random bits after it.
                let word = c << (32 - l) | (g.next() as u32 >> l);
                let bytes = word.to_be_bytes();
                let mut bits = LaneBits::new(&bytes);
                assert!(bits.refill());
                assert_eq!(dec.next(&mut bits), Some(p), "{kind:?} pattern {p:#06x}");
                assert_eq!(bits.consumed, l as u64);
            }
        }
    }
}

/// Expected values from the KX7 reference tool, `kdaexact.py`. Produced by running the
/// ignored test `kx7_dump_reference_fixtures` (raw little-endian u16 files), then, with
/// python3 -I and numpy, for each fixture `a`: `f = np.bincount(a.ravel(), minlength=65536)`,
/// `length_limited_lengths(f, 16)` (`lengths`: one hex digit, length - 1, per used pattern in
/// ascending order; `unlimited_max`: `length_limited_lengths(f, 64).max()`), and for each
/// pack entry `run(a, 0, 16, layout='lane16', ksplit=4, pack=p, ...)` with the entry's code
/// and interleave arguments, then `dump_pack(d, a, r, p, source=False)`: the SHA-256 of each
/// .bin file and the header fields listed.
const REFERENCE: &str = include_str!("../fixtures/kx7_reference.json");

fn reference_fixtures() -> Vec<(&'static str, usize, usize, Vec<u16>)> {
    vec![
        ("fp8_born", 64, 4096, fp8_born(64, 4096, 0x4b58_3701)),
        ("fp8_narrow", 64, 4096, fp8_born_over(64, 4096, 0x4b58_3703, 8)),
        ("zipf", 64, 4096, zipf(64, 4096, 0x4b58_3702, 8192)),
        ("promote", 256, 4096, promote(256, 4096, 0x4b58_3704)),
    ]
}

fn reference_packs() -> [(&'static str, Kx7Layout, CodeKind); 4] {
    [
        ("lane16_ksplit4", Kx7Layout::lane16(4), CodeKind::Huffman),
        ("v11", V11_LAYOUT, V11_CODE),
        ("v11_padded", Kx7Layout::lane16(4).interleaved(false), V11_CODE),
        ("huffman_ragged", Kx7Layout::lane16(4).interleaved(true), CodeKind::Huffman),
    ]
}

/// The file set of a pack as the tool names it.
fn files(pack: &Kx7Pack) -> Vec<(&'static str, Vec<u8>)> {
    let mut f = vec![
        ("table", pack.table_bin()),
        ("streams", pack.streams().to_vec()),
        ("offsets", pack.offsets_bin()),
        ("lengths", pack.lengths_bin()),
    ];
    if let Some(rare) = pack.rare_bin() {
        f.push(("rare", rare));
    }
    f
}

/// Header fields that describe the pack (not histogram statistics or timings).
const HEADER_FIELDS: [&str; 30] = [
    "shape",
    "n",
    "chunk",
    "layout",
    "ksplit",
    "maxlen",
    "code",
    "distinct_patterns",
    "table_bytes",
    "pbits",
    "rare_bits",
    "esc_len",
    "n_rare",
    "interleave_rows",
    "quad_stride",
    "ragged",
    "padding_bytes",
    "padding_bpw",
    "stream_bytes",
    "chunks",
    "stream_bpw",
    "index_bpw",
    "table_bpw",
    "total_bpw",
    "bytes_bf16",
    "bytes_kx7",
    "esc_index",
    "n_sym",
    "streams_per_row",
    "symbols_per_stream",
];

fn header_mismatches(mine: &Value, theirs: &Value) -> Vec<String> {
    let mut out: Vec<String> = HEADER_FIELDS
        .iter()
        .filter(|&&f| !theirs[f].is_null() && mine[f] != theirs[f])
        .map(|f| format!("{f}: {} vs {}", mine[*f], theirs[*f]))
        .collect();
    let rows = |j: &Value| -> Vec<Value> {
        j["components"].as_array().into_iter().flatten().map(|c| c["rows"].clone()).collect()
    };
    if !theirs["components"].as_array().map_or(true, |c| c.is_empty()) && rows(mine) != rows(theirs) {
        out.push(format!("components: {:?} vs {:?}", rows(mine), rows(theirs)));
    }
    out
}

#[test]
fn kx7_code_lengths_and_packs_match_kdaexact() {
    let reference: Value = serde_json::from_str(REFERENCE).unwrap();
    for (name, n, k, w) in reference_fixtures() {
        let r = &reference["fixtures"][name];
        assert_eq!((r["n"].as_u64(), r["k"].as_u64()), (Some(n as u64), Some(k as u64)), "{name}");
        let hist = histogram(&Kx7Source::matrix(&w, n, k).unwrap());
        let lengths = code_lengths(&hist).unwrap();
        let got: String =
            lengths.iter().filter(|&&l| l > 0).map(|&l| char::from_digit(l as u32 - 1, 16).unwrap()).collect();
        assert_eq!(got, r["lengths"].as_str().unwrap(), "{name}: code lengths differ from the tool's");
        let freqs: Vec<u64> = hist.iter().copied().filter(|&f| f > 0).collect();
        let unlimited = huffman_depths(&freqs).into_iter().max().unwrap();
        assert_eq!(unlimited as u64, r["unlimited_max"].as_u64().unwrap(), "{name}");
        for (entry, layout, kind) in reference_packs() {
            let d = &r[entry];
            let pack = encode_matrix(&w, n, k, layout, kind).unwrap();
            for (file, bytes) in files(&pack) {
                assert_eq!(
                    sha(&bytes),
                    d[format!("{file}_sha256")].as_str().unwrap_or("absent"),
                    "{name} {entry}: {file}.bin"
                );
            }
            assert!(d["rare_sha256"].is_null() || pack.rare_bin().is_some(), "{name} {entry}: rare.bin missing");
            let mine: Value = serde_json::from_str(&pack.header_json(&Kx7Labels::default(), false)).unwrap();
            assert_eq!(header_mismatches(&mine, &d["header"]), Vec::<String>::new(), "{name} {entry}");
        }
    }
    // Coverage: fp8_narrow takes the unlimited Huffman code, the others package-merge; the
    // promote fixture promotes.
    let r = &reference["fixtures"];
    assert!(r["fp8_narrow"]["unlimited_max"].as_u64().unwrap() <= 16);
    assert!(r["fp8_born"]["unlimited_max"].as_u64().unwrap() > 16 && r["zipf"]["unlimited_max"].as_u64().unwrap() > 16);
    assert_eq!(r["promote"]["v11"]["header"]["n_rare"].as_u64(), Some(4096));
}

#[test]
#[ignore = "writes the reference fixtures to CUTEAFD_KX7_FIXTURE_DIR"]
fn kx7_dump_reference_fixtures() {
    let dir = std::env::var("CUTEAFD_KX7_FIXTURE_DIR").expect("set CUTEAFD_KX7_FIXTURE_DIR");
    for (name, n, k, w) in reference_fixtures() {
        std::fs::write(format!("{dir}/{name}.{n}x{k}.u16"), le16(&w)).unwrap();
    }
}

#[test]
fn kx7_dump_and_load_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let (q, kk) = (fp8_born(16, 1024, 61), zipf(16, 1024, 62, 2048));
    let src = Kx7Source::fused(&[(&q, 16), (&kk, 16)], 1024).unwrap();
    let labels = Kx7Labels { label: "q + k".into(), components: vec!["q".into(), "k".into()] };
    for (i, (layout, kind)) in
        [(V11_LAYOUT, V11_CODE), (Kx7Layout::frag32(2), CodeKind::Huffman), (Kx7Layout::lane16(2), CodeKind::Huffman)]
            .into_iter()
            .enumerate()
    {
        let path = dir.path().join(format!("p{i}"));
        let pack = encode(&src, layout, kind).unwrap();
        pack.dump(&path, &labels, Some(&src)).unwrap();
        let back = Kx7Pack::load(&path).unwrap();
        assert_eq!(back, pack);
        let header: Value = serde_json::from_slice(&std::fs::read(path.join("header.json")).unwrap()).unwrap();
        assert_eq!(header["components"][1]["name"], "k");
        assert_eq!(header["files"]["rare"].is_null(), kind == CodeKind::Huffman);
        let source = std::fs::read(path.join("source.bf16.bin")).unwrap();
        assert_eq!(source, le16(&[q.as_slice(), &kk].concat()));
        assert_eq!(std::fs::read(path.join("rare.bin")).is_ok(), kind != CodeKind::Huffman);
    }
    // Refusals on load: a header that disagrees with its files, a missing file, a bad layout.
    let path = dir.path().join("p0");
    let text = std::fs::read_to_string(path.join("header.json")).unwrap();
    let edit = |from: &str, to: &str| {
        assert!(text.contains(from), "{from}");
        std::fs::write(path.join("header.json"), text.replacen(from, to, 1)).unwrap();
        Kx7Pack::load(&path)
    };
    let sym_line = text
        .lines()
        .find(|l| l.trim_start().starts_with("\"n_sym\""))
        .unwrap()
        .trim()
        .trim_end_matches(',')
        .to_string();
    assert!(matches!(edit(&sym_line, "\"n_sym\": 1"), Err(Kx7Error::Header { field: "n_sym", .. })));
    assert!(matches!(
        edit("\"layout\": \"lane16\"", "\"layout\": \"lane8\""),
        Err(Kx7Error::Header { field: "layout", .. })
    ));
    assert!(matches!(edit("\"ragged\": true", "\"ragged\": false"), Err(Kx7Error::Index { .. })));
    assert!(matches!(edit("\"code\": \"twoclass\"", "\"code\": \"huffman\""), Err(Kx7Error::Table { .. })));
    std::fs::write(path.join("header.json"), &text).unwrap();
    std::fs::remove_file(path.join("lengths.bin")).unwrap();
    assert!(matches!(Kx7Pack::load(&path), Err(Kx7Error::Io { .. })));
}

#[test]
fn kx7_refuses_bad_shapes() {
    let w = vec![0u16; 64 * 256];
    let shape = |r: Result<Kx7Pack>| matches!(r, Err(Kx7Error::Shape { .. }));
    let h = CodeKind::Huffman;
    assert!(shape(encode_matrix(&w[..8 * 256], 8, 256, Kx7Layout::frag32(1), h)));
    assert!(shape(encode_matrix(&w[..64 * 40], 64, 40, Kx7Layout::frag32(1), h)));
    assert!(shape(encode_matrix(&w, 64, 256, Kx7Layout::frag32(32), h)));
    assert!(shape(encode_matrix(&w, 64, 256, Kx7Layout::frag32(0), h)));
    assert!(shape(encode_matrix(&w, 64, 256, Kx7Layout::lane16(8), h)));
    assert!(shape(encode_matrix(&w, 64, 256, Kx7Layout::lane16(1).with_align(3), h)));
    assert!(shape(encode_matrix(&w, 64, 256, Kx7Layout::frag32(1).interleaved(true), h)));
    assert!(shape(encode_matrix(&w[..12 * 256], 12, 256, Kx7Layout::lane16(1).interleaved(true), h)));
    assert!(matches!(encode_matrix(&w, 64, 255, Kx7Layout::frag32(1), h), Err(Kx7Error::Source { .. })));
    assert!(matches!(Kx7Source::fused(&[], 256), Err(Kx7Error::Source { .. })));
    assert!(matches!(Kx7Source::fused(&[(&w[..256], 1); 9], 256), Err(Kx7Error::Source { .. })));
    assert!(matches!(Kx7Source::fused(&[(&w[..256], 1), (&w[..0], 0)], 256), Err(Kx7Error::Source { .. })));
    let src = Kx7Source::matrix(&w, 64, 256).unwrap();
    let mut cw = vec![0u32; PATTERNS];
    assert!(matches!(pack_streams(&src, Kx7Layout::lane16(1), &cw), Err(Kx7Error::Uncoded { pattern: 0, .. })));
    cw.truncate(10);
    assert!(matches!(pack_streams(&src, Kx7Layout::lane16(1), &cw), Err(Kx7Error::Code { .. })));
}

#[test]
fn kx7_refuses_bad_codes() {
    let hist = histogram(&Kx7Source::matrix(&fp8_born(64, 1024, 71), 64, 1024).unwrap());
    let code = |kind| Kx7Code::build(&hist, kind);
    for (pbits, rare_bits) in [(0, 12), (17, 12), (11, 0), (11, 16)] {
        assert!(
            matches!(code(CodeKind::TwoClass { pbits, rare_bits }), Err(Kx7Error::Code { .. })),
            "{pbits} {rare_bits}"
        );
    }
    assert!(matches!(Kx7Code::build(&[0; 10], CodeKind::Huffman), Err(Kx7Error::Code { .. })));
    assert!(matches!(Kx7Code::build(&vec![0; PATTERNS], V11_CODE), Err(Kx7Error::Code { .. })));
    // Most of the alphabet: far more than 2^11 frequent symbols after promotion.
    let mut g = Mix(3);
    let random: Vec<u16> = (0..64 * 2048).map(|_| g.next() as u16).collect();
    let hist = histogram(&Kx7Source::matrix(&random, 64, 2048).unwrap());
    assert!(matches!(Kx7Code::build(&hist, V11_CODE), Err(Kx7Error::Code { .. })));
    assert!(Kx7Code::build(&hist, CodeKind::Huffman).is_ok());
    // A rare mass of 0.4%: ESC takes 8 bits, 8 + 12 > 16.
    let mut w: Vec<u16> = (0..262144).map(|i| pattern(i % 200)).collect();
    (0..1000).for_each(|j| w[j * 262] = pattern(1000 + j));
    let hist = histogram(&Kx7Source::matrix(&w, 64, 4096).unwrap());
    match Kx7Code::build(&hist, V11_CODE) {
        Err(Kx7Error::Code { reason }) => assert!(reason.contains("exceeds 16"), "{reason}"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn kx7_refuses_bad_tables() {
    let w = fp8_born(32, 1024, 81);
    let hist = histogram(&Kx7Source::matrix(&w, 32, 1024).unwrap());
    let huff = Kx7Code::build(&hist, CodeKind::Huffman).unwrap();
    let two = Kx7Code::build(&hist, V11_CODE).unwrap();
    let table = |t: &[u16]| Kx7Code::from_tables(CodeKind::Huffman, t, &[], None);
    let good = huff.table();
    assert_eq!(table(&good).unwrap(), huff);
    assert!(matches!(table(&good[..10]), Err(Kx7Error::Truncated { what: "code table", .. })));
    assert!(matches!(table(&good[..good.len() - 1]), Err(Kx7Error::Table { .. })));
    let first_len = (0..16).find(|&l| good[l] >= 2).unwrap();
    let first_sym = TABLE_COUNTS + good[..first_len].iter().map(|&c| c as usize).sum::<usize>();
    let mut swapped = good.clone();
    swapped.swap(first_sym, first_sym + 1);
    assert!(matches!(table(&swapped), Err(Kx7Error::Table { .. })));
    let mut kraft = good.clone();
    let (long, short) = ((0..16).rev().find(|&l| kraft[l] > 0).unwrap(), (0..16).find(|&l| kraft[l] > 0).unwrap());
    kraft[long] -= 1;
    kraft[short] += 1;
    assert!(matches!(table(&kraft), Err(Kx7Error::Table { .. })));
    let mut one = vec![0u16; 16];
    one[2] = 1;
    one.push(0x3c00);
    assert!(matches!(table(&one), Err(Kx7Error::Table { .. })));
    assert!(matches!(Kx7Code::from_tables(CodeKind::Huffman, &good, &[1], None), Err(Kx7Error::Table { .. })));
    // Two-class: the ESC slot, its place, the rare list.
    let (t, rare, esc) = (two.table(), two.rare().to_vec(), two.esc_index());
    let tc = |t: &[u16], rare: &[u16], esc: Option<usize>| Kx7Code::from_tables(V11_CODE, t, rare, esc);
    assert_eq!(tc(&t, &rare, esc).unwrap(), two);
    let e = esc.unwrap();
    assert!(matches!(tc(&t, &rare, None), Err(Kx7Error::Table { .. })));
    assert!(matches!(tc(&t, &rare, Some(e + 1)), Err(Kx7Error::Table { .. })));
    assert!(matches!(tc(&t, &[], esc), Err(Kx7Error::Table { .. })));
    let mut unsorted = rare.clone();
    unsorted.swap(0, 1);
    assert!(matches!(tc(&t, &unsorted, esc), Err(Kx7Error::Table { .. })));
    let mut overlap = rare.clone();
    overlap[0] = t[TABLE_COUNTS + if e == 0 { 1 } else { 0 }];
    overlap.sort_unstable();
    assert!(matches!(tc(&t, &overlap, esc), Err(Kx7Error::Table { .. })));
    let too_many: Vec<u16> = (0..4097u32).map(|i| (i * 13 + 7) as u16).collect();
    assert!(matches!(tc(&t, &too_many, esc), Err(Kx7Error::Table { .. })));
    // The ESC slot moved ahead of a symbol of its length breaks the canonical order.
    let esc_len = (two.esc_len()) as usize;
    let group_start = TABLE_COUNTS + t[..esc_len - 1].iter().map(|&c| c as usize).sum::<usize>();
    if t[esc_len - 1] >= 2 && group_start < TABLE_COUNTS + e {
        let mut moved = t.clone();
        moved.swap(group_start, TABLE_COUNTS + e);
        assert!(matches!(tc(&moved, &rare, Some(group_start - TABLE_COUNTS)), Err(Kx7Error::Table { .. })));
    }
    assert!(matches!(Kx7Code::from_tables(CodeKind::Huffman, &t, &rare, esc), Err(Kx7Error::Table { .. })));
    // Device tables: primary bits, and a two-class code wider than the table.
    assert!(matches!(huff.device_tables(0), Err(Kx7Error::Table { .. })));
    let promoted = promote(256, 4096, 3);
    let wide = Kx7Code::build(&histogram(&Kx7Source::matrix(&promoted, 256, 4096).unwrap()), WIDE).unwrap();
    assert!(wide.table()[11] > 0, "the fixture needs 12-bit codes");
    assert!(matches!(wide.device_tables(11), Err(Kx7Error::Table { .. })));
}

#[test]
fn kx7_refuses_bad_offsets_lengths_and_streams() {
    let w = fp8_born(32, 1024, 91);
    for (layout, kind) in [(V11_LAYOUT, V11_CODE), (Kx7Layout::frag32(2), CodeKind::Huffman)] {
        let p = encode_matrix(&w, 32, 1024, layout, kind).unwrap();
        let parts = |f: &dyn Fn(&mut Kx7Header, &mut Vec<u32>, &mut Vec<u16>, &mut Vec<u8>)| {
            let (mut h, mut o, mut l, mut s) =
                (p.header().clone(), p.offsets().to_vec(), p.lengths().to_vec(), p.streams().to_vec());
            f(&mut h, &mut o, &mut l, &mut s);
            Kx7Pack::from_parts(h, p.code().clone(), o, l, s)
        };
        assert!(matches!(
            parts(&|_, _, l, _| {
                l.pop();
            }),
            Err(Kx7Error::Truncated { what: "stream lengths", .. })
        ));
        assert!(matches!(
            parts(&|_, o, _, _| {
                o.pop();
            }),
            Err(Kx7Error::Truncated { what: "stream offsets", .. })
        ));
        assert!(matches!(parts(&|_, o, _, _| o[3] += 16), Err(Kx7Error::Index { entry: 3, .. })));
        assert!(matches!(parts(&|_, _, l, _| l[0] += 200), Err(Kx7Error::Index { .. })));
        assert!(matches!(
            parts(&|_, _, _, s| {
                s.pop();
            }),
            Err(Kx7Error::Truncated { what: "streams", .. })
        ));
        assert!(matches!(parts(&|h, _, _, _| h.part_ends = vec![16, 16, 32]), Err(Kx7Error::Header { .. })));
        assert!(matches!(parts(&|h, _, _, _| h.part_ends = vec![16]), Err(Kx7Error::Header { .. })));
        // A length one byte short that keeps the placement (same 16-byte padded size).
        let s = (0..p.lengths().len()).find(|&s| p.lengths()[s] % 16 >= 2).unwrap();
        let shorter = parts(&|_, _, l, _| l[s] -= 1).unwrap();
        assert_eq!(
            decode(&shorter),
            Err(Kx7Error::StreamLength {
                stream: s,
                expected: p.lengths()[s] as usize,
                actual: p.lengths()[s] as usize - 1
            })
        );
    }
}

#[test]
fn kx7_detects_corrupt_streams() {
    let w = fp8_born(32, 1024, 41);
    for (layout, kind) in
        [(V11_LAYOUT, V11_CODE), (Kx7Layout::frag32(2), CodeKind::Huffman), (Kx7Layout::lane16(2), CodeKind::Huffman)]
    {
        let p = encode_matrix(&w, 32, 1024, layout, kind).unwrap();
        let rebuilt = |streams: Vec<u8>| {
            Kx7Pack::from_parts(
                p.header().clone(),
                p.code().clone(),
                p.offsets().to_vec(),
                p.lengths().to_vec(),
                streams,
            )
            .unwrap()
        };
        // Nonzero padding after a stream's last symbol (a stream whose bits end mid-byte).
        let s = (0..p.lengths().len())
            .find(|&s| {
                let mut pos = Vec::new();
                layout.positions(32, 1024, s, &mut pos);
                pos.iter().map(|&(r, c)| p.code().length(w[r * 1024 + c])).sum::<u32>() % 8 != 0
            })
            .expect("a stream ending mid-byte");
        let e = extent(&layout, p.offsets(), p.lengths(), s);
        let last = p.lengths()[s] as usize - 1;
        let at = e.quad(last / QUAD) + last % QUAD;
        let mut streams = p.streams().to_vec();
        streams[at] |= 1;
        assert_eq!(decode(&rebuilt(streams)), Err(Kx7Error::Padding { stream: s }), "{layout:?}");
        // Flipped bits inside streams: never a silent wrong answer.
        let mut g = Mix(77);
        for _ in 0..200 {
            let mut streams = p.streams().to_vec();
            let at = g.below(streams.len() as u64) as usize;
            streams[at] ^= 1 << g.below(8);
            match verify_matrix(&rebuilt(streams), &w) {
                Err(
                    Kx7Error::Mismatch { .. }
                    | Kx7Error::StreamLength { .. }
                    | Kx7Error::InvalidCode { .. }
                    | Kx7Error::Padding { .. }
                    | Kx7Error::Overrun { .. },
                ) => {}
                other => panic!("{layout:?}: flip at byte {at}: {other:?}"),
            }
        }
        // A mismatch names its first position.
        let mut other = w.clone();
        other[5 * 1024 + 77] ^= 1;
        assert_eq!(
            verify_matrix(&p, &other),
            Err(Kx7Error::Mismatch { row: 5, col: 77, expected: other[5 * 1024 + 77], actual: w[5 * 1024 + 77] })
        );
    }
}

// ---------------------------------------------------------------- real tensors

struct Shards {
    tensors: std::collections::HashMap<String, (std::path::PathBuf, u64, Vec<usize>)>,
}

impl Shards {
    fn open(dir: &str) -> Self {
        use std::io::Read;
        let mut tensors = std::collections::HashMap::new();
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().and_then(|e| e.to_str()) != Some("safetensors") {
                continue;
            }
            let mut f = std::fs::File::open(&path).unwrap();
            let mut len = [0u8; 8];
            f.read_exact(&mut len).unwrap();
            let len = u64::from_le_bytes(len);
            let mut json = vec![0u8; len as usize];
            f.read_exact(&mut json).unwrap();
            let header: Value = serde_json::from_slice(&json).unwrap();
            for (name, meta) in header.as_object().unwrap() {
                if name == "__metadata__" || meta["dtype"] != "BF16" {
                    continue;
                }
                let shape = meta["shape"].as_array().unwrap().iter().map(|d| d.as_u64().unwrap() as usize).collect();
                let start = meta["data_offsets"][0].as_u64().unwrap();
                tensors.insert(name.clone(), (path.clone(), 8 + len + start, shape));
            }
        }
        Self { tensors }
    }

    fn load(&self, name: &str) -> (Vec<u16>, usize, usize) {
        use std::io::{Read, Seek, SeekFrom};
        let (path, offset, shape) = self.tensors.get(name).unwrap_or_else(|| panic!("{name} not in the shards"));
        let (n, k) = (shape[0], shape[1]);
        let mut f = std::fs::File::open(path).unwrap();
        f.seek(SeekFrom::Start(*offset)).unwrap();
        let mut bytes = vec![0u8; 2 * n * k];
        f.read_exact(&mut bytes).unwrap();
        (bytes.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect(), n, k)
    }
}

fn seconds<T>(f: impl FnOnce() -> T) -> (T, f64) {
    let t = std::time::Instant::now();
    let out = f();
    (out, t.elapsed().as_secs_f64())
}

/// Packs layer 10's fused q;k;v and o_proj as the tool's reference packs were made
/// (`--layout lane16 --ksplit 4 --interleave 8 --ragged`, `--code twoclass --pbits 11` or
/// Huffman) and compares every .bin file byte for byte and the header's structural fields;
/// also loads each reference pack, decodes it and compares it with the source.
/// CUTEAFD_KX7_PACKS names the directory holding `in4tc11R` and `out4tc11R` (required) and
/// `inR`, `outR` (compared when present). With CUTEAFD_KX7_LUTS, a directory of
/// `<pack>.lut.bin` files (the reference bench's `build_lut` words for each pack, u32
/// little-endian), the device tables are compared with them too.
#[test]
#[ignore = "reads GLM-5.3-Flash shards (CUTEAFD_KX7_SHARDS) and reference packs (CUTEAFD_KX7_PACKS)"]
fn kx7_matches_reference_packs() {
    let shards =
        std::env::var("CUTEAFD_KX7_SHARDS").expect("set CUTEAFD_KX7_SHARDS to a GLM-5.3-Flash shard directory");
    let packs = std::path::PathBuf::from(std::env::var("CUTEAFD_KX7_PACKS").expect("set CUTEAFD_KX7_PACKS"));
    let luts = std::env::var("CUTEAFD_KX7_LUTS").ok().map(std::path::PathBuf::from);
    let shards = Shards::open(&shards);
    let name = |p: &str| format!("model.language_model.layers.10.self_attn.{p}_proj.weight");
    let (q, k, v, o) =
        (shards.load(&name("q")), shards.load(&name("k")), shards.load(&name("v")), shards.load(&name("o")));
    let qkv = Kx7Source::fused(&[(q.0.as_slice(), q.1), (k.0.as_slice(), k.1), (v.0.as_slice(), v.1)], k.2).unwrap();
    let out = Kx7Source::matrix(&o.0, o.1, o.2).unwrap();
    let one = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    let threads = rayon::current_num_threads();
    let ragged = Kx7Layout::lane16(4).interleaved(true);
    let cases = [
        ("in4tc11R", &qkv, V11_LAYOUT, V11_CODE, true),
        ("out4tc11R", &out, V11_LAYOUT, V11_CODE, true),
        ("inR", &qkv, ragged, CodeKind::Huffman, false),
        ("outR", &out, ragged, CodeKind::Huffman, false),
    ];
    let mut matched = 0;
    for (dir_name, src, layout, kind, required) in cases {
        let dir = packs.join(dir_name);
        if !dir.join("header.json").exists() {
            assert!(!required, "{dir_name} missing under CUTEAFD_KX7_PACKS");
            println!("{dir_name}: absent, skipped");
            continue;
        }
        let weights = (src.n() * src.k()) as f64;
        let (pack1, enc1) = seconds(|| one.install(|| encode(src, layout, kind).unwrap()));
        let (pack, enc) = seconds(|| encode(src, layout, kind).unwrap());
        assert_eq!(pack1, pack);
        for (file, bytes) in files(&pack) {
            let theirs = std::fs::read(dir.join(format!("{file}.bin"))).unwrap();
            assert_eq!((bytes.len(), sha(&bytes)), (theirs.len(), sha(&theirs)), "{dir_name}: {file}.bin");
            println!("{dir_name} {file}.bin {} bytes sha256 {} identical", bytes.len(), sha(&bytes));
        }
        let mine: Value = serde_json::from_str(&pack.header_json(&Kx7Labels::default(), false)).unwrap();
        let theirs: Value = serde_json::from_slice(&std::fs::read(dir.join("header.json")).unwrap()).unwrap();
        assert_eq!(header_mismatches(&mine, &theirs), Vec::<String>::new(), "{dir_name} header.json");
        let tables = le32(&pack.device_tables().unwrap());
        if let Some(luts) = &luts {
            let theirs = std::fs::read(luts.join(format!("{dir_name}.lut.bin"))).unwrap();
            assert_eq!(sha(&tables), sha(&theirs), "{dir_name}: device tables differ from build_lut");
            println!(
                "{dir_name} device tables {} words sha256 {} identical to build_lut",
                tables.len() / 4,
                sha(&tables)
            );
        }
        let (loaded, load_s) = seconds(|| Kx7Pack::load(&dir).unwrap());
        assert_eq!(loaded, pack, "{dir_name}: the loaded reference pack differs from ours");
        let (decoded1, dec1) = seconds(|| one.install(|| decode(&loaded).unwrap()));
        let (_, dec) = seconds(|| decode(&loaded).unwrap());
        for (r, row) in src.rows.iter().enumerate() {
            assert_eq!(*row, &decoded1[r * src.k()..(r + 1) * src.k()], "{dir_name} row {r}");
        }
        let c = pack.code();
        let esc_mass = match kind {
            CodeKind::TwoClass { .. } => {
                let hist = histogram(src);
                c.rare().iter().map(|&p| hist[p as usize]).sum::<u64>() as f64 / weights
            }
            CodeKind::Huffman => 0.0,
        };
        println!(
            "{dir_name} [{}, {}] {}: n_sym {}, n_rare {}, esc_len {}, rare mass {esc_mass:.4}; \
             {:.4} bits/weight stored (streams {:.4}, padding {} B); device tables {} words; \
             encode {enc1:.2} s (1 thread) / {enc:.3} s ({threads}); load {load_s:.2} s; \
             decode {dec1:.2} s / {dec:.3} s",
            src.n(),
            src.k(),
            kind.name(),
            c.n_sym(),
            c.n_rare(),
            c.esc_len(),
            pack.stored_bytes() as f64 * 8.0 / weights,
            pack.streams().len() as f64 * 8.0 / weights,
            pack.padding_bytes(),
            pack.device_tables().unwrap().len(),
        );
        matched += 1;
    }
    println!("{matched} reference packs identical");
}
