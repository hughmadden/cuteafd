//! KX7 v1: an exact (lossless) code for BF16 matrices whose values were born FP8, the
//! storage of GLM-5.3-Flash's KDA `q/k/v/o` projections. Pure CPU: the packer, the decode
//! tables a GPU loads, and a scalar reference decoder that walks every stream the way one
//! GPU lane does, so it is the executable reference for the kernels.
//!
//! The code. Each 16-bit BF16 pattern is one symbol. One canonical Huffman code per pack
//! (per fused source): the unlimited Huffman lengths when the longest is at most 16 bits,
//! else package-merge limited to 16, with every tie broken exactly as the reference tool
//! (`kdaexact.py`: heap ties by pattern rank then node creation order, package-merge by a
//! stable sort on weight with packages ahead of leaves). Codes are canonical: symbols in
//! (length, pattern) order, first code 0, next = (prev + 1) << (len - prevlen). Bits are
//! written MSB-first within each byte.
//!
//! The core is layout-agnostic. A [`Kx7Layout`] only says which (row, col) positions each
//! stream holds, in order, how streams are aligned, and which streams together cover a band
//! of whole rows; histogram, code, bit packing and the lane walk never look further.
//! Layouts:
//!
//! * `frag32(S)`: K is cut into S slices; one stream per (slice s, 16-row tile t, lane l of
//!   32), ordered s, t, l. Per 16-wide K step j of its slice the stream holds the 8 values of
//!   lane l's `mma.m16n8k16` A fragment, with g = l / 4, c = 2 (l % 4), r0 = 16t + g,
//!   r1 = r0 + 8, k0 = 16j: W[r0][k0+c], W[r0][k0+c+1], W[r1][k0+c], W[r1][k0+c+1],
//!   W[r0][k0+c+8], W[r0][k0+c+9], W[r1][k0+c+8], W[r1][k0+c+9]. Streams 16-byte aligned.
//! * `lane16(S)`: as `kdaexact.py --layout lane16 --ksplit S` (SPEC 2.2): a row holds 4S
//!   streams; stream s*4 + t of row r covers inputs s*K/S + 64c + 16t + i, c over the
//!   slice's 64-wide chunks, i = 0..15. Streams ordered row-major, byte-aligned, so the
//!   table, streams and index are byte-identical to that tool's `dump_pack` files.
//!
//! The pack. A 32-word header ([`Kx7Header::to_words`]), the code table (u16 counts of
//! symbols per code length 1..16, then the symbols in canonical order: `dump_pack`'s
//! `table.bin`), one u32 byte offset per stream in stream order plus a final total
//! (`offsets.bin`), and the concatenated streams (`streams.bin`), each zero-padded to its
//! layout's alignment. Decode tables are built from the code table by a separate function
//! ([`two_level_tables`]) so their format can change without touching the pack.
//!
//! A lane may read up to [`READ_AHEAD_BYTES`] past the end of its stream (it refills 32 bits
//! whenever fewer than 32 are buffered). A device copy of the streams must therefore carry
//! that many bytes after the last stream; the reference decoder reads them as zeros.
use rayon::prelude::*;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use thiserror::Error;

/// Header word 0: the bytes "KX7\0" read as a little-endian u32.
pub const MAGIC: u32 = u32::from_le_bytes(*b"KX7\0");
pub const VERSION: u32 = 1;
/// Longest code, bits.
pub const MAXLEN: u32 = 16;
/// Header words; table sections and the stream index follow on 16-byte boundaries.
pub const HEADER_WORDS: usize = 32;
/// Most source tensors one pack can fuse.
pub const MAX_PARTS: usize = 8;
/// Header word 12 for the code of this module: canonical Huffman limited to [`MAXLEN`].
pub const CODE_HUFFMAN: u32 = 1;
/// Bytes a lane may read past the end of its stream.
pub const READ_AHEAD_BYTES: usize = 16;
/// The lane16 K split the reference kernels use.
pub const LANE16_DEFAULT_SLICES: usize = 4;
/// Code-table words before the symbols: one u16 count per code length 1..=16.
pub const TABLE_COUNTS: usize = 16;

const PATTERNS: usize = 1 << 16;
/// Streams one packing task writes; streams of a task are contiguous.
const ENCODE_GROUP: usize = 64;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum Kx7Error {
    #[error("KX7 {layout:?} S={slices} cannot hold [{n}, {k}]: {reason}")]
    Shape { n: usize, k: usize, layout: LayoutKind, slices: usize, reason: &'static str },
    #[error("KX7 source: {reason} (expected {expected}, found {found})")]
    Source { reason: &'static str, expected: usize, found: usize },
    #[error("KX7 code: {reason}")]
    Code { reason: String },
    #[error("KX7 header word {word} ({field}) is {found}: {reason}")]
    Header { word: usize, field: &'static str, found: u32, reason: &'static str },
    #[error("KX7 {what} truncated: needs {need}, has {have}")]
    Truncated { what: &'static str, need: usize, have: usize },
    #[error("KX7 table: {reason}")]
    Table { reason: String },
    #[error("KX7 index entry {entry} is {found}: {reason}")]
    Index { entry: usize, found: u64, reason: &'static str },
    #[error("KX7 streams total {bytes} bytes, past the u32 index range")]
    StreamsTooLarge { bytes: u64 },
    #[error("KX7 stream {stream}: pattern {pattern:#06x} has no codeword")]
    Uncoded { stream: usize, pattern: u16 },
    #[error("KX7 stream {stream}, symbol {symbol}: the bits match no codeword")]
    InvalidCode { stream: usize, symbol: usize },
    #[error("KX7 stream {stream} reads past the streams and their {READ_AHEAD_BYTES}-byte read-ahead")]
    Overrun { stream: usize },
    #[error("KX7 stream {stream}: its symbols take {expected} padded bytes, the index gives {actual}")]
    StreamLength { stream: usize, expected: usize, actual: usize },
    #[error("KX7 stream {stream}: padding after the last symbol is not zero")]
    Padding { stream: usize },
    #[error("KX7 verification failed at [{row}, {col}]: source {expected:#06x}, decoded {actual:#06x}")]
    Mismatch { row: usize, col: usize, expected: u16, actual: u16 },
}

type Result<T> = std::result::Result<T, Kx7Error>;

// ---------------------------------------------------------------- layouts

/// Header word 2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayoutKind {
    Frag32 = 1,
    Lane16 = 2,
}

impl LayoutKind {
    fn from_word(w: u32) -> Option<Self> {
        match w {
            1 => Some(Self::Frag32),
            2 => Some(Self::Lane16),
            _ => None,
        }
    }
}

/// Which positions each stream holds and how streams are aligned (module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Kx7Layout {
    pub kind: LayoutKind,
    /// K slices S.
    pub slices: usize,
    /// Byte alignment of every stream start and length (a power of two up to 256).
    pub align: usize,
}

/// frag32's split: the smallest S in {1, 2, 4, 8} that gives at least 1024 (slice, tile)
/// warps with K % (16 S) == 0, else the largest S in that set that divides K (q/k/v
/// [8192, 4096] -> 2, o [4096, 8192] -> 4).
pub fn default_slices(n: usize, k: usize) -> usize {
    let fits = |s: usize| k % (16 * s) == 0;
    [1, 2, 4, 8]
        .into_iter()
        .find(|&s| fits(s) && (n / 16) * s >= 1024)
        .or_else(|| [8, 4, 2, 1].into_iter().find(|&s| fits(s)))
        .unwrap_or(1)
}

impl Kx7Layout {
    pub fn frag32(slices: usize) -> Self {
        Self { kind: LayoutKind::Frag32, slices, align: 16 }
    }
    pub fn lane16(slices: usize) -> Self {
        Self { kind: LayoutKind::Lane16, slices, align: 1 }
    }
    pub fn with_align(self, align: usize) -> Self {
        Self { align, ..self }
    }

    fn shape_error(&self, n: usize, k: usize, reason: &'static str) -> Kx7Error {
        Kx7Error::Shape { n, k, layout: self.kind, slices: self.slices, reason }
    }

    /// Whether an [n, k] matrix fits this layout.
    pub fn check(&self, n: usize, k: usize) -> Result<()> {
        let s = self.slices;
        if n == 0 || k == 0 || n > u32::MAX as usize || k > u32::MAX as usize {
            return Err(self.shape_error(n, k, "needs 0 < n, k < 2^32"));
        }
        if s == 0 || s > 256 {
            return Err(self.shape_error(n, k, "needs 1 <= S <= 256"));
        }
        if !self.align.is_power_of_two() || self.align > 256 {
            return Err(self.shape_error(n, k, "needs the alignment a power of two up to 256"));
        }
        match self.kind {
            LayoutKind::Frag32 if n % 16 != 0 => Err(self.shape_error(n, k, "frag32 needs n % 16 == 0")),
            LayoutKind::Frag32 if k % (16 * s) != 0 => Err(self.shape_error(n, k, "frag32 needs k % (16 S) == 0")),
            LayoutKind::Lane16 if k % (64 * s) != 0 => Err(self.shape_error(n, k, "lane16 needs k % (64 S) == 0")),
            _ => Ok(()),
        }
    }

    pub fn streams(&self, n: usize, _k: usize) -> usize {
        match self.kind {
            LayoutKind::Frag32 => self.slices * (n / 16) * 32,
            LayoutKind::Lane16 => n * 4 * self.slices,
        }
    }

    pub fn symbols_per_stream(&self, k: usize) -> usize {
        match self.kind {
            LayoutKind::Frag32 => k / (2 * self.slices),
            LayoutKind::Lane16 => k / (4 * self.slices),
        }
    }

    /// Rows of one band: the streams of a band hold exactly its rows, every column.
    pub fn band_rows(&self) -> usize {
        match self.kind {
            LayoutKind::Frag32 => 16,
            LayoutKind::Lane16 => 1,
        }
    }

    /// The streams of band `band`, in stream order.
    pub fn band_streams(&self, n: usize, band: usize, out: &mut Vec<usize>) {
        out.clear();
        match self.kind {
            LayoutKind::Frag32 => {
                let tiles = n / 16;
                for s in 0..self.slices {
                    out.extend((0..32).map(|l| (s * tiles + band) * 32 + l));
                }
            }
            LayoutKind::Lane16 => {
                let spr = 4 * self.slices;
                out.extend(band * spr..(band + 1) * spr);
            }
        }
    }

    /// The (row, col) positions of stream `stream`, in the order its symbols are coded.
    pub fn positions(&self, n: usize, k: usize, stream: usize, out: &mut Vec<(usize, usize)>) {
        out.clear();
        match self.kind {
            LayoutKind::Frag32 => {
                let tiles = n / 16;
                let lane = stream % 32;
                let tile = (stream / 32) % tiles;
                let slice = stream / 32 / tiles;
                let steps = k / (16 * self.slices);
                let (r0, c) = (16 * tile + lane / 4, 2 * (lane % 4));
                let r1 = r0 + 8;
                for j in slice * steps..(slice + 1) * steps {
                    let k0 = 16 * j + c;
                    out.extend_from_slice(&[
                        (r0, k0),
                        (r0, k0 + 1),
                        (r1, k0),
                        (r1, k0 + 1),
                        (r0, k0 + 8),
                        (r0, k0 + 9),
                        (r1, k0 + 8),
                        (r1, k0 + 9),
                    ]);
                }
            }
            LayoutKind::Lane16 => {
                let spr = 4 * self.slices;
                let (row, slice, t) = (stream / spr, (stream % spr) / 4, stream % 4);
                let ks = k / self.slices;
                for c in 0..ks / 64 {
                    let base = slice * ks + 64 * c + 16 * t;
                    out.extend((base..base + 16).map(|col| (row, col)));
                }
            }
        }
    }
}

// ---------------------------------------------------------------- source

/// The matrix a pack codes: one [n, k] tensor, or several [n_i, k] tensors fused by rows
/// (q; k; v) under one code. Rows are borrowed, never copied.
#[derive(Clone)]
pub struct Kx7Source<'a> {
    k: usize,
    rows: Vec<&'a [u16]>,
    part_ends: Vec<usize>,
}

impl std::fmt::Debug for Kx7Source<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Kx7Source")
            .field("n", &self.n())
            .field("k", &self.k)
            .field("part_ends", &self.part_ends)
            .finish()
    }
}

impl<'a> Kx7Source<'a> {
    /// One row-major [n, k] matrix.
    pub fn matrix(w: &'a [u16], n: usize, k: usize) -> Result<Self> {
        Self::fused(&[(w, n)], k)
    }

    /// Parts (data, rows), each row-major [rows, k], concatenated in order.
    pub fn fused(parts: &[(&'a [u16], usize)], k: usize) -> Result<Self> {
        if parts.is_empty() || parts.len() > MAX_PARTS {
            return Err(Kx7Error::Source {
                reason: "parts must number 1 to 8",
                expected: MAX_PARTS,
                found: parts.len(),
            });
        }
        if k == 0 {
            return Err(Kx7Error::Source { reason: "k must be positive", expected: 1, found: 0 });
        }
        let mut rows = Vec::new();
        let mut part_ends = Vec::with_capacity(parts.len());
        for &(w, n) in parts {
            if n == 0 {
                return Err(Kx7Error::Source { reason: "every part needs rows", expected: 1, found: 0 });
            }
            if w.len() != n * k {
                return Err(Kx7Error::Source {
                    reason: "part length must be rows * k",
                    expected: n * k,
                    found: w.len(),
                });
            }
            rows.extend(w.chunks_exact(k));
            part_ends.push(rows.len());
        }
        Ok(Self { k, rows, part_ends })
    }

    pub fn n(&self) -> usize {
        self.rows.len()
    }
    pub fn k(&self) -> usize {
        self.k
    }
    /// Exclusive end row of each part.
    pub fn part_ends(&self) -> &[usize] {
        &self.part_ends
    }
    pub fn row(&self, r: usize) -> &'a [u16] {
        self.rows[r]
    }
}

/// Occurrences of each of the 65,536 patterns (parallel, deterministic).
pub fn histogram(src: &Kx7Source<'_>) -> Vec<u64> {
    src.rows
        .par_iter()
        .fold(
            || vec![0u64; PATTERNS],
            |mut h, row| {
                row.iter().for_each(|&v| h[v as usize] += 1);
                h
            },
        )
        .reduce(
            || vec![0u64; PATTERNS],
            |mut a, b| {
                a.iter_mut().zip(b).for_each(|(x, y)| *x += y);
                a
            },
        )
}

// ---------------------------------------------------------------- code lengths

/// Unlimited Huffman depths of `freqs` (symbols in ascending pattern order), as
/// `kdaexact.py: _huffman_lengths`: a min-heap on (weight, id), leaves numbered by rank,
/// merged nodes from n upward in creation order.
fn huffman_depths(freqs: &[u64]) -> Vec<u32> {
    let n = freqs.len();
    if n == 1 {
        return vec![1];
    }
    let mut heap: BinaryHeap<Reverse<(u64, usize)>> = freqs.iter().enumerate().map(|(i, &f)| Reverse((f, i))).collect();
    let mut parent = vec![0usize; 2 * n - 1];
    let mut next = n;
    while heap.len() > 1 {
        let (Some(Reverse((f1, a))), Some(Reverse((f2, b)))) = (heap.pop(), heap.pop()) else { unreachable!() };
        parent[a] = next;
        parent[b] = next;
        heap.push(Reverse((f1 + f2, next)));
        next += 1;
    }
    // A parent's id exceeds its children's; the root is the last node.
    let mut depth = vec![0u32; 2 * n - 1];
    for node in (0..2 * n - 2).rev() {
        depth[node] = depth[parent[node]] + 1;
    }
    depth.truncate(n);
    depth
}

#[derive(Clone, Copy)]
enum Coin {
    Leaf(u32),
    Package(u32, u32),
}

/// Length-limited depths by package-merge, as `kdaexact.py: package_merge`: `items` are
/// (weight, pattern) sorted ascending; each level is the stable sort by weight of the
/// previous level's pairs followed by the leaves, so packages precede leaves on ties.
fn package_merge(items: &[(u64, u16)], maxlen: u32) -> Result<Vec<u32>> {
    let n = items.len();
    if n == 1 {
        return Ok(vec![1]);
    }
    if n > 1usize << maxlen {
        return Err(Kx7Error::Code { reason: format!("{n} symbols cannot take codes of at most {maxlen} bits") });
    }
    let leaves: Vec<(u64, Coin)> = items.iter().enumerate().map(|(i, &(f, _))| (f, Coin::Leaf(i as u32))).collect();
    let mut levels: Vec<Vec<(u64, Coin)>> = vec![leaves.clone()];
    for _ in 0..maxlen - 1 {
        let prev = levels.last().expect("level 0");
        let packages: Vec<(u64, Coin)> = (0..prev.len().saturating_sub(1))
            .step_by(2)
            .map(|i| (prev[i].0 + prev[i + 1].0, Coin::Package(i as u32, i as u32 + 1)))
            .collect();
        let mut level = Vec::with_capacity(packages.len() + n);
        let (mut p, mut l) = (0, 0);
        while p < packages.len() || l < n {
            if l == n || (p < packages.len() && packages[p].0 <= leaves[l].0) {
                level.push(packages[p]);
                p += 1;
            } else {
                level.push(leaves[l]);
                l += 1;
            }
        }
        levels.push(level);
    }
    // Buy face value n - 1 at denomination 1/2: the first 2n - 2 coins of the top level,
    // each adding one to the length of every leaf it holds.
    let mut lengths = vec![0u32; n];
    let top = levels.last().expect("top level");
    let mut bought = vec![0u32; top.len()];
    bought.iter_mut().take(2 * n - 2).for_each(|t| *t = 1);
    for depth in (0..levels.len()).rev() {
        let mut below = vec![0u32; if depth > 0 { levels[depth - 1].len() } else { 0 }];
        for (&(_, coin), &t) in levels[depth].iter().zip(&bought) {
            if t == 0 {
                continue;
            }
            match coin {
                Coin::Leaf(i) => lengths[i as usize] += t,
                Coin::Package(a, b) => {
                    below[a as usize] += t;
                    below[b as usize] += t;
                }
            }
        }
        bought = below;
    }
    Ok(lengths)
}

/// Code lengths per pattern (65,536 entries, 0 = absent) with every length at most
/// [`MAXLEN`]: `kdaexact.py: length_limited_lengths(f, 16)`, tie for tie.
pub fn code_lengths(hist: &[u64]) -> Result<Vec<u8>> {
    if hist.len() != PATTERNS {
        return Err(Kx7Error::Code { reason: format!("histogram has {} bins, not 65536", hist.len()) });
    }
    let syms: Vec<u16> = (0..PATTERNS).filter(|&s| hist[s] > 0).map(|s| s as u16).collect();
    if syms.is_empty() {
        return Err(Kx7Error::Code { reason: "empty histogram".into() });
    }
    let freqs: Vec<u64> = syms.iter().map(|&s| hist[s as usize]).collect();
    let mut depths = huffman_depths(&freqs);
    let mut lengths = vec![0u8; PATTERNS];
    if depths.iter().any(|&d| d > MAXLEN) {
        let mut items: Vec<(u64, u16)> = freqs.iter().copied().zip(syms.iter().copied()).collect();
        items.sort_unstable();
        depths = package_merge(&items, MAXLEN)?;
        for (&(_, s), &d) in items.iter().zip(&depths) {
            lengths[s as usize] = d as u8;
        }
    } else {
        for (&s, &d) in syms.iter().zip(&depths) {
            lengths[s as usize] = d as u8;
        }
    }
    Ok(lengths)
}

// ---------------------------------------------------------------- canonical code

/// A canonical prefix code over 16-bit patterns, lengths 1..=16.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalCode {
    lengths: Vec<u8>,
    symbols: Vec<u16>,
    counts: [u32; MAXLEN as usize + 1],
}

impl CanonicalCode {
    /// From per-pattern lengths (65,536 entries, 0 = absent).
    pub fn from_lengths(lengths: Vec<u8>) -> Result<Self> {
        if lengths.len() != PATTERNS {
            return Err(Kx7Error::Code { reason: format!("{} lengths, not 65536", lengths.len()) });
        }
        let mut counts = [0u32; MAXLEN as usize + 1];
        for &l in &lengths {
            if l as u32 > MAXLEN {
                return Err(Kx7Error::Code { reason: format!("a code of {l} bits exceeds {MAXLEN}") });
            }
            counts[l as usize] += 1;
        }
        counts[0] = 0;
        let mut symbols = Vec::with_capacity(counts.iter().sum::<u32>() as usize);
        for len in 1..=MAXLEN as u8 {
            symbols.extend((0..PATTERNS).filter(|&s| lengths[s] == len).map(|s| s as u16));
        }
        Self::checked(lengths, symbols, counts).map_err(|reason| Kx7Error::Code { reason })
    }

    /// The code of a histogram: [`code_lengths`] then canonical assignment.
    pub fn from_histogram(hist: &[u64]) -> Result<Self> {
        Self::from_lengths(code_lengths(hist)?)
    }

    /// From a code table: u16 counts for lengths 1..=16, then the symbols in canonical order.
    /// A u16 count wraps, so the one code with all 65,536 patterns at 16 bits stores 0.
    pub fn from_table(table: &[u16]) -> Result<Self> {
        if table.len() < TABLE_COUNTS {
            return Err(Kx7Error::Truncated { what: "code table", need: TABLE_COUNTS, have: table.len() });
        }
        let symbols = table[TABLE_COUNTS..].to_vec();
        let mut counts = [0u32; MAXLEN as usize + 1];
        for (l, &c) in table[..TABLE_COUNTS].iter().enumerate() {
            counts[l + 1] = c as u32;
        }
        if symbols.len() == PATTERNS && counts.iter().all(|&c| c == 0) {
            counts[MAXLEN as usize] = PATTERNS as u32;
        }
        let total: u32 = counts.iter().sum();
        if total as usize != symbols.len() {
            return Err(Kx7Error::Table {
                reason: format!("counts sum to {total}, the table holds {} symbols", symbols.len()),
            });
        }
        let mut lengths = vec![0u8; PATTERNS];
        let mut i = 0;
        for len in 1..=MAXLEN as usize {
            for _ in 0..counts[len] {
                let s = symbols[i] as usize;
                if lengths[s] != 0 {
                    return Err(Kx7Error::Table { reason: format!("pattern {s:#06x} appears twice") });
                }
                lengths[s] = len as u8;
                i += 1;
            }
        }
        Self::checked(lengths, symbols, counts).map_err(|reason| Kx7Error::Table { reason })
    }

    fn checked(
        lengths: Vec<u8>,
        symbols: Vec<u16>,
        counts: [u32; MAXLEN as usize + 1],
    ) -> std::result::Result<Self, String> {
        if symbols.is_empty() {
            return Err("no symbols".into());
        }
        if symbols.len() == 1 && counts[1] != 1 {
            return Err("a one-symbol code must be one bit long".into());
        }
        let kraft: u64 = (1..=MAXLEN as usize).map(|l| (counts[l] as u64) << (MAXLEN as usize - l)).sum();
        if kraft > 1 << MAXLEN {
            return Err(format!("Kraft sum {kraft}/65536 exceeds 1"));
        }
        for w in symbols.windows(2) {
            let (la, lb) = (lengths[w[0] as usize], lengths[w[1] as usize]);
            if la > lb || (la == lb && w[0] >= w[1]) {
                return Err(format!("symbols {:#06x}, {:#06x} are not in (length, pattern) order", w[0], w[1]));
            }
        }
        Ok(Self { lengths, symbols, counts })
    }

    pub fn n_sym(&self) -> usize {
        self.symbols.len()
    }
    /// Symbols in canonical order.
    pub fn symbols(&self) -> &[u16] {
        &self.symbols
    }
    /// Per-pattern lengths, 0 = absent.
    pub fn lengths(&self) -> &[u8] {
        &self.lengths
    }
    /// Symbols of length `len` (1..=16).
    pub fn count(&self, len: u32) -> u32 {
        self.counts[len as usize]
    }

    /// The first code of each length 0..=17, not left-aligned (canonical recurrence).
    fn first_codes(&self) -> [u32; MAXLEN as usize + 2] {
        let mut first = [0u32; MAXLEN as usize + 2];
        for l in 1..=MAXLEN as usize + 1 {
            first[l] = (first[l - 1] + self.counts[l - 1]) << 1;
        }
        first
    }

    /// Left-aligned (16-bit) first code of length `len`.
    fn start(&self, len: u32) -> u32 {
        self.first_codes()[len as usize] << (MAXLEN - len)
    }

    /// Per pattern: `code << 8 | length` (0 = absent).
    pub fn codewords(&self) -> Vec<u32> {
        let mut first = self.first_codes();
        let mut out = vec![0u32; PATTERNS];
        for &s in &self.symbols {
            let l = self.lengths[s as usize] as usize;
            out[s as usize] = first[l] << 8 | l as u32;
            first[l] += 1;
        }
        out
    }

    /// The code table: u16 counts for lengths 1..=16 (wrapping), then the symbols.
    pub fn table(&self) -> Vec<u16> {
        let mut t: Vec<u16> = self.counts[1..].iter().map(|&c| c as u16).collect();
        t.extend_from_slice(&self.symbols);
        t
    }
}

// ---------------------------------------------------------------- decode tables

/// Table section word 0 for [`two_level_tables`].
pub const TABLE_TWO_LEVEL: u32 = 1;
/// Primary table index bits: the top 11 of a 16-bit peek.
pub const PRIMARY_BITS: u32 = 11;
/// Word offsets inside a two-level table section.
pub const TL_PRIMARY: usize = 4;
pub const TL_LONG: usize = TL_PRIMARY + (1 << PRIMARY_BITS);
pub const TL_COUNTS: usize = TL_LONG + 4 * TL_LONG_LENGTHS;
pub const TL_SYMBOLS: usize = TL_COUNTS + MAXLEN as usize;
/// Long code lengths PRIMARY_BITS + 1 ..= MAXLEN.
pub const TL_LONG_LENGTHS: usize = (MAXLEN - PRIMARY_BITS) as usize;
/// Primary entry flags (low byte).
pub const FLAG_LONG: u32 = 0;
pub const FLAG_LEAF: u32 = 1;
pub const FLAG_UNUSED: u32 = 2;

/// The two-level decode tables a GPU loads (into shared memory), as one u32 section:
///
/// | words | content |
/// |---|---|
/// | 0 | [`TABLE_TWO_LEVEL`] |
/// | 1 | n_sym |
/// | 2 | [`PRIMARY_BITS`] (11) |
/// | 3 | section words: `TL_SYMBOLS + ceil(n_sym / 2)` |
/// | 4 .. 2052 | primary[2048], indexed by `peek16 >> 5` |
/// | 2052 .. 2072 | long[L - 12] for L = 12..=16: {limit, start, base, count} |
/// | 2072 .. 2088 | counts[L - 1] for L = 1..=16 |
/// | 2088 .. | the symbols in canonical order, u16 pairs (symbol 2i in the low half) |
///
/// A primary entry is `pattern << 16 | len << 8 | flag`: flag 1, a leaf (len <= 11);
/// flag 0, a long code whose `len` is the shortest code length under that 11-bit prefix
/// (where the length search starts) and pattern 0; flag 2, a prefix no code takes. For the
/// long arrays, `start` is the first code of length L left-aligned to 16 bits, `limit` the
/// exclusive end of length-L codes in that space (`start + count << (16 - L)`), and `base`
/// the canonical index of length L's first symbol, so a long code decodes as
/// `L = len; while peek >= limit[L] { L++ }`, `symbols[base[L] + ((peek - start[L]) >> (16 - L))]`.
/// A one-symbol code fills every primary entry with its leaf (len 1), as `kdaexact.py`.
/// Sections 4, 2052, 2072 and 2088 words are 16-byte aligned.
pub fn two_level_tables(code: &CanonicalCode) -> Vec<u32> {
    let n_sym = code.n_sym();
    let mut t = vec![0u32; TL_SYMBOLS + n_sym.div_ceil(2)];
    t[0] = TABLE_TWO_LEVEL;
    t[1] = n_sym as u32;
    t[2] = PRIMARY_BITS;
    t[3] = t.len() as u32;
    let primary = &mut t[TL_PRIMARY..TL_LONG];
    primary.fill(FLAG_UNUSED);
    if n_sym == 1 {
        primary.fill((code.symbols[0] as u32) << 16 | 1 << 8 | FLAG_LEAF);
    } else {
        let cw = code.codewords();
        for &s in code.symbols.iter().rev() {
            let (c, l) = (cw[s as usize] >> 8, cw[s as usize] & 0xff);
            if l <= PRIMARY_BITS {
                let lo = (c << (PRIMARY_BITS - l)) as usize;
                let hi = ((c + 1) << (PRIMARY_BITS - l)) as usize;
                primary[lo..hi].fill((s as u32) << 16 | l << 8 | FLAG_LEAF);
            } else {
                // Reverse canonical order: the last write per prefix is its shortest code.
                primary[(c >> (l - PRIMARY_BITS)) as usize] = l << 8 | FLAG_LONG;
            }
        }
    }
    let mut base = 0u32;
    for l in 1..=MAXLEN {
        if l > PRIMARY_BITS {
            let start = code.start(l);
            let e = TL_LONG + 4 * (l - PRIMARY_BITS - 1) as usize;
            t[e..e + 4].copy_from_slice(&[start + (code.count(l) << (MAXLEN - l)), start, base, code.count(l)]);
        }
        t[TL_COUNTS + l as usize - 1] = code.count(l);
        base += code.count(l);
    }
    for (i, &s) in code.symbols.iter().enumerate() {
        t[TL_SYMBOLS + i / 2] |= (s as u32) << (16 * (i % 2));
    }
    t
}

/// The bit reader of one GPU lane: a 64-bit buffer whose valid bits sit at the top, refilled
/// with the next big-endian 32-bit word whenever it holds fewer than 32 bits.
pub struct LaneBits<'a> {
    bytes: &'a [u8],
    next: usize,
    buf: u64,
    nbits: u32,
    consumed: u64,
}

impl<'a> LaneBits<'a> {
    fn new(bytes: &'a [u8], start: usize) -> Self {
        Self { bytes, next: start, buf: 0, nbits: 0, consumed: 0 }
    }

    fn refill(&mut self) -> bool {
        if self.nbits >= 32 {
            return true;
        }
        if self.next + 4 > self.bytes.len() + READ_AHEAD_BYTES {
            return false;
        }
        let word = (0..4).fold(0u32, |w, i| w << 8 | *self.bytes.get(self.next + i).unwrap_or(&0) as u32);
        self.buf |= (word as u64) << (32 - self.nbits);
        self.nbits += 32;
        self.next += 4;
        true
    }

    /// The next 16 bits, MSB-aligned.
    pub fn peek16(&self) -> u32 {
        (self.buf >> 48) as u32
    }

    /// Drops `n` bits (at most the 32 a refill guarantees).
    pub fn consume(&mut self, n: u32) {
        debug_assert!(n <= self.nbits);
        self.buf <<= n;
        self.nbits -= n;
        self.consumed += n as u64;
    }
}

/// One symbol from a lane's bits (at least 32 buffered); `None` when they match no code.
pub trait SymbolDecoder: Sync {
    fn next(&self, bits: &mut LaneBits<'_>) -> Option<u16>;
}

/// The decoder of a [`two_level_tables`] section.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TwoLevelDecoder {
    primary: Vec<u32>,
    long: [[u32; 4]; TL_LONG_LENGTHS],
    symbols: Vec<u16>,
}

impl TwoLevelDecoder {
    /// Parses a table section and checks it word for word against the tables its own counts
    /// and symbols rebuild; returns the decoder and that code.
    pub fn parse(t: &[u32]) -> Result<(Self, CanonicalCode)> {
        if t.len() < TL_SYMBOLS {
            return Err(Kx7Error::Truncated { what: "two-level tables", need: TL_SYMBOLS, have: t.len() });
        }
        if t[0] != TABLE_TWO_LEVEL || t[2] != PRIMARY_BITS {
            return Err(Kx7Error::Table {
                reason: format!("kind {} with {} primary bits is not two-level/11", t[0], t[2]),
            });
        }
        let n_sym = t[1] as usize;
        let words = TL_SYMBOLS + n_sym.div_ceil(2);
        if n_sym > PATTERNS || t[3] as usize != words || t.len() != words {
            return Err(Kx7Error::Truncated { what: "two-level tables", need: words, have: t.len() });
        }
        let mut table: Vec<u16> = t[TL_COUNTS..TL_SYMBOLS].iter().map(|&c| c as u16).collect();
        table.extend((0..n_sym).map(|i| (t[TL_SYMBOLS + i / 2] >> (16 * (i % 2))) as u16));
        let code = CanonicalCode::from_table(&table)?;
        let rebuilt = two_level_tables(&code);
        if let Some(w) = (0..words).find(|&w| rebuilt[w] != t[w]) {
            return Err(Kx7Error::Table {
                reason: format!("word {w} is {:#x}, the code gives {:#x}", t[w], rebuilt[w]),
            });
        }
        let mut long = [[0u32; 4]; TL_LONG_LENGTHS];
        for (i, e) in long.iter_mut().enumerate() {
            e.copy_from_slice(&t[TL_LONG + 4 * i..TL_LONG + 4 * i + 4]);
        }
        let decoder = Self { primary: t[TL_PRIMARY..TL_LONG].to_vec(), long, symbols: code.symbols.clone() };
        Ok((decoder, code))
    }
}

impl SymbolDecoder for TwoLevelDecoder {
    fn next(&self, bits: &mut LaneBits<'_>) -> Option<u16> {
        let peek = bits.peek16();
        let e = self.primary[(peek >> (MAXLEN - PRIMARY_BITS)) as usize];
        let mut len = (e >> 8) & 0xff;
        match e & 0xff {
            FLAG_LEAF => {
                bits.consume(len);
                Some((e >> 16) as u16)
            }
            FLAG_LONG => loop {
                if len > MAXLEN || len <= PRIMARY_BITS {
                    return None;
                }
                let [limit, start, base, _] = self.long[(len - PRIMARY_BITS - 1) as usize];
                if peek < limit {
                    let offset = peek.wrapping_sub(start) >> (MAXLEN - len);
                    let sym = *self.symbols.get(base.wrapping_add(offset) as usize)?;
                    bits.consume(len);
                    return Some(sym);
                }
                len += 1;
            },
            _ => None,
        }
    }
}

// ---------------------------------------------------------------- header

/// The pack header, 32 u32 words:
///
/// | word | field |
/// |---|---|
/// | 0 | [`MAGIC`] ("KX7\0") |
/// | 1 | [`VERSION`] (1) |
/// | 2 | layout: 1 frag32, 2 lane16 |
/// | 3 | n, rows over all parts |
/// | 4 | k |
/// | 5 | S, K slices |
/// | 6 | maxlen (16) |
/// | 7 | n_sym |
/// | 8 | stream bytes (the index's final entry) |
/// | 9 | streams |
/// | 10 | symbols per stream |
/// | 11 | stream alignment, bytes |
/// | 12 | code: [`CODE_HUFFMAN`] |
/// | 13 | parts (1..=8) |
/// | 14, 15 | 0 |
/// | 16 .. 24 | exclusive end row of each part (part i covers [end[i-1], end[i])), unused 0 |
/// | 24 .. 32 | 0 |
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Kx7Header {
    pub layout: Kx7Layout,
    pub n: usize,
    pub k: usize,
    pub n_sym: usize,
    pub stream_bytes: usize,
    pub part_ends: Vec<usize>,
}

impl Kx7Header {
    pub fn streams(&self) -> usize {
        self.layout.streams(self.n, self.k)
    }
    pub fn symbols_per_stream(&self) -> usize {
        self.layout.symbols_per_stream(self.k)
    }

    pub fn to_words(&self) -> [u32; HEADER_WORDS] {
        let mut w = [0u32; HEADER_WORDS];
        w[..14].copy_from_slice(&[
            MAGIC,
            VERSION,
            self.layout.kind as u32,
            self.n as u32,
            self.k as u32,
            self.layout.slices as u32,
            MAXLEN,
            self.n_sym as u32,
            self.stream_bytes as u32,
            self.streams() as u32,
            self.symbols_per_stream() as u32,
            self.layout.align as u32,
            CODE_HUFFMAN,
            self.part_ends.len() as u32,
        ]);
        for (i, &e) in self.part_ends.iter().enumerate() {
            w[16 + i] = e as u32;
        }
        w
    }

    pub fn from_words(w: &[u32]) -> Result<Self> {
        if w.len() < HEADER_WORDS {
            return Err(Kx7Error::Truncated { what: "header", need: HEADER_WORDS, have: w.len() });
        }
        let bad = |word: usize, field: &'static str, reason: &'static str| Kx7Error::Header {
            word,
            field,
            found: w[word],
            reason,
        };
        if w[0] != MAGIC {
            return Err(bad(0, "magic", "not \"KX7\\0\""));
        }
        if w[1] != VERSION {
            return Err(bad(1, "version", "only version 1 is defined"));
        }
        let kind = LayoutKind::from_word(w[2]).ok_or_else(|| bad(2, "layout", "not 1 (frag32) or 2 (lane16)"))?;
        if w[6] != MAXLEN {
            return Err(bad(6, "maxlen", "must be 16"));
        }
        if w[12] != CODE_HUFFMAN {
            return Err(bad(12, "code", "must be 1 (canonical Huffman)"));
        }
        let layout = Kx7Layout { kind, slices: w[5] as usize, align: w[11] as usize };
        let (n, k) = (w[3] as usize, w[4] as usize);
        layout.check(n, k)?;
        if w[7] == 0 || w[7] as usize > PATTERNS {
            return Err(bad(7, "n_sym", "must be 1 to 65536"));
        }
        if w[9] as usize != layout.streams(n, k) {
            return Err(bad(9, "streams", "disagrees with the layout"));
        }
        if w[10] as usize != layout.symbols_per_stream(k) {
            return Err(bad(10, "symbols per stream", "disagrees with the layout"));
        }
        let parts = w[13] as usize;
        if parts == 0 || parts > MAX_PARTS {
            return Err(bad(13, "parts", "must be 1 to 8"));
        }
        let mut part_ends = Vec::with_capacity(parts);
        for i in 0..parts {
            let end = w[16 + i] as usize;
            if end <= part_ends.last().copied().unwrap_or(0) || end > n {
                return Err(bad(16 + i, "part end", "part ends must rise within n"));
            }
            part_ends.push(end);
        }
        if part_ends[parts - 1] != n {
            return Err(bad(16 + parts - 1, "part end", "the last part must end at n"));
        }
        if let Some(i) = [14, 15].into_iter().chain(16 + parts..HEADER_WORDS).find(|&i| w[i] != 0) {
            return Err(bad(i, "reserved", "must be 0"));
        }
        Ok(Self { layout, n, k, n_sym: w[7] as usize, stream_bytes: w[8] as usize, part_ends })
    }
}

// ---------------------------------------------------------------- pack

/// One packed matrix: header, code, stream index and streams (module docs).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Kx7Pack {
    header: Kx7Header,
    code: CanonicalCode,
    index: Vec<u32>,
    streams: Vec<u8>,
}

impl Kx7Pack {
    pub fn header(&self) -> &Kx7Header {
        &self.header
    }
    pub fn code(&self) -> &CanonicalCode {
        &self.code
    }
    /// u32 byte offset of every stream, in stream order, then the total.
    pub fn index(&self) -> &[u32] {
        &self.index
    }
    pub fn streams(&self) -> &[u8] {
        &self.streams
    }
    /// The code table (`dump_pack`'s table.bin).
    pub fn table(&self) -> Vec<u16> {
        self.code.table()
    }
    /// The two-level decode tables for this pack's code.
    pub fn decode_tables(&self) -> Vec<u32> {
        two_level_tables(&self.code)
    }

    /// Bytes held: header, code table, index and streams.
    pub fn stored_bytes(&self) -> usize {
        4 * HEADER_WORDS + 2 * (TABLE_COUNTS + self.code.n_sym()) + 4 * self.index.len() + self.streams.len()
    }

    /// Reassembles and validates a pack from its parts.
    pub fn from_parts(header: &[u32], table: &[u16], index: Vec<u32>, streams: Vec<u8>) -> Result<Self> {
        let header = Kx7Header::from_words(header)?;
        let code = CanonicalCode::from_table(table)?;
        if code.n_sym() != header.n_sym {
            return Err(Kx7Error::Table {
                reason: format!("{} symbols, the header says {}", code.n_sym(), header.n_sym),
            });
        }
        let ns = header.streams();
        if index.len() != ns + 1 {
            return Err(Kx7Error::Truncated { what: "stream index", need: ns + 1, have: index.len() });
        }
        if index[0] != 0 {
            return Err(Kx7Error::Index { entry: 0, found: index[0] as u64, reason: "the first stream starts at 0" });
        }
        let align = header.layout.align as u32;
        for (i, w) in index.windows(2).enumerate() {
            if w[1] < w[0] {
                return Err(Kx7Error::Index { entry: i + 1, found: w[1] as u64, reason: "offsets must not fall" });
            }
            if w[1] % align != 0 {
                return Err(Kx7Error::Index {
                    entry: i + 1,
                    found: w[1] as u64,
                    reason: "offset breaks the alignment",
                });
            }
        }
        if index[ns] as usize != header.stream_bytes {
            return Err(Kx7Error::Index {
                entry: ns,
                found: index[ns] as u64,
                reason: "the total differs from the header",
            });
        }
        if streams.len() != header.stream_bytes {
            return Err(Kx7Error::Truncated { what: "streams", need: header.stream_bytes, have: streams.len() });
        }
        Ok(Self { header, code, index, streams })
    }
}

// ---------------------------------------------------------------- encode

/// MSB-first bit writer into a zeroed region.
struct BitWriter<'a> {
    out: &'a mut [u8],
    pos: usize,
    acc: u64,
    n: u32,
}

impl BitWriter<'_> {
    fn put(&mut self, code: u32, len: u32) {
        self.acc = self.acc << len | code as u64;
        self.n += len;
        while self.n >= 8 {
            self.n -= 8;
            self.out[self.pos] = (self.acc >> self.n) as u8;
            self.pos += 1;
        }
        self.acc &= (1 << self.n) - 1;
    }

    fn finish(self) {
        if self.n > 0 {
            self.out[self.pos] = (self.acc << (8 - self.n)) as u8;
        }
    }
}

/// Packs every stream of `layout` over `src` with a per-pattern codebook (`code << 8 | len`,
/// len <= 24; [`CanonicalCode::codewords`]): returns the index and the streams. Two parallel
/// passes, stream lengths then bits; the output does not depend on the thread count.
pub fn pack_streams(src: &Kx7Source<'_>, layout: Kx7Layout, codewords: &[u32]) -> Result<(Vec<u32>, Vec<u8>)> {
    let (n, k) = (src.n(), src.k());
    layout.check(n, k)?;
    if codewords.len() != PATTERNS {
        return Err(Kx7Error::Code { reason: format!("{} codewords, not 65536", codewords.len()) });
    }
    let ns = layout.streams(n, k);
    let bits: Vec<u64> = (0..ns)
        .into_par_iter()
        .map_init(Vec::new, |pos, stream| {
            layout.positions(n, k, stream, pos);
            let mut bits = 0u64;
            for &(r, c) in pos.iter() {
                let v = src.rows[r][c];
                let len = codewords[v as usize] & 0xff;
                if len == 0 {
                    return Err(Kx7Error::Uncoded { stream, pattern: v });
                }
                bits += len as u64;
            }
            Ok(bits)
        })
        .collect::<Result<_>>()?;
    let mut index = Vec::with_capacity(ns + 1);
    let mut total = 0u64;
    index.push(0);
    for &b in &bits {
        total += b.div_ceil(8).next_multiple_of(layout.align as u64);
        if total > u32::MAX as u64 {
            return Err(Kx7Error::StreamsTooLarge { bytes: total });
        }
        index.push(total as u32);
    }
    let mut streams = vec![0u8; total as usize];
    let mut groups = Vec::with_capacity(ns.div_ceil(ENCODE_GROUP));
    let mut rest = streams.as_mut_slice();
    for first in (0..ns).step_by(ENCODE_GROUP) {
        let end = (first + ENCODE_GROUP).min(ns);
        let (head, tail) = std::mem::take(&mut rest).split_at_mut((index[end] - index[first]) as usize);
        groups.push((first, end, head));
        rest = tail;
    }
    groups.into_par_iter().for_each_init(Vec::new, |pos, (first, end, buf)| {
        let base = index[first] as usize;
        for stream in first..end {
            layout.positions(n, k, stream, pos);
            let region = &mut buf[index[stream] as usize - base..index[stream + 1] as usize - base];
            let mut w = BitWriter { out: region, pos: 0, acc: 0, n: 0 };
            for &(r, c) in pos.iter() {
                let cw = codewords[src.rows[r][c] as usize];
                w.put(cw >> 8, cw & 0xff);
            }
            w.finish();
        }
    });
    Ok((index, streams))
}

/// Packs `src` in `layout` under its own canonical Huffman code.
pub fn encode(src: &Kx7Source<'_>, layout: Kx7Layout) -> Result<Kx7Pack> {
    layout.check(src.n(), src.k())?;
    let code = CanonicalCode::from_histogram(&histogram(src))?;
    let (index, streams) = pack_streams(src, layout, &code.codewords())?;
    let header = Kx7Header {
        layout,
        n: src.n(),
        k: src.k(),
        n_sym: code.n_sym(),
        stream_bytes: streams.len(),
        part_ends: src.part_ends.clone(),
    };
    Ok(Kx7Pack { header, code, index, streams })
}

/// Packs one row-major [n, k] matrix.
pub fn encode_matrix(w: &[u16], n: usize, k: usize, layout: Kx7Layout) -> Result<Kx7Pack> {
    encode(&Kx7Source::matrix(w, n, k)?, layout)
}

// ---------------------------------------------------------------- decode

/// Decodes one stream as a lane does, then checks it fills its index entry exactly
/// (padding to the alignment, zero bits after the last symbol).
fn walk_stream<D: SymbolDecoder>(dec: &D, pack: &Kx7Pack, stream: usize, out: &mut [u16]) -> Result<()> {
    let (start, end) = (pack.index[stream] as usize, pack.index[stream + 1] as usize);
    let mut bits = LaneBits::new(&pack.streams, start);
    for (symbol, o) in out.iter_mut().enumerate() {
        if !bits.refill() {
            return Err(Kx7Error::Overrun { stream });
        }
        *o = dec.next(&mut bits).ok_or(Kx7Error::InvalidCode { stream, symbol })?;
    }
    let used = bits.consumed.div_ceil(8) as usize;
    let padded = used.next_multiple_of(pack.header.layout.align);
    if padded != end - start {
        return Err(Kx7Error::StreamLength { stream, expected: padded, actual: end - start });
    }
    let tail_bits = (bits.consumed % 8) as u32;
    let partial = tail_bits != 0 && pack.streams[start + used - 1] & (0xffu8 >> tail_bits) != 0;
    if partial || pack.streams[start + used..end].iter().any(|&b| b != 0) {
        return Err(Kx7Error::Padding { stream });
    }
    Ok(())
}

/// Decodes a pack with any symbol decoder into the [n, k] row-major source (parts in order).
pub fn decode_with<D: SymbolDecoder>(pack: &Kx7Pack, dec: &D) -> Result<Vec<u16>> {
    let (n, k, layout) = (pack.header.n, pack.header.k, pack.header.layout);
    let band_rows = layout.band_rows();
    let mut out = vec![0u16; n * k];
    out.par_chunks_mut(band_rows * k).enumerate().try_for_each_init(
        || (Vec::new(), Vec::new(), vec![0u16; layout.symbols_per_stream(k)]),
        |(streams, pos, syms), (band, chunk)| {
            layout.band_streams(n, band, streams);
            for &stream in streams.iter() {
                walk_stream(dec, pack, stream, syms)?;
                layout.positions(n, k, stream, pos);
                for (&(r, c), &v) in pos.iter().zip(syms.iter()) {
                    chunk[(r - band * band_rows) * k + c] = v;
                }
            }
            Ok(())
        },
    )?;
    Ok(out)
}

/// The reference decoder: builds this pack's two-level tables (the image a GPU loads),
/// parses them back and walks every stream with them.
pub fn decode(pack: &Kx7Pack) -> Result<Vec<u16>> {
    let (dec, _) = TwoLevelDecoder::parse(&pack.decode_tables())?;
    decode_with(pack, &dec)
}

/// Decodes and compares bit for bit; the error names the first differing position.
pub fn verify(pack: &Kx7Pack, src: &Kx7Source<'_>) -> Result<()> {
    let h = &pack.header;
    if (src.n(), src.k()) != (h.n, h.k) || src.part_ends() != h.part_ends.as_slice() {
        return Err(Kx7Error::Source { reason: "source rows differ from the pack", expected: h.n, found: src.n() });
    }
    let out = decode(pack)?;
    for (r, (row, got)) in src.rows.iter().zip(out.chunks_exact(h.k)).enumerate() {
        if let Some(c) = (0..h.k).find(|&c| row[c] != got[c]) {
            return Err(Kx7Error::Mismatch { row: r, col: c, expected: row[c], actual: got[c] });
        }
    }
    Ok(())
}

/// [`verify`] against one row-major [n, k] matrix.
pub fn verify_matrix(pack: &Kx7Pack, w: &[u16]) -> Result<()> {
    verify(pack, &Kx7Source::matrix(w, pack.header.n, pack.header.k)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};
    use std::collections::HashSet;

    /// SplitMix64: fixtures must be identical on every machine, so no floating-point
    /// library calls (only IEEE-exact +, *, /, sqrt).
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
    }

    /// 85% exactly an E4M3 value times a power of two (3 mantissa bits), 15% ordinary BF16 at
    /// the same 16 exponents (the shape of `kdaexact.py`'s self-test fixture).
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

    /// Zipf(1.5) over `ranks` ranks, rank r mapped to pattern (r + 1) * 40503 mod 2^16 so
    /// frequency order and pattern order differ: steep enough that the unlimited Huffman code
    /// exceeds 16 bits and package-merge takes over.
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
                let rank = cdf.partition_point(|&c| c < u).min(ranks - 1);
                ((rank as u32 + 1).wrapping_mul(40503) & 0xffff) as u16
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
        let mut g = Mix(seed);
        for i in (1..w.len()).rev() {
            w.swap(i, g.below(i as u64 + 1) as usize);
        }
        w
    }

    fn round_trip(w: &[u16], n: usize, k: usize, layout: Kx7Layout) -> Kx7Pack {
        let pack = encode_matrix(w, n, k, layout).unwrap();
        verify_matrix(&pack, w).unwrap();
        assert_eq!(decode(&pack).unwrap(), w);
        let h = pack.header();
        let again =
            Kx7Pack::from_parts(&h.to_words(), &pack.table(), pack.index().to_vec(), pack.streams().to_vec()).unwrap();
        assert_eq!(again, pack);
        pack
    }

    fn sha(bytes: &[u8]) -> String {
        Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
    }
    fn le16(v: &[u16]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }
    fn le32(v: &[u32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
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
        layout.positions(n, k, (1 * 2 + 1) * 32 + 5, &mut pos);
        let mut want = Vec::new();
        for j in 2..4 {
            let k0 = 16 * j + 2;
            want.extend([
                (17, k0),
                (17, k0 + 1),
                (25, k0),
                (25, k0 + 1),
                (17, k0 + 8),
                (17, k0 + 9),
                (25, k0 + 8),
                (25, k0 + 9),
            ]);
        }
        assert_eq!(pos, want);
    }

    #[test]
    fn kx7_lane16_follows_kdaexact_ksplit_permutation() {
        // kdaexact.py lane16_permutation(K, ksplit): row-major streams of K / (4 S) inputs,
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
    fn kx7_round_trips_fp8_born_fixtures() {
        let cases = [
            (16, 16, Kx7Layout::frag32(1)),
            (64, 4096, Kx7Layout::frag32(2)),
            (32, 8192, Kx7Layout::frag32(4)),
            (64, 4096, Kx7Layout::frag32(default_slices(64, 4096))),
            (16, 64, Kx7Layout::lane16(1)),
            (64, 4096, Kx7Layout::lane16(LANE16_DEFAULT_SLICES)),
            (32, 8192, Kx7Layout::lane16(4).with_align(16)),
        ];
        for (i, (n, k, layout)) in cases.into_iter().enumerate() {
            let w = fp8_born(n, k, 100 + i as u64);
            let pack = round_trip(&w, n, k, layout);
            let bpw = pack.stored_bytes() as f64 * 8.0 / (n * k) as f64;
            assert!(n * k < 65536 || bpw < 10.5, "{layout:?} [{n}, {k}]: {bpw} bits/weight");
        }
    }

    #[test]
    fn kx7_round_trips_odd_distributions() {
        // One symbol: a 1-bit code, every primary entry its leaf.
        for (n, k) in [(16, 16), (64, 4096)] {
            let w = vec![0x3c00u16; n * k];
            let pack = round_trip(&w, n, k, Kx7Layout::frag32(1));
            assert_eq!(pack.code().symbols(), &[0x3c00]);
            assert_eq!(pack.code().count(1), 1);
            assert!(pack.decode_tables()[TL_PRIMARY..TL_LONG].iter().all(|&e| e == 0x3c00 << 16 | 1 << 8 | FLAG_LEAF));
            if k % 64 == 0 {
                round_trip(&w, n, k, Kx7Layout::lane16(4));
            }
        }
        // 4,096 equally frequent patterns: every code 12 bits, every symbol on the long path.
        let w = uniform4096(64, 4096, 7);
        let pack = round_trip(&w, 64, 4096, Kx7Layout::frag32(2));
        assert_eq!((pack.code().n_sym(), pack.code().count(12)), (4096, 4096));
        assert!(pack.decode_tables()[TL_PRIMARY..TL_LONG].iter().all(|&e| e == 12 << 8 | FLAG_LONG));
        round_trip(&w, 64, 4096, Kx7Layout::lane16(4));
        // Skewed: the unlimited code passes 16 bits and package-merge caps it.
        let w = zipf(64, 4096, 11, 8192);
        let hist = histogram(&Kx7Source::matrix(&w, 64, 4096).unwrap());
        let freqs: Vec<u64> = hist.iter().copied().filter(|&f| f > 0).collect();
        assert!(huffman_depths(&freqs).into_iter().max().unwrap() > 16);
        let pack = round_trip(&w, 64, 4096, Kx7Layout::frag32(2));
        assert!(pack.code().count(16) > 0);
        round_trip(&w, 64, 4096, Kx7Layout::lane16(4));
        // Random patterns over most of the alphabet.
        let mut g = Mix(3);
        let w: Vec<u16> = (0..64 * 2048).map(|_| g.next() as u16).collect();
        round_trip(&w, 64, 2048, Kx7Layout::frag32(2));
        // All 65,536 patterns, 4 times each: every code 16 bits, the u16 count wraps to 0.
        let w: Vec<u16> = (0..256 * 1024).map(|i| (i as u32).wrapping_mul(40503) as u16).collect();
        let pack = round_trip(&w, 256, 1024, Kx7Layout::lane16(4));
        assert_eq!((pack.code().n_sym(), pack.code().count(16), pack.table()[15]), (65536, 65536, 0));
    }

    #[test]
    fn kx7_fused_source_round_trips_under_one_code() {
        let k = 4096;
        let (q, kk, v) = (fp8_born(16, k, 21), fp8_born(32, k, 22), zipf(16, k, 23, 4096));
        let src = Kx7Source::fused(&[(&q, 16), (&kk, 32), (&v, 16)], k).unwrap();
        let whole: Vec<u16> = [q.as_slice(), &kk, &v].concat();
        for layout in [Kx7Layout::lane16(4), Kx7Layout::frag32(2)] {
            let pack = encode(&src, layout).unwrap();
            verify(&pack, &src).unwrap();
            assert_eq!(decode(&pack).unwrap(), whole);
            assert_eq!(pack.header().part_ends, vec![16, 48, 64]);
            let words = pack.header().to_words();
            assert_eq!((words[13], &words[16..20]), (3, &[16u32, 48, 64, 0][..]));
            // The same rows as one matrix give the same streams; only the part record differs.
            let single = encode_matrix(&whole, 64, k, layout).unwrap();
            assert_eq!(
                (single.streams(), single.index(), single.table()),
                (pack.streams(), pack.index(), pack.table())
            );
            assert!(matches!(verify(&single, &src), Err(Kx7Error::Source { .. })));
        }
    }

    #[test]
    fn kx7_encoding_is_deterministic() {
        let w = fp8_born(64, 4096, 5);
        for layout in [Kx7Layout::frag32(2), Kx7Layout::lane16(4)] {
            let a = encode_matrix(&w, 64, 4096, layout).unwrap();
            let b = encode_matrix(&w, 64, 4096, layout).unwrap();
            let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
            let c = pool.install(|| encode_matrix(&w, 64, 4096, layout).unwrap());
            for p in [&b, &c] {
                assert_eq!(p.header().to_words(), a.header().to_words());
                assert_eq!((p.table(), p.index(), p.streams()), (a.table(), a.index(), a.streams()));
                assert_eq!(p.decode_tables(), a.decode_tables());
            }
        }
    }

    #[test]
    fn kx7_tables_decode_every_codeword() {
        for w in [fp8_born(64, 4096, 9), zipf(64, 4096, 10, 8192), uniform4096(16, 4096, 2)] {
            let code = CanonicalCode::from_histogram(&histogram(&Kx7Source::matrix(&w, w.len() / 4096, 4096).unwrap()))
                .unwrap();
            let (dec, back) = TwoLevelDecoder::parse(&two_level_tables(&code)).unwrap();
            assert_eq!(back, code);
            let cw = code.codewords();
            let mut g = Mix(1);
            for &s in code.symbols() {
                let (c, l) = (cw[s as usize] >> 8, cw[s as usize] & 0xff);
                // The codeword left-aligned in a 32-bit word, random bits after it.
                let word = c << (32 - l) | (g.next() as u32 >> l);
                let bytes = word.to_be_bytes();
                let mut bits = LaneBits::new(&bytes, 0);
                assert!(bits.refill());
                assert_eq!(dec.next(&mut bits), Some(s));
                assert_eq!(bits.consumed, l as u64);
            }
        }
    }

    /// Expected values from the KX7 reference tool, `kdaexact.py`. Produced by running the
    /// ignored test `kx7_dump_reference_fixtures` (raw little-endian u16 files), then, with
    /// python3 -I and numpy, for each fixture `a` ([64, 4096]):
    /// `f = np.bincount(a.ravel(), minlength=65536)`, `lens = length_limited_lengths(f, 16)`
    /// (`lengths`: one hex digit, length - 1, per used pattern in ascending pattern order;
    /// `unlimited_max`: `length_limited_lengths(f, 64).max()`), and from the tool's
    /// `--ksplit`/`--dump` revision `run(a, 0, 16, layout='lane16', ksplit=4, pack=p)` then
    /// `dump_pack(d, a, r, p, source=False)`: the SHA-256 of its table.bin, streams.bin and
    /// offsets.bin. Both revisions gave identical lengths.
    const REFERENCE: &str = include_str!("fixtures/kx7_reference.json");

    fn reference_fixtures() -> Vec<(&'static str, usize, usize, Vec<u16>)> {
        vec![
            ("fp8_born", 64, 4096, fp8_born(64, 4096, 0x4b58_3701)),
            ("fp8_narrow", 64, 4096, fp8_born_over(64, 4096, 0x4b58_3703, 8)),
            ("zipf", 64, 4096, zipf(64, 4096, 0x4b58_3702, 8192)),
        ]
    }

    #[test]
    fn kx7_code_lengths_and_lane16_pack_match_kdaexact() {
        let reference: serde_json::Value = serde_json::from_str(REFERENCE).unwrap();
        for (name, n, k, w) in reference_fixtures() {
            let r = &reference["fixtures"][name];
            assert_eq!((r["n"].as_u64(), r["k"].as_u64()), (Some(n as u64), Some(k as u64)), "{name}");
            let hist = histogram(&Kx7Source::matrix(&w, n, k).unwrap());
            let lengths = code_lengths(&hist).unwrap();
            let got: String =
                lengths.iter().filter(|&&l| l > 0).map(|&l| char::from_digit(l as u32 - 1, 16).unwrap()).collect();
            assert_eq!(got, r["lengths"].as_str().unwrap(), "{name}: code lengths differ from kdaexact.py");
            let freqs: Vec<u64> = hist.iter().copied().filter(|&f| f > 0).collect();
            let unlimited = huffman_depths(&freqs).into_iter().max().unwrap();
            assert_eq!(unlimited as u64, r["unlimited_max"].as_u64().unwrap(), "{name}");
            let pack = encode_matrix(&w, n, k, Kx7Layout::lane16(4)).unwrap();
            let d = &r["lane16_ksplit4"];
            assert_eq!(sha(&le16(&pack.table())), d["table_sha256"].as_str().unwrap(), "{name}: table.bin");
            assert_eq!(sha(pack.streams()), d["streams_sha256"].as_str().unwrap(), "{name}: streams.bin");
            assert_eq!(sha(&le32(pack.index())), d["offsets_sha256"].as_str().unwrap(), "{name}: offsets.bin");
        }
        // Coverage: fp8_narrow takes the unlimited Huffman code, the others package-merge.
        let r = &reference["fixtures"];
        assert!(r["fp8_narrow"]["unlimited_max"].as_u64().unwrap() <= 16);
        assert!(
            r["fp8_born"]["unlimited_max"].as_u64().unwrap() > 16 && r["zipf"]["unlimited_max"].as_u64().unwrap() > 16
        );
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
    fn kx7_refuses_bad_shapes() {
        let w = vec![0u16; 64 * 256];
        let shape = |r: Result<Kx7Pack>| matches!(r, Err(Kx7Error::Shape { .. }));
        assert!(shape(encode_matrix(&w[..8 * 256], 8, 256, Kx7Layout::frag32(1))));
        assert!(shape(encode_matrix(&w[..64 * 40], 64, 40, Kx7Layout::frag32(1))));
        assert!(shape(encode_matrix(&w, 64, 256, Kx7Layout::frag32(32))));
        assert!(shape(encode_matrix(&w, 64, 256, Kx7Layout::frag32(0))));
        assert!(shape(encode_matrix(&w, 64, 256, Kx7Layout::lane16(8))));
        assert!(shape(encode_matrix(&w, 64, 256, Kx7Layout::lane16(1).with_align(3))));
        assert!(matches!(encode_matrix(&w, 64, 255, Kx7Layout::frag32(1)), Err(Kx7Error::Source { .. })));
        assert!(matches!(Kx7Source::fused(&[], 256), Err(Kx7Error::Source { .. })));
        assert!(matches!(Kx7Source::fused(&[(&w[..256], 1); 9], 256), Err(Kx7Error::Source { .. })));
        assert!(matches!(Kx7Source::fused(&[(&w[..256], 1), (&w[..0], 0)], 256), Err(Kx7Error::Source { .. })));
        let src = Kx7Source::matrix(&w, 64, 256).unwrap();
        let mut cw = vec![0u32; PATTERNS];
        assert!(matches!(pack_streams(&src, Kx7Layout::lane16(1), &cw), Err(Kx7Error::Uncoded { pattern: 0, .. })));
        cw.truncate(10);
        assert!(matches!(pack_streams(&src, Kx7Layout::lane16(1), &cw), Err(Kx7Error::Code { .. })));
    }

    fn sample_pack() -> Kx7Pack {
        let (q, k) = (fp8_born(16, 1024, 31), fp8_born(16, 1024, 32));
        encode(&Kx7Source::fused(&[(&q, 16), (&k, 16)], 1024).unwrap(), Kx7Layout::frag32(2)).unwrap()
    }

    fn reassemble(header: &[u32], table: &[u16], index: Vec<u32>, streams: Vec<u8>) -> Result<Kx7Pack> {
        Kx7Pack::from_parts(header, table, index, streams)
    }

    #[test]
    fn kx7_refuses_bad_headers() {
        let p = sample_pack();
        let good = p.header().to_words();
        let header = |word: usize, value: u32| {
            let mut h = good;
            h[word] = value;
            reassemble(&h, &p.table(), p.index().to_vec(), p.streams().to_vec())
        };
        let at = |r: Result<Kx7Pack>| match r {
            Err(Kx7Error::Header { word, .. }) => Some(word),
            _ => None,
        };
        assert_eq!(at(header(0, u32::from_le_bytes(*b"KX8\0"))), Some(0));
        assert_eq!(at(header(1, 2)), Some(1));
        assert_eq!(at(header(2, 3)), Some(2));
        assert_eq!(at(header(6, 15)), Some(6));
        assert_eq!(at(header(7, 0)), Some(7));
        assert_eq!(at(header(9, good[9] + 32)), Some(9));
        assert_eq!(at(header(10, 7)), Some(10));
        assert_eq!(at(header(12, 2)), Some(12));
        assert_eq!(at(header(13, 0)), Some(13));
        assert_eq!(at(header(13, 9)), Some(13));
        assert_eq!(at(header(13, 3)), Some(18));
        assert_eq!(at(header(16, 33)), Some(16));
        assert_eq!(at(header(16, 32)), Some(17));
        assert_eq!(at(header(17, 31)), Some(17));
        assert_eq!(at(header(14, 1)), Some(14));
        assert_eq!(at(header(31, 1)), Some(31));
        assert!(matches!(header(3, 24), Err(Kx7Error::Shape { .. })));
        assert!(matches!(header(5, 3), Err(Kx7Error::Shape { .. })));
        assert!(matches!(header(11, 24), Err(Kx7Error::Shape { .. })));
        assert!(matches!(header(7, good[7] + 1), Err(Kx7Error::Table { .. })));
        assert!(matches!(header(8, good[8] + 16), Err(Kx7Error::Index { .. })));
        assert!(matches!(
            reassemble(&good[..31], &p.table(), p.index().to_vec(), p.streams().to_vec()),
            Err(Kx7Error::Truncated { what: "header", .. })
        ));
    }

    #[test]
    fn kx7_refuses_bad_tables() {
        let p = sample_pack();
        let h = p.header().to_words();
        let table = |t: Vec<u16>| reassemble(&h, &t, p.index().to_vec(), p.streams().to_vec());
        let good = p.table();
        assert!(matches!(table(good[..10].to_vec()), Err(Kx7Error::Truncated { what: "code table", .. })));
        assert!(matches!(table(good[..good.len() - 1].to_vec()), Err(Kx7Error::Table { .. })));
        let first_len = (0..16).find(|&l| good[l] >= 2).unwrap();
        let first_sym = TABLE_COUNTS + good[..first_len].iter().map(|&c| c as usize).sum::<usize>();
        let mut swapped = good.clone();
        swapped.swap(first_sym, first_sym + 1);
        assert!(matches!(table(swapped), Err(Kx7Error::Table { .. })));
        let mut twice = good.clone();
        twice[first_sym + 1] = twice[first_sym];
        assert!(matches!(table(twice), Err(Kx7Error::Table { .. })));
        let mut kraft = good.clone();
        let (long, short) = ((0..16).rev().find(|&l| kraft[l] > 0).unwrap(), (0..16).find(|&l| kraft[l] > 0).unwrap());
        kraft[long] -= 1;
        kraft[short] += 1;
        assert!(matches!(table(kraft), Err(Kx7Error::Table { .. })));
        let mut one = vec![0u16; 16];
        one[2] = 1;
        one.push(0x3c00);
        assert!(matches!(CanonicalCode::from_table(&one), Err(Kx7Error::Table { .. })));
        assert!(matches!(CanonicalCode::from_lengths(vec![0; PATTERNS]), Err(Kx7Error::Code { .. })));
        assert!(matches!(code_lengths(&vec![0; PATTERNS]), Err(Kx7Error::Code { .. })));
        // Decode-table sections: truncation, kind, and any word the code does not give.
        let t = p.decode_tables();
        assert!(TwoLevelDecoder::parse(&t).is_ok());
        assert!(matches!(TwoLevelDecoder::parse(&t[..100]), Err(Kx7Error::Truncated { .. })));
        assert!(matches!(TwoLevelDecoder::parse(&t[..t.len() - 1]), Err(Kx7Error::Truncated { .. })));
        for (word, value) in [
            (0, 2),
            (2, 12),
            (TL_PRIMARY + 5, t[TL_PRIMARY + 5] ^ 1 << 8),
            (TL_LONG + 1, 7),
            (TL_SYMBOLS, t[TL_SYMBOLS] ^ 1),
        ] {
            let mut bad = t.clone();
            bad[word] = value;
            assert!(matches!(TwoLevelDecoder::parse(&bad), Err(Kx7Error::Table { .. })), "word {word}");
        }
    }

    #[test]
    fn kx7_refuses_bad_index_and_streams() {
        let p = sample_pack();
        let (h, t) = (p.header().to_words(), p.table());
        let index = |f: &dyn Fn(&mut Vec<u32>)| {
            let mut i = p.index().to_vec();
            f(&mut i);
            reassemble(&h, &t, i, p.streams().to_vec())
        };
        assert!(matches!(
            index(&|i| {
                i.pop();
            }),
            Err(Kx7Error::Truncated { what: "stream index", .. })
        ));
        assert!(matches!(index(&|i| i[0] = 16), Err(Kx7Error::Index { entry: 0, .. })));
        assert!(matches!(index(&|i| i[5] = i[7]), Err(Kx7Error::Index { entry: 6, .. })));
        assert!(matches!(index(&|i| i[5] += 1), Err(Kx7Error::Index { entry: 5, .. })));
        let last = p.index().len() - 1;
        assert!(matches!(index(&|i| i[last] += 16), Err(Kx7Error::Index { .. })));
        let mut short = p.streams().to_vec();
        short.pop();
        assert!(matches!(
            reassemble(&h, &t, p.index().to_vec(), short),
            Err(Kx7Error::Truncated { what: "streams", .. })
        ));
        // A stream boundary moved by one alignment unit: both neighbours misdecode.
        let moved = index(&|i| i[3] += 16).unwrap();
        assert!(matches!(
            decode(&moved),
            Err(Kx7Error::StreamLength { .. } | Kx7Error::InvalidCode { .. } | Kx7Error::Padding { .. })
        ));
    }

    #[test]
    fn kx7_detects_corrupt_streams() {
        let w = fp8_born(32, 1024, 41);
        let p = encode_matrix(&w, 32, 1024, Kx7Layout::frag32(2)).unwrap();
        let rebuilt = |streams: Vec<u8>| {
            Kx7Pack::from_parts(&p.header().to_words(), &p.table(), p.index().to_vec(), streams).unwrap()
        };
        // Nonzero padding after a stream's last symbol.
        let pad = (0..p.header().streams()).find(|&s| {
            let mut bits = 0u64;
            let mut pos = Vec::new();
            p.header().layout.positions(32, 1024, s, &mut pos);
            for &(r, c) in &pos {
                bits += p.code().lengths()[w[r * 1024 + c] as usize] as u64;
            }
            bits.div_ceil(8) < (p.index()[s + 1] - p.index()[s]) as u64
        });
        let s = pad.expect("a stream with a padding byte");
        let mut streams = p.streams().to_vec();
        streams[p.index()[s + 1] as usize - 1] = 0x80;
        assert_eq!(decode(&rebuilt(streams)), Err(Kx7Error::Padding { stream: s }));
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
                other => panic!("flip at byte {at}: {other:?}"),
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

    // ------------------------------------------------------------ real tensors

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
                let header: serde_json::Value = serde_json::from_slice(&json).unwrap();
                for (name, meta) in header.as_object().unwrap() {
                    if name == "__metadata__" || meta["dtype"] != "BF16" {
                        continue;
                    }
                    let shape =
                        meta["shape"].as_array().unwrap().iter().map(|d| d.as_u64().unwrap() as usize).collect();
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

    #[test]
    #[ignore = "reads GLM-5.3-Flash safetensors shards from CUTEAFD_KX7_SHARDS"]
    fn kx7_real_glm_tensors() {
        let dir =
            std::env::var("CUTEAFD_KX7_SHARDS").expect("set CUTEAFD_KX7_SHARDS to a GLM-5.3-Flash shard directory");
        let layer: usize = std::env::var("CUTEAFD_KX7_LAYER").ok().and_then(|l| l.parse().ok()).unwrap_or(10);
        let shards = Shards::open(&dir);
        let name = |p: &str| format!("model.language_model.layers.{layer}.self_attn.{p}_proj.weight");
        let (q, k, v, o) =
            (shards.load(&name("q")), shards.load(&name("k")), shards.load(&name("v")), shards.load(&name("o")));
        let one = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        let threads = rayon::current_num_threads();
        type Case<'a> = (&'a str, Vec<(&'a [u16], usize)>, usize, Kx7Layout);
        let cases: Vec<Case> = vec![
            ("k_proj", vec![(k.0.as_slice(), k.1)], k.2, Kx7Layout::frag32(default_slices(k.1, k.2))),
            ("o_proj", vec![(o.0.as_slice(), o.1)], o.2, Kx7Layout::frag32(default_slices(o.1, o.2))),
            ("k_proj", vec![(k.0.as_slice(), k.1)], k.2, Kx7Layout::lane16(4)),
            ("o_proj", vec![(o.0.as_slice(), o.1)], o.2, Kx7Layout::lane16(4)),
            (
                "q;k;v",
                vec![(q.0.as_slice(), q.1), (k.0.as_slice(), k.1), (v.0.as_slice(), v.1)],
                k.2,
                Kx7Layout::lane16(4),
            ),
            (
                "q;k;v",
                vec![(q.0.as_slice(), q.1), (k.0.as_slice(), k.1), (v.0.as_slice(), v.1)],
                k.2,
                Kx7Layout::frag32(default_slices(q.1 + k.1 + v.1, k.2)),
            ),
        ];
        println!("layer {layer}, {threads} threads");
        for (label, parts, kk, layout) in cases {
            let src = Kx7Source::fused(&parts, kk).unwrap();
            let weights = (src.n() * kk) as f64;
            let hist = histogram(&src);
            let code = CanonicalCode::from_histogram(&hist).unwrap();
            let mass = |min: u8| {
                hist.iter().zip(code.lengths()).filter(|&(_, &l)| l > min).map(|(&f, _)| f).sum::<u64>() as f64
                    / weights
            };
            let rate = hist.iter().zip(code.lengths()).map(|(&f, &l)| f * l as u64).sum::<u64>() as f64 / weights;
            let entropy: f64 = hist
                .iter()
                .filter(|&&f| f > 0)
                .map(|&f| {
                    let p = f as f64 / weights;
                    -p * p.log2()
                })
                .sum();
            let (pack1, enc1) = seconds(|| one.install(|| encode(&src, layout).unwrap()));
            let (pack, enc) = seconds(|| encode(&src, layout).unwrap());
            assert_eq!((pack1.streams(), pack1.index()), (pack.streams(), pack.index()));
            let (out1, dec1) = seconds(|| one.install(|| decode(&pack).unwrap()));
            let (_, dec) = seconds(|| decode(&pack).unwrap());
            for (r, row) in src.rows.iter().enumerate() {
                assert_eq!(*row, &out1[r * kk..(r + 1) * kk], "{label} row {r}");
            }
            verify(&pack, &src).unwrap();
            let h = pack.header();
            let stream_bpw = pack.streams().len() as f64 * 8.0 / weights;
            let index_bpw = pack.index().len() as f64 * 32.0 / weights;
            let table_bpw = (4 * HEADER_WORDS + 2 * pack.table().len()) as f64 * 8.0 / weights;
            println!(
                "{label} [{}, {kk}] {:?} S={} align={}: {} streams x {} symbols, n_sym {}, H {entropy:.4}, \
                 code rate {rate:.4}; bits/weight stream {stream_bpw:.4} + index {index_bpw:.4} + header/table {table_bpw:.5} = {:.4}; \
                 symbol mass with code > 11 bits {:.4}, > 12 bits {:.4}; decode tables {} B; \
                 encode {enc1:.2} s (1 thread) / {enc:.3} s ({threads}); decode {dec1:.2} s / {dec:.3} s",
                h.n, layout.kind, layout.slices, layout.align, h.streams(), h.symbols_per_stream(), h.n_sym,
                pack.stored_bytes() as f64 * 8.0 / weights, mass(11), mass(12), 4 * pack.decode_tables().len(),
            );
            if layout.kind == LayoutKind::Lane16 {
                println!(
                    "  sha256 table.bin {} streams.bin {} offsets.bin {}",
                    sha(&le16(&pack.table())),
                    sha(pack.streams()),
                    sha(&le32(pack.index()))
                );
            }
        }
    }
}
