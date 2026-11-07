//! KX7: an exact (lossless) code for BF16 matrices whose values were born FP8, the storage
//! of GLM-5.3-Flash's KDA `q/k/v/o` projections. Pure CPU: the packer, the decode tables the
//! reference kernels load, and a scalar decoder that walks every stream the way one GPU lane
//! does, so it is the executable reference for the kernels. Format v1.1, the one the GEMV
//! reads, is [`V11_LAYOUT`] with [`V11_CODE`]; its file set is byte-identical to the KX7
//! reference tool's (`kdaexact.py --layout lane16 --ksplit 4 --interleave 8 --ragged --code
//! twoclass --pbits 11 --dump`).
//!
//! Codes ([`CodeKind`]); each 16-bit BF16 pattern is one symbol, bits are MSB-first, codes
//! canonical (symbols in (length, value) order, first code 0, next = (prev + 1) << (len -
//! prevlen)), and every tie is broken as the tool breaks it (heap ties by symbol rank then
//! node creation order; package-merge by a stable sort on weight, packages ahead of leaves):
//!
//! * `Huffman`: one code over all patterns, the unlimited Huffman lengths when the longest
//!   is at most 16 bits, else package-merge limited to 16.
//! * `TwoClass { pbits, rare_bits }`: patterns whose unlimited Huffman length exceeds pbits
//!   are rare (the most frequent rare ones promoted until at most 2^rare_bits remain); one
//!   code limited to pbits covers the frequent patterns and an ESC symbol weighted by the
//!   rare mass (ESC sorts after every pattern); a rare pattern is ESC's code followed by
//!   rare_bits raw bits, its rank among the rare patterns by value. Every code fits 16 bits.
//!
//! Layouts ([`Kx7Layout`]): the core never looks past each stream's ordered (row, col) list:
//!
//! * `lane16(S)`: a row holds 4S streams; stream s*4 + t of row r covers inputs
//!   s*K/S + 64c + 16t + i, c over the slice's 64-wide chunks, i = 0..15; stream id
//!   r * 4S + s*4 + t.
//! * `frag32(S)`: one stream per (slice s, 16-row tile t, lane l of 32), id (s * n/16 + t) *
//!   32 + l; per 16-wide K step j it holds lane l's `mma.m16n8k16` A fragment, g = l / 4,
//!   c = 2 (l % 4), r0 = 16t + g, r1 = r0 + 8, k0 = 16j: W[r0][k0+c], W[r0][k0+c+1],
//!   W[r1][k0+c], W[r1][k0+c+1], W[r0][k0+c+8], W[r0][k0+c+9], W[r1][k0+c+8], W[r1][k0+c+9].
//!
//! Placement ([`Placement`]): every stream's bytes are its bits zero-padded to a whole byte
//! (its length, `lengths.bin`). `Contiguous { align }` concatenates streams in id order, each
//! padded to `align`. `Interleaved { ragged }` (lane16): the 32 streams of each 8-row block
//! and slice, lane l = 4 (row % 8) + t, are cut into 16-byte quads (zero-padded to a whole
//! quad); quad q of lane l sits at group_base + (q * 32 + l) * 16 for q below the group's
//! shortest stream in quads (ragged) or its longest (padded, shorter streams zero-filled);
//! ragged, each lane's remaining quads then follow contiguously in lane order. Groups follow
//! in (block, slice) order. `offsets.bin` holds each stream's quad 0 address and the total.
//!
//! A lane may read up to [`READ_AHEAD_BYTES`] past the end of its stream; the reference
//! decoder reads them as zeros, and a device copy needs that many zero bytes after the
//! streams.
use rayon::prelude::*;
use serde_json::{json, Value};
use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap};
use std::path::Path;
use thiserror::Error;

pub const VERSION: u32 = 1;
/// Longest code, bits.
pub const MAXLEN: u32 = 16;
/// Most source tensors one pack can fuse.
pub const MAX_PARTS: usize = 8;
/// Bytes a lane may read past the end of its stream.
pub const READ_AHEAD_BYTES: usize = 16;
/// The lane16 K split the reference kernels use.
pub const LANE16_DEFAULT_SLICES: usize = 4;
/// u16 counts (code lengths 1..=16) at the head of `table.bin`.
pub const TABLE_COUNTS: usize = 16;
/// A stream quad, bytes.
pub const QUAD: usize = 16;
/// Rows of one interleaved group: 8 rows x 4 lanes = the 32 lanes of a warp.
pub const INTERLEAVE_ROWS: usize = 8;
/// Distance between consecutive interleaved quads of one stream.
pub const QUAD_STRIDE: usize = 32 * QUAD;
/// The two-class code's escape symbol (sorts after every 16-bit pattern).
pub const ESC: u32 = 1 << 16;
/// ESC's slot in `table.bin` (the header's `esc_index` names the slot).
pub const ESC_SLOT: u16 = 0xFFFF;
/// Primary table bits of the reference kernels (`KX7_PBITS`).
pub const DEVICE_PBITS: u32 = 11;
/// Low byte of a Huffman node entry; byte 1 of a two-class ESC entry.
pub const ENTRY_NODE: u32 = 0xFF;
/// Format v1.1: lane16, K split 4, 8-row ragged interleave.
pub const V11_LAYOUT: Kx7Layout =
    Kx7Layout { kind: LayoutKind::Lane16, slices: 4, placement: Placement::Interleaved { ragged: true } };
/// Format v1.1: the two-class code with an 11-bit primary and 12 rare bits.
pub const V11_CODE: CodeKind = CodeKind::TwoClass { pbits: 11, rare_bits: 12 };

const PATTERNS: usize = 1 << 16;
/// Streams one contiguous packing task writes.
const ENCODE_GROUP: usize = 64;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum Kx7Error {
    #[error("KX7 {layout:?} S={slices} cannot hold [{n}, {k}]: {reason}")]
    Shape { n: usize, k: usize, layout: LayoutKind, slices: usize, reason: &'static str },
    #[error("KX7 source: {reason} (expected {expected}, found {found})")]
    Source { reason: &'static str, expected: usize, found: usize },
    #[error("KX7 code: {reason}")]
    Code { reason: String },
    #[error("KX7 table: {reason}")]
    Table { reason: String },
    #[error("KX7 header {field}: {reason}")]
    Header { field: &'static str, reason: String },
    #[error("KX7 {what} truncated: needs {need}, has {have}")]
    Truncated { what: &'static str, need: usize, have: usize },
    #[error("KX7 offset {entry} is {found}: {reason}")]
    Index { entry: usize, found: u64, reason: &'static str },
    #[error("KX7 streams total {bytes} bytes, past the u32 offset range")]
    StreamsTooLarge { bytes: u64 },
    #[error("KX7 stream {stream} takes {bytes} bytes, past the u16 length range")]
    StreamTooLong { stream: usize, bytes: u64 },
    #[error("KX7 stream {stream}: pattern {pattern:#06x} has no codeword")]
    Uncoded { stream: usize, pattern: u16 },
    #[error("KX7 stream {stream}, symbol {symbol}: the bits match no codeword")]
    InvalidCode { stream: usize, symbol: usize },
    #[error("KX7 stream {stream} reads past its bytes and their {READ_AHEAD_BYTES}-byte read-ahead")]
    Overrun { stream: usize },
    #[error("KX7 stream {stream}: its symbols take {expected} bytes, lengths.bin gives {actual}")]
    StreamLength { stream: usize, expected: usize, actual: usize },
    #[error("KX7 stream {stream}: padding after the last symbol is not zero")]
    Padding { stream: usize },
    #[error("KX7 verification failed at [{row}, {col}]: source {expected:#06x}, decoded {actual:#06x}")]
    Mismatch { row: usize, col: usize, expected: u16, actual: u16 },
    #[error("KX7 {path}: {reason}")]
    Io { path: String, reason: String },
}

type Result<T> = std::result::Result<T, Kx7Error>;

// ---------------------------------------------------------------- layouts

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayoutKind {
    Frag32,
    Lane16,
}

impl LayoutKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::Frag32 => "frag32",
            Self::Lane16 => "lane16",
        }
    }
}

/// Where stream bytes go (module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Placement {
    /// Streams in id order, each padded to `align` bytes (a power of two up to 256).
    Contiguous { align: usize },
    /// lane16 only: 8-row x slice groups of 32 quad-interleaved streams.
    Interleaved { ragged: bool },
}

/// Which positions each stream holds and where its bytes go (module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Kx7Layout {
    pub kind: LayoutKind,
    /// K slices S.
    pub slices: usize,
    pub placement: Placement,
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
    pub const fn frag32(slices: usize) -> Self {
        Self { kind: LayoutKind::Frag32, slices, placement: Placement::Contiguous { align: 16 } }
    }
    pub const fn lane16(slices: usize) -> Self {
        Self { kind: LayoutKind::Lane16, slices, placement: Placement::Contiguous { align: 1 } }
    }
    pub const fn with_align(self, align: usize) -> Self {
        Self { placement: Placement::Contiguous { align }, ..self }
    }
    pub const fn interleaved(self, ragged: bool) -> Self {
        Self { placement: Placement::Interleaved { ragged }, ..self }
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
        match self.placement {
            Placement::Contiguous { align } if !align.is_power_of_two() || align > 256 => {
                return Err(self.shape_error(n, k, "needs the alignment a power of two up to 256"))
            }
            Placement::Interleaved { .. } if self.kind != LayoutKind::Lane16 => {
                return Err(self.shape_error(n, k, "interleaving needs lane16"))
            }
            Placement::Interleaved { .. } if n % INTERLEAVE_ROWS != 0 => {
                return Err(self.shape_error(n, k, "interleaving needs n % 8 == 0"))
            }
            _ => {}
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

    /// The streams of band `band`, in id order.
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

    /// Interleaved placement: the 32 streams of group `g` ((block, slice) order) by lane.
    fn group_streams(&self, g: usize) -> [usize; 32] {
        let spr = 4 * self.slices;
        let (block, slice) = (g / self.slices, g % self.slices);
        std::array::from_fn(|l| (INTERLEAVE_ROWS * block + l / 4) * spr + slice * 4 + l % 4)
    }

    /// Interleaved placement: the group of stream `stream` and its lane.
    fn group_of(&self, stream: usize) -> (usize, usize) {
        let spr = 4 * self.slices;
        let (row, w) = (stream / spr, stream % spr);
        ((row / INTERLEAVE_ROWS) * self.slices + w / 4, 4 * (row % INTERLEAVE_ROWS) + w % 4)
    }
}

fn quads(len: u16) -> usize {
    (len as usize).div_ceil(QUAD)
}

/// Where the bytes of one stream sit: quad q at `first + q * stride` for q < `interleaved`,
/// else `tail + (q - interleaved) * 16`; `span` bytes in all (its padded extent).
struct Extent {
    first: usize,
    stride: usize,
    interleaved: usize,
    tail: usize,
    span: usize,
}

impl Extent {
    fn quad(&self, q: usize) -> usize {
        if q < self.interleaved {
            self.first + q * self.stride
        } else {
            self.tail + (q - self.interleaved) * QUAD
        }
    }
}

/// Stream byte offsets (its first byte, or quad 0, then the total) from the stream lengths.
fn place(layout: &Kx7Layout, n: usize, k: usize, lengths: &[u16]) -> Result<Vec<u32>> {
    let ns = layout.streams(n, k);
    let mut offsets = vec![0u32; ns + 1];
    let mut total = 0u64;
    let fits = |t: u64| if t > u32::MAX as u64 { Err(Kx7Error::StreamsTooLarge { bytes: t }) } else { Ok(t as u32) };
    match layout.placement {
        Placement::Contiguous { align } => {
            for (i, &len) in lengths.iter().enumerate() {
                offsets[i] = fits(total)?;
                total += (len as u64).next_multiple_of(align as u64);
            }
        }
        Placement::Interleaved { ragged } => {
            for g in 0..(n / INTERLEAVE_ROWS) * layout.slices {
                let ids = layout.group_streams(g);
                let nq = ids.map(|i| quads(lengths[i]));
                let most = *nq.iter().max().expect("32 lanes");
                let rounds = if ragged { *nq.iter().min().expect("32 lanes") } else { most };
                for (l, &i) in ids.iter().enumerate() {
                    offsets[i] = fits(total + (l * QUAD) as u64)?;
                }
                let tails: usize = nq.iter().map(|&q| q.saturating_sub(rounds)).sum();
                total += (rounds * QUAD_STRIDE + tails * QUAD) as u64;
            }
        }
    }
    offsets[ns] = fits(total)?;
    Ok(offsets)
}

/// The extent of stream `stream` under validated offsets and lengths.
fn extent(layout: &Kx7Layout, offsets: &[u32], lengths: &[u16], stream: usize) -> Extent {
    let first = offsets[stream] as usize;
    match layout.placement {
        Placement::Contiguous { align } => {
            let span = (lengths[stream] as usize).next_multiple_of(align);
            Extent { first, stride: QUAD, interleaved: usize::MAX, tail: 0, span }
        }
        Placement::Interleaved { ragged } => {
            let (g, lane) = layout.group_of(stream);
            let nq = layout.group_streams(g).map(|i| quads(lengths[i]));
            let own = nq[lane];
            if !ragged {
                let rounds = *nq.iter().max().expect("32 lanes");
                return Extent { first, stride: QUAD_STRIDE, interleaved: rounds, tail: 0, span: rounds * QUAD };
            }
            let rounds = *nq.iter().min().expect("32 lanes");
            let before: usize = nq[..lane].iter().map(|&q| q - rounds).sum();
            let tail = first - lane * QUAD + rounds * QUAD_STRIDE + before * QUAD;
            Extent { first, stride: QUAD_STRIDE, interleaved: rounds, tail, span: own * QUAD }
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

/// Unlimited Huffman depths of `freqs` (in symbol order), as the tool's `_huffman_lengths`:
/// a min-heap on (weight, id), leaves numbered by position, merged nodes from n upward.
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

/// Length-limited depths by package-merge, as the tool's `package_merge`: `items` are
/// (weight, symbol) sorted ascending; each level is the stable sort by weight of the previous
/// level's pairs followed by the leaves, so packages precede leaves on ties.
fn package_merge(items: &[(u64, u32)], maxlen: u32) -> Result<Vec<u32>> {
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

/// Code lengths of (weight, symbol) items listed in ascending symbol order, limited to
/// `maxlen`: unlimited Huffman when it fits, else package-merge.
fn limited_lengths(items: &[(u64, u32)], maxlen: u32) -> Result<Vec<u32>> {
    let freqs: Vec<u64> = items.iter().map(|&(f, _)| f).collect();
    let depths = huffman_depths(&freqs);
    if depths.iter().all(|&d| d <= maxlen) {
        return Ok(depths);
    }
    let mut sorted: Vec<(u64, u32, usize)> = items.iter().enumerate().map(|(i, &(f, s))| (f, s, i)).collect();
    sorted.sort_unstable();
    let merged = package_merge(&sorted.iter().map(|&(f, s, _)| (f, s)).collect::<Vec<_>>(), maxlen)?;
    let mut out = vec![0u32; items.len()];
    for (&(_, _, i), &d) in sorted.iter().zip(&merged) {
        out[i] = d;
    }
    Ok(out)
}

fn used_patterns(hist: &[u64]) -> Result<Vec<(u64, u32)>> {
    if hist.len() != PATTERNS {
        return Err(Kx7Error::Code { reason: format!("histogram has {} bins, not 65536", hist.len()) });
    }
    let items: Vec<(u64, u32)> = (0..PATTERNS).filter(|&s| hist[s] > 0).map(|s| (hist[s], s as u32)).collect();
    if items.is_empty() {
        return Err(Kx7Error::Code { reason: "empty histogram".into() });
    }
    Ok(items)
}

/// Huffman code lengths per pattern (65,536 entries, 0 = absent), at most [`MAXLEN`]: the
/// tool's `length_limited_lengths(f, 16)`, tie for tie.
pub fn code_lengths(hist: &[u64]) -> Result<Vec<u8>> {
    let items = used_patterns(hist)?;
    let mut lengths = vec![0u8; PATTERNS];
    for (&(_, s), d) in items.iter().zip(limited_lengths(&items, MAXLEN)?) {
        lengths[s as usize] = d as u8;
    }
    Ok(lengths)
}

// ---------------------------------------------------------------- codes

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodeKind {
    Huffman,
    TwoClass { pbits: u32, rare_bits: u32 },
}

impl CodeKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::Huffman => "huffman",
            Self::TwoClass { .. } => "twoclass",
        }
    }
}

/// One pack's code: the canonical code over its table symbols (ESC among them for the
/// two-class code) and, two-class, the rare patterns in rank order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Kx7Code {
    kind: CodeKind,
    /// (length, symbol) in canonical order; ESC is [`ESC`].
    order: Vec<(u32, u32)>,
    rare: Vec<u16>,
    esc_index: Option<usize>,
    esc_len: u32,
    /// Per pattern: `code << 8 | length` (0 = absent), rare patterns included.
    codewords: Vec<u32>,
}

impl Kx7Code {
    /// The code of a histogram.
    pub fn build(hist: &[u64], kind: CodeKind) -> Result<Self> {
        let items = used_patterns(hist)?;
        let code_err = |reason: String| Kx7Error::Code { reason };
        match kind {
            CodeKind::Huffman => {
                let lengths = limited_lengths(&items, MAXLEN)?;
                let mut order: Vec<(u32, u32)> = items.iter().zip(lengths).map(|(&(_, s), l)| (l, s)).collect();
                order.sort_unstable();
                Self::assemble(kind, order, Vec::new()).map_err(code_err)
            }
            CodeKind::TwoClass { pbits, rare_bits } => {
                check_two_class(pbits, rare_bits).map_err(code_err)?;
                // The tool's twoclass_code: rare = unlimited length > pbits, ordered by
                // (-weight, pattern); the most frequent are promoted until 2^rare_bits remain.
                let depths = huffman_depths(&items.iter().map(|&(f, _)| f).collect::<Vec<_>>());
                let mut rare: Vec<(u64, u32)> =
                    items.iter().zip(&depths).filter(|&(_, &d)| d > pbits).map(|(&it, _)| it).collect();
                rare.sort_unstable_by_key(|&(f, s)| (Reverse(f), s));
                let keep = rare.len().min(1 << rare_bits);
                rare.drain(..rare.len() - keep);
                let mut is_rare = vec![false; PATTERNS];
                rare.iter().for_each(|&(_, s)| is_rare[s as usize] = true);
                let mut coded: Vec<(u64, u32)> = items.iter().copied().filter(|&(_, s)| !is_rare[s as usize]).collect();
                if !rare.is_empty() {
                    coded.push((rare.iter().map(|&(f, _)| f).sum(), ESC));
                }
                let lengths = limited_lengths(&coded, pbits)?;
                let mut order: Vec<(u32, u32)> = coded.iter().zip(lengths).map(|(&(_, s), l)| (l, s)).collect();
                order.sort_unstable();
                let mut rare: Vec<u16> = rare.iter().map(|&(_, s)| s as u16).collect();
                rare.sort_unstable();
                Self::assemble(kind, order, rare).map_err(code_err)
            }
        }
    }

    /// From `table.bin` (u16 counts for lengths 1..=16, then the symbols in canonical order,
    /// [`ESC_SLOT`] at `esc_index`) and `rare.bin` (two-class: patterns in rank order). A u16
    /// count wraps: the one Huffman code with all 65,536 patterns at 16 bits stores 0.
    pub fn from_tables(kind: CodeKind, table: &[u16], rare: &[u16], esc_index: Option<usize>) -> Result<Self> {
        let bad = |reason: String| Kx7Error::Table { reason };
        if table.len() < TABLE_COUNTS {
            return Err(Kx7Error::Truncated { what: "code table", need: TABLE_COUNTS, have: table.len() });
        }
        let symbols = &table[TABLE_COUNTS..];
        let mut counts: Vec<usize> = table[..TABLE_COUNTS].iter().map(|&c| c as usize).collect();
        if symbols.len() == PATTERNS && counts.iter().all(|&c| c == 0) {
            counts[MAXLEN as usize - 1] = PATTERNS;
        }
        let total: usize = counts.iter().sum();
        if total != symbols.len() {
            return Err(bad(format!("counts sum to {total}, the table holds {} symbols", symbols.len())));
        }
        if let Some(e) = esc_index {
            if kind == CodeKind::Huffman || e >= symbols.len() || symbols[e] != ESC_SLOT {
                return Err(bad(format!("esc_index {e} names no ESC slot")));
            }
        }
        let mut order = Vec::with_capacity(total);
        for (l, &c) in counts.iter().enumerate() {
            for _ in 0..c {
                let i = order.len();
                order.push((l as u32 + 1, if esc_index == Some(i) { ESC } else { symbols[i] as u32 }));
            }
        }
        Self::assemble(kind, order, rare.to_vec()).map_err(bad)
    }

    /// Validates a canonical order (and rare list) and derives every codeword.
    fn assemble(kind: CodeKind, order: Vec<(u32, u32)>, rare: Vec<u16>) -> std::result::Result<Self, String> {
        let limit = match kind {
            CodeKind::Huffman => MAXLEN,
            CodeKind::TwoClass { pbits, rare_bits } => {
                check_two_class(pbits, rare_bits)?;
                pbits
            }
        };
        if order.is_empty() {
            return Err("no symbols".into());
        }
        if order.len() == 1 && order[0].0 != 1 {
            return Err("a one-symbol code must be one bit long".into());
        }
        for w in order.windows(2) {
            if w[0] >= w[1] {
                return Err(format!("symbols {:#x}, {:#x} are not in (length, value) order", w[0].1, w[1].1));
            }
        }
        if let Some(&(l, s)) = order.iter().find(|&&(l, s)| l == 0 || l > limit || s > ESC) {
            return Err(format!("symbol {s:#x} takes {l} bits (1 to {limit} allowed)"));
        }
        let kraft: u64 = order.iter().map(|&(l, _)| 1u64 << (MAXLEN - l)).sum();
        if kraft > 1 << MAXLEN {
            return Err(format!("Kraft sum {kraft}/65536 exceeds 1"));
        }
        let esc_index = order.iter().position(|&(_, s)| s == ESC);
        let mut codewords = vec![0u32; PATTERNS];
        let (mut code, mut prev, mut esc) = (0u32, 0u32, (0u32, 0u32));
        for &(l, s) in &order {
            code <<= l - prev;
            if s == ESC {
                esc = (code, l);
            } else if codewords[s as usize] != 0 {
                return Err(format!("pattern {s:#06x} appears twice"));
            } else {
                codewords[s as usize] = code << 8 | l;
            }
            code += 1;
            prev = l;
        }
        match kind {
            CodeKind::Huffman if esc_index.is_some() || !rare.is_empty() => {
                return Err("a Huffman code has no ESC or rare symbols".into())
            }
            CodeKind::TwoClass { rare_bits, .. } => {
                if esc_index.is_some() == rare.is_empty() {
                    return Err("ESC must be coded exactly when rare symbols exist".into());
                }
                if rare.len() > 1 << rare_bits {
                    return Err(format!("{} rare symbols exceed 2^{rare_bits}", rare.len()));
                }
                if !rare.is_empty() && esc.1 + rare_bits > MAXLEN {
                    return Err(format!("ESC code {} bits + {rare_bits} raw bits exceeds 16", esc.1));
                }
                for (rank, &s) in rare.iter().enumerate() {
                    if rank > 0 && rare[rank - 1] >= s {
                        return Err(format!("rare patterns {:#06x}, {s:#06x} are not ascending", rare[rank - 1]));
                    }
                    if codewords[s as usize] != 0 {
                        return Err(format!("pattern {s:#06x} is both frequent and rare"));
                    }
                    codewords[s as usize] = ((esc.0 << rare_bits) | rank as u32) << 8 | (esc.1 + rare_bits);
                }
            }
            _ => {}
        }
        Ok(Self { kind, order, rare, esc_index, esc_len: esc.1, codewords })
    }

    pub fn kind(&self) -> CodeKind {
        self.kind
    }
    /// Table symbols (ESC included): the header's `n_sym`.
    pub fn n_sym(&self) -> usize {
        self.order.len()
    }
    pub fn n_frequent(&self) -> usize {
        self.order.len() - usize::from(self.esc_index.is_some())
    }
    pub fn n_rare(&self) -> usize {
        self.rare.len()
    }
    /// ESC's position in the canonical order.
    pub fn esc_index(&self) -> Option<usize> {
        self.esc_index
    }
    /// ESC's code length (0 without rare symbols).
    pub fn esc_len(&self) -> u32 {
        self.esc_len
    }
    /// Rare patterns in rank order.
    pub fn rare(&self) -> &[u16] {
        &self.rare
    }
    /// Per pattern: `code << 8 | length` (0 = absent).
    pub fn codewords(&self) -> &[u32] {
        &self.codewords
    }
    /// Per pattern code length in bits (0 = absent), rare patterns at ESC + raw bits.
    pub fn length(&self, pattern: u16) -> u32 {
        self.codewords[pattern as usize] & 0xff
    }
    /// Patterns with a codeword (frequent and rare).
    pub fn coded_patterns(&self) -> usize {
        self.codewords.iter().filter(|&&c| c != 0).count()
    }
    /// The header's `maxlen`: 16, or ESC + rare bits for the two-class code (as the tool).
    pub fn maxlen(&self) -> u32 {
        match self.kind {
            CodeKind::Huffman => MAXLEN,
            CodeKind::TwoClass { rare_bits, .. } => self.esc_len + rare_bits,
        }
    }

    /// `table.bin`: u16 counts for lengths 1..=16 (wrapping), then the symbols in canonical
    /// order with [`ESC_SLOT`] for ESC.
    pub fn table(&self) -> Vec<u16> {
        let mut t = vec![0u16; TABLE_COUNTS];
        for &(l, _) in &self.order {
            t[l as usize - 1] = t[l as usize - 1].wrapping_add(1);
        }
        t.extend(self.order.iter().map(|&(_, s)| if s == ESC { ESC_SLOT } else { s as u16 }));
        t
    }

    /// The decode tables of the reference kernels (`build_lut`), as u32 words for shared
    /// memory. Huffman: 2^pbits primary entries, a leaf `pattern << 16 | len` or, for a prefix
    /// of longer codes, a node `index << 16 | (32 - bits) << 8 | 0xFF` naming a sub-table of
    /// 2^bits leaves (bits = longest code under the prefix - pbits) at word `index`; sub-tables
    /// follow in prefix order. Two-class: 2^pbits entries, a leaf or, for ESC,
    /// `0xFF << 8 | (esc_len + rare_bits)`, then 2^rare_bits u16 rare patterns by rank (zero
    /// padded), two to a word. Unused entries are 0.
    pub fn device_tables(&self, pbits: u32) -> Result<Vec<u32>> {
        let bad = |reason: String| Kx7Error::Table { reason };
        if pbits == 0 || pbits > MAXLEN {
            return Err(bad(format!("{pbits} primary bits")));
        }
        let primary = 1usize << pbits;
        let mut words = vec![0u32; primary];
        let mut code = 0u32;
        let mut prev = 0u32;
        let mut long: Vec<(u32, u32, u32, u32)> = Vec::new();
        for &(l, s) in &self.order {
            code <<= l - prev;
            prev = l;
            let c = code;
            code += 1;
            if l > pbits {
                if s == ESC || matches!(self.kind, CodeKind::TwoClass { .. }) {
                    return Err(bad(format!("a two-class code of {l} bits exceeds {pbits} primary bits")));
                }
                long.push((c >> (l - pbits), l, s, c));
                continue;
            }
            let entry = match (self.kind, s == ESC) {
                (CodeKind::TwoClass { rare_bits, .. }, true) => ENTRY_NODE << 8 | (l + rare_bits),
                _ => s << 16 | l,
            };
            words[(c << (pbits - l)) as usize..((c + 1) << (pbits - l)) as usize].fill(entry);
        }
        match self.kind {
            CodeKind::TwoClass { rare_bits, .. } => {
                let start = words.len();
                words.resize(start + (1usize << rare_bits).div_ceil(2), 0);
                for (i, &r) in self.rare.iter().enumerate() {
                    words[start + i / 2] |= (r as u32) << (16 * (i % 2));
                }
            }
            CodeKind::Huffman => {
                let mut longest: BTreeMap<u32, u32> = BTreeMap::new();
                for &(p, l, _, _) in &long {
                    let e = longest.entry(p).or_insert(l);
                    *e = (*e).max(l);
                }
                let mut base: BTreeMap<u32, usize> = BTreeMap::new();
                for (&p, &l) in &longest {
                    let index = words.len();
                    if index >= PATTERNS {
                        return Err(bad("sub-tables pass the 16-bit entry index".into()));
                    }
                    base.insert(p, index);
                    words[p as usize] = (index as u32) << 16 | (32 - (l - pbits)) << 8 | ENTRY_NODE;
                    words.resize(index + (1 << (l - pbits)), 0);
                }
                for &(p, l, s, c) in &long {
                    let bits = longest[&p] - pbits;
                    let rem = c & ((1 << (l - pbits)) - 1);
                    let shift = bits - (l - pbits);
                    let at = base[&p];
                    words[at + (rem << shift) as usize..at + ((rem + 1) << shift) as usize].fill(s << 16 | l);
                }
            }
        }
        Ok(words)
    }
}

fn check_two_class(pbits: u32, rare_bits: u32) -> std::result::Result<(), String> {
    if pbits == 0 || pbits > MAXLEN || rare_bits == 0 || rare_bits >= MAXLEN {
        return Err(format!("two-class needs 1 <= pbits <= 16 and 1 <= rare_bits <= 15 (got {pbits}, {rare_bits})"));
    }
    Ok(())
}

// ---------------------------------------------------------------- lane decoding

/// The bit reader of one GPU lane over its stream's bytes: a 64-bit buffer whose valid bits
/// sit at the top, refilled with the next big-endian 32-bit word whenever it holds fewer than
/// 32 bits.
pub struct LaneBits<'a> {
    bytes: &'a [u8],
    next: usize,
    buf: u64,
    nbits: u32,
    consumed: u64,
}

impl<'a> LaneBits<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, next: 0, buf: 0, nbits: 0, consumed: 0 }
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

/// A decoder over [`Kx7Code::device_tables`], step for step the reference kernels' lookup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceDecoder {
    kind: CodeKind,
    pbits: u32,
    words: Vec<u32>,
    esc_len: u32,
    n_rare: usize,
}

impl DeviceDecoder {
    pub fn new(code: &Kx7Code, pbits: u32) -> Result<Self> {
        let words = code.device_tables(pbits)?;
        Ok(Self { kind: code.kind, pbits, words, esc_len: code.esc_len, n_rare: code.n_rare() })
    }

    /// The table words it decodes with.
    pub fn words(&self) -> &[u32] {
        &self.words
    }
}

impl SymbolDecoder for DeviceDecoder {
    fn next(&self, bits: &mut LaneBits<'_>) -> Option<u16> {
        let peek = bits.peek16();
        let mut e = self.words[(peek >> (MAXLEN - self.pbits)) as usize];
        let sym = match self.kind {
            CodeKind::TwoClass { rare_bits, .. } if (e >> 8) & 0xff == ENTRY_NODE => {
                let rank = (peek >> (MAXLEN - self.esc_len - rare_bits)) & ((1 << rare_bits) - 1);
                if rank as usize >= self.n_rare {
                    return None;
                }
                let w = self.words[(1 << self.pbits) + rank as usize / 2];
                (w >> (16 * (rank % 2))) as u16
            }
            CodeKind::TwoClass { .. } => (e >> 16) as u16,
            CodeKind::Huffman => {
                if e & 0xff > MAXLEN {
                    let shr = (e >> 8) & 0xff;
                    let sub = ((peek << 16) << self.pbits) >> shr;
                    e = *self.words.get(((e >> 16) + sub) as usize)?;
                }
                (e >> 16) as u16
            }
        };
        let len = e & 0xff;
        if len == 0 || len > MAXLEN {
            return None;
        }
        bits.consume(len);
        Some(sym)
    }
}

// ---------------------------------------------------------------- pack

/// What a pack codes and how it is laid out.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Kx7Header {
    pub layout: Kx7Layout,
    pub n: usize,
    pub k: usize,
    /// Exclusive end row of each fused part.
    pub part_ends: Vec<usize>,
}

impl Kx7Header {
    pub fn streams(&self) -> usize {
        self.layout.streams(self.n, self.k)
    }
    pub fn symbols_per_stream(&self) -> usize {
        self.layout.symbols_per_stream(self.k)
    }
    pub fn streams_per_row(&self) -> usize {
        self.streams() / self.n
    }

    fn check(&self) -> Result<()> {
        self.layout.check(self.n, self.k)?;
        let ends = &self.part_ends;
        if ends.is_empty() || ends.len() > MAX_PARTS || ends.windows(2).any(|w| w[0] >= w[1]) || ends[0] == 0 {
            return Err(Kx7Error::Header {
                field: "components",
                reason: format!("part ends {ends:?} must rise, 1 to 8 parts"),
            });
        }
        if *ends.last().expect("parts") != self.n {
            return Err(Kx7Error::Header {
                field: "components",
                reason: format!("the last part must end at row {}", self.n),
            });
        }
        Ok(())
    }
}

/// Names for `header.json` (the tool's `label` and component names).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Kx7Labels {
    pub label: String,
    /// One name per fused part.
    pub components: Vec<String>,
}

/// One packed matrix: its file set in memory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Kx7Pack {
    header: Kx7Header,
    code: Kx7Code,
    offsets: Vec<u32>,
    lengths: Vec<u16>,
    streams: Vec<u8>,
}

fn le16(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn le32(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn round4(x: f64) -> f64 {
    format!("{x:.4}").parse().expect("a formatted float")
}
fn round5(x: f64) -> f64 {
    format!("{x:.5}").parse().expect("a formatted float")
}

impl Kx7Pack {
    pub fn header(&self) -> &Kx7Header {
        &self.header
    }
    pub fn code(&self) -> &Kx7Code {
        &self.code
    }
    /// u32 offset of every stream (its first byte or quad 0), then the total (`offsets.bin`).
    pub fn offsets(&self) -> &[u32] {
        &self.offsets
    }
    /// u16 byte length of every stream (`lengths.bin`).
    pub fn lengths(&self) -> &[u16] {
        &self.lengths
    }
    /// `streams.bin`.
    pub fn streams(&self) -> &[u8] {
        &self.streams
    }
    pub fn table_bin(&self) -> Vec<u8> {
        le16(&self.code.table())
    }
    /// `rare.bin` (two-class codes only).
    pub fn rare_bin(&self) -> Option<Vec<u8>> {
        matches!(self.code.kind, CodeKind::TwoClass { .. }).then(|| le16(&self.code.rare))
    }
    pub fn offsets_bin(&self) -> Vec<u8> {
        le32(&self.offsets)
    }
    pub fn lengths_bin(&self) -> Vec<u8> {
        le16(&self.lengths)
    }
    /// Stream bytes beyond the streams' own lengths (alignment and interleave padding).
    pub fn padding_bytes(&self) -> usize {
        self.streams.len() - self.lengths.iter().map(|&l| l as usize).sum::<usize>()
    }
    /// Primary bits a decoder uses: the two-class code's, else [`DEVICE_PBITS`].
    pub fn device_pbits(&self) -> u32 {
        match self.code.kind {
            CodeKind::TwoClass { pbits, .. } => pbits,
            CodeKind::Huffman => DEVICE_PBITS,
        }
    }
    /// The reference kernels' decode tables for this pack.
    pub fn device_tables(&self) -> Result<Vec<u32>> {
        self.code.device_tables(self.device_pbits())
    }

    /// Bytes held: streams, u32 offsets, u16 lengths, table and rare table.
    pub fn stored_bytes(&self) -> usize {
        self.streams.len()
            + 4 * self.offsets.len()
            + 2 * self.lengths.len()
            + 2 * (TABLE_COUNTS + self.code.n_sym())
            + 2 * self.code.n_rare()
    }

    /// Reassembles and validates a pack: the offsets must be exactly the placement of the
    /// lengths, and the streams exactly as long as the placement says.
    pub fn from_parts(
        header: Kx7Header,
        code: Kx7Code,
        offsets: Vec<u32>,
        lengths: Vec<u16>,
        streams: Vec<u8>,
    ) -> Result<Self> {
        header.check()?;
        let ns = header.streams();
        if lengths.len() != ns {
            return Err(Kx7Error::Truncated { what: "stream lengths", need: ns, have: lengths.len() });
        }
        if offsets.len() != ns + 1 {
            return Err(Kx7Error::Truncated { what: "stream offsets", need: ns + 1, have: offsets.len() });
        }
        let want = place(&header.layout, header.n, header.k, &lengths)?;
        if let Some(i) = (0..=ns).find(|&i| want[i] != offsets[i]) {
            return Err(Kx7Error::Index {
                entry: i,
                found: offsets[i] as u64,
                reason: "not where the lengths place it",
            });
        }
        if streams.len() != offsets[ns] as usize {
            return Err(Kx7Error::Truncated { what: "streams", need: offsets[ns] as usize, have: streams.len() });
        }
        Ok(Self { header, code, offsets, lengths, streams })
    }

    /// `header.json`: the tool's structural and byte-accounting fields (rates rounded as it
    /// rounds them; no histogram statistics or timings).
    pub fn header_json(&self, labels: &Kx7Labels, with_source: bool) -> String {
        let (h, c) = (&self.header, &self.code);
        let weights = (h.n * h.k) as f64;
        let ns = h.streams();
        let stat_table = 2 * TABLE_COUNTS + 2 * c.coded_patterns();
        let stream_bpw = self.streams.len() as f64 * 8.0 / weights;
        let index_bpw = 16.0 * ns as f64 / weights;
        let table_bpw = stat_table as f64 * 8.0 / weights;
        let mut j = serde_json::Map::new();
        let mut put = |key: &str, v: Value| {
            j.insert(key.to_string(), v);
        };
        put("magic", json!("KX7"));
        put("version", json!(VERSION));
        put("label", json!(labels.label));
        put("shape", json!([h.n, h.k]));
        put("n", json!(h.n * h.k));
        put("chunk", json!(h.symbols_per_stream()));
        put("layout", json!(h.layout.kind.name()));
        put("ksplit", json!(h.layout.slices));
        put("maxlen", json!(c.maxlen()));
        put("code", json!(c.kind.name()));
        put("distinct_patterns", json!(c.coded_patterns()));
        put("table_bytes", json!(2 * (TABLE_COUNTS + c.n_sym())));
        if let CodeKind::TwoClass { pbits, rare_bits } = c.kind {
            put("pbits", json!(pbits));
            put("rare_bits", json!(rare_bits));
            put("esc_len", json!(c.esc_len));
            put("n_frequent", json!(c.n_frequent()));
            put("n_rare", json!(c.n_rare()));
        }
        match h.layout.placement {
            Placement::Interleaved { ragged } => {
                put("interleave_rows", json!(INTERLEAVE_ROWS));
                put("quad_stride", json!(QUAD_STRIDE));
                put("ragged", json!(ragged));
                put("padding_bytes", json!(self.padding_bytes()));
                put("padding_bpw", json!(round4(self.padding_bytes() as f64 * 8.0 / weights)));
            }
            Placement::Contiguous { align } => {
                put("stream_align", json!(align));
                put("padding_bytes", json!(self.padding_bytes()));
            }
        }
        put("stream_bytes", json!(self.streams.len()));
        put("chunks", json!(ns));
        put("stream_bpw", json!(round4(stream_bpw)));
        put("index_bpw", json!(round4(index_bpw)));
        put("table_bpw", json!(round5(table_bpw)));
        put("total_bpw", json!(round4(stream_bpw + index_bpw + table_bpw)));
        put("bytes_bf16", json!(2 * h.n * h.k));
        put("bytes_kx7", json!(self.streams.len() + 2 * ns + stat_table));
        if matches!(c.kind, CodeKind::TwoClass { .. }) {
            put("esc_index", json!(c.esc_index.map_or(-1, |e| e as i64)));
            put("rare_format", json!("u16 [n_rare] patterns in rank order; rank = the rare_bits raw bits after ESC"));
        }
        if !matches!(h.layout.placement, Placement::Interleaved { .. }) {
            put("quad_stride", json!(QUAD));
        }
        put("lengths_format", json!("u16 [N*streams_per_row] byte length of each stream (lengths.bin)"));
        put("n_sym", json!(c.n_sym()));
        put("streams_per_row", json!(h.streams_per_row()));
        put("symbols_per_stream", json!(h.symbols_per_stream()));
        let components: Vec<Value> = h
            .part_ends
            .iter()
            .enumerate()
            .map(|(i, &end)| {
                let start = if i == 0 { 0 } else { h.part_ends[i - 1] };
                json!({"name": labels.components.get(i).cloned().unwrap_or_default(), "rows": [start, end]})
            })
            .collect();
        put("components", Value::Array(components));
        let rare = matches!(c.kind, CodeKind::TwoClass { .. });
        put(
            "files",
            json!({"table": "table.bin", "streams": "streams.bin", "offsets": "offsets.bin",
                   "source": if with_source { json!("source.bf16.bin") } else { Value::Null },
                   "rare": if rare { json!("rare.bin") } else { Value::Null }, "lengths": "lengths.bin"}),
        );
        put(
            "offsets_format",
            json!(concat!(
                "u32 [N*streams_per_row + 1] byte offsets of each stream (its first quad) into streams.bin, ",
                "final = total; stream byte p at offset + (p // 16) * quad_stride + p % 16"
            )),
        );
        put("table_format", json!("u16 counts[16] for code lengths 1..16, then n_sym symbols u16 in canonical order"));
        serde_json::to_string_pretty(&Value::Object(j)).expect("JSON") + "\n"
    }

    /// Writes the file set to `dir` (created if missing); `source` adds `source.bf16.bin`.
    pub fn dump(&self, dir: &Path, labels: &Kx7Labels, source: Option<&Kx7Source<'_>>) -> Result<()> {
        let io =
            |path: &Path, e: std::io::Error| Kx7Error::Io { path: path.display().to_string(), reason: e.to_string() };
        std::fs::create_dir_all(dir).map_err(|e| io(dir, e))?;
        let write = |name: &str, bytes: &[u8]| {
            let p = dir.join(name);
            std::fs::write(&p, bytes).map_err(|e| io(&p, e))
        };
        write("header.json", self.header_json(labels, source.is_some()).as_bytes())?;
        write("table.bin", &self.table_bin())?;
        if let Some(rare) = self.rare_bin() {
            write("rare.bin", &rare)?;
        }
        write("streams.bin", &self.streams)?;
        write("offsets.bin", &self.offsets_bin())?;
        write("lengths.bin", &self.lengths_bin())?;
        if let Some(src) = source {
            write("source.bf16.bin", &src.rows.iter().flat_map(|r| le16(r)).collect::<Vec<u8>>())?;
        }
        Ok(())
    }

    /// Reads and validates a file set (this module's or the tool's `dump_pack`).
    pub fn load(dir: &Path) -> Result<Self> {
        let read = |name: &str| {
            let p = dir.join(name);
            std::fs::read(&p).map_err(|e| Kx7Error::Io { path: p.display().to_string(), reason: e.to_string() })
        };
        let u16s = |b: Vec<u8>| -> Vec<u16> { b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect() };
        let text = read("header.json")?;
        let j: Value = serde_json::from_slice(&text)
            .map_err(|e| Kx7Error::Header { field: "header.json", reason: e.to_string() })?;
        let missing = |field: &'static str| Kx7Error::Header { field, reason: "missing or not a number".into() };
        let int = |field: &'static str| j[field].as_u64().map(|v| v as usize).ok_or_else(|| missing(field));
        let shape = j["shape"].as_array().filter(|s| s.len() == 2).ok_or_else(|| missing("shape"))?;
        let (n, k) = (
            shape[0].as_u64().ok_or_else(|| missing("shape"))? as usize,
            shape[1].as_u64().ok_or_else(|| missing("shape"))? as usize,
        );
        let kind = match j["layout"].as_str() {
            Some("lane16") => LayoutKind::Lane16,
            Some("frag32") => LayoutKind::Frag32,
            other => {
                return Err(Kx7Error::Header { field: "layout", reason: format!("{other:?} is not lane16 or frag32") })
            }
        };
        let slices = int("ksplit")?;
        let placement = match (j["quad_stride"].as_u64(), j["interleave_rows"].as_u64()) {
            (Some(512), Some(8)) => Placement::Interleaved { ragged: j["ragged"].as_bool().unwrap_or(false) },
            (Some(16) | None, None | Some(0)) => Placement::Contiguous {
                align: j["stream_align"]
                    .as_u64()
                    .map_or(if kind == LayoutKind::Frag32 { 16 } else { 1 }, |a| a as usize),
            },
            (q, r) => {
                return Err(Kx7Error::Header {
                    field: "quad_stride",
                    reason: format!("{q:?} with {r:?} interleave rows"),
                })
            }
        };
        let layout = Kx7Layout { kind, slices, placement };
        let code_kind = match j["code"].as_str().unwrap_or("huffman") {
            "huffman" => CodeKind::Huffman,
            "twoclass" => CodeKind::TwoClass { pbits: int("pbits")? as u32, rare_bits: int("rare_bits")? as u32 },
            other => {
                return Err(Kx7Error::Header { field: "code", reason: format!("{other:?} is not huffman or twoclass") })
            }
        };
        let esc_index = j["esc_index"].as_i64().and_then(|e| usize::try_from(e).ok());
        let rare = match code_kind {
            CodeKind::TwoClass { .. } => u16s(read("rare.bin")?),
            CodeKind::Huffman => Vec::new(),
        };
        let code = Kx7Code::from_tables(code_kind, &u16s(read("table.bin")?), &rare, esc_index)?;
        let mut part_ends = Vec::new();
        for c in j["components"].as_array().into_iter().flatten() {
            part_ends.push(c["rows"][1].as_u64().ok_or_else(|| missing("components"))? as usize);
        }
        if part_ends.is_empty() {
            part_ends.push(n);
        }
        let header = Kx7Header { layout, n, k, part_ends };
        let offsets =
            read("offsets.bin")?.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let pack = Self::from_parts(header, code, offsets, u16s(read("lengths.bin")?), read("streams.bin")?)?;
        let agree = [
            ("n_sym", pack.code.n_sym()),
            ("streams_per_row", pack.header.streams_per_row()),
            ("symbols_per_stream", pack.header.symbols_per_stream()),
            ("stream_bytes", pack.streams.len()),
            ("maxlen", pack.code.maxlen() as usize),
        ];
        let two_class = [("n_rare", pack.code.n_rare()), ("esc_len", pack.code.esc_len as usize)];
        let checks =
            agree.iter().chain(if matches!(code_kind, CodeKind::TwoClass { .. }) { &two_class[..] } else { &[] });
        for &(field, value) in checks {
            if let Some(v) = j[field].as_u64() {
                if v as usize != value {
                    return Err(Kx7Error::Header { field, reason: format!("{v} disagrees with the files' {value}") });
                }
            }
        }
        Ok(pack)
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

fn write_stream(src: &Kx7Source<'_>, codewords: &[u32], pos: &[(usize, usize)], out: &mut [u8]) {
    let mut w = BitWriter { out, pos: 0, acc: 0, n: 0 };
    for &(r, c) in pos {
        let cw = codewords[src.rows[r][c] as usize];
        w.put(cw >> 8, cw & 0xff);
    }
    w.finish();
}

/// Packs every stream of `layout` over `src` with a per-pattern codebook (`code << 8 | len`,
/// len <= 24): returns offsets, lengths and streams. Two parallel passes, lengths then bits;
/// the output does not depend on the thread count.
pub fn pack_streams(
    src: &Kx7Source<'_>,
    layout: Kx7Layout,
    codewords: &[u32],
) -> Result<(Vec<u32>, Vec<u16>, Vec<u8>)> {
    let (n, k) = (src.n(), src.k());
    layout.check(n, k)?;
    if codewords.len() != PATTERNS {
        return Err(Kx7Error::Code { reason: format!("{} codewords, not 65536", codewords.len()) });
    }
    let ns = layout.streams(n, k);
    let lengths: Vec<u16> = (0..ns)
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
            u16::try_from(bits.div_ceil(8)).map_err(|_| Kx7Error::StreamTooLong { stream, bytes: bits.div_ceil(8) })
        })
        .collect::<Result<_>>()?;
    let offsets = place(&layout, n, k, &lengths)?;
    let mut streams = vec![0u8; offsets[ns] as usize];
    // Contiguous output regions, each holding whole streams: runs of streams, or groups.
    let regions: Vec<(usize, Vec<usize>)> = match layout.placement {
        Placement::Contiguous { .. } => (0..ns)
            .step_by(ENCODE_GROUP)
            .map(|a| (offsets[a] as usize, (a..(a + ENCODE_GROUP).min(ns)).collect()))
            .collect(),
        Placement::Interleaved { .. } => (0..(n / INTERLEAVE_ROWS) * layout.slices)
            .map(|g| {
                let ids = layout.group_streams(g);
                (offsets[ids[0]] as usize, ids.to_vec())
            })
            .collect(),
    };
    let mut slices = Vec::with_capacity(regions.len());
    let mut rest = streams.as_mut_slice();
    for (i, (start, ids)) in regions.iter().enumerate() {
        let end = regions.get(i + 1).map_or(offsets[ns] as usize, |r| r.0);
        let (head, tail) = std::mem::take(&mut rest).split_at_mut(end - start);
        slices.push((*start, ids, head));
        rest = tail;
    }
    slices.into_par_iter().for_each_init(
        || (Vec::new(), Vec::new()),
        |(pos, scratch), (base, ids, region)| {
            for &stream in ids.iter() {
                layout.positions(n, k, stream, pos);
                let e = extent(&layout, &offsets, &lengths, stream);
                match layout.placement {
                    Placement::Contiguous { .. } => {
                        let at = e.first - base;
                        write_stream(src, codewords, pos, &mut region[at..at + lengths[stream] as usize]);
                    }
                    Placement::Interleaved { .. } => {
                        let nq = quads(lengths[stream]);
                        scratch.clear();
                        scratch.resize(nq * QUAD, 0u8);
                        write_stream(src, codewords, pos, scratch);
                        for q in 0..nq {
                            let at = e.quad(q) - base;
                            region[at..at + QUAD].copy_from_slice(&scratch[q * QUAD..(q + 1) * QUAD]);
                        }
                    }
                }
            }
        },
    );
    Ok((offsets, lengths, streams))
}

/// Packs `src` in `layout` under its own code of `kind`. Refuses a code the reference
/// kernels' tables cannot hold (a Huffman code whose sub-tables pass the 16-bit entry index).
pub fn encode(src: &Kx7Source<'_>, layout: Kx7Layout, kind: CodeKind) -> Result<Kx7Pack> {
    layout.check(src.n(), src.k())?;
    let code = Kx7Code::build(&histogram(src), kind)?;
    let header = Kx7Header { layout, n: src.n(), k: src.k(), part_ends: src.part_ends.clone() };
    let mut pack = Kx7Pack { header, code, offsets: Vec::new(), lengths: Vec::new(), streams: Vec::new() };
    pack.device_tables()?;
    (pack.offsets, pack.lengths, pack.streams) = pack_streams(src, layout, pack.code.codewords())?;
    Ok(pack)
}

/// Packs one row-major [n, k] matrix.
pub fn encode_matrix(w: &[u16], n: usize, k: usize, layout: Kx7Layout, kind: CodeKind) -> Result<Kx7Pack> {
    encode(&Kx7Source::matrix(w, n, k)?, layout, kind)
}

// ---------------------------------------------------------------- decode

/// Decodes one stream as a lane does: gathers its quads (interleaved) or bytes (contiguous),
/// walks them, then checks the bits fill its length exactly and the padding is zero.
fn walk_stream<D: SymbolDecoder>(
    dec: &D,
    pack: &Kx7Pack,
    stream: usize,
    scratch: &mut Vec<u8>,
    out: &mut [u16],
) -> Result<()> {
    let layout = &pack.header.layout;
    let e = extent(layout, &pack.offsets, &pack.lengths, stream);
    let bytes: &[u8] = match layout.placement {
        Placement::Contiguous { .. } => &pack.streams[e.first..e.first + e.span],
        Placement::Interleaved { .. } => {
            scratch.clear();
            for q in 0..e.span / QUAD {
                let at = e.quad(q);
                scratch.extend_from_slice(&pack.streams[at..at + QUAD]);
            }
            scratch
        }
    };
    let mut bits = LaneBits::new(bytes);
    for (symbol, o) in out.iter_mut().enumerate() {
        if !bits.refill() {
            return Err(Kx7Error::Overrun { stream });
        }
        *o = dec.next(&mut bits).ok_or(Kx7Error::InvalidCode { stream, symbol })?;
    }
    let used = bits.consumed.div_ceil(8) as usize;
    let len = pack.lengths[stream] as usize;
    if used != len {
        return Err(Kx7Error::StreamLength { stream, expected: used, actual: len });
    }
    let tail_bits = (bits.consumed % 8) as u32;
    if (tail_bits != 0 && bytes[used - 1] & (0xffu8 >> tail_bits) != 0) || bytes[used..].iter().any(|&b| b != 0) {
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
        || (Vec::new(), Vec::new(), Vec::new(), vec![0u16; layout.symbols_per_stream(k)]),
        |(streams, pos, scratch, syms), (band, chunk)| {
            layout.band_streams(n, band, streams);
            for &stream in streams.iter() {
                walk_stream(dec, pack, stream, scratch, syms)?;
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

/// The reference decoder: the reference kernels' tables ([`Kx7Pack::device_tables`]) and
/// lookup, every stream walked from where the placement puts it.
pub fn decode(pack: &Kx7Pack) -> Result<Vec<u16>> {
    decode_with(pack, &DeviceDecoder::new(&pack.code, pack.device_pbits())?)
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

// ---------------------------------------------------------------- device image

/// The device buffers an engine holds for one pack, in bytes: the decode tables
/// ([`Kx7Pack::device_tables`], u32 words), the stream offsets (`offsets.bin`), the stream
/// lengths (`lengths.bin`) and the streams followed by [`READ_AHEAD_BYTES`] zeros. The code
/// table and rare table stay on the host (the device tables hold what a lane reads).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Kx7DeviceBytes {
    pub tables: u64,
    pub offsets: u64,
    pub lengths: u64,
    pub streams: u64,
}

/// Words of the decode tables of a `kind` code: exact for the two-class code (2^pbits primary
/// entries and 2^rare_bits u16 ranks), the most a Huffman code's sub-tables can take.
pub fn device_table_words(kind: CodeKind) -> u64 {
    match kind {
        CodeKind::TwoClass { pbits, rare_bits } => (1u64 << pbits) + (1u64 << rare_bits).div_ceil(2),
        CodeKind::Huffman => PATTERNS as u64 + (1 << (MAXLEN - DEVICE_PBITS)),
    }
}

impl Kx7DeviceBytes {
    /// The buffers of `pack`.
    pub fn of(pack: &Kx7Pack) -> Result<Self> {
        Ok(Self {
            tables: 4 * pack.device_tables()?.len() as u64,
            offsets: 4 * pack.offsets.len() as u64,
            lengths: 2 * pack.lengths.len() as u64,
            streams: (pack.streams.len() + READ_AHEAD_BYTES) as u64,
        })
    }

    /// The admission bound of an [n, k] pack in `layout` and `kind`, known before any weight is
    /// read: streams (with their padding) of at most 8 bits a weight (FP8-born BF16 takes about
    /// 7.5 in v1.1), the exact offsets and lengths and the largest decode tables. A pack above
    /// it is refused ([`Self::within`]).
    pub fn bound(n: usize, k: usize, layout: Kx7Layout, kind: CodeKind) -> Result<Self> {
        layout.check(n, k)?;
        let streams = layout.streams(n, k) as u64;
        Ok(Self {
            tables: 4 * device_table_words(kind),
            offsets: 4 * (streams + 1),
            lengths: 2 * streams,
            streams: n as u64 * k as u64 + READ_AHEAD_BYTES as u64,
        })
    }

    /// Whether every buffer fits the matching one of `bound`.
    pub fn within(&self, bound: &Self) -> bool {
        self.tables <= bound.tables
            && self.offsets <= bound.offsets
            && self.lengths <= bound.lengths
            && self.streams <= bound.streams
    }

    /// The buffers' bytes, each at least `floor` (an allocator's minimum).
    pub fn total(&self, floor: u64) -> u64 {
        [self.tables, self.offsets, self.lengths, self.streams].iter().map(|&b| b.max(floor)).sum()
    }
}

/// Bytes of a decoded [n, k] BF16 matrix in a decode workspace, rounded up to 256.
pub fn decoded_bytes(n: usize, k: usize) -> u64 {
    (n as u64 * k as u64 * 2).next_multiple_of(256)
}

/// Packs one row-major [n, k] BF16 matrix and proves the pack against it bit for bit with the
/// reference decoder (the device tables and lane walk the kernels use): a pack an engine may
/// upload.
pub fn encode_verified(w: &[u16], n: usize, k: usize, layout: Kx7Layout, kind: CodeKind) -> Result<Kx7Pack> {
    let pack = encode_matrix(w, n, k, layout, kind)?;
    verify_matrix(&pack, w)?;
    Ok(pack)
}

/// [`encode_verified`] of every (matrix, n, k) in parallel, in order; the first error wins.
pub fn encode_verified_all(
    matrices: &[(&[u16], usize, usize)],
    layout: Kx7Layout,
    kind: CodeKind,
) -> Result<Vec<Kx7Pack>> {
    matrices.par_iter().map(|&(w, n, k)| encode_verified(w, n, k, layout, kind)).collect()
}

#[cfg(test)]
mod tests;
