//! GLM-5.3 Flash weight variants on the coordinator (contract §2, §4).
//!
//! Sites (`cuteafd_loader::families::glm5_flash::variants`), each applied to the
//! BF16 sublayer output before `mhc_post_pre` mixes it into the streams:
//! - `Attn`: the attention output (`delta`) at the end of [`super::engine`]'s
//!   `attention` (per packed span for KDA);
//! - `SharedExpert` / `DenseFfn`: the shared expert's (`shared`) or dense MLP's
//!   (`delta`) output at the end of `ffn_on`, before the routed combine or reduce.
//!
//! Two operators:
//! - steering (`y -= A (Bᵀ y)`): `steer_rows` over the §4 tables;
//! - a stored delta (`y += ΔW x`, example-org L44): ΔW in FP8 on the sublayer's input
//!   (KDA: the gated-norm output, `y` in the program's scratch; FFN: the SwiGLU
//!   activation) for every row, FP32 out, then `add_rows` on the rows whose variant stores
//!   that delta. A decode step (up to [`GlmfVariants::slice_rows`] rows) runs the W8A16
//!   GEMV into its own stable buffers; a wider prefill chunk runs one whole-height W8A8 pass
//!   (ΔW read once) with its output and workspace in scratch the program has finished with
//!   ([`wide_delta_bytes`]), falling back to `slice_rows` slices only if that room is short.
//!
//! Table pointers and the per-row variant ids (`w.variants`, uploaded with the
//! step's tables) have stable addresses, so captured decode graphs keep their keys
//! and always launch at configured sites; a prefill chunk with only base rows
//! launches nothing. Base rows (`vidx == 0`) are never written. The kernels are
//! `cuteafd_ffi::steer_rows` (`SteerRows`, `AddRows`), bound (their modules loaded on the
//! coordinator device) when the sites upload, before any decode graph capture.
use crate::shared::fp8_linear::{Fp8Scales, Fp8Weight};
use crate::shared::memory::DeviceAllocation;
use anyhow::{ensure, Context, Result};
use cuteafd_core::variants::{delta_switch, site_tables, SiteKey, SiteKind, SiteOp, VariantPlan, VariantRecord};
use cuteafd_ffi::NativeLibrary;
use cuteafd_loader::families::glm5_flash::{GlmNextAttention, GlmNextConfig};
use std::collections::BTreeMap;
use std::ffi::c_void;

/// ΔW rows packed per upload chunk (a multiple of the pack kernel's 16-row tiles), so the
/// transient BF16 copy stays small.
const PACK_ROWS: usize = 256;

/// One steering site's tables in the §4 layout: `a`, `b` FP32 `[V+1][k_max][h]`, `k` i32 `[V+1]`.
struct SteerSite<'a> {
    a: DeviceAllocation<'a>,
    b: DeviceAllocation<'a>,
    k: DeviceAllocation<'a>,
    k_max: usize,
}

/// One delta site: ΔW in FP8 (per row × 128-K scales) and which variants store it.
struct DeltaSite<'a> {
    weight: Fp8Weight<'a>,
    on: DeviceAllocation<'a>,
}

/// Every configured site's device state, plus what `/v1/models` and stats report.
pub(crate) struct GlmfVariants<'a> {
    steer: BTreeMap<SiteKey, SteerSite<'a>>,
    deltas: BTreeMap<SiteKey, DeltaSite<'a>>,
    /// A delta GEMV's FP32 output `[slice_rows, hidden]` before the masked add.
    out: Option<DeviceAllocation<'a>>,
    scratch: Option<DeviceAllocation<'a>>,
    slice_rows: usize,
    hidden: usize,
    /// Table slots past base: the largest variant index.
    slots: usize,
    /// The bound kernels: steering when a site steers, the masked add when one stores a delta.
    steer_rows: Option<cuteafd_ffi::steer_rows::SteerRows<'a>>,
    add_rows: Option<cuteafd_ffi::steer_rows::AddRows<'a>>,
    pub records: Vec<VariantRecord>,
    pub plans: Vec<VariantPlan>,
}

/// Every site key some plan edits with `op`'s kind (steering or delta).
fn keys(plans: &[VariantPlan], delta: bool) -> Vec<SiteKey> {
    let mut keys: Vec<SiteKey> = plans.iter().flat_map(|p| p.sites.iter())
        .filter(|(_, op)| op.is_delta() == delta).map(|(key, _)| *key).collect();
    keys.sort();
    keys.dedup();
    keys
}

/// Each delta site's `(input width K)`, from the first plan that stores it.
fn delta_shapes(plans: &[VariantPlan]) -> Vec<(SiteKey, usize)> {
    keys(plans, true).into_iter().map(|key| {
        let cols = plans.iter().find_map(|p| match p.site(key) {
            Some(SiteOp::Delta { cols, .. }) => Some(*cols as usize),
            _ => None,
        }).unwrap_or(0);
        (key, cols)
    }).collect()
}

/// What the variants hold on the coordinator GPU, charged before the KV pool is sized.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct VariantBytes {
    /// Steering tables (`a`, `b`, `k` per site).
    pub tables: u64,
    /// FP8 deltas, their scales and per-site switches.
    pub deltas: u64,
    /// The delta GEMV's output and scratch, and the largest transient BF16 upload chunk.
    pub workspace: u64,
    /// Per-row variant ids of every step workspace.
    pub rows: u64,
}
impl VariantBytes {
    pub fn total(&self) -> u64 { self.tables + self.deltas + self.workspace + self.rows }
}

fn rounded(bytes: usize) -> u64 { bytes.max(256) as u64 }

/// Device bytes for `plans` with delta slices of `slice_rows` rows; `gemv_scratch` is the
/// native GEMV's workspace for one slice of the given `(k, n)` shapes, and `table_rows` the
/// rows of every step workspace (one `i32` each).
pub(crate) fn device_bytes(plans: &[VariantPlan], hidden: usize, slice_rows: usize, table_rows: &[usize],
    gemv_scratch: impl Fn(&[(usize, usize)]) -> Result<usize>) -> Result<VariantBytes> {
    let tables = keys(plans, false).into_iter().map(|key| {
        let (a, b, k, _) = site_tables(plans, key, hidden);
        rounded(a.len() * 4) + rounded(b.len() * 4) + rounded(k.len() * 4)
    }).sum();
    let shapes = delta_shapes(plans);
    let slots = plans.iter().map(|p| usize::from(p.index)).max().unwrap_or(0) + 1;
    let deltas = shapes.iter().map(|&(_, k)| rounded(hidden * k) + rounded(hidden * k / 128 * 4) + rounded(slots * 4)).sum();
    let workspace = if shapes.is_empty() { 0 } else {
        let gemv: Vec<(usize, usize)> = shapes.iter().map(|&(_, k)| (k, hidden)).collect();
        let widest = shapes.iter().map(|&(_, k)| k).max().unwrap_or(0);
        rounded(slice_rows * hidden * 4) + rounded(gemv_scratch(&gemv)?) + rounded(PACK_ROWS * widest * 2)
    };
    let rows = if plans.is_empty() { 0 } else { table_rows.iter().map(|&r| rounded(r * 4)).sum() };
    Ok(VariantBytes { tables, deltas, workspace, rows })
}

impl<'a> GlmfVariants<'a> {
    /// Upload every configured site on the current device and `stream` (outside the
    /// weight ledger; admission charged [`device_bytes`]).
    pub fn upload(library: &'a NativeLibrary, stream: *mut c_void, plans: Vec<VariantPlan>, records: Vec<VariantRecord>,
        hidden: usize, slice_rows: usize) -> Result<Self> {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("weight_variants");
        ensure!(plans.iter().all(|p| p.hidden as usize == hidden), "variant tables do not match hidden {hidden}");
        ensure!(slice_rows > 0, "delta slices need rows");
        let put = |values: &[u8]| -> Result<DeviceAllocation<'a>> {
            let dev = DeviceAllocation::new(library, values.len().max(256))?;
            library.copy_h2d(cuteafd_ffi::CuteafdDeviceBuffer { bytes: values.len(), ..dev.buffer }, values)?;
            Ok(dev)
        };
        let mut steer = BTreeMap::new();
        for key in keys(&plans, false) {
            let (a, b, k, k_max) = site_tables(&plans, key, hidden);
            steer.insert(key, SteerSite { a: put(bytes_of(&a))?, b: put(bytes_of(&b))?, k: put(bytes_of(&k))?, k_max });
        }
        let shapes = delta_shapes(&plans);
        let mut deltas = BTreeMap::new();
        for &(key, cols) in &shapes {
            ensure!(cols % 128 == 0 && hidden % 16 == 0, "delta at {key}: [{hidden}, {cols}] needs whole 128-K blocks");
            let distinct: Vec<&Vec<u16>> = plans.iter().filter_map(|p| match p.site(key) {
                Some(SiteOp::Delta { weight, .. }) => Some(weight),
                _ => None,
            }).collect();
            // One ΔW per site: a second variant with its own delta at the same site would need its own copy.
            ensure!(distinct.windows(2).all(|w| w[0] == w[1]),
                "delta at {key}: variants store different deltas at one site (one per site is supported)");
            let weight = Fp8Weight::allocate(library, hidden, cols)?;
            let (packed, scale) = weight.buffers();
            let chunk = DeviceAllocation::new(library, PACK_ROWS * cols * 2)?;
            for first in (0..hidden).step_by(PACK_ROWS) {
                let rows = PACK_ROWS.min(hidden - first);
                library.copy_h2d(cuteafd_ffi::CuteafdDeviceBuffer { bytes: rows * cols * 2, ..chunk.buffer },
                    bytes_of(&distinct[0][first * cols..(first + rows) * cols]))?;
                // SAFETY: `chunk` holds `rows` BF16 rows of ΔW; rows `first..first + rows` of the packed copy
                // (16-row tiles are contiguous, `first` a multiple of 16) and their scales are inside `weight`.
                unsafe {
                    library.fp8_w8a16_pack(chunk.buffer.ptr.cast_const(), packed.ptr.cast::<u8>().add(first * cols).cast(),
                        scale.ptr.cast::<u8>().add(first * cols / 128 * 4).cast(), rows, cols, Fp8Scales::Best.code(), stream)?;
                    // The chunk is rewritten by the next copy: drain first.
                    library.cuda_stream_synchronize(stream)?;
                }
            }
            deltas.insert(key, DeltaSite { weight, on: put(bytes_of(&delta_switch(&plans, key)))? });
        }
        let (out, scratch) = if shapes.is_empty() { (None, None) } else {
            let gemv: Vec<(usize, usize)> = shapes.iter().map(|&(_, k)| (k, hidden)).collect();
            (Some(DeviceAllocation::new(library, slice_rows * hidden * 4)?),
                Some(crate::shared::fp8_linear::scratch(library, slice_rows, &gemv)?))
        };
        let slots = plans.iter().map(|p| usize::from(p.index)).max().unwrap_or(0);
        let steer_rows = (!steer.is_empty()).then(|| library.steer_rows()).transpose()?;
        let add_rows = (!deltas.is_empty()).then(|| library.add_rows()).transpose()?;
        Ok(Self { steer, deltas, out, scratch, slice_rows, hidden, slots, steer_rows, add_rows, records, plans })
    }

    pub fn site_count(&self) -> usize { self.steer.len() + self.deltas.len() }
    pub fn delta_count(&self) -> usize { self.deltas.len() }
    pub fn slice_rows(&self) -> usize { self.slice_rows }

    /// Steer `rows` BF16 rows of `x` (row stride `hidden`) at `key`, if any variant steers it.
    pub fn steer(&self, key: SiteKey, x: *mut c_void, rows: usize, vidx: *const c_void, stream: *mut c_void) -> Result<()> {
        let Some(site) = self.steer.get(&key) else { return Ok(()) };
        let kernel = self.steer_rows.as_ref().context("steer_rows binding")?;
        let shape = cuteafd_ffi::steer_rows::SteerRowsShape { rows, h: self.hidden, ld: self.hidden, k_max: site.k_max,
            variants: self.slots };
        // SAFETY: `x` holds `rows` live BF16 rows of `hidden` columns and `vidx` `rows` i32 ids on the
        // binding device, ordered on `stream`; the tables are this struct's live allocations of the §4
        // shapes (256-byte aligned), and `x` overlaps none of them.
        unsafe {
            kernel.launch_raw(x.cast(), vidx.cast(), site.a.buffer.ptr.cast_const().cast(),
                site.b.buffer.ptr.cast_const().cast(), site.k.buffer.ptr.cast_const().cast(), shape, stream)
        }.with_context(|| format!("steer_rows at {key}"))
    }

    pub fn has_delta(&self, key: SiteKey) -> bool { self.deltas.contains_key(&key) }

    /// `y += ΔW x` on the rows whose variant stores the delta at `key`: `x` is `rows` BF16 rows
    /// of the site tensor's input (row stride K), `y` the BF16 output (stride `hidden`). `spare`
    /// is device memory nothing reads until the next program (a wide chunk's output and
    /// workspace go there).
    #[allow(clippy::too_many_arguments)]
    pub fn delta(&self, library: &NativeLibrary, key: SiteKey, x: *const c_void, rows: usize, y: *mut c_void,
        vidx: *const c_void, spare: Option<(*mut c_void, usize)>, stream: *mut c_void) -> Result<()> {
        let Some(site) = self.deltas.get(&key) else { return Ok(()) };
        let add = self.add_rows.as_ref().context("add_rows binding")?;
        let (h, k) = (self.hidden, site.weight.k);
        let add_shape = |rows| cuteafd_ffi::steer_rows::AddRowsShape { rows, h, ld_y: h, ld_d: h, variants: self.slots };
        if rows > self.slice_rows {
            let (out_bytes, workspace) = wide_delta_bytes(rows, h, k);
            if let Some((room, _)) = spare.filter(|&(room, len)| room as usize % 256 == 0 && out_bytes + workspace <= len) {
                let (packed, scale) = site.weight.buffers();
                // SAFETY: `room` holds `out_bytes` FP32 output rows then `workspace` bytes that no other
                // work reads before the next program on this stream; `x`, `y` and `vidx` hold `rows`
                // rows; the stream orders the pass before the add.
                return unsafe {
                    library.fp8_linear(x, packed.ptr.cast_const(), scale.ptr.cast_const(), room, true, rows, k, h,
                        crate::shared::fp8_linear::Fp8Rows::W8a8.code(), room.byte_add(out_bytes), workspace, stream)?;
                    add.launch_raw(y.cast(), room.cast_const().cast(), vidx.cast(), site.on.buffer.ptr.cast_const().cast(),
                        add_shape(rows), stream)
                }.with_context(|| format!("delta at {key} ({rows} rows, one pass)"));
            }
            tracing::debug!(%key, rows, "delta: no spare room for one pass; slices of {}", self.slice_rows);
        }
        let (out, scratch) = (self.out.as_ref().context("delta output")?, self.scratch.as_ref().context("delta scratch")?);
        for first in (0..rows).step_by(self.slice_rows) {
            let n = self.slice_rows.min(rows - first);
            // SAFETY: rows `first..first + n` of `x`, `y` and `vidx` lie inside the caller's buffers; `out`
            // holds `slice_rows` FP32 rows and `scratch` was sized for `slice_rows` rows of this shape. The
            // stream orders the GEMV before the add and the next slice's GEMV after it.
            unsafe {
                site.weight.apply(library, x.byte_add(first * k * 2), out.buffer.ptr, true, n, 0, h, scratch, stream)?;
                add.launch_raw(y.byte_add(first * h * 2).cast(), out.buffer.ptr.cast_const().cast(),
                    vidx.byte_add(first * 4).cast(), site.on.buffer.ptr.cast_const().cast(), add_shape(n), stream)
            }.with_context(|| format!("delta at {key}"))?;
        }
        Ok(())
    }
}

/// A whole-height delta pass over `rows` rows of input width `k`: `(output, workspace)` bytes,
/// the FP32 output (256-byte aligned) then the W8A8 workspace that holds every row's E4M3
/// activations and per-128 scales at once, so `fp8_linear` reads ΔW in one tile (its fixed
/// row-scale and maxima regions fit in the 1 MiB margin).
pub(crate) fn wide_delta_bytes(rows: usize, hidden: usize, k: usize) -> (usize, usize) {
    let tiled = rows.next_multiple_of(128);
    ((rows * hidden * 4).next_multiple_of(256), (k * tiled).next_multiple_of(256) + (k / 128 * tiled * 4).next_multiple_of(256)
        + (1 << 20))
}

/// Why a derived variant cannot serve on GLM Flash's sites (`None`: it can).
pub(crate) fn unsupported(plan: &VariantPlan, cfg: &GlmNextConfig) -> Option<String> {
    plan.sites.iter().find_map(|(key, op)| {
        let layer = usize::from(key.layer);
        if layer >= cfg.layers { return Some(format!("{key}: layer {layer} is not run")); }
        let dense = cfg.dense[layer];
        match (key.kind, op) {
            (SiteKind::Ffn, _) => Some(format!("{key}: GLM Flash has no combined-FFN site (its edits are per tensor)")),
            (SiteKind::SharedExpert, _) if dense => Some(format!("{key}: layer {layer} is dense")),
            (SiteKind::DenseFfn, _) if !dense => Some(format!("{key}: layer {layer} is an MoE layer")),
            (SiteKind::Attn, SiteOp::Delta { .. }) if cfg.attention[layer] != GlmNextAttention::Kda =>
                Some(format!("{key}: a delta on an MLA o_proj is not supported (its input stays inside the o program)")),
            (_, SiteOp::Delta { cols, .. }) => {
                let want = match key.kind {
                    SiteKind::Attn => cfg.kda_heads * cfg.kda_head_dim,
                    SiteKind::SharedExpert => cfg.moe_intermediate,
                    _ => cfg.dense_intermediate,
                };
                (*cols as usize != want).then(|| format!("{key}: delta input width {cols}, the site's is {want}"))
            }
            _ => None,
        }
    })
}

/// The launch's derived variants: the available plans and their records.
pub(crate) struct Derived {
    pub plans: Vec<VariantPlan>,
    pub records: Vec<VariantRecord>,
}

/// Derive and verify every `--variant` against the served snapshot (or its `base=`).
/// A variant that fails verification, or needs a site this engine cannot serve, is
/// unavailable; base serving continues.
pub(crate) fn derive(snapshot: &std::path::Path, served_id: &str, specs: &[(u8, cuteafd_core::variants::VariantSpec)],
    cfg: &GlmNextConfig) -> Result<Derived> {
    let parallel = std::thread::available_parallelism().map_or(4, |n| (n.get() / 8).clamp(1, 4));
    let (mut plans, mut records) = (Vec::new(), Vec::new());
    for (index, spec) in specs {
        let outcome = cuteafd_loader::variants::derive_variant(snapshot, served_id, *index, spec, parallel)
            .with_context(|| format!("variant {}", spec.name))?;
        let verify = outcome.verify_summary();
        let Some(plan) = outcome.plan else {
            tracing::warn!(variant = %spec.name, rejection = outcome.rejection.as_deref().unwrap_or("rejected"),
                "weight variant unavailable: not an output-side low-rank edit; a weight-set variant is required");
            continue;
        };
        ensure!(plan.hidden as usize == cfg.hidden, "variant {}: hidden {} differs from the model's {}", spec.name,
            plan.hidden, cfg.hidden);
        if let Some(reason) = unsupported(&plan, cfg) {
            tracing::warn!(variant = %spec.name, %reason, "weight variant unavailable on GLM Flash");
            continue;
        }
        tracing::info!(variant = %spec.name, id = %plan.public_id, sites = plan.sites.len(),
            deltas = plan.sites.iter().filter(|(_, op)| op.is_delta()).count(), k_max = plan.k_max(),
            namespace = format_args!("{:016x}", plan.namespace), %verify, "weight variant available");
        records.push(VariantRecord::of(&plan, served_id, verify));
        plans.push(plan);
    }
    Ok(Derived { plans, records })
}

/// Whether a step launches its variant sites: a decode step always (its captured graphs keep
/// one topology), a prefill step only when some row is not base.
pub(crate) fn launches(decode: bool, variants: &[i32]) -> bool {
    decode || variants.iter().any(|&v| v != 0)
}

/// Byte offset of the KDA program's gated-norm output (`o_proj`'s input, `[rows, D]` BF16) in
/// its scratch: after the in-projection, f|gate, convolved q|k|v and recurrent output regions,
/// each 1024-byte aligned (`kda_scratch_bytes` in the SparkInfer glmf programs).
pub(crate) fn kda_norm_offset(rows: usize, in_width: usize, d: usize) -> usize {
    [rows * in_width * 2, rows * 4 * d, rows * 6 * d, rows * 2 * d].iter().map(|n| n.next_multiple_of(1024)).sum()
}

/// Byte offset of the FFN program's SwiGLU activation (`down_proj`'s input, `[rows, I]` BF16) in
/// its scratch: after the gate|up rows, 1024-byte aligned.
pub(crate) fn ffn_hidden_offset(rows: usize, inter: usize) -> usize {
    (rows * 4 * inter).next_multiple_of(1024)
}

/// Bytes at the start of the KDA program's scratch that are dead once it returns: its
/// in-projection rows `[rows, P]` (BF16), read only inside the program. They end before the
/// gated-norm output a delta reads.
pub(crate) fn kda_spare_bytes(rows: usize, in_width: usize) -> usize {
    rows * in_width * 2
}

/// Where the FFN program's scratch use ends: the SwiGLU activation's end, 1024-byte aligned.
/// The program never writes past it.
pub(crate) fn ffn_spare_offset(rows: usize, inter: usize) -> usize {
    (ffn_hidden_offset(rows, inter) + rows * inter * 2).next_multiple_of(1024)
}

fn bytes_of<T: Copy>(values: &[T]) -> &[u8] {
    // SAFETY: plain numeric slices (f32, i32, u16) have no padding and u8 has alignment 1.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}

#[cfg(test)]
mod tests;
