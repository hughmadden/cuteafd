//! Per-request weight variants on V4.1 (contract §2, §4). A variant row gets
//! `y -= A (Bᵀ y)` on the BF16 `wo_b` output of each edited layer, in place,
//! before the attention mHC post (`hc_post`) adds it to the residual. Base rows
//! are never written, and nothing here exists when no variant is configured.
//!
//! Where it runs: [`BackboneBlockWave::enqueue_ffn`](super::v41_block) is the
//! one consumer of the projected `wo_b` rows on every path (the captured sparse
//! tail, dual-head tails, the TP2 output gather and its FFN graphs), so one
//! launch there covers prefill, decode, MTP verify, decoder replay and the
//! distributed lanes. The kernel reads a per-row variant index (`vidx`) from a
//! lane buffer with a fixed address; the batch's rows are uploaded beside the
//! projection positions before each layer's tail (`AttentionTail::prepare`).
//!
//! Graphs: whether the node is in a graph depends only on the layer and the
//! row count, never on the batch, so no graph key gains a variant axis. Per
//! recipe (V4.1 prefill is one request per chunk), a batch with no variant row
//! skips the launch on shapes that are never captured.
use crate::shared::memory::{DeviceAllocation, HostAllocation};
use anyhow::{bail, ensure, Context, Result};
use cuteafd_core::variants::{site_tables, SiteKey, SiteKind, VariantPlan};
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary};

/// V4.1 residual width (the `wo_b` output columns).
pub(crate) const V41_HIDDEN: usize = 5120;
const LAYERS: usize = 40;

/// One distinct site table in the kernel's layout: `a`/`b` `[V+1][k_max][H]`
/// (slot 0 zero), `k` `[V+1]`.
#[derive(Debug, Clone, PartialEq)]
struct HostTable { a: Vec<f32>, b: Vec<f32>, k: Vec<i32>, k_max: usize }
impl HostTable {
    fn bytes(&self) -> usize { (self.a.len() + self.b.len()) * 4 + self.k.len() * 4 }
}

/// Host copy of every configured variant's `wo_b` tables. Sites with equal
/// tables share one (Keys 1017: one direction and α for all 26 layers).
pub(crate) struct V41SteerTables {
    tables: Vec<HostTable>,
    layers: [Option<u16>; LAYERS],
    variants: usize,
}
impl V41SteerTables {
    /// None when no variant is configured: the engine is then exactly today's.
    pub fn new(plans: &[VariantPlan]) -> Result<Option<Self>> {
        if plans.is_empty() { return Ok(None); }
        let mut tables: Vec<HostTable> = Vec::new();
        let mut layers = [None; LAYERS];
        for plan in plans {
            ensure!(plan.hidden as usize == V41_HIDDEN, "variant {}: hidden {} is not V4.1's {V41_HIDDEN}", plan.name, plan.hidden);
            if let Some((site, _)) = plan.sites.iter().find(|(s, _)| s.kind != SiteKind::Attn || usize::from(s.layer) >= LAYERS) {
                bail!("variant {}: site {site} is not a V4.1 attention output (wo_b) site", plan.name);
            }
        }
        for (layer, slot) in layers.iter_mut().enumerate() {
            let key = SiteKey { layer: layer as u16, kind: SiteKind::Attn };
            if plans.iter().all(|p| p.site(key).is_none()) { continue; }
            let (a, b, k, k_max) = site_tables(plans, key, V41_HIDDEN);
            let table = HostTable { a, b, k, k_max };
            let index = tables.iter().position(|t| *t == table).unwrap_or_else(|| { tables.push(table); tables.len() - 1 });
            *slot = Some(index as u16);
        }
        let variants = plans.iter().map(|p| usize::from(p.index)).max().unwrap_or(0);
        Ok(Some(Self { tables, layers, variants }))
    }
    pub fn edits(&self, layer: usize) -> bool { self.layers.get(layer).is_some_and(Option::is_some) }
    /// Table bytes on a GPU holding the layers `holds` selects: each distinct
    /// table once. The planner charges this before sizing the KV pool.
    pub fn device_bytes(&self, holds: impl Fn(usize) -> bool) -> usize {
        let mut used = vec![false; self.tables.len()];
        for (layer, table) in self.layers.iter().enumerate() {
            if let (Some(t), true) = (table, holds(layer)) { used[usize::from(*t)] = true; }
        }
        self.tables.iter().zip(used).filter(|(_, u)| *u).map(|(t, _)| t.bytes()).sum()
    }
    /// Per-lane mask buffer bytes (one i32 per row).
    pub fn lane_bytes(capacity: usize) -> usize { capacity * 4 }
    /// Planner demands on `gpu`: its tables (weights) and `lanes` row masks.
    pub fn demands(&self, gpu: u8, holds: impl Fn(usize) -> bool, capacity: usize, lanes: usize) -> Vec<cuteafd_loader::placement::Demand> {
        use cuteafd_core::memory_layout::{Basis, Category};
        use cuteafd_loader::placement::Demand;
        vec![Demand::new(gpu, Category::Weights, "weight variant wo_b tables", self.device_bytes(holds) as u64, Basis::Formula),
            Demand::new(gpu, Category::Workspace, "weight variant row masks", (lanes * Self::lane_bytes(capacity)) as u64, Basis::Formula)]
    }
}

/// Each row's variant in `rows` order (`CacheBatch::expert_rows`, the order
/// of positions and of the block's tokens), from `(request id, variant)`.
pub(crate) fn variant_rows(rows: &[super::v41_backbone_router::ExpertRow], requests: &[(u64, u8)]) -> Result<Vec<u8>> {
    rows.iter().map(|row| requests.iter().find(|(id, _)| *id == row.request_id).map(|(_, v)| *v)
        .with_context(|| format!("batch row of request {} has no request", row.request_id))).collect()
}

/// One configured variant after derivation.
pub(crate) struct V41Variant {
    pub plan: VariantPlan,
    pub derivation: cuteafd_loader::V41Derivation,
}

impl V41Variant {
    /// What `/v1/models` and stats show for this variant.
    pub fn record(&self, served: &str) -> cuteafd_core::variants::VariantRecord {
        let d = &self.derivation;
        let fold = |f: fn(&cuteafd_loader::V41SiteReport) -> f32, max: bool| d.sites.iter().map(f)
            .fold(if max { f32::MIN } else { f32::MAX }, if max { f32::max } else { f32::min });
        cuteafd_core::variants::VariantRecord::of(&self.plan, served, serde_json::json!({
            "alpha": d.alpha, "shared_direction": d.shared_direction, "direction_cos_min": d.direction_cos_min,
            "next_share_max": fold(|s| s.verify.next_share, true), "proj_cos_min": fold(|s| s.verify.proj_cos, false),
            "rebuild_exact_min": fold(|s| s.rebuild_exact, false), "unchanged": d.unchanged.len()}))
    }
}

/// Derive every `--variant` against the served stock snapshot (or its `base=`),
/// reading the edited `wo_b` tensors from the variant's own file. A variant
/// whose edit fails the recipe's bars is logged and left out; base serving and
/// the other variants continue.
pub(crate) fn load_variants(specs: &[(u8, cuteafd_core::variants::VariantSpec)], served: &str,
    catalog: &cuteafd_loader::OfficialV41Catalog) -> Result<Vec<V41Variant>> {
    let mut loaded = Vec::new();
    for (index, spec) in specs {
        ensure!(spec.mode.is_none(), "variant {}: DeepSeek V4.1 has no variant modes (got mode={})",
            spec.name, spec.mode.as_deref().unwrap_or_default());
        let started = std::time::Instant::now();
        let source = cuteafd_loader::V41VariantSource::open(std::path::Path::new(&spec.source), LAYERS as u16)
            .with_context(|| format!("variant {}", spec.name))?;
        let base_catalog = spec.base.as_ref().map(|b| cuteafd_loader::read_official_v41_catalog(std::path::Path::new(b)))
            .transpose().with_context(|| format!("variant {} base", spec.name))?;
        let base: &dyn cuteafd_loader::TensorBytes = base_catalog.as_ref().unwrap_or(catalog);
        let derivation = cuteafd_loader::derive_v41_variant(base, &source, V41_HIDDEN, &Default::default())
            .with_context(|| format!("variant {} derivation", spec.name))?;
        if !derivation.accept {
            tracing::error!(variant = %spec.name, reason = derivation.reason.as_deref().unwrap_or(""),
                "weight variant refused; it will not be served");
            continue;
        }
        let plan = VariantPlan::new(*index, spec.name.clone(), spec.public_id(served), V41_HIDDEN as u32, derivation.ops.clone())?;
        tracing::info!(variant = %spec.name, id = %plan.public_id, file = %source.file.display(), sites = plan.sites.len(),
            alpha = ?derivation.alpha, shared_direction = derivation.shared_direction,
            direction_cos_min = derivation.direction_cos_min, seconds = started.elapsed().as_secs_f32(),
            "weight variant derived");
        loaded.push(V41Variant { plan, derivation });
    }
    Ok(loaded)
}

/// One table resident on a GPU.
struct DeviceTable<'a> { a: DeviceAllocation<'a>, b: DeviceAllocation<'a>, k: DeviceAllocation<'a>, k_max: i32 }
/// The tables a GPU's layers use, uploaded once at load.
pub(crate) struct V41SteerWeights<'a> {
    device: i32,
    tables: Vec<Option<DeviceTable<'a>>>,
    layers: [Option<u16>; LAYERS],
    variants: usize,
}
fn as_bytes<T: Copy>(values: &[T]) -> &[u8] {
    // SAFETY: f32/i32 are plain data without padding; the view spans exactly the slice.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values)) }
}
impl<'a> V41SteerWeights<'a> {
    /// Upload the tables used by the layers on the current device `device`.
    pub fn upload(library: &'a NativeLibrary, device: i32, host: &V41SteerTables, holds: impl Fn(usize) -> bool) -> Result<Self> {
        let mut tables: Vec<Option<DeviceTable<'a>>> = (0..host.tables.len()).map(|_| None).collect();
        let mut layers = [None; LAYERS];
        for (layer, table) in host.layers.iter().enumerate() {
            let Some(index) = *table else { continue };
            if !holds(layer) { continue; }
            layers[layer] = Some(index);
            let slot = &mut tables[usize::from(index)];
            if slot.is_some() { continue; }
            let t = &host.tables[usize::from(index)];
            let upload = |values: &[u8]| -> Result<DeviceAllocation<'a>> {
                let allocation = DeviceAllocation::named(library, values.len(), "v41 variant steering tables")?;
                library.copy_h2d(allocation.buffer, values)?;
                Ok(allocation)
            };
            *slot = Some(DeviceTable { a: upload(as_bytes(&t.a))?, b: upload(as_bytes(&t.b))?, k: upload(as_bytes(&t.k))?,
                k_max: i32::try_from(t.k_max)? });
        }
        Ok(Self { device, tables, layers, variants: host.variants })
    }
    fn site(&self, layer: usize) -> Option<&DeviceTable<'a>> {
        self.layers.get(layer).copied().flatten().and_then(|i| self.tables[usize::from(i)].as_ref())
    }
    pub fn device(&self) -> i32 { self.device }
}

/// Whether a layer's `wo_b` rows get the steering launch. Captured graphs must
/// not depend on the batch, so any shape that may be captured always launches
/// (the kernel leaves base rows untouched); a never-captured shape with no
/// variant row skips it (V4.1 prefill chunks are one request each).
pub(crate) fn launches(edits_layer: bool, any_variant_row: bool, may_capture: bool) -> bool {
    edits_layer && (any_variant_row || may_capture)
}

/// Per-lane steering state inside the block: the mask buffer and its staging.
pub(crate) struct BlockSteer<'w, 'a> {
    weights: &'w V41SteerWeights<'a>,
    vidx: DeviceAllocation<'a>,
    staging: HostAllocation<'a>,
    rows: Option<Vec<i32>>,
    any: bool,
    /// Dual-head tails and the TP2 FFN bank capture every row count.
    capture_every_shape: bool,
    #[cfg(test)]
    pub(crate) launched: Vec<(usize, usize)>,
}
impl<'w, 'a> BlockSteer<'w, 'a> {
    pub fn new(library: &'a NativeLibrary, weights: &'w V41SteerWeights<'a>, capacity: usize) -> Result<Self> {
        Ok(Self { weights, vidx: DeviceAllocation::named(library, V41SteerTables::lane_bytes(capacity), "v41 variant row mask")?,
            staging: HostAllocation::new(library, V41SteerTables::lane_bytes(capacity))?, rows: None, any: false,
            capture_every_shape: false, #[cfg(test)] launched: Vec::new() })
    }
    pub fn capture_every_shape(&mut self) { self.capture_every_shape = true; }
    /// This pass's per-row variant indices, in the batch's row order.
    pub fn set_rows(&mut self, rows: &[u8]) -> Result<()> {
        ensure!(rows.len() * 4 <= self.vidx.buffer.bytes, "variant rows exceed lane capacity");
        if let Some(&bad) = rows.iter().find(|&&v| usize::from(v) > self.weights.variants) {
            bail!("row variant {bad} is not configured (have {})", self.weights.variants);
        }
        self.any = rows.iter().any(|&v| v != 0);
        self.rows = Some(rows.iter().map(|&v| i32::from(v)).collect());
        Ok(())
    }
    /// Stable addresses a containing graph captures.
    pub fn identity(&self, layer: usize) -> [usize; 2] {
        [self.vidx.buffer.ptr as usize, self.weights.site(layer).map_or(0, |t| t.a.buffer.ptr as usize)]
    }
    /// Queue the mask upload beside the projection positions. Staging holds the
    /// same rows for every layer of a pass, so a rewrite while an earlier
    /// layer's copy is queued cannot change what that copy reads.
    ///
    /// # Safety
    /// The caller drains `stream` before releasing or reusing this lane.
    pub unsafe fn upload(&mut self, rows: usize, stream: *mut std::ffi::c_void) -> Result<()> {
        let values = self.rows.as_ref().context("variant rows were not staged for this pass")?;
        ensure!(values.len() == rows, "variant rows {} differ from the pass's {rows} rows", values.len());
        self.staging.bytes_mut()[..rows * 4].copy_from_slice(as_bytes(values));
        unsafe { self.vidx.library.copy_host_buffer_h2d_async(self.vidx.buffer, self.staging.buffer, rows * 4, stream) }
    }
    /// Steer the projected `wo_b` rows of `layer` in place on `stream`.
    ///
    /// # Safety
    /// `x` is this lane's BF16 `[rows, 5120]` projection output, ordered on
    /// `stream` after its producer and before its consumer; the mask upload
    /// for this pass precedes it on the same stream.
    pub unsafe fn enqueue(&mut self, layer: usize, x: CuteafdDeviceBuffer, rows: usize, stream: *mut std::ffi::c_void) -> Result<()> {
        let may_capture = self.capture_every_shape || super::graph_policy::captures_shape(rows as u32);
        let Some(table) = self.weights.site(layer).filter(|_| launches(true, self.any, may_capture)) else { return Ok(()) };
        ensure!(x.bytes == rows * V41_HIDDEN * 2 && x.device_id == self.weights.device, "steered wo_b rows differ");
        #[cfg(test)]
        self.launched.push((layer, rows));
        unsafe {
            steer_rows(self.vidx.library, x, V41_HIDDEN as i64, rows as i32, V41_HIDDEN as i32, self.vidx.buffer,
                table.a.buffer, table.b.buffer, table.k.buffer, table.k_max, stream)
        }
    }
}

/// `cuteafd_steer_rows` (contract §4). Stand-in until the `wv-kernel` binding
/// lands in `cuteafd-ffi`: refuses loudly instead of serving unsteered rows.
#[allow(clippy::too_many_arguments)]
unsafe fn steer_rows(_library: &NativeLibrary, _x: CuteafdDeviceBuffer, _ld: i64, _rows: i32, _h: i32,
    _vidx: CuteafdDeviceBuffer, _a: CuteafdDeviceBuffer, _b: CuteafdDeviceBuffer, _k: CuteafdDeviceBuffer,
    _k_max: i32, _stream: *mut std::ffi::c_void) -> Result<()> {
    bail!("cuteafd_steer_rows is not bound in cuteafd-ffi yet (wv-kernel); variant rows cannot be served")
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_core::variants::{bf16_rne, bf16_to_f32, steer_rows_ref, SiteOp};
    use anyhow::{ensure, Context};

    fn plan(index: u8, alpha: f32, r: &[f32], layers: impl IntoIterator<Item = u16>) -> VariantPlan {
        VariantPlan::new(index, format!("v{index}"), format!("m-v{index}"), V41_HIDDEN as u32,
            layers.into_iter().map(|l| (SiteKey { layer: l, kind: SiteKind::Attn }, SiteOp::rank1(alpha, r))).collect()).unwrap()
    }
    fn unit(seed: usize) -> Vec<f32> {
        let mut r: Vec<f32> = (0..V41_HIDDEN).map(|i| (((i * 31 + seed * 7) % 97) as f32 - 48.0) / 48.0).collect();
        let n = r.iter().map(|x| x * x).sum::<f32>().sqrt();
        r.iter_mut().for_each(|x| *x /= n);
        r
    }

    /// Keys 1017: one direction and α on layers 10–35 is one table (~80 KiB
    /// with the unused base slot), charged once per GPU that holds those layers.
    #[test]
    fn shared_direction_is_one_table_and_the_planner_charges_it_once() -> Result<()> {
        let tables = V41SteerTables::new(&[plan(1, 3.5, &unit(1), 10..=35)])?.unwrap();
        assert_eq!(tables.tables.len(), 1);
        assert!((0..10).chain(36..40).all(|l| !tables.edits(l)) && (10..=35).all(|l| tables.edits(l)));
        let one = 2 * 2 * V41_HIDDEN * 4 + 2 * 4;
        assert_eq!(tables.device_bytes(|_| true), one);
        assert!(one <= 81 << 10, "{one} bytes");
        // Split over two GPUs at layer 20: both hold some edited layer.
        assert_eq!(tables.device_bytes(|l| l < 20), one);
        assert_eq!(tables.device_bytes(|l| l < 10), 0, "a GPU without edited layers holds nothing");
        assert_eq!(V41SteerTables::lane_bytes(64), 256);
        let demands = tables.demands(0, |_| true, 4096, 2);
        assert_eq!(demands.iter().map(|d| d.bytes).collect::<Vec<_>>(), [one as u64, 2 * 4096 * 4]);
        assert_eq!(demands[0].category, cuteafd_core::memory_layout::Category::Weights);
        // A second variant with its own direction gives distinct tables only where it edits.
        let two = V41SteerTables::new(&[plan(1, 3.5, &unit(1), 10..=35), plan(2, 2.0, &unit(2), 30..=39)])?.unwrap();
        assert_eq!(two.tables.len(), 3, "10-29, 30-35 and 36-39 differ");
        assert!(V41SteerTables::new(&[])?.is_none(), "no variants: nothing exists");
        Ok(())
    }

    #[test]
    fn non_wo_b_sites_are_refused() {
        let ffn = VariantPlan::new(1, "x", "m-x", V41_HIDDEN as u32,
            vec![(SiteKey { layer: 3, kind: SiteKind::Ffn }, SiteOp::rank1(1.0, &unit(0)))]).unwrap();
        assert!(V41SteerTables::new(&[ffn]).is_err());
        let past = VariantPlan::new(1, "x", "m-x", V41_HIDDEN as u32,
            vec![(SiteKey { layer: 40, kind: SiteKind::Attn }, SiteOp::rank1(1.0, &unit(0)))]).unwrap();
        assert!(V41SteerTables::new(&[past]).is_err());
    }

    /// A mixed batch (a 3-row verify, a decode row, a 4-row prefill chunk)
    /// stages each row's variant at that row's index in the token order.
    #[test]
    fn mixed_batch_mask_follows_expert_row_order() -> Result<()> {
        use super::super::v41_backbone_router::ExpertRow;
        use cuteafd_transport::ExpertV2SourceKind as Kind;
        // Built exactly as CacheBatch::expert_rows: request by request, rows in position order.
        let batch = [(41u64, 100u64, 3u32, Kind::MtpVerify), (7, 55, 1, Kind::Decode), (19, 0, 4, Kind::Prefill)];
        let rows: Vec<ExpertRow> = batch.iter().flat_map(|&(id, position, n, kind)|
            (0..n).map(move |i| ExpertRow { request_id: id, position: position + u64::from(i), kind })).collect();
        let mask = variant_rows(&rows, &[(19, 0), (41, 2), (7, 1)])?;
        assert_eq!(mask, [2, 2, 2, 1, 0, 0, 0, 0]);
        let positions: Vec<u64> = rows.iter().map(|r| r.position).collect();
        assert_eq!(positions, [100, 101, 102, 55, 0, 1, 2, 3], "mask index i is token row i");
        assert!(variant_rows(&rows, &[(41, 2), (7, 1)]).is_err(), "every row needs its request");
        Ok(())
    }

    #[test]
    fn launch_rule_never_depends_on_the_batch_for_captured_shapes() {
        for any in [false, true] {
            assert!(launches(true, any, true), "captured shapes always carry the node");
            assert!(!launches(false, any, true) && !launches(false, any, false), "unedited layers never launch");
        }
        assert!(launches(true, true, false));
        assert!(!launches(true, false, false), "base-only eager chunks skip the kernel");
    }

    /// Juliette gate 2 (component): layer L's real `wo_b` projection on the
    /// GPU, steered by the kernel, against the published W' applied exactly
    /// (FP8 values dequantized, FP32 accumulation). Base rows stay bit-identical,
    /// the kernel matches the host reference, and the steered rows land on W'x
    /// within the GEMM's own error plus the FP8 requantization floor.
    ///   CUTEAFD_NATIVE_LIB=... CUTEAFD_SNAPSHOT=<stock 1017> CUTEAFD_VARIANT_SOURCE=<overlay> \
    ///   [CUTEAFD_VARIANT_LAYERS=10,22,35] TEST --exact --ignored --nocapture
    #[test]
    #[ignore = "GPU gate: native library with steer_rows, stock V4.1 1017 snapshot, Keys overlay"]
    fn component_wo_b_steering_matches_published_weights() -> Result<()> {
        use crate::families::deepseek_v41::v41_attention_output::AttentionOutputWeights;
        use crate::shared::memory::device::{Allocation, Device};
        let env = |key: &str| std::env::var(key).with_context(|| format!("{key} is required"));
        let library = unsafe { NativeLibrary::load(env("CUTEAFD_NATIVE_LIB")?)? };
        let catalog = cuteafd_loader::read_official_v41_catalog(std::path::Path::new(&env("CUTEAFD_SNAPSHOT")?))?;
        let overlay = env("CUTEAFD_VARIANT_SOURCE")?;
        let layers: Vec<usize> = std::env::var("CUTEAFD_VARIANT_LAYERS").unwrap_or_else(|_| "10,22,35".into())
            .split(',').map(str::parse).collect::<std::result::Result<_, _>>()?;
        let spec: cuteafd_core::variants::VariantSpec = format!("name=abliterated,source={overlay}").parse()?;
        let loaded = load_variants(&[(1, spec)], "served", &catalog)?;
        let variant = loaded.first().context("the overlay was refused")?;
        let (alpha, r) = match &variant.plan.sites[0].1 {
            SiteOp::Steer { a, b, .. } => (a[0] / b[0], b.clone()),
            SiteOp::Delta {} => unreachable!(),
        };
        let tables = V41SteerTables::new(std::slice::from_ref(&variant.plan))?.context("tables")?;
        let device = Device { library: &library, id: 0 };
        let steer_weights = device.run(|| V41SteerWeights::upload(&library, 0, &tables, |_| true))?;
        let source = cuteafd_loader::V41VariantSource::open(std::path::Path::new(&overlay), LAYERS as u16)?;
        let rows = 16usize;
        let mask: Vec<u8> = (0..rows).map(|i| (i % 2) as u8).collect();
        let mut steer = device.run(|| BlockSteer::new(&library, &steer_weights, rows))?;
        steer.capture_every_shape();
        steer.set_rows(&mask)?;
        let read = |buffer: CuteafdDeviceBuffer| -> Result<Vec<u16>> {
            let mut bytes = vec![0u8; buffer.bytes];
            device.run(|| library.copy_d2h(&mut bytes, buffer))?;
            Ok(bytes.chunks_exact(2).map(|b| u16::from_ne_bytes([b[0], b[1]])).collect())
        };
        let gemv = |w: &[f32], z: &[f32]| -> Vec<f32> {
            let mut out = vec![0f32; w.len() / z.len()];
            let threads = std::thread::available_parallelism().map_or(8, |n| n.get());
            let per = out.len().div_ceil(threads);
            std::thread::scope(|scope| for (i, chunk) in out.chunks_mut(per).enumerate() {
                scope.spawn(move || for (j, o) in chunk.iter_mut().enumerate() {
                    let row = &w[(i * per + j) * z.len()..][..z.len()];
                    *o = row.iter().zip(z).map(|(a, b)| f64::from(*a) * f64::from(*b)).sum::<f64>() as f32;
                });
            });
            out
        };
        let norm = |v: &[f32]| v.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
        let rel = |a: &[f32], b: &[f32]| norm(&a.iter().zip(b).map(|(x, y)| x - y).collect::<Vec<_>>()) / norm(b);
        let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| f64::from(*x) * f64::from(*y)).sum::<f64>();
        eprintln!("alpha={alpha:.4}  layer  gemm_floor  fp8_floor  base_vs_pub  steered_vs_pub  r_component_err");
        for layer in layers {
            let weights = device.own(|| AttentionOutputWeights::load(&library, &catalog, layer,
                AttentionOutputWeights::device_bytes(&library, &catalog, layer)?, 1 << 20))?;
            let mut wave = device.own(|| weights.wave(rows as u32, usize::MAX))?;
            let input = Allocation::new(device, rows * 65536)?;
            let host: Vec<u8> = (0..rows * 32768).flat_map(|i| {
                let value = (((i * 2654435761) % 4093) as f32 / 4093.0 - 0.5) * 0.25;
                ((value.to_bits() >> 16) as u16).to_ne_bytes()
            }).collect();
            device.run(|| library.copy_h2d(input.buffer, &host))?;
            let tokens: Vec<u64> = (0..rows as u64).map(|i| 1000 + i).collect();
            let attention = crate::families::deepseek_v41::v41_sparse_attention::QueuedSparseAttention {
                values: CuteafdDeviceBuffer { bytes: rows * 65536, ..input.buffer }, rows, layer };
            let ([z_buffer, y_buffer], stream) = wave.wo_b_buffers(rows);
            let y = device.run(|| unsafe {
                wave.prepare_chain_graph(&tokens, stream)?;
                let y = wave.enqueue_chain_graph(&attention, stream)?;
                library.cuda_stream_synchronize(stream)?;
                Ok(y)
            })?;
            ensure!(y.ptr == y_buffer.ptr && y.bytes == y_buffer.bytes, "projection output buffer differs");
            let (z, base_out) = (read(z_buffer)?, read(y_buffer)?);
            device.run(|| unsafe {
                steer.upload(rows, stream)?;
                steer.enqueue(layer, y_buffer, rows, stream)?;
                library.cuda_stream_synchronize(stream)
            })?;
            let steered = read(y_buffer)?;
            let mut reference = base_out.clone();
            let t = &tables.tables[0];
            let vidx: Vec<i32> = mask.iter().map(|&v| i32::from(v)).collect();
            steer_rows_ref(&mut reference, V41_HIDDEN, rows, V41_HIDDEN, &vidx, &t.a, &t.b, &t.k, t.k_max);
            let name = format!("layers.{layer}.attn.wo_b");
            let w = cuteafd_loader::read_fp8_block32(&catalog, &name)?.dequantize();
            let w_pub = cuteafd_loader::read_fp8_block32(&source, &name)?.dequantize();
            let mut worst = [0f64; 5];
            for row in 0..rows {
                let span = row * V41_HIDDEN..(row + 1) * V41_HIDDEN;
                if mask[row] == 0 {
                    ensure!(steered[span.clone()] == base_out[span.clone()], "layer {layer} base row {row} was written");
                    continue;
                }
                let ulp = steered[span.clone()].iter().zip(&reference[span.clone()]).map(|(a, b)| a.abs_diff(*b)).max().unwrap_or(0);
                ensure!(ulp <= 1, "layer {layer} row {row}: kernel differs from the host reference by {ulp} BF16 ulp");
                let f = |bits: &[u16]| bits.iter().map(|&b| bf16_to_f32(b)).collect::<Vec<f32>>();
                let zr = f(&z[row * 8192..(row + 1) * 8192]);
                let (y_gpu, y_steered) = (f(&base_out[span.clone()]), f(&steered[span]));
                let (y_host, y_pub) = (gemv(&w, &zr), gemv(&w_pub, &zr));
                let projected: Vec<f32> = { let c = dot(&r, &y_host) as f32; y_host.iter().zip(&r).map(|(y, r)| y - alpha * r * c).collect() };
                let (c_s, c_pub) = (dot(&r, &y_steered), dot(&r, &y_pub));
                let row_err = [rel(&y_gpu, &y_host), rel(&projected, &y_pub), rel(&y_gpu, &y_pub), rel(&y_steered, &y_pub),
                    (c_s - c_pub).abs() / c_pub.abs().max(1e-12)];
                for (w, e) in worst.iter_mut().zip(row_err) { *w = w.max(e); }
            }
            eprintln!("            {layer:>5}  {:.5}     {:.5}    {:.5}      {:.5}          {:.5}", worst[0], worst[1], worst[2], worst[3], worst[4]);
            ensure!(worst[3] <= 1.1 * (worst[0] + worst[1]) + 1e-3,
                "layer {layer}: steered rows miss W'x ({:.5}) beyond the GEMM ({:.5}) plus FP8 ({:.5}) floors", worst[3], worst[0], worst[1]);
            ensure!(worst[3] < 0.5 * worst[2], "layer {layer}: steering does not move the output toward W'x");
        }
        Ok(())
    }

    /// The host reference with the uploaded table layout leaves base rows
    /// bit-identical and steers variant rows by `x − α r (rᵀx)`.
    #[test]
    fn table_layout_steers_only_variant_rows() -> Result<()> {
        let r = unit(3);
        let tables = V41SteerTables::new(&[plan(1, 3.5, &r, [12])])?.unwrap();
        let t = &tables.tables[0];
        let rows = 3;
        let x: Vec<u16> = (0..rows * V41_HIDDEN).map(|i| bf16_rne(((i as f32) * 0.013).sin())).collect();
        let mut y = x.clone();
        steer_rows_ref(&mut y, V41_HIDDEN, rows, V41_HIDDEN, &[0, 1, 0], &t.a, &t.b, &t.k, t.k_max);
        assert_eq!(&y[..V41_HIDDEN], &x[..V41_HIDDEN]);
        assert_eq!(&y[2 * V41_HIDDEN..], &x[2 * V41_HIDDEN..]);
        let row: Vec<f32> = x[V41_HIDDEN..2 * V41_HIDDEN].iter().map(|&v| bf16_to_f32(v)).collect();
        let dot: f32 = row.iter().zip(&r).map(|(a, b)| a * b).sum();
        for c in (0..V41_HIDDEN).step_by(97) {
            let want = row[c] - 3.5 * r[c] * dot;
            assert!((bf16_to_f32(y[V41_HIDDEN + c]) - want).abs() <= want.abs() / 64.0 + 1e-4, "col {c}");
        }
        Ok(())
    }
}
