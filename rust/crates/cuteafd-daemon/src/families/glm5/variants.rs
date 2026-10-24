//! GLM-5.3 full weight variants on the coordinator (contract §2, §4): one
//! device table per configured site, `steer_rows` launched on the BF16
//! sublayer output before it enters `glm_norm`'s residual add.
//!
//! Sites (`cuteafd_loader::families::glm5::variants`): `Attn` at the end of
//! `o_proj` (`delta`), `DenseFfn` after a dense layer's FFN (`delta`), `Ffn`
//! right after `reduce` (routed + shared in `delta`; in decode the reduce runs
//! at the start of the next layer's segment, so it uses layer `index − 1`),
//! `SharedExpert` on the shared expert (`shared`) before `reduce` adds it.
//! Table pointers are baked into the captured decode graphs, which always
//! launch at configured sites; per-row variant ids arrive in the workspace's
//! `variants` buffer by H2D beside the step's other tables, so the graph key
//! never gains a variant axis. Base rows (`vidx == 0`) are never written. The kernel is
//! `cuteafd_ffi::steer_rows`, bound (its modules loaded) when the tables upload.
use crate::shared::memory::DeviceAllocation;
use anyhow::{ensure, Context, Result};
use cuteafd_core::variants::{site_tables, SiteKey, VariantPlan, VariantRecord};
use cuteafd_ffi::NativeLibrary;
use std::collections::BTreeMap;
use std::ffi::c_void;

/// One site's tables in the §4 layout: `a`, `b` FP32 `[V+1][k_max][h]`, `k` i32 `[V+1]`.
struct SiteTable<'a> {
    a: DeviceAllocation<'a>,
    b: DeviceAllocation<'a>,
    k: DeviceAllocation<'a>,
    k_max: usize,
}

/// Every configured site's device tables, plus what `/v1/models` and stats report.
pub(crate) struct GlmVariantTables<'a> {
    sites: BTreeMap<SiteKey, SiteTable<'a>>,
    hidden: usize,
    /// The bound kernel (modules loaded on the coordinator device before any capture).
    steer: cuteafd_ffi::steer_rows::SteerRows<'a>,
    /// Table slots past base: the largest variant index.
    slots: usize,
    pub records: Vec<VariantRecord>,
    pub plans: Vec<VariantPlan>,
}

/// The configured site keys: every key some plan edits.
pub(crate) fn configured_sites(plans: &[VariantPlan]) -> Vec<SiteKey> {
    let mut keys: Vec<SiteKey> = plans.iter().flat_map(|p| p.sites.iter().map(|(key, _)| *key)).collect();
    keys.sort();
    keys.dedup();
    keys
}

/// Device bytes the tables take (charged in admission before the KV pool is sized).
pub(crate) fn table_bytes(plans: &[VariantPlan], hidden: usize) -> u64 {
    configured_sites(plans).into_iter().map(|key| {
        let (a, b, k, _) = site_tables(plans, key, hidden);
        [a.len() * 4, b.len() * 4, k.len() * 4].into_iter().map(|bytes| bytes.max(256) as u64).sum::<u64>()
    }).sum()
}

impl<'a> GlmVariantTables<'a> {
    /// Upload every configured site's tables on the current device (outside the weight
    /// ledger; admission charged [`table_bytes`]).
    pub fn upload(library: &'a NativeLibrary, plans: Vec<VariantPlan>, records: Vec<VariantRecord>, hidden: usize)
        -> Result<Self> {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("weight_variants");
        ensure!(plans.iter().all(|p| p.hidden as usize == hidden), "variant tables do not match hidden {hidden}");
        let mut sites = BTreeMap::new();
        for key in configured_sites(&plans) {
            let (a, b, k, k_max) = site_tables(&plans, key, hidden);
            let put = |values: &[u8]| -> Result<DeviceAllocation<'a>> {
                let dev = DeviceAllocation::new(library, values.len().max(256))?;
                library.copy_h2d(cuteafd_ffi::CuteafdDeviceBuffer { bytes: values.len(), ..dev.buffer }, values)?;
                Ok(dev)
            };
            sites.insert(key, SiteTable { a: put(f32_bytes(&a))?, b: put(f32_bytes(&b))?, k: put(i32_bytes(&k))?, k_max });
        }
        let slots = plans.iter().map(|p| usize::from(p.index)).max().unwrap_or(0);
        Ok(Self { sites, hidden, steer: library.steer_rows()?, slots, records, plans })
    }

    pub fn configured(&self, key: SiteKey) -> bool { self.sites.contains_key(&key) }
    pub fn site_count(&self) -> usize { self.sites.len() }

    /// Steer `rows` BF16 rows of `x` (row stride `hidden`) at `key`, if any variant edits it.
    pub fn launch(&self, key: SiteKey, x: *mut c_void, rows: usize, vidx: *const c_void, stream: *mut c_void) -> Result<()> {
        let Some(site) = self.sites.get(&key) else { return Ok(()) };
        let shape = cuteafd_ffi::steer_rows::SteerRowsShape { rows, h: self.hidden, ld: self.hidden, k_max: site.k_max,
            variants: self.slots };
        // SAFETY: `x` holds `rows` live BF16 rows of `hidden` columns and `vidx` `rows` i32 ids on the
        // binding device, ordered on `stream`; the tables are this struct's live allocations of the
        // §4 shapes (DeviceAllocation is 256-byte aligned), and `x` overlaps none of them.
        unsafe {
            self.steer.launch_raw(x.cast(), vidx.cast(), site.a.buffer.ptr.cast_const().cast(),
                site.b.buffer.ptr.cast_const().cast(), site.k.buffer.ptr.cast_const().cast(), shape, stream)
        }.with_context(|| format!("steer_rows at {key}"))
    }
}

/// The launch's derived variants: the available plans, their records and table bytes.
pub(crate) struct Derived {
    pub plans: Vec<VariantPlan>,
    pub records: Vec<VariantRecord>,
    pub bytes: u64,
}

/// Derive and verify every `--variant` against the served snapshot (or its `base=`).
/// A variant whose tensors fail verification is unavailable; base serving continues.
pub(crate) fn derive(snapshot: &std::path::Path, served_id: &str, specs: &[(u8, cuteafd_core::variants::VariantSpec)],
    hidden: usize) -> Result<Derived> {
    let parallel = std::thread::available_parallelism().map_or(4, |n| (n.get() / 8).clamp(1, 4));
    let (mut plans, mut records) = (Vec::new(), Vec::new());
    for (index, spec) in specs {
        let outcome = cuteafd_loader::variants::derive_variant(snapshot, served_id, *index, spec, parallel)
            .with_context(|| format!("variant {}", spec.name))?;
        let verify = outcome.verify_summary();
        match outcome.plan {
            Some(plan) => {
                ensure!(plan.hidden as usize == hidden, "variant {}: hidden {} differs from the model's {hidden}", spec.name, plan.hidden);
                tracing::info!(variant = %spec.name, id = %plan.public_id, sites = plan.sites.len(), k_max = plan.k_max(),
                    namespace = format_args!("{:016x}", plan.namespace), %verify, "weight variant available");
                records.push(VariantRecord::of(&plan, served_id, verify));
                plans.push(plan);
            }
            None => tracing::warn!(variant = %spec.name, rejection = outcome.rejection.as_deref().unwrap_or("rejected"),
                "weight variant unavailable: not an output-side low-rank edit; a weight-set variant is required"),
        }
    }
    let bytes = table_bytes(&plans, hidden);
    Ok(Derived { plans, records, bytes })
}

/// All rows of a step share one variant (`Some`), or they mix (`None`).
pub(crate) fn uniform_variant(variants: &[i32]) -> Option<u8> {
    let first = *variants.first()?;
    variants.iter().all(|&v| v == first).then(|| u8::try_from(first).ok()).flatten()
}

fn f32_bytes(values: &[f32]) -> &[u8] {
    // SAFETY: f32 has no padding and u8 has alignment 1; the slice covers the same memory.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}
fn i32_bytes(values: &[i32]) -> &[u8] {
    // SAFETY: as above, for i32.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_core::variants::{SiteKind, SiteOp};

    fn plan(index: u8, keys: &[(u16, SiteKind)], k: u8, h: usize) -> VariantPlan {
        let sites = keys.iter().map(|&(layer, kind)| (SiteKey { layer, kind }, SiteOp::Steer { k,
            a: (0..usize::from(k) * h).map(|i| i as f32).collect(), b: vec![0.5; usize::from(k) * h] })).collect();
        VariantPlan::new(index, format!("v{index}"), format!("m-v{index}"), h as u32, sites).unwrap()
    }

    #[test]
    fn sites_are_the_union_and_bytes_follow_the_largest_rank() {
        let h = 128;
        let a = plan(1, &[(2, SiteKind::Attn), (2, SiteKind::DenseFfn), (3, SiteKind::Ffn)], 1, h);
        let b = plan(2, &[(3, SiteKind::Ffn), (4, SiteKind::SharedExpert)], 2, h);
        let keys = configured_sites(&[a.clone(), b.clone()]);
        assert_eq!(keys.len(), 4);
        // L3 ffn: both variants, k_max 2, slots 3: a and b are 3·2·128 f32 each, k is 3 i32 (256-byte floor).
        let l3 = SiteKey { layer: 3, kind: SiteKind::Ffn };
        let (ta, _, k, k_max) = site_tables(&[a.clone(), b.clone()], l3, h);
        assert_eq!((ta.len(), k.as_slice(), k_max), (3 * 2 * h, &[0, 1, 2][..], 2));
        // Slots run to the largest index (3) at every site; k_max is per site.
        let rank1 = 2 * (3 * h * 4) + 256;
        let rank2 = 2 * (3 * 2 * h * 4) + 256;
        let expected = 2 * rank1 + 2 * rank2;
        assert_eq!(table_bytes(&[a, b], h), expected as u64);
    }

    #[test]
    fn uniform_variant_detects_mixed_steps() {
        assert_eq!(uniform_variant(&[1, 1, 1]), Some(1));
        assert_eq!(uniform_variant(&[0, 0]), Some(0));
        assert_eq!(uniform_variant(&[0, 1]), None);
        assert_eq!(uniform_variant(&[]), None);
    }
}
