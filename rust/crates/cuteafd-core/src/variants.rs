//! Per-request weight variants: a request picks a variant by model id, and
//! only its rows get `y -= A (Bᵀ y)` on the BF16 output of each edited
//! sublayer before that output enters the residual kernel. No second weight
//! set, no wire change, no graph-key axis; base rows (`variant == 0`) are
//! never written.
//!
//! Interface contract: `designs/weight-variants-contract.md` (v1, 2026-10-24).
//! This module holds the shared types (§2), the `--variant` flag (§6), the
//! prefix namespace (§5) and the host reference of the steering kernel (§4).
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::fmt;

/// Configured variants per launch, besides the base (index 0).
pub const MAX_VARIANTS: usize = 7;
/// Largest rank a steering site may carry (the general kernel export's bound).
pub const MAX_STEER_RANK: usize = 32;

/// Where a family applies a site's operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SiteKind {
    /// Attention output projection (`o_proj`, `wo_b`).
    Attn,
    /// FFN output after the routed and shared combine.
    Ffn,
    /// Shared expert output, before the combine adds it.
    SharedExpert,
    /// A dense (non-MoE) layer's FFN output.
    DenseFfn,
}
impl SiteKind {
    fn tag(self) -> u8 {
        match self { Self::Attn => 0, Self::Ffn => 1, Self::SharedExpert => 2, Self::DenseFfn => 3 }
    }
    pub fn name(self) -> &'static str {
        match self { Self::Attn => "attn", Self::Ffn => "ffn", Self::SharedExpert => "shared_expert", Self::DenseFfn => "dense_ffn" }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
pub struct SiteKey {
    pub layer: u16,
    pub kind: SiteKind,
}
impl fmt::Display for SiteKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "L{}.{}", self.layer, self.kind.name()) }
}

/// What a variant does at one site.
#[derive(Debug, Clone, PartialEq)]
pub enum SiteOp {
    /// `y -= A (Bᵀ y)`: `a` and `b` are `[k][hidden]` row-major. A rank-1
    /// projection `W' = (I − α r rᵀ) W` is `k = 1`, `a = α·r`, `b = r`.
    Steer { k: u8, a: Vec<f32>, b: Vec<f32> },
    /// Reserved for a stored weight delta (`y += ΔW x` on variant rows; GLM
    /// Flash L44). Not built in v1.
    Delta {},
}
impl SiteOp {
    /// The rank-1 projection `y -= alpha r (rᵀ y)`.
    pub fn rank1(alpha: f32, r: &[f32]) -> Self {
        Self::Steer { k: 1, a: r.iter().map(|v| alpha * v).collect(), b: r.to_vec() }
    }
    pub fn rank(&self) -> usize {
        match self { Self::Steer { k, .. } => usize::from(*k), Self::Delta {} => 0 }
    }
}

/// One configured variant, ready for a family to upload.
#[derive(Debug, Clone, PartialEq)]
pub struct VariantPlan {
    /// 1..=7, in `--variant` order.
    pub index: u8,
    pub name: String,
    pub public_id: String,
    pub hidden: u32,
    pub sites: Vec<(SiteKey, SiteOp)>,
    /// Prefix-cache namespace ([`namespace`] of `sites`); never 0.
    pub namespace: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum VariantError {
    #[error("--variant {0}")]
    Flag(String),
    #[error("variant {name}: site {site}: {reason}")]
    Site { name: String, site: SiteKey, reason: String },
    #[error("variant {name}: {reason}")]
    Plan { name: String, reason: String },
}

impl VariantPlan {
    /// Validate shapes and sort sites by key, then compute the namespace.
    pub fn new(index: u8, name: impl Into<String>, public_id: impl Into<String>, hidden: u32,
        mut sites: Vec<(SiteKey, SiteOp)>) -> Result<Self, VariantError> {
        let name = name.into();
        if !(1..=MAX_VARIANTS as u8).contains(&index) {
            return Err(VariantError::Plan { name, reason: format!("index {index} is outside 1..={MAX_VARIANTS}") });
        }
        sites.sort_by_key(|(key, _)| *key);
        if let Some(pair) = sites.windows(2).find(|pair| pair[0].0 == pair[1].0) {
            return Err(VariantError::Site { name, site: pair[0].0, reason: "listed twice".into() });
        }
        let h = hidden as usize;
        for (site, op) in &sites {
            let bad = |reason: String| VariantError::Site { name: name.clone(), site: *site, reason };
            match op {
                SiteOp::Steer { k, a, b } => {
                    let k = usize::from(*k);
                    if k == 0 || k > MAX_STEER_RANK { return Err(bad(format!("rank {k} is outside 1..={MAX_STEER_RANK}"))); }
                    if a.len() != k * h || b.len() != k * h {
                        return Err(bad(format!("tables hold {}/{} values, expected {k}x{h}", a.len(), b.len())));
                    }
                    if a.iter().chain(b).any(|v| !v.is_finite()) { return Err(bad("non-finite table value".into())); }
                }
                SiteOp::Delta {} => return Err(bad("delta sites are not supported in v1".into())),
            }
        }
        let namespace = namespace(hidden, &sites);
        Ok(Self { index, name, public_id: public_id.into(), hidden, sites, namespace })
    }
    /// The largest rank over this variant's sites.
    pub fn k_max(&self) -> usize { self.sites.iter().map(|(_, op)| op.rank()).max().unwrap_or(0) }
    pub fn site(&self, key: SiteKey) -> Option<&SiteOp> {
        self.sites.binary_search_by_key(&key, |(k, _)| *k).ok().map(|i| &self.sites[i].1)
    }
    /// Device table bytes: `a` and `b` at the variant's own rank per site.
    pub fn table_bytes(&self) -> u64 {
        self.sites.iter().map(|(_, op)| 2 * op.rank() as u64 * u64::from(self.hidden) * 4).sum()
    }
}

/// Prefix-cache namespace of a variant (§5): the first 8 bytes (LE) of the
/// SHA-256 of its serialized `(SiteKey, SiteOp)` list, so changed tables can
/// never restore old pages. Never 0, which is the base namespace.
pub fn namespace(hidden: u32, sites: &[(SiteKey, SiteOp)]) -> u64 {
    let mut hash = Sha256::new();
    hash.update(b"cuteafd-variant-v1");
    hash.update(hidden.to_le_bytes());
    hash.update((sites.len() as u64).to_le_bytes());
    for (key, op) in sites {
        hash.update(key.layer.to_le_bytes());
        hash.update([key.kind.tag()]);
        match op {
            SiteOp::Steer { k, a, b } => {
                hash.update([0, *k]);
                for v in a.iter().chain(b) { hash.update(v.to_bits().to_le_bytes()); }
            }
            SiteOp::Delta {} => hash.update([1, 0]),
        }
    }
    let digest = hash.finalize();
    let value = u64::from_le_bytes(digest[..8].try_into().unwrap());
    if value == 0 { 1 } else { value }
}

/// One `--variant name=<n>,source=<path>[,base=<path>][,id=<public id>][,mode=<m>]` (§6).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VariantSpec {
    pub name: String,
    /// Snapshot directory or safetensors file holding the edited tensors.
    pub source: String,
    /// The checkpoint the edit is measured against; the served snapshot when absent.
    pub base: Option<String>,
    /// Public model id; `<served id>-<name>` when absent.
    pub id: Option<String>,
    /// Family-specific mode (GLM-5.3 full: `ffn` or `shared-only`).
    pub mode: Option<String>,
}
impl VariantSpec {
    pub fn public_id(&self, served: &str) -> String {
        self.id.clone().unwrap_or_else(|| format!("{served}-{}", self.name))
    }
}
impl std::str::FromStr for VariantSpec {
    type Err = VariantError;
    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let bad = |reason: String| VariantError::Flag(reason);
        let (mut name, mut source, mut base, mut id, mut mode) = (None, None, None, None, None);
        for part in text.split(',') {
            let (key, value) = part.split_once('=').ok_or_else(|| bad(format!("{part:?} is not key=value")))?;
            if value.is_empty() { return Err(bad(format!("{key} is empty"))); }
            let slot = match key {
                "name" => &mut name, "source" => &mut source, "base" => &mut base, "id" => &mut id, "mode" => &mut mode,
                other => return Err(bad(format!("unknown key {other:?} (name, source, base, id, mode)"))),
            };
            if slot.replace(value.to_string()).is_some() { return Err(bad(format!("{key} given twice"))); }
        }
        let name = name.ok_or_else(|| bad("needs name=".into()))?;
        if !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_') {
            return Err(bad(format!("name {name:?} must be [A-Za-z0-9_-]")));
        }
        if id.as_deref().is_some_and(|id| id.chars().any(char::is_whitespace)) {
            return Err(bad("id contains whitespace".into()));
        }
        let source = source.ok_or_else(|| bad(format!("{name}: needs source=")))?;
        Ok(Self { name, source, base, id, mode })
    }
}

/// Check a launch's variant list: at most [`MAX_VARIANTS`], unique names and ids.
pub fn validate_specs(specs: &[VariantSpec], served: &str) -> Result<(), VariantError> {
    if specs.len() > MAX_VARIANTS {
        return Err(VariantError::Flag(format!("at most {MAX_VARIANTS} variants, got {}", specs.len())));
    }
    for (i, spec) in specs.iter().enumerate() {
        let id = spec.public_id(served);
        if id == served { return Err(VariantError::Flag(format!("{}: id equals the served id {served}", spec.name))); }
        for other in &specs[..i] {
            if other.name == spec.name { return Err(VariantError::Flag(format!("name {} given twice", spec.name))); }
            if other.public_id(served) == id { return Err(VariantError::Flag(format!("id {id} given twice"))); }
        }
    }
    Ok(())
}

/// What `/v1/models`, the request log and `/v1/stats` say about a variant.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct VariantRecord {
    pub index: u8,
    pub name: String,
    pub public_id: String,
    /// The served (base) model id.
    pub base: String,
    pub sites: usize,
    pub k_max: usize,
    /// Verification summary from the derivation (family-defined fields).
    pub verify: serde_json::Value,
}
impl VariantRecord {
    pub fn of(plan: &VariantPlan, base: &str, verify: serde_json::Value) -> Self {
        Self { index: plan.index, name: plan.name.clone(), public_id: plan.public_id.clone(), base: base.into(),
            sites: plan.sites.len(), k_max: plan.k_max(), verify }
    }
    /// The `"variant"` object a `/v1/models` record carries.
    pub fn model_field(&self) -> serde_json::Value {
        serde_json::json!({"name": self.name, "base": self.base, "sites": self.sites, "k_max": self.k_max, "verify": self.verify})
    }
}

/// BF16 round-to-nearest-even from FP32 (NaN stays NaN).
pub fn bf16_rne(value: f32) -> u16 {
    let bits = value.to_bits();
    if value.is_nan() { return ((bits >> 16) | 0x40) as u16; }
    let round = 0x7fff + ((bits >> 16) & 1);
    (bits.wrapping_add(round) >> 16) as u16
}
pub fn bf16_to_f32(value: u16) -> f32 { f32::from_bits(u32::from(value) << 16) }

/// Host reference of `cuteafd_steer_rows` (§4): `x` is BF16 bits `[rows, ld]`,
/// steered in place on its first `h` columns. `a`/`b` are `[V+1][k_max][h]`
/// (slot 0 unused), `k` is `[V+1]`. Per row the reduction runs in FP32 in
/// column order, so a row's result never depends on other rows.
#[allow(clippy::too_many_arguments)]
pub fn steer_rows_ref(x: &mut [u16], ld: usize, rows: usize, h: usize, vidx: &[i32], a: &[f32], b: &[f32],
    k: &[i32], k_max: usize) {
    assert!(ld >= h && vidx.len() >= rows && (rows == 0 || x.len() >= (rows - 1) * ld + h));
    assert_eq!(a.len(), k.len() * k_max * h);
    assert_eq!(b.len(), a.len());
    let mut t = vec![0f32; k_max];
    for r in 0..rows {
        let v = vidx[r];
        if v <= 0 || v as usize >= k.len() { continue; }
        let v = v as usize;
        let kv = k[v].clamp(0, k_max as i32) as usize;
        if kv == 0 { continue; }
        let row = &mut x[r * ld..r * ld + h];
        for (j, tj) in t.iter_mut().enumerate().take(kv) {
            let bj = &b[(v * k_max + j) * h..][..h];
            *tj = row.iter().zip(bj).map(|(&x, &b)| bf16_to_f32(x) * b).fold(0.0, |s, p| s + p);
        }
        for (c, x) in row.iter_mut().enumerate() {
            let mut y = bf16_to_f32(*x);
            for (j, tj) in t.iter().enumerate().take(kv) { y -= a[(v * k_max + j) * h + c] * tj; }
            *x = bf16_rne(y);
        }
    }
}

/// Pack plans into the kernel's per-site table layout (§4): for one site,
/// `(a, b, k, k_max)` with slot 0 zero and slots in variant index order.
/// `k_max` is the largest rank at this site (1 for the fast path).
pub fn site_tables(plans: &[VariantPlan], key: SiteKey, hidden: usize) -> (Vec<f32>, Vec<f32>, Vec<i32>, usize) {
    let slots = plans.iter().map(|p| usize::from(p.index)).max().unwrap_or(0) + 1;
    let k_max = plans.iter().filter_map(|p| p.site(key)).map(SiteOp::rank).max().unwrap_or(0).max(1);
    let (mut a, mut b, mut k) = (vec![0f32; slots * k_max * hidden], vec![0f32; slots * k_max * hidden], vec![0i32; slots]);
    for plan in plans {
        if let Some(SiteOp::Steer { k: rank, a: pa, b: pb }) = plan.site(key) {
            let v = usize::from(plan.index);
            let n = usize::from(*rank) * hidden;
            a[v * k_max * hidden..][..n].copy_from_slice(pa);
            b[v * k_max * hidden..][..n].copy_from_slice(pb);
            k[v] = i32::from(*rank);
        }
    }
    (a, b, k, k_max)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(layer: u16, kind: SiteKind) -> SiteKey { SiteKey { layer, kind } }

    #[test]
    fn spec_parses_every_key_and_rejects_bad_input() {
        let spec: VariantSpec = "name=abliterated,source=/m/keys,base=/m/base,id=glm-ab,mode=shared-only".parse().unwrap();
        assert_eq!(spec, VariantSpec { name: "abliterated".into(), source: "/m/keys".into(), base: Some("/m/base".into()),
            id: Some("glm-ab".into()), mode: Some("shared-only".into()) });
        let short: VariantSpec = "name=x,source=/s".parse().unwrap();
        assert_eq!(short.public_id("zai-org/GLM-5.3"), "zai-org/GLM-5.3-x");
        for bad in ["source=/s", "name=x", "name=x,source=/s,colour=red", "name=x,source=/s,source=/t", "name=a b,source=/s",
            "name=x,source=", "name=x,source=/s,id=a b", "garbage"] {
            assert!(bad.parse::<VariantSpec>().is_err(), "{bad}");
        }
    }

    #[test]
    fn launch_list_limits() {
        let spec = |n: &str| VariantSpec { name: n.into(), source: "/s".into(), base: None, id: None, mode: None };
        let specs: Vec<_> = (0..7).map(|i| spec(&format!("v{i}"))).collect();
        validate_specs(&specs, "m").unwrap();
        let mut eight = specs.clone();
        eight.push(spec("v7"));
        assert!(validate_specs(&eight, "m").is_err());
        assert!(validate_specs(&[spec("a"), spec("a")], "m").is_err());
        let mut same_id = vec![spec("a"), spec("b")];
        same_id[1].id = Some("m-a".into());
        assert!(validate_specs(&same_id, "m").is_err());
        let mut served = vec![spec("a")];
        served[0].id = Some("m".into());
        assert!(validate_specs(&served, "m").is_err());
    }

    #[test]
    fn plan_validates_and_namespaces_differ() {
        let r = vec![0.6f32, 0.8, 0.0, 0.0];
        let a = VariantPlan::new(1, "a", "m-a", 4, vec![(key(3, SiteKind::Attn), SiteOp::rank1(3.0, &r)),
            (key(2, SiteKind::Ffn), SiteOp::rank1(3.0, &r))]).unwrap();
        assert_eq!(a.sites[0].0, key(2, SiteKind::Ffn), "sites sort by key");
        assert_eq!(a.k_max(), 1);
        assert_eq!(a.table_bytes(), 2 * 2 * 4 * 4);
        assert_ne!(a.namespace, 0);
        let b = VariantPlan::new(1, "a", "m-a", 4, vec![(key(3, SiteKind::Attn), SiteOp::rank1(3.5, &r)),
            (key(2, SiteKind::Ffn), SiteOp::rank1(3.0, &r))]).unwrap();
        assert_ne!(a.namespace, b.namespace, "a changed table changes the namespace");
        let same = VariantPlan::new(2, "renamed", "other", 4, a.sites.clone()).unwrap();
        assert_eq!(a.namespace, same.namespace, "the namespace depends on tables only");
        assert!(VariantPlan::new(0, "x", "x", 4, vec![]).is_err());
        assert!(VariantPlan::new(8, "x", "x", 4, vec![]).is_err());
        assert!(VariantPlan::new(1, "x", "x", 4, vec![(key(1, SiteKind::Attn), SiteOp::rank1(1.0, &r[..3]))]).is_err());
        assert!(VariantPlan::new(1, "x", "x", 4, vec![(key(1, SiteKind::Attn), SiteOp::rank1(1.0, &r)),
            (key(1, SiteKind::Attn), SiteOp::rank1(2.0, &r))]).is_err());
        assert!(VariantPlan::new(1, "x", "x", 4, vec![(key(1, SiteKind::Attn), SiteOp::Delta {})]).is_err());
    }

    #[test]
    fn bf16_rounding_is_nearest_even() {
        assert_eq!(bf16_rne(1.0), 0x3f80);
        // 1 + 2^-8 is exactly halfway between 1 and the next BF16: ties to even (1.0).
        assert_eq!(bf16_rne(1.0 + 1.0 / 256.0), 0x3f80);
        // 1 + 3·2^-8 ties between 0x3f81 and 0x3f82: even is 0x3f82.
        assert_eq!(bf16_rne(1.0 + 3.0 / 256.0), 0x3f82);
        assert_eq!(bf16_to_f32(bf16_rne(-2.5)), -2.5);
    }

    #[test]
    fn host_reference_matches_fp32_math_and_skips_base_rows() {
        let h = 8;
        let r: Vec<f32> = (0..h).map(|i| (i as f32 + 1.0) / 14.282857).collect();
        let plan = VariantPlan::new(1, "a", "m-a", h as u32, vec![(key(0, SiteKind::Attn), SiteOp::rank1(3.0, &r))]).unwrap();
        let (a, b, k, k_max) = site_tables(&[plan], key(0, SiteKind::Attn), h);
        assert_eq!((k.as_slice(), k_max), (&[0, 1][..], 1));
        let xs: Vec<f32> = (0..2 * h).map(|i| (i as f32 * 0.37).sin()).collect();
        let mut x: Vec<u16> = xs.iter().map(|&v| bf16_rne(v)).collect();
        let before = x.clone();
        steer_rows_ref(&mut x, h, 2, h, &[0, 1], &a, &b, &k, k_max);
        assert_eq!(&x[..h], &before[..h], "base row untouched");
        let row: Vec<f32> = before[h..].iter().map(|&v| bf16_to_f32(v)).collect();
        let t: f32 = row.iter().zip(&r).map(|(x, r)| x * r).sum();
        for c in 0..h {
            let want = row[c] - 3.0 * r[c] * t;
            assert!((bf16_to_f32(x[h + c]) - want).abs() <= want.abs() / 128.0 + 1e-6, "col {c}");
        }
        // A site this variant does not edit (k = 0) leaves every row alone.
        let mut y = before.clone();
        steer_rows_ref(&mut y, h, 2, h, &[1, 1], &a, &b, &[0, 0], k_max);
        assert_eq!(y, before);
    }

    #[test]
    fn host_reference_is_batch_invariant() {
        let h = 16;
        let r: Vec<f32> = (0..2 * h).map(|i| ((i * 7 % 11) as f32 - 5.0) / 10.0).collect();
        let plan = VariantPlan::new(2, "k2", "m-k2", h as u32, vec![(key(4, SiteKind::Ffn),
            SiteOp::Steer { k: 2, a: r.iter().map(|v| v * 0.5).collect(), b: r.clone() })]).unwrap();
        let (a, b, k, k_max) = site_tables(&[plan], key(4, SiteKind::Ffn), h);
        assert_eq!(k_max, 2);
        let row: Vec<u16> = (0..h).map(|i| bf16_rne((i as f32).cos())).collect();
        let mut alone = row.clone();
        steer_rows_ref(&mut alone, h, 1, h, &[2], &a, &b, &k, k_max);
        let mut batch: Vec<u16> = (0..5 * h).map(|i| bf16_rne((i as f32 * 0.1).sin())).collect();
        batch[3 * h..4 * h].copy_from_slice(&row);
        steer_rows_ref(&mut batch, h, 5, h, &[0, 2, 0, 2, 2], &a, &b, &k, k_max);
        assert_eq!(&batch[3 * h..4 * h], &alone[..]);
    }
}
