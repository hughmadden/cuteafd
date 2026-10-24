//! Per-request weight variants: a request picks a variant by model id, and
//! only its rows get `y -= A (Bᵀ y)` on the BF16 output of each edited
//! sublayer before that output enters the residual kernel. No second weight
//! set, no wire change, no graph-key axis; base rows (`variant == 0`) are
//! never written.
//!
//! Interface contract: `designs/weight-variants-contract.md` (v1, 2026-10-24).
//! This module holds the shared types (§2), the `--variant` flag (§6), the
//! prefix namespace (§5); the steering kernel's host reference (§4) is re-exported from
//! `crate::steer_rows`.
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
    /// A stored weight delta: `y += ΔW x` on variant rows, where `x` is the
    /// input of the tensor that writes the site (`Attn`: the `o_proj` input;
    /// `SharedExpert`/`DenseFfn`: the `down_proj` input). For an edit that is
    /// not low-rank (GLM Flash dealignai L44). `weight` is the BF16 bits of
    /// `ΔW = W' − W`, `[rows][cols]` with `rows` = hidden; the family picks
    /// the device format.
    Delta { rows: u32, cols: u32, weight: Vec<u16> },
}
impl SiteOp {
    /// The rank-1 projection `y -= alpha r (rᵀ y)`.
    pub fn rank1(alpha: f32, r: &[f32]) -> Self {
        Self::Steer { k: 1, a: r.iter().map(|v| alpha * v).collect(), b: r.to_vec() }
    }
    pub fn rank(&self) -> usize {
        match self { Self::Steer { k, .. } => usize::from(*k), Self::Delta { .. } => 0 }
    }
    pub fn is_delta(&self) -> bool { matches!(self, Self::Delta { .. }) }
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
    #[error("derivation: {0}")]
    Derive(String),
    #[error("tensor is unchanged")]
    Unchanged,
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
                SiteOp::Delta { rows, cols, weight } => {
                    if site.kind == SiteKind::Ffn { return Err(bad("a delta needs one tensor's input; Ffn combines two".into())); }
                    if *rows != hidden || *cols == 0 || weight.len() != *rows as usize * *cols as usize {
                        return Err(bad(format!("delta holds {} values as {rows}x{cols}, expected {hidden} rows", weight.len())));
                    }
                    if weight.iter().any(|&v| !bf16_to_f32(v).is_finite()) { return Err(bad("non-finite delta value".into())); }
                }
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
    /// Delta values (`rows · cols` per delta site); bytes per value depend on the family's device format.
    pub fn delta_values(&self) -> u64 {
        self.sites.iter().map(|(_, op)| match op { SiteOp::Delta { weight, .. } => weight.len() as u64, _ => 0 }).sum()
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
            SiteOp::Delta { rows, cols, weight } => {
                hash.update([1, 0]);
                hash.update(rows.to_le_bytes());
                hash.update(cols.to_le_bytes());
                for v in weight { hash.update(v.to_le_bytes()); }
            }
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

/// Per-variant serving counters (contract §7), indexed by variant (0 = base).
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct VariantCounters {
    pub requests: u64,
    pub prompt_tokens: u64,
    pub generated_tokens: u64,
    /// Rows run with this variant's steering (prefill and decode/verify rows).
    pub steered_rows: u64,
    pub prefix_hits: u64,
    pub prefix_lookups: u64,
    pub drafted: u64,
    pub accepted: u64,
}

/// `/v1/stats` `variants`: the configured variants' records and their counters, plus base.
#[derive(Debug, Clone, Default)]
pub struct VariantStats {
    records: Vec<VariantRecord>,
    counters: [VariantCounters; MAX_VARIANTS + 1],
}
impl VariantStats {
    pub fn new(records: Vec<VariantRecord>) -> Self { Self { records, ..Self::default() } }
    pub fn is_empty(&self) -> bool { self.records.is_empty() }
    /// A variant's name for logs (`base` for 0 or an unknown index).
    pub fn name(&self, index: u8) -> &str {
        self.records.iter().find(|r| r.index == index).map_or("base", |r| r.name.as_str())
    }
    fn at(&mut self, index: u8) -> &mut VariantCounters { &mut self.counters[usize::from(index).min(MAX_VARIANTS)] }
    pub fn admitted(&mut self, index: u8, prompt_tokens: usize, looked_up: bool, hit: bool) {
        let c = self.at(index);
        c.requests += 1;
        c.prompt_tokens += prompt_tokens as u64;
        c.prefix_lookups += u64::from(looked_up);
        c.prefix_hits += u64::from(looked_up && hit);
    }
    pub fn steered(&mut self, index: u8, rows: usize) {
        if index != 0 { self.at(index).steered_rows += rows as u64; }
    }
    pub fn finished(&mut self, index: u8, generated: usize, drafted: u64, accepted: u64) {
        let c = self.at(index);
        c.generated_tokens += generated as u64;
        c.drafted += drafted;
        c.accepted += accepted;
    }
    /// `{name: {requests, prompt_tokens, generated_tokens, steered_rows, prefix_hits,
    /// prefix_lookups, draft_acceptance, sites, k_max}}`, base included for comparison.
    pub fn json(&self) -> serde_json::Value {
        let entry = |c: &VariantCounters, sites: usize, k_max: usize| serde_json::json!({
            "requests": c.requests, "prompt_tokens": c.prompt_tokens, "generated_tokens": c.generated_tokens,
            "steered_rows": c.steered_rows, "prefix_hits": c.prefix_hits, "prefix_lookups": c.prefix_lookups,
            "draft_acceptance": (c.drafted > 0).then(|| c.accepted as f64 / c.drafted as f64),
            "sites": sites, "k_max": k_max});
        let mut out = serde_json::Map::new();
        out.insert("base".into(), entry(&self.counters[0], 0, 0));
        for record in &self.records {
            out.insert(record.name.clone(), entry(&self.counters[usize::from(record.index)], record.sites, record.k_max));
        }
        serde_json::Value::Object(out)
    }
}

// The kernel's host reference (one copy, `crate::steer_rows`): the same fixed reduction
// order as `cuteafd_steer_rows`, so CPU tests match the GPU bit for bit.
pub use crate::steer_rows::{add_rows_ref, bf16_to_f32, f32_to_bf16_rne as bf16_rne, steer_rows_ref};

/// Per-variant switch of one delta site, `[V+1]` with slot 0 zero: `on[v]` is 1
/// when variant `v` stores a delta at `key`.
pub fn delta_switch(plans: &[VariantPlan], key: SiteKey) -> Vec<i32> {
    let slots = plans.iter().map(|p| usize::from(p.index)).max().unwrap_or(0) + 1;
    let mut on = vec![0i32; slots];
    for plan in plans { if plan.site(key).is_some_and(SiteOp::is_delta) { on[usize::from(plan.index)] = 1; } }
    on
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
        let delta = |rows: u32, cols: u32| SiteOp::Delta { rows, cols, weight: vec![bf16_rne(0.5); (rows * cols) as usize] };
        let d = VariantPlan::new(1, "x", "x", 4, vec![(key(1, SiteKind::Attn), delta(4, 8)),
            (key(1, SiteKind::SharedExpert), SiteOp::rank1(1.0, &r))]).unwrap();
        assert_eq!((d.k_max(), d.delta_values(), d.table_bytes()), (1, 32, 2 * 4 * 4));
        assert_ne!(d.namespace, VariantPlan::new(1, "x", "x", 4, vec![(key(1, SiteKind::Attn), delta(4, 4)),
            (key(1, SiteKind::SharedExpert), SiteOp::rank1(1.0, &r))]).unwrap().namespace, "delta bytes are in the namespace");
        assert!(VariantPlan::new(1, "x", "x", 4, vec![(key(1, SiteKind::Attn), delta(3, 8))]).is_err(), "rows must be hidden");
        assert!(VariantPlan::new(1, "x", "x", 4, vec![(key(1, SiteKind::Ffn), delta(4, 8))]).is_err(), "Ffn has no single input");
        assert!(VariantPlan::new(1, "x", "x", 4, vec![(key(1, SiteKind::Attn),
            SiteOp::Delta { rows: 4, cols: 2, weight: vec![0x7fc0; 8] })]).is_err(), "non-finite");
    }

    #[test]
    fn stats_count_per_variant_and_report_base() {
        let record = VariantRecord { index: 2, name: "abl".into(), public_id: "m-abl".into(), base: "m".into(),
            sites: 96, k_max: 1, verify: serde_json::json!({}) };
        let mut stats = VariantStats::new(vec![record]);
        stats.admitted(2, 100, true, true);
        stats.admitted(0, 50, true, false);
        stats.steered(2, 99);
        stats.steered(0, 7);
        stats.finished(2, 20, 10, 6);
        let json = stats.json();
        assert_eq!(json["abl"]["requests"], 1);
        assert_eq!(json["abl"]["steered_rows"], 99);
        assert_eq!(json["abl"]["prefix_hits"], 1);
        assert_eq!(json["abl"]["draft_acceptance"], 0.6);
        assert_eq!(json["abl"]["sites"], 96);
        assert_eq!(json["base"]["steered_rows"], 0);
        assert_eq!(json["base"]["prefix_hits"], 0);
        assert!(json["base"]["draft_acceptance"].is_null());
        assert_eq!((stats.name(2), stats.name(0), stats.name(5)), ("abl", "base", "base"));
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
        let h = 128;
        let norm = ((1..=h).map(|i| (i * i) as f32).sum::<f32>()).sqrt();
        let r: Vec<f32> = (0..h).map(|i| (i as f32 + 1.0) / norm).collect();
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
    fn add_rows_writes_only_switched_rows() {
        let h = 4;
        let mut y: Vec<u16> = (0..3 * h).map(|i| bf16_rne(i as f32 - 5.0)).collect();
        y[0] = 0x8000; // -0.0 on a base row stays -0.0 (never written)
        let before = y.clone();
        let d: Vec<f32> = (0..3 * h).map(|i| 0.25 * i as f32).collect();
        // Row 1 is variant 2 (no delta here), row 2 variant 1 (stores the delta).
        add_rows_ref(&mut y, h, &d, h, 3, h, &[0, 2, 1], &[0, 1, 0]);
        assert_eq!(&y[..2 * h], &before[..2 * h], "base row and a variant without the delta untouched");
        for c in 0..h { assert_eq!(bf16_to_f32(y[2 * h + c]), bf16_to_f32(before[2 * h + c]) + d[2 * h + c]); }
        let plan = VariantPlan::new(2, "d", "m-d", h as u32, vec![(key(7, SiteKind::Attn),
            SiteOp::Delta { rows: h as u32, cols: 2, weight: vec![0; 2 * h] })]).unwrap();
        assert_eq!(delta_switch(&[plan.clone()], key(7, SiteKind::Attn)), [0, 0, 1]);
        assert_eq!(delta_switch(&[plan], key(7, SiteKind::SharedExpert)), [0, 0, 0]);
    }

    #[test]
    fn host_reference_is_batch_invariant() {
        let h = 128;
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
