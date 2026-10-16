//! Spark owners: manifests describe execution, checkpoint selection describes weights.
use crate::placement::SparkAssignment;
use cuteafd_core::memory_layout::{Basis, Category, Item};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use thiserror::Error;

#[derive(Debug, Error)]
#[error("V4.1 Spark inventory: {0}")]
pub struct SparkInventoryError(String);
type Result<T> = std::result::Result<T, SparkInventoryError>;
fn invalid(message: impl Into<String>) -> SparkInventoryError { SparkInventoryError(message.into()) }
fn add(values: impl IntoIterator<Item = u64>) -> Result<u64> {
    values.into_iter().try_fold(0u64, |a, b| a.checked_add(b).ok_or_else(|| invalid("byte sum overflow")))
}
fn mul(values: &[u64]) -> Result<u64> {
    values.iter().try_fold(1u64, |a, b| a.checked_mul(*b).ok_or_else(|| invalid("byte product overflow")))
}
fn number(value: &Value, key: &str) -> Result<u64> {
    value[key].as_u64().ok_or_else(|| invalid(format!("missing unsigned {key}")))
}
fn item(category: Category, name: &str, bytes: u64, basis: Basis) -> Item {
    Item::new(category, name, "", bytes, basis)
}

/// Every manifest is fingerprinted, including its object hashes and capacity state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SparkManifestProvenance {
    pub sha256: String,
    pub revision: String,
    pub capacity_rows: Vec<u32>,
    pub physical_sms: u32,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SparkMemoryDomains {
    pub device: Vec<Item>,
    pub pinned: Vec<Item>,
    pub host: Vec<Item>,
}
impl SparkMemoryDomains {
    pub fn total(&self) -> Result<u64> {
        add(self.device.iter().chain(&self.pinned).chain(&self.host).map(|i| i.bytes))
    }
}

/// Actual loader high-water owners, not a multiple of resident weights. Maxima
/// are across sequential layer loads; simultaneous banks/read lanes are summed
/// by the checkpoint loader before constructing this value.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SparkLoadInventory {
    pub device_staging_bytes: u64,
    pub pinned_staging_bytes: u64,
    pub host_read_scratch_bytes: u64,
    pub source: String,
}

impl SparkLoadInventory {
    pub fn native(catalog: &crate::OfficialV41Catalog, selections: &[crate::V41ExpertSelection]) -> Result<Self> {
        if selections.is_empty() { return Err(invalid("no selected native layer load plans")); }
        let mut staging = 0u64;
        let mut scratch = 0u64;
        for &selection in selections {
            let (bytes, read) = if catalog.nvfp4().is_some() {
                let plan = catalog.nvfp4_expert_staging(selection).map_err(|e| invalid(e.to_string()))?;
                (plan.staging_bytes(), plan.minimum_read_scratch_bytes())
            } else {
                let plan = catalog.expert_staging(selection).map_err(|e| invalid(e.to_string()))?;
                (plan.staging_bytes(), plan.minimum_read_scratch_bytes())
            };
            staging = staging.max(bytes as u64);
            scratch = scratch.max(read as u64);
        }
        Ok(Self { device_staging_bytes: staging, pinned_staging_bytes: mul(&[staging, 16])?,
            host_read_scratch_bytes: mul(&[scratch, 64, 16])?,
            source: "checkpoint staging plans; 16 reader lanes, 64 source rows per pread; sequential layers".into() })
    }

    pub fn exl3(catalog: &crate::OfficialV41Catalog, plans: &[crate::V41Exl3Residency]) -> Result<Self> {
        if plans.is_empty() { return Err(invalid("no selected EXL3 layer load plans")); }
        let mut pinned = 0u64;
        let mut read = 0u64;
        for plan in plans {
            if plan.loads.len() != plan.experts.checked_mul(9).ok_or_else(|| invalid("EXL3 load count overflow"))? {
                return Err(invalid("EXL3 staging inventory mismatch"));
            }
            let staging = plan.loads.chunks(9).map(|jobs| add(jobs.iter().map(|j| j.bytes as u64)))
                .collect::<Result<Vec<_>>>()?.into_iter().max().unwrap_or(0);
            let init = plan.buffers.iter().map(|b| mul(&[b.initial_words.len() as u64, 4]))
                .collect::<Result<Vec<_>>>()?.into_iter().max().unwrap_or(0);
            pinned = pinned.max(mul(&[staging.max(init), 16, 2])?);
            read = read.max(mul(&[plan.scratch_bytes(catalog).map_err(|e| invalid(e.to_string()))? as u64, 64, 16, 2])?);
        }
        Ok(Self { device_staging_bytes: 0, pinned_staging_bytes: pinned, host_read_scratch_bytes: read,
            source: "checkpoint EXL3 residency jobs; 9 jobs per expert, 16 reader lanes, 2 banks, 64 rows per pread; sequential layers".into() })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SparkExecutionInventory {
    pub compiled_capacity: u32,
    pub live_capacity: u32,
    pub retained_capacities: Vec<u32>,
    pub physical_sms: u32,
    pub logical_intermediate: u32,
    pub stored_intermediate: u32,
    pub manifests: Vec<SparkManifestProvenance>,
    pub retained: SparkMemoryDomains,
    pub load: SparkLoadInventory,
}

fn capacities(compiled: u32, live: u32, retained: &[u32]) -> Result<()> {
    if live == 0 || compiled < live || compiled > 4096 || retained.is_empty()
        || retained.iter().any(|&r| r == 0 || r > compiled)
        || retained.iter().copied().collect::<BTreeSet<_>>().len() != retained.len()
        || !retained.contains(&compiled) {
        return Err(invalid("unsupported compiled/live/retained capacity state"));
    }
    Ok(())
}
fn provenance(manifest: &Value, rows: Vec<u32>, sms: u32) -> Result<SparkManifestProvenance> {
    let revision = manifest["sparkinfer_revision"].as_str().filter(|s| !s.is_empty())
        .ok_or_else(|| invalid("missing package revision"))?;
    Ok(SparkManifestProvenance { sha256: format!("{:x}", Sha256::digest(
        serde_json::to_vec(manifest).map_err(|e| invalid(e.to_string()))?)),
        revision: revision.into(), capacity_rows: rows, physical_sms: sms })
}

impl SparkExecutionInventory {
    /// Native Spark owns separate main, serial and grouped-decode arenas. The
    /// compact reducer owns code/handles, not another device scratch arena.
    pub fn native(manifest: &Value, physical_sms: u32, compiled: u32, live: u32,
        retained: &[u32], load: SparkLoadInventory) -> Result<Self> {
        capacities(compiled, live, retained)?;
        if manifest["schema"] != 1 || manifest["capability"] != serde_json::json!([12, 1])
            || physical_sms == 0 || number(manifest, "physical_sms")? != physical_sms as u64 {
            return Err(invalid("native package schema/architecture/physical SM mismatch"));
        }
        let tp = match manifest["role"].as_str() {
            Some("spark") => 4, Some("spark_tp2") => 2,
            Some("spark_tp3") => 3, Some("spark_tp6") => 6,
            _ => return Err(invalid("not a supported native Spark role")),
        };
        if number(manifest, "spark_tp_degree")? != tp {
            return Err(invalid("native role/TP degree mismatch"));
        }
        let geometry = &manifest["geometry"];
        let logical = number(geometry, "intermediate")?;
        let stored = number(geometry, "kernel_intermediate")?;
        if number(geometry, "hidden")? != 5120 || number(geometry, "experts")? != 384
            || number(geometry, "topk")? != 6 || logical != 2304 / tp
            || stored < logical || stored > 2304 || stored % 32 != 0 {
            return Err(invalid("native package is not the selected V4.1 Spark geometry"));
        }
        let mut expected = vec![compiled];
        if compiled > 1 { expected.push(1); }
        if compiled > 80 { expected.push(80); }
        if retained != expected { return Err(invalid("native retained state must be main, m1, m80 in owner order")); }
        let input = match manifest["input_format"].as_str() {
            Some("bf16") => 5120 * 2, Some("fp8_k32") => 5120 + 5120 / 32,
            _ => return Err(invalid("unsupported native Spark ingress")),
        };
        let variants = manifest["variants"].as_array().ok_or_else(|| invalid("missing native variants"))?;
        let mut seen = BTreeSet::new();
        for variant in variants {
            if !seen.insert(number(variant, "capacity_rows")?) { return Err(invalid("duplicate native capacity")); }
        }
        let mut device = Vec::new();
        for &rows in retained {
            let variant = variants.iter().find(|v| v["capacity_rows"] == rows)
                .ok_or_else(|| invalid(format!("missing native capacity {rows}")))?;
            device.push(item(Category::Workspace, &format!("expert_scratch_m{rows}"),
                number(variant, "core_scratch_nbytes")?, Basis::Exact));
        }
        device.push(item(Category::Workspace, "expert_hidden", mul(&[live as u64, input])?, Basis::Formula));
        device.push(item(Category::Workspace, "expert_routes", mul(&[live as u64, 6, 8])?, Basis::Formula));
        device.push(item(Category::Workspace, "expert_compact_output", mul(&[live as u64, 5120, 2])?, Basis::Formula));
        // Reducer function handles carry no separately allocated device payload.
        device.push(item(Category::Workspace, "compact_reducer_scratch", 0, Basis::Exact));
        Ok(Self { compiled_capacity: compiled, live_capacity: live, retained_capacities: retained.into(),
            physical_sms, logical_intermediate: logical as u32, stored_intermediate: stored as u32,
            manifests: vec![provenance(manifest, retained.into(), physical_sms)?],
            retained: SparkMemoryDomains { device, pinned: vec![item(Category::Workspace,
                "expert_route_staging", mul(&[live as u64, 6, 8])?, Basis::Formula)], host: vec![] }, load })
    }

    /// The selected schedule's actual capacity manifests, not a coordinator
    /// export. The existing helper preserves alias/shared/private LUT semantics.
    pub fn exl3(manifests: &[Value], physical_sms: u32, compiled: u32, live: u32,
        retained: &[u32], load: SparkLoadInventory) -> Result<Self> {
        capacities(compiled, live, retained)?;
        let expected = cuteafd_core::Exl3RowPolicy::Nearest.capacities(live as usize)
            .map_err(|e| invalid(e.to_string()))?;
        if retained != expected || expected.last() != Some(&compiled) {
            return Err(invalid("EXL3 retained states differ from V4.1 row policy"));
        }
        if physical_sms == 0 || manifests.len() != retained.len() {
            return Err(invalid("EXL3 manifest/retained state count mismatch"));
        }
        let mut sources = Vec::new();
        let mut width = None;
        for (manifest, &rows) in manifests.iter().zip(retained) {
            let intermediate = number(manifest, "intermediate")?;
            if manifest["schema"] != "cuteafd.v41-exl3-aot.v1" || manifest["output_dtype"] != "bf16"
                || number(manifest, "hidden")? != 5120 || number(manifest, "experts")? != 384
                || number(manifest, "top_k")? != 6 || number(manifest, "sms")? != physical_sms as u64
                || number(manifest, "capacity")? != rows as u64 || intermediate == 0
                || intermediate > 2304 || intermediate % 128 != 0
                || width.is_some_and(|w| w != intermediate) {
                return Err(invalid("EXL3 package geometry/capacity/physical SM mismatch"));
            }
            width = Some(intermediate);
            sources.push(provenance(manifest, vec![rows], physical_sms)?);
        }
        let arena = crate::serving_capacity::exl3_workspace_bytes(manifests, true)
            .map_err(|e| invalid(e.to_string()))?;
        Ok(Self { compiled_capacity: compiled, live_capacity: live, retained_capacities: retained.into(),
            physical_sms, logical_intermediate: width.unwrap() as u32, stored_intermediate: width.unwrap() as u32,
            manifests: sources, retained: SparkMemoryDomains {
                device: vec![item(Category::Workspace, "exl3_retained_capacity_arenas", arena, Basis::Exact),
                    item(Category::Workspace, "expert_hidden", mul(&[live as u64, 5280])?, Basis::Formula),
                    item(Category::Workspace, "expert_routes", mul(&[live as u64, 6, 8])?, Basis::Formula)],
                pinned: vec![item(Category::Workspace, "expert_route_staging", mul(&[live as u64, 6, 8])?, Basis::Formula)],
                host: vec![] }, load })
    }
}

/// Calibration must come from an isolated same-process phase pair. Neither
/// MemAvailable nor total CUDA use minus the ledger qualifies a runtime term.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SparkRuntimeProvenance {
    pub source: String,
    pub method: SparkRuntimeMethod,
    pub phase: String,
    pub pid: u32,
    pub quiet_host: bool,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SparkRuntimeMethod {
    ProcessAttributedPhaseDelta,
    MeasuredResidual,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SparkRuntimeTerm {
    pub item: Item,
    pub provenance: SparkRuntimeProvenance,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualifiedSparkRuntime {
    pub arch: String,
    pub driver: String,
    pub physical_sms: u32,
    /// Fingerprints of the selected capacity packages, not just family name.
    pub manifest_sha256: Vec<String>,
    pub device: Vec<SparkRuntimeTerm>,
    pub pinned: Vec<SparkRuntimeTerm>,
    pub host: Vec<SparkRuntimeTerm>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SparkRankInventory {
    pub rank: u32,
    /// Serving owners; domains are disjoint even on GB10's unified pool.
    pub device: Vec<Item>,
    pub pinned: Vec<Item>,
    pub host: Vec<Item>,
    /// Additional load-only owners. They do not coexist with execution/rings.
    pub load_transient: SparkMemoryDomains,
    /// Full independent unified-memory peaks, including resident and runtime.
    pub load_peak: u64,
    pub serve_peak: u64,
    pub manifests: Vec<SparkManifestProvenance>,
    pub runtime_provenance: Vec<SparkRuntimeProvenance>,
    pub load_provenance: String,
}

pub fn v41_spark_inventory(assignment: &SparkAssignment, execution: &SparkExecutionInventory,
    runtime: &QualifiedSparkRuntime) -> Result<SparkRankInventory> {
    capacities(execution.compiled_capacity, execution.live_capacity, &execution.retained_capacities)?;
    if runtime.arch != "sm_121" || runtime.driver.is_empty() || runtime.physical_sms != execution.physical_sms
        || execution.load.source.is_empty() || assignment.slices.is_empty()
        || assignment.slices.iter().any(|s| s.width != execution.logical_intermediate
            || s.stored_width != execution.stored_intermediate) {
        return Err(invalid("assignment/execution/runtime qualification mismatch (idle rank needs no execution)"));
    }
    let fingerprints: Vec<_> = execution.manifests.iter().map(|m| m.sha256.clone()).collect();
    if fingerprints != runtime.manifest_sha256 { return Err(invalid("runtime calibrated for different package manifests")); }
    let all: Vec<_> = runtime.device.iter().chain(&runtime.pinned).chain(&runtime.host).collect();
    for term in &all {
        let p = &term.provenance;
        if p.source.is_empty() || p.phase.is_empty() || p.pid == 0 || !p.quiet_host
            || term.item.basis != Basis::Calibrated
            || (p.method == SparkRuntimeMethod::MeasuredResidual && !term.item.group.contains("measured_residual")) {
            return Err(invalid("unqualified runtime term; residuals must be named measured_residual"));
        }
    }
    for domain in [&runtime.device, &runtime.pinned, &runtime.host] {
        let mut groups = BTreeSet::new();
        if domain.iter().any(|term| !groups.insert(&term.item.group)) {
            return Err(invalid("duplicate runtime owner in one memory domain"));
        }
    }
    if !runtime.device.iter().any(|t| t.item.group == "cuda_context")
        || !runtime.device.iter().any(|t| t.item.group == "expert_modules")
        || !runtime.host.iter().any(|t| t.item.group.contains("measured_residual")
            && t.provenance.method == SparkRuntimeMethod::MeasuredResidual) {
        return Err(invalid("qualified context, modules and physical host residual are required"));
    }
    if runtime.device.iter().any(|t| matches!(t.item.group.as_str(), "cuda_context" | "expert_modules")
        && t.provenance.method != SparkRuntimeMethod::ProcessAttributedPhaseDelta) {
        return Err(invalid("context/modules require isolated process-attributable phase deltas"));
    }
    let mut retained = execution.retained.clone();
    retained.device.insert(0, Item::new(Category::Experts, "expert_weights", "checkpoint_native",
        assignment.resident_bytes, Basis::Exact));
    // Intake is already the input/routes/output owners above. EXL3 output is
    // inside its manifest arena. Never add assignment.io.intake a second time.
    retained.host.push(item(Category::Transport, "expert_exchange", assignment.io.host_exchange, Basis::Formula));
    retained.host.push(item(Category::Transport, "expert_row_indices", assignment.io.row_indices, Basis::Formula));
    retained.pinned.push(item(Category::Transport, "mapped_endpoint_rings", assignment.io.mapped_rings, Basis::Formula));
    retained.device.extend(runtime.device.iter().map(|t| t.item.clone()));
    retained.pinned.extend(runtime.pinned.iter().map(|t| t.item.clone()));
    retained.host.extend(runtime.host.iter().map(|t| t.item.clone()));
    let load_transient = SparkMemoryDomains {
        device: vec![item(Category::Staging, "expert_load_device", execution.load.device_staging_bytes, Basis::Exact)],
        pinned: vec![item(Category::Staging, "expert_load_pinned", execution.load.pinned_staging_bytes, Basis::Exact)],
        host: vec![item(Category::Staging, "expert_read_scratch", execution.load.host_read_scratch_bytes, Basis::Exact)],
    };
    let load_peak = add([assignment.resident_bytes, add(all.iter().map(|t| t.item.bytes))?, load_transient.total()?])?;
    Ok(SparkRankInventory { rank: assignment.rank, load_peak, serve_peak: retained.total()?,
        device: retained.device, pinned: retained.pinned, host: retained.host, load_transient,
        load_provenance: execution.load.source.clone(),
        manifests: execution.manifests.clone(), runtime_provenance: all.iter().map(|t| t.provenance.clone()).collect() })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn load() -> SparkLoadInventory {
        SparkLoadInventory { device_staging_bytes: 100, pinned_staging_bytes: 200,
            host_read_scratch_bytes: 300, source: "test checkpoint staging plans".into() }
    }
    fn native() -> Value {
        json!({"schema":1,"role":"spark_tp3","spark_tp_degree":3,"capability":[12,1],
            "physical_sms":48,"sparkinfer_revision":"test","input_format":"fp8_k32",
            "geometry":{"experts":384,"hidden":5120,"intermediate":768,"kernel_intermediate":768,"topk":6},
            "variants":[{"capacity_rows":1,"core_scratch_nbytes":11},
                {"capacity_rows":80,"core_scratch_nbytes":80},
                {"capacity_rows":4096,"core_scratch_nbytes":4096}]})
    }
    fn execution() -> SparkExecutionInventory {
        SparkExecutionInventory::native(&native(), 48, 4096, 1024, &[4096, 1, 80], load()).unwrap()
    }
    fn runtime(e: &SparkExecutionInventory) -> QualifiedSparkRuntime {
        let term = |name, bytes, method| SparkRuntimeTerm {
            item: item(Category::Runtime, name, bytes, Basis::Calibrated),
            provenance: SparkRuntimeProvenance { source: "test quiet-host census".into(),
                method, phase: "before/after isolated same-process phase".into(), pid: 42, quiet_host: true } };
        QualifiedSparkRuntime { arch: "sm_121".into(), driver: "test".into(), physical_sms: 48,
            manifest_sha256: e.manifests.iter().map(|m| m.sha256.clone()).collect(),
            device: vec![term("cuda_context", 10, SparkRuntimeMethod::ProcessAttributedPhaseDelta),
                term("expert_modules", 20, SparkRuntimeMethod::ProcessAttributedPhaseDelta)],
            pinned: vec![], host: vec![term("host_measured_residual", 30, SparkRuntimeMethod::MeasuredResidual)] }
    }
    fn assignment() -> SparkAssignment {
        SparkAssignment { rank: 1, resident_bytes: 1000, slices: vec![cuteafd_core::SparkSlice {
            layer: 5, first: 768, width: 768, stored_width: 768, tiers: vec![], projection_counts: vec![] }],
            io: crate::placement::SparkIoBytes { intake: u64::MAX, host_exchange: 111, row_indices: 222, mapped_rings: 333 } }
    }
    #[test]
    fn native_sums_retained_arenas_instead_of_coordinator_maximum() {
        let e = execution();
        assert_eq!(e.retained.device.iter().filter(|i| i.group.starts_with("expert_scratch"))
            .map(|i| i.bytes).sum::<u64>(), 4096 + 11 + 80);
        assert_eq!(e.retained.pinned[0].bytes, 1024 * 6 * 8);
        let r = runtime(&e);
        let result = v41_spark_inventory(&assignment(), &e, &r).unwrap();
        assert_eq!(result.load_peak, 1000 + 60 + 100 + 200 + 300);
        assert_eq!(result.serve_peak, 1000 + 60 + e.retained.total().unwrap() + 111 + 222 + 333);
        assert!(!result.device.iter().any(|i| i.group == "intake"));
        assert_eq!(result.device.iter().chain(&result.pinned).chain(&result.host).map(|i| i.bytes).sum::<u64>(), result.serve_peak);
    }
    #[test]
    fn native_rejects_unsupported_states_and_package_mismatches() {
        for (sms, compiled, live, states) in [(47,4096,1024,vec![4096,1,80]),
            (48,4096,4097,vec![4096,1,80]), (48,4096,1024,vec![4096]),
            (48,4096,1024,vec![4096,80,1]), (48,80,80,vec![80,1,80])] {
            assert!(SparkExecutionInventory::native(&native(), sms, compiled, live, &states, load()).is_err());
        }
        for key in ["role", "physical_sms", "variants", "geometry", "sparkinfer_revision"] {
            let mut m = native(); m[key] = Value::Null;
            assert!(SparkExecutionInventory::native(&m,48,4096,1024,&[4096,1,80],load()).is_err(), "{key}");
        }
        let mut duplicate = native(); let v = duplicate["variants"][0].clone();
        duplicate["variants"].as_array_mut().unwrap().push(v);
        assert!(SparkExecutionInventory::native(&duplicate,48,4096,1024,&[4096,1,80],load()).is_err());
    }
    #[test]
    fn exl3_preserves_shared_max_and_private_capacity_state() {
        let manifests: Vec<_> = [1,16,80].into_iter().map(|rows| json!({
            "schema":"cuteafd.v41-exl3-aot.v1", "output_dtype":"bf16", "hidden":5120,
            "intermediate":768,"experts":384,"top_k":6,"sms":48,"capacity":rows,
            "sparkinfer_revision":"test","input_format":"e4m3_k32",
            "trellis_lut":{"bytes":32},"buffers":{
                "scratch":{"bytes":rows*100,"allocation":"scratch","dtype":"bf16","zero_on_create":false},
                "alias":{"bytes":99999,"allocation":"scratch","dtype":"bf16","zero_on_create":false},
                "state":{"bytes":0,"allocation":"state","dtype":"i32","zero_on_create":true}}})).collect();
        let e = SparkExecutionInventory::exl3(&manifests,48,80,16,&[1,16,80],load()).unwrap();
        assert_eq!(e.retained.device[0].bytes,8000 + 3*(32+16));
        assert_eq!(e.retained.device[1].bytes,16*5280);
        assert!(SparkExecutionInventory::exl3(&manifests,47,80,16,&[1,16,80],load()).is_err());
        assert!(SparkExecutionInventory::exl3(&manifests[2..],48,80,16,&[80],load()).is_err());
    }
    #[test]
    fn runtime_requires_isolated_evidence_and_matching_package() {
        let e=execution(); let a=assignment();
        for bad in 0..9 {
            let mut r=runtime(&e);
            match bad {
                0 => r.device[0].provenance.quiet_host=false,
                1 => r.device[0].provenance.pid=0,
                2 => r.device[0].item.basis=Basis::Exact,
                3 => r.manifest_sha256[0]="different".into(),
                4 => r.device[0].provenance.method=SparkRuntimeMethod::MeasuredResidual,
                5 => r.host[0].item.group="context".into(),
                6 => r.host.clear(),
                7 => r.device.push(r.device[0].clone()),
                _ => r.host[0].provenance.method=SparkRuntimeMethod::ProcessAttributedPhaseDelta,
            }
            assert!(v41_spark_inventory(&a,&e,&r).is_err(),"{bad}");
        }
        let mut overflow=e.clone(); overflow.load.device_staging_bytes=u64::MAX;
        assert!(v41_spark_inventory(&a,&overflow,&runtime(&overflow)).is_err());
    }
}

#[cfg(test)]
mod checkpoint_load_tests {
    use super::*;
    #[test]
    fn native_load_inventory_matches_checkpoint_staging_not_resident_size() {
        for nvfp4 in [false,true] {
            let dir=tempfile::tempdir().unwrap();
            if nvfp4 { crate::plan::testing::write_v41_nvfp4_snapshot(dir.path()); }
            else { crate::plan::testing::write_v41_snapshot(dir.path()); }
            let catalog=crate::read_expert_catalog(dir.path()).unwrap();
            let selection=crate::V41ExpertSelection::BackboneTp { layer:3,expert:0,rank:1,world:3 };
            let inventory=SparkLoadInventory::native(&catalog,&[selection]).unwrap();
            let (staging,scratch)=if nvfp4 {
                let plan=catalog.nvfp4_expert_staging(selection).unwrap();
                (plan.staging_bytes() as u64,plan.minimum_read_scratch_bytes() as u64)
            } else {
                let plan=catalog.expert_staging(selection).unwrap();
                (plan.staging_bytes() as u64,plan.minimum_read_scratch_bytes() as u64)
            };
            assert_eq!(inventory.device_staging_bytes,staging);
            assert_eq!(inventory.pinned_staging_bytes,staging*16);
            assert_eq!(inventory.host_read_scratch_bytes,scratch*64*16);
            assert!(SparkLoadInventory::native(&catalog,&[]).is_err());
        }
    }
    #[test]
    fn exl3_load_inventory_matches_simultaneous_banks_not_all_layers() {
        let dir=tempfile::tempdir().unwrap();
        crate::plan::testing::write_v41_exl3_snapshot(dir.path(),3);
        let catalog=crate::read_expert_catalog(dir.path()).unwrap();
        let manifest=catalog.exl3().unwrap();
        let plans=[3,4].into_iter().map(|layer|manifest.residency(crate::V41Exl3Layer::Backbone(layer),3,1).unwrap()).collect::<Vec<_>>();
        let result=SparkLoadInventory::exl3(&catalog,&plans).unwrap();
        let staging=plans.iter().map(|p|p.loads.chunks(9).map(|jobs|jobs.iter().map(|j|j.bytes as u64).sum::<u64>()).max().unwrap()).max().unwrap();
        let init=plans.iter().flat_map(|p|&p.buffers).map(|b|b.initial_words.len() as u64*4).max().unwrap();
        let scratch=plans.iter().map(|p|p.scratch_bytes(&catalog).unwrap() as u64).max().unwrap();
        assert_eq!(result.device_staging_bytes,0);
        assert_eq!(result.pinned_staging_bytes,staging.max(init)*16*2);
        assert_eq!(result.host_read_scratch_bytes,scratch*64*16*2);
    }
}
