//! Select native programs from target-backbone execution slices, never table names.
use anyhow::{Context, Result, ensure};
use cuteafd_ffi::native_dense::{DenseGeometry, Encoding};
use cuteafd_ffi::native_dense_glue::{Op, op_of};
use cuteafd_ffi::programs::Programs;
use super::{
    DenseAxis, DenseEncoding, DensePlan, DenseRecord, DenseSelection, ShardSide,
};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Requirement { encoding: DenseEncoding, k: usize, n: usize, side: usize }

fn requirements(selection: &DenseSelection, records: &[DenseRecord], layers: usize,
    plan: &DensePlan) -> Result<Vec<Requirement>> {
    let backbone = selection.backbone(records, layers)?;
    let mut result = Vec::new();
    for slice in &plan.tensors {
        if slice.absorbed.is_some() { continue; }
        let record = backbone.iter().find(|r| r.tensor == slice.tensor).context("native slice record")?;
        if record.shape.len() != 2 { continue; }
        let (mut n, mut k) = (record.shape[0], record.shape[1]);
        match slice.axis { DenseAxis::Output => n = slice.channels, DenseAxis::Input => k = slice.channels }
        let side = if record.side == ShardSide::Input { 2 } else { 1 };
        let next = Requirement { encoding: record.encoding, k, n, side };
        if !result.contains(&next) { result.push(next); }
    }
    Ok(result)
}

fn matches(required: Requirement, actual: DenseGeometry, a8: bool) -> bool {
    let encoding = match required.encoding {
        DenseEncoding::Trellis { bits: 4 } => actual.encoding == Encoding::Trellis
            || (a8 && actual.encoding == Encoding::TrellisA8),
        DenseEncoding::Int4G32 => actual.encoding == Encoding::Int4G32,
        DenseEncoding::Mxfp8 => actual.encoding == Encoding::Mxfp8,
        DenseEncoding::RowFp8 => actual.encoding == Encoding::RowFp8,
        _ => false,
    };
    encoding && actual.input_features == required.k as u64
        && actual.output_features == required.n as u64 && actual.side == required.side as u64
}

/// Exported routes define the activation domain; callers retain their existing auto policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivationDomain { pub a16: bool, pub a8: bool }

#[derive(Debug, thiserror::Error)]
pub enum ActivationError {
    #[error("EXLR8 explicitly pinned {requested} activations outside exported domain {domain:?}")]
    Explicit { requested: super::super::PrefillActivations, domain: ActivationDomain },
    #[error("EXLR8 auto policy selected {selected} outside exported domain {domain:?}")]
    Auto { selected: super::super::PrefillActivations, domain: ActivationDomain },
}

impl ActivationDomain {
    pub fn contains(self, value: super::super::PrefillActivations) -> bool {
        match value { super::super::PrefillActivations::A16 => self.a16,
            super::super::PrefillActivations::A8 => self.a8 }
    }
    /// Dense and routed experts must agree on the activation contract.
    pub fn with_experts(self, manifest: &serde_json::Value,
        family: cuteafd_ffi::exlr8::KernelFamily) -> Result<Self> {
        use cuteafd_ffi::exlr8::KernelKind;
        let entries = manifest["programs"].as_array().context("manifest programs")?;
        let supported = |kind| -> Result<bool> {
            let name = family.program(kind);
            if !entries.iter().any(|entry| entry["name"] == name) { return Ok(false); }
            let spec = super::super::workspace::program_spec(manifest, name)?;
            let entry = entries.iter().find(|entry| entry["name"] == name).context("expert program entry")?;
            cuteafd_ffi::exlr8::validate_entry(&spec, entry)?;
            Ok(true)
        };
        ensure!(supported(KernelKind::GroupedA16)?, "EXLR8 lacks exported A16 expert prefill");
        let a8 = supported(KernelKind::GroupedA8)?;
        Ok(Self { a16: self.a16, a8: self.a8 && a8 })
    }

    pub fn resolve(self, pin: Option<super::super::PrefillActivations>,
        existing_auto: super::super::PrefillActivations)
        -> std::result::Result<super::super::PrefillActivations, ActivationError> {
        let selected = pin.unwrap_or(existing_auto);
        if self.contains(selected) { return Ok(selected); }
        Err(match pin {
            Some(requested) => ActivationError::Explicit { requested, domain: self },
            None => ActivationError::Auto { selected, domain: self },
        })
    }
}

/// Every owned rank must have all of its native dense routes. Unrelated package
/// entries cannot enable A8 for a family with uncovered projection geometry.
pub fn activation_domain(manifest: &serde_json::Value, selection: &DenseSelection,
    records: &[DenseRecord], layers: usize, plans: &[DensePlan]) -> Result<ActivationDomain> {
    ensure!(!plans.is_empty(), "native activation domain requires owned rank plans");
    let mut a8 = true;
    for plan in plans {
        select_manifest(manifest, selection, records, layers, plan, false)?;
        let required = requirements(selection, records, layers, plan)?;
        let entries = manifest["programs"].as_array().context("manifest programs")?;
        let mut covered = vec![false; required.len()];
        let mut quantized = BTreeSet::new();
        for entry in entries {
            let name = entry["name"].as_str().context("manifest program name")?;
            if !name.starts_with("exlr8_dense_") { continue; }
            let spec = super::super::workspace::program_spec(manifest, name)?;
            if let Some(op) = op_of(entry)? {
                cuteafd_ffi::native_dense_glue::validate_entry(&spec, entry)?;
                if let Op::Mx8Quant { features } = op { quantized.insert(features as usize); }
                continue;
            }
            let geometry = cuteafd_ffi::native_dense::validate_spec(&spec)?;
            if geometry.encoding != Encoding::TrellisA8 { continue; }
            for (index, &requirement) in required.iter().enumerate() {
                covered[index] |= matches(requirement, geometry, true);
            }
        }
        a8 &= required.iter().zip(covered).all(|(requirement, covered)|
            !matches!(requirement.encoding, DenseEncoding::Trellis { bits: 4 })
                || (covered && quantized.contains(&requirement.k)));
        if a8 { select_manifest(manifest, selection, records, layers, plan, true)?; }
    }
    Ok(ActivationDomain { a16: true, a8 })
}

/// One admission validator for CPU planning and runtime adapters. Geometry comes
/// from the family's config reader, while availability comes from the package.
pub fn resolve_activations(manifest: &serde_json::Value, config: &serde_json::Value,
    records: &[DenseRecord], layers: usize, plans: &[DensePlan],
    pin: Option<super::super::PrefillActivations>, existing_auto: super::super::PrefillActivations)
    -> Result<super::super::PrefillActivations> {
    let selection = DenseSelection::from_hf(config)?;
    let geometry = super::super::ExpertGeometry::from_hf(config)?;
    ensure!(geometry.intermediate % 128 == 0, "EXLR8 expert intermediate must divide into units");
    let text = config.get("text_config").unwrap_or(config);
    let top_k = text["num_experts_per_tok"].as_u64().context("EXLR8 top-k missing")?;
    let family = cuteafd_ffi::exlr8::KernelFamily::from_geometry(geometry.hidden,
        geometry.intermediate / 128, usize::try_from(top_k)?, geometry.experts)?;
    Ok(activation_domain(manifest, &selection, records, layers, plans)?
        .with_experts(manifest, family)?.resolve(pin, existing_auto)?)
}

/// Vocabulary roles are removed by DenseSelection before geometry matching. A
/// row-FP8 backbone can therefore reuse the same linear as a vocabulary head.
pub fn select(programs: &Programs<'_>, manifest: &serde_json::Value,
    selection: &DenseSelection, records: &[DenseRecord], layers: usize,
    plan: &DensePlan, a8: bool) -> Result<BTreeSet<String>> {
    select_entries(manifest, selection, records, layers, plan, a8, |name| Ok(programs.spec(name)?.clone()))
}

/// CPU admission uses the identical selection over the exported manifest, without CUDA.
pub fn select_manifest(manifest: &serde_json::Value, selection: &DenseSelection,
    records: &[DenseRecord], layers: usize, plan: &DensePlan, a8: bool) -> Result<BTreeSet<String>> {
    select_entries(manifest, selection, records, layers, plan, a8,
        |name| super::super::workspace::program_spec(manifest, name))
}

fn select_entries(manifest: &serde_json::Value, selection: &DenseSelection, records: &[DenseRecord],
    layers: usize, plan: &DensePlan, a8: bool,
    spec: impl Fn(&str) -> Result<cuteafd_ffi::programs::ProgramSpec>) -> Result<BTreeSet<String>> {
    let required = requirements(selection, records, layers, plan)?;
    let mut covered = vec![0u8; required.len()];
    let mut selected = BTreeSet::new();
    for entry in manifest["programs"].as_array().context("manifest programs")? {
        let name = entry["name"].as_str().context("manifest program name")?;
        if !name.starts_with("exlr8_dense_") { continue; }
        let spec = spec(name)?;
        let trellis = |r: &&Requirement| matches!(r.encoding, DenseEncoding::Trellis { bits: 4 });
        let needed = if let Some(op) = op_of(entry)? {
            match op {
                Op::TrellisInput { features, .. } | Op::Mx8Quant { features } =>
                    (!matches!(op, Op::Mx8Quant { .. }) || a8)
                        && required.iter().filter(trellis).any(|r| r.k == features as usize),
                Op::TrellisOutput { features } => required.iter().filter(trellis).any(|r| r.n == features as usize),
                Op::SwigluSplit { inter } => required.iter().filter(trellis).any(|r| r.n == inter as usize),
                Op::NarrowPack { width, .. } => required.iter().any(|r| r.n == width as usize),
                Op::IndexWeights { width, outputs } => required.iter().any(|r|
                    r.encoding == DenseEncoding::Bf16 && (r.k, r.n) == (width as usize, outputs as usize)),
            }
        } else {
            let geometry = cuteafd_ffi::native_dense::validate_spec(&spec)?;
            let mut needed = false;
            for (index, &requirement) in required.iter().enumerate() {
                if matches(requirement, geometry, a8) {
                    covered[index] |= match geometry.encoding {
                        Encoding::TrellisA8 => 4,
                        Encoding::Trellis if spec.geometry["input_dtype"] == "bf16" => 2,
                        _ => 1,
                    };
                    needed = true;
                }
            }
            needed
        };
        if needed { selected.insert(name.to_owned()); }
    }
    for (requirement, covered) in required.iter().zip(covered) {
        if matches!(requirement.encoding, DenseEncoding::Trellis { .. } | DenseEncoding::Int4G32
            | DenseEncoding::Mxfp8 | DenseEncoding::RowFp8) {
            let routes = if matches!(requirement.encoding, DenseEncoding::Trellis { .. }) {
                if a8 { 7 } else { 3 }
            } else { 1 };
            ensure!(covered & routes == routes, "native backbone lacks required decode/prefill linear routes for {requirement:?}");
        }
    }
    Ok(selected)
}

/// Vocabulary is a separate owner, selected by its exact validated geometry.
pub fn vocabulary(programs: &Programs<'_>, hidden: usize, vocabulary: usize) -> Result<BTreeSet<String>> {
    let mut selected = BTreeSet::new();
    for encoding in [Encoding::Embedding, Encoding::RowFp8] {
        let matches = programs.names().filter(|name| name.starts_with("exlr8_dense_"))
            .filter_map(|name| cuteafd_ffi::native_dense::validate_spec(programs.spec(name).ok()?).ok()
                .filter(|g| g.encoding == encoding && g.input_features == hidden as u64
                    && g.output_features == vocabulary as u64).map(|_| name.to_owned())).collect::<Vec<_>>();
        ensure!(matches.len() == 1, "native vocabulary requires one {encoding:?} kernel at {hidden}x{vocabulary}");
        selected.extend(matches);
    }
    Ok(selected)
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::{DenseExecution, DenseOwnership};

    #[test]
    fn activation_domain_preserves_auto_and_types_explicit_refusals() {
        use super::super::super::PrefillActivations::{A16, A8};
        let both = ActivationDomain { a16: true, a8: true };
        assert_eq!(both.resolve(None, A16).unwrap(), A16);
        assert_eq!(both.resolve(None, A8).unwrap(), A8);
        assert_eq!(both.resolve(Some(A16), A8).unwrap(), A16);
        assert_eq!(both.resolve(Some(A8), A16).unwrap(), A8);
        let a16 = ActivationDomain { a16: true, a8: false };
        assert_eq!(a16.resolve(None, A16).unwrap(), A16);
        assert!(matches!(a16.resolve(Some(A8), A16), Err(ActivationError::Explicit { requested: A8, .. })));
        assert!(matches!(a16.resolve(None, A8), Err(ActivationError::Auto { selected: A8, .. })));
    }

    #[test]
    #[ignore = "requires published headers and complete actual exporter manifest; CPU only"]
    fn published_activation_domains_cover_dense_and_expert_routes() {
        use super::super::super::PrefillActivations::{A16, A8};
        let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(
            std::env::var_os("CUTEAFD_EXLR8_ACTIVATION_MANIFEST").expect("complete manifest")).unwrap()).unwrap();
        for (variable, has_a8) in [("CUTEAFD_EXLR8_GLM_SNAPSHOT", true),
            ("CUTEAFD_EXLR8_FLASH_SNAPSHOT", false)] {
            let root = std::path::PathBuf::from(std::env::var_os(variable).expect(variable));
            let config: serde_json::Value = serde_json::from_slice(&std::fs::read(root.join("config.json")).unwrap()).unwrap();
            let geometry = super::super::super::ExpertGeometry::from_hf(&config).unwrap();
            let selection = DenseSelection::from_hf(&config).unwrap();
            let records = DenseRecord::inventory(&root).unwrap();
            let backbone = selection.backbone(&records, geometry.layers).unwrap();
            for world in [1, 2] {
                let plans = (0..world).map(|rank| DensePlan::for_rank_owned(&backbone, world, rank,
                    &Default::default()).unwrap()).collect::<Vec<_>>();
                let domain = activation_domain(&manifest, &selection, &records, geometry.layers, &plans).unwrap();
                assert_eq!(domain, ActivationDomain { a16: true, a8: has_a8 });
                assert_eq!(resolve_activations(&manifest, &config, &records, geometry.layers,
                    &plans, None, A16).unwrap(), A16);
                let pinned = resolve_activations(&manifest, &config, &records, geometry.layers,
                    &plans, Some(A8), A16);
                if has_a8 {
                    assert_eq!(pinned.unwrap(), A8);
                    assert_eq!(resolve_activations(&manifest, &config, &records, geometry.layers,
                        &plans, None, A8).unwrap(), A8);
                } else {
                    assert!(matches!(pinned.unwrap_err().downcast_ref::<ActivationError>(),
                        Some(ActivationError::Explicit { requested: A8, .. })));
                }
                let mut missing = manifest.clone();
                missing["programs"].as_array_mut().unwrap().retain(|e| e["name"] != "exlr8_a8_grouped");
                if has_a8 {
                    assert!(resolve_activations(&missing, &config, &records, geometry.layers,
                        &plans, Some(A8), A16).unwrap_err().is::<ActivationError>());
                    let mut missing_quant = manifest.clone();
                    missing_quant["programs"].as_array_mut().unwrap().retain(|e|
                        !matches!(op_of(e).unwrap(), Some(Op::Mx8Quant { .. })));
                    assert!(!activation_domain(&missing_quant, &selection, &records,
                        geometry.layers, &plans).unwrap().a8);
                    let mut bad_expert = manifest.clone();
                    bad_expert["programs"].as_array_mut().unwrap().iter_mut()
                        .find(|e| e["name"] == "exlr8_a8_grouped").unwrap()["scalars"] = serde_json::json!([]);
                    assert!(!resolve_activations(&bad_expert, &config, &records, geometry.layers,
                        &plans, Some(A8), A16).unwrap_err().is::<ActivationError>());
                }
                let selected = select_manifest(&manifest, &selection, &records, geometry.layers, &plans[0], false).unwrap();
                let name = selected.iter().find(|name| {
                    let spec = super::super::super::workspace::program_spec(&manifest, name).unwrap();
                    cuteafd_ffi::native_dense::validate_spec(&spec).is_ok()
                }).unwrap();
                let mut absent = manifest.clone();
                absent["programs"].as_array_mut().unwrap().retain(|e| e["name"] != *name);
                assert!(activation_domain(&absent, &selection, &records, geometry.layers, &plans).is_err());
                let mut duplicate = manifest.clone();
                let entry = duplicate["programs"].as_array().unwrap().iter().find(|e| e["name"] == *name).unwrap().clone();
                duplicate["programs"].as_array_mut().unwrap().push(entry);
                assert!(activation_domain(&duplicate, &selection, &records, geometry.layers, &plans).is_err());
                let mut malformed = manifest.clone();
                malformed["programs"].as_array_mut().unwrap().iter_mut().find(|e| e["name"] == *name).unwrap()["scalars"] = serde_json::json!([]);
                assert!(activation_domain(&malformed, &selection, &records, geometry.layers, &plans).is_err());
                eprintln!("{variable} TP{world}: {domain:?}, auto a16");
            }
        }
    }

    #[test]
    fn requirements_consume_external_whole_mla_owner_without_replanning() {
        use crate::families::glm5_flash::{GlmNextAttention, GlmNextConfig};
        let configs: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../tests/fixtures/exlr8-dense-configs.json")).unwrap();
        let cfg = GlmNextConfig::from_hf(&configs["flash"]).unwrap();
        let selection = DenseSelection::from_hf(&configs["flash"]).unwrap();
        let mla = cfg.attention.iter().position(|a| *a == GlmNextAttention::Mla).unwrap();
        let mut ownership = DenseOwnership::new(DenseExecution::Replicated);
        ownership.whole_mla = Some(mla);
        let record = DenseRecord { tensor: format!("{}.self_attn.q_a_proj.weight", selection.layer_prefix(mla).unwrap()),
            encoding: DenseEncoding::Mxfp8, shape: vec![cfg.q_lora_rank, cfg.hidden],
            side: ShardSide::Replicated, ownership, channels_per_shard: 0,
            rotation_id: 0, rotation: None, hessian_solved: false, components: vec![], native_blocks: 0 };
        let records = [record];
        let owners = std::collections::BTreeMap::from([(mla, 1)]);
        for rank in 0..2 {
            let plan = DensePlan::for_rank_owned(&records, 2, rank, &owners).unwrap();
            let required = requirements(&selection, &records, cfg.layers, &plan).unwrap();
            assert_eq!(required.len(), rank);
            if rank == 1 { assert_eq!((required[0].k, required[0].n), (cfg.hidden, cfg.q_lora_rank)); }
        }
    }

    #[test]
    #[ignore = "requires native checkpoint headers and full exported union manifest; CPU only"]
    fn published_headers_select_unique_program_geometry_for_every_rank() {
        let path = std::env::var_os("CUTEAFD_EXLR8_DENSE_MANIFEST").expect("union manifest");
        let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        let spec = |name: &str| super::super::super::workspace::program_spec(&manifest, name);
        for variable in ["CUTEAFD_EXLR8_GLM_SNAPSHOT", "CUTEAFD_EXLR8_FLASH_SNAPSHOT"] {
            let root = std::path::PathBuf::from(std::env::var_os(variable).expect(variable));
            let config: serde_json::Value = serde_json::from_slice(&std::fs::read(root.join("config.json")).unwrap()).unwrap();
            let text = config.get("text_config").unwrap_or(&config);
            let layers = text["num_hidden_layers"].as_u64().unwrap() as usize;
            let selection = DenseSelection::from_hf(&config).unwrap();
            let records = DenseRecord::inventory(&root).unwrap();
            let backbone = selection.backbone(&records, layers).unwrap();
            for world in [1, 2] { for rank in 0..world {
                let plan = DensePlan::for_rank_owned(&backbone, world, rank, &Default::default()).unwrap();
                let selected = select_manifest(&manifest, &selection, &records, layers, &plan, false).unwrap();
                assert_eq!(selected, select_entries(&manifest, &selection, &records, layers, &plan, false, &spec).unwrap());
                assert!(!selected.is_empty());
                let (mut cores, mut companions) = (Vec::new(), Vec::new());
                for name in &selected {
                    let e = manifest["programs"].as_array().unwrap().iter().find(|e| e["name"] == *name).unwrap();
                    if let Some(op) = op_of(e).unwrap() { assert!(!companions.contains(&op), "{name}"); companions.push(op); }
                    else {
                        let s = spec(name).unwrap();
                        let g = cuteafd_ffi::native_dense::validate_spec(&s).unwrap();
                        let key = (format!("{:?}", g.encoding), g.input_features, g.output_features, g.side,
                            s.geometry["input_dtype"].as_str().unwrap_or("f16").to_owned());
                        assert!(!cores.contains(&key), "{name}"); cores.push(key);
                    }
                }
                eprintln!("{variable} TP{world} rank{rank}: {} selected cores, {} companions", cores.len(), companions.len());
            } }
        }
    }

    #[test]
    fn backbone_kernel_selection_uses_roles_not_row_fp8_encoding() {
        let configs: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../tests/fixtures/exlr8-dense-configs.json")).unwrap();
        let selection = DenseSelection::from_hf(&configs["flash"]).unwrap();
        let record = |tensor: &str| DenseRecord { tensor: tensor.into(),
            encoding: DenseEncoding::RowFp8, shape: vec![154880, 4096],
            side: ShardSide::Output, ownership: DenseOwnership::new(DenseExecution::RankZero),
            channels_per_shard: 128, rotation_id: 0, rotation: None, hessian_solved: false,
            components: vec![], native_blocks: 0 };
        let records = [record("model.language_model.embed_tokens.weight"), record("lm_head.weight"),
            record("model.language_model.layers.45.shared_head.head.weight"),
            record("model.language_model.layers.0.mlp.gate_proj.weight")];
        let backbone = selection.backbone(&records, 45).unwrap();
        let owned = |world, rank| DensePlan::for_rank_owned(&backbone, world, rank, &Default::default()).unwrap();
        let needed = requirements(&selection, &records, 45, &owned(1, 0)).unwrap();
        assert_eq!(needed, vec![Requirement { encoding: DenseEncoding::RowFp8,
            k: 4096, n: 154880, side: 1 }]);
        assert!(matches(needed[0], DenseGeometry { encoding: Encoding::RowFp8,
            input_features: 4096, output_features: 154880, side: 1, max_rows: 4096 }, false));
        assert!(!matches(needed[0], DenseGeometry { encoding: Encoding::Embedding,
            input_features: 4096, output_features: 154880, side: 1, max_rows: 4096 }, false));
        assert!(requirements(&selection, &records, 45, &owned(2, 1)).unwrap().is_empty());
    }
}
